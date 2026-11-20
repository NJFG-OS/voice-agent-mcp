#!/usr/bin/env python3
"""AetherVoice voice relay - the ONE public port.

Binds 0.0.0.0:8000 and nothing else. vLLM answers on 8002, TTS on 8080 (loopback).
Every Twilio stream gets its OWN session context, resolved from
that stream's custom parameters (workspace_id + agent_id), so two concurrent
calls never share an agent, a transcript, a tool set or a speech buffer.
"""
import asyncio
import audioop
import base64
import hmac
import io
import json
import os
import re
import sys
import time
import wave

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

HOST = "0.0.0.0"
PORT = 8000
VLLM = os.environ.get("AETHER_VLLM_URL", "http://127.0.0.1:8002")
TTS = os.environ.get("AETHER_TTS_URL", "http://127.0.0.1:8080")
WHISPER = os.environ.get("AETHER_WHISPER_URL", "http://127.0.0.1:8001")
APP = os.environ.get("AETHER_APP_ORIGIN", "https://deft-aether-voice-core.base44.app").rstrip("/")
SECRET = os.environ.get("WEBHOOK_SECRET") or os.environ.get("AETHER_WEBHOOK_SECRET") or ""
LOG_DIR = os.environ.get("AETHER_LOG_DIR", "/workspace/aether/logs")
SERVED = os.environ.get("AETHER_SERVED_MODEL", "").strip()
MODELS_TRIED_RAW = os.environ.get("AETHER_MODELS_TRIED", "").strip()
BOOTED = time.time()

# tenant -> (pointer mtime, model id). The pointer file is re-read only when it
# changes, so a promotion is picked up on the next turn without a restart.
ADAPTER_CACHE = {}

# The mode the launcher actually started vLLM in: "lora" or "base", written by
# run_vllm_safe.py before the engine binds and re-read only when it changes. After a
# base fallback the pointer is still on disk, so routing to the alias would 404 every
# completion: the mode file, not the pointer, decides.
MODE_FILE = os.environ.get("AETHER_VLLM_MODE_FILE") or os.path.join(LOG_DIR, "vllm_mode.json")
MODE_CACHE = {"stamp": None, "mode": ""}
BASE_MODEL = (os.environ.get("AETHER_VLLM_MODEL") or "").strip()


def vllm_mode():
    try:
        stamp = os.path.getmtime(MODE_FILE)
    except Exception:
        return ""
    if MODE_CACHE["stamp"] == stamp:
        return MODE_CACHE["mode"]
    mode = ""
    try:
        with open(MODE_FILE) as handle:
            mode = str(json.load(handle).get("mode") or "")
    except Exception:
        mode = ""
    MODE_CACHE.update(stamp=stamp, mode=mode)
    return mode


def answer_model():
    """The model id a completion is answered with.

    The tenant's validated adapter when one is promoted, otherwise the base model.
    The pointer file is the same one the engine read at boot, so the weights loaded
    and the name routed to can never disagree.
    """
    # The launcher fell back to the base model, so the alias it never loaded must not
    # be named. The base id travels as AETHER_VLLM_MODEL; SERVED may itself be the
    # alias, which is exactly why it is not used here.
    if vllm_mode() == "base":
        return BASE_MODEL or SERVED
    tenant = (os.environ.get("AETHER_TENANT_ID") or "").strip()
    if not tenant:
        return SERVED
    root = os.environ.get("AETHER_TENANT_ROOT", "/workspace/tenants")
    pointer = os.path.join(root, tenant, "adapters", "llm_lora", "current_adapter.json")
    try:
        stamp = os.path.getmtime(pointer)
    except Exception:
        return SERVED
    cached = ADAPTER_CACHE.get(tenant)
    if cached and cached[0] == stamp:
        return cached[1]
    name = SERVED
    try:
        with open(pointer) as handle:
            data = json.load(handle)
        path = str(data.get("adapter_path") or "")
        if path and os.path.isdir(path):
            name = re.sub(r"[^a-z0-9_]", "_", tenant.lower())[:40]
    except Exception:
        name = SERVED
    ADAPTER_CACHE[tenant] = (stamp, name)
    return name

# Twilio mu-law is 8 kHz mono, 160 bytes = 20 ms per frame.
FRAME_BYTES = 160
VOICE_RMS = 420          # speech energy floor on 16-bit PCM
SILENCE_FRAMES = 38      # ~0.76 s of silence closes an utterance
MIN_SPEECH_FRAMES = 6    # ignore blips shorter than ~120 ms

# workspace_id -> config pushed by the app (or fetched per stream). Per workspace,
# never one process-wide agent.
CONFIGS = {}
# streamSid -> Session. One entry per live call, removed when the call ends.
SESSIONS = {}


def _runtime_pins():
    """The torch/vLLM this image was built with, from its build-time gate."""
    path = os.environ.get("AETHER_RUNTIME_PINS", "/opt/aether/runtime-pins.json")
    try:
        with open(path) as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return {
        "torch": data.get("torch", ""),
        "vllm": data.get("vllm", ""),
        "base_image": data.get("base_image", ""),
    }


def _models_tried():
    """'id:code,id:code' -> {"id": "code"}, for /health and the ready report."""
    out = {}
    for pair in MODELS_TRIED_RAW.split(","):
        if ":" in pair:
            name, code = pair.rsplit(":", 1)
            out[name.strip()] = code.strip()
    return out


def _auth_ok(request):
    if not SECRET:
        return False
    provided = request.headers.get("x-webhook-secret") or request.query.get("token") or ""
    return hmac.compare_digest(provided, SECRET)


def _scrub(line):
    bad = ("HF_TOKEN", "WEBHOOK_SECRET", "Authorization", "hf_", "sk-")
    if any(mark in line for mark in bad):
        return ""
    return line.rstrip()[:400]


def _tail(path, lines):
    try:
        with open(path, "r", errors="replace") as handle:
            rows = handle.readlines()[-lines:]
    except Exception:
        return []
    return [row for row in (_scrub(line) for line in rows) if row]


async def _code(url, total=2.0):
    try:
        async with ClientSession(timeout=ClientTimeout(total=total)) as session:
            async with session.get(url) as response:
                return response.status
    except Exception:
        return 0


async def _vllm_state():
    """loading vs ready: vLLM only answers /v1/models once weights are in VRAM."""
    try:
        async with ClientSession(timeout=ClientTimeout(total=2.0)) as session:
            async with session.get(VLLM + "/v1/models") as response:
                if response.status != 200:
                    return "loading", []
                data = await response.json()
                ids = [row.get("id") for row in (data.get("data") or []) if row.get("id")]
                return ("ready" if ids else "loading"), ids
    except Exception:
        return "loading", []


async def _whisper_state():
    """200 means the worker is up; ready means its model has finished loading."""
    try:
        async with ClientSession(timeout=ClientTimeout(total=2.0)) as session:
            async with session.get(WHISPER + "/health") as response:
                if response.status != 200:
                    return {"ok": False, "ready": False, "engine": "none"}
                return await response.json()
    except Exception:
        return {"ok": False, "ready": False, "engine": "none"}


async def _deps_ready():
    """The three upstream services this relay needs before it takes a call."""
    vllm_state, _ = await _vllm_state()
    tts_ok = (await _code(TTS + "/health", 1.5)) == 200
    whisper = await _whisper_state()
    missing = []
    if not whisper.get("ok") or not whisper.get("ready"):
        missing.append("whisper")
    if vllm_state != "ready":
        missing.append("vllm_" + str(vllm_state))
    if not tts_ok:
        missing.append("tts")
    return (not missing), ",".join(missing)


def _boot_stage():
    """The step start_all.sh last announced.

    The relay binds before the engines load, so this is what /health reports while
    a boot is still in progress: a stall names its own stage instead of looking
    like a pod that never started.
    """
    try:
        with open(os.path.join(os.environ.get("AETHER_LOG_DIR", "/tmp/aether-logs"), "stage.json")) as handle:
            data = json.load(handle)
        return {"stage": data.get("stage", ""), "detail": data.get("detail", ""), "since": data.get("since", "")}
    except Exception:
        return {}


async def health(_request):
    vllm_state, served = await _vllm_state()
    tts_ok = (await _code(TTS + "/health", 1.5)) == 200
    whisper = await _whisper_state()
    deps_ok, missing = await _deps_ready()
    return web.json_response({
        "ok": True,
        "vllm": vllm_state,
        "vllm_ok": vllm_state == "ready",
        "model": answer_model() or SERVED or (served[0] if served else ""),
        "vllm_mode": vllm_mode(),
        "models_tried": _models_tried(),
        "runtime": _runtime_pins(),
        "tts": tts_ok,
        "whisper": bool(whisper.get("ok")),
        "whisper_ready": bool(whisper.get("ready")),
        "deps_ok": deps_ok,
        "deps_missing": missing,
        "uptime_s": int(time.time() - BOOTED),
        "sessions": len(SESSIONS),
        "workspaces_armed": len(CONFIGS),
        "stage": _boot_stage(),
        "bind": HOST + ":" + str(PORT),
    })


async def forward_get(request):
    path = request.path_qs
    try:
        async with ClientSession(timeout=ClientTimeout(total=60)) as session:
            async with session.get(VLLM + path) as response:
                body = await response.read()
                return web.Response(
                    body=body,
                    status=response.status,
                    content_type=response.content_type or "application/json",
                )
    except Exception as error:
        return web.json_response({"error": "vllm_unreachable", "detail": str(error)}, status=502)


async def forward_post(request):
    path = request.path_qs
    raw = await request.read()
    # A stale model name must never 404 a live call: the served id is the only
    # truth this pod can honour, so every completion is rewritten to it.
    if path.split("?")[0].rstrip("/").endswith("/chat/completions"):
        target = answer_model()
        if target:
            try:
                payload = json.loads(raw or b"{}")
                if isinstance(payload, dict) and payload.get("model") != target:
                    payload["model"] = target
                    raw = json.dumps(payload).encode("utf-8")
            except Exception:
                pass
    try:
        async with ClientSession(timeout=ClientTimeout(total=180)) as session:
            async with session.post(
                VLLM + path, data=raw, headers={"content-type": request.headers.get("content-type", "application/json")}
            ) as response:
                body = await response.read()
                return web.Response(
                    body=body,
                    status=response.status,
                    content_type=response.content_type or "application/json",
                )
    except Exception as error:
        return web.json_response({"error": "vllm_unreachable", "detail": str(error)}, status=502)


async def forward_tts(request):
    """The app reaches the loopback speech worker through this one public port.

    Voice catalog, previews and saved clips all live on 8080, which is loopback
    only. Proxying them here keeps a single public port and no extra exposure.
    """
    path = request.path_qs
    target = TTS + (path[len("/tts"):] if path.startswith("/tts") else path)
    try:
        async with ClientSession(timeout=ClientTimeout(total=90)) as session:
            if request.method == "GET":
                async with session.get(target) as response:
                    body = await response.read()
                    return web.Response(
                        body=body,
                        status=response.status,
                        content_type=response.content_type or "application/json",
                    )
            raw = await request.read()
            async with session.post(
                target,
                data=raw,
                headers={"content-type": request.headers.get("content-type", "application/json")},
            ) as response:
                body = await response.read()
                return web.Response(
                    body=body,
                    status=response.status,
                    content_type=response.content_type or "application/json",
                )
    except Exception as error:
        return web.json_response({"error": "tts_unreachable", "detail": str(error)[:200]}, status=502)


async def session_config(request):
    """Arm one workspace's profile. Never a single global agent."""
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid_json"}, status=400)
    workspace_id = str(payload.get("workspace_id") or "")
    if not workspace_id:
        return web.json_response({"ok": False, "error": "workspace_id_required"}, status=400)
    agent = payload.get("agent") or {}
    CONFIGS[workspace_id] = {
        "ok": True,
        "workspace_id": workspace_id,
        "agent": agent,
        "tools": payload.get("tools") or [],
        "workflow": payload.get("workflow"),
        "audio": payload.get("audio") or {},
        "pushed_at": time.time(),
    }
    entry = CONFIGS[workspace_id]
    return web.json_response({
        "ok": True,
        "armed": bool(agent.get("uuid")),
        "agent": agent.get("name") or "",
        "voice_id": agent.get("voice_id") or "",
        "tools": len(entry["tools"]),
        "workflow": (entry["workflow"] or {}).get("name") or "",
        "uptime_s": int(time.time() - BOOTED),
    })


async def session_end(request):
    """Park one workspace: drop its armed profile and close its live sessions."""
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    workspace_id = str(payload.get("workspace_id") or "")
    for sid, session in list(SESSIONS.items()):
        if not workspace_id or session.workspace_id == workspace_id:
            SESSIONS.pop(sid, None)
    if workspace_id:
        CONFIGS.pop(workspace_id, None)
    return web.json_response({"ok": True, "workspace_id": workspace_id, "sessions": len(SESSIONS)})


async def logs(request):
    """Sanitised log tails for the admin diagnostics panel (secret-gated)."""
    if not _auth_ok(request):
        return web.json_response({"ok": False, "error": "unauthorized"}, status=401)
    try:
        lines = max(20, min(int(request.query.get("lines", 200)), 500))
    except Exception:
        lines = 200
    files = {}
    for name in ("boot_diag", "supervisor", "runtime", "whisper", "vllm", "tts", "relay", "watchdog"):
        files[name] = _tail(os.path.join(LOG_DIR, name + ".log"), lines)
    # The launcher falls back to the container disk when the volume log dir is
    # unusable, and that fallback is the only trace of such a boot.
    if not files.get("boot_diag"):
        files["boot_diag"] = _tail("/tmp/aether-logs/boot_diag.log", lines)
    return web.json_response({
        "ok": True,
        "pod": os.environ.get("RUNPOD_POD_ID", ""),
        "workspace_id": os.environ.get("AETHER_WORKSPACE_ID", ""),
        "uptime_s": int(time.time() - BOOTED),
        "served_model": SERVED,
        "sessions": len(SESSIONS),
        "workspaces_armed": len(CONFIGS),
        "logs": files,
    })


# ---------------------------------------------------------------------------
# App calls: config, transcription, tools. The pod holds no tenant data of its
# own — the app answers every question scoped to the workspace that asked.
# ---------------------------------------------------------------------------

async def app_json(path, payload=None, method="POST", timeout=20):
    if not APP:
        return None
    headers = {"x-webhook-secret": SECRET}
    kwargs = {}
    if payload is not None:
        kwargs["json"] = payload
        headers["content-type"] = "application/json"
    try:
        async with ClientSession(timeout=ClientTimeout(total=timeout)) as session:
            async with session.request(method, APP + path, headers=headers, **kwargs) as response:
                if response.status != 200:
                    return None
                return await response.json()
    except Exception:
        return None


async def resolve_config(workspace_id, agent_uuid):
    """This stream's own workspace profile, fetched fresh from the app."""
    if not workspace_id:
        return {}
    query = "/api/pod/agent-config?workspace_id=" + workspace_id
    if agent_uuid:
        query += "&agent_uuid=" + agent_uuid
    live = await app_json(query, None, method="GET", timeout=12)
    if live and live.get("ok"):
        CONFIGS[workspace_id] = live
        return live
    return CONFIGS.get(workspace_id) or {}


async def tts_wav(text, voice, tone):
    if not text:
        return None
    body = {"text": text, "voice": voice or "", "tone": tone or ""}
    try:
        async with ClientSession(timeout=ClientTimeout(total=45)) as session:
            async with session.post(TTS + "/synthesize", json=body) as response:
                if response.status != 200:
                    return None
                return await response.read()
    except Exception:
        return None


async def vllm_chat(body):
    target = answer_model()
    if target:
        body["model"] = target
    try:
        async with ClientSession(timeout=ClientTimeout(total=90)) as session:
            async with session.post(VLLM + "/v1/chat/completions", json=body) as response:
                if response.status != 200:
                    return None
                return await response.json()
    except Exception:
        return None


def mulaw_to_wav(payload):
    pcm = audioop.ulaw2lin(payload, 2)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(8000)
        out.writeframes(pcm)
    return buffer.getvalue()


def wav_to_mulaw(payload):
    with wave.open(io.BytesIO(payload), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if channels > 1:
        frames = audioop.tomono(frames, width, 0.5, 0.5)
    if width != 2:
        frames = audioop.lin2lin(frames, width, 2)
    if rate != 8000:
        frames, _ = audioop.ratecv(frames, 2, 1, rate, 8000, None)
    return audioop.lin2ulaw(frames, 2)


class Session:
    """One Twilio stream. Its agent, transcript and tools belong to this call only."""

    def __init__(self, stream_sid, workspace_id, agent_uuid, call_sid):
        self.stream_sid = stream_sid
        self.workspace_id = workspace_id
        self.agent_uuid = agent_uuid
        self.call_sid = call_sid
        self.config = {}
        self.history = []
        self.audio = bytearray()
        self.speech_frames = 0
        self.silence_frames = 0
        self.speaking = False
        self.turn = None
        self.busy = False
        self.generation = 0
        self.started = time.time()
        # Prompt fragment handed back by the app's workflow graph for this turn.
        self.turn_note = ""
        # Last thing the agent said, reported with the next turn so the app can
        # persist it on the call's workflow trace.
        self.last_reply = ""

    def agent(self):
        return (self.config or {}).get("agent") or {}

    def tools(self):
        return (self.config or {}).get("tools") or []

    def system_prompt(self):
        agent = self.agent()
        parts = [agent.get("system_prompt") or "You are a helpful voice assistant."]
        if agent.get("business_purpose"):
            parts.append("Legitimate business purpose: " + str(agent["business_purpose"]))
        if agent.get("guardrails_note"):
            parts.append(str(agent["guardrails_note"]))
        if self.turn_note:
            parts.append("Active workflow step: " + self.turn_note)
        parts.append("You are on a live phone call. Reply in one or two short spoken sentences, plain text, no markdown.")
        return "\\n".join(parts)

    # ---- caller audio in ----
    async def feed(self, ws, payload):
        try:
            pcm = audioop.ulaw2lin(payload, 2)
            level = audioop.rms(pcm, 2)
        except Exception:
            return
        if level >= VOICE_RMS:
            # Barge-in: the caller speaking over the agent stops it immediately.
            if self.speaking:
                await self.interrupt(ws)
            self.audio.extend(payload)
            self.speech_frames += 1
            self.silence_frames = 0
            return
        if not self.speech_frames:
            return
        self.audio.extend(payload)
        self.silence_frames += 1
        if self.silence_frames < SILENCE_FRAMES:
            return
        chunk = bytes(self.audio)
        frames = self.speech_frames
        self.audio = bytearray()
        self.speech_frames = 0
        self.silence_frames = 0
        if frames < MIN_SPEECH_FRAMES or self.busy:
            return
        self.turn = asyncio.create_task(self.respond(ws, chunk))

    async def interrupt(self, ws):
        self.generation += 1
        self.speaking = False
        try:
            await ws.send_json({"event": "clear", "streamSid": self.stream_sid})
        except Exception:
            pass

    # ---- agent audio out ----
    async def say(self, ws, text):
        if not text:
            return
        agent = self.agent()
        wav = await tts_wav(text, agent.get("voice_id"), agent.get("emotional_tone"))
        if not wav:
            return
        try:
            payload = wav_to_mulaw(wav)
        except Exception:
            return
        mark = self.generation
        self.speaking = True
        try:
            for start in range(0, len(payload), FRAME_BYTES):
                if mark != self.generation or ws.closed:
                    break
                chunk = payload[start:start + FRAME_BYTES]
                await ws.send_json({
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": base64.b64encode(chunk).decode("ascii")},
                })
                await asyncio.sleep(0.02)
            if mark == self.generation and not ws.closed:
                await ws.send_json({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": "reply"}})
        finally:
            if mark == self.generation:
                self.speaking = False

    # ---- one conversational turn ----
    async def respond(self, ws, payload):
        self.busy = True
        try:
            heard = await self.transcribe(mulaw_to_wav(payload))
            if not heard:
                return
            try:
                await ws.send_json({"type": "user_text", "text": heard})
            except Exception:
                return
            answer = await self.brain(heard)
            if not answer:
                return
            try:
                await ws.send_json({"type": "assistant_text", "text": answer})
            except Exception:
                return
            await self.say(ws, answer)
        finally:
            self.busy = False

    async def transcribe(self, wav):
        result = await app_json("/api/pod/transcribe", {
            "workspace_id": self.workspace_id,
            "agent_uuid": self.agent_uuid,
            "call_sid": self.call_sid,
            "audio_base64": base64.b64encode(wav).decode("ascii"),
            "mime": "audio/wav",
        }, timeout=45)
        if not result:
            return ""
        return str(result.get("text") or "").strip()

    async def workflow_turn(self, text):
        """One server-side graph step. The app owns the workflow; the pod never does."""
        if not self.workspace_id:
            return None
        return await app_json("/api/pod/workflow-turn", {
            "workspace_id": self.workspace_id,
            "agent_uuid": self.agent_uuid,
            "call_sid": self.call_sid,
            "text": text,
            "previous_reply": self.last_reply,
        }, timeout=25)

    async def brain(self, text):
        # The graph is walked before the model speaks, so this turn already knows
        # which step it is in - or that the workflow ended the call.
        step = await self.workflow_turn(text)
        if step and step.get("ended"):
            closing = str(step.get("closing") or "Thank you for calling. Goodbye.")
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": closing})
            self.last_reply = closing
            return closing
        self.turn_note = str((step or {}).get("prompt_override") or "")
        self.history.append({"role": "user", "content": text})
        tools = self.tools()
        for _ in range(3):
            body = {
                "messages": [{"role": "system", "content": self.system_prompt()}] + self.history[-12:],
                "max_tokens": 300,
            }
            temperature = self.agent().get("temperature")
            if isinstance(temperature, (int, float)):
                body["temperature"] = float(temperature)
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            data = await vllm_chat(body)
            if not data:
                return ""
            choices = data.get("choices") or [{}]
            message = (choices[0] or {}).get("message") or {}
            calls = message.get("tool_calls") or []
            if not calls:
                answer = str(message.get("content") or "").strip()
                self.history.append({"role": "assistant", "content": answer})
                self.last_reply = answer
                return answer
            self.history.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": calls,
            })
            for call in calls:
                function = call.get("function") or {}
                result = await self.run_tool(function.get("name"), function.get("arguments"))
                self.history.append({
                    "role": "tool",
                    "tool_call_id": call.get("id") or "",
                    "name": function.get("name") or "",
                    "content": json.dumps(result)[:1800],
                })
        return ""

    async def run_tool(self, slug, raw_arguments):
        try:
            arguments = json.loads(raw_arguments or "{}")
        except Exception:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        result = await app_json("/api/pod/tool-exec", {
            "workspace_id": self.workspace_id,
            "agent_uuid": self.agent_uuid,
            "call_sid": self.call_sid,
            "slug": slug or "",
            "arguments": arguments,
        }, timeout=20)
        return result or {"ok": False, "error": "tool_unreachable"}


# ---------------------------------------------------------------------------
# Twilio Media Streams
# ---------------------------------------------------------------------------

async def ws_twilio(request):
    ws = web.WebSocketResponse(heartbeat=25, max_msg_size=0)
    await ws.prepare(request)
    session = None
    try:
        async for message in ws:
            if message.type != WSMsgType.TEXT:
                continue
            try:
                event = json.loads(message.data)
            except Exception:
                continue
            kind = event.get("event")

            if kind == "start":
                start = event.get("start") or {}
                params = start.get("customParameters") or {}
                stream_sid = start.get("streamSid") or ""
                workspace_id = str(params.get("workspace_id") or "")
                agent_uuid = str(params.get("agent_id") or params.get("agent_uuid") or "")
                call_sid = str(params.get("call_sid") or start.get("callSid") or "")
                if not stream_sid:
                    continue
                if not workspace_id:
                    await ws.send_json({
                        "type": "assistant_text",
                        "text": "This number is not bound to a workspace yet.",
                    })
                    continue
                # No call is accepted until the Ear, the Brain and the Mouth all
                # answer. The supervisor only starts this relay behind those gates;
                # this check covers a service that died after the pod reported ready.
                deps_ok, missing = await _deps_ready()
                if not deps_ok:
                    await ws.send_json({
                        "type": "deps_not_ready",
                        "detail": missing,
                        "text": "One moment, the assistant is still starting up.",
                    })
                    continue
                session = Session(stream_sid, workspace_id, agent_uuid, call_sid)
                SESSIONS[stream_sid] = session
                session.config = await resolve_config(workspace_id, agent_uuid)
                blocked = str((session.config or {}).get("blocked_reason") or "")
                greeting = session.agent().get("greeting") or ""
                # Spoken as a task, never inline: the receive loop must stay free to
                # read caller audio so the greeting itself can be barged in on.
                if blocked:
                    asyncio.create_task(session.say(ws, blocked))
                elif greeting:
                    await ws.send_json({"type": "assistant_text", "text": greeting})
                    asyncio.create_task(session.say(ws, greeting))

            elif kind == "media" and session:
                payload = (event.get("media") or {}).get("payload") or ""
                if not payload:
                    continue
                try:
                    await session.feed(ws, base64.b64decode(payload))
                except Exception:
                    continue

            elif kind == "stop":
                break
    finally:
        if session and SESSIONS.get(session.stream_sid) is session:
            SESSIONS.pop(session.stream_sid, None)
    return ws


async def ws_audio(request):
    """Browser test widget: text turns, own workspace, no shared agent."""
    ws = web.WebSocketResponse(heartbeat=25, max_msg_size=0)
    await ws.prepare(request)
    workspace_id = request.query.get("workspace_id") or ""
    agent_uuid = request.query.get("agent_id") or ""
    session = Session("widget-" + str(int(time.time())), workspace_id, agent_uuid, "") if workspace_id else None
    if session:
        session.config = await resolve_config(workspace_id, agent_uuid)
    agent = session.agent() if session else {}
    await ws.send_json({
        "type": "ready",
        "agent": agent.get("name") or "",
        "voice_id": agent.get("voice_id") or "",
        "uptime_s": int(time.time() - BOOTED),
    })
    greeting = agent.get("greeting") or ""
    if greeting:
        await ws.send_json({"type": "assistant_text", "text": greeting})
        wav = await tts_wav(greeting, agent.get("voice_id"), agent.get("emotional_tone"))
        if wav:
            await ws.send_bytes(wav)
    async for message in ws:
        if message.type == WSMsgType.TEXT:
            try:
                data = json.loads(message.data)
            except Exception:
                continue
            kind = data.get("type")
            if kind in ("user_text", "text") and session:
                said = str(data.get("text") or "").strip()
                if not said:
                    continue
                await ws.send_json({"type": "user_text", "text": said})
                answer = await session.brain(said)
                if answer:
                    await ws.send_json({"type": "assistant_text", "text": answer})
                    wav = await tts_wav(answer, agent.get("voice_id"), agent.get("emotional_tone"))
                    if wav:
                        await ws.send_bytes(wav)
            elif kind == "ping":
                await ws.send_json({"type": "pong"})
        elif message.type == WSMsgType.ERROR:
            break
    return ws


def _build():
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_get("/logs", logs)
    app.router.add_get("/v1/models", forward_get)
    app.router.add_post("/v1/chat/completions", forward_post)
    app.router.add_get("/tts/voices", forward_tts)
    app.router.add_post("/tts/voices", forward_tts)
    app.router.add_post("/tts/synthesize", forward_tts)
    app.router.add_post("/session/config", session_config)
    app.router.add_get("/session/config", session_config)
    app.router.add_post("/session/end", session_end)
    app.router.add_get("/ws/audio", ws_audio)
    app.router.add_get("/ws", ws_audio)
    app.router.add_get("/twilio", ws_twilio)
    return app


async def main():
    runner = web.AppRunner(_build(), access_log=None)
    await runner.setup()
    try:
        site = web.TCPSite(runner, HOST, PORT)
        await site.start()
    except OSError as error:
        print("voice_relay: cannot bind " + HOST + ":" + str(PORT) + " - " + str(error), file=sys.stderr, flush=True)
        return 1
    print("voice_relay: listening on " + HOST + ":" + str(PORT), flush=True)
    await asyncio.Event().wait()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
