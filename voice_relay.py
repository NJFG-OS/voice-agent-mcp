#!/usr/bin/env python3
"""AetherVoice voice relay - the ONE public port.

Binds 0.0.0.0:8000. vLLM answers on 8002, TTS on 8080 (published for the provider's
port probe and pod-secret gated; normally reached as <relay>/tts/*).
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
import resource
import sys
import time
import wave

from aiohttp import ClientSession, ClientTimeout, TCPConnector, WSMsgType, web

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

# The file-descriptor budget this process must own. A live call holds a Twilio media
# socket plus the relay's upstream connections, and the image's soft default (1024)
# is exhausted by a handful of concurrent calls - which surfaced as
# "[Errno 24] Too many open files" and took the only public port with it. The
# supervisor raises the limit for its children; this raises it again here, so a relay
# started any other way (a manual run, a self-refresh) still has the same target.
NOFILE_TARGET = int(os.environ.get("AETHER_NOFILE_TARGET") or 65536)


def raise_nofile():
    """Best effort: the soft limit goes up to the target, never above the hard one."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:
        return "unavailable"
    want = NOFILE_TARGET if hard in (resource.RLIM_INFINITY, -1) else min(NOFILE_TARGET, hard)
    if soft >= want:
        return str(soft)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
        return str(want)
    except Exception:
        return str(soft)


NOFILE = raise_nofile()

# ONE pooled HTTP session for the relay's whole life.
#
# Every health poll and every proxy call used to build its own ClientSession, and each
# one left a connector, a socket and a resolver entry behind. Under the app's polling
# that reached the pod's file-descriptor budget and the relay died with
# "[Errno 24] Too many open files" - taking the only public port with it. The session
# is created once, inside the running loop, and reused; a per-call deadline is passed
# on the request instead of on the session.
SESSION = {"client": None}


def session():
    """The shared session, created on first use inside the running loop."""
    client = SESSION["client"]
    if client is None or client.closed:
        client = ClientSession(
            connector=TCPConnector(
                limit=64,
                limit_per_host=16,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            ),
            timeout=ClientTimeout(total=30),
            trust_env=False,
        )
        SESSION["client"] = client
    return client


async def close_session():
    client = SESSION["client"]
    SESSION["client"] = None
    if client is not None and not client.closed:
        await client.close()

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
FRAME_MS = 20
FRAME_SECONDS = FRAME_MS / 1000.0
VOICE_RMS = 420          # speech energy floor on 16-bit PCM


# ---- turn-taking policy ------------------------------------------------------
# Every value below is overridable from the app: the config it pushes carries a turn_taking block, so
# a shop's phone manner is a setting rather than a constant buried in a service. These are the
# defaults the app ships with.
def _ms(name, default):
    try:
        value = int(os.environ.get(name) or default)
    except Exception:
        value = default
    return value if value > 0 else default


SILENCE_HOLD_MS = _ms("HANK_SILENCE_MS", 450)             # an ordinary thought
SILENCE_SIMPLE_MS = _ms("HANK_SILENCE_SIMPLE_MS", 375)    # yes/no, confirmation
SILENCE_DETAIL_MS = _ms("HANK_SILENCE_DETAIL_MS", 700)    # year/make/model, VIN, symptoms, contact
SILENCE_HOLD_LONG_MS = _ms("HANK_SILENCE_HOLD_MS", 2000)  # "hold on", "let me check"
DEAD_AIR_MS = _ms("HANK_DEAD_AIR_MS", 6000)               # check in rather than sit in silence
MIN_SPEECH_MS = _ms("HANK_MIN_SPEECH_MS", 250)            # audio below this is noise, not a turn
BACKCHANNEL_MS = _ms("HANK_BACKCHANNEL_MS", 600)          # "yeah"/"uh-huh" under this is an ack
ECHO_GUARD_MS = _ms("HANK_ECHO_GUARD_MS", 200)            # deaf window after playback / clear
FILLER_AFTER_MS = _ms("HANK_FILLER_AFTER_MS", 350)        # a slower tool earns a cached filler

DEFAULT_POLICY = {
    "silence_ms": SILENCE_HOLD_MS,
    "silence_simple_ms": SILENCE_SIMPLE_MS,
    "silence_detail_ms": SILENCE_DETAIL_MS,
    "silence_hold_ms": SILENCE_HOLD_LONG_MS,
    "dead_air_ms": DEAD_AIR_MS,
    "min_speech_ms": MIN_SPEECH_MS,
    "backchannel_ms": BACKCHANNEL_MS,
    "echo_guard_ms": ECHO_GUARD_MS,
    "filler_after_ms": FILLER_AFTER_MS,
}

# A backchannel is an acknowledgement, never an answer. It only suppresses a barge-in while Hank is
# mid-sentence, so a real "yes" to a question that has finished asking is still a turn.
BACKCHANNELS = {
    "yeah", "yep", "yea", "yes", "yup", "uh", "huh", "uhhuh", "uhhuhh", "mm", "mmhmm", "mhm",
    "hm", "hmm", "ok", "okay", "k", "right", "sure", "alright", "gotcha", "got", "it", "i", "see",
    "thanks", "thank", "you", "no", "nope", "nah",
}
# Words that always take the turn, however short: a correction or a stop is not an acknowledgement.
STOP_WORDS = {"stop", "wait", "cancel", "hold", "actually", "wrong", "listen", "quiet", "shush"}

# The static lines the bridge pre-synthesizes at the start of a call, so they replay with no TTS wait.
# Shop-independent by design: that is exactly what makes them cacheable.
PHRASES = {
    "filler": "One moment, let me check that.",
    "check_in": "Are you still there? Take your time.",
    "hold_ack": "No problem, take your time.",
    "lookup": "Let me pull that up.",
    "triage": "Got it. And how long has that been happening?",
}
# (voice, text) -> 8 kHz mu-law bytes. Filled by warm_phrases() and reused by every call on the pod.
PHRASE_AUDIO = {}


def phrase_audio(text, voice=""):
    return PHRASE_AUDIO.get((voice or "", text))


def frame_count(ms):
    return max(1, int(round(float(ms) / FRAME_MS)))

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
    # The whole runtime the image gated, not just the Brain: /health is the one place
    # an operator can prove which numpy/torch/transformers/whisper/tts a pod really
    # runs, so a pin that drifted is visible without a shell.
    return {
        "torch": data.get("torch", ""),
        "vllm": data.get("vllm", ""),
        "base_image": data.get("base_image", ""),
        "numpy": data.get("numpy", ""),
        "numpy_source": data.get("numpy_source", ""),
        "transformers": data.get("transformers", ""),
        "whisper": data.get("whisper", ""),
        "tts": data.get("tts", ""),
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
        async with session().get(url, timeout=ClientTimeout(total=total)) as response:
            return response.status
    except Exception:
        return 0


async def _vllm_state():
    """loading vs ready: vLLM only answers /v1/models once weights are in VRAM."""
    try:
        async with session().get(VLLM + "/v1/models", timeout=ClientTimeout(total=2.0)) as response:
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
        async with session().get(WHISPER + "/health", timeout=ClientTimeout(total=2.0)) as response:
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
        "nofile": NOFILE,
        "bind": HOST + ":" + str(PORT),
    })


async def forward_get(request):
    path = request.path_qs
    try:
        async with session().get(VLLM + path, timeout=ClientTimeout(total=60)) as response:
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
        async with session().post(
            VLLM + path,
            data=raw,
            headers={"content-type": request.headers.get("content-type", "application/json")},
            timeout=ClientTimeout(total=180),
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
    """The app reaches the speech worker through this one public port.

    Voice catalog, previews and saved clips all live on 8080, which the provider
    probes directly but only answers beyond /health to a caller holding the pod
    secret. Proxying them here keeps the app on a single public port.
    """
    path = request.path_qs
    target = TTS + (path[len("/tts"):] if path.startswith("/tts") else path)
    try:
        if request.method == "GET":
            async with session().get(target, timeout=ClientTimeout(total=90)) as response:
                body = await response.read()
                return web.Response(
                    body=body,
                    status=response.status,
                    content_type=response.content_type or "application/json",
                )
        raw = await request.read()
        async with session().post(
            target,
            data=raw,
            headers={"content-type": request.headers.get("content-type", "application/json")},
            timeout=ClientTimeout(total=90),
        ) as response:
            body = await response.read()
            # A saved clip spoken in a preset voice is reported by the worker as
            # X-Aether-Voice; it is passed through so the app can tell the two apart.
            note = response.headers.get("X-Aether-Voice")
            return web.Response(
                body=body,
                status=response.status,
                content_type=response.content_type or "application/json",
                headers={"X-Aether-Voice": note} if note else None,
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
    # Named state, never "session": that name is the relay's shared HTTP session.
    for sid, state in list(SESSIONS.items()):
        if not workspace_id or state.workspace_id == workspace_id:
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
        async with session().request(
            method, APP + path, headers=headers, timeout=ClientTimeout(total=timeout), **kwargs
        ) as response:
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
        async with session().post(
            TTS + "/synthesize", json=body, timeout=ClientTimeout(total=45)
        ) as response:
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
        async with session().post(
            VLLM + "/v1/chat/completions", json=body, timeout=ClientTimeout(total=90)
        ) as response:
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


CLAUSE_RE = re.compile(r"[^.!?;:]+[.!?;:]*\s*")


def speakable_clauses(text):
    """The reply cut where a person would breathe.

    Synthesizing a whole paragraph before sending one byte made the caller wait for the last
    sentence to hear the first. Each clause is its own TTS job, so audio starts as soon as the
    opening words exist; a short tail is folded into the clause before it instead of becoming a
    one-word job.
    """
    parts = [part.strip() for part in CLAUSE_RE.findall(text or "") if part.strip()]
    if not parts:
        return [text.strip()] if (text or "").strip() else []
    out = []
    for part in parts:
        if out and len(part) < 12:
            out[-1] = (out[-1] + " " + part).strip()
        else:
            out.append(part)
    return out


class Session:
    """One Twilio stream. Its agent, transcript, tools and turn-taking state belong to this call only."""

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
        # Every response is one generation. A barge-in, a clear or a new response retires the old
        # number, and any audio, mark or TTS chunk still arriving under it is dropped.
        self.generation = 0
        self.turn_id = 0
        self.delivered = []
        self.tts_tasks = {}
        self.brain_task = None
        self.dead_air_task = None
        self.hold_ms = SILENCE_HOLD_MS
        self.hold_from_app = 0
        self.echo_until = 0.0
        self.last_activity = time.time()
        self.barge_frames = 0
        self.started = time.time()
        # Prompt fragment handed back by the app's workflow graph for this turn.
        self.turn_note = ""
        # Last thing the agent said, reported with the next turn so the app can
        # persist it on the call's workflow trace.
        self.last_reply = ""
        # The live structured intake, as the app last confirmed it.
        self.intake = {}

    # ---- policy ----
    def policy(self):
        pushed = (self.config or {}).get("turn_taking") or {}
        policy = dict(DEFAULT_POLICY)
        for key, default in DEFAULT_POLICY.items():
            value = pushed.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                policy[key] = int(value)
        return policy

    def log_turn(self, event, **fields):
        """One structured line per turn event: ids, timings and booleans. Never an audio payload."""
        record = {
            "event": event,
            "stream": self.stream_sid,
            "call_sid": self.call_sid,
            "workspace_id": self.workspace_id,
            "turn_id": self.turn_id,
            "generation_id": self.generation,
        }
        record.update({key: value for key, value in fields.items() if value is not None})
        print("voice_relay: " + json.dumps(record, default=str), flush=True)

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
        # The transcript is the only record of what the caller already said, so it is
        # named here: re-asking a captured detail is the most common wasted turn, and
        # every wasted turn is a second of silence on a live call.
        parts.append("Never ask again for anything the caller has already told you in this call - use what they said.")
        return "\n".join(parts)

    def generation_pending(self):
        return bool(self.brain_task and not self.brain_task.done())

    def is_backchannel(self, text):
        """A short standalone acknowledgement: 'yeah', 'uh-huh', 'okay'. Never a correction."""
        words = re.sub(r"[^a-z' ]", " ", (text or "").lower()).split()
        if not words or len(words) > 3:
            return False
        if any(word in STOP_WORDS for word in words):
            return False
        return all(word in BACKCHANNELS for word in words)

    def hold_for(self, text):
        """The silence this reply earns, from the question it just asked."""
        policy = self.policy()
        low = (text or "").lower()
        if re.search(r"hold on|one moment|one sec|just a (moment|second)|let me (check|look|pull|see|find)", low):
            return policy["silence_hold_ms"]
        if re.search(
            r"\b(year|make|model|vin|licen[cs]e plate|plate number|mileage|odometer|symptom|address|"
            r"phone number|callback|e-?mail|best number|spell)\b",
            low,
        ):
            return policy["silence_detail_ms"]
        if re.search(
            r"\b(is that|do you|does it|did you|are you|can you|would you|will you|have you|"
            r"any (noise|lights|warning))\b",
            low,
        ):
            return policy["silence_simple_ms"]
        return policy["silence_ms"]

    # ---- cached phrases ----
    async def warm_phrases(self):
        """Synthesize the static lines once per pod and voice, so they play with no TTS wait."""
        agent = self.agent()
        voice = agent.get("voice_id") or ""
        tone = agent.get("emotional_tone")
        warmed = 0
        for text in PHRASES.values():
            if phrase_audio(text, voice) is not None:
                warmed += 1
                continue
            wav = await tts_wav(text, voice, tone)
            if not wav:
                continue
            try:
                PHRASE_AUDIO[(voice, text)] = wav_to_mulaw(wav)
                warmed += 1
            except Exception:
                continue
        self.log_turn("phrases_warm", cached=warmed, phrases=len(PHRASES))

    # ---- caller audio in ----
    async def feed(self, ws, payload):
        try:
            pcm = audioop.ulaw2lin(payload, 2)
            level = audioop.rms(pcm, 2)
        except Exception:
            return
        now = time.time()
        # ECHO GUARD. Right after playback starts or a clear is sent, what arrives is the line's own
        # echo and the frames already buffered - never a reason for Hank to interrupt himself.
        if now < self.echo_until:
            return
        policy = self.policy()
        if level >= VOICE_RMS:
            if self.dead_air_task and not self.dead_air_task.done():
                self.dead_air_task.cancel()
            self.last_activity = now
            self.audio.extend(payload)
            self.speech_frames += 1
            self.silence_frames = 0
            if self.speaking or self.generation_pending():
                # While Hank is talking only SUSTAINED speech takes the turn: a "yeah" that stops
                # after 300 ms is an acknowledgement, a caller who keeps going is interrupting.
                self.barge_frames += 1
                if self.barge_frames * FRAME_MS >= policy["backchannel_ms"]:
                    self.barge_frames = 0
                    await self.interrupt(ws, reason="sustained_speech")
            return
        if not self.speech_frames:
            return
        self.audio.extend(payload)
        self.silence_frames += 1
        if self.silence_frames * FRAME_MS < self.hold_ms:
            return
        # The hold expired: the caller has finished a thought.
        chunk = bytes(self.audio)
        frames = self.speech_frames
        self.audio = bytearray()
        self.speech_frames = 0
        self.silence_frames = 0
        self.barge_frames = 0
        spoke_ms = frames * FRAME_MS
        if spoke_ms < policy["min_speech_ms"]:
            self.log_turn("noise_ignored", speech_ms=spoke_ms)
            return
        if self.busy:
            return
        heard = ""
        if self.speaking or self.generation_pending():
            # The utterance ended inside the backchannel window. It only counts if it was not just
            # an acknowledgement: Hank's sentence survives a "mm-hm".
            heard = await self.transcribe(mulaw_to_wav(chunk))
            if heard and spoke_ms < policy["backchannel_ms"] and self.is_backchannel(heard):
                self.log_turn("backchannel_ignored", speech_ms=spoke_ms, words=len(heard.split()))
                return
            await self.interrupt(ws, reason="caller_took_turn")
        self.turn_id += 1
        self.log_turn("turn_start", speech_ms=spoke_ms, silence_ms=self.hold_ms)
        self.turn = asyncio.create_task(self.respond(ws, chunk, heard))

    async def interrupt(self, ws, reason="speech"):
        """Stop Hank mid-word: flush the carrier, retire the generation, cancel its work."""
        if not (self.speaking or self.generation_pending()):
            return
        retired = self.generation
        self.generation += 1
        self.speaking = False
        cancelled = 0
        for task in list(self.tts_tasks.values()):
            if not task.done():
                task.cancel()
                cancelled += 1
        self.tts_tasks = {}
        if self.brain_task and not self.brain_task.done():
            self.brain_task.cancel()
            cancelled += 1
        self.brain_task = None
        cleared = await self.send_clear(ws)
        self.log_turn(
            "barge_in",
            reason=reason,
            retired_generation=retired,
            clear_sent=cleared,
            tts_cancelled=cancelled,
            delivered=len(self.delivered),
        )

    async def send_clear(self, ws):
        """Twilio's own flush: everything queued for this stream is dropped, not played later."""
        try:
            await ws.send_json({"event": "clear", "streamSid": self.stream_sid})
        except Exception:
            return False
        self.echo_until = time.time() + self.policy()["echo_guard_ms"] / 1000.0
        return True

    # ---- agent audio out ----
    async def synth(self, text):
        agent = self.agent()
        wav = await tts_wav(text, agent.get("voice_id"), agent.get("emotional_tone"))
        if not wav:
            return None
        try:
            return wav_to_mulaw(wav)
        except Exception:
            return None

    async def say(self, ws, text, generation=None):
        if not text:
            return
        if generation is None:
            self.generation += 1
            generation = self.generation
        voice = self.agent().get("voice_id") or ""
        clauses = speakable_clauses(text)
        self.speaking = True
        self.delivered = []
        # The first frames of a response are the moment the line starts echoing.
        self.echo_until = time.time() + self.policy()["echo_guard_ms"] / 1000.0
        tasks = {}
        try:
            # SENTENCE-LEVEL STREAMING. Every clause is its own TTS job, dispatched now and played in
            # order, so the caller hears the opening words while the rest is still being made.
            for index, clause in enumerate(clauses):
                if phrase_audio(clause, voice) is None:
                    tasks[index] = asyncio.create_task(self.synth(clause))
            self.tts_tasks = tasks
            for index, clause in enumerate(clauses):
                if generation != self.generation or ws.closed:
                    break
                payload = phrase_audio(clause, voice)
                if payload is None:
                    task = tasks.get(index)
                    if task is None:
                        continue
                    try:
                        payload = await asyncio.wait_for(asyncio.shield(task), timeout=15)
                    except Exception:
                        payload = None
                if not payload:
                    continue
                if generation != self.generation or ws.closed:
                    break
                if await self.stream(ws, payload, generation):
                    self.delivered.append(clause)
            if generation == self.generation and not ws.closed:
                await ws.send_json({
                    "event": "mark",
                    "streamSid": self.stream_sid,
                    "mark": {"name": "reply", "generation": generation},
                })
        finally:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            if generation == self.generation:
                self.speaking = False
                self.schedule_dead_air(ws)

    async def stream(self, ws, payload, generation):
        """Send one clause's frames. False the moment the generation it belongs to is retired."""
        for start in range(0, len(payload), FRAME_BYTES):
            if generation != self.generation or ws.closed:
                return False
            chunk = payload[start:start + FRAME_BYTES]
            await ws.send_json({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": base64.b64encode(chunk).decode("ascii")},
            })
            await asyncio.sleep(FRAME_SECONDS)
        return True

    # ---- dead air ----
    def schedule_dead_air(self, ws):
        if self.dead_air_task and not self.dead_air_task.done():
            self.dead_air_task.cancel()
        self.dead_air_task = asyncio.create_task(self.check_in(ws))

    async def check_in(self, ws):
        """A question followed by nothing is a stuck call, not a silent one."""
        policy = self.policy()
        try:
            await asyncio.sleep(policy["dead_air_ms"] / 1000.0)
        except asyncio.CancelledError:
            return
        if ws.closed or self.speaking or self.busy or self.speech_frames:
            return
        waited = int((time.time() - self.last_activity) * 1000)
        if waited < policy["dead_air_ms"]:
            return
        self.log_turn("dead_air_check_in", silence_ms=waited)
        await self.say(ws, PHRASES["check_in"])

    # ---- one conversational turn ----
    async def respond(self, ws, payload, heard=""):
        self.busy = True
        turn_id = self.turn_id
        started = time.time()
        try:
            if not heard:
                heard = await self.transcribe(mulaw_to_wav(payload))
            if not heard:
                self.log_turn("empty_transcript", turn_id=turn_id)
                return
            try:
                await ws.send_json({"type": "user_text", "text": heard})
            except Exception:
                return
            self.brain_task = asyncio.create_task(self.brain(heard))
            policy = self.policy()
            answer = None
            try:
                # A tool that runs long must not be dead air: the cached filler plays while it works.
                answer = await asyncio.wait_for(
                    asyncio.shield(self.brain_task), timeout=policy["filler_after_ms"] / 1000.0
                )
            except asyncio.TimeoutError:
                self.log_turn("filler_played", turn_id=turn_id, wait_ms=int((time.time() - started) * 1000))
                await self.say(ws, PHRASES["filler"])
                answer = await self.brain_task
            finally:
                self.brain_task = None
            if not answer:
                self.log_turn("empty_answer", turn_id=turn_id)
                return
            try:
                await ws.send_json({"type": "assistant_text", "text": answer})
            except Exception:
                return
            self.hold_ms = self.hold_from_app or self.hold_for(answer)
            await self.say(ws, answer)
            self.note_interrupted()
        finally:
            self.busy = False

    def note_interrupted(self):
        """Only the words the caller actually heard belong in the conversation."""
        if not self.delivered:
            return
        spoken = " ".join(self.delivered).strip()
        if not spoken:
            return
        for entry in reversed(self.history):
            if entry.get("role") == "assistant" and entry.get("content"):
                if len(spoken) + 12 < len(str(entry["content"])):
                    entry["content"] = spoken + " [interrupted]"
                    self.last_reply = entry["content"]
                break

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
            "intake": self.intake,
        }, timeout=25)

    async def brain(self, text):
        # The graph is walked before the model speaks, so this turn already knows
        # which step it is in - or that the workflow ended the call.
        step = await self.workflow_turn(text)
        if step:
            # The app is the record for the intake and for the hold this question earns.
            if isinstance(step.get("intake"), dict):
                self.intake = step["intake"]
            hold = step.get("silence_hold_ms")
            if isinstance(hold, (int, float)) and not isinstance(hold, bool) and hold > 0:
                self.hold_from_app = int(hold)
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
                # Spoken turns are short by contract: a smaller cap is a shorter wait
                # before the first syllable, and the reply is trimmed by the prompt.
                "max_tokens": 220,
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
            "intake": self.intake,
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
                # TENANT FENCE. This pod was deployed for exactly one workspace and
                # serves exactly one adapter (AETHER_TENANT_ID travels in its env), so
                # a stream for any other workspace is refused here rather than answered
                # with this business's fine-tune. Empty env (a manually launched pod)
                # keeps the old per-stream behaviour.
                owner = (os.environ.get("AETHER_WORKSPACE_ID") or "").strip()
                if owner and workspace_id != owner:
                    print("voice_relay: refused foreign workspace " + workspace_id + " (this pod serves " + owner + ")", flush=True)
                    await ws.send_json({
                        "type": "assistant_text",
                        "text": "This assistant is not available on this number.",
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
                asyncio.create_task(session.warm_phrases())
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


async def diag(request):
    """One line's readiness, as booleans: no secret, no endpoint address, no audio."""
    deps_ok, missing = await _deps_ready()
    return web.json_response({
        "ok": True,
        "shop_id": request.query.get("shop_id") or "",
        "uptime_s": int(time.time() - BOOTED),
        "deps_ready": deps_ok,
        "deps_missing": missing,
        "sessions": len(SESSIONS),
        "phrases_cached": len(PHRASE_AUDIO),
        "policy": DEFAULT_POLICY,
        "model": answer_model(),
        "runtime": _runtime_pins(),
    })

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
    app.router.add_get("/diag", diag)
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
    print("voice_relay: listening on " + HOST + ":" + str(PORT) + " nofile=" + NOFILE, flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        # One session for the process, closed once: its sockets go with it.
        await close_session()
        await runner.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
