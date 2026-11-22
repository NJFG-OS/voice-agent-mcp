#!/usr/bin/env python3
"""AetherVoice TTS worker - 0.0.0.0:8080, pod-secret gated.

Kokoro first, Chatterbox as the fallback. The engine is proven by a real warm-up
synthesis, not by an import: /health answers 200 only once this worker can actually
render speech, so the relay's tts gate means capability, not presence.

GET  /voices      the catalog: preset speakers + every saved reference clip
POST /voices      save a 10-30s wav (+ transcript) as a reusable voice
POST /synthesize  render speech with a preset speaker id or a saved clip
"""
import base64
import hmac
import io
import json
import os
import resource
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "0.0.0.0"
# 8080 matches the RunPod template's declared ports. It binds 0.0.0.0 because the
# provider's own HTTP probe only sees a port bound on all interfaces: a loopback bind
# stayed on Initializing forever. The bind is a readiness fix, not a new public
# surface - loopback may use the worker freely, and any other caller must present the
# pod secret before it can synthesize speech or write a voice.
PORT = 8080
SECRET = os.environ.get("WEBHOOK_SECRET") or os.environ.get("AETHER_WEBHOOK_SECRET") or ""
_CACHE = {"engine": None, "chatterbox": None, "kokoro": {}}
# The warm-up's verdict. ready stays false until a real synthesis has succeeded, so
# an importable module can never be reported as a working voice.
_STATE = {"ready": False, "detail": "starting", "checked_at": ""}

# ThreadingHTTPServer is one thread per request, so the worker's own file-descriptor
# budget matters as much as the relay's. The supervisor raises it for its children;
# this raises it again for any other way the worker is started.
NOFILE_TARGET = int(os.environ.get("AETHER_NOFILE_TARGET") or 65536)
# How long /synthesize waits for a warm-up that is still running before answering 503.
# A caller that arrives during a cold start gets speech a moment later instead of an
# error it would have to retry by hand.
WARM_WAIT = int(os.environ.get("AETHER_TTS_WARM_WAIT") or 25)
# Why the last synthesis fell back (a saved clip with no clone engine, or a failed
# clone), reported as a response header so the app can tell a real clone from a preset.
_LAST = {"fallback": ""}


def raise_nofile():
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

VOICE_DIR = os.environ.get("AETHER_VOICE_DIR", "/workspace/tts_voices")
# Tenant voices live under /workspace/tenants/<id>/voices. They are listed beside
# the shared presets so one business never silently borrows another's clips.
TENANT_ROOT = os.environ.get("AETHER_TENANT_ROOT", "/workspace/tenants")
TENANT_ID = os.environ.get("AETHER_TENANT_ID", "")
PREVIEW_TEXT = "Hi, this is how I sound. I can help with your call today."

# The preset speakers the engine can actually render, across US and UK voices.
BUILTIN = [
    {"id": "af_heart", "name": "Heart", "gender": "female", "language": "en-US", "note": "Warm, conversational"},
    {"id": "af_bella", "name": "Bella", "gender": "female", "language": "en-US", "note": "Bright, upbeat"},
    {"id": "af_nicole", "name": "Nicole", "gender": "female", "language": "en-US", "note": "Soft, low-key"},
    {"id": "af_sarah", "name": "Sarah", "gender": "female", "language": "en-US", "note": "Clear, professional"},
    {"id": "af_sky", "name": "Sky", "gender": "female", "language": "en-US", "note": "Light, friendly"},
    {"id": "am_adam", "name": "Adam", "gender": "male", "language": "en-US", "note": "Steady, trustworthy"},
    {"id": "am_michael", "name": "Michael", "gender": "male", "language": "en-US", "note": "Deep, assured"},
    {"id": "bf_emma", "name": "Emma", "gender": "female", "language": "en-GB", "note": "Refined British"},
    {"id": "bf_isabella", "name": "Isabella", "gender": "female", "language": "en-GB", "note": "Crisp British"},
    {"id": "bm_george", "name": "George", "gender": "male", "language": "en-GB", "note": "Formal British"},
    {"id": "bm_lewis", "name": "Lewis", "gender": "male", "language": "en-GB", "note": "Calm British"},
]
KNOWN = set(item["id"] for item in BUILTIN)


def tenant_voice_dir():
    if not TENANT_ID:
        return ""
    path = os.path.join(TENANT_ROOT, TENANT_ID, "voices")
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        return ""
    return path


def voice_dir():
    # A saved clip goes to the tenant's own folder; without a tenant it stays shared.
    path = tenant_voice_dir() or VOICE_DIR
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path


def voice_dirs():
    return [d for d in (tenant_voice_dir(), VOICE_DIR) if d]


def safe_id(raw):
    return "".join(ch for ch in str(raw or "") if ch.isalnum() or ch in "-_").strip("-_")[:48]


def read_text(path):
    try:
        with open(path, "r", errors="replace") as handle:
            return handle.read().strip()[:600]
    except Exception:
        return ""


def reference_for(voice_id):
    """A saved clip is the clone prompt. Without one there is nothing to clone."""
    voice_id = safe_id(voice_id)
    if not voice_id:
        return None
    for folder in voice_dirs():
        wav = os.path.join(folder, voice_id + ".wav")
        if os.path.exists(wav) and os.path.getsize(wav) > 1024:
            return {"wav": wav, "text": read_text(os.path.join(folder, voice_id + ".txt"))}
    return None


def catalog():
    """Preset speakers plus every saved reference clip, in one shape for the app."""
    voices = [dict(item, kind="builtin") for item in BUILTIN]
    seen = set(item["id"] for item in BUILTIN)
    for folder in voice_dirs():
        try:
            names = sorted(os.listdir(folder))
        except Exception:
            names = []
        for name in names:
            if not name.endswith(".wav"):
                continue
            voice_id = safe_id(name[:-4])
            if not voice_id or voice_id in seen:
                continue
            seen.add(voice_id)
            prompt = read_text(os.path.join(folder, voice_id + ".txt"))
            voices.append({
                "id": voice_id,
                "name": voice_id.replace("_", " ").title(),
                "gender": "",
                "language": "",
                "note": prompt[:120] or "Saved reference clip",
                "kind": "reference",
            })
    return voices


def engine():
    if _CACHE["engine"] is not None:
        return _CACHE["engine"]
    name = "none"
    try:
        import chatterbox  # noqa: F401
        name = "chatterbox"
    except Exception:
        try:
            import kokoro  # noqa: F401
            name = "kokoro"
        except Exception:
            name = "none"
    _CACHE["engine"] = name
    return name


def pcm_wav(samples, rate):
    import numpy as np
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("int16").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(int(rate) or 24000)
        out.writeframes(pcm)
    return buffer.getvalue()


def kokoro_engine(speaker):
    """One pipeline per language, kept: rebuilding it per reply is a model reload."""
    lang = "b" if str(speaker).startswith("b") else "a"
    cached = _CACHE["kokoro"].get(lang)
    if cached is None:
        from kokoro import KPipeline
        cached = KPipeline(lang_code=lang)
        _CACHE["kokoro"][lang] = cached
    return cached


def synth_kokoro(text, voice, speed=1.0):
    import numpy as np

    speaker = voice if voice in KNOWN else "af_heart"
    pipeline = kokoro_engine(speaker)
    chunks = []
    for _, _, audio in pipeline(text, voice=speaker, speed=float(speed or 1.0)):
        chunks.append(np.asarray(audio, dtype="float32"))
    if not chunks:
        return None
    return pcm_wav(np.concatenate(chunks), 24000)


def synth_chatterbox(text, reference=None):
    """Zero-shot clone from a saved clip; without one its own speaker is used."""
    import numpy as np
    from chatterbox.tts import ChatterboxTTS

    model = _CACHE.get("chatterbox")
    if model is None:
        model = ChatterboxTTS.from_pretrained(device="cuda")
        _CACHE["chatterbox"] = model
    if reference and reference.get("wav"):
        audio = model.generate(text, audio_prompt_path=reference["wav"])
    else:
        audio = model.generate(text)
    samples = np.asarray(audio.squeeze().cpu().numpy(), dtype="float32")
    return pcm_wav(samples, int(getattr(model, "sr", 24000) or 24000))


def synth(text, voice, speed=1.0):
    """A saved clip wins; otherwise the engine renders its own preset speaker.

    A saved clip is a CLONE prompt: only the clone engine can speak it. Falling
    straight through to a preset speaker kept audio flowing, but it silently spoke
    someone else's voice with nothing in the response to say so - which is
    indistinguishable from a broken clone. The fallback is recorded and returned as a
    header, so the app (and an operator reading tts.log) can see it.
    """
    _LAST["fallback"] = ""
    reference = reference_for(voice)
    if reference:
        if engine() == "chatterbox":
            try:
                return synth_chatterbox(text, reference)
            except Exception as error:
                _LAST["fallback"] = "clone_failed"
                print("tts: clone failed for " + str(voice) + ": " + str(error)[:160], flush=True)
        else:
            _LAST["fallback"] = "clone_engine_unavailable"
            print(
                "tts: voice " + str(voice) + " is a saved clip but no clone engine is installed - using a preset speaker",
                flush=True,
            )
    try:
        return synth_kokoro(text, voice, speed)
    except Exception:
        return None


KOKORO_REPO = "hexgrad/Kokoro-82M"


def json_blob_ok(path):
    """A cached blob left empty by an interrupted download is served forever."""
    try:
        if not path or not os.path.exists(path) or os.path.getsize(path) < 2:
            return False
        with open(path) as handle:
            json.load(handle)
        return True
    except Exception:
        return False


def repair_kokoro_cache():
    """Drop a corrupt cached config.json and fetch it again.

    HF_HOME lives on the network volume, so a pod stopped mid-download leaves a
    0-byte blob behind and the hub keeps treating it as a cache hit. The pipeline
    then dies inside json.load with "Expecting value: line 1 column 1 (char 0)" -
    an error that names neither the file nor the cause, and it fails identically on
    every later boot until that blob is gone.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return ""
    for _ in range(2):
        try:
            path = hf_hub_download(repo_id=KOKORO_REPO, filename="config.json")
        except Exception as error:
            return "config.json download failed: " + str(error)[:140]
        if json_blob_ok(path):
            return ""
        try:
            size = os.path.getsize(path)
        except Exception:
            size = 0
        print("tts: dropping corrupt cached config.json (" + str(size) + " bytes)", flush=True)
        try:
            os.remove(path)
        except Exception:
            return "the cached config.json cannot be replaced"
    return "config.json is still unreadable after a re-download"


def warm():
    """Prove the engine can render speech before anything depends on it.

    An importable module is not a working voice: the weights may not be cached, the
    GPU may be out of memory, or the package may be a stub. One short synthesis is
    the only evidence that lets /health report ready. A corrupt cached config is
    repaired first, and a failed attempt is retried: a cold download can fail once
    and succeed on the next try.
    """
    name = engine()
    if name == "none":
        _STATE.update(ready=False, detail="no TTS engine installed")
        print("tts: no engine installed", flush=True)
        return
    if name == "kokoro":
        note = repair_kokoro_cache()
        if note:
            print("tts: cache repair: " + note, flush=True)
    note = ""
    for attempt in (1, 2, 3):
        try:
            if name == "chatterbox":
                audio = synth_chatterbox("Aether voice ready.", None)
            else:
                audio = synth_kokoro("Aether voice ready.", "af_heart", 1.0)
        except Exception as error:
            note = str(error)[:200]
            print("tts: warm-up attempt " + str(attempt) + " failed: " + note, flush=True)
            if attempt < 3:
                time.sleep(3)
            continue
        if audio:
            _STATE.update(
                ready=True,
                detail="",
                checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            print("tts: warm-up ready engine=" + name, flush=True)
            return
        note = "warm-up produced no audio"
    _STATE.update(
        ready=False,
        detail=(name + ": " + note)[:200],
        checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    print("tts: warm-up failed " + note, flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("tts: " + (fmt % args) + chr(10))
        sys.stderr.flush()

    def _send(self, status, body, content_type, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _allowed(self):
        """Loopback may use the worker; anyone else needs the pod secret.

        The port is published so the provider can probe it, so /health (and / for a
        bare probe) answer without a secret and expose nothing but readiness.
        Synthesis and voice writes spend GPU time and write to the volume, so those
        require the secret the app and the relay already carry.
        """
        peer = str(self.client_address[0] if self.client_address else "")
        if peer in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
            return True
        if self.path.split("?")[0] in ("/", "/health"):
            return True
        if not SECRET:
            return False
        given = (self.headers.get("Authorization") or "").replace("Bearer", "").strip()
        given = given or (self.headers.get("x-webhook-secret") or "").strip()
        return bool(given) and hmac.compare_digest(given, SECRET)

    def do_GET(self):
        path = self.path.split("?")[0]
        if not self._allowed():
            self._json(401, {"error": "unauthorized"})
            return
        if path == "/":
            self._json(200, {"ok": True, "service": "aether-tts", "engine": engine()})
            return
        if path == "/health":
            # 200 only once a real synthesis succeeded: the relay's tts gate is
            # capability, so a module that imports but cannot render is not ready.
            ready = bool(_STATE.get("ready"))
            self._json(200 if ready else 503, {
                "ok": ready,
                "ready": ready,
                "engine": engine(),
                "detail": _STATE.get("detail") or "",
                "checked_at": _STATE.get("checked_at") or "",
                "voices": len(catalog()),
                "clone_engine": engine() == "chatterbox",
                "nofile": NOFILE,
            })
            return
        if path == "/voices":
            self._json(200, {
                "ok": True,
                "engine": engine(),
                "clone_engine": engine() == "chatterbox",
                "dir": voice_dir(),
                "preview_text": PREVIEW_TEXT,
                "voices": catalog(),
            })
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        if not self._allowed():
            self._json(401, {"error": "unauthorized"})
            return
        length = int(self.headers.get("content-length") or 0)
        try:
            data = json.loads(self.rfile.read(length) or b"{}") if length else {}
        except Exception:
            self._json(400, {"error": "invalid_json"})
            return

        if path == "/voices":
            self._save_voice(data)
            return
        if path != "/synthesize":
            self._json(404, {"error": "not_found"})
            return

        text = str(data.get("text") or "").strip()[:1200]
        if not text:
            self._json(400, {"error": "text_required"})
            return
        # A cold start answers /health 503 until a real synthesis succeeds. A caller
        # that arrives in that window waits for the warm-up instead of being told the
        # voice is unavailable and having to retry by hand.
        waited = 0.0
        while not _STATE.get("ready") and engine() != "none" and waited < WARM_WAIT:
            time.sleep(0.5)
            waited += 0.5
        try:
            audio = synth(text, data.get("voice"), float(data.get("speed") or 1.0))
        except Exception as error:
            self._json(502, {"error": "tts_failed", "detail": str(error)[:200]})
            return
        if not audio:
            # 501 means this image has no engine at all; 503 means the engine is here
            # but not speaking yet, which is a retry, not a missing model.
            if engine() == "none":
                self._json(501, {"error": "tts_unavailable", "engine": "none"})
            else:
                self._json(503, {
                    "error": "tts_warming",
                    "engine": engine(),
                    "detail": _STATE.get("detail") or "warm-up still running",
                })
            return
        headers = {}
        if _LAST.get("fallback"):
            headers["X-Aether-Voice"] = _LAST["fallback"]
        self._send(200, audio, "audio/wav", headers)

    def _save_voice(self, data):
        """A saved clip lives on the pod's volume, so it survives every restart."""
        voice_id = safe_id(data.get("id") or data.get("name"))
        encoded = str(data.get("audio_base64") or "")
        if not voice_id:
            self._json(400, {"error": "id_required"})
            return
        try:
            audio = base64.b64decode(encoded)
        except Exception:
            self._json(400, {"error": "audio_invalid"})
            return
        if len(audio) < 8000 or len(audio) > 12 * 1024 * 1024:
            self._json(400, {"error": "audio_size", "detail": "Upload a 10-30 second .wav sample."})
            return
        try:
            with open(os.path.join(voice_dir(), voice_id + ".wav"), "wb") as handle:
                handle.write(audio)
            prompt = str(data.get("transcript") or "").strip()[:600]
            if prompt:
                with open(os.path.join(voice_dir(), voice_id + ".txt"), "w") as handle:
                    handle.write(prompt)
        except Exception as error:
            self._json(500, {"error": "save_failed", "detail": str(error)[:200]})
            return
        self._json(200, {"ok": True, "id": voice_id, "bytes": len(audio), "voices": len(catalog())})


if __name__ == "__main__":
    # Warm in the background: the port answers immediately so the supervisor's gate
    # sees a live worker, while /health stays 503 until the model can really speak.
    threading.Thread(target=warm, daemon=True).start()
    print("tts: engine=" + engine() + " listening on " + HOST + ":" + str(PORT) + " nofile=" + NOFILE + " (warming)", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
