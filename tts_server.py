#!/usr/bin/env python3
"""AetherVoice TTS worker - loopback only (127.0.0.1:8080).

Kokoro first, Chatterbox as the fallback. The engine is proven by a real warm-up
synthesis, not by an import: /health answers 200 only once this worker can actually
render speech, so the relay's tts gate means capability, not presence.

GET  /voices      the catalog: preset speakers + every saved reference clip
POST /voices      save a 10-30s wav (+ transcript) as a reusable voice
POST /synthesize  render speech with a preset speaker id or a saved clip
"""
import base64
import io
import json
import os
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "127.0.0.1"
# 8080 matches the RunPod template's declared ports; loopback keeps speech internal.
PORT = 8080
_CACHE = {"engine": None, "chatterbox": None, "kokoro": {}}
# The warm-up's verdict. ready stays false until a real synthesis has succeeded, so
# an importable module can never be reported as a working voice.
_STATE = {"ready": False, "detail": "starting", "checked_at": ""}

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
    """A saved clip wins; otherwise the engine renders its own preset speaker."""
    reference = reference_for(voice)
    if reference:
        try:
            return synth_chatterbox(text, reference)
        except Exception:
            pass
    try:
        return synth_kokoro(text, voice, speed)
    except Exception:
        return None


def warm():
    """Prove the engine can render speech before anything depends on it.

    An importable module is not a working voice: the weights may not be cached, the
    GPU may be out of memory, or the package may be a stub. One short synthesis is
    the only evidence that lets /health report ready.
    """
    name = engine()
    if name == "none":
        _STATE.update(ready=False, detail="no TTS engine installed")
        print("tts: no engine installed", flush=True)
        return
    try:
        if name == "chatterbox":
            audio = synth_chatterbox("Aether voice ready.", None)
        else:
            audio = synth_kokoro("Aether voice ready.", "af_heart", 1.0)
    except Exception as error:
        _STATE.update(ready=False, detail=(name + ": " + str(error))[:200])
        print("tts: warm-up failed " + str(error)[:200], flush=True)
        return
    ok = bool(audio)
    _STATE.update(
        ready=ok,
        detail="" if ok else name + ": warm-up produced no audio",
        checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    print("tts: warm-up " + ("ready" if ok else "failed") + " engine=" + name, flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("tts: " + (fmt % args) + chr(10))
        sys.stderr.flush()

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
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
            })
            return
        if path == "/voices":
            self._json(200, {
                "ok": True,
                "engine": engine(),
                "dir": voice_dir(),
                "preview_text": PREVIEW_TEXT,
                "voices": catalog(),
            })
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self):
        path = self.path.split("?")[0]
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
        try:
            audio = synth(text, data.get("voice"), float(data.get("speed") or 1.0))
        except Exception as error:
            self._json(502, {"error": "tts_failed", "detail": str(error)[:200]})
            return
        if not audio:
            self._json(501, {"error": "tts_unavailable", "engine": engine()})
            return
        self._send(200, audio, "audio/wav")

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
    print("tts: engine=" + engine() + " listening on " + HOST + ":" + str(PORT) + " (warming)", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
