#!/usr/bin/env python3
"""AetherVoice Whisper worker - loopback only, model loads in the background."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "127.0.0.1"
PORT = int(os.environ.get("AETHER_WHISPER_PORT", "8001"))
MODEL = os.environ.get("AETHER_WHISPER_MODEL", "base")
# ready is the MODEL's state, not the process's: /health answers as soon as the
# worker is up, while ok/ready stay false until the model can actually transcribe.
STATE = {"ready": False, "engine": "none", "model": MODEL, "stage": "loading", "detail": "starting"}


def load():
    try:
        from faster_whisper import WhisperModel
    except Exception as error:
        STATE.update(engine="none", stage="failed", detail="faster_whisper_not_installed: " + str(error)[:160])
        return
    first = ""
    try:
        WhisperModel(MODEL, device="cuda", compute_type="float16")
        STATE.update(ready=True, engine="faster-whisper-cuda", stage="ready", detail=MODEL)
        return
    except Exception as error:
        first = str(error)[:160]
    try:
        WhisperModel(MODEL, device="cpu", compute_type="int8")
        STATE.update(ready=True, engine="faster-whisper-cpu", stage="ready", detail=MODEL + " (cuda failed: " + first + ")")
    except Exception as error:
        STATE.update(engine="none", stage="failed", detail="load_failed: " + str(error)[:160])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("whisper: " + (fmt % args) + chr(10))
        sys.stderr.flush()

    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            # 200 as soon as the worker answers, so the supervisor's gate stays fast.
            # ok/ready are the MODEL's verdict and are what the relay's deps gate on.
            ready = bool(STATE["ready"])
            self._send(200, {
                "ok": ready,
                "ready": ready,
                "engine": STATE["engine"],
                "model": STATE["model"],
                "stage": STATE["stage"],
                "detail": STATE["detail"],
            })
            return
        self._send(404, {"error": "not_found"})


if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()
    print("whisper: listening on " + HOST + ":" + str(PORT), flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
