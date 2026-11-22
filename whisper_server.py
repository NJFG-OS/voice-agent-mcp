#!/usr/bin/env python3
"""AetherVoice Whisper worker - 0.0.0.0:8001 (health only), model loads in the background."""
import glob
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 0.0.0.0 because 8001 is a template-declared port: RunPod's proxy connects to the
# pod's interface, never to loopback. Safe to publish because this worker answers
# /health only - it has no inference route that could transcribe for a stranger.
# AETHER_WHISPER_HOST=127.0.0.1 pins it back to loopback for an operator who wants
# the port closed (the relay keeps working either way).
HOST = os.environ.get("AETHER_WHISPER_HOST", "0.0.0.0")
PORT = int(os.environ.get("AETHER_WHISPER_PORT", "8001"))
MODEL = os.environ.get("AETHER_WHISPER_MODEL", "base")
ATTEMPTS = 3
# ready is the MODEL's state, not the process's: /health answers as soon as the
# worker is up, while ok/ready stay false until the model can actually transcribe.
STATE = {
    "ready": False,
    "engine": "none",
    "model": MODEL,
    "stage": "loading",
    "detail": "starting",
    "attempt": 0,
    "repo": "",
    "repaired": 0,
}


def repo_id(model):
    """The Hugging Face repo faster-whisper downloads this model from."""
    try:
        from faster_whisper.utils import _MODELS
        if model in _MODELS:
            return _MODELS[model]
    except Exception:
        pass
    return model if "/" in model else "Systran/faster-whisper-" + model


def cache_dir(repo):
    """This model's own cache folder - the only one this worker ever touches."""
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    owner, _, name = repo.partition("/")
    return os.path.join(home, "hub", "models--" + owner + "--" + (name or owner))


def snapshot_dirs(repo):
    return sorted(glob.glob(os.path.join(cache_dir(repo), "snapshots", "*")))


def _drop(path, removed):
    """Remove one artifact: the snapshot link AND the blob it points at.

    The snapshot entry is a symlink into blobs/. Deleting only the link leaves the
    corrupt blob in place and the hub hands it back on the next download.
    """
    target = ""
    try:
        if os.path.islink(path):
            target = os.path.realpath(path)
    except Exception:
        target = ""
    for candidate in (path, target):
        if not candidate:
            continue
        try:
            if os.path.exists(candidate) or os.path.islink(candidate):
                os.remove(candidate)
                removed.append(os.path.basename(candidate))
        except Exception:
            pass


def scrub_cache(repo, deep):
    """Drop this model's corrupt cache artifacts, never another model's.

    First pass removes only zero-byte files and unparseable JSON. A retry after a
    failed load also drops the weight files, so a truncated model.bin is downloaded
    again instead of failing identically forever.
    """
    removed = []
    for folder in snapshot_dirs(repo):
        try:
            names = sorted(os.listdir(folder))
        except Exception:
            continue
        for name in names:
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            try:
                size = os.path.getsize(path)
            except Exception:
                size = 0
            bad = size == 0
            if not bad and name.endswith(".json"):
                try:
                    with open(path) as handle:
                        json.load(handle)
                except Exception:
                    bad = True
            if not bad and deep and (name.endswith(".bin") or name.endswith(".safetensors")):
                bad = True
            if bad:
                _drop(path, removed)
    return removed


def note_repaired(removed, repo, why):
    if not removed:
        return
    STATE["repaired"] = int(STATE.get("repaired") or 0) + len(removed)
    print(
        "whisper: " + why + " - dropped " + str(len(removed)) + " cache artifact(s) from "
        + repo + ": " + ", ".join(removed[:8]),
        flush=True,
    )


def load():
    """Load the model, repairing this model's cache and retrying before giving up."""
    try:
        from faster_whisper import WhisperModel
    except Exception as error:
        STATE.update(engine="none", stage="failed", detail="faster_whisper_not_installed: " + str(error)[:160])
        print("whisper: faster_whisper is not installed: " + str(error)[:200], flush=True)
        return

    repo = repo_id(MODEL)
    STATE["repo"] = repo
    folders = snapshot_dirs(repo)
    print(
        "whisper: model=" + MODEL + " repo=" + repo + " cache=" + cache_dir(repo)
        + " snapshots=" + str(len(folders)) + " hf_home=" + str(os.environ.get("HF_HOME") or "default"),
        flush=True,
    )
    note_repaired(scrub_cache(repo, False), repo, "pre-load cache check")

    first = ""
    for attempt in range(1, ATTEMPTS + 1):
        STATE["attempt"] = attempt
        if attempt > 1:
            # A failed load is the only evidence that the weights themselves are
            # damaged, so the retry re-downloads them (config included).
            note_repaired(scrub_cache(repo, True), repo, "retry " + str(attempt))
            time.sleep(2)
        for device, compute in (("cuda", "float16"), ("cpu", "int8")):
            try:
                WhisperModel(MODEL, device=device, compute_type=compute)
            except Exception as error:
                note = str(error)[:200]
                if not first:
                    first = note
                print("whisper: attempt " + str(attempt) + " " + device + " failed: " + note, flush=True)
                continue
            detail = MODEL if not first else MODEL + " (" + device + " after " + first + ")"
            STATE.update(ready=True, engine="faster-whisper-" + device, stage="ready", detail=detail[:400])
            print(
                "whisper: ready engine=" + STATE["engine"] + " model=" + MODEL + " attempt=" + str(attempt),
                flush=True,
            )
            return

    STATE.update(engine="none", stage="failed", detail=("load_failed: " + first)[:400])
    print(
        "whisper: model could not be loaded after " + str(ATTEMPTS) + " attempt(s) - the relay reports it unready",
        flush=True,
    )


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
            # 200 as soon as the worker answers, so the supervisor's liveness check
            # stays fast. ok/ready are the MODEL's verdict and are what the relay's
            # deps gate and the supervisor's ready gate both read.
            ready = bool(STATE["ready"])
            self._send(200, {
                "ok": ready,
                "ready": ready,
                "engine": STATE["engine"],
                "model": STATE["model"],
                "stage": STATE["stage"],
                "detail": STATE["detail"],
                "attempt": STATE["attempt"],
                "repo": STATE["repo"],
                "repaired": STATE["repaired"],
            })
            return
        self._send(404, {"error": "not_found"})


if __name__ == "__main__":
    print("whisper: binding " + HOST + ":" + str(PORT) + " (GET /health only)", flush=True)
    threading.Thread(target=load, daemon=True).start()
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
