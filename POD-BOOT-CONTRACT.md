# Pod boot contract

A terminated pod is replaced by a fresh container: `/tmp`, `/etc`, `/opt` and
`/start.sh` come back as the image ships them, and only `/workspace` persists.
Nothing may need a manual fix inside a new pod, so everything that decides a boot
lives in one of two places: the image (this repository) or the app's served
generation (fetched, sha256-verified, onto the container disk).

## Ports

| Port | Service | Bind | Declared on the RunPod template |
| ---- | ------- | ---- | ------------------------------- |
| 8000 | voice_relay.py (call path, /health) | 0.0.0.0 | 8000/http |
| 8001 | whisper_server.py (the Ear, GET /health only) | 0.0.0.0 | 8001/http |
| 8002 | vLLM (run_vllm_safe.py) | 0.0.0.0 | 8002/http |
| 8080 | tts_server.py (the Mouth) | 0.0.0.0 | 8080/http |

Every service binds 0.0.0.0. RunPod's service proxy connects to the pod's own
interface, so a loopback bind answers the relay but leaves
`<pod>-8001.proxy.runpod.net` and `<pod>-8002.proxy.runpod.net` with nothing to
connect to - the port reads as dead from outside while the process inside is
healthy. `AETHER_WHISPER_HOST=127.0.0.1` or `AETHER_VLLM_HOST=127.0.0.1` closes
either port again without touching the relay's own path to it.

Whisper is safe to publish: it serves `GET /health` and nothing else, so the port
exposes readiness, not transcription.

## Where each script comes from

Baked into the image (this repository):

- `Dockerfile` - base image, build-time runtime gate, nginx 8001 block strip,
  file-descriptor limits, transformers/numpy pins.
- `post_start.sh` - the hook `/start.sh` runs once sshd and Jupyter are up.
  Raises `ulimit -n 65536`, clears legacy volume scripts, repairs `/workspace`
  caches without chmod/chown, frees 8001 and 8002, fetches and verifies the served
  generation, then execs `start_all.sh`.

Served by the app (`/api/pod-code`, sha256-verified onto `/tmp/aether/run`):

- `start_all.sh` - supervisor: mount/version/runtime gates, stale worker and stale
  pid/lock cleanup, service launchers, per-service gates, listener proof, ready
  report, watchdog.
- `whisper_server.py`, `run_vllm_safe.py`, `voice_relay.py`, `tts_server.py`.

The copies of those five files in this repository are mirrors of what the app
serves; the app is authoritative and a runtime fix ships without an image rebuild.

## Boot order

1. `/start.sh` (provider) -> `/post_start.sh`.
2. nofile raised, legacy volume scripts removed, stale owned workers stopped.
3. ports 8001 and 8002 proven free; nginx is stopped only if it still holds 8001.
4. served generation fetched, every sha256 matched, tree verified.
5. `start_all.sh`: mount gate -> version gate -> runtime gate (torch, vllm, vllm._C).
6. stale pid/lock/stage/mode files cleared, supervisor lock taken (holder must be a
   live start_all.sh process, never a recycled pid).
7. relay 8000 first, then whisper 8001, vLLM 8002 and TTS 8080 together.
8. gates: vLLM /health, whisper model ready, TTS health.
9. every port re-checked with a TCP connect; verdict written to listeners.json and
   carried in the ready report.
10. watchdog: reports ready, heartbeats, stops the pod after 120 minutes unhealthy.

## Verifying a fresh pod

    ss -tlpn | grep -E ':800[012]|:8080'
    curl -s localhost:8001/health
    curl -s localhost:8002/health
    curl -s localhost:8000/health

Through the provider proxy:

    https://<pod-id>-8000.proxy.runpod.net/health
    https://<pod-id>-8001.proxy.runpod.net/health
    https://<pod-id>-8002.proxy.runpod.net/health

`/workspace/aether/logs/supervisor.log` carries the stage lines and the per-port
verdict; `listeners.json` in the same directory holds the machine-readable answer.

## Pins that must not drift

- `transformers==4.47.1` (runtime-candidates.json) - Kokoro's warm-up fails on the
  version its TTS wheel pulls in under PyTorch 2.4.x. The Dockerfile asserts the
  pin from the finished image.
- `ulimit -n 65536` in the image (limits.conf, profile.d, bash.bashrc), in
  `post_start.sh`, in `start_all.sh` and in each worker - a live call holds a media
  socket plus upstream connections and `[Errno 24]` takes port 8000 with it.
- `/workspace` is storage only: HF cache, pip cache, voices, logs. No executables,
  no chmod, no chown - it is owned by nobody:nogroup and both are refused.
