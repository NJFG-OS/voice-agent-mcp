#!/bin/bash
# AetherVoice start_all.sh - the image's own runtime, deterministic generation.
#
# Run by /post_start.sh (which the image's /start.sh runs once SSH and Jupyter are
# up), from the sha256-verified served tree on the container disk.
#
# /workspace is persistent storage ONLY: HF cache, voices, logs. No executables, no
# venv, no chmod that can fail a boot. vLLM is baked into the image.
set -uo pipefail

WS="${AETHER_WORKSPACE_DIR:-/workspace}"

# Logs first: a boot that fails before this point has nowhere to explain itself.
# No chmod - the directory may be owned by another uid and chmod would only add
# a failing command to the boot.
LOG_DIR="${WS}/aether/logs"
mkdir -p "$LOG_DIR" 2>/dev/null || true
if [ ! -w "$LOG_DIR" ]; then
  LOG_DIR=/tmp/aether-logs
  mkdir -p "$LOG_DIR" 2>/dev/null || true
fi

SUPERVISOR_LOG="${LOG_DIR}/supervisor.log"
DIAG="${LOG_DIR}/boot_diag.log"
FAIL_FILE="${LOG_DIR}/boot_failed.json"
LEGACY_FAIL_FILE="${LOG_DIR}/boot_fail.json"
RUNTIME_PROBE_FILE="${LOG_DIR}/runtime-probe.txt"
# The stage this boot is in, written atomically before every long step. The relay
# serves it from /health, so a stalled boot is diagnosable over port 8000 alone.
STAGE_FILE="${LOG_DIR}/stage.json"
# Written once, the first time a stall inside the boot window is provable, so the
# early failure report is not repeated on every poll.
STALL_FILE="${LOG_DIR}/boot_stalled.flag"

# The generation this script IS. /api/pod-code substitutes its served version here,
# so the pin is compared against the bytes about to run, not a guess.
SERVED_VERSION="test"
PINNED_VERSION="${AETHER_POD_VERSION:-}"

# What the image's build-time gate selected. Read for diagnostics only.
PINS_FILE="${AETHER_RUNTIME_PINS:-/opt/aether/runtime-pins.json}"
GPU_UTIL="${AETHER_GPU_MEMORY_UTILIZATION:-0.65}"
MAX_MODEL_LEN="${AETHER_MAX_MODEL_LEN_24:-4096}"
MAX_NUM_SEQS="${AETHER_MAX_NUM_SEQS_24:-4}"

RELAY_PORT=8000
WHISPER_PORT=8001
VLLM_PORT=8002
TTS_PORT=8080

BOOT_WINDOW="${AETHER_BOOT_WINDOW:-900}"
WHISPER_TIMEOUT="${AETHER_WHISPER_BOOT_TIMEOUT:-420}"
# Weights download on a cold cache: the window has to cover ~16GB over fuse.ge.
VLLM_TIMEOUT="${AETHER_VLLM_BOOT_TIMEOUT:-${AETHER_BOOT_WINDOW:-1800}}"
# A short note only: the worker answers its port immediately and its /health turns
# 200 once a real synthesis succeeds, which is what the relay gates on. Waiting the
# full warm-up here would delay the relay for no benefit.
TTS_TIMEOUT="${AETHER_TTS_BOOT_TIMEOUT:-60}"
RELAY_TIMEOUT="${AETHER_RELAY_BOOT_TIMEOUT:-180}"
MIN_VRAM_MB="${AETHER_MIN_VRAM_MB:-23000}"

# The served model: the first preference the app sent, else the explicit model.
MODEL="${AETHER_VLLM_MODEL:-}"
if [ -z "$MODEL" ] && [ -n "${AETHER_VLLM_MODEL_PREFS:-}" ]; then
  MODEL="$(printf '%s' "${AETHER_VLLM_MODEL_PREFS}" | cut -d, -f1 | tr -d ' ')"
fi
[ -n "$MODEL" ] || MODEL="meta-llama/Llama-3.1-8B-Instruct"

GPU_NAME=""
VRAM_GB=0
VRAM_MB=0
PY_BIN="${AETHER_PYTHON:-}"

APP="${AETHER_APP_ORIGIN:-}"
POD="${RUNPOD_POD_ID:-}"
WS_ID="${AETHER_WORKSPACE_ID:-}"
NODE_ID="${AETHER_NODE_ID:-}"
SECRET="${WEBHOOK_SECRET:-${AETHER_WEBHOOK_SECRET:-}}"

export HF_HOME="${HF_HOME:-${WS}/hf-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WS}/pip-cache}"
export PYTHONUNBUFFERED=1
export AETHER_VLLM_URL="http://127.0.0.1:${VLLM_PORT}"
export AETHER_TTS_URL="http://127.0.0.1:${TTS_PORT}"
export AETHER_WHISPER_URL="http://127.0.0.1:${WHISPER_PORT}"
export AETHER_VLLM_MODEL="${MODEL}"
export AETHER_VLLM_PORT="${VLLM_PORT}"
export AETHER_WHISPER_PORT="${WHISPER_PORT}"
export AETHER_SERVED_MODEL="${MODEL}"
export AETHER_LOG_DIR="${LOG_DIR}"
export AETHER_GPU_MEMORY_UTILIZATION="${GPU_UTIL}"
export AETHER_MAX_MODEL_LEN_24="${MAX_MODEL_LEN}"
export AETHER_MAX_NUM_SEQS_24="${MAX_NUM_SEQS}"
export AETHER_RUNTIME_PINS="${PINS_FILE}"

# One log for the whole boot, mirrored into boot_diag.log for the diagnostics panel.
log() {
  local line="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
  printf '%s\n' "$line" >> "$SUPERVISOR_LOG" 2>/dev/null || printf '%s\n' "$line"
  printf '%s\n' "$line" >> "$DIAG" 2>/dev/null || true
}

hold() {
  log "container held open for inspection: $1"
  sleep infinity
}

# Announce the step this boot is in. Atomic (write then rename), so the relay never
# reads half a record. This is the file /health reports as "stage".
stage() {
  local name="$1" detail="${2:-}"
  printf '{"stage":"%s","detail":"%s","since":"%s"}' \
    "$name" "${detail//\"/}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${STAGE_FILE}.tmp" 2>/dev/null && mv "${STAGE_FILE}.tmp" "$STAGE_FILE" 2>/dev/null
  log "stage: $name${detail:+ ($detail)}"
}

# ---------------------------------------------------------------------------
# Stale Aether workers. Ownership is decided by the process's OWN COMMAND LINE,
# never by a port number: killing whatever listens on 8000-8080 could take down a
# listener that is not ours. A process is ours only when its command line names one
# of this pod's roots AND one of its served scripts, and it is never this script.
# ---------------------------------------------------------------------------
owned_worker_pids() {
  local dir pid cmd marker
  [ -d /proc ] || return 0
  for dir in /proc/[0-9]*; do
    pid="${dir#/proc/}"
    [ "$pid" = "$$" ] && continue
    cmd="$(tr '\0' ' ' < "$dir/cmdline" 2>/dev/null || true)"
    [ -n "$cmd" ] || continue
    # The engine is this image's own: nothing else in the pod runs vLLM.
    case "$cmd" in
      *vllm.entrypoints.openai.api_server*) printf '%s\n' "$pid"; continue ;;
    esac
    # Every other worker is ours only by SCRIPT NAME...
    case "$cmd" in
      *voice_relay.py*|*tts_server.py*|*whisper_server.py*|*run_vllm_safe.py*|*train_lora.py*) ;;
      *) continue ;;
    esac
    # ...AND when it also names one of this pod's own roots. An unset root is skipped,
    # never turned into a wildcard that would match every process on the box.
    for marker in "${AETHER_ROOT:-}" "${AETHER_SCRIPT_CACHE:-/tmp/aether/run}" "/opt/aether" "/tmp/aether" "${WS}/aether"; do
      [ -n "$marker" ] || continue
      case "$cmd" in
        *"$marker"*) printf '%s\n' "$pid"; break ;;
      esac
    done
  done
}

clear_stale_workers() {
  local pids pid
  pids="$(owned_worker_pids)"
  [ -n "$pids" ] || return 0
  # TERM first, so a launcher forwards it to its vLLM child instead of orphaning it
  # on the GPU; KILL only for whatever ignored it.
  for pid in $pids; do kill -TERM "$pid" 2>/dev/null || true; done
  sleep 2
  for pid in $pids; do
    kill -0 "$pid" 2>/dev/null || continue
    kill -9 "$pid" 2>/dev/null || true
  done
}
clear_stale_workers
log "stale Aether workers cleared (ownership by command line, never by port)"

# Every boot starts its service logs empty. The logs live on the persistent volume,
# so without this a failure's tail mixes generations - an old vLLM traceback reads
# as this boot's crash and sends the diagnosis down the wrong path.
for _svc in whisper vllm tts relay runtime supervisor watchdog; do : > "${LOG_DIR}/${_svc}.log" 2>/dev/null || true; done
rm -f "$FAIL_FILE" "$LEGACY_FAIL_FILE" 2>/dev/null || true
: > "$DIAG" 2>/dev/null || true

# ---------------------------------------------------------------------------
# The failure record. boot_failed.json is the machine-readable one the app and an
# operator read; boot_fail.json is kept because existing tooling already reads it.
# Both carry the vLLM tail, which is the answer in almost every case.
# ---------------------------------------------------------------------------
write_fail() {
  local stage="$1" reason="$2" detail="${3:-}" py
  py="${PY_BIN:-}"
  [ -n "$py" ] || py="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
  [ -n "$py" ] || return 0
  "$py" - "$FAIL_FILE" "$LEGACY_FAIL_FILE" "$stage" "$reason" "$detail" "$LOG_DIR" "$MODEL" "$POD" "$SERVED_VERSION" <<'PY'
import datetime, json, os, sys
path, legacy, stage, reason, detail, log_dir, model, pod, version = sys.argv[1:10]


def tail(name, lines):
    try:
        with open(os.path.join(log_dir, name + ".log"), errors="replace") as handle:
            return handle.read().splitlines()[-lines:]
    except Exception:
        return []


data = {
    "stage": stage,
    "reason": reason,
    "detail": detail[:4000],
    "pod_version": version,
    "model": model,
    "pod_id": pod,
    "log_dir": log_dir,
    "vllm_log_tail": tail("vllm", 200),
    "supervisor_log_tail": tail("supervisor", 120),
    "ts": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
}
for target in (path, legacy):
    try:
        with open(target, "w") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
    except Exception:
        pass
print(json.dumps({"stage": stage, "reason": reason, "pod_id": pod}))
PY
}

# ---------------------------------------------------------------------------
# Reporting. A "ready" report is what binds the pod's tunnel in the app; a
# "boot_failure" report records the stage and the log tail. Neither stops the pod.
# Bodies are built by python3, so no JSON ever passes through shell quoting.
# ---------------------------------------------------------------------------
report() {
  local kind="$1" reason="${2:-}" stage="${3:-}" body code
  if [ -z "$APP" ] || [ -z "$SECRET" ]; then
    log "$kind report skipped - AETHER_APP_ORIGIN or WEBHOOK_SECRET is unset"
    return 0
  fi
  body=$(python3 - "$kind" "$POD" "$WS_ID" "$NODE_ID" "$reason" "$stage" "${AETHER_SERVED_MODEL:-}" "$GPU_NAME" "$VRAM_GB" <<'PY'
import datetime, json, os, sys
kind, pod, ws, node, reason, stage, served, gpu, vram = sys.argv[1:10]
if kind == "ready":
    print(json.dumps({
        "pod_id": pod, "workspace_id": ws, "node_id": node,
        "status": "ready", "port": 8000,
        "ports": {"8000": "relay", "8001": "whisper", "8002": "vllm", "8080": "tts"},
        "mount_ok": True, "gpu_ok": True,
        "whisper_ok": True, "vllm_ok": True, "tts_ok": True, "relay_ok": True,
        "gpu_name": gpu, "vram_gb": vram, "served_model": served,
        "tunnel_url": "https://%s-8000.proxy.runpod.net" % pod,
    }))
else:
    tail = []
    log_dir = os.environ.get("AETHER_LOG_DIR", "")
    for name in ("boot_diag", "supervisor", "runtime", "whisper", "vllm", "tts", "relay"):
        try:
            rows = open(os.path.join(log_dir, name + ".log"), errors="replace").read().splitlines()[-40:]
        except Exception:
            rows = []
        for row in rows:
            if not any(mark in row for mark in ("HF_TOKEN", "WEBHOOK_SECRET", "Authorization", "hf_", "sk-")):
                tail.append(name + ": " + row)
    print(json.dumps({
        "probe": "boot_failure", "stage": stage, "pod_id": pod,
        "workspace_id": ws, "node_id": node, "error": reason,
        "gpu_name": gpu, "vram_gb": vram, "served_model": served,
        "log_tail": "\n".join(tail)[-4000:],
        "diagnostic_mode": True,
        "ts": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
    }))
PY
)
  code=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' -X POST "$APP/api/runpod-webhook" \
    -H 'content-type: application/json' -H "x-webhook-secret: $SECRET" --data-binary "$body" || true)
  log "$kind report http=$code${reason:+ reason=$reason}"
}

# A gate that fails: record it, tell the app, and hold. Never exit.
gate_fail() {
  local stage="$1" reason="$2" detail="${3:-}"
  log "BOOT PROBLEM stage=$stage reason=$reason"
  write_fail "$stage" "$reason" "$detail" >> "$DIAG" 2>&1 || true
  report boot_failure "$reason" "$stage"
  hold "$stage: $reason"
}

# ---------------------------------------------------------------------------
# THE MOUNT GATE. /workspace is persistent storage: without it the pod would write
# a generation into a disk that disappears with it.
# ---------------------------------------------------------------------------
MOUNT_OK=1
if command -v mountpoint >/dev/null 2>&1; then
  mountpoint -q "$WS" || MOUNT_OK=0
else
  grep -qs " $WS " /proc/mounts || MOUNT_OK=0
fi
if [ "$MOUNT_OK" != "1" ]; then
  log "mount gate: $WS is NOT the persistent volume"
  ls -la "$WS" >> "$DIAG" 2>&1 || true
  gate_fail mount workspace_not_mounted "$WS is not a mount point"
fi
log "mount gate: $WS is the persistent volume"
log "logs: $LOG_DIR (no chmod)"

# ---------------------------------------------------------------------------
# THE VERSION GATE. The pod runs the generation it was pinned to, or it does not
# run at all: a half-shipped generation is how a stale script kept serving.
# ---------------------------------------------------------------------------
version_gate() {
  local served
  log "version gate: script=$SERVED_VERSION pinned=${PINNED_VERSION:-unset}"
  if [ -z "$PINNED_VERSION" ]; then
    log "version gate: AETHER_POD_VERSION is unset - running $SERVED_VERSION unpinned (set the pin on the template)"
    return 0
  fi
  if [ "$PINNED_VERSION" != "$SERVED_VERSION" ]; then
    # A RunPod restart retains its creation-time env. Accept a new generation ONLY
    # when the baked entrypoint verified the complete manifest on container disk.
    if [ "${AETHER_SELF_REFRESHED:-0}" = "1" ] && "${PY_BIN:-python3}" - "${AETHER_ROOT:-}" "$SERVED_VERSION" <<'PYVERIFY'
import hashlib, json, pathlib, sys
root, version = pathlib.Path(sys.argv[1]), sys.argv[2]
manifest = json.loads((root / ".boot-manifest.json").read_text())
files = manifest.get("files") or []
assert manifest.get("pod_version") == version and files
assert any(row["name"] == "start_all.sh" for row in files)
for row in files:
    if row["name"] == "VERSION":
        continue
    assert pathlib.Path(row["name"]).name == row["name"]
    assert hashlib.sha256((root / row["name"]).read_bytes()).hexdigest() == row["sha256"]
PYVERIFY
    then
      log "version gate: verified restart upgrade $PINNED_VERSION -> $SERVED_VERSION"
      PINNED_VERSION="$SERVED_VERSION"
      export AETHER_POD_VERSION="$SERVED_VERSION"
    else
      gate_fail version version_mismatch "pod pinned $PINNED_VERSION but no verified $SERVED_VERSION tree exists"
    fi
  fi
  if [ -n "$APP" ]; then
    served=$(curl -sS -m 10 "$APP/api/pod-code?name=VERSION" 2>/dev/null | tr -d '[:space:]' || true)
    if [ -n "$served" ] && [ "$served" != "$PINNED_VERSION" ]; then
      gate_fail version version_mismatch "pod pinned $PINNED_VERSION but the app serves $served"
    fi
    log "version gate: manifest=${served:-unreachable} pin=$PINNED_VERSION"
  fi
  log "version gate: ok"
}
version_gate

# ---------------------------------------------------------------------------
# Preflight - diagnostics only, never an exit.
# ---------------------------------------------------------------------------
preflight() {
  local missing="" free_gb raw

  for _var in AETHER_APP_ORIGIN WEBHOOK_SECRET AETHER_WORKSPACE_ID AETHER_NODE_ID; do
    case "$_var" in
      WEBHOOK_SECRET) [ -n "${WEBHOOK_SECRET:-${AETHER_WEBHOOK_SECRET:-}}" ] || missing="$missing $_var" ;;
      *) eval "_v=\${$_var:-}"; [ -n "$_v" ] || missing="$missing $_var" ;;
    esac
  done
  [ -z "$missing" ] || log "preflight: missing env:${missing} - reports will be limited"

  if printf 'ok' > "${WS}/aether/.write-test" 2>/dev/null; then
    rm -f "${WS}/aether/.write-test" 2>/dev/null || true
    log "preflight: $WS/aether writable"
  else
    log "preflight: $WS/aether is NOT writable"
  fi

  free_gb=$(df -Pm "${WS}" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024}')
  log "preflight: free disk ${free_gb:-unknown}GB on ${WS}"

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "preflight: nvidia-smi is not on PATH - no GPU visible"
    return 0
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    log "preflight: nvidia-smi failed - no usable CUDA GPU"
    return 0
  fi

  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | tr -d '\r')
  # MiB, never whole GB: a 24GB card reports 24564 MiB and rounding that to 23GB
  # rejected an RTX 4090. The driver can still be initialising, so the read retries.
  for _try in 1 2 3 4 5 6 7 8 9 10; do
    raw=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc '0-9')
    VRAM_MB=${raw:-0}
    [ "${VRAM_MB:-0}" -gt 0 ] && break
    sleep 2
  done
  VRAM_GB=$(( (VRAM_MB + 1023) / 1024 ))
  log "preflight: GPU=${GPU_NAME:-unknown} VRAM=${VRAM_MB}MiB (${VRAM_GB}GB) floor=${MIN_VRAM_MB}MiB"
  if [ "${VRAM_MB:-0}" -lt "$MIN_VRAM_MB" ]; then
    log "preflight: GPU is below the ${MIN_VRAM_MB}MiB floor - vLLM will refuse to start (see vllm.log)"
  fi
}
preflight

# ---------------------------------------------------------------------------
# THE RUNTIME: the image's own python. Nothing is installed into it here - a
# runtime torch install is what pulled a CUDA stack and broke every boot.
# ---------------------------------------------------------------------------
runtime_gate() {
  [ -n "$PY_BIN" ] || PY_BIN="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
  [ -n "$PY_BIN" ] || gate_fail runtime no_python_interpreter "no python3 on PATH"
  [ -x "$PY_BIN" ] || gate_fail runtime python_not_executable "$PY_BIN is not executable"
  log "runtime: $PY_BIN ($("$PY_BIN" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo unknown))"
  log "runtime: torch $("$PY_BIN" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unavailable)"
}
runtime_gate

# ---------------------------------------------------------------------------
# THE RUNTIME GATE. vLLM was installed at IMAGE BUILD time and only after the same
# imports below passed, so this asserts the running interpreter rather than
# installing anything. vllm._C is the compiled extension: importing it proves the
# wheel matches this image's torch and CUDA stack. No private torch symbol is
# asserted — requiring one that torch 2.4.x does not export is what reported
# vllm_import_failed on healthy boots.
# ---------------------------------------------------------------------------
verify_vllm_runtime() {
  local gate rc
  if [ -s "$PINS_FILE" ]; then
    log "runtime pins: $(tr -d '\n' < "$PINS_FILE" 2>/dev/null)"
  else
    log "runtime pins: $PINS_FILE is missing — this image was built without the gate"
  fi
  gate=$("$PY_BIN" - <<'PYGATE' 2>&1
import torch
import vllm
import vllm._C
print("torch", torch.__version__, "vllm", vllm.__version__)
PYGATE
)
  rc=$?
  printf '%s\n' "$gate" >> "${LOG_DIR}/runtime.log" 2>/dev/null || true
  if [ "$rc" -eq 0 ]; then
    log "runtime gate: $gate"
    log "runtime gate: vllm + vllm._C present — the image's vLLM is the one this pod runs"
    return 0
  fi
  printf '%s\n' "$gate" | tail -n 40 >> "$SUPERVISOR_LOG" 2>/dev/null || true
  gate_fail vllm vllm_import_failed "$(printf '%s' "$gate" | tail -n 20)"
}
verify_vllm_runtime

# ---------------------------------------------------------------------------
# Pre-vLLM diagnostics: the GPU, the runtime versions actually importable, the
# cache the weights land in and whether Hugging Face will serve this repo. The
# token itself is never printed - only whether one is set.
# ---------------------------------------------------------------------------
runtime_diagnostics() {
  local first code token_state
  log "diagnostics: ---- pre-vLLM ----"
  first=$(nvidia-smi 2>/dev/null | head -n1 | tr -d '\r' | cut -c1-240)
  log "diagnostics: nvidia-smi: ${first:-unavailable}"
  log "diagnostics: torch: $("$PY_BIN" -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo unavailable)"
  log "diagnostics: vllm: $("$PY_BIN" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unavailable)"
  log "diagnostics: runtime pins: $(tr -d '\n' < "$PINS_FILE" 2>/dev/null || echo missing)"
  log "diagnostics: python: $PY_BIN"
  log "diagnostics: profile: gpu_util=$GPU_UTIL max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS"
  # The relay cannot bind 8000 without this, and a missing wheel used to show only as
  # a restart loop in relay.log. Naming it here makes it a one-line diagnosis.
  log "diagnostics: aiohttp: $("$PY_BIN" -c 'import aiohttp; print(aiohttp.__version__)' 2>/dev/null || echo MISSING)"
  log "diagnostics: HF_HOME=${HF_HOME}"
  mkdir -p "${HF_HOME}" 2>/dev/null || true
  ls -la "${HF_HOME}" 2>/dev/null | head -n 10 | while IFS= read -r _row; do log "diagnostics: hf-cache: ${_row}"; done
  if printf 'aether-cache-write-test' > "${HF_HOME}/.write-test" 2>/dev/null; then
    log "diagnostics: hf-cache write test: ok"
    rm -f "${HF_HOME}/.write-test" 2>/dev/null || true
  else
    log "diagnostics: hf-cache write test: FAILED - ${HF_HOME} is not writable"
  fi
  token_state="not set"
  [ -n "${HF_TOKEN:-}" ] && token_state="set"
  case "$MODEL" in
    */*)
      if [ -n "${HF_TOKEN:-}" ]; then
        code=$(curl -s -o /dev/null -m 15 -w '%{http_code}' -H "Authorization: Bearer ${HF_TOKEN}" "https://huggingface.co/${MODEL}/resolve/main/config.json" 2>/dev/null || true)
      else
        code=$(curl -s -o /dev/null -m 15 -w '%{http_code}' "https://huggingface.co/${MODEL}/resolve/main/config.json" 2>/dev/null || true)
      fi
      log "diagnostics: hf-repo: ${MODEL} status=${code:-000} token=${token_state}"
      ;;
    *) log "diagnostics: hf-repo: ${MODEL} is not a Hugging Face repo id - repo check skipped, token=${token_state}" ;;
  esac
  log "diagnostics: ---- end ----"
}
runtime_diagnostics

# Adapter routing. One source of truth: the same current_adapter.json the vLLM
# launcher reads. A validated tenant adapter becomes the served model name, so the
# relay's requests land on that business's LoRA; with nothing approved, the shared
# base model keeps serving exactly as before.
# ---------------------------------------------------------------------------
# Where the child scripts live. post_start.sh lands the verified generation in its
# run directory (AETHER_SCRIPT_CACHE, /tmp/aether/run by default) and exports it as
# AETHER_ROOT. When this supervisor is started any other way - a stale copy left on
# the volume, a manual background run - that variable is absent, and defaulting to
# the workspace cache pointed every child at a directory holding only old artifacts: python
# answered "[Errno 2] No such file or directory: /workspace/aether/voice_relay.py" on
# every restart. The directory is therefore resolved by finding the scripts, never
# assumed from a path baked in at build time.
# ---------------------------------------------------------------------------
SCRIPT_SENTINEL="voice_relay.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
SCRIPT_SEARCH="${AETHER_ROOT:-} ${SCRIPT_DIR:-} ${AETHER_SCRIPT_CACHE:-/tmp/aether/run} /opt/aether ${WS}/aether"
ROOT=""
for _candidate in ${AETHER_ROOT:-} "${SCRIPT_DIR:-}" "${AETHER_SCRIPT_CACHE:-/tmp/aether/run}" "/opt/aether" "${WS}/aether"; do
  [ -n "$_candidate" ] || continue
  if [ -f "${_candidate}/${SCRIPT_SENTINEL}" ]; then ROOT="$_candidate"; break; fi
done
if [ -n "$ROOT" ]; then
  log "script directory: $ROOT (searched:${SCRIPT_SEARCH})"
else
  # Nothing found: keep the old default so the per-script resolution below can still
  # try, and let it record a named failure instead of an Errno 2 in a child log.
  ROOT="${AETHER_ROOT:-${WS}/aether}"
  log "script directory: NONE of${SCRIPT_SEARCH} holds ${SCRIPT_SENTINEL} - falling back to $ROOT"
fi
TENANT_ROOT="${AETHER_TENANT_ROOT:-/workspace/tenants}"
TENANT_ID="${AETHER_TENANT_ID:-}"
ADAPTER_ALIAS=""
if [ -n "$TENANT_ID" ] && [ -f "${TENANT_ROOT}/${TENANT_ID}/adapters/llm_lora/current_adapter.json" ]; then
  ADAPTER_ALIAS=$("$PY_BIN" - "${TENANT_ROOT}/${TENANT_ID}/adapters/llm_lora/current_adapter.json" "$TENANT_ID" <<'PY'
import json, os, re, sys
pointer, tenant = sys.argv[1:3]
try:
    data = json.load(open(pointer))
except Exception:
    raise SystemExit(0)
path = str(data.get("adapter_path") or "")
if path and os.path.isdir(path):
    print(re.sub(r"[^a-z0-9_]", "_", tenant.lower())[:40])
PY
)
fi
if [ -n "${ADAPTER_ALIAS:-}" ]; then
  export AETHER_SERVED_MODEL="$ADAPTER_ALIAS"
  export AETHER_LORA_ALIAS="$ADAPTER_ALIAS"
  log "adapter routing: serving tenant ${TENANT_ID} adapter as model=${ADAPTER_ALIAS}"
else
  log "adapter routing: no approved adapter - serving base model ${MODEL}"
fi

# ---------------------------------------------------------------------------
# Independent launchers. Each service is a nohup background process with its own
# log and its own restart loop; no health gate here can stop its siblings or the
# container. The loop is bounded in rate (10s backoff), never in attempts.
# ---------------------------------------------------------------------------
run_forever() {
  local name="$1" script="$2" max_crashes="${3:-5}" path="" _dir
  # The cap is a number or it is the default. A non-numeric value made the comparison
  # below [ "" -gt 0 ], which is false forever, so the loop could never stop and a
  # dead engine was retried until the boot window expired.
  case "$max_crashes" in ''|*[!0-9]*) max_crashes=5 ;; esac
  # Resolve THIS script against the verified directory and then the fallbacks. One
  # directory is never assumed to hold all of them, and a script that is genuinely
  # absent is named in the failure record instead of reaching python as a bad path.
  for _dir in "$ROOT" "${AETHER_SCRIPT_CACHE:-/tmp/aether/run}" "/opt/aether" "${WS}/aether"; do
    [ -n "$_dir" ] || continue
    if [ -f "${_dir}/${script}" ]; then path="${_dir}/${script}"; break; fi
  done
  if [ -z "$path" ]; then
    log "FATAL: ${script} is not in $ROOT, ${AETHER_SCRIPT_CACHE:-/tmp/aether/run}, /opt/aether or ${WS}/aether"
    write_fail "script_missing" "${script}_missing" "${script} was not found in any script directory"
    report boot_failure "script_missing" "${name}"
    return 1
  fi
  [ "$path" = "${ROOT}/${script}" ] || log "${script} resolved from $path (not $ROOT)"
  log "launching $name: $path (restart cap ${max_crashes} crashes/min)"
  nohup bash -c '
    crashes=0
    while :; do
      began=$(date +%s)
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting '"$path"'"
      "'"$PY_BIN"'" "'"$path"'"
      rc=$?
      ran=$(( $(date +%s) - began ))
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] '"$script"' exited rc=$rc after ${ran}s"
      # A service that dies inside a minute is crashing, not restarting. Without a
      # cap the loop hides that behind endless retries and the boot gate just waits
      # out its window - which is how an engine that OOMs on every start looked like
      # a 20-minute hang. After the cap the loop stops so the gate can name it.
      if [ "$ran" -lt 60 ]; then crashes=$((crashes + 1)); else crashes=0; fi
      if [ "'"$max_crashes"'" -gt 0 ] && [ "$crashes" -ge "'"$max_crashes"'" ]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] '"$script"' crashed ${crashes}x inside 60s - restart loop stopped"
        # break, never exit: this is a child loop, and the supervisor itself must stay
        # alive (an exit here would restart the whole container).
        break
      fi
      sleep 10
    done
  ' >> "${LOG_DIR}/${name}.log" 2>&1 &
  echo "$!" > "${LOG_DIR}/${name}.pid"
  log "launched $name pid=$! log=${LOG_DIR}/${name}.log"
}

# Poll one service's /health. Returns non-zero on timeout; the caller decides
# whether that is a hold (vLLM) or just a note (TTS).
wait_health() {
  local name="$1" url="$2" timeout="$3" waited=0
  while [ "$waited" -lt "$timeout" ]; do
    if curl -fsS -m 5 "$url" >/dev/null 2>&1; then
      log "health: $name ready after ${waited}s"
      return 0
    fi
    sleep 10
    waited=$((waited + 10))
  done
  log "health: $name did not answer within ${timeout}s"
  return 1
}

# The engine gate, with progress. Every 30s it records how long the engine has been
# starting and the last line its own log produced, so a stall is visible while it is
# happening instead of being inferred from a silent port. Past the boot window it
# writes the failure record ONCE and keeps waiting: the engine may still come up.
wait_health_progress() {
  local name="$1" url="$2" timeout="$3" waited=0 last
  while [ "$waited" -lt "$timeout" ]; do
    if curl -fsS -m 5 "$url" >/dev/null 2>&1; then
      log "health: $name ready after ${waited}s"
      return 0
    fi
    # A bounded restart loop that gave up leaves no process behind: fail at once
    # instead of waiting out a window for a service that is no longer running.
    if [ -f "${LOG_DIR}/${name}.pid" ] && ! kill -0 "$(cat "${LOG_DIR}/${name}.pid" 2>/dev/null)" 2>/dev/null; then
      log "health: $name has no running process (its restart loop stopped) - failing now"
      return 1
    fi
    if [ $((waited % 30)) -eq 0 ]; then
      last=$(tail -n 1 "${LOG_DIR}/${name}.log" 2>/dev/null | tr -d '\r' | cut -c1-180)
      log "health: $name still starting after ${waited}s (last log: ${last:-none})"
      if [ "$waited" -ge "$BOOT_WINDOW" ] && [ ! -f "$STALL_FILE" ]; then
        : > "$STALL_FILE" 2>/dev/null || true
        stage "${name}_stalled" "no /health after ${waited}s"
        write_fail "$name" "${name}_stalled" "no /health on $url after ${waited}s"
        report boot_failure "${name}_stalled" "$name"
      fi
    fi
    sleep 10
    waited=$((waited + 10))
  done
  log "health: $name did not answer within ${timeout}s"
  return 1
}

# 1) The relay FIRST. It is the pod's only public port and it needs no engine to
#    bind, so it starts before anything loads. Binding 8000 only after the engines
#    were healthy is what made a slow boot look like a dead pod: until something
#    listens on 8000 the RunPod proxy answers 502, so there was no way to see how
#    far the boot had got. /health now reports the stage from the first second.
stage relay_start "binding 8000 before the engines load"
run_forever relay voice_relay.py
wait_health relay "http://127.0.0.1:${RELAY_PORT}/health" "$RELAY_TIMEOUT" \
  || log "relay did not answer within ${RELAY_TIMEOUT}s - the watchdog keeps reporting"

# 2) The engines are launched TOGETHER. They are independent processes, so gating
#    one behind another only added their load times up: Whisper's model used to hold
#    vLLM's start for up to ${WHISPER_TIMEOUT}s, and only then did the weights
#    download begin - which is how a cold boot reached 20+ minutes. Concurrent start
#    is the single biggest saving here.
stage whisper_start "faster-whisper loading"
run_forever whisper whisper_server.py
stage vllm_start "engine launching (weights download on a cold cache is the slow part)"
# Five crashes inside a minute is not a flaky start: the engine cannot run here, and
# waiting out a 30-minute window to say so helps nobody.
run_forever vllm run_vllm_safe.py 5
stage tts_start "speech worker loading"
run_forever tts tts_server.py

# 3) vLLM - the Brain. This gate decides: an engine that never binds 8002 is a named
#    failure with the vLLM tail attached, never a silent 10s restart loop.
stage vllm_wait "waiting for /health on ${VLLM_PORT}"
if ! wait_health_progress vllm "http://127.0.0.1:${VLLM_PORT}/health" "$VLLM_TIMEOUT"; then
  log "vllm.log tail (200 lines):"
  tail -n 200 "${LOG_DIR}/vllm.log" >> "$SUPERVISOR_LOG" 2>/dev/null || true
  gate_fail vllm vllm_unhealthy "no /health on ${VLLM_PORT} within ${VLLM_TIMEOUT}s"
fi
stage vllm_ready "engine answered /health on ${VLLM_PORT}"

# 4) The Ear and the Voice are noted, not gated: the relay's own deps_ok reports
#    them, and a process that is alive still serves once its model finishes loading.
wait_health whisper "http://127.0.0.1:${WHISPER_PORT}/health" "$WHISPER_TIMEOUT" \
  || log "whisper is not answering yet - continuing, the relay reports it"
wait_health tts "http://127.0.0.1:${TTS_PORT}/health" "$TTS_TIMEOUT" \
  || log "tts: no /health yet - keeping the process alive"

stage services_ready "whisper=${WHISPER_PORT} vllm=${VLLM_PORT} tts=${TTS_PORT} relay=${RELAY_PORT}"
log "boot complete: whisper=${WHISPER_PORT} vllm=${VLLM_PORT} tts=${TTS_PORT} relay=${RELAY_PORT}"

# ---------------------------------------------------------------------------
# Watchdog. Every 30s it reads the relay's own health; once every service gate is
# green it reports ready (that is what binds the tunnel). It never restarts or
# stops a service. Every 30 minutes it heartbeats the app, which answers whether
# this pod should stop (idle workspace); and after 120 consecutive minutes of an
# unhealthy relay it stops the pod itself, which is the only local way to end the
# provider billing.
# ---------------------------------------------------------------------------
heartbeat() {
  local healthy="$1" body resp
  [ -n "$APP" ] && [ -n "$SECRET" ] || return 0
  body=$(python3 - "$POD" "$WS_ID" "$NODE_ID" "$healthy" <<'PY'
import json, sys
pod, ws, node, healthy = sys.argv[1:5]
print(json.dumps({"probe": "heartbeat", "pod_id": pod, "workspace_id": ws,
                  "node_id": node, "healthy": healthy == "1"}))
PY
)
  resp=$(curl -sS -m 10 -X POST "$APP/api/runpod-webhook" \
    -H 'content-type: application/json' -H "x-webhook-secret: $SECRET" --data-binary "$body" 2>/dev/null || true)
  log "heartbeat healthy=$healthy response=$(printf '%s' "$resp" | tr -d '\n' | cut -c1-200)"
  if printf '%s' "$resp" | grep -q '"stop": *true'; then
    log "the app asked this pod to stop (idle or unhealthy) - stopping it now"
    if command -v runpodctl >/dev/null 2>&1; then
      runpodctl pod stop "$POD" >> "$DIAG" 2>&1 || true
    else
      log "runpodctl is unavailable - the app must stop this pod"
    fi
    hold "stopped on the app's heartbeat instruction"
  fi
}

watchdog() {
  local started ready_reported=0 failure_reported=0 unhealthy_since=0 now health stage reason last_beat
  started=$(date +%s)
  last_beat=$started
  while :; do
    sleep 30
    now=$(date +%s)
    health=$(curl -sS -m 5 "http://127.0.0.1:${RELAY_PORT}/health" 2>/dev/null || true)
    if printf '%s' "$health" | grep -q '"bind"'; then
      unhealthy_since=0
      if [ "$ready_reported" = "0" ] && printf '%s' "$health" | grep -q '"deps_ok": *true'; then
        report ready
        ready_reported=1
        log "all service gates green - reported ready"
      fi
    else
      [ "$unhealthy_since" = "0" ] && unhealthy_since=$now
      if [ "$ready_reported" = "0" ] && [ "$failure_reported" = "0" ] && [ $((now - started)) -ge "$BOOT_WINDOW" ]; then
        stage=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("stage",""))' "$FAIL_FILE" 2>/dev/null || true)
        reason=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("reason",""))' "$FAIL_FILE" 2>/dev/null || true)
        log "engine not ready after ${BOOT_WINDOW}s (stage=${stage:-none}) - pod stays up for inspection"
        report boot_failure "${reason:-engine_not_ready}" "${stage:-unknown}"
        failure_reported=1
      fi
    fi
    if [ "$unhealthy_since" != "0" ] && [ $((now - unhealthy_since)) -ge 7200 ]; then
      log "watchdog: relay unhealthy for 120 consecutive minutes - stopping the pod"
      unhealthy_since=$now
      heartbeat 0
      if command -v runpodctl >/dev/null 2>&1; then
        runpodctl pod stop "$POD" >> "$DIAG" 2>&1 || true
      else
        log "runpodctl is unavailable - the app must stop this pod"
      fi
    fi
    if [ $((now - last_beat)) -ge 1800 ]; then
      last_beat=$now
      if [ "$unhealthy_since" = "0" ]; then heartbeat 1; else heartbeat 0; fi
    fi
  done
}

watchdog >> "${LOG_DIR}/watchdog.log" 2>&1 &

log "container held open with sleep infinity"
sleep infinity
