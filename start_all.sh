#!/usr/bin/env bash
# =============================================================================
# AetherVoice — baked container entrypoint  (/opt/aether/start_all.sh)
# Base image: ghcr.io/njfg-os/voice-agent-mcp:0.2.0
#
# What this file is
#   The image's ONE entrypoint. It never installs a runtime, never blocks on
#   venv_missing, and never writes startup scripts into /workspace. It:
#     1. picks a Python that already carries torch + vLLM (baked venv, /opt/venv,
#        conda, /usr/local, /usr/bin, PATH) and records why each one lost
#     2. asks the app for the served POD_VERSION and reuses its verified cache
#     3. otherwise fetches the app-owned scripts, validates them (bash -n /
#        compile) and writes a sha256 manifest — the same verification the app's
#        own bootstrap uses
#     4. hands off with `exec` to the verified served supervisor, so a future
#        runtime change ships from the app with no image rebuild
#
# Storage contract
#   container disk   /opt/aether (baked scripts)   /tmp/aether/run (verified cache)
#                    /tmp (probe + boot-failure records)
#   /workspace       hf-cache, pip-cache, tts_voices, aether/logs — model caches
#                    and voices only; NO startup script is ever required there
#
# Webhook secret: WEBHOOK_SECRET or AETHER_WEBHOOK_SECRET, whichever is set.
# =============================================================================
set -uo pipefail

APP="${AETHER_APP_ORIGIN:-https://deft-aether-voice-core.base44.app}"
APP="${APP%/}"
BAKED_ROOT="/opt/aether"
CACHE_DIR="${AETHER_SCRIPT_CACHE:-/tmp/aether/run}"
OFFLINE_ROOT="${AETHER_BAKED_SUPERVISOR_DIR:-${BAKED_ROOT}/pod}"
WORKSPACE_DIR="${AETHER_WORKSPACE_DIR:-/workspace}"
SECRET="${WEBHOOK_SECRET:-${AETHER_WEBHOOK_SECRET:-}}"
BOOTSTRAP_ONLY="${AETHER_BOOTSTRAP_ONLY:-0}"
PROBE_FILE="/tmp/aether-runtime-probe.txt"
FAIL_FILE="/tmp/aether-boot-fail.json"
NAMES="start_all.sh run_vllm_safe.py whisper_server.py voice_relay.py tts_server.py"

export AETHER_APP_ORIGIN="$APP"
export AETHER_WORKSPACE_DIR="$WORKSPACE_DIR"
export AETHER_BOOTSTRAP_ONLY="$BOOTSTRAP_ONLY"
export HF_HOME="${HF_HOME:-${WORKSPACE_DIR}/hf-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WORKSPACE_DIR}/pip-cache}"
export AETHER_VOICE_DIR="${AETHER_VOICE_DIR:-${WORKSPACE_DIR}/tts_voices}"
export PYTHONUNBUFFERED=1
# Both aliases always carry the same value, whichever one the template set.
export WEBHOOK_SECRET="$SECRET"
export AETHER_WEBHOOK_SECRET="$SECRET"

log() { printf 'AetherVoice[boot]: %s\n' "$*" >&2; }

# -----------------------------------------------------------------------------
# Boot-failure record: one JSON, stamped with an aware UTC datetime (never
# utcnow()), written to the container disk and posted to the app when reachable.
# -----------------------------------------------------------------------------
fail_report() {
  local reason="$1" detail="${2:-}" body="" code=""
  if command -v "${PY:-python3}" >/dev/null 2>&1; then
    body=$("${PY:-python3}" - "$reason" "$detail" "${RUNPOD_POD_ID:-}" \
      "${AETHER_WORKSPACE_ID:-}" "${AETHER_NODE_ID:-}" "${WANT:-}" <<'PYJSON'
import datetime, json, sys
reason, detail, pod, ws, node, want = sys.argv[1:7]
print(json.dumps({
    "probe": "boot_failure",
    "stage": "entrypoint",
    "reason": reason,
    "detail": detail[:1500],
    "pod_id": pod,
    "workspace_id": ws,
    "node_id": node,
    "served_version": want,
    "ts": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
}))
PYJSON
)
  fi
  [ -n "$body" ] && printf '%s\n' "$body" > "$FAIL_FILE" 2>/dev/null
  log "BOOT FAILED reason=$reason ${detail:+detail=$detail}"
  if [ -n "$body" ] && [ -n "$SECRET" ]; then
    code=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' -X POST "$APP/api/runpod-webhook" \
      -H 'content-type: application/json' -H "x-webhook-secret: $SECRET" \
      --data-binary "$body" 2>/dev/null || true)
    log "failure report http=${code:-none}"
  fi
}

# -----------------------------------------------------------------------------
# 1. Python runtime — accepted, never demanded.
#    Every interpreter the image could ship is probed for torch AND vLLM (as an
#    importable module or as its CLI beside the interpreter). The first one that
#    passes is used. If none passes we still continue on the best available
#    python3: the services report their own health, and a missing wheel is not a
#    reason to refuse to boot a paid GPU pod.
# -----------------------------------------------------------------------------
runtime_candidates() {
  {
    printf '%s\n' "${AETHER_PYTHON:-}"
    printf '%s\n' "${BAKED_ROOT}/venv/bin/python" "${BAKED_ROOT}/venv/bin/python3"
    printf '%s\n' /opt/venv/bin/python /opt/venv/bin/python3
    printf '%s\n' /opt/conda/bin/python /opt/conda/bin/python3 /opt/miniconda3/bin/python
    printf '%s\n' /usr/local/bin/python3 /usr/local/bin/python
    printf '%s\n' /usr/bin/python3 /usr/bin/python
    command -v python3 2>/dev/null || true
    command -v python 2>/dev/null || true
  } | awk 'NF && !seen[$0]++'
}

select_python() {
  local py bin
  : > "$PROBE_FILE" 2>/dev/null || true
  while IFS= read -r py; do
    [ -n "$py" ] || continue
    if [ ! -x "$py" ]; then
      printf '%s=absent ' "$py" >> "$PROBE_FILE" 2>/dev/null
      continue
    fi
    if ! "$py" -c 'import torch' >/dev/null 2>&1; then
      printf '%s=no_torch ' "$py" >> "$PROBE_FILE" 2>/dev/null
      continue
    fi
    bin=$(dirname "$py")
    if "$py" -c 'import vllm' >/dev/null 2>&1 || [ -x "${bin}/vllm" ] || command -v vllm >/dev/null 2>&1; then
      printf '%s=ok ' "$py" >> "$PROBE_FILE" 2>/dev/null
      printf '%s' "$py"
      return 0
    fi
    printf '%s=no_vllm ' "$py" >> "$PROBE_FILE" 2>/dev/null
  done <<EOF
$(runtime_candidates)
EOF
  return 1
}

PY="$(select_python || true)"
if [ -n "$PY" ]; then
  log "runtime: $PY"
else
  PY="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
  if [ -z "$PY" ]; then
    fail_report no_python_interpreter "no python3/python on PATH and no baked interpreter found"
    exit 1
  fi
  log "runtime: $PY (no interpreter reported torch+vLLM yet — continuing; services report their own health)"
fi
log "runtime probe: $(tr -s ' ' <<<"$(cat "$PROBE_FILE" 2>/dev/null || echo none)")"
export AETHER_PYTHON="$PY"
export PATH="$(dirname "$PY"):$PATH"

# Model caches and voices are the only things this pod needs to persist.
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$AETHER_VOICE_DIR" 2>/dev/null || \
  log "warning: could not create every workspace cache directory"

# -----------------------------------------------------------------------------
# 2. Served-script refresh + version / sha256 verification.
#    Cache lives on the container disk: /workspace never holds a startup script.
# -----------------------------------------------------------------------------
served_version() {
  curl -fsS -m 15 "$APP/api/pod-code?name=VERSION" 2>/dev/null | tr -d '[:space:]'
}

cache_valid() {
  [ -n "${WANT:-}" ] || return 1
  [ -s "$CACHE_DIR/start_all.sh" ] || return 1
  [ -s "$CACHE_DIR/.boot-manifest.json" ] || return 1
  "$PY" - "$CACHE_DIR" "$WANT" <<'PYVERIFY' >/dev/null 2>&1
import hashlib, json, os, sys
root, want = sys.argv[1], sys.argv[2]
names = ("start_all.sh", "run_vllm_safe.py", "whisper_server.py", "voice_relay.py", "tts_server.py")
try:
    data = json.load(open(os.path.join(root, ".boot-manifest.json")))
except Exception:
    sys.exit(1)
if data.get("version") != want:
    sys.exit(1)
for name in names:
    path = os.path.join(root, name)
    if not os.path.isfile(path):
        sys.exit(1)
    if data.get("sha256", {}).get(name) != hashlib.sha256(open(path, "rb").read()).hexdigest():
        sys.exit(1)
sys.exit(0)
PYVERIFY
}

write_manifest() {
  "$PY" - "$CACHE_DIR" "$WANT" <<'PYMANIFEST' >/dev/null 2>&1
import hashlib, json, os, sys
root, want = sys.argv[1], sys.argv[2]
names = ("start_all.sh", "run_vllm_safe.py", "whisper_server.py", "voice_relay.py", "tts_server.py")
sha = {}
for name in names:
    sha[name] = hashlib.sha256(open(os.path.join(root, name), "rb").read()).hexdigest()
tmp = os.path.join(root, ".boot-manifest.json.tmp")
with open(tmp, "w") as handle:
    json.dump({"version": want, "sha256": sha}, handle, indent=2, sort_keys=True)
os.replace(tmp, os.path.join(root, ".boot-manifest.json"))
with open(os.path.join(root, "VERSION"), "w") as handle:
    handle.write(want + "\n")
PYMANIFEST
}

fetch_all() {
  local name ok=1
  mkdir -p "$CACHE_DIR" 2>/dev/null || { log "cannot create $CACHE_DIR"; return 1; }
  for name in $NAMES; do
    if ! curl -fsS -m 60 "$APP/api/pod-code?name=$name" -o "$CACHE_DIR/$name.tmp" 2>/dev/null; then
      log "fetch failed: $name"
      ok=0
      rm -f "$CACHE_DIR/$name.tmp" 2>/dev/null
      continue
    fi
    case "$name" in
      *.sh)
        if ! bash -n "$CACHE_DIR/$name.tmp" 2>/dev/null; then
          log "$name failed bash -n — not swapped in"
          ok=0
          rm -f "$CACHE_DIR/$name.tmp" 2>/dev/null
          continue
        fi ;;
      *.py)
        if ! "$PY" -c 'import sys; compile(open(sys.argv[1]).read(), sys.argv[1], "exec")' "$CACHE_DIR/$name.tmp" 2>/dev/null; then
          log "$name failed to compile — not swapped in"
          ok=0
          rm -f "$CACHE_DIR/$name.tmp" 2>/dev/null
          continue
        fi ;;
    esac
    mv -f "$CACHE_DIR/$name.tmp" "$CACHE_DIR/$name"
    log "verified and landed: $name"
  done
  # The manifest is written only after every file landed and validated, so a
  # half-write can never stamp itself as a good cache.
  [ "$ok" = "1" ] || return 1
  write_manifest || return 1
}

WANT=""
attempt=0
while [ "$attempt" -lt 12 ] && [ -z "$WANT" ]; do
  attempt=$(( attempt + 1 ))
  WANT="$(served_version || true)"
  [ -n "$WANT" ] || sleep 5
done

if [ -n "$WANT" ]; then
  log "served version: $WANT"
else
  log "app unreachable at $APP — falling back to cached or baked scripts"
fi

if [ -n "$WANT" ] && cache_valid; then
  RUN_ROOT="$CACHE_DIR"
  log "cache hit $WANT — version and sha256 agree"
elif [ -n "$WANT" ] && fetch_all; then
  RUN_ROOT="$CACHE_DIR"
  log "cache miss — fetched $WANT and verified every file"
elif [ -s "$CACHE_DIR/start_all.sh" ]; then
  RUN_ROOT="$CACHE_DIR"
  log "warning: using the cached scripts without a verified manifest"
elif [ -s "$OFFLINE_ROOT/start_all.sh" ]; then
  RUN_ROOT="$OFFLINE_ROOT"
  log "warning: using the baked offline scripts in $OFFLINE_ROOT"
else
  fail_report no_supervisor "no verified served scripts and no baked supervisor at $OFFLINE_ROOT"
  exit 1
fi

# -----------------------------------------------------------------------------
# 3. Hand off. The served supervisor owns preflight, service gates and the ready
#    report; it must run from the SAME directory as its service scripts, so
#    AETHER_ROOT points at the verified tree and its own self-refresh is skipped
#    (this entrypoint already did that work).
# -----------------------------------------------------------------------------
export AETHER_ROOT="$RUN_ROOT"
export AETHER_SELF_REFRESHED=1
log "handing off to $RUN_ROOT/start_all.sh (bootstrap_only=$BOOTSTRAP_ONLY)"

if [ "${DEBUG_BOOT:-false}" = "true" ]; then
  log "DEBUG_BOOT on — supervisor runs as a child so this container stays inspectable"
  bash "$RUN_ROOT/start_all.sh"
  rc=$?
  log "supervisor exited rc=$rc — holding the container open (DEBUG_BOOT)"
  sleep infinity
fi

exec bash "$RUN_ROOT/start_all.sh"
