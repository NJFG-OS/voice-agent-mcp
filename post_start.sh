#!/usr/bin/env bash
# =============================================================================
# AetherVoice — /post_start.sh  (baked into the pod image)
#
# WHY THE APP STARTS HERE AND NOT FROM AN ENTRYPOINT
#   RunPod's base images own the container entrypoint. /start.sh brings up nginx,
#   sshd, the web terminal and Jupyter, exports the environment, and only THEN runs
#   `bash /post_start.sh`, waiting on it before its final `sleep infinity`.
#   (runpod/containers, container-template/start.sh.)
#   So this image sets no ENTRYPOINT and never touches /start.sh: the pod stays
#   reachable over SSH and the web terminal, and the Aether stack starts from the
#   documented hook, after those services are already up.
#
# WHAT IT DOES
#   1. clears the legacy script generation an older boot left on /workspace
#   2. picks the image's python — the one carrying torch and the build-time vLLM
#   3. records the runtime pins the image was built with (runtime-pins.json)
#   4. asks the app for the served generation + sha256 manifest, fetches and
#      verifies it onto the CONTAINER DISK (/tmp/aether/run), never /workspace
#   5. hands off to the verified supervisor (start_all.sh), which runs the staged
#      Whisper -> vLLM -> TTS -> relay gates and reports readiness
#
# FAILURE CONTRACT
#   /start.sh runs with `set -e`, so returning non-zero here would kill the
#   container and turn one broken boot into a restart loop. Every failure path
#   writes boot_failed.json + the log tail, reports to the app, and HOLDS
#   (sleep infinity). This script never terminates a pod.
#
# /workspace: HF cache, pip cache, voices and logs only. No executable is ever
# required there, so a wiped volume can never break a boot.
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
MANIFEST_FILE="${CACHE_DIR}/.served-manifest.json"
PINS_FILE="${AETHER_RUNTIME_PINS:-${BAKED_ROOT}/runtime-pins.json}"

export AETHER_APP_ORIGIN="$APP"
export AETHER_WORKSPACE_DIR="$WORKSPACE_DIR"
export AETHER_BOOTSTRAP_ONLY="$BOOTSTRAP_ONLY"
export AETHER_RUNTIME_PINS="$PINS_FILE"
export HF_HOME="${HF_HOME:-${WORKSPACE_DIR}/hf-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WORKSPACE_DIR}/pip-cache}"
export AETHER_VOICE_DIR="${AETHER_VOICE_DIR:-${WORKSPACE_DIR}/tts_voices}"
export PYTHONUNBUFFERED=1
# Both aliases always carry the same value, whichever one the template set.
export WEBHOOK_SECRET="$SECRET"
export AETHER_WEBHOOK_SECRET="$SECRET"

log() { printf 'AetherVoice[post_start]: %s\n' "$*" >&2; }
log "post-start hook /post_start.sh pid=$$ (run by the image's /start.sh)"

# -----------------------------------------------------------------------------
# 0. Legacy volume generation. Older images wrote their scripts into
#    /workspace/aether; those files are frozen and nothing reads them any more, but
#    a stale copy is how a pod once kept running a retired supervisor. Clear them.
# -----------------------------------------------------------------------------
for _legacy in post_start.sh start_all.sh run_vllm_safe.py ensure_vllm_overlay.py \
               whisper_server.py voice_relay.py tts_server.py; do
  rm -f "${WORKSPACE_DIR}/aether/${_legacy}" 2>/dev/null || true
done
log "cleared the legacy volume script generation under ${WORKSPACE_DIR}/aether"

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
    "stage": "post_start",
    "reason": reason,
    "detail": detail[:1500],
    "pod_id": pod,
    "workspace_id": ws,
    "node_id": node,
    "served_version": want,
    "diagnostic_mode": True,
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

# The only way out of a failed boot: hold. /start.sh already brought the pod up, so
# an operator can still SSH in, read the logs and inspect the volume.
hold() {
  log "container held open for inspection: $1"
  sleep infinity
}

# -----------------------------------------------------------------------------
# 1. Python runtime — the image's own. It carries torch AND the vLLM the build gate
#    accepted; nothing is installed here, ever.
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
  local py
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
    printf '%s=ok ' "$py" >> "$PROBE_FILE" 2>/dev/null
    printf '%s' "$py"
    return 0
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
    hold "no_python_interpreter"
  fi
  log "runtime: $PY (no interpreter reported torch yet — the supervisor's gate decides)"
fi
log "runtime probe: $(tr -s ' ' <<<"$(cat "$PROBE_FILE" 2>/dev/null || echo none)")"
export AETHER_PYTHON="$PY"
export PATH="$(dirname "$PY"):$PATH"

# -----------------------------------------------------------------------------
# 1b. /workspace bootstrap.
#
# The volume outlives every pod that mounts it, so a previous generation can leave
# three things behind that break the next boot: half-written cache files (a killed
# download leaves *.incomplete that blocks the next fetch), directories owned by a
# uid this container does not run as, and workers still holding 8000/8001/8002/8080.
# Each is repaired or cleared here, before any service starts, so terminate ->
# reactivate always boots from a known state and never inherits a stale process.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Stale Aether workers. Ownership is decided by the process's OWN COMMAND LINE,
# never by a port number: killing whatever listens on 8000-8080 could take down a
# listener that is not ours (a sidecar, a debug server, another agent). A process is
# ours only when its command line names one of this image's roots AND one of its
# served scripts, and it is never this script.
# -----------------------------------------------------------------------------
owned_worker_pids() {
  local dir pid cmd
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
    # ...AND when it also names one of this image's own roots.
    case "$cmd" in
      *"$CACHE_DIR"*|*"$OFFLINE_ROOT"*|*"$BAKED_ROOT"*|*"$WORKSPACE_DIR/aether"*|*"/tmp/aether"*) printf '%s\n' "$pid" ;;
    esac
  done
}

clear_owned_workers() {
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
  log "stopped previous-container worker(s): $(printf '%s' "$pids" | tr '\n' ' ')"
}

# Whisper (the Ear) owns 8001. The image strips the base config's 8001 listener at
# build time, so this only acts on a pod whose nginx still holds the port: whisper
# would otherwise exit with "Address already in use" while the provider kept showing
# 8001 Ready, because nginx - not Whisper - was the process answering there.
port_8001_free() {
  "${PY:-python3}" -c 'import socket, sys
sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("0.0.0.0", 8001))
except OSError:
    sys.exit(1)
finally:
    sock.close()' 2>/dev/null
}

ensure_port_8001_free() {
  if port_8001_free; then
    log "port 8001: free for Whisper"
    return 0
  fi
  log "port 8001: already bound - stopping nginx so Whisper (the Ear) can own it"
  if command -v nginx >/dev/null 2>&1; then
    nginx -s stop 2>/dev/null || pkill -x nginx 2>/dev/null || true
    sleep 2
  fi
  if port_8001_free; then
    log "port 8001: released for Whisper"
  else
    log "port 8001: still bound by another process - whisper reports the failure in its own log"
  fi
}

bootstrap_workspace() {
  local dir port pid name
  for dir in "$HF_HOME" "$PIP_CACHE_DIR" "$AETHER_VOICE_DIR" \
             "${WORKSPACE_DIR}/logs" "${WORKSPACE_DIR}/tts_voices" \
             "${WORKSPACE_DIR}/tenants" "${WORKSPACE_DIR}/platform"; do
    mkdir -p "$dir" 2>/dev/null || log "warning: cannot create $dir"
    # A directory owned by another uid is why a later write failed silently.
    [ -w "$dir" ] || chmod -R u+rwX "$dir" 2>/dev/null || true
  done

  # Half-written downloads are never resumable across a pod, and a stale one blocks
  # the next fetch. Model weights that completed are left untouched.
  find "$HF_HOME" -maxdepth 5 -type f \( -name '*.incomplete' -o -name '*.tmp' \) -delete 2>/dev/null || true
  find "$HF_HOME" -maxdepth 5 -type d -name '*.lock' -prune -exec rm -rf {} + 2>/dev/null || true
  find "$PIP_CACHE_DIR" -maxdepth 3 -type f -name '*.tmp' -delete 2>/dev/null || true

  # Nothing from a previous container is ever reused: only OUR workers are stopped,
  # by command line. A port-wide kill would also take down a listener that merely
  # happens to sit on 8000-8080 and is not this app's.
  clear_owned_workers
  ensure_port_8001_free

  date -u +%FT%TZ > "${WORKSPACE_DIR}/.aether-last-boot" 2>/dev/null || true
  log "workspace bootstrap: caches repaired, stale workers cleared (8000/8001/8002/8080)"
}
bootstrap_workspace

# -----------------------------------------------------------------------------
# 2. Runtime pins the image was built with. Logged on every boot so a pod's own
#    diagnostics answer "which torch/vLLM is this?" without a shell.
# -----------------------------------------------------------------------------
if [ -s "$PINS_FILE" ]; then
  log "runtime pins: $(tr -d '\n' < "$PINS_FILE" 2>/dev/null)"
else
  log "runtime pins: $PINS_FILE is missing — this image was built without the gate"
fi

# -----------------------------------------------------------------------------
# 3. Served generation + sha256 manifest. The app's manifest is the authority: a
#    file is only landed once its digest matches the one the app published. The
#    cache lives on the container disk — /workspace never holds a script.
# -----------------------------------------------------------------------------
served_version() {
  curl -fsS -m 15 "$APP/api/pod-code?name=VERSION" 2>/dev/null | tr -d '[:space:]'
}

fetch_manifest() {
  mkdir -p "$CACHE_DIR" 2>/dev/null || return 1
  if ! curl -fsS -m 30 "$APP/api/pod-code?name=manifest" -o "$MANIFEST_FILE.tmp" 2>/dev/null; then
    rm -f "$MANIFEST_FILE.tmp" 2>/dev/null
    return 1
  fi
  if ! "$PY" - "$MANIFEST_FILE.tmp" "$WANT" <<'PYMANIFESTCHECK' >/dev/null 2>&1
import json, sys
path, want = sys.argv[1], sys.argv[2]
data = json.load(open(path))
files = data.get("files") or []
if not want or data.get("pod_version") != want or not files:
    sys.exit(1)
if any(not entry.get("name") or not entry.get("sha256") for entry in files):
    sys.exit(1)
sys.exit(0)
PYMANIFESTCHECK
  then
    rm -f "$MANIFEST_FILE.tmp" 2>/dev/null
    return 1
  fi
  mv -f "$MANIFEST_FILE.tmp" "$MANIFEST_FILE"
}

manifest_names() {
  "$PY" - "$1" <<'PYNAMES' 2>/dev/null
import json, sys
for entry in json.load(open(sys.argv[1])).get("files", []):
    if entry.get("name") and entry["name"] != "VERSION":
        print(entry["name"])
PYNAMES
}

manifest_digest() {  # $1 = manifest, $2 = file name
  "$PY" - "$1" "$2" <<'PYDIGEST' 2>/dev/null
import json, sys
for entry in json.load(open(sys.argv[1])).get("files", []):
    if entry.get("name") == sys.argv[2]:
        print(entry.get("sha256", ""))
        break
PYDIGEST
}

file_digest() {
  "$PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1" 2>/dev/null
}

# The whole tree agrees with the manifest it was landed against.
verify_tree() {  # $1 = root
  "$PY" - "$1" <<'PYVERIFY' >/dev/null 2>&1
import hashlib, json, os, sys
root = sys.argv[1]
try:
    data = json.load(open(os.path.join(root, ".boot-manifest.json")))
except Exception:
    sys.exit(1)
files = data.get("files") or []
if not files or not data.get("pod_version"):
    sys.exit(1)
for entry in files:
    if entry["name"] == "VERSION":
        continue
    path = os.path.join(root, entry["name"])
    if not os.path.isfile(path):
        sys.exit(1)
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != entry["sha256"]:
        sys.exit(1)
sys.exit(0)
PYVERIFY
}

cache_valid() {
  [ -n "${WANT:-}" ] || return 1
  [ -s "$CACHE_DIR/start_all.sh" ] || return 1
  [ -s "$CACHE_DIR/.boot-manifest.json" ] || return 1
  "$PY" - "$CACHE_DIR/.boot-manifest.json" "$WANT" <<'PYCACHEVER' >/dev/null 2>&1
import json, sys
if json.load(open(sys.argv[1])).get("pod_version") != sys.argv[2]:
    sys.exit(1)
sys.exit(0)
PYCACHEVER
  verify_tree "$CACHE_DIR"
}

fetch_all() {
  local name want_sha got_sha ok=1
  fetch_manifest || { log "served manifest unavailable for $WANT"; return 1; }
  for name in $(manifest_names "$MANIFEST_FILE"); do
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
    want_sha="$(manifest_digest "$MANIFEST_FILE" "$name")"
    got_sha="$(file_digest "$CACHE_DIR/$name.tmp")"
    if [ -z "$want_sha" ] || [ "$want_sha" != "$got_sha" ]; then
      log "$name does not match the served sha256 — not swapped in"
      ok=0
      rm -f "$CACHE_DIR/$name.tmp" 2>/dev/null
      continue
    fi
    mv -f "$CACHE_DIR/$name.tmp" "$CACHE_DIR/$name"
    log "verified and landed: $name"
  done
  # The manifest is stamped only after every file landed and matched, so a
  # half-write can never certify itself as a good cache.
  [ "$ok" = "1" ] || return 1
  cp -f "$MANIFEST_FILE" "$CACHE_DIR/.boot-manifest.json" 2>/dev/null || return 1
  printf '%s\n' "$WANT" > "$CACHE_DIR/VERSION" 2>/dev/null || true
  verify_tree "$CACHE_DIR" || { log "landed tree failed final verification"; return 1; }
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
  log "cache hit $WANT — generation and every sha256 agree"
elif [ -n "$WANT" ] && fetch_all; then
  RUN_ROOT="$CACHE_DIR"
  log "cache miss — fetched $WANT and matched every served sha256"
elif [ -s "$CACHE_DIR/start_all.sh" ]; then
  RUN_ROOT="$CACHE_DIR"
  log "warning: using the cached scripts without a verified manifest"
elif [ -s "$OFFLINE_ROOT/start_all.sh" ]; then
  RUN_ROOT="$OFFLINE_ROOT"
  log "warning: using the baked offline scripts in $OFFLINE_ROOT"
else
  fail_report no_supervisor "no verified served scripts and no baked supervisor at $OFFLINE_ROOT"
  hold "no_supervisor"
fi

# -----------------------------------------------------------------------------
# 4. Hand off. The served supervisor owns preflight, the runtime gate, the staged
#    service gates and the ready report; it must run from the SAME directory as its
#    service scripts, so AETHER_ROOT points at the verified tree.
# -----------------------------------------------------------------------------
export AETHER_ROOT="$RUN_ROOT"
export AETHER_SELF_REFRESHED=1
log "handing off to $RUN_ROOT/start_all.sh (bootstrap_only=$BOOTSTRAP_ONLY)"

case "${DEBUG_BOOT:-false}" in
  true|TRUE|1|yes) DEBUG_HOLD=1 ;;
  *) DEBUG_HOLD=0 ;;
esac
if [ "$DEBUG_HOLD" = "1" ]; then
  log "DEBUG_BOOT on — supervisor runs as a child so this container stays inspectable"
  bash "$RUN_ROOT/start_all.sh"
  rc=$?
  # Bounded, never infinite: even an opted-in debug hold must stop billing on its
  # own, so a failed boot can never sit on a paid GPU indefinitely.
  hold_seconds="${AETHER_DEBUG_HOLD_SECONDS:-900}"
  log "supervisor exited rc=$rc — holding the container open ${hold_seconds}s (DEBUG_BOOT)"
  sleep "$hold_seconds"
  exit 0
fi

exec bash "$RUN_ROOT/start_all.sh"