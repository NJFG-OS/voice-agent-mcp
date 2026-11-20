#!/usr/bin/env bash
# =============================================================================
# AetherVoice — baked container entrypoint  (/opt/aether/start_all.sh)
# Baked into: ghcr.io/njfg-os/voice-agent-mcp:0.2.2
#
# What this file is
#   The image's ONE entrypoint — the boot-contract bootstrapper. It never
#   installs a runtime, never blocks on venv_missing, and never writes a startup
#   script into /workspace. It:
#     1. picks a Python that already carries torch (baked venv, /opt/venv, conda,
#        /usr/local, /usr/bin, PATH) and records why each candidate lost
#     2. asks the app for the served generation and its sha256 MANIFEST
#        (/api/pod-code?name=manifest) — the manifest is the authority, this
#        image never computes its own idea of what the tree should be
#     3. reuses the verified cache when the generation and every digest agree,
#        otherwise fetches the served files, syntax-checks each one and proves
#        its sha256 against the manifest before landing it
#     4. hands off with `exec` to the verified served supervisor, so a runtime
#        fix ships from the app with no image rebuild
#
# Storage contract
#   container disk   /opt/aether (this file)   /tmp/aether/run (verified cache)
#                    /tmp (probe + boot-failure records)
#   /workspace       hf-cache, pip-cache, tts_voices, aether/logs — model caches
#                    and voices only; NO startup script is ever required there
#
# The served supervisor owns preflight, the vLLM overlay, the staged service
# gates and the ready report. The generation pin (AETHER_POD_VERSION vs the
# served VERSION) is enforced there, on the bytes that are about to run.
#
# Webhook secret: WEBHOOK_SECRET or AETHER_WEBHOOK_SECRET, whichever is set.
# =============================================================================
set -uo pipefail

APP="${AETHER_APP_ORIGIN:-******}"
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
log "entrypoint /opt/aether/start_all.sh pid=$$"

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
#    Every interpreter the image could ship is probed for torch. The first one
#    that passes is used; vLLM itself arrives as the app-managed --no-deps
#    overlay on the container disk, so an interpreter without it is still a
#    usable runtime. If nothing passes we continue on the best available
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
    exit 1
  fi
  log "runtime: $PY (no interpreter reported torch yet — continuing; services report their own health)"
fi
log "runtime probe: $(tr -s ' ' <<<"$(cat "$PROBE_FILE" 2>/dev/null || echo none)")"
export AETHER_PYTHON="$PY"
export PATH="$(dirname "$PY"):$PATH"

# Model caches and voices are the only things this pod needs to persist.
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$AETHER_VOICE_DIR" 2>/dev/null || \
  log "warning: could not create every workspace cache directory"

# -----------------------------------------------------------------------------
# 2. Served generation + sha256 manifest. The app's manifest is the authority:
#    a file is only landed once its digest matches the one the app published.
#    The cache lives on the container disk — /workspace never holds a script.
# -----------------------------------------------------------------------------
served_version() {
  curl -fsS -m 15 "$APP/api/pod-code?name=VERSION" 2>/dev/null | tr -d '[:space:]'
}

# The manifest must name the generation we asked for and carry digests.
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
  exit 1
fi

# -----------------------------------------------------------------------------
# 3. Hand off. The served supervisor owns preflight, the vLLM overlay, the
#    staged service gates and the ready report; it must run from the SAME
#    directory as its service scripts, so AETHER_ROOT points at the verified
#    tree (this entrypoint already did the refresh work).
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
  hold="${AETHER_DEBUG_HOLD_SECONDS:-900}"
  log "supervisor exited rc=$rc — holding the container open ${hold}s (DEBUG_BOOT), then exiting"
  sleep "$hold"
  exit "$rc"
fi

exec bash "$RUN_ROOT/start_all.sh"