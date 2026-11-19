#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/opt/aether:/opt/CosyVoice:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/workspace/vllm_cache}"
export AETHER_VLLM_MODEL="${AETHER_VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export AETHER_VLLM_PORT="${AETHER_VLLM_PORT:-8002}"
export AETHER_RELAY_PORT="${AETHER_RELAY_PORT:-8000}"

LOG_DIR="/workspace/aether/logs"
mkdir -p "$LOG_DIR" "/workspace/vllm_cache" "/workspace/whisper_models" "/workspace/CosyVoice_models" "/workspace/tts_voices"

log() {
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*" | tee -a "$LOG_DIR/start_all.log"
}

log "Starting AetherVoice baked runtime..."

python3 /opt/aether/run_vllm_safe.py --model "$AETHER_VLLM_MODEL" --port "$AETHER_VLLM_PORT" >> "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${AETHER_VLLM_PORT}/health" >/dev/null 2>&1 || curl -sf "http://127.0.0.1:${AETHER_VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    log "vLLM ready on port ${AETHER_VLLM_PORT}"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    log "BOOT_FAILURE: vLLM exited early. Check $LOG_DIR/vllm.log"
    exit 1
  fi
  sleep 2
done

python3 /opt/aether/whisper_server.py >> "$LOG_DIR/whisper.log" 2>&1 &
python3 /opt/aether/tts_server.py >> "$LOG_DIR/tts.log" 2>&1 &
uvicorn voice_relay:app --app-dir /opt/aether --host 0.0.0.0 --port "$AETHER_RELAY_PORT" >> "$LOG_DIR/relay.log" 2>&1 &
RELAY_PID=$!

for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:${AETHER_RELAY_PORT}/health" >/dev/null 2>&1; then
    log "Voice relay ready on port ${AETHER_RELAY_PORT}"
    break
  fi
  sleep 1
done

wait "$VLLM_PID" "$RELAY_PID"
