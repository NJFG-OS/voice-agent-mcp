#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/workspace/aether/logs"
mkdir -p "$LOG_DIR"
chmod 777 "$LOG_DIR" || true

log() { echo "[${1:-boot}] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${2:-$*}" | tee -a "$LOG_DIR/supervisor.log" ; }

MODEL="${AETHER_VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
VLLM_PORT="${AETHER_VLLM_PORT:-8002}"
RELAY_PORT="${AETHER_RELAY_PORT:-8000}"
WHISPER_PORT="${AETHER_WHISPER_PORT:-8001}"
TTS_PORT="${AETHER_TTS_PORT:-8003}"

GPU_MEM_UTIL="${AETHER_GPU_MEMORY_UTILIZATION:-0.65}"

# Conservative defaults for 24GB tier
MAX_MODEL_LEN_24="${AETHER_MAX_MODEL_LEN_24:-4096}"
MAX_SEQS_24="${AETHER_MAX_SEQS_24:-4}"

# Optional: allow bigger tier values (fallback GPUs)
MAX_MODEL_LEN_48="${AETHER_MAX_MODEL_LEN_48:-8192}"
MAX_SEQS_48="${AETHER_MAX_SEQS_48:-8}"

# --- GPU detect / VRAM floor ---
VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
log boot "GPU=${GPU_NAME} VRAM=${VRAM_MIB}MiB"

if [ -z "$VRAM_MIB" ]; then
  log boot "BOOT FAILED: no_gpu_detected"
  sleep infinity
fi

# 24GB floor ~ 23000 MiB
if [ "$VRAM_MIB" -lt 23000 ]; then
  log boot "BOOT FAILED: gpu_too_small need>=24GB got=${VRAM_MIB}MiB"
  sleep infinity
fi

if [ "$VRAM_MIB" -ge 47000 ]; then
  MAX_MODEL_LEN="$MAX_MODEL_LEN_48"
  MAX_NUM_SEQS="$MAX_SEQS_48"
else
  MAX_MODEL_LEN="$MAX_MODEL_LEN_24"
  MAX_NUM_SEQS="$MAX_SEQS_24"
fi

log boot "vLLM config: model=${MODEL} gpu_mem_util=${GPU_MEM_UTIL} max_len=${MAX_MODEL_LEN} max_seqs=${MAX_NUM_SEQS}"

# --- Preflight imports (fail early with useful info) ---
python3 - <<PY 2>&1 | tee -a "$LOG_DIR/supervisor.log"
import torch, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", getattr(vllm, "__version__", "unknown"))
PY

# --- Start Whisper (implement whisper_server.py to serve /health on 8001) ---
log boot "Starting Whisper on ${WHISPER_PORT}"
nohup python3 /opt/aether/pod/whisper_server.py --port "$WHISPER_PORT" > "$LOG_DIR/whisper.log" 2>&1 &

# --- Start vLLM ---
log boot "Starting vLLM on ${VLLM_PORT}"
nohup python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --model "$MODEL" \
  --port "$VLLM_PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enforce-eager \
  > "$LOG_DIR/vllm.log" 2>&1 &

# Wait vLLM health
for i in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    log boot "vLLM healthy"
    break
  fi
  sleep 2
  if [ "$i" -eq 240 ]; then
    log boot "BOOT FAILED: vllm_unhealthy"
    tail -n 200 "$LOG_DIR/vllm.log" | tee -a "$LOG_DIR/supervisor.log" || true
    sleep infinity
  fi
done

# --- Start TTS (CosyVoice) ---
log boot "Starting TTS on ${TTS_PORT}"
nohup python3 /opt/aether/pod/tts_server.py --port "$TTS_PORT" > "$LOG_DIR/tts.log" 2>&1 &

# --- Start relay last ---
log boot "Starting relay on ${RELAY_PORT}"
nohup python3 /opt/aether/pod/voice_relay.py --port "$RELAY_PORT" > "$LOG_DIR/relay.log" 2>&1 &

log boot "All services launched; holding container"
sleep infinity
