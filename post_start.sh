#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/aether/logs /workspace/vllm_cache /workspace/whisper_models /workspace/CosyVoice_models /workspace/tts_voices || true
bash /opt/aether/start_all.sh >> /workspace/aether/logs/supervisor.log 2>&1 &
