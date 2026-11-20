# syntax=docker/dockerfile:1
FROM runpod/pytorch:2.1.0-cu1320-torch2130-ubuntu2404

LABEL org.opencontainers.image.title="aether-voice-agent" \
      org.opencontainers.image.description="AetherVoice pod: baked runtime (vLLM + Whisper + CosyVoice) + deterministic boot" \
      org.opencontainers.image.source="https://github.com/NJFG-OS/voice-agent-mcp"

USER root

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf-cache \
    TRANSFORMERS_CACHE=/workspace/hf-cache \
    XDG_CACHE_HOME=/workspace/.cache \
    PIP_CACHE_DIR=/workspace/pip-cache \
    AETHER_VOICE_DIR=/workspace/tts_voices

# System deps for audio + CosyVoice + common ML wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs \
      ffmpeg sox libsox-dev \
      libsndfile1 \
      build-essential \
      ca-certificates curl bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/aether

# Install pinned Python deps (no runtime installs)
COPY requirements.txt /opt/aether/requirements.txt
RUN pip install --no-cache-dir -r /opt/aether/requirements.txt

# Bake CosyVoice source + its Python deps
RUN git clone --recursive https://github.com/QwenAudio/CosyVoice.git /opt/CosyVoice \
 && pip install --no-cache-dir -r /opt/CosyVoice/requirements.txt

# Bake your service code (no runtime fetch)
COPY pod/ /opt/aether/pod/
RUN chmod +x /opt/aether/pod/start_all.sh

# Build-time compatibility gate: fail build if torch/vllm mismatch
RUN python3 - <<'PY'\nimport torch\nprint('torch', torch.__version__, 'cuda', torch.version.cuda)\nimport vllm\nprint('vllm', getattr(vllm, '__version__', 'unknown'))\nPY

# Runpod pod-safe hook: base /start.sh calls /post_start.sh
COPY post_start.sh /post_start.sh
RUN chmod +x /post_start.sh

VOLUME ["/workspace"]
EXPOSE 8000 8001 8002 8003 8080
