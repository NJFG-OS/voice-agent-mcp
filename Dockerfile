# 1) Use an official Runpod PyTorch base that is Python 3.12
FROM runpod/pytorch:REPLACE_WITH_PY312_TAG

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf-cache \
    TRANSFORMERS_CACHE=/workspace/hf-cache \
    XDG_CACHE_HOME=/workspace/.cache \
    PIP_CACHE_DIR=/workspace/pip-cache

USER root

# 2) System deps for audio + CosyVoice + common native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs \
      ffmpeg sox libsox-dev \
      libsndfile1 \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3) Copy and install pinned Python deps (NO runtime installs)
WORKDIR /opt/aether
COPY requirements.txt /opt/aether/requirements.txt
RUN pip install --no-cache-dir -r /opt/aether/requirements.txt

# 4) Bake CosyVoice repo code (or bake your fork)
RUN git clone --recursive https://github.com/QwenAudio/CosyVoice.git /opt/CosyVoice \
 && pip install --no-cache-dir -r /opt/CosyVoice/requirements.txt

# 5) Bake your service code into the image
COPY pod/ /opt/aether/pod/
RUN chmod +x /opt/aether/pod/start_all.sh

# 6) Build-time compatibility gate (prevents shipping broken torch/vllm combos)
RUN python3 - <<'PY'\nimport torch\nprint('torch', torch.__version__, 'cuda', torch.version.cuda)\nimport vllm\nprint('vllm', getattr(vllm, '__version__', 'unknown'))\nPY

# 7) Runpod pod-safe startup: keep /start.sh running; it will call /post_start.sh
COPY post_start.sh /post_start.sh
RUN chmod +x /post_start.sh

VOLUME ["/workspace"]
EXPOSE 8000 8001 8002 8003 8080

ENTRYPOINT ["/bin/bash", "/opt/aether/start_all.sh"]
