FROM runpod/pytorch:2.1.0-cu1320-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/vllm_cache \
    TRANSFORMERS_CACHE=/workspace/vllm_cache \
    XDG_CACHE_HOME=/workspace/.cache \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AETHER_APP_DIR=/opt/aether \
    COSYVOICE_DIR=/opt/CosyVoice \
    WORKSPACE_DIR=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs \
      ffmpeg sox libsox-dev \
      libsndfile1 \
      build-essential curl wget unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt

RUN git clone --recursive https://github.com/QwenAudio/CosyVoice.git /opt/CosyVoice \
    && pip install --no-cache-dir -r /opt/CosyVoice/requirements.txt

WORKDIR /opt/aether
COPY start_all.sh /opt/aether/start_all.sh
COPY voice_relay.py /opt/aether/voice_relay.py
COPY tts_server.py /opt/aether/tts_server.py
COPY whisper_server.py /opt/aether/whisper_server.py
COPY run_vllm_safe.py /opt/aether/run_vllm_safe.py
RUN chmod +x /opt/aether/start_all.sh

COPY post_start.sh /post_start.sh
RUN chmod +x /post_start.sh
