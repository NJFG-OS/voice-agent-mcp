# syntax=docker/dockerfile:1
# =============================================================================
# AetherVoice — voice-agent-mcp image  (generation 0.4.1)
#
#   docker build -t ghcr.io/njfg-os/voice-agent-mcp:0.4.1 .
#   docker push ghcr.io/njfg-os/voice-agent-mcp:0.4.1
#
# WHY 0.4.1 EXISTS
#   Every pod reported vllm_import_failed, and the cause was the gate itself: both
#   this build gate and the pod's boot gate imported torch.library.infer_schema,
#   which torch 2.4.x does not export at all and which vLLM never imports, so a
#   healthy vLLM was rejected on every boot. The gate now asserts the real contract
#   — torch, vllm and vllm._C — and the manifest's vLLM versions are real published
#   releases, so a candidate can actually pass.
#
# WHAT CHANGED FROM 0.3.x
#   * The Ear and the Mouth are baked in too: faster-whisper and one TTS engine
#     (kokoro, or chatterbox as the fallback) are installed and import-gated in the
#     same build step as vLLM. A pod can no longer boot with tts:false in its own
#     health contract, and no pod installs a runtime at boot.
#   * Port contract is explicit: relay 8000 (public), whisper 8001, vllm 8002 and
#     tts 8080 all loopback. The stale CosyVoice :5000 assumption is gone.
#
# WHAT CHANGED FROM 0.2.x
#   * The base is the official RunPod PyTorch image again, so /start.sh, sshd and
#     the web terminal are the provider's own and are never replaced.
#   * vLLM is installed HERE, at build time, through select_runtime_pins.py. No pod
#     runs `pip install` any more: a runtime install is what pulled a CUDA stack
#     and broke every boot.
#   * The build gate is the release gate. torch, vllm, vllm._C, faster_whisper and
#     a TTS module must all import in the image's own interpreter or the RUN fails,
#     no image is produced and nothing is published.
#   * The app stack starts from /post_start.sh — the hook RunPod's /start.sh runs
#     after sshd and Jupyter are already up. There is deliberately NO ENTRYPOINT
#     and NO CMD here: setting either would clobber the provider's startup.
#
# /workspace stays persistent storage only: HF cache, pip cache, voices, logs.
# The served scripts are fetched and sha256-verified onto the container disk
# (/tmp/aether/run), so a runtime fix still ships from the app without a rebuild.
#
# The image tag RunPod runs must equal POD_IMAGE in src/lib/runpod.server.js.
# Bump both together.
# =============================================================================
ARG BASE_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="aether-voice-agent" \
      org.opencontainers.image.description="AetherVoice pod: RunPod PyTorch base, build-time vLLM gate, app-verified boot" \
      org.opencontainers.image.source="https://github.com/NJFG-OS/voice-agent-mcp"

USER root
ENV DEBIAN_FRONTEND=noninteractive

# curl is what /post_start.sh uses to fetch the served generation and report a
# failed boot. No-op when the base already ships it.
RUN if ! command -v curl >/dev/null 2>&1 || ! command -v bash >/dev/null 2>&1; then \
      apt-get update \
      && apt-get install -y --no-install-recommends curl ca-certificates bash \
      && rm -rf /var/lib/apt/lists/*; \
    fi

# The base image's python is the ONLY runtime this pod uses. Fail the build here
# rather than on a paid GPU if it has no pip.
RUN python3 -m pip --version

ARG BASE_IMAGE

# Defaults. The RunPod template overrides the tenant-specific ones
# (AETHER_WORKSPACE_ID, AETHER_NODE_ID, WEBHOOK_SECRET) at pod creation — they are
# deliberately NOT baked here.
ENV AETHER_APP_ORIGIN=https://deft-aether-voice-core.base44.app \
    AETHER_SCRIPT_CACHE=/tmp/aether/run \
    AETHER_VOICE_DIR=/workspace/tts_voices \
    AETHER_RUNTIME_PINS=/opt/aether/runtime-pins.json \
    HF_HOME=/workspace/hf-cache \
    PIP_CACHE_DIR=/workspace/pip-cache \
    AETHER_WHISPER_PORT=8001 \
    AETHER_VLLM_PORT=8002 \
    AETHER_TTS_PORT=8080 \
    PYTHONUNBUFFERED=1 \
    AETHER_IMAGE_VERSION=0.4.1

COPY runtime-candidates.json /opt/aether/runtime-candidates.json
COPY select_runtime_pins.py /opt/aether/select_runtime_pins.py

# THE BUILD GATE. Tries the declared vLLM ladder in order, installs the candidate,
# and requires torch + torch.library.infer_schema + vllm + vllm._C to import. It
# then installs and import-gates faster-whisper and one TTS engine. The selected
# pins are written to /opt/aether/runtime-pins.json for the pod's own diagnostics.
# No candidate passing = this step fails = no image, no publish.
RUN python3 /opt/aether/select_runtime_pins.py \
      --manifest /opt/aether/runtime-candidates.json \
      --base "${BASE_IMAGE}" \
      --out /opt/aether/runtime-pins.json \
    && test -s /opt/aether/runtime-pins.json \
    && python3 -c "import json; pins=json.load(open('/opt/aether/runtime-pins.json')); assert pins.get('vllm'), 'pins name no vllm'; assert pins.get('torch'), 'pins name no torch'; print('runtime pins:', json.dumps(pins))"

# The same three imports the pod's health contract depends on, proven once more from
# the finished image so a later layer can never shadow them.
RUN python3 -c "import torch, vllm, faster_whisper; print('brain+ear ok', torch.__version__, vllm.__version__)" \
    && python3 -c "import importlib, json; pins=json.load(open('/opt/aether/runtime-pins.json')); importlib.import_module(pins['tts_module']); print('mouth ok', pins['tts'])"

# The app stack. /start.sh (the base image's entrypoint) runs this after sshd and
# Jupyter are up, so the pod stays reachable while the engine boots.
COPY post_start.sh /post_start.sh
RUN chmod +x /post_start.sh \
    && bash -n /post_start.sh \
    && test -x /post_start.sh

# Persistent model caches, voices and logs only.
VOLUME ["/workspace"]

# relay (public), whisper, vllm, tts — the app's fixed port map.
EXPOSE 8000 8001 8002 8080

# NO ENTRYPOINT and NO CMD on purpose: the base image's /start.sh stays the
# container entrypoint and it runs /post_start.sh itself. Overriding either would
# take SSH and the web terminal away from the operator.