# syntax=docker/dockerfile:1
# =============================================================================
# AetherVoice — voice-agent-mcp image
#
# Builds the NEXT generation on top of the current one, so the baked Python
# runtime is kept and only the boot entrypoint is replaced.
#
#   docker build -t ghcr.io/njfg-os/voice-agent-mcp:0.2.2 .
#   docker push ghcr.io/njfg-os/voice-agent-mcp:0.2.2
#
# Rules this file follows
#   * BASE_IMAGE is the PREVIOUS generation, never the tag being built. A tag is
#     never built from itself: that loop rewrites its own base on every push and
#     the image grows a layer each time.
#   * /opt/aether/start_all.sh is the container entrypoint (chmod +x, syntax
#     checked at build time). It never installs a runtime and never fails with
#     venv_missing — it accepts whatever Python the base image already has.
#   * ENTRYPOINT is set explicitly so a CMD inherited from the base cannot
#     bypass the entrypoint.
#   * /workspace is only ever used for persistent model caches and voices.
#
# It does NOT freeze the pod's service code: the entrypoint verifies the app's
# served POD_VERSION (sha256 manifest) and runs that, so runtime fixes ship
# without rebuilding this image. The COPY below is the offline fallback only.
#
# The image RunPod runs must be the tag the app pins (POD_IMAGE in
# src/lib/runpod.server.js). Bump both together.
# =============================================================================
ARG BASE_IMAGE=ghcr.io/njfg-os/voice-agent-mcp:0.2.0
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="aether-voice-agent" \
      org.opencontainers.image.description="AetherVoice pod: app-verified boot, baked runtime" \
      org.opencontainers.image.source="https://github.com/NJFG-OS/voice-agent-mcp"

USER root

# curl is the only hard dependency the entrypoint adds (served-script fetch and
# the boot-failure report). No-op when the base image already ships it.
RUN if ! command -v curl >/dev/null 2>&1 || ! command -v bash >/dev/null 2>&1; then \
      apt-get update \
      && apt-get install -y --no-install-recommends curl ca-certificates bash \
      && rm -rf /var/lib/apt/lists/*; \
    fi

# Defaults. The RunPod template overrides the tenant-specific ones
# (AETHER_WORKSPACE_ID, AETHER_NODE_ID, WEBHOOK_SECRET) at pod creation — they
# are deliberately NOT baked here.
ENV AETHER_APP_ORIGIN=https://deft-aether-voice-core.base44.app \
    AETHER_BOOTSTRAP_ONLY=0 \
    AETHER_SCRIPT_CACHE=/tmp/aether/run \
    AETHER_VOICE_DIR=/workspace/tts_voices \
    HF_HOME=/workspace/hf-cache \
    PIP_CACHE_DIR=/workspace/pip-cache \
    PYTHONUNBUFFERED=1 \
    AETHER_IMAGE_VERSION=0.2.2

# The entrypoint. Everything else (supervisor, whisper, vLLM launcher, relay,
# TTS) is fetched and sha256-verified from the app at boot.
COPY start_all.sh /opt/aether/start_all.sh

# OPTIONAL offline fallback: uncomment to bake the service scripts too, so the
# pod can boot with the app unreachable. Put them in ./pod/ next to this file.
# COPY pod/ /opt/aether/pod/

RUN chmod +x /opt/aether/start_all.sh \
    && bash -n /opt/aether/start_all.sh \
    && test -x /opt/aether/start_all.sh

# Persistent model caches and saved voices only.
VOLUME ["/workspace"]

# relay (public), whisper, vllm, tts — the app's fixed port map.
EXPOSE 8000 8001 8002 8080

ENTRYPOINT ["/bin/bash", "/opt/aether/start_all.sh"]
