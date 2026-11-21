# syntax=docker/dockerfile:1
# =============================================================================
# AetherVoice — voice-agent-mcp image  (generation 0.4.4)
#
#   docker build -t ghcr.io/njfg-os/voice-agent-mcp:0.4.4 .
#   docker push ghcr.io/njfg-os/voice-agent-mcp:0.4.4
#
# WHY 0.4.4 EXISTS
#   Two things a fresh pod used to have to be fixed for by hand are now baked in.
#   1. NUMPY IS PINNED. numpy 2.x is what pyarrow and datasets resolve against, but
#      torch 2.4.x wheels were built against 1.x, so the pin can break the very
#      runtime it is meant to support. select_runtime_pins.py therefore installs the
#      manifest's numpy AFTER the engines and imports the WHOLE stack together
#      (torch, vllm, vllm._C, transformers, faster-whisper, the chosen TTS engine and
#      pyarrow/datasets when present). A pin that fails that gate is rolled back to
#      the numpy the image already had and recorded in the pins, so no image is ever
#      published with a numpy that cannot import its own runtime.
#   2. THE FILE-DESCRIPTOR BUDGET IS BAKED. limits.conf, /etc/profile.d and
#      /etc/bash.bashrc all carry 65536, and /post_start.sh raises it before the stack
#      starts. A live call holds a Twilio media socket plus the relay's upstream
#      connections; the base image's soft limit of 1024 is what a busy pod exhausted
#      into "[Errno 24] Too many open files", taking the only public port with it.
#   /start.sh is the provider's own and is deliberately NOT touched: the app's stack
#   begins at /post_start.sh, so the limit is raised there and inherited by every
#   service.
#
# WHY 0.4.3 EXISTS
#   The base image's nginx config listens on 8001, the port the Ear owns. Whisper died
#   with "OSError: [Errno 98] Address already in use" and the provider showed 8001
#   Ready because nginx was answering there. The 8001 server block is now stripped at
#   build time (strip_nginx_8001.py) and the result is validated, so every pod starts
#   with the port free. Nothing else in the runtime contract changed.
#
# WHY 0.4.2 EXISTS
#   The Mouth's transformers is pinned. Kokoro's warm-up failed on the transformers
#   its TTS wheel pulled in, and the in-container fix (transformers==4.47.1) lived
#   only as long as that container: every fresh pod hit the same failure again. The
#   pin is installed from runtime-candidates.json AFTER the engines, the chosen
#   engine is re-imported, and the finished image asserts the exact version. The
#   runtime contract is otherwise 0.4.1's.
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
    AETHER_NOFILE_TARGET=65536 \
    AETHER_IMAGE_VERSION=0.4.4

COPY runtime-candidates.json /opt/aether/runtime-candidates.json
COPY select_runtime_pins.py /opt/aether/select_runtime_pins.py
COPY strip_nginx_8001.py /opt/aether/strip_nginx_8001.py

# THE 8001 COLLISION. The base image's /etc/nginx/nginx.conf carries a `vscode server`
# block that listens on 8001 and proxies to localhost:8000, which is the port the Ear
# (whisper_server.py) must own. With both on it, Whisper exited with
# "OSError: [Errno 98] Address already in use" on every boot and the provider still
# showed 8001 Ready, because nginx was the process answering there. The block is
# removed from the image here, so the port is free before any pod starts; the script
# refuses to write a config whose braces do not balance, and nginx is syntax-checked
# when the base ships it. /post_start.sh keeps a guard for a pod whose nginx still
# holds the port.
RUN python3 /opt/aether/strip_nginx_8001.py \
    && if command -v nginx >/dev/null 2>&1; then nginx -t 2>&1 | tail -n 3 || true; fi

# THE CUDA DRIVER STUB — scoped to the gate steps below, never to the image.
# A docker build has no GPU and no NVIDIA driver, so libcuda.so.1 does not exist and
# `import vllm._C` fails with "libcuda.so.1: cannot open shared object file" before
# any wheel can be judged. The CUDA devel base ships the driver stub the toolkit
# itself uses for exactly this: resolving the driver API with no driver present.
# It is exposed ONLY through LD_LIBRARY_PATH inside the two gate RUN steps, so it is
# never added to the image's default linker path: at runtime the real driver mounted
# by the NVIDIA container runtime always wins.
RUN set -eu; \
    mkdir -p /opt/aether/cuda-stub; \
    stub=""; \
    for c in /usr/local/cuda/lib64/stubs/libcuda.so \
             /usr/local/cuda/lib64/stubs/libcuda.so.1 \
             /usr/local/cuda/compat/libcuda.so.1; do \
      if [ -e "$c" ]; then stub="$c"; break; fi; \
    done; \
    if [ -z "$stub" ]; then echo "the base image ships no CUDA driver stub (libcuda.so)" >&2; exit 1; fi; \
    ln -sf "$stub" /opt/aether/cuda-stub/libcuda.so.1; \
    ls -l /opt/aether/cuda-stub/libcuda.so.1

# THE BUILD GATE. Tries the declared vLLM ladder in order, installs the candidate,
# and requires torch + vllm + vllm._C to import. It then installs and import-gates
# faster-whisper and one TTS engine. The selected pins are written to
# /opt/aether/runtime-pins.json for the pod's own diagnostics.
# No candidate passing = this step fails = no image, no publish.
RUN export LD_LIBRARY_PATH="/opt/aether/cuda-stub:${LD_LIBRARY_PATH:-}" \
    && python3 /opt/aether/select_runtime_pins.py \
      --manifest /opt/aether/runtime-candidates.json \
      --base "${BASE_IMAGE}" \
      --out /opt/aether/runtime-pins.json \
    && test -s /opt/aether/runtime-pins.json \
    && python3 -c "import json; pins=json.load(open('/opt/aether/runtime-pins.json')); assert pins.get('vllm'), 'pins name no vllm'; assert pins.get('torch'), 'pins name no torch'; print('runtime pins:', json.dumps(pins))"

# The same imports the pod's health contract depends on, proven once more from the
# finished image so a later layer can never shadow them. The stub is scoped to this
# RUN exactly as above.
RUN export LD_LIBRARY_PATH="/opt/aether/cuda-stub:${LD_LIBRARY_PATH:-}" \
    && python3 -c "import torch, vllm, faster_whisper; print('brain+ear ok', torch.__version__, vllm.__version__)" \
    && python3 -c "import importlib, json; pins=json.load(open('/opt/aether/runtime-pins.json')); importlib.import_module(pins['tts_module']); print('mouth ok', pins['tts'])"

# THE MOUTH'S TRANSFORMERS, proven from the finished image. Kokoro's warm-up failed
# on the transformers its TTS wheel pulled in; the in-container fix
# (transformers==4.47.1) never survived that container, so every fresh pod hit the
# same failure. select_runtime_pins.py installs the manifest's voice.transformers
# after the engines; this step proves the pin is really the version in the image, so
# a build that lost it fails here instead of shipping a pod that cannot speak.
RUN python3 -c "import json, transformers; pins=json.load(open('/opt/aether/runtime-pins.json')); want=pins.get('transformers',''); assert want, 'pins name no transformers'; assert want.split('==')[-1] == transformers.__version__, 'transformers %s is not %s' % (transformers.__version__, want); print('mouth transformers ok', want)"

# THE NUMPY PIN AND THE PINS FILE, proven from the finished image. The selector
# records the numpy the image actually imports (and whether the manifest's pin was
# kept or rolled back), so this asserts the two agree instead of trusting the log.
RUN python3 -c "import json, numpy, torch, vllm, transformers, faster_whisper; pins=json.load(open('/opt/aether/runtime-pins.json')); want=pins.get('numpy',''); assert want, 'pins name no numpy'; assert want == numpy.__version__, 'numpy %s is not %s' % (numpy.__version__, want); print('runtime pins aligned:', json.dumps({k: pins.get(k) for k in ('torch','vllm','numpy','numpy_source','transformers','whisper','tts')}))"

# THE FILE-DESCRIPTOR BUDGET, baked into the image. One live call holds a Twilio media
# socket plus the relay's upstream connections; the base image's soft limit of 1024 is
# what a busy pod exhausted into "[Errno 24] Too many open files", which takes the only
# public port with it. The served supervisor, the relay and the speech worker raise it
# again at runtime, so a container started any other way still gets the same budget.
# /start.sh is the provider's own entrypoint and is never rewritten here.
RUN set -eu; \
    touch /etc/security/limits.conf; \
    if ! grep -q 'aether-nofile' /etc/security/limits.conf; then \
      printf '%s\n' \
        '# aether-nofile: one call holds a media socket plus upstream connections' \
        'root soft nofile 65536' \
        'root hard nofile 65536' \
        '* soft nofile 65536' \
        '* hard nofile 65536' >> /etc/security/limits.conf; \
    fi; \
    printf '%s\n' 'ulimit -n 65536 2>/dev/null || true' > /etc/profile.d/aether-nofile.sh; \
    if ! grep -q 'aether-nofile' /etc/bash.bashrc 2>/dev/null; then \
      printf '%s\n' '# aether-nofile' 'ulimit -n 65536 2>/dev/null || true' >> /etc/bash.bashrc; \
    fi; \
    bash -lc 'ulimit -n' | tail -n 1

# THE RELAY'S OWN DEPENDENCIES, proven at build time. voice_relay.py binds 8000 with
# aiohttp: if that wheel is missing the relay crash-loops on import and the pod's only
# public port never opens, which reads as a dead pod rather than a missing package.
# aiohttp normally arrives with the base image's Jupyter, but "normally" is not a
# contract, so it is installed here when absent and then import-gated like the rest.
RUN if python3 -c "import aiohttp" 2>/dev/null; then \
      echo "relay deps ok $(python3 -c 'import aiohttp; print(aiohttp.__version__)')"; \
    else \
      python3 -m pip install --no-cache-dir aiohttp \
      && python3 -c "import aiohttp; print('relay deps installed', aiohttp.__version__)"; \
    fi

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