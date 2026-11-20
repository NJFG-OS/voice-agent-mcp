#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/aether/logs || true

# Start your stack in background so /start.sh can finish bringing up SSH/Jupyter/web terminal
bash /opt/aether/pod/start_all.sh >> /workspace/aether/logs/supervisor.log 2>&1 &
