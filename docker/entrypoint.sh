#!/usr/bin/env bash
set -euo pipefail

# Sidecars, then the server manager as the main process. The manager starts
# the selected backend (llama.cpp or vLLM) on 127.0.0.1:8080 using
# /state/server.json; on the first run it seeds that file from the env
# (.env). After that, /server owns the settings.

if [[ "${HW_METRICS:-1}" == "1" ]]; then
  python3 /opt/hw-metrics/hw-metrics.py --host 127.0.0.1 --port 8083 &
fi
nginx

echo "monitor http://<host>:8888/monitor/"
echo "server  http://<host>:8888/server/"

exec python3 /opt/server/manager.py --host 127.0.0.1 --port 8090
