#!/usr/bin/env bash
set -euo pipefail

# Sidecars, then the server manager as the main process. The manager starts
# the selected backend (llama.cpp or vLLM) on 127.0.0.1:8080 using
# /state/server.json; on the first run it seeds that file from the env
# (.env). After that, /server owns the settings.

# Live code: docker-compose.yml mounts the repo's python/ at /opt/src (read-only, a directory so
# git checkouts that replace files still show through). Point the image's copies at it, so edits
# need no rebuild: /server and /monitor on a page reload, the manager on
# POST /server/api/manager/restart, hw-metrics on a container restart. Without the mount (the
# image run on its own) the copies baked in by docker/Dockerfile are used.
if [[ -d /opt/src/Server && -d /opt/src/Dashboard ]]; then
  ln -sf /opt/src/Server/manager.py /opt/server/manager.py
  ln -sf /opt/src/Server/index.html /opt/server/www/index.html
  ln -sf /opt/src/Dashboard/monitor.html /opt/monitor/index.html
  ln -sf /opt/src/Dashboard/hw-metrics.py /opt/hw-metrics/hw-metrics.py
  echo "live code from /opt/src (the repo's python/)"
fi

if [[ "${HW_METRICS:-1}" == "1" ]]; then
  python3 /opt/hw-metrics/hw-metrics.py --host 127.0.0.1 --port 8083 &
fi
nginx

echo "monitor http://<host>:8888/monitor/"
echo "server  http://<host>:8888/server/"

exec python3 /opt/server/manager.py --host 127.0.0.1 --port 8090
