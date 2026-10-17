#!/usr/bin/env bash
# Container health (pages + server manager), then the model backend if one is loaded.
set -euo pipefail

PORT="${LLM_PORT:-${LLAMA_PORT:-8888}}"
BASE="${BASE_URL:-http://127.0.0.1:${PORT}}"

echo "GET $BASE/server/api/health"
curl -fsS "$BASE/server/api/health"
echo
if curl -fsS -o /dev/null "$BASE/health" 2>/dev/null; then
  echo "GET $BASE/v1/models"
  curl -fsS "$BASE/v1/models"
  echo
else
  echo "no model loaded — start one at $BASE/server/"
fi
