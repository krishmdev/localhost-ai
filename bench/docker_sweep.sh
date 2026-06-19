#!/usr/bin/env bash
# Concurrency sweep against the compose stack (CPU in Docker). Brings the stack up, runs the
# sweep, saves a Grafana screenshot, and always tears the stack down.
set -euo pipefail
cd "$(dirname "$0")/.."
port=${LHAI_HOST_PORT:-8410}
out=${OUT:-bench/results/cpu-docker.json}
export LHAI_MEM_LIMIT=${LHAI_MEM_LIMIT:-4g}
docker compose -p lhai up -d --build
trap 'docker compose -p lhai down' EXIT
uv run python bench/loadgen.py --url "http://127.0.0.1:$port" --label cpu-docker \
  --calibrate-slo "${SLO_FACTOR:-3}" --modes "${MODES:-fixed:1,fixed:32,aimd}" \
  --concurrency "${CONCURRENCY:-1,4,8,16,32,64}" --max-tokens "${MAX_TOKENS:-128}" \
  --duration "${DURATION:-45}" --out "$out"
if [ -n "${SCREENSHOT:-}" ]; then
  scripts/grafana_screenshot.sh "$SCREENSHOT" "${SCREENSHOT_RANGE:-now-15m}"
fi
