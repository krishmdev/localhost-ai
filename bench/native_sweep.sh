#!/usr/bin/env bash
# Same sweep against a native `lhai serve` (Apple GPU via MPS on a Mac).
set -euo pipefail
cd "$(dirname "$0")/.."
port=${PORT:-8411}
device=${DEVICE:-mps}
out=${OUT:-bench/results/$device-native.json}
LHAI_DEVICE=$device uv run lhai serve --port "$port" --log-level warning &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
uv run python bench/loadgen.py --url "http://127.0.0.1:$port" --label "$device-native" \
  --calibrate-slo "${SLO_FACTOR:-3}" --modes "${MODES:-fixed:1,fixed:32,aimd}" \
  --concurrency "${CONCURRENCY:-1,4,8,16,32,64}" --max-tokens "${MAX_TOKENS:-128}" \
  --duration "${DURATION:-45}" --out "$out"
