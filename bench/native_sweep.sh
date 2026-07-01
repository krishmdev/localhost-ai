#!/usr/bin/env bash
# Same sweep against a native `lhai serve` (Apple GPU via MPS on a Mac). For an MLX preset:
#   LHAI_MODEL=qwen2.5-0.5b-mlx4 LABEL=mlx-qwen2.5-0.5b bench/native_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."
port=${PORT:-8411}
device=${DEVICE:-mps}
label=${LABEL:-$device-native}
out=${OUT:-bench/results/$label.json}
LHAI_DEVICE=$device uv run lhai serve --port "$port" --log-level warning &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
uv run python bench/loadgen.py --url "http://127.0.0.1:$port" --label "$label" \
  --calibrate-slo "${SLO_FACTOR:-3}" --modes "${MODES:-fixed:1,fixed:32,aimd}" \
  --concurrency "${CONCURRENCY:-1,4,8,16,32,64}" --max-tokens "${MAX_TOKENS:-128}" \
  --duration "${DURATION:-45}" --out "$out"
