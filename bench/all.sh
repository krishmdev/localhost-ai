#!/usr/bin/env bash
# Every measurement in RESULTS.md, in one go. On a shared machine, run it under whatever
# serializes heavy jobs there (`make bench LHAI_LEASE="..."`).
set -euo pipefail
cd "$(dirname "$0")/.."
docker ps --format '{{.Names}}' | grep -q '^lhai-' && { echo "lhai stack already up"; exit 1; }
bench/wait_quiet.sh
bench/native_sweep.sh
bench/wait_quiet.sh
SCREENSHOT=docs/grafana.png bench/docker_sweep.sh
bench/wait_quiet.sh
uv run python bench/mempressure.py --mem-limit "${MEM_LIMIT:-1500m}" \
  --out bench/results/cpu-mempressure.json
