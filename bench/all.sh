#!/usr/bin/env bash
# Every measurement in RESULTS.md, in one go. Run it under the compute lease:
#   ../.tools/compute_lease.py run localhost-ai-bench -- bench/all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
docker ps --format '{{.Names}}' | grep -q '^lhai-' && { echo "lhai stack already up"; exit 1; }
bench/native_sweep.sh
SCREENSHOT=docs/grafana.png bench/docker_sweep.sh
uv run python bench/mempressure.py --mem-limit "${MEM_LIMIT:-1500m}" \
  --out bench/results/cpu-mempressure.json
