#!/usr/bin/env bash
# The MLX measurements in RESULTS.md: runner-only load/decode numbers for every downloaded MLX
# preset, then a served sweep per preset. Run it under whatever serializes heavy jobs on the
# machine (`LHAI_LEASE=... make bench-mlx`); a second GPU job at the same time skews every number.
set -euo pipefail
cd "$(dirname "$0")/.."
direct=bench/results/mlx-direct.json
rm -f "$direct"
# preset, results label, served modes, concurrency levels, seconds per point (on fd 3, so
# nothing in the loop can eat the list from stdin)
while read -r preset label modes conc dur <&3; do
  if ! LHAI_MODEL=$preset uv run lhai models verify >/dev/null 2>&1; then
    echo "[mlx] $preset not downloaded, skipping" >&2
    continue
  fi
  bench/wait_quiet.sh
  uv run python bench/mlx_direct.py "$preset" --out "$direct"
  bench/wait_quiet.sh
  LHAI_MODEL=$preset LABEL=$label MODES=$modes CONCURRENCY=$conc DURATION=$dur \
    bench/native_sweep.sh
done 3<<'EOF'
qwen2.5-0.5b-mlx4 mlx-qwen2.5-0.5b fixed:1,fixed:32,aimd 1,4,8,16,32 45
gemma-4-e4b-mlx4 mlx-gemma-4-e4b fixed:1,fixed:16,aimd 1,4,8,16 90
qwen3.5-9b-mlx4 mlx-qwen3.5-9b fixed:1,fixed:16,aimd 1,4,8,16 90
EOF
bench/wait_quiet.sh
uv run python bench/mlx_qmm.py --out bench/results/mlx-qmm.json
