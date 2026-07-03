#!/usr/bin/env bash
# Prefix caching on a shared-system-prompt workload: the same sweep with the cache off and on,
# every request carrying bench/system_prompt.txt as its system message. Run it under whatever
# serializes heavy jobs on the machine.
set -euo pipefail
cd "$(dirname "$0")/.."
# preset, results label, served modes, concurrency levels, seconds per point
while read -r preset label modes conc dur <&3; do
  if ! LHAI_MODEL=$preset uv run lhai models verify >/dev/null 2>&1; then
    echo "[prefix] $preset not downloaded, skipping" >&2
    continue
  fi
  for cache in off on; do
    bench/wait_quiet.sh
    LHAI_PREFIX_CACHE=$([ "$cache" = on ] && echo 1 || echo 0) LHAI_MODEL=$preset \
      LABEL=prefix-$label-$cache MODES=$modes CONCURRENCY=$conc DURATION=$dur \
      SYSTEM=bench/system_prompt.txt bench/native_sweep.sh
  done
done 3<<'EOF_LIST'
smollm2-135m smollm2-135m fixed:16 1,4,8,16 30
qwen3.5-9b-mlx4 qwen3.5-9b fixed:8 1,4,8 60
EOF_LIST
