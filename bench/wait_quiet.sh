#!/usr/bin/env bash
# Timing runs share the machine with whatever else is running. Wait (up to MAX_WAIT seconds)
# until the 1-minute load average drops below LOAD_MAX, then log what it was.
max_wait=${MAX_WAIT:-3600}
limit=${LOAD_MAX:-6}
waited=0
while :; do
  load=$(sysctl -n vm.loadavg | awk '{print $2}')
  if awk -v l="$load" -v m="$limit" 'BEGIN{exit !(l < m)}'; then break; fi
  if [ "$waited" -ge "$max_wait" ]; then echo "[quiet] giving up after ${waited}s, load $load" >&2; break; fi
  sleep 30; waited=$((waited + 30))
done
echo "[quiet] load $load after ${waited}s wait" >&2
