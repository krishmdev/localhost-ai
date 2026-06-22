#!/usr/bin/env bash
# Timing runs share the machine with whatever else is running (the lease only serializes other
# heavy jobs). Wait up to MAX_WAIT seconds until the host CPU is at least IDLE_MIN percent idle,
# then log what it was. macOS load averages count blocked threads too, so idle time is the
# better signal here.
max_wait=${MAX_WAIT:-2400}
want=${IDLE_MIN:-75}
waited=0
while :; do
  idle=$(top -l 2 -n 0 -s 3 | awk '/CPU usage/ {v=$7} END {sub("%", "", v); print v}')
  if awk -v i="$idle" -v w="$want" 'BEGIN{exit !(i >= w)}'; then break; fi
  if [ "$waited" -ge "$max_wait" ]; then
    echo "[quiet] giving up after ${waited}s, cpu idle ${idle}%" >&2; break
  fi
  sleep 30; waited=$((waited + 36))
done
echo "[quiet] cpu idle ${idle}% after ${waited}s wait" >&2
