#!/usr/bin/env bash
# Headless-Chrome screenshot of the provisioned dashboard (anonymous viewer, kiosk mode).
set -euo pipefail
out=${1:-docs/grafana.png}
from=${2:-now-15m}
port=${GRAFANA_HOST_PORT:-3410}
chrome=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
url="http://127.0.0.1:$port/d/lhai-overview/?orgId=1&kiosk&from=$from&to=now&refresh="
"$chrome" --headless=new --disable-gpu --hide-scrollbars --window-size=1600,1500 \
  --virtual-time-budget=20000 --screenshot="$out" "$url" >/dev/null 2>&1
echo "saved $out"
