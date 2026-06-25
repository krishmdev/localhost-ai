#!/usr/bin/env bash
# Meant to run under a wrapper that denies outbound network except localhost (on macOS a
# sandbox-exec profile; `make offline-check LHAI_OFFLINE_RUN=...`). Checks the pinned files, proves the server process itself can't reach out, and
# runs the demo round trip.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== models verify (offline)"
uv run --offline lhai models verify
echo "== egress canary in the CLI process"
uv run --offline lhai egress-check --timeout 2
port=${PORT:-8413}
uv run --offline lhai serve --port "$port" --log-level warning &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null || true' EXIT
for _ in $(seq 120); do curl -sf "localhost:$port/readyz" >/dev/null && break; sleep 1; done
echo "== egress canary inside the running server process"
curl -sf "localhost:$port/v1/admin/egress"
echo
echo "== demo round trip"
curl -sf "localhost:$port/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name three planets."}],"max_tokens":24,"temperature":0}'
echo
uv run --offline python scripts/ws_client.py --url "ws://127.0.0.1:$port" --max-tokens 24 \
  --prompt "Explain RAM in one sentence."
echo "offline check passed"
