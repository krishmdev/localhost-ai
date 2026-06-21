#!/usr/bin/env bash
# Start a native server, exercise REST, SSE, WebSocket and metrics once, then stop it.
set -euo pipefail
cd "$(dirname "$0")/.."
port=${PORT:-8412}
uv run lhai serve --port "$port" --log-level warning &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null || true' EXIT
for _ in $(seq 120); do curl -sf "localhost:$port/readyz" >/dev/null && break; sleep 1; done
curl -sf "localhost:$port/readyz"; echo
curl -sf "localhost:$port/v1/models"; echo
curl -sf "localhost:$port/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name three planets."}],"max_tokens":24,"temperature":0}'
echo
curl -sN "localhost:$port/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi."}],"max_tokens":8,"stream":true,"stream_options":{"include_usage":true}}' | tail -4
uv run python scripts/ws_client.py --url "ws://127.0.0.1:$port" --max-tokens 24 \
  --prompt "Explain RAM in one sentence." --prompt "Name a prime number."
uv run python scripts/ws_top.py --url "ws://127.0.0.1:$port" --once
curl -sf "localhost:$port/metrics/" | grep -E '^lhai_(batch_limit|generated_tokens_total|decode_step_p95_seconds) '
