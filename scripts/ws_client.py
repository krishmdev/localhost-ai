"""Stream one or more generations over the multiplexed WebSocket API.

    uv run python scripts/ws_client.py --prompt "Explain RAM" --prompt "Name three planets"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import websockets


async def main(args: argparse.Namespace) -> int:
    url = args.url.rstrip("/") + "/v1/ws/generate"
    async with websockets.connect(url) as ws:
        for i, prompt in enumerate(args.prompt):
            await ws.send(json.dumps({"type": "generate", "id": f"r{i}",
                                      "messages": [{"role": "user", "content": prompt}],
                                      "max_tokens": args.max_tokens,
                                      "temperature": args.temperature}))
        texts = {f"r{i}": "" for i in range(len(args.prompt))}
        pending = set(texts)
        single = len(texts) == 1
        while pending:
            msg = json.loads(await ws.recv())
            rid = msg.get("id")
            if msg["type"] == "token":
                texts[rid] += msg["text"]
                if single:
                    print(msg["text"], end="", flush=True)
            elif msg["type"] == "done":
                pending.discard(rid)
                t = msg["timings"]
                if single:
                    print()
                else:
                    print(f"--- {rid}: {args.prompt[int(rid[1:])]}\n{texts[rid]}")
                n = msg["usage"]["completion_tokens"]
                print(f"[{rid}] {msg['finish_reason']}, {n} tokens, "
                      f"ttft {t['ttft_ms']} ms, tpot {t['tpot_ms']} ms", file=sys.stderr)
            elif msg["type"] == "error":
                print(f"[{rid}] error: {msg['message']}", file=sys.stderr)
                pending.discard(rid)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000")
    ap.add_argument("--prompt", action="append", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    sys.exit(asyncio.run(main(ap.parse_args())))
