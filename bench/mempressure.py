"""Memory-pressure scenario: the server container gets a small cgroup memory limit and long
generations, then fixed:32 and aimd run the same load. For each mode the server is recreated,
the load runs, and `docker inspect` records whether the kernel OOM-killed the container.

    python bench/mempressure.py --mem-limit 1500m --out bench/results/cpu-mempressure.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import loadgen  # noqa: E402

PROJECT = "lhai"


def compose(*args: str, env: dict[str, str]) -> None:
    subprocess.run(["docker", "compose", "-p", PROJECT, *args], check=True, cwd=HERE.parent,
                   env={**os.environ, **env})


def inspect_server() -> dict:
    cid = subprocess.run(["docker", "compose", "-p", PROJECT, "ps", "-aq", "server"],
                         capture_output=True, text=True, cwd=HERE.parent).stdout.strip()
    if not cid:
        return {"error": "no server container"}
    raw = subprocess.run(["docker", "inspect", cid], capture_output=True, text=True).stdout
    st = json.loads(raw)[0]["State"]
    return {"status": st["Status"], "oom_killed": st["OOMKilled"], "exit_code": st["ExitCode"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem-limit", default="1500m")
    ap.add_argument("--modes", default="fixed:32,aimd")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--duration", type=float, default=90)
    ap.add_argument("--slo-tpot-ms", type=float, default=1000.0,
                    help="loose on purpose: this scenario is about memory, not latency")
    ap.add_argument("--port", default=os.environ.get("LHAI_HOST_PORT", "8410"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    env = {"LHAI_MEM_LIMIT": args.mem_limit}
    url = f"http://127.0.0.1:{args.port}"
    runs = []
    try:
        compose("up", "-d", "--build", env=env)
        for mode in args.modes.split(","):
            compose("up", "-d", "--force-recreate", "--no-deps", "server", env=env)
            asyncio.run(loadgen.wait_ready(url))
            la = loadgen.parse(["--url", url, "--out", "/dev/null", "--label", "cpu-mempressure",
                                "--max-tokens", str(args.max_tokens),
                                "--duration", str(args.duration), "--warmup", "0",
                                "--request-timeout", "600"])
            asyncio.run(loadgen.set_controller(url, "", mode, args.slo_tpot_ms))
            prompts = [json.loads(line)["prompt"]
                       for line in (HERE / "prompts.jsonl").read_text().splitlines() if line]
            t0 = time.time()
            s = asyncio.run(loadgen.run_point(url, mode, args.concurrency, la, prompts,
                                              args.slo_tpot_ms))
            state = inspect_server()
            peak = max((p["mem_used_bytes"] or 0 for p in s.trace), default=None)
            runs.append({**asdict(s), "container": state, "wall_s": round(time.time() - t0, 1),
                         "peak_mem_used_bytes": peak})
            print(f"{mode}: completed={s.completed} errors={s.errors} {state} "
                  f"peak={peak}", file=sys.stderr, flush=True)
    finally:
        compose("down", env=env)

    out = {
        "label": "cpu-mempressure",
        "throughput_source": "server_counter",
        "config": vars(args),
        "runs": runs,
        "manifest": loadgen.manifest(None, {"target": "cpu-docker-mempressure",
                                            "mem_limit": args.mem_limit,
                                            "max_tokens": args.max_tokens}),
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
