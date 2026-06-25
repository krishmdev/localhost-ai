"""Closed-loop load generator for the OpenAI-compatible streaming endpoint.

For each controller mode (switched with PUT /v1/admin/controller) and each concurrency level, N
workers send streaming chat requests back to back for --duration seconds after a --warmup.
Client-side TTFT is the time to the first content chunk; TPOT is (last token - first token) /
(tokens - 1) for each request; tokens/s counts completion tokens from the usage chunk. A
WebSocket telemetry subscriber records the server's batch limit, running rows and p95 TPOT once
per interval, so every run also has an L(t) trace.

    python bench/loadgen.py --url http://127.0.0.1:8410 --modes fixed:1,fixed:32,aimd \
        --concurrency 1,4,8,16,32,64 --max-tokens 128 --out bench/results/cpu-docker.json
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import websockets

HERE = Path(__file__).resolve().parent


@dataclass
class Result:
    ok: bool
    start: float
    end: float
    ttft: float | None = None
    tpot: float | None = None
    tokens: int = 0
    status: str = ""


@dataclass
class RunSummary:
    mode: str
    concurrency: int
    duration_s: float
    slo_tpot_ms: float
    completed: int
    errors: int
    rejected_429: int
    req_per_s: float
    out_tok_per_s: float
    ttft_p50_ms: float | None
    ttft_p95_ms: float | None
    req_tpot_p50_ms: float | None  # client-side, per request: (last - first token) / (n - 1)
    req_tpot_p95_ms: float | None
    e2e_p50_ms: float | None
    e2e_p95_ms: float | None
    slo_attainment: float | None
    server_alive_after: bool
    host_load_1m: list = field(default_factory=list)  # [before, after] the point
    host_cpu_idle_before: float | None = None  # percent, from `top` (macOS hosts only)
    error_kinds: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def ms(v: float | None) -> float | None:
    return None if v is None else round(v * 1e3, 2)


def cpu_idle() -> float | None:
    try:
        out = subprocess.run(["top", "-l", "2", "-n", "0", "-s", "1"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [ln for ln in out.splitlines() if ln.startswith("CPU usage")]
    try:
        return float(lines[-1].split(",")[2].split("%")[0])
    except (IndexError, ValueError):
        return None


async def one_request(client: httpx.AsyncClient, prompt: str, max_tokens: int,
                      seed: int) -> Result:
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.7, "top_p": 0.95, "seed": seed, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    first = last = None
    n_chunks = 0
    tokens = 0
    try:
        async with client.stream("POST", "/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                await r.aread()
                return Result(False, t0, time.perf_counter(), status=f"http_{r.status_code}")
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if "error" in obj:
                    return Result(False, t0, time.perf_counter(), status="stream_error")
                if obj.get("usage"):
                    tokens = obj["usage"]["completion_tokens"]
                for ch in obj.get("choices", []):
                    if ch.get("delta", {}).get("content"):
                        now = time.perf_counter()
                        first = first or now
                        last = now
                        n_chunks += 1
    except (httpx.HTTPError, OSError) as exc:
        return Result(False, t0, time.perf_counter(), status=type(exc).__name__)
    end = time.perf_counter()
    tpot = (last - first) / (tokens - 1) if first and last and tokens > 1 else None
    return Result(True, t0, end, ttft=None if first is None else first - t0, tpot=tpot,
                  tokens=tokens, status="ok")


async def telemetry(url: str, token: str, sink: list, stop: asyncio.Event, t_origin: float):
    ws_url = url.replace("http", "ws", 1) + "/v1/ws/telemetry"
    if token:
        ws_url += f"?token={token}"
    with contextlib.suppress(Exception):
        async with websockets.connect(ws_url, open_timeout=5) as ws:
            while not stop.is_set():
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
                except TimeoutError:
                    continue
                if msg.get("type") != "telemetry":
                    continue
                mem = msg.get("memory") or {}
                d = msg.get("last_decision") or {}
                sink.append({
                    "t": round(time.perf_counter() - t_origin, 3),
                    "batch_limit": msg.get("batch_limit"),
                    "running": msg.get("running"),
                    "queued": msg.get("queued"),
                    "decode_step_p95_ms": msg.get("decode_step_p95_ms"),
                    "controller_p95_ms": msg.get("controller_p95_ms"),
                    "iteration_p95_ms": msg.get("iteration_p95_ms"),
                    "tokens_per_s": msg.get("tokens_per_s"),
                    "busy_ratio": msg.get("busy_ratio"),
                    "headroom_frac": mem.get("headroom_frac"),
                    "mem_used_bytes": mem.get("used_bytes"),
                    "action": d.get("action"),
                })


async def run_point(url: str, mode: str, conc: int, args, prompts: list[str],
                    slo_ms: float) -> RunSummary:
    limits = httpx.Limits(max_connections=conc + 4, max_keepalive_connections=conc + 4)
    timeout = httpx.Timeout(args.request_timeout, connect=10)
    results: list[Result] = []
    trace: list = []
    stop_tel = asyncio.Event()
    t_origin = time.perf_counter()
    tel = asyncio.create_task(telemetry(url, args.admin_token, trace, stop_tel, t_origin))
    rng = random.Random(f"{args.seed}-{mode}-{conc}")
    load_before = round(os.getloadavg()[0], 2)
    idle_before = cpu_idle()
    deadline = t_origin + args.warmup + args.duration
    measure_from = t_origin + args.warmup

    async with httpx.AsyncClient(base_url=url, limits=limits, timeout=timeout) as client:
        async def worker(wid: int) -> None:
            while time.perf_counter() < deadline:
                p = prompts[rng.randrange(len(prompts))]
                r = await one_request(client, p, args.max_tokens, rng.randrange(1 << 30))
                results.append(r)
                if r.status in ("ConnectError", "RemoteProtocolError", "ReadError"):
                    await asyncio.sleep(0.5)  # server gone or restarting; don't spin
                elif r.status == "http_429":
                    await asyncio.sleep(0.2)

        await asyncio.gather(*(worker(i) for i in range(conc)))
    stop_tel.set()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(tel, 5)

    # count requests that started after warmup and finished before the deadline (plus the
    # tail that finished within one request timeout after it, so long requests aren't dropped)
    window = [r for r in results if r.start >= measure_from]
    ok = [r for r in window if r.ok]
    errs = [r for r in window if not r.ok]
    span = max(1e-9, max((r.end for r in window), default=deadline) - measure_from)
    tpots = [r.tpot for r in ok if r.tpot is not None]
    kinds: dict[str, int] = {}
    for r in errs:
        kinds[r.status] = kinds.get(r.status, 0) + 1
    alive = await server_ready(url)
    return RunSummary(
        mode=mode, concurrency=conc, duration_s=round(span, 2), slo_tpot_ms=slo_ms,
        completed=len(ok), errors=len(errs), rejected_429=kinds.get("http_429", 0),
        req_per_s=round(len(ok) / span, 3),
        out_tok_per_s=round(sum(r.tokens for r in ok) / span, 2),
        ttft_p50_ms=ms(pct([r.ttft for r in ok if r.ttft is not None], 0.5)),
        ttft_p95_ms=ms(pct([r.ttft for r in ok if r.ttft is not None], 0.95)),
        req_tpot_p50_ms=ms(pct(tpots, 0.5)), req_tpot_p95_ms=ms(pct(tpots, 0.95)),
        e2e_p50_ms=ms(pct([r.end - r.start for r in ok], 0.5)),
        e2e_p95_ms=ms(pct([r.end - r.start for r in ok], 0.95)),
        slo_attainment=(round(sum(t <= slo_ms / 1e3 for t in tpots) / len(tpots), 4)
                        if tpots else None),
        server_alive_after=alive, error_kinds=kinds, trace=trace,
        host_load_1m=[load_before, round(os.getloadavg()[0], 2)],
        host_cpu_idle_before=idle_before,
    )


async def preflight(url: str, token: str) -> dict:
    """What the server's memory probe sees right before the run, plus host swap. On a Mac the
    MPS limit is capped by what the OS can still hand out, so a busy machine shrinks it."""
    ws_url = url.replace("http", "ws", 1) + "/v1/ws/telemetry"
    if token:
        ws_url += f"?token={token}"
    snap: dict = {}
    with contextlib.suppress(Exception):
        async with websockets.connect(ws_url, open_timeout=5) as ws:
            snap = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True)
    mem = snap.get("memory") or {}
    out = {"device": snap.get("device"), "dtype": snap.get("dtype"), "memory": mem,
           "host_swap": swap.stdout.strip() or None, "warnings": []}
    limit = mem.get("limit_bytes") or 0
    if snap.get("device") == "mps" and limit < 4 * 2**30:
        out["warnings"].append(f"MPS memory limit is only {limit / 2**30:.2f} GiB; other "
                               "processes are holding unified memory")
    for w in out["warnings"]:
        print(f"preflight warning: {w}", file=sys.stderr)
    return out


async def server_ready(url: str, timeout: float = 3.0) -> bool:
    try:
        async with httpx.AsyncClient(base_url=url, timeout=timeout) as c:
            return (await c.get("/readyz")).status_code == 200
    except httpx.HTTPError:
        return False


async def wait_ready(url: str, timeout: float = 300.0) -> None:
    t = time.perf_counter()
    while time.perf_counter() - t < timeout:
        if await server_ready(url):
            return
        await asyncio.sleep(1)
    raise SystemExit(f"server at {url} not ready after {timeout:.0f}s")


async def set_controller(url: str, token: str, mode: str, slo_ms: float | None) -> dict:
    kind, _, batch = mode.partition(":")
    body: dict = {"mode": kind}
    if batch:
        body["batch"] = int(batch)
    if slo_ms:
        body["slo_tpot_ms"] = slo_ms
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(base_url=url, timeout=10) as c:
        r = await c.put("/v1/admin/controller", json=body, headers=headers)
        r.raise_for_status()
        return r.json()


def basic_manifest(extra: dict) -> dict:
    import platform

    import torch
    import transformers

    return {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": {"machine": platform.machine(), "os": platform.platform(terse=True),
                     "python": platform.python_version(), "cpu_count": os.cpu_count()},
            "libs": {"torch": torch.__version__, "transformers": transformers.__version__},
            "load_1m": round(os.getloadavg()[0], 1), "extra": extra}


def manifest(cmd: str | None, extra: dict) -> dict:
    """Host/workload record for the results file. LHAI_RUN_MANIFEST may point at an external
    script that prints one JSON object; otherwise a basic built-in record is used."""
    path = cmd or os.environ.get("LHAI_RUN_MANIFEST", "")
    if not path or not Path(path).exists():
        return basic_manifest(extra)
    kv = [f"{k}={v}" for k, v in extra.items()]
    out = subprocess.run([sys.executable, path, *kv], capture_output=True, text=True, timeout=60)
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {"error": out.stderr[-500:]}


async def main_async(args) -> dict:
    prompts = [json.loads(line)["prompt"] for line in Path(args.prompts).read_text().splitlines()
               if line.strip()]
    await wait_ready(args.url)
    async with httpx.AsyncClient(base_url=args.url, timeout=10) as c:
        models = (await c.get("/v1/models")).json()
    pre = await preflight(args.url, args.admin_token)
    modes = args.modes.split(",")
    concs = [int(x) for x in args.concurrency.split(",")]
    slo = args.slo_tpot_ms
    calibration = None
    runs: list[dict] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    if args.calibrate_slo:
        # SLO = factor x the median TPOT of a single request running alone.
        await set_controller(args.url, args.admin_token, "fixed:1", None)
        cal = await run_point(args.url, "fixed:1", 1, args, prompts, slo or 0)
        slo = round(args.calibrate_slo * cal.req_tpot_p50_ms, 1)
        calibration = {"factor": args.calibrate_slo, "baseline_tpot_p50_ms": cal.req_tpot_p50_ms,
                       "slo_tpot_ms": slo}
        print(f"calibrated SLO: {slo} ms ({args.calibrate_slo} x {cal.req_tpot_p50_ms} ms)",
              file=sys.stderr)

    for mode in modes:
        for conc in concs:
            if not await server_ready(args.url):
                if args.wait_restart:
                    await wait_ready(args.url, args.wait_restart)
                else:
                    runs.append({"mode": mode, "concurrency": conc, "skipped": "server down"})
                    continue
            await set_controller(args.url, args.admin_token, mode, slo)
            await asyncio.sleep(args.settle)
            s = await run_point(args.url, mode, conc, args, prompts, slo or 0)
            print(f"{mode:>9} c={conc:<3} req/s={s.req_per_s:<7} tok/s={s.out_tok_per_s:<8} "
                  f"ttft_p95={s.ttft_p95_ms} req_tpot_p95={s.req_tpot_p95_ms} "
                  f"slo={s.slo_attainment} "
                  f"err={s.errors} alive={s.server_alive_after}", file=sys.stderr, flush=True)
            runs.append(asdict(s))

    extra = {"target": args.label, "model": models["data"][0]["id"], "max_tokens": args.max_tokens,
             "duration_s": args.duration}
    return {
        "label": args.label,
        "started": started,
        "config": {k: v for k, v in vars(args).items() if k not in ("admin_token",)},
        "model": models["data"][0],
        "slo_tpot_ms": slo,
        "calibration": calibration,
        "preflight": pre,
        "runs": runs,
        "manifest": manifest(args.manifest, extra),
    }


def parse(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8410")
    ap.add_argument("--label", default="unlabeled", help="e.g. cpu-docker, mps-native")
    ap.add_argument("--modes", default="fixed:1,fixed:32,aimd")
    ap.add_argument("--concurrency", default="1,4,8,16,32,64")
    ap.add_argument("--duration", type=float, default=45.0)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompts", default=str(HERE / "prompts.jsonl"))
    ap.add_argument("--slo-tpot-ms", type=float, default=None)
    ap.add_argument("--calibrate-slo", type=float, default=None,
                    help="set the SLO to this multiple of single-request p50 TPOT")
    ap.add_argument("--admin-token", default=os.environ.get("LHAI_ADMIN_TOKEN", ""))
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--wait-restart", type=float, default=0.0,
                    help="if the server is down between points, wait this long for it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default=None, help="path to run_manifest.py")
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse(argv)
    out = asyncio.run(main_async(args))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
