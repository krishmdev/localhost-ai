"""Same model, same prompts, same client: this server against mlx_lm.server, llama.cpp's
llama-server and Ollama on one Mac.

Each engine is started from scratch, warmed up, then driven by a closed loop of N streaming
clients per concurrency level (the same prompt order for every engine), and stopped before the
next one starts. Everything is measured on the client side so no engine gets its own counter:

- throughput: completion tokens whose chunk arrived inside the measurement window, per second
  (each request's usage count spread over its content chunks);
- TTFT: time to the first non-empty content chunk;
- TPOT: (last chunk - first chunk) / (completion tokens - 1) per request;
- memory: the server's whole process tree (Ollama serves from a child runner), sampled every
  250 ms. `footprint` is macOS's phys_footprint, what Activity Monitor calls Memory, which
  includes Metal buffers. `rss` also counts resident file-backed pages; for llama.cpp and Ollama,
  which mmap the GGUF, it comes out about one weights-file larger than footprint.

    uv run python bench/baseline.py --engines lhai-aimd,mlx_lm,llama.cpp,ollama \
        --concurrency 1,2,4,8,16,32 --out bench/results/baseline-qwen2.5-3b.json

Run it under the machine's compute lease; the engines share one GPU with everything else.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ctypes
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from loadgen import manifest, ms, pct  # noqa: E402

LHAI_PRESET = "qwen2.5-3b-mlx4"
OLLAMA_MODEL = "qwen2.5:3b-instruct"


# --- memory of a process tree (macOS) --------------------------------------------------------

class _RusageV2(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
        "wired_size", "resident_size", "phys_footprint", "proc_start_abstime",
        "proc_exit_abstime", "child_user_time", "child_system_time", "child_pkg_idle_wkups",
        "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime", "diskio_bytesread",
        "diskio_byteswritten")]


_libc = ctypes.CDLL(None) if sys.platform == "darwin" else None


def footprint(pid: int) -> int | None:
    """phys_footprint of one process, None if it can't be read (gone, or not macOS)."""
    if _libc is None:
        return None
    buf = _RusageV2()
    if _libc.proc_pid_rusage(pid, 2, ctypes.byref(buf)) != 0:  # RUSAGE_INFO_V2
        return None
    return int(buf.phys_footprint)


def tree_memory(pid: int) -> tuple[int, int]:
    """(footprint, rss) summed over a process and all its descendants."""
    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0, 0
    fp = rss = 0
    for p in procs:
        with contextlib.suppress(psutil.Error):
            rss += p.memory_info().rss
            fp += footprint(p.pid) or 0
    return fp, rss


class MemSampler:
    def __init__(self, pid: int, every_s: float = 0.25) -> None:
        self.pid, self.every_s = pid, every_s
        self.peak_fp = self.peak_rss = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            fp, rss = tree_memory(self.pid)
            self.peak_fp, self.peak_rss = max(self.peak_fp, fp), max(self.peak_rss, rss)
            self._stop.wait(self.every_s)

    def __enter__(self) -> MemSampler:
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join()


# --- engines ---------------------------------------------------------------------------------

@dataclass
class Engine:
    name: str
    cmd: list[str]
    ready_path: str
    model: str | None  # value of the request's "model" field; None leaves it out
    env: dict[str, str] = field(default_factory=dict)
    weights: str = ""
    quant: str = ""
    version: str = ""


def sh(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (out.stdout + out.stderr).strip()


def mlx_snapshot() -> Path:
    sys.path.insert(0, str(ROOT / "src"))
    from localhost_ai.config import Settings
    from localhost_ai.models.registry import Registry, local_path

    s = Settings()
    return local_path(Registry(s.models_file).get(LHAI_PRESET), s.models_dir)


def ollama_blob() -> str:
    """The GGUF file behind Ollama's tag, so llama-server runs the very same weights."""
    for line in sh(["ollama", "show", "--modelfile", OLLAMA_MODEL]).splitlines():
        if line.startswith("FROM /"):
            return line.split(" ", 1)[1].strip()
    raise SystemExit(f"ollama has no {OLLAMA_MODEL}; `ollama pull {OLLAMA_MODEL}` first")


def lhai_version() -> str:
    rev = sh(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"])
    return f"localhost-ai {rev}"


def make_engine(name: str, port: int, slots: int, ctx: int) -> Engine:
    if name.startswith("lhai-"):
        mode = name.removeprefix("lhai-")
        env = {"LHAI_MODEL": LHAI_PRESET, "LHAI_MAX_CONTEXT": str(ctx)}
        if mode == "aimd":
            env["LHAI_CONTROLLER"] = "aimd"
        elif mode.startswith("fixed"):
            env |= {"LHAI_CONTROLLER": "fixed", "LHAI_FIXED_BATCH": str(slots)}
        else:
            raise SystemExit(f"unknown lhai mode {mode!r}")
        return Engine(name, ["uv", "run", "lhai", "serve", "--port", str(port),
                             "--log-level", "warning"], "/readyz", None, env,
                      weights=f"mlx-community/Qwen2.5-3B-Instruct-4bit ({LHAI_PRESET})",
                      quant="MLX 4-bit, group size 64", version=lhai_version())
    if name == "mlx_lm":
        import mlx_lm

        # decode concurrency is left at its default (32); unseeded requests are batched
        return Engine(name, ["uv", "run", "mlx_lm.server", "--model", str(mlx_snapshot()),
                             "--port", str(port), "--log-level", "WARNING"],
                      "/v1/models", None,
                      weights="mlx-community/Qwen2.5-3B-Instruct-4bit (same files)",
                      quant="MLX 4-bit, group size 64", version=f"mlx-lm {mlx_lm.__version__}")
    if name == "llama.cpp":
        exe = shutil.which("llama-server")
        if exe is None:
            raise SystemExit("llama-server not found (brew install llama.cpp)")
        version = sh([exe, "--version"]).splitlines()
        return Engine(name, [exe, "-m", ollama_blob(), "--port", str(port), "-np", str(slots),
                             "-c", str(slots * ctx), "-ngl", "99", "--jinja", "--no-webui"],
                      "/health", None, weights=f"Ollama's {OLLAMA_MODEL} GGUF blob",
                      quant="GGUF Q4_K_M",
                      version="llama.cpp " + next((v for v in version if "version" in v), "?"))
    if name == "ollama":
        env = {"OLLAMA_HOST": f"127.0.0.1:{port}", "OLLAMA_NUM_PARALLEL": str(slots),
               "OLLAMA_CONTEXT_LENGTH": str(ctx), "OLLAMA_KEEP_ALIVE": "-1",
               "OLLAMA_MAX_LOADED_MODELS": "1"}
        return Engine(name, ["ollama", "serve"], "/api/version", OLLAMA_MODEL, env,
                      weights=f"{OLLAMA_MODEL}", quant="GGUF Q4_K_M",
                      version=sh(["ollama", "--version"]).splitlines()[-1])
    raise SystemExit(f"unknown engine {name!r}")


class Server:
    def __init__(self, eng: Engine, port: int, log: Path) -> None:
        self.eng, self.url, self.log = eng, f"http://127.0.0.1:{port}", log
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        fh = self.log.open("w")
        self.proc = subprocess.Popen(self.eng.cmd, cwd=ROOT, env={**os.environ, **self.eng.env},
                                     stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.proc.pid, signal.SIGTERM)
        try:
            self.proc.wait(20)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(10)

    async def wait_ready(self, timeout: float = 600) -> float:
        t0 = time.perf_counter()
        async with httpx.AsyncClient(base_url=self.url, timeout=5) as c:
            while time.perf_counter() - t0 < timeout:
                if self.proc.poll() is not None:
                    raise SystemExit(f"{self.eng.name} exited; see {self.log}")
                with contextlib.suppress(httpx.HTTPError):
                    if (await c.get(self.eng.ready_path)).status_code == 200:
                        return time.perf_counter() - t0
                await asyncio.sleep(0.5)
        raise SystemExit(f"{self.eng.name} not ready after {timeout:.0f}s; see {self.log}")


# --- load ------------------------------------------------------------------------------------

@dataclass
class Res:
    ok: bool
    start: float
    end: float
    chunks: list[float] = field(default_factory=list)
    tokens: int | None = None  # from the usage chunk
    status: str = "ok"

    @property
    def n_tokens(self) -> int:
        return self.tokens if self.tokens is not None else len(self.chunks)


async def one(client: httpx.AsyncClient, model: str | None, prompt: str, max_tokens: int) -> Res:
    # No seed: mlx_lm.server only batches unseeded requests.
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.7, "top_p": 0.95, "stream": True,
            "stream_options": {"include_usage": True}}
    if model:
        body["model"] = model
    r = Res(False, time.perf_counter(), 0.0)
    try:
        async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                await resp.aread()
                r.status = f"http_{resp.status_code}"
                r.end = time.perf_counter()
                return r
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                if line[6:].strip() == "[DONE]":
                    break
                obj = json.loads(line[6:])
                if "error" in obj:
                    r.status = "stream_error"
                    r.end = time.perf_counter()
                    return r
                if obj.get("usage"):
                    r.tokens = obj["usage"].get("completion_tokens")
                for ch in obj.get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        r.chunks.append(time.perf_counter())
    except (httpx.HTTPError, OSError) as exc:
        r.status = type(exc).__name__
        r.end = time.perf_counter()
        return r
    r.ok, r.end = True, time.perf_counter()
    return r


async def point(url: str, eng: Engine, conc: int, prompts: list[str], args,
                pid: int) -> dict:
    rng = random.Random(f"{args.seed}-{conc}")  # the same prompt order for every engine
    order = [prompts[rng.randrange(len(prompts))] for _ in range(100_000)]
    nxt = iter(order)
    results: list[Res] = []
    t0 = time.perf_counter()
    lo, hi = t0 + args.warmup, t0 + args.warmup + args.duration
    limits = httpx.Limits(max_connections=conc + 4, max_keepalive_connections=conc + 4)
    with MemSampler(pid) as mem:
        async with httpx.AsyncClient(base_url=url, limits=limits,
                                     timeout=httpx.Timeout(args.request_timeout,
                                                           connect=10)) as c:
            async def worker() -> None:
                while time.perf_counter() < hi:
                    r = await one(c, eng.model, next(nxt), args.max_tokens)
                    results.append(r)
                    if not r.ok:
                        await asyncio.sleep(0.2)

            await asyncio.gather(*(worker() for _ in range(conc)))
    span = hi - lo
    tokens = 0.0
    for r in results:
        if r.chunks:
            per_chunk = r.n_tokens / len(r.chunks)
            tokens += per_chunk * sum(lo <= t <= hi for t in r.chunks)
    done = [r for r in results if lo <= r.end <= hi]
    ok = [r for r in done if r.ok and r.chunks]
    ttft = [r.chunks[0] - r.start for r in ok]
    tpot = [(r.chunks[-1] - r.chunks[0]) / (r.n_tokens - 1) for r in ok if r.n_tokens > 1]
    kinds: dict[str, int] = {}
    for r in done:
        if not r.ok:
            kinds[r.status] = kinds.get(r.status, 0) + 1
    usage = sum(r.tokens is not None for r in ok)
    return {
        "concurrency": conc, "duration_s": round(span, 2), "completed": len(ok),
        "errors": sum(kinds.values()), "error_kinds": kinds,
        "out_tok_per_s": round(tokens / span, 2),
        "tokens_source": "usage" if ok and usage == len(ok) else "chunks" if not usage
        else "mixed",
        "mean_completion_tokens": round(sum(r.n_tokens for r in ok) / len(ok), 1) if ok else None,
        "ttft_p50_ms": ms(pct(ttft, 0.5)), "ttft_p99_ms": ms(pct(ttft, 0.99)),
        "tpot_p50_ms": ms(pct(tpot, 0.5)), "tpot_p99_ms": ms(pct(tpot, 0.99)),
        "peak_footprint_bytes": mem.peak_fp, "peak_rss_bytes": mem.peak_rss,
        "host_load_1m": round(os.getloadavg()[0], 2),
    }


async def run_engine(eng: Engine, port: int, prompts: list[str], args) -> dict:
    srv = Server(eng, port, Path(args.logs) / f"{eng.name}.log")
    srv.start()
    try:
        ready_s = await srv.wait_ready()
        async with httpx.AsyncClient(base_url=srv.url, timeout=300) as c:
            for p in prompts[:2]:  # loads the model (Ollama) and compiles kernels
                r = await one(c, eng.model, p, 16)
                if not r.ok:
                    raise SystemExit(f"{eng.name} warm-up failed: {r.status}; see {srv.log}")
        loaded_fp, loaded_rss = tree_memory(srv.proc.pid)
        points = []
        for conc in args.concurrency:
            pt = await point(srv.url, eng, conc, prompts, args, srv.proc.pid)
            print(f"{eng.name:>10} c={conc:<3} tok/s={pt['out_tok_per_s']:<8} "
                  f"ttft p50/p99={pt['ttft_p50_ms']}/{pt['ttft_p99_ms']} "
                  f"tpot p50/p99={pt['tpot_p50_ms']}/{pt['tpot_p99_ms']} "
                  f"peak fp={pt['peak_footprint_bytes'] / 2**30:.2f}G "
                  f"rss={pt['peak_rss_bytes'] / 2**30:.2f}G err={pt['errors']}",
                  file=sys.stderr, flush=True)
            points.append(pt)
            await asyncio.sleep(args.settle)
    finally:
        srv.stop()
    return {"name": eng.name, "version": eng.version, "weights": eng.weights,
            "quant": eng.quant, "cmd": [Path(a).name if a.startswith("/") else a
                                        for a in eng.cmd],
            "env": eng.env, "ready_s": round(ready_s, 1),
            "loaded_footprint_bytes": loaded_fp, "loaded_rss_bytes": loaded_rss,
            "points": points}


async def main_async(args) -> dict:
    prompts = [json.loads(ln)["prompt"] for ln in Path(args.prompts).read_text().splitlines()
               if ln.strip()]
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    engines = []
    for i, name in enumerate(args.engines):
        port = args.port + i
        eng = make_engine(name, port, args.slots, args.ctx)
        engines.append(await run_engine(eng, port, prompts, args))
        await asyncio.sleep(args.settle)
    cfg = {k: v for k, v in vars(args).items() if k not in ("logs",)}
    cfg["prompts"] = Path(args.prompts).name
    return {"label": args.label, "started": started, "config": cfg, "engines": engines,
            "manifest": manifest(args.manifest, {"target": args.label,
                                                 "engines": ",".join(args.engines)})}


def parse(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engines", default="lhai-aimd,lhai-fixed,mlx_lm,llama.cpp,ollama",
                    type=lambda s: s.split(","))
    ap.add_argument("--concurrency", default="1,2,4,8,16,32",
                    type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--slots", type=int, default=32,
                    help="parallel sequences each engine is configured for")
    ap.add_argument("--ctx", type=int, default=2048, help="context per sequence")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--prompts", default=str(HERE / "prompts.jsonl"))
    ap.add_argument("--port", type=int, default=8431)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="baseline-qwen2.5-3b")
    ap.add_argument("--manifest", default=None, help="path to run_manifest.py")
    ap.add_argument("--logs", default=str(HERE / "results" / "baseline-logs.tmp"))
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse(argv)
    out = asyncio.run(main_async(args))
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
