"""Does a seeded (or greedy) request produce byte-identical text alone, alone again, and in a batch?

For each temperature the prompts run one at a time twice (run-to-run determinism at batch 1),
then together at each width, once with every row admitted at the start and once with half the
rows joining after four steps (the way a live server fills a batch). A row counts as identical
when its decoded text matches the first alone run exactly.

    uv run python bench/seedcheck.py --preset qwen3.5-9b-mlx4 --widths 8,16 --temps 0.8,0 \
        --out bench/results/seedcheck-qwen3.5-9b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from loadgen import manifest  # noqa: E402

from localhost_ai.config import Settings  # noqa: E402
from localhost_ai.device import configure  # noqa: E402
from localhost_ai.engine.controller import FixedController  # noqa: E402
from localhost_ai.engine.request import ErrorEvent, Request, SamplingParams  # noqa: E402
from localhost_ai.engine.scheduler import Scheduler  # noqa: E402
from localhost_ai.models.loader import load  # noqa: E402
from localhost_ai.models.registry import Registry  # noqa: E402


def run(m, ids, params, idx, limit, join=None):
    sched = Scheduler(m.runner, m.tokenizer, FixedController(limit))
    def report(ev):
        if isinstance(ev, ErrorEvent):
            print(f"row error: {ev.message}", file=sys.stderr)

    reqs = [Request(ids[i], params[i], on_event=report) for i in idx]
    first, late = (reqs[:join], reqs[join:]) if join else (reqs, [])
    for r in first:
        sched.add(r)
    n = 0
    while sched.has_work() or late:
        sched.step()
        n += 1
        if late and n == 4:
            for r in late:
                sched.add(r)
            late = []
    return [(m.tokenizer.decode(r.generated), list(r.generated), r.finish_reason) for r in reqs]


def compare(ref, got):
    """Rows whose text differs from the reference. A row that ended in an error (or with no
    tokens) is listed under `failed` instead, since that is not a sampling difference."""
    rows, failed = [], []
    for i, ((a, at, af), (b, bt, bf)) in enumerate(zip(ref, got, strict=True)):
        if "error" in (af, bf) or not at or not bt:
            failed.append({"row": i, "finish": [af, bf], "tokens": [len(at), len(bt)]})
            continue
        if a == b:
            continue
        k = next((j for j in range(min(len(at), len(bt))) if at[j] != bt[j]),
                 min(len(at), len(bt)))
        rows.append({"row": i, "first_token_diff": k})
    return {"identical": len(ref) - len(rows) - len(failed), "of": len(ref) - len(failed),
            "diverged": rows, "failed": failed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", default="qwen3.5-9b-mlx4")
    ap.add_argument("--widths", default="8,16")
    ap.add_argument("--temps", default="0.8,0")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = Settings()
    m = load(Registry(s.models_file).get(args.preset), configure("auto", "auto", 0), s.models_dir)
    widths = [int(x) for x in args.widths.split(",")]
    n = max(widths)
    prompts = [json.loads(line)["prompt"]
               for line in (HERE / "prompts.jsonl").read_text().splitlines() if line.strip()]
    prompts = [prompts[i % len(prompts)] for i in range(n)]
    ids = [m.encode_chat([{"role": "user", "content": p}]) for p in prompts]
    cases = []
    for temp in [float(x) for x in args.temps.split(",")]:
        params = [SamplingParams(temperature=temp, top_p=0.95, max_tokens=args.max_tokens,
                                 seed=1000 + i) for i in range(n)]
        t = time.perf_counter()
        alone = [run(m, ids, params, [i], 1)[0] for i in range(n)]
        again = [run(m, ids, params, [i], 1)[0] for i in range(n)]
        cases.append({"temperature": temp, "mode": "alone-repeat", "width": 1,
                      **compare(alone, again)})
        for w in widths:
            for join in (None, w // 2):
                got = run(m, ids, params, list(range(w)), w, join=join)
                cases.append({"temperature": temp, "mode": "batched-join" if join else "batched",
                              "width": w, **compare(alone[:w], got)})
        print(json.dumps([c for c in cases if c["temperature"] == temp]), file=sys.stderr)
        print(f"temperature {temp}: {time.perf_counter() - t:.0f}s", file=sys.stderr)
    out = {"preset": args.preset, "max_tokens": args.max_tokens, "top_p": 0.95, "cases": cases,
           "manifest": manifest(args.manifest, {"target": "seedcheck", "preset": args.preset})}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
