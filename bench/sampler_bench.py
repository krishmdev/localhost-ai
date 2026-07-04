"""CPU time of one sampling step, the old full-vocabulary sort against the candidate-set sampler
in engine/sampling.py, on real next-token logits, and a same-seed token check between the two.

Logits are collected from an MLX preset decoding the benchmark prompts at temperature 0.7, then
both samplers run over the same logits for batches of 1 to 32 rows, with the loadgen's settings
(temperature 0.7, top-p 0.95). Also reports how often a row needed the full-sort fallback.

    uv run python bench/sampler_bench.py --preset qwen2.5-3b-mlx4 --out bench/results/sampler.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from loadgen import manifest  # noqa: E402

from localhost_ai.config import Settings  # noqa: E402
from localhost_ai.device import configure  # noqa: E402
from localhost_ai.engine import sampling  # noqa: E402
from localhost_ai.engine.request import SamplingParams  # noqa: E402
from localhost_ai.models.loader import load  # noqa: E402
from localhost_ai.models.registry import Registry  # noqa: E402


def full_sort(logits: torch.Tensor, params: list[SamplingParams],
              generators: list[torch.Generator]) -> list[int]:
    """The sampler before the candidate set (sampling.py at d2717ca), sampled rows only."""
    temps = torch.tensor([p.temperature for p in params]).unsqueeze(1)
    sub = logits / temps
    sorted_logits, sorted_idx = sub.sort(dim=-1, descending=True)
    vocab = sub.shape[-1]
    ranks = torch.arange(vocab).unsqueeze(0)
    top_k = torch.tensor([p.top_k if p.top_k > 0 else vocab for p in params])
    mask = ranks >= top_k.unsqueeze(1)
    probs = sorted_logits.masked_fill(mask, float("-inf")).softmax(dim=-1)
    top_p = torch.tensor([p.top_p for p in params]).unsqueeze(1)
    mask |= (probs.cumsum(dim=-1) - probs) >= top_p
    probs = sorted_logits.masked_fill(mask, float("-inf")).softmax(dim=-1)
    out = []
    for j in range(len(params)):
        pick = torch.multinomial(probs[j], 1, generator=generators[j])
        out.append(int(sorted_idx[j, pick]))
    return out


def collect(preset: str, steps: int) -> torch.Tensor:
    s = Settings()
    m = load(Registry(s.models_file).get(preset), configure("auto", "auto", 0), s.models_dir)
    prompts = [json.loads(ln)["prompt"] for ln in (HERE / "prompts.jsonl").read_text()
               .splitlines() if ln.strip()]
    params = SamplingParams(temperature=0.7, top_p=0.95)
    rows = []
    for i, p in enumerate(prompts):
        g = sampling.make_generator(i)
        state, logits = m.runner.prefill([m.encode_chat([{"role": "user", "content": p}])])
        for _ in range(steps):
            rows.append(logits[0].clone())
            tok = sampling.sample(logits, [params], [g])
            if tok[0] in m.runner.eos_ids:
                break
            logits = m.runner.decode(state, tok)
    return torch.stack(rows)


SETTINGS = {"t0.7-p0.95": dict(temperature=0.7, top_p=0.95),  # the loadgen's
            "t1.0-p0.9": dict(temperature=1.0, top_p=0.9),
            "t0.8-p0.95-k40": dict(temperature=0.8, top_p=0.95, top_k=40),
            "t1.0-p1.0": dict(temperature=1.0, top_p=1.0)}  # API default: always sorts


def params_for(batch: int, seeded: bool, setting: dict, base: int = 0) -> list[SamplingParams]:
    return [SamplingParams(seed=base + i if seeded else None, **setting) for i in range(batch)]


def timed(fn, logits: torch.Tensor, batch: int, reps: int, seeded: bool) -> float:
    params = params_for(batch, seeded, SETTINGS["t0.7-p0.95"])
    gens = [sampling.make_generator(i) for i in range(batch)]
    n = logits.shape[0] // batch
    fn(logits[:batch], params, gens)  # warm-up
    t = time.perf_counter()
    for r in range(reps):
        i = (r % n) * batch
        fn(logits[i:i + batch], params, gens)
    return (time.perf_counter() - t) / reps


def same_seed_check(logits: torch.Tensor, batch: int, setting: dict) -> dict:
    """Seeded rows: the old full sort and the current sampler, each with its own copy of every
    row's generator, over consecutive real logit rows (a generator advances across steps, so
    any difference in how a draw consumes it shows up in later tokens). Also counts the rows
    the current sampler had to sort in full."""
    mismatches, compared, sorted_rows = 0, 0, 0
    first = None
    orig = sampling._full_sort

    def counting(rows, k, p):
        nonlocal sorted_rows
        sorted_rows += len(rows)
        return orig(rows, k, p)

    sampling._full_sort = counting
    try:
        for seed in (0, 7, 42):
            old_gen = [sampling.make_generator(seed + i) for i in range(batch)]
            new_gen = [sampling.make_generator(seed + i) for i in range(batch)]
            for start in range(0, len(logits) - batch + 1, batch):
                chunk = logits[start:start + batch]
                params = params_for(batch, True, setting, seed)
                old = full_sort(chunk, params, old_gen)
                new = sampling.sample(chunk, params, new_gen)
                for i, (a, b) in enumerate(zip(old, new, strict=True)):
                    compared += 1
                    if a != b:
                        mismatches += 1
                        first = first or {"seed": seed, "logit_row": start + i,
                                          "full_sort": a, "candidates": b}
    finally:
        sampling._full_sort = orig
    return {"compared_tokens": compared, "mismatches": mismatches, "first_mismatch": first,
            "full_sorted_rows": sorted_rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", default="qwen2.5-3b-mlx4")
    ap.add_argument("--steps", type=int, default=32, help="decode steps per prompt")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logits = collect(args.preset, args.steps)
    kept = [int((p > 0).sum()) for p, _ in
            sampling.nucleus(logits / 0.7, params_for(len(logits), False, SETTINGS["t0.7-p0.95"]))]
    batches = [int(x) for x in args.batches.split(",")]
    equivalence = {name: {str(b): same_seed_check(logits, b, st) for b in batches}
                   for name, st in SETTINGS.items()}
    for name, per in equivalence.items():
        print(name, per, file=sys.stderr)
    mismatches = sum(v["mismatches"] for per in equivalence.values() for v in per.values())
    rows = []
    for b in batches:
        old = timed(full_sort, logits, b, args.reps, True)
        seeded = timed(sampling.sample, logits, b, args.reps, True)
        unseeded = timed(sampling.sample, logits, b, args.reps, False)
        rows.append({"batch": b, "full_sort_ms": round(old * 1e3, 3),
                     "seeded_ms": round(seeded * 1e3, 3),
                     "unseeded_ms": round(unseeded * 1e3, 3)})
        print(rows[-1], file=sys.stderr)
    out = {"preset": args.preset, "vocab": logits.shape[1], "logit_rows": logits.shape[0],
           "candidates": sampling.CANDIDATES, "timing_setting": SETTINGS["t0.7-p0.95"],
           "nucleus_size_p50": sorted(kept)[len(kept) // 2], "nucleus_size_max": max(kept),
           "torch_threads": torch.get_num_threads(), "same_seed_mismatches": mismatches,
           "same_seed_check": equivalence, "runs": rows,
           "manifest": manifest(args.manifest, {"target": "sampler"})}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    if mismatches:
        raise SystemExit(f"{mismatches} same-seed token mismatches")


if __name__ == "__main__":
    main()
