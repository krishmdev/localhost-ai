"""CPU time of one sampling step, the old full-vocabulary sort against the candidate-set sampler
in engine/sampling.py, on real next-token logits.

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


def timed(fn, logits: torch.Tensor, batch: int, reps: int) -> float:
    params = [SamplingParams(temperature=0.7, top_p=0.95)] * batch
    gens = [sampling.make_generator(i) for i in range(batch)]
    n = logits.shape[0] // batch
    fn(logits[:batch], params, gens)  # warm-up
    t = time.perf_counter()
    for r in range(reps):
        i = (r % n) * batch
        fn(logits[i:i + batch], params, gens)
    return (time.perf_counter() - t) / reps


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
    params = [SamplingParams(temperature=0.7, top_p=0.95)] * logits.shape[0]
    probs = sampling.nucleus(logits / 0.7, params)
    fallbacks = sum(ids is not None and ids.numel() == logits.shape[1] for _, ids in probs)
    kept = [int((p > 0).sum()) for p, _ in probs]
    rows = []
    for b in [int(x) for x in args.batches.split(",")]:
        old = timed(full_sort, logits, b, args.reps)
        new = timed(sampling.sample, logits, b, args.reps)
        rows.append({"batch": b, "full_sort_ms": round(old * 1e3, 3),
                     "candidates_ms": round(new * 1e3, 3), "speedup": round(old / new, 1)})
        print(rows[-1], file=sys.stderr)
    out = {"preset": args.preset, "vocab": logits.shape[1], "logit_rows": logits.shape[0],
           "candidates": sampling.CANDIDATES, "temperature": 0.7, "top_p": 0.95,
           "fallback_rows": fallbacks, "nucleus_size_p50": sorted(kept)[len(kept) // 2],
           "nucleus_size_max": max(kept), "torch_threads": torch.get_num_threads(),
           "runs": rows, "manifest": manifest(args.manifest, {"target": "sampler"})}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
