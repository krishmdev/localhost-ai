"""What response_format costs per token: the same prompts decoded through the scheduler with
and without a JSON-schema constraint, one row and a batch of eight, on an MLX preset.

Two numbers per case:
- mask_ms_per_row_token: CPU time in the constraint (llguidance mask, unpacking it, the
  masked_fill and advancing the matcher), per row per step;
- step_ms: wall time of a scheduler iteration (decode + sampling) averaged over decode steps.

Greedy decoding, so both variants are deterministic; the constrained answers are checked to
parse.

    uv run python bench/json_overhead.py --preset qwen2.5-3b-mlx4 \
        --out bench/results/json-overhead.json
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
from localhost_ai.engine import sampling  # noqa: E402
from localhost_ai.engine.constrain import Grammars, JSONConstraint  # noqa: E402
from localhost_ai.engine.controller import FixedController  # noqa: E402
from localhost_ai.engine.request import Request, SamplingParams  # noqa: E402
from localhost_ai.engine.scheduler import Scheduler  # noqa: E402
from localhost_ai.models.loader import load  # noqa: E402
from localhost_ai.models.registry import Registry  # noqa: E402

SCHEMA = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "required": ["topic", "summary", "points", "confidence"],
    "properties": {"topic": {"type": "string"}, "summary": {"type": "string"},
                   "points": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                   "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}}}
PROMPTS = ["Explain how a refrigerator works.", "Summarize the causes of the French Revolution.",
           "Give advice for a first marathon.", "What is a hash table?",
           "Describe the water cycle.", "Why do cats purr?", "How does compound interest work?",
           "Compare trains and planes for a 500 km trip."]


class MaskTimer:
    """Times the constraint's share of sampling: sampling.constrain (llguidance mask, unpacking,
    masked_fill) and JSONConstraint.advance, patched for the duration of a run."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self.rows = 0

    def __enter__(self) -> MaskTimer:
        self._constrain, self._advance = sampling.constrain, JSONConstraint.advance

        def constrain(logits, constraints, _f=self._constrain):
            t = time.perf_counter()
            out = _f(logits, constraints)
            self.seconds += time.perf_counter() - t
            self.rows += sum(c is not None for c in constraints or ())
            return out

        def advance(c, token, _f=self._advance):
            t = time.perf_counter()
            _f(c, token)
            self.seconds += time.perf_counter() - t

        sampling.constrain, JSONConstraint.advance = constrain, advance
        return self

    def __exit__(self, *exc) -> None:
        sampling.constrain, JSONConstraint.advance = self._constrain, self._advance


def run(m, grammars, batch: int, constrained: bool, max_tokens: int) -> dict:
    sched = Scheduler(m.runner, m.tokenizer, FixedController(batch))
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    reqs = []
    for p in PROMPTS[:batch]:
        msg = [{"role": "system", "content": "Answer in JSON."}, {"role": "user", "content": p}]
        c = grammars.constraint(SCHEMA) if constrained else None
        reqs.append(Request(m.encode_chat(msg), params, constraint=c))
    for r in reqs:
        sched.add(r)
    with MaskTimer() as mt:
        sched.step()  # prefill and the first token; not in step_ms
        steps, t_steps = 0, 0.0
        while sched.has_work():
            t = time.perf_counter()
            sched.step()
            t_steps += time.perf_counter() - t
            steps += 1
    texts = [m.tokenizer.decode(r.generated, skip_special_tokens=True) for r in reqs]
    parsed = 0
    for r, t in zip(reqs, texts, strict=True):
        if constrained and r.finish_reason == "stop":
            json.loads(t)
            parsed += 1
    return {"batch": batch, "constrained": constrained, "decode_steps": steps,
            "tokens": sum(len(r.generated) for r in reqs),
            "finish": [r.finish_reason for r in reqs], "parsed": parsed if constrained else None,
            "step_ms": round(t_steps / max(1, steps) * 1e3, 3),
            "mask_ms_per_row_token": round(mt.seconds / mt.rows * 1e3, 4) if mt.rows else None,
            "sample": texts[0][:300]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", default="qwen2.5-3b-mlx4")
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = Settings()
    m = load(Registry(s.models_file).get(args.preset), configure("auto", "auto", 0), s.models_dir)
    grammars = Grammars.from_hf(m.tokenizer, m.runner.eos_ids)
    t = time.perf_counter()
    grammars.constraint(SCHEMA)
    first_s = time.perf_counter() - t
    t = time.perf_counter()
    grammars.constraint(SCHEMA)
    compile_s = time.perf_counter() - t
    runs = []
    for b in [int(x) for x in args.batches.split(",")]:
        for rep in range(args.repeats):
            for constrained in (False, True):
                r = run(m, grammars, b, constrained, args.max_tokens)
                r["repeat"] = rep
                print({k: v for k, v in r.items() if k != "sample"}, file=sys.stderr)
                runs.append(r)
    out = {"preset": args.preset, "vocab": len(m.tokenizer), "max_tokens": args.max_tokens,
           "schema": SCHEMA, "first_constraint_s": round(first_s, 3),
           "constraint_compile_ms": round(compile_s * 1e3, 2), "runs": runs,
           "manifest": manifest(args.manifest, {"target": "json-overhead",
                                                "preset": args.preset})}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
