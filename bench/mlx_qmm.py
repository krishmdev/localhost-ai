"""Time MLX's 4-bit quantized matmul against the number of rows it multiplies, the shape a
decode step has (one token per row). It explains the decode-step column in the mlx_direct
results: whether a batch of B rows costs about one weight read or B of them.

    uv run python bench/mlx_qmm.py --out bench/results/mlx-qmm.json

The weights are a stack of distinct matrices larger than the GPU caches, so every call reads
its weights from memory, as a real forward pass does."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from loadgen import manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4096, help="input features (Qwen3.5-9B hidden)")
    ap.add_argument("--n", type=int, default=12288, help="output features (its MLP width)")
    ap.add_argument("--mats", type=int, default=20)
    ap.add_argument("--rows", default="1,2,4,8,9,12,16,32,64")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mx.random.seed(0)
    w = mx.random.normal((args.n, args.k)).astype(mx.bfloat16)
    mats = [mx.quantize(w + i, group_size=64, bits=4) for i in range(args.mats)]
    mx.eval(mats)
    del w
    weight_bytes = sum(q.nbytes + s.nbytes + b.nbytes for q, s, b in mats)

    runs = []
    for m in [int(x) for x in args.rows.split(",")]:
        x = mx.random.normal((m, 1, args.k)).astype(mx.bfloat16)  # [batch, 1 token, hidden]

        def step(x=x):
            mx.eval([mx.quantized_matmul(x, q, s, b, transpose=True, group_size=64, bits=4)
                     for q, s, b in mats])

        step()
        t = time.perf_counter()
        for _ in range(args.iters):
            step()
        ms = (time.perf_counter() - t) / args.iters * 1e3
        runs.append({"rows": m, "ms": round(ms, 3),
                     "weight_gb_per_s": round(weight_bytes / (ms / 1e3) / 1e9, 1)})
        print(json.dumps(runs[-1]), flush=True)

    out = {"k": args.k, "n": args.n, "mats": args.mats, "bits": 4, "group_size": 64,
           "weight_bytes": weight_bytes, "mlx": mx.__version__, "runs": runs,
           "manifest": manifest(args.manifest, {"bench": "mlx_qmm"})}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
