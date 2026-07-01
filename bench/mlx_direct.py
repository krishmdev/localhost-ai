"""Load an MLX preset and time the runner directly, without HTTP or the controller: load time and
memory, then greedy decode at a few fixed batch sizes. It answers "does this model fit and how
fast is it on this machine"; bench/native_sweep.sh measures the served system.

    LHAI_RUN_MANIFEST=~/path/run_manifest.py \\
        uv run python bench/mlx_direct.py qwen2.5-0.5b-mlx4 --out bench/results/mlx-direct.json

Several presets can share one results file; each run replaces its own entry."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from loadgen import manifest  # noqa: E402

from localhost_ai.config import get_settings  # noqa: E402
from localhost_ai.device import DeviceConfig  # noqa: E402
from localhost_ai.memory import MlxProbe  # noqa: E402
from localhost_ai.models.loader import load  # noqa: E402
from localhost_ai.models.registry import Registry  # noqa: E402

GiB = 2**30


def mem() -> dict:
    return {"active_gib": round(mx.get_active_memory() / GiB, 3),
            "peak_gib": round(mx.get_peak_memory() / GiB, 3),
            "cache_gib": round(mx.get_cache_memory() / GiB, 3)}


def decode_run(m, prompts: list[list[int]], steps: int) -> dict:
    """Prefill the batch, then `steps` greedy decode steps; EOS is ignored so every row decodes
    the same number of tokens."""
    r = m.runner
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    state, logits = r.prefill(prompts)
    t1 = time.perf_counter()
    tokens = logits.argmax(-1).tolist()
    first = list(tokens)
    for _ in range(steps):
        tokens = r.decode(state, tokens).argmax(-1).tolist()
    t2 = time.perf_counter()
    out = {"batch": len(prompts), "prompt_tokens": [len(p) for p in prompts],
           "prefill_s": round(t1 - t0, 3), "decode_steps": steps,
           "decode_s": round(t2 - t1, 3),
           "decode_step_ms": round((t2 - t1) / steps * 1e3, 2),
           "decode_tok_per_s": round(len(prompts) * steps / (t2 - t1), 1),
           "sample": m.tokenizer.decode(first[:1]), **mem()}
    del state
    r.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preset")
    ap.add_argument("--batches", default="1,4,8,16")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s = get_settings()
    spec = Registry(s.models_file).get(args.preset)
    probe_before = MlxProbe().snapshot()
    mx.reset_peak_memory()
    m = load(spec, DeviceConfig(torch.device("mps"), torch.float16, 1), s.models_dir)
    loaded = {"load_s": round(m.load_s, 2), **mem()}
    probe_after = MlxProbe().snapshot()

    prompts = [json.loads(line)["prompt"]
               for line in (HERE / "prompts.jsonl").read_text().splitlines()]
    ids = [m.encode_chat([{"role": "user", "content": p}]) for p in prompts]

    # one greedy answer, as a sanity check that the model produces text
    q = m.encode_chat([{"role": "user", "content": "What is the capital of France? One word."}])
    state, logits = m.runner.prefill([q])
    answer = []
    for _ in range(16):
        t = int(logits.argmax(-1)[0])
        if t in m.runner.eos_ids:
            break
        answer.append(t)
        logits = m.runner.decode(state, [t])
    del state

    runs = []
    for b in [int(x) for x in args.batches.split(",")]:
        batch = [ids[i % len(ids)] for i in range(b)]
        runs.append(decode_run(m, batch, args.steps))
        print(json.dumps(runs[-1]), flush=True)

    entry = {
        "preset": spec.name, "repo": spec.repo, "revision": spec.revision,
        "quant": m.quant, "dtype": m.dtype_name,
        "kv_bytes_per_token": m.runner.kv_bytes_per_token,
        "row_state_bytes": m.runner.row_state_bytes,
        "metal_recommended_working_set_gib": round(MlxProbe()._max / GiB, 2),
        "probe_before_load": {"used_gib": round(probe_before.used / GiB, 3),
                              "limit_gib": round(probe_before.limit / GiB, 3)},
        "probe_after_load": {"used_gib": round(probe_after.used / GiB, 3),
                             "limit_gib": round(probe_after.limit / GiB, 3),
                             "headroom_frac": round(probe_after.headroom_frac, 3)},
        "load": loaded, "answer": m.tokenizer.decode(answer), "runs": runs,
        "manifest": manifest(args.manifest, {"bench": "mlx_direct", "preset": spec.name}),
    }
    out = Path(args.out)
    data = json.loads(out.read_text()) if out.exists() else {}
    data[spec.name] = entry
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(json.dumps({k: entry[k] for k in ("preset", "load", "answer", "probe_after_load")}))


if __name__ == "__main__":
    main()
