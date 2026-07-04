"""Multi-LoRA on the real Llama-3.2-3B-Instruct 4-bit with EduAI's adapters, against mlx-lm.

Two adapters from the EduAI training run are served at once: the final one (600 iterations)
and its 300-iteration checkpoint. Every prompt is sent three times in one mixed batch (base,
each adapter), and each row's greedy tokens must equal mlx-lm's `stream_generate` on a model
loaded with that adapter (or none). Needs the mlx extra, the llama-3.2-3b-mlx4 weights and the
EduAI adapters; skips otherwise.

    LHAI_EDUAI_ADAPTER=/path/to/eduai/adapters/llama32-3b-eduai uv run pytest -q -m mlx_model \
        tests/test_lora_model.py"""

import os
from pathlib import Path

import pytest

from localhost_ai.config import REPO_ROOT, get_settings
from localhost_ai.engine.controller import FixedController
from localhost_ai.engine.request import Request, SamplingParams
from localhost_ai.engine.scheduler import Scheduler

pytestmark = pytest.mark.mlx_model

N_TOKENS = 24
PROMPTS = ["What gas do plants take in from the air during photosynthesis?",
           "Why does ice float on water?",
           "Name the organelle that makes ATP in a cell."]
ADAPTER_DIR = Path(os.environ.get("LHAI_EDUAI_ADAPTER",
                                  REPO_ROOT.parent / "eduai/adapters/llama32-3b-eduai"))
CHECKPOINT = "0000300_adapters.safetensors"


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    pytest.importorskip("mlx_lm")
    from localhost_ai.models.registry import Registry, local_path

    s = get_settings()
    try:
        base = local_path(Registry(s.models_file).get("llama-3.2-3b-mlx4"), s.models_dir)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"llama-3.2-3b-mlx4 not downloaded ({exc})")
    if not (ADAPTER_DIR / "adapters.safetensors").exists():
        pytest.skip(f"EduAI adapters not found at {ADAPTER_DIR}")
    # mlx-lm loads `adapters.safetensors` from a directory, so give the checkpoint one
    ckpt = tmp_path_factory.mktemp("eduai-300")
    (ckpt / "adapters.safetensors").symlink_to(ADAPTER_DIR / CHECKPOINT)
    (ckpt / "adapter_config.json").symlink_to(ADAPTER_DIR / "adapter_config.json")
    return {"base": base, "eduai": ADAPTER_DIR, "eduai-300": ckpt}


def mlx_lm_tokens(base: Path, adapter: Path | None) -> list[list[int]]:
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    model, tok = load(str(base), adapter_path=str(adapter) if adapter else None)
    out = []
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        # the last response carries the final token (EOS, or the N_TOKENS-th)
        out.append([r.token for r in stream_generate(model, tok, ids, max_tokens=N_TOKENS)])
    del model
    mx.clear_cache()
    return out


@pytest.fixture(scope="module")
def reference(paths):
    return {name: mlx_lm_tokens(paths["base"], None if name == "base" else paths[name])
            for name in ("base", "eduai", "eduai-300")}


@pytest.fixture(scope="module")
def served(paths, reference):  # after the references, so only one model is loaded at a time
    import torch

    from localhost_ai.device import DeviceConfig
    from localhost_ai.models.loader import load
    from localhost_ai.models.registry import Registry

    s = get_settings()
    dev = DeviceConfig(torch.device("cpu"), torch.float32, threads=torch.get_num_threads())
    return load(Registry(s.models_file).get("llama-3.2-3b-mlx4"), dev, s.models_dir,
                adapters=[("eduai", str(ADAPTER_DIR)),
                          ("eduai-300", str(ADAPTER_DIR / CHECKPOINT))])


def run(m, route, limit):
    sched = Scheduler(m.runner, m.tokenizer, FixedController(limit))
    params = SamplingParams(temperature=0.0, max_tokens=N_TOKENS)
    reqs = []
    for p in PROMPTS:
        for a in route:
            r = Request(m.encode_chat([{"role": "user", "content": p}]), params, adapter=a)
            reqs.append(r)
            sched.add(r)
    while sched.has_work():
        sched.step()
    return reqs


def test_adapters_change_the_answers(reference):
    assert reference["eduai"] != reference["base"]
    assert reference["eduai"] != reference["eduai-300"]


def test_mixed_batch_matches_mlx_lm_per_adapter(served, reference):
    route = [None, "eduai", "eduai-300"]
    reqs = run(served, route, limit=len(route) * len(PROMPTS))
    for i, r in enumerate(reqs):
        name = route[i % len(route)] or "base"
        want = reference[name][i // len(route)]
        assert r.generated == want, (name, PROMPTS[i // len(route)],
                                     served.tokenizer.decode(r.generated),
                                     served.tokenizer.decode(want))


def test_single_adapter_batches_match_mlx_lm(served, reference):
    for name in ("eduai", "eduai-300"):
        reqs = run(served, [name], limit=len(PROMPTS))
        for i, r in enumerate(reqs):
            assert r.generated == reference[name][i], name
