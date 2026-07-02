"""Batched decoding on the MLX backend must give the same greedy tokens as running each prompt
alone, including rows that join mid-stream. Runs on the real 4-bit checkpoints, so it needs the
mlx extra and the pinned weights; presets that aren't downloaded are skipped.

    uv run pytest -q -m mlx_model
    LHAI_MLX_PRESETS=qwen3.5-9b-mlx4,gemma-4-e4b-mlx4 uv run pytest -q -m mlx_model"""

import os

import pytest

from localhost_ai.config import get_settings
from localhost_ai.device import DeviceConfig
from localhost_ai.engine.controller import FixedController
from localhost_ai.engine.request import Request, SamplingParams
from localhost_ai.engine.scheduler import Scheduler

pytestmark = pytest.mark.mlx_model

N_TOKENS = 24
PROMPTS = [
    "Name three planets.",
    "Write one sentence about the ocean and why it is salty.",
    "2 + 2 =",
    "List the first five prime numbers, separated by commas, and explain what makes a number "
    "prime in one short sentence.",
    "Hello",
]
PRESETS = os.environ.get("LHAI_MLX_PRESETS", "qwen2.5-0.5b-mlx4").split(",")


@pytest.fixture(scope="module", params=PRESETS)
def loaded(request):
    pytest.importorskip("mlx_lm")
    import torch

    from localhost_ai.models.loader import load
    from localhost_ai.models.registry import Registry, local_path

    s = get_settings()
    spec = Registry(s.models_file).get(request.param)
    try:
        local_path(spec, s.models_dir)
    except Exception as exc:  # noqa: BLE001 - only a missing snapshot skips; load errors fail
        pytest.skip(f"{request.param} not downloaded ({type(exc).__name__}); "
                    f"LHAI_MODEL={request.param} uv run lhai models pull")
    dev = DeviceConfig(torch.device("mps"), torch.float16, threads=1)  # unused by mlx
    m = load(spec, dev, s.models_dir)
    yield m
    m.runner.release()


def run(loaded, prompts, limit, join_after=None):
    sched = Scheduler(loaded.runner, loaded.tokenizer, FixedController(limit))
    params = SamplingParams(temperature=0.0, max_tokens=N_TOKENS)
    reqs = [Request(loaded.encode_chat([{"role": "user", "content": p}]), params)
            for p in prompts]
    late = reqs[join_after:] if join_after else []
    for r in reqs[: join_after or len(reqs)]:
        sched.add(r)
    steps = 0
    while sched.has_work() or late:
        sched.step()
        steps += 1
        if late and steps == 5:  # join while the first rows are mid-generation
            for r in late:
                sched.add(r)
            late = []
    return [r.generated for r in reqs]


def test_batched_matches_sequential(loaded):
    alone = [run(loaded, [p], limit=1)[0] for p in PROMPTS]
    batched = run(loaded, PROMPTS, limit=len(PROMPTS), join_after=2)
    for p, a, b in zip(PROMPTS, alone, batched, strict=True):
        same = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
        assert a == b, f"{p!r}: diverged at token {same}"


def test_answers_are_text(loaded):
    out = run(loaded, ["What is the capital of France? Answer in one word."], limit=1)[0]
    assert "Paris" in loaded.tokenizer.decode(out)
