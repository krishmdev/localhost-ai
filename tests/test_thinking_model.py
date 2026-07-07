"""max_thinking_tokens on the real thinking checkpoints (Qwen3.5 `<think>`, Gemma 4's thought
channel), with enable_thinking on per request: the budget closes the block with the model's own
marker, the answer comes after it, and a JSON response_format applies to the answer only.

    LHAI_MLX_PRESETS=qwen3.5-9b-mlx4,gemma-4-e4b-mlx4 uv run pytest -q -m mlx_model \
        tests/test_thinking_model.py"""

import json
import os

import pytest

from localhost_ai.config import get_settings
from localhost_ai.engine.controller import FixedController
from localhost_ai.engine.request import DoneEvent, Request, SamplingParams, TokenEvent
from localhost_ai.engine.scheduler import Scheduler
from localhost_ai.engine.thinking import ThinkSpec, wrap

pytestmark = pytest.mark.mlx_model

PRESETS = os.environ.get("LHAI_MLX_PRESETS", "qwen3.5-9b-mlx4").split(",")
ON = {"enable_thinking": True}
MARKERS = ("<think>", "</think>", "<|channel>", "<channel|>")


@pytest.fixture(scope="module", params=PRESETS)
def loaded(request):
    pytest.importorskip("mlx_lm")
    import torch

    from localhost_ai.device import DeviceConfig
    from localhost_ai.models.loader import load
    from localhost_ai.models.registry import Registry, local_path

    s = get_settings()
    spec = Registry(s.models_file).get(request.param)
    try:
        local_path(spec, s.models_dir)
    except Exception as exc:  # noqa: BLE001 - only a missing snapshot skips
        pytest.skip(f"{request.param} not downloaded ({type(exc).__name__})")
    m = load(spec, DeviceConfig(torch.device("mps"), torch.float16, threads=1), s.models_dir)
    if ThinkSpec.from_tokenizer(m.tokenizer) is None:
        pytest.skip(f"{request.param} has no thinking markers")
    yield m
    m.runner.release()


def generate(m, prompts, budget, fmt=None, max_tokens=96):
    spec = ThinkSpec.from_tokenizer(m.tokenizer)
    sched = Scheduler(m.runner, m.tokenizer, FixedController(len(prompts)))
    grammars = None
    if fmt is not None:
        from localhost_ai.engine.constrain import Grammars

        grammars = Grammars.from_hf(m.tokenizer, m.runner.eos_ids)
    out = []
    for p in prompts:
        ids = m.encode_chat([{"role": "user", "content": p}], ON)
        inner = grammars.constraint(fmt) if grammars else None
        events = []
        r = Request(ids, SamplingParams(temperature=0.0, max_tokens=max_tokens),
                    on_event=events.append,
                    constraint=wrap(spec, ids, m.template_kwargs(ON), budget, inner,
                                    m.runner.eos_ids))
        out.append((r, events))
        sched.add(r)
    while sched.has_work():
        sched.step()
    res = []
    for r, events in out:
        think = "".join(e.text for e in events if isinstance(e, TokenEvent) and e.reasoning)
        content = "".join(e.text for e in events if isinstance(e, TokenEvent) and not e.reasoning)
        done = next(e for e in events if isinstance(e, DoneEvent))
        res.append((r, think, content, done))
    return spec, res


def test_budget_closes_the_block_with_the_models_marker(loaded):
    spec, [(r, think, content, done)] = generate(
        loaded, ["How many prime numbers are there below 30? Answer with the number."], 16)
    assert done.thinking_tokens == 16, (think, content)
    assert r.constraint.forced
    i = next(k for k in range(len(r.generated))
             if r.generated[k:k + len(spec.force_ids)] == list(spec.force_ids))
    assert i >= 16
    assert think.strip() and content.strip()
    assert not any(mk in think + content for mk in MARKERS)
    assert done.finish_reason in ("stop", "length")


def test_rows_in_a_batch_each_keep_their_own_budget(loaded):
    budgets = [4, 12, 24]
    prompts = ["Name a prime number above 50.", "What is 17 times 3?", "Spell 'cat' backwards."]
    spec = ThinkSpec.from_tokenizer(loaded.tokenizer)
    sched = Scheduler(loaded.runner, loaded.tokenizer, FixedController(3))
    rows = []
    for p, b in zip(prompts, budgets, strict=True):
        ids = loaded.encode_chat([{"role": "user", "content": p}], ON)
        r = Request(ids, SamplingParams(temperature=0.0, max_tokens=64),
                    constraint=wrap(spec, ids, loaded.template_kwargs(ON), b, None))
        rows.append(r)
        sched.add(r)
    while sched.has_work():
        sched.step()
    for r, b in zip(rows, budgets, strict=True):
        assert r.constraint.tokens <= b
        if r.constraint.forced:
            assert r.constraint.tokens == b


def test_json_answer_after_the_block(loaded):
    _, [(r, think, content, done)] = generate(
        loaded, ["Give a JSON object with keys name and age for a 30 year old called Ana."], 16,
        fmt={"type": "json_object"}, max_tokens=128)
    assert done.thinking_tokens <= 16 and think.strip()
    assert done.finish_reason == "stop", content
    assert isinstance(json.loads(content), dict)
