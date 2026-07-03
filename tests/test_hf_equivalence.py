"""Batched decoding must produce the same greedy tokens as running each prompt alone.

This is the test that guards engine/kv.py against changes in transformers' cache internals.
It needs the pinned SmolLM2-135M files (`make models`) and runs on CPU in fp32."""

import pytest
import torch

from localhost_ai.config import get_settings
from localhost_ai.device import DeviceConfig
from localhost_ai.engine.controller import FixedController
from localhost_ai.engine.request import Request, SamplingParams
from localhost_ai.engine.scheduler import Scheduler

pytestmark = pytest.mark.model

N_TOKENS = 32
PROMPTS = [
    "Name three planets.",
    "Write one sentence about the ocean and why it is salty.",
    "2 + 2 =",
    "List the first five prime numbers, separated by commas, and explain what makes a number "
    "prime in one short sentence.",
    "Hello",
]


@pytest.fixture(scope="module")
def loaded():
    from localhost_ai.models.loader import load
    from localhost_ai.models.registry import Registry

    s = get_settings()
    try:
        spec = Registry(s.models_file).get("smollm2-135m")
        dev = DeviceConfig(torch.device("cpu"), torch.float32, threads=torch.get_num_threads())
        return load(spec, dev, s.models_dir)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"pinned model not available ({exc}); run `make models`")


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


def first_mismatch(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)


def test_batched_matches_sequential(loaded):
    alone = [run(loaded, [p], limit=1)[0] for p in PROMPTS]
    batched = run(loaded, PROMPTS, limit=len(PROMPTS), join_after=2)
    for p, a, b in zip(PROMPTS, alone, batched, strict=True):
        assert a == b, f"{p!r}: diverged at token {first_mismatch(a, b)}"


def test_logits_match_after_merge(loaded):
    r = loaded.runner
    ids = [loaded.encode_chat([{"role": "user", "content": p}]) for p in PROMPTS[:3]]
    ref = [r.prefill([s])[1][0] for s in ids]
    s01, l01 = r.prefill(ids[:2])
    s2, l2 = r.prefill(ids[2:])
    merged = r.merge(s01, s2)
    step = r.decode(merged, torch.cat([l01, l2]).argmax(-1).tolist())
    for i, s in enumerate(ids):
        tok = int(ref[i].argmax())
        st, _ = r.prefill([s])
        solo = r.decode(st, [tok])
        torch.testing.assert_close(step[i], solo[0], atol=1e-4, rtol=1e-4)
    # rows that finish are filtered out and the rest keep decoding correctly
    kept = r.select(merged, [0, 2])
    nxt = r.decode(kept, step[[0, 2]].argmax(-1).tolist())
    assert nxt.shape[0] == 2


def test_preempted_request_matches_uninterrupted(loaded):
    """Recompute preemption: drop a row's KV mid-generation, re-prefill prompt + generated
    tokens later, and the greedy continuation must be unchanged."""
    alone = run(loaded, PROMPTS[:3], limit=3)
    sched = Scheduler(loaded.runner, loaded.tokenizer, FixedController(3))
    params = SamplingParams(temperature=0.0, max_tokens=N_TOKENS)
    reqs = [Request(loaded.encode_chat([{"role": "user", "content": p}]), params)
            for p in PROMPTS[:3]]
    for r in reqs:
        sched.add(r)
    for _ in range(8):
        sched.step()
    sched._preempt(2)
    assert sum(r.preemptions for r in reqs) == 2
    while sched.has_work():
        sched.step()
    assert [r.generated for r in reqs] == alone


def test_crop_after_failure_mid_stack(loaded):
    """An error raised inside layer 17 leaves layers 0-16 with one extra position. The runner
    crops them back, and the next decode matches a clean run."""
    r = loaded.runner
    ids = [loaded.encode_chat([{"role": "user", "content": p}]) for p in PROMPTS[:2]]
    state, logits = r.prefill(ids)
    toks = logits.argmax(-1).tolist()
    clean_state, _ = r.prefill(ids)
    expected = r.decode(clean_state, toks)

    layer = loaded.runner.model.model.layers[17]

    def boom(*_a, **_k):
        raise torch.OutOfMemoryError("injected")

    h = layer.register_forward_pre_hook(boom)
    with pytest.raises(torch.OutOfMemoryError):
        r.decode(state, toks)
    h.remove()
    lengths = {layer_.keys.shape[2] for layer_ in state.cache.layers}
    assert lengths == {state.mask.shape[1]}
    got = r.decode(state, toks)
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


SYSTEM = ("You are a careful assistant for a small team. Answer in plain English, keep answers "
          "short, say when you are unsure, and never invent facts, names, numbers or sources. "
          "If a question is ambiguous, answer the most likely reading and say so.")


def run_system(loaded, prompts, limit, join_after=None):
    sched = Scheduler(loaded.runner, loaded.tokenizer, FixedController(limit))
    params = SamplingParams(temperature=0.0, max_tokens=N_TOKENS)
    reqs = [Request(loaded.encode_chat([{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": p}]), params)
            for p in prompts]
    late = reqs[join_after:] if join_after else []
    for r in reqs[: join_after or len(reqs)]:
        sched.add(r)
    steps = 0
    while sched.has_work() or late:
        sched.step()
        steps += 1
        if late and steps == 5:
            for r in late:
                sched.add(r)
            late = []
    return [r.generated for r in reqs]


def test_prefix_cache_gives_the_same_greedy_tokens(loaded):
    from localhost_ai.engine.prefix import PrefixCache

    alone = [run_system(loaded, [p], limit=1)[0] for p in PROMPTS]
    loaded.runner.prefix = pc = PrefixCache(budget_bytes=1 << 30, min_tokens=16)
    try:
        cached = run_system(loaded, PROMPTS, limit=len(PROMPTS), join_after=2)
    finally:
        loaded.runner.prefix = None
    assert pc.hits >= len(PROMPTS) - 1 and pc.hit_tokens > 16 * (len(PROMPTS) - 1)
    for p, a, b in zip(PROMPTS, alone, cached, strict=True):
        assert a == b, f"{p!r}: diverged at token {first_mismatch(a, b)}"
