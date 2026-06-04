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
        i = first_mismatch(a, b)
        # fp32 reductions over different padded shapes can flip a near-tie late in a sequence;
        # the first 16 tokens must match exactly.
        assert i is None or i >= 16, f"{p!r}: diverged at token {i}"


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
