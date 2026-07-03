"""The scheduler driving a real MLXModelRunner (tiny random-weight models built in-process, like
test_mlx_runner.py): seeded sampling alone vs inside a batch, the lost-cache recompute after an
OOM, and cancellation. Needs the mlx extra, so it skips elsewhere."""

import pytest

pytest.importorskip("mlx.core")

from test_mlx_runner import tiny_gemma4, tiny_llama, tiny_qwen3_5  # noqa: E402

from localhost_ai.engine.controller import FixedController  # noqa: E402
from localhost_ai.engine.mlx_runner import MLXModelRunner  # noqa: E402
from localhost_ai.engine.request import DoneEvent, Request, SamplingParams  # noqa: E402
from localhost_ai.engine.scheduler import Scheduler  # noqa: E402

PROMPTS = [[3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14], [15, 16], [17, 18, 19, 20, 21],
           [22, 23, 24, 25, 26, 27]]
N = 12


class Letters:
    """Tokenizer stand-in: one letter per id, no special tokens."""

    eos_token_id = None

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + i % 26) for i in ids)


@pytest.fixture(scope="module", params=["llama", "hybrid", "windowed"])
def runner(request):
    model = {"llama": tiny_llama, "hybrid": tiny_qwen3_5, "windowed": tiny_gemma4}[request.param]
    return MLXModelRunner(model(), eos_ids=frozenset())


def run(runner, prompts, limit, params, join_after=None, each_step=None):
    sched = Scheduler(runner, Letters(), FixedController(limit))
    events: dict[int, list] = {}
    reqs = []
    for i, p in enumerate(prompts):
        events[i] = []
        reqs.append(Request(list(p), params[i], on_event=events[i].append))
    late = reqs[join_after:] if join_after else []
    for r in reqs[: join_after or len(reqs)]:
        sched.add(r)
    steps = 0
    while sched.has_work() or late:
        sched.step()
        steps += 1
        if each_step:
            each_step(sched, steps, reqs)
        if late and steps == 3:
            for r in late:
                sched.add(r)
            late = []
    return reqs, events, sched


def done(events):
    return next(e for e in events if isinstance(e, DoneEvent))


def test_seeded_sampling_is_the_same_alone_and_in_a_batch(runner):
    params = [SamplingParams(temperature=0.9, top_p=0.95, max_tokens=N, seed=100 + i)
              for i in range(len(PROMPTS))]
    alone = [run(runner, [p], 1, [params[i]])[0][0].generated for i, p in enumerate(PROMPTS)]
    batched, _, _ = run(runner, PROMPTS, len(PROMPTS), params, join_after=2)
    assert [r.generated for r in batched] == alone


def test_lost_cache_is_recomputed_with_the_same_output(runner, monkeypatch):
    params = [SamplingParams(temperature=0.0, max_tokens=N)] * len(PROMPTS)
    alone = [run(runner, [p], 1, params[:1])[0][0].generated for p in PROMPTS]

    real = runner._forward
    fired = {"n": 0}
    full = {"n": 0}

    def flaky(cache, tokens):
        # One failure in the third full-batch forward pass (the first is the prefill's last
        # tokens, so this is a decode step), with the error mlx raises when Metal can't hand
        # out a buffer.
        if len(tokens) == len(PROMPTS):
            full["n"] += 1
            if full["n"] == 3 and fired["n"] == 0:
                fired["n"] += 1
                raise RuntimeError("[malloc] Unable to allocate 1073741824 bytes.")
        return real(cache, tokens)

    monkeypatch.setattr(runner, "_forward", flaky)
    reqs, events, sched = run(runner, PROMPTS, len(PROMPTS), params)
    assert fired["n"] == 1 and sched.ooms == 1
    assert sched.preemptions == len(PROMPTS)  # every row was recomputed
    assert [r.generated for r in reqs] == alone
    assert all(done(events[i]).finish_reason == "length" for i in events)


def test_cancel_mid_stream_leaves_the_other_rows_unchanged(runner):
    params = [SamplingParams(temperature=0.0, max_tokens=N)] * len(PROMPTS)
    alone = [run(runner, [p], 1, params[:1])[0][0].generated for p in PROMPTS]

    def cancel_second(sched, steps, reqs):
        if steps == 4:
            reqs[1].cancel()

    reqs, events, _ = run(runner, PROMPTS, len(PROMPTS), params, each_step=cancel_second)
    assert done(events[1]).finish_reason == "cancelled"
    for i, r in enumerate(reqs):
        if i != 1:
            assert r.generated == alone[i]


def test_prefix_cache_gives_the_same_greedy_tokens(runner):
    from localhost_ai.engine.prefix import PrefixCache

    system = list(range(40, 52))
    prompts = [system + p for p in PROMPTS]
    params = [SamplingParams(temperature=0.0, max_tokens=N)] * len(prompts)
    plain = [run(runner, [p], 1, params[:1])[0][0].generated for p in prompts]
    cached = MLXModelRunner(runner.model, eos_ids=frozenset(),
                            prefix_cache=PrefixCache(budget_bytes=1 << 30, min_tokens=8))
    reqs, _, sched = run(cached, prompts, len(prompts), params, join_after=2)
    assert [r.generated for r in reqs] == plain
    assert cached.prefix.hits >= len(prompts) - 1
