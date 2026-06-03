import pytest
from fakes import EOS, FakeRunner, FakeTokenizer, reference

from localhost_ai.engine.controller import AIMDConfig, AIMDController, FixedController
from localhost_ai.engine.request import DoneEvent, ErrorEvent, Request, SamplingParams, TokenEvent
from localhost_ai.engine.scheduler import QueueFull, Scheduler, SchedulerConfig
from localhost_ai.memory import FakeProbe

GREEDY = SamplingParams(temperature=0.0, max_tokens=12)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.001
        return self.t


def make(limit=8, runner=None, controller=None, probe=None, **cfg):
    return Scheduler(runner or FakeRunner(), FakeTokenizer(),
                     controller or FixedController(limit), probe=probe,
                     cfg=SchedulerConfig(**cfg), clock=Clock())


def req(prompt, params=GREEDY):
    events = []
    r = Request(prompt_ids=list(prompt), params=params, on_event=events.append)
    r.events = events
    return r


def drain(s, max_steps=10_000, each=None):
    for i in range(max_steps):
        if each:
            each(i)
        if not s.step() and not s.has_work():
            return i
    raise AssertionError("scheduler did not finish")


def done(r):
    return next(e for e in r.events if isinstance(e, (DoneEvent, ErrorEvent)))


PROMPTS = [[1, 2, 3], [4] * 11, [5, 6], list(range(20)), [7], [8, 9, 10, 11, 12]]


def test_batched_output_matches_sequential():
    s = make(limit=4)
    rs = [req(p) for p in PROMPTS]
    for r in rs:
        s.add(r)
    drain(s)
    for p, r in zip(PROMPTS, rs, strict=True):
        assert r.generated == reference(p, 12)
        assert done(r).finish_reason == "length"


def test_requests_join_mid_stream():
    s = make(limit=8)
    rs = [req(p) for p in PROMPTS]
    s.add(rs[0])

    def feed(i):
        if 1 <= i <= len(rs) - 1:
            s.add(rs[i])

    drain(s, each=feed)
    for p, r in zip(PROMPTS, rs, strict=True):
        assert r.generated == reference(p, 12)


def test_fifo_admission_and_limit_respected():
    runner = FakeRunner()
    s = make(limit=2, runner=runner)
    rs = [req([i + 1]) for i in range(5)]
    for r in rs:
        s.add(r)
    seen = []
    drain(s, each=lambda _: seen.append(len(s.running)))
    assert max(seen) <= 2
    admitted = sorted(rs, key=lambda r: r.admitted_at)
    assert admitted == rs


def test_eos_and_stop_strings():
    runner = FakeRunner(eos_after=6)
    s = make(runner=runner)
    a = req([1, 2, 3])  # 3 prompt tokens + 3 generated -> EOS on the 4th
    s.add(a)
    drain(s)
    assert a.generated[-1] == EOS and done(a).finish_reason == "stop"

    s = make()
    ref = reference([1, 2, 3], 12)
    text = FakeTokenizer().decode(ref)
    stop = text[4:6]
    b = req([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=12, stop=(stop,)))
    s.add(b)
    drain(s)
    emitted = "".join(e.text for e in b.events if isinstance(e, TokenEvent))
    assert done(b).finish_reason == "stop"
    assert emitted == text[: text.find(stop)]


def test_cancel_mid_stream():
    s = make(limit=4)
    a, b = req([1, 2]), req([3, 4])
    s.add(a)
    s.add(b)
    s.step()
    s.step()
    a.cancel()
    drain(s)
    assert done(a).finish_reason == "cancelled"
    assert len(a.generated) < 12
    assert b.generated == reference([3, 4], 12)


def test_cancel_while_queued():
    s = make(limit=1)
    a, b = req([1]), req([2])
    s.add(a)
    s.add(b)
    b.cancel()
    drain(s)
    assert done(b).finish_reason == "cancelled" and b.generated == []


def test_queue_full_raises():
    s = make(max_queue=2)
    s.add(req([1]))
    s.add(req([2]))
    with pytest.raises(QueueFull):
        s.add(req([3]))


def test_oom_preempts_newest_and_output_is_unchanged():
    # 4 rows fit until the padded batch grows past 60 tokens; then the newest row is preempted,
    # re-queued at the front, and re-prefilled with its generated tokens later.
    runner = FakeRunner(oom_above_tokens=60)
    ctl = AIMDController(AIMDConfig(initial=4, min_batch=1))
    s = make(runner=runner, controller=ctl)
    prompts = [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12], [13, 14]]
    rs = [req(p, SamplingParams(temperature=0.0, max_tokens=14)) for p in prompts]
    for r in rs:
        s.add(r)
    drain(s)
    assert s.preemptions >= 1 and s.ooms >= 1
    assert any(r.preemptions for r in rs)
    for p, r in zip(prompts, rs, strict=True):
        assert r.generated == reference(p, 14)
        assert done(r).finish_reason == "length"


def test_oom_is_reported_to_controller():
    runner = FakeRunner(oom_above_tokens=30)
    ctl = AIMDController(AIMDConfig(initial=8))
    s = make(runner=runner, controller=ctl, control_interval_s=0.0)
    for p in PROMPTS[:4]:
        s.add(req(p))
    drain(s)
    assert any(d.action == "oom_backoff" for d in ctl.log)
    assert ctl.limit < 8


def test_single_request_oom_fails_cleanly():
    s = make(runner=FakeRunner(oom_above_tokens=3))
    r = req([1, 2, 3, 4, 5])
    s.add(r)
    drain(s)
    assert isinstance(done(r), ErrorEvent)


def test_memory_decision_sheds_rows():
    probe = FakeProbe(limit=1_000_000, used=100_000)
    ctl = AIMDController(AIMDConfig(initial=6))
    s = make(runner=FakeRunner(), controller=ctl, probe=probe, control_interval_s=0.0)
    rs = [req([i + 1], SamplingParams(temperature=0.0, max_tokens=40)) for i in range(6)]
    for r in rs:
        s.add(r)
    s.step()
    s.step()
    assert len(s.running) == 6
    probe.used = 950_000  # headroom 5% < low watermark
    s.step()
    assert ctl.last.action == "mem_decrease" and ctl.last.shed
    assert ctl.limit < 6 and len(s.running) <= ctl.limit
    probe.used = 100_000
    drain(s)
    for r in rs:
        assert r.generated == reference(r.prompt_ids, 40)


def test_kv_budget_limits_admission():
    # 1000 bytes/token; budget = 0.85 * 200_000 - 100_000 = 70 tokens of KV
    probe = FakeProbe(limit=200_000, used=100_000)
    s = make(limit=16, runner=FakeRunner(), probe=probe)
    rs = [req([1] * 10, SamplingParams(temperature=0.0, max_tokens=10)) for _ in range(8)]
    for r in rs:
        s.add(r)
    s.step()
    # each row projects 10 + 5 tokens, so 4 rows (60 tokens) fit and a 5th (75) doesn't
    assert len(s.running) == 4


def test_seeded_sampling_is_batch_independent():
    params = SamplingParams(temperature=0.9, top_p=0.9, top_k=10, max_tokens=16, seed=1234)
    alone = make(limit=1)
    a = req([1, 2, 3], params)
    alone.add(a)
    drain(alone)

    mixed = make(limit=8)
    others = [req(p, SamplingParams(temperature=1.0, max_tokens=16, seed=i))
              for i, p in enumerate(PROMPTS)]
    b = req([1, 2, 3], params)
    for r in others[:3] + [b] + others[3:]:
        mixed.add(r)
    drain(mixed)
    assert a.generated == b.generated


def test_stats_snapshot():
    s = make(limit=4, control_interval_s=0.0)
    for p in PROMPTS:
        s.add(req(p))
    drain(s)
    st = s.stats()
    assert st["batch_limit"] == 4 and st["running"] == 0
    assert st["tokens_per_s"] > 0
