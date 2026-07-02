import math
from dataclasses import dataclass

import pytest
from fakes import EOS, FakeRunner, FakeTokenizer, VirtualClock, reference

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


def test_backend_allocator_errors_count_as_oom():
    from localhost_ai.engine.runner import is_oom

    # the message MLX raises when a buffer is too big for Metal (seen on an M1 Pro)
    assert is_oom(RuntimeError("[metal::malloc] Attempting to allocate 274877906944 bytes which "
                               "is greater than the maximum allowed buffer size of 9534832640 "
                               "bytes."))
    # the other two allocator failures in mlx 0.32.2 (strings from libmlx): the buffer-count
    # limit, and Metal failing to hand out the buffer at all, which is the usual real OOM
    assert is_oom(RuntimeError("[metal::malloc] Resource limit (499000) exceeded."))
    assert is_oom(RuntimeError("[malloc] Unable to allocate 1073741824 bytes."))
    assert is_oom(RuntimeError("MPS backend out of memory (MPS allocated: 1.2 GB)"))
    assert is_oom(MemoryError())
    assert not is_oom(RuntimeError("shapes (2,3) and (4,) cannot be broadcast"))


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


def test_merge_oom_recomputes_whole_batch():
    runner = FakeRunner(merge_ooms=1)
    s = make(limit=8, runner=runner, controller=AIMDController(AIMDConfig(initial=8)))
    rs = [req(p) for p in PROMPTS]
    s.add(rs[0])
    s.step()
    for r in rs[1:]:
        s.add(r)
    drain(s)
    assert s.ooms == 1
    for p, r in zip(PROMPTS, rs, strict=True):
        assert r.generated == reference(p, 12)


@pytest.mark.parametrize("controller", ["fixed", "aimd"])
def test_oom_does_not_thrash(controller):
    runner = FakeRunner(oom_above_tokens=60)
    ctl = FixedController(4) if controller == "fixed" else AIMDController(AIMDConfig(initial=4))
    s = make(runner=runner, controller=ctl)
    prompts = [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12], [13, 14]] * 2
    rs = [req(p, SamplingParams(temperature=0.0, max_tokens=14)) for p in prompts]
    for r in rs:
        s.add(r)
    steps = drain(s)
    assert s.ooms <= 4, f"{s.ooms} OOMs in {steps} steps"
    for p, r in zip(prompts, rs, strict=True):
        assert r.generated == reference(p, 14)


@pytest.mark.parametrize("controller", ["fixed", "aimd"])
def test_prefill_oom_with_nothing_running_does_not_livelock(controller):
    # Two 40-token prompts prefilled together (80 padded tokens) OOM; each alone fits. With
    # nothing running, the fixed limit never drops, so only the prefill cap gets us out.
    runner = FakeRunner(oom_above_tokens=60)
    ctl = FixedController(4) if controller == "fixed" else AIMDController(AIMDConfig(initial=4))
    s = make(runner=runner, controller=ctl)
    prompts = [[i % 30 + 1] * 40 for i in range(2)]
    rs = [req(p, SamplingParams(temperature=0.0, max_tokens=8)) for p in prompts]
    for r in rs:
        s.add(r)
    steps = drain(s, max_steps=500)
    assert s.ooms <= 4, f"{s.ooms} OOMs in {steps} steps"
    for p, r in zip(prompts, rs, strict=True):
        assert r.generated == reference(p, 8)


def test_prefill_time_does_not_drag_the_limit():
    # Closed loop, short outputs, frequent admissions: every iteration with a join pays a
    # prefill (~0.4 ms/token, a 300-token prompt ~ 125 ms). The controller only sees decode
    # steps, so L must still sit in the S1 band around b* = (SLO - t0) / k = 32.
    clock = VirtualClock()
    runner = FakeRunner(clock=clock, t0=0.018, k=0.001, prefill_s=0.005,
                        prefill_per_token=0.0004)
    ctl = AIMDController(AIMDConfig(slo_tpot_s=0.050, initial=16, max_batch=128))
    s = Scheduler(runner, FakeTokenizer(), ctl, cfg=SchedulerConfig(max_queue=512), clock=clock)
    params = SamplingParams(temperature=0.0, max_tokens=64)
    prompt = [i % 38 for i in range(300)]
    demand = 80
    live = []
    limits = []
    last_tick = -1.0
    while clock.t < 150:
        live = [r for r in live if r.finish_reason is None]
        while len(live) < demand:
            r = Request(prompt_ids=prompt, params=params)
            s.add(r)
            live.append(r)
        s.step()
        clock.t += 1e-4  # scheduler overhead, keeps the clock moving
        if s._last_tick != last_tick:
            last_tick = s._last_tick
            limits.append((clock.t, ctl.limit))
    after = [L for t, L in limits if t > 30]
    lo, hi = math.floor(0.8 * 32) - 1, 32 + math.ceil(32 / 10) + 1
    assert after and all(lo <= L <= hi for L in after), (min(after), max(after))


def test_kv_ceiling_does_not_undercut_admission():
    # Before the fix, the ceiling assumed every row was as long as the longest recent request
    # (prompt + half its max_tokens), so one long request in a batch of short ones clamped L
    # far below what the admission budget had just allowed.
    probe = FakeProbe(limit=1_000_000, used=100_000)  # ~750 tokens of KV at 1000 B/token
    ctl = AIMDController(AIMDConfig(initial=16, max_batch=64))
    s = make(runner=FakeRunner(), controller=ctl, probe=probe, control_interval_s=0.0)
    s.add(req([1] * 40, SamplingParams(temperature=0.0, max_tokens=400)))
    for i in range(10):
        s.add(req([i % 30 + 1] * 5, SamplingParams(temperature=0.0, max_tokens=20)))
    s.step()
    admitted = len(s.running)
    assert admitted == 11  # the admission budget fits all of them
    s.step()  # control tick with the mixed batch running
    assert s.kv_ceiling is not None and s.kv_ceiling >= admitted
    assert ctl.limit >= admitted
    assert ctl.last.action != "clamp"


def test_fixed_per_row_state_counts_against_the_kv_budget():
    # A hybrid model (Qwen3.5 on MLX) holds recurrent state per row on top of its KV. At 50 KV
    # tokens' worth per row, a 750-token budget fits 11 rows of 5 + 10 projected tokens, not 20.
    @dataclass
    class HybridRunner(FakeRunner):
        row_state_bytes: int = 50_000

    def run(runner):
        probe = FakeProbe(limit=1_000_000, used=100_000)  # ~750 tokens of budget at 1000 B/token
        ctl = AIMDController(AIMDConfig(initial=32, max_batch=64))
        s = make(runner=runner, controller=ctl, probe=probe, control_interval_s=0.0)
        for i in range(20):
            s.add(req([i % 30 + 1] * 5, SamplingParams(temperature=0.0, max_tokens=20)))
        s.step()
        admitted = len(s.running)
        s.step()
        return admitted, s

    plain_admitted, plain = run(FakeRunner())
    admitted, s = run(HybridRunner())
    assert plain_admitted == 20 and plain.row_bytes(10) == 10_000
    assert admitted == 11 and s.row_bytes(10) == 60_000
    # the ceiling divides by est_seq_len tokens plus 50 tokens' worth per row
    assert s.kv_ceiling is not None and plain.kv_ceiling is not None
    assert s.kv_ceiling < plain.kv_ceiling / 2
    drain(s)


def test_transient_low_headroom_preserves_limit_but_gates_new_admission():
    probe = FakeProbe(limit=1_000_000, used=100_000)
    ctl = AIMDController(AIMDConfig(initial=11, max_batch=16))
    s = make(runner=FakeRunner(), controller=ctl, probe=probe, control_interval_s=0.0)
    running = [req([i + 1] * 5, SamplingParams(temperature=0.0, max_tokens=40))
               for i in range(8)]
    waiting = req([20] * 5, SamplingParams(temperature=0.0, max_tokens=40))
    for r in running:
        s.add(r)
    s.step()
    s.add(waiting)
    assert len(s.running) == 8 and s.queue_depth == 1

    # Headroom is below the reserve but above the emergency watermark. The predictive
    # ceiling conflicts with eight safely running rows, and the per-request admission
    # budget still blocks the ninth.
    probe.used = 882_000
    s.step()
    assert s.kv_ceiling is not None and s.kv_ceiling < 8
    assert ctl.limit == 11 and ctl.last.action == "hold" and not ctl.last.shed
    assert s.queue_depth == 1 and waiting.admitted_at is None

    # A fresh favorable snapshot permits the queued request without waiting for all eight
    # rows to drain. The same path still enforces the prefill and KV budgets.
    probe.used = 100_000
    s.step()
    assert waiting.admitted_at is not None and len(s.running) == 9
    drain(s)
    for r in [*running, waiting]:
        assert isinstance(done(r), DoneEvent)


@pytest.mark.parametrize("controller", ["fixed", "aimd"])
def test_lost_cache_rebuilds_a_smaller_batch(controller):
    # MLX-style: a failed decode loses the whole batch cache, so every row is recomputed. The
    # rebuilt batch has to be smaller and stay that way until a row finishes; otherwise a fixed
    # limit re-admits the same batch and OOMs forever without decoding a token.
    from fakes import LosingFakeRunner

    runner = LosingFakeRunner(oom_at_rows=3)
    ctl = FixedController(4) if controller == "fixed" else AIMDController(AIMDConfig(initial=4))
    s = make(runner=runner, controller=ctl)
    prompts = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10]]
    rs = [req(p, SamplingParams(temperature=0.0, max_tokens=20)) for p in prompts]
    for r in rs:
        s.add(r)
    steps = drain(s, max_steps=2000)
    assert s.ooms <= 3, f"{s.ooms} OOMs, {s.preemptions} preemptions in {steps} steps"
    for p, r in zip(prompts, rs, strict=True):
        assert done(r).finish_reason == "length"
        assert r.generated == reference(p, 20)


def test_runner_row_bytes_and_live_kv_bytes_drive_the_budget():
    # A runner that knows its cache layout (the MLX runner) prices rows itself and reports the
    # bytes its cache really holds; the scheduler uses those instead of tokens x bytes/token.
    @dataclass
    class LayoutRunner(FakeRunner):
        def row_bytes(self, tokens):
            return 200_000  # e.g. a 256-token allocation step, whatever the length

        def kv_bytes(self, state):
            return 123_456

    probe = FakeProbe(limit=1_000_000, used=100_000)  # 750_000 bytes of budget
    s = make(runner=LayoutRunner(), controller=FixedController(16), probe=probe,
             control_interval_s=0.0)
    for i in range(8):
        s.add(req([i + 1] * 5, SamplingParams(temperature=0.0, max_tokens=20)))
    s.step()
    assert len(s.running) == 3  # 3 x 200_000 fits in 750_000, 4 doesn't
    s.step()
    assert s.kv_bytes == 123_456
    assert s.kv_ceiling == (900_000 - 150_000 + 123_456) // 200_000
    drain(s)
