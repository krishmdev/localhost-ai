"""Batch-limit controllers.

`AIMDController` adapts the integer batch limit L once per control interval from three signals:
p95 time-per-output-token (TPOT) against an SLO, live memory headroom, and OOM events. The update
rule, in priority order:

1. OOM (never gated): L <- max(Lmin, L // 2), new epoch, 3 intervals of no increases.
2. Headroom below the low watermark (live reading): L <- max(Lmin, floor(0.8 L)), new epoch.
3. Fewer than n_min fresh samples, or the epoch is younger than one interval: hold.
4. Fresh p95 > SLO: L <- max(Lmin, floor(0.8 L)), new epoch.
5. Increase cooldown still running: hold.
6. Fresh p95 < 0.9 SLO, headroom above the high watermark and the batch saturated
   (queue > 0 and running >= L): L <- L + max(1, L // 10), new epoch.
7. Otherwise (p95 in the [0.9 SLO, SLO] deadband): hold.
8. Clamp L to [Lmin, min(Lmax, KV ceiling)].

Fresh evidence: every TPOT sample is tagged with the epoch it was measured in, and only samples
from the current epoch, measured while the running batch was within the current limit, count.
Any change of L starts a new epoch and clears the window. Without this, one slow sample left in a
5 s window keeps triggering decreases after the batch has already shrunk (the stale-window
cascade), because each decrease would be judged on latency measured at the old, larger batch.

AIMD does not settle on one value. Under steady load it saws between roughly 0.8 b* and b*
(b* = the largest batch that meets the SLO)."""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

from ..memory import MemSnapshot

Action = Literal["oom_backoff", "mem_decrease", "slo_decrease", "increase", "hold", "clamp"]


@dataclass(frozen=True)
class Sample:
    t: float
    tpot_s: float
    epoch: int
    batch: int


@dataclass(frozen=True)
class Observation:
    now: float
    running: int
    queued: int
    mem: MemSnapshot | None = None
    kv_ceiling: int | None = None


@dataclass(frozen=True)
class Decision:
    t: float
    action: Action
    reason: str
    limit_before: int
    limit: int
    epoch: int  # epoch after this decision
    evidence_epoch: int  # epoch whose samples this decision looked at
    p95_ms: float | None
    fresh: int
    headroom_frac: float | None
    # True when the scheduler should preempt rows above the new limit right away instead of
    # letting them finish (memory and OOM decisions only).
    shed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AIMDConfig:
    slo_tpot_s: float = 0.100
    min_batch: int = 1
    max_batch: int = 64
    initial: int = 16
    interval_s: float = 1.0
    window_s: float = 5.0
    n_min: int = 20
    mem_low_wm: float = 0.10
    mem_high_wm: float = 0.20
    decrease: float = 0.8
    increase_frac: float = 0.10
    deadband: float = 0.9
    oom_cooldown: int = 3


class Controller(Protocol):
    mode: str

    @property
    def limit(self) -> int: ...
    @property
    def epoch(self) -> int: ...
    def observe(self, sample: Sample) -> None: ...
    def report_oom(self, now: float) -> None: ...
    def tick(self, obs: Observation) -> Decision: ...
    def state(self) -> dict: ...


def p95(values: list[float]) -> float:
    """95th percentile with linear interpolation (NumPy's default method)."""
    s = sorted(values)
    pos = 0.95 * (len(s) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def kv_ceiling(mem: MemSnapshot, reserve: float, kv_in_use: int, bytes_per_token: int,
               est_seq_len: int) -> int:
    """How many rows of `est_seq_len` tokens fit in memory: everything below the reserve line,
    minus what's used by things other than the KV cache, divided by the per-row KV size."""
    budget = mem.limit * (1.0 - reserve) - mem.used + kv_in_use
    per_row = max(1, bytes_per_token * max(1, est_seq_len))
    return max(0, math.floor(budget / per_row))


class _Decisions:
    def __init__(self, maxlen: int = 256) -> None:
        self.log: deque[Decision] = deque(maxlen=maxlen)

    @property
    def last(self) -> Decision | None:
        return self.log[-1] if self.log else None


class FixedController(_Decisions):
    mode = "fixed"

    def __init__(self, batch: int) -> None:
        super().__init__()
        self._limit = max(1, batch)
        self._epoch = 0
        self._oom = False

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def epoch(self) -> int:
        return self._epoch

    def observe(self, sample: Sample) -> None:
        pass

    def report_oom(self, now: float) -> None:
        self._oom = True

    def tick(self, obs: Observation) -> Decision:
        hr = obs.mem.headroom_frac if obs.mem else None
        reason = "fixed mode (oom seen, limit unchanged)" if self._oom else "fixed mode"
        self._oom = False
        d = Decision(obs.now, "hold", reason, self._limit, self._limit, self._epoch, self._epoch,
                     None, 0, hr)
        self.log.append(d)
        return d

    def state(self) -> dict:
        return {"mode": self.mode, "limit": self._limit, "epoch": self._epoch}


class AIMDController(_Decisions):
    mode = "aimd"

    def __init__(self, cfg: AIMDConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or AIMDConfig()
        c = self.cfg
        self._limit = min(max(c.initial, c.min_batch), c.max_batch)
        self._epoch = 0
        self._epoch_started: float | None = None
        self._window: deque[Sample] = deque()
        self._cooldown = 0
        self._oom_pending = False
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def epoch(self) -> int:
        return self._epoch

    def set_slo(self, tpot_s: float) -> None:
        with self._lock:
            self.cfg.slo_tpot_s = tpot_s

    def observe(self, sample: Sample) -> None:
        with self._lock:
            if sample.epoch == self._epoch:
                self._window.append(sample)

    def report_oom(self, now: float) -> None:
        with self._lock:
            self._oom_pending = True

    def _fresh(self, now: float) -> list[float]:
        horizon = now - self.cfg.window_s
        while self._window and self._window[0].t < horizon:
            self._window.popleft()
        return [s.tpot_s for s in self._window if s.batch <= self._limit]

    def _new_epoch(self, now: float) -> None:
        self._epoch += 1
        self._epoch_started = now
        self._window.clear()

    def tick(self, obs: Observation) -> Decision:
        with self._lock:
            return self._tick(obs)

    def _tick(self, obs: Observation) -> Decision:
        c = self.cfg
        now = obs.now
        if self._epoch_started is None:
            self._epoch_started = now
        before = self._limit
        evidence_epoch = self._epoch
        hr = obs.mem.headroom_frac if obs.mem is not None else None
        cooling = self._cooldown > 0
        if cooling:
            self._cooldown -= 1

        fresh = self._fresh(now)
        p = p95(fresh) if fresh else None
        age = now - self._epoch_started
        shed = False
        new = before

        if self._oom_pending:
            self._oom_pending = False
            new = max(c.min_batch, before // 2)
            self._cooldown = c.oom_cooldown
            action, reason, shed = "oom_backoff", "out-of-memory during a step", True
            self._new_epoch(now)
        elif hr is not None and hr < c.mem_low_wm:
            new = max(c.min_batch, math.floor(c.decrease * before))
            action = "mem_decrease"
            reason = f"headroom {hr:.1%} < low watermark {c.mem_low_wm:.0%}"
            shed = True
        elif len(fresh) < c.n_min or age < c.interval_s:
            action = "hold"
            reason = f"waiting for fresh samples ({len(fresh)}/{c.n_min}, epoch age {age:.1f}s)"
        elif p > c.slo_tpot_s:
            new = max(c.min_batch, math.floor(c.decrease * before))
            action = "slo_decrease"
            reason = f"p95 TPOT {p * 1e3:.1f} ms > SLO {c.slo_tpot_s * 1e3:.0f} ms"
        elif cooling:
            action, reason = "hold", "increase cooldown after OOM"
        elif (p < c.deadband * c.slo_tpot_s and (hr is None or hr > c.mem_high_wm)
              and obs.queued > 0 and obs.running >= before):
            new = before + max(1, math.floor(c.increase_frac * before))
            action = "increase"
            reason = f"p95 TPOT {p * 1e3:.1f} ms < {c.deadband:.0%} of SLO and batch saturated"
        else:
            action = "hold"
            if p >= c.deadband * c.slo_tpot_s:
                reason = f"p95 TPOT {p * 1e3:.1f} ms in deadband"
            elif hr is not None and hr <= c.mem_high_wm:
                reason = f"headroom {hr:.1%} <= high watermark"
            else:
                reason = "batch not saturated"

        ceiling = c.max_batch if obs.kv_ceiling is None else min(c.max_batch, obs.kv_ceiling)
        clamped = min(max(new, c.min_batch), max(c.min_batch, ceiling))
        if clamped != new and action in ("hold", "increase"):
            action = "clamp"
            reason = f"limit clamped to KV/max ceiling {ceiling}"
            shed = clamped < before
        new = clamped

        if new != before and action != "oom_backoff":
            self._new_epoch(now)
        self._limit = new
        d = Decision(now, action, reason, before, new, self._epoch, evidence_epoch,
                     None if p is None else p * 1e3, len(fresh), hr, shed)
        self.log.append(d)
        return d

    def state(self) -> dict:
        c = self.cfg
        return {
            "mode": self.mode,
            "limit": self._limit,
            "epoch": self._epoch,
            "cooldown": self._cooldown,
            "slo_tpot_ms": c.slo_tpot_s * 1e3,
            "min_batch": c.min_batch,
            "max_batch": c.max_batch,
        }
