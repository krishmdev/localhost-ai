"""Discrete-time simulator for the AIMD batch controller (fixed seeds, no sleeping).

Model, per control interval of 1 s:
- A closed-loop load keeps `demand` requests outstanding, so the queue is never empty.
- Decode-step latency is t0 + k * b for a running batch of b rows, times (1 + N(0, sigma)).
  Steps are simulated back to back until the interval is used up; each one is a TPOT sample
  tagged with the controller epoch at the moment it was measured.
- Samples reach the controller one interval late (measurement delay). The step in flight at a
  tick finishes after it, so time is continuous across intervals.
- When L rises, queued requests are admitted at once. When L falls, rows finish naturally
  (`drain` of the running rows per interval) unless the decision says to shed, in which case
  the newest rows are preempted immediately, as the real scheduler does.
- Memory: used = base + external + running * per_row, as fractions of the limit. The KV ceiling
  is computed the same way the engine computes it.

b* is the capacity boundary, the largest batch whose noise-free latency meets the SLO."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from localhost_ai.engine.controller import (
    AIMDConfig,
    AIMDController,
    Observation,
    Sample,
    kv_ceiling,
)
from localhost_ai.memory import MemSnapshot

MEM_UNIT = 1_000_000  # the probe works in bytes; 1.0 of the limit = 1e6 "bytes"


@dataclass
class Scenario:
    name: str
    t0: float = 0.018
    k: float = 0.001
    slo: float = 0.050
    sigma: float = 0.0
    intervals: int = 200
    demand: int = 200
    drain: float = 0.2
    delay: float = 1.0
    base_mem: float = 0.30
    per_row_mem: float = 0.01
    reserve: float = 0.15
    initial: int = 16
    seed: int = 0
    # (interval, new k) load step, (interval, external memory fraction), interval of an OOM
    k_step: tuple[int, float] | None = None
    mem_shock_at: int | None = None
    oom_at: int | None = None

    def b_star(self, k: float | None = None) -> int:
        return math.floor((self.slo - self.t0) / (k if k is not None else self.k))


@dataclass
class Trace:
    scenario: Scenario
    limit: list[int] = field(default_factory=list)
    running: list[int] = field(default_factory=list)
    p95_true: list[float] = field(default_factory=list)  # p95 of latency actually experienced
    headroom: list[float] = field(default_factory=list)
    used: list[float] = field(default_factory=list)
    ceiling: list[int] = field(default_factory=list)
    b_star: list[int] = field(default_factory=list)
    decisions: list = field(default_factory=list)


def run(sc: Scenario) -> Trace:
    rng = random.Random(sc.seed)
    cfg = AIMDConfig(slo_tpot_s=sc.slo, initial=sc.initial, max_batch=256)
    ctl = AIMDController(cfg)
    tr = Trace(sc)
    pending: list[Sample] = []
    running = min(ctl.limit, sc.demand)
    external = 0.0
    k = sc.k
    t_cursor = 0.0

    for i in range(sc.intervals):
        now = float(i)
        if sc.k_step and i == sc.k_step[0]:
            k = sc.k_step[1]
        if sc.mem_shock_at is not None and i == sc.mem_shock_at:
            used_now = sc.base_mem + external + running * sc.per_row_mem
            external += (1.0 - used_now) / 2  # an outside allocation takes half the headroom

        # deliver samples whose measurement delay has elapsed
        ready = [s for s in pending if s.t + sc.delay <= now]
        pending = [s for s in pending if s.t + sc.delay > now]
        for s in ready:
            ctl.observe(s)

        oom = ctl.report_oom(now) if sc.oom_at is not None and i == sc.oom_at else None

        used = sc.base_mem + external + running * sc.per_row_mem
        mem = MemSnapshot(used=int(used * MEM_UNIT), limit=MEM_UNIT,
                          headroom=max(0, int((1.0 - used) * MEM_UNIT)))
        ceil_ = kv_ceiling(mem, sc.reserve, int(running * sc.per_row_mem * MEM_UNIT),
                           int(sc.per_row_mem * MEM_UNIT), 1)
        d = ctl.tick(Observation(now=now, running=running, queued=sc.demand - running,
                                 mem=mem, kv_ceiling=ceil_))
        if oom is not None:
            tr.decisions.append(oom)
        tr.decisions.append(d)

        L = ctl.limit
        if L >= running:
            running = min(L, sc.demand)
        elif d.shed or oom is not None:
            running = L
        else:
            running = max(L, running - math.ceil(sc.drain * running))

        # decode steps run back to back; the one in flight at a tick finishes after it
        t = max(t_cursor, now)
        lat_this = []
        while t < now + 1.0:
            lat = (sc.t0 + k * running) * max(0.05, 1.0 + rng.gauss(0.0, sc.sigma))
            t += lat
            lat_this.append(lat)
            pending.append(Sample(t=t, tpot_s=lat, epoch=ctl.epoch, batch=running))
        t_cursor = t

        used = sc.base_mem + external + running * sc.per_row_mem
        tr.limit.append(L)
        tr.running.append(running)
        tr.p95_true.append(sorted(lat_this)[max(0, math.ceil(0.95 * len(lat_this)) - 1)]
                           if lat_this else float("nan"))
        tr.used.append(used)
        tr.headroom.append(1.0 - used)
        tr.ceiling.append(ceil_)
        tr.b_star.append(sc.b_star(k))
    return tr


def band(b_star: int) -> tuple[int, int]:
    return math.floor(0.8 * b_star) - 1, b_star + math.ceil(b_star / 10) + 1


SCENARIOS: dict[str, Scenario] = {
    "S1": Scenario("S1 steady load, no noise"),
    "S2": Scenario("S2 steady load, 5% noise", sigma=0.05, seed=2),
    "S3": Scenario("S3 steady load, 15% noise", sigma=0.15, seed=3),
    "S4": Scenario("S4 load step at t=100 (b* halves)", k_step=(100, 0.002), seed=4),
    "S5": Scenario("S5 memory shock at t=100 (headroom halves)", slo=0.5, per_row_mem=0.02,
                   mem_shock_at=100, seed=5),
    "S6": Scenario("S6 OOM at t=100", oom_at=100, seed=6),
}
