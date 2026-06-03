"""Iteration-level continuous batching.

One `step()` is one scheduler iteration:
1. drop cancelled requests; run the controller if a control interval has passed;
2. admit waiting requests while the batch is under the controller's limit L, the prefill token
   budget and the KV budget;
3. prefill the admitted requests together and sample their first token;
4. merge them into the running batch (left-pad + concat) and run one decode step for every row;
5. filter out rows that finished.

The wall time of an iteration is the inter-token latency (TPOT) every row that was already
running experienced, so that's the sample fed to the controller.

On an out-of-memory error the newest row is preempted by recompute (vLLM style): its KV is
dropped, it goes back to the front of the queue, and when it's admitted again the prompt plus
the tokens generated so far are prefilled. Its sampler state is kept, so the output is the same
as an uninterrupted run."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..memory import MemoryProbe, MemSnapshot
from .controller import Controller, Decision, Observation, Sample, kv_ceiling
from .detok import Decoder, IncrementalDetokenizer
from .request import DoneEvent, ErrorEvent, FinishReason, Request, TokenEvent
from .runner import ModelRunner, is_oom
from .sampling import make_generator, sample


class QueueFull(Exception):
    def __init__(self, depth: int, retry_after_s: float) -> None:
        super().__init__(f"queue full ({depth} waiting)")
        self.depth = depth
        self.retry_after_s = retry_after_s


class NullMetrics:
    def step(self, duration_s: float, batch: int, prefill_tokens: int) -> None: ...
    def token(self, n: int) -> None: ...
    def finished(self, req: Request, done: DoneEvent) -> None: ...
    def decision(self, d: Decision) -> None: ...
    def preempted(self) -> None: ...
    def oom(self) -> None: ...
    def gauges(self, sched: Scheduler) -> None: ...


@dataclass
class SchedulerConfig:
    max_queue: int = 256
    max_prefill_tokens_per_step: int = 2048
    max_context: int = 2048
    control_interval_s: float = 1.0
    mem_reserve: float = 0.15
    stats_window_s: float = 5.0


class Scheduler:
    def __init__(self, runner: ModelRunner, tokenizer: Decoder, controller: Controller,
                 probe: MemoryProbe | None = None, cfg: SchedulerConfig | None = None,
                 metrics: Any = None, clock: Callable[[], float] = time.perf_counter) -> None:
        self.runner = runner
        self.tokenizer = tokenizer
        self.controller = controller
        self.probe = probe
        self.cfg = cfg or SchedulerConfig()
        self.metrics = metrics or NullMetrics()
        self.clock = clock

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.state: Any = None
        self._lock = threading.Lock()
        self._next_controller: Controller | None = None

        self.mem: MemSnapshot | None = probe.snapshot() if probe else None
        self._last_tick = clock()
        self._recent_len: deque[int] = deque(maxlen=64)
        self.busy_s = 0.0
        self.busy_ratio = 0.0
        self._busy_at_tick = 0.0
        self.steps: deque[tuple[float, float, int]] = deque()  # (t, tpot_s, batch)
        self.ttfts: deque[tuple[float, float]] = deque()
        self.tokens: deque[tuple[float, int]] = deque()
        self.kv_ceiling: int | None = None
        self.last_decision: Decision | None = None
        self.preemptions = 0
        self.ooms = 0
        self._stats: dict = self._compute_stats()

    # --- called from other threads ---------------------------------------------------------

    def add(self, req: Request) -> int:
        with self._lock:
            if len(self.waiting) >= self.cfg.max_queue:
                raise QueueFull(len(self.waiting), self.cfg.control_interval_s)
            self.waiting.append(req)
            return len(self.waiting)

    def swap_controller(self, controller: Controller) -> None:
        self._next_controller = controller

    @property
    def queue_depth(self) -> int:
        return len(self.waiting)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # --- compute thread --------------------------------------------------------------------

    def step(self) -> bool:
        now = self.clock()
        if self._next_controller is not None:
            self.controller, self._next_controller = self._next_controller, None
        self._drop_cancelled()
        if now - self._last_tick >= self.cfg.control_interval_s:
            self.tick(now)
        if not self.running and not self.waiting:
            return False

        t_start = self.clock()
        had_rows = len(self.running)
        prefill_tokens = self._admit_and_prefill()
        if self.running:
            self._decode()
        t_end = self.clock()
        dur = t_end - t_start
        self.busy_s += dur
        if had_rows:
            self.controller.observe(Sample(t=t_end, tpot_s=dur, epoch=self.controller.epoch,
                                           batch=had_rows))
            self.steps.append((t_end, dur, had_rows))
        self.metrics.step(dur, len(self.running), prefill_tokens)
        return True

    def tick(self, now: float) -> Decision:
        elapsed = max(1e-9, now - self._last_tick)
        self.busy_ratio = min(1.0, (self.busy_s - self._busy_at_tick) / elapsed)
        self._busy_at_tick = self.busy_s
        self._last_tick = now
        self._trim_stats(now)
        if self.probe is not None:
            self.mem = self.probe.snapshot()
        kv_in_use = self.kv_tokens * self.runner.kv_bytes_per_token
        self.kv_ceiling = None
        if self.mem is not None:
            self.kv_ceiling = kv_ceiling(self.mem, self.cfg.mem_reserve, kv_in_use,
                                         self.runner.kv_bytes_per_token, self.est_seq_len)
        d = self.controller.tick(Observation(now=now, running=len(self.running),
                                             queued=len(self.waiting), mem=self.mem,
                                             kv_ceiling=self.kv_ceiling))
        self.last_decision = d
        if d.shed and len(self.running) > self.controller.limit:
            self._preempt(len(self.running) - self.controller.limit)
        self._stats = self._compute_stats()
        self.metrics.decision(d)
        self.metrics.gauges(self)
        return d

    @property
    def kv_tokens(self) -> int:
        return self.runner.padded_tokens(self.state) if self.state is not None else 0

    @property
    def est_seq_len(self) -> int:
        """Optimistic per-row KV length: prompt + half of max_tokens, over recent and queued
        requests. When it's wrong, recompute preemption catches the OOM."""
        queued = [r.num_prompt + r.params.max_tokens // 2 for r in list(self.waiting)[:64]]
        return max([*self._recent_len, *queued], default=256)

    def _trim_stats(self, now: float) -> None:
        horizon = now - self.cfg.stats_window_s
        for dq in (self.steps, self.ttfts, self.tokens):
            while dq and dq[0][0] < horizon:
                dq.popleft()

    def _drop_cancelled(self) -> None:
        with self._lock:
            dead = [r for r in self.waiting if r.cancelled]
            for r in dead:
                self.waiting.remove(r)
        for r in dead:
            self._finish(r, "cancelled")
        gone = [i for i, r in enumerate(self.running) if r.cancelled]
        if gone:
            for i in gone:
                self._finish(self.running[i], "cancelled")
            self._keep([i for i, r in enumerate(self.running) if not r.cancelled])

    def _kv_budget_tokens(self) -> int | None:
        if self.mem is None:
            return None
        kv_in_use = self.kv_tokens * self.runner.kv_bytes_per_token
        budget = self.mem.headroom - self.cfg.mem_reserve * self.mem.limit + kv_in_use
        return int(budget // self.runner.kv_bytes_per_token)

    def _admit(self) -> list[Request]:
        limit = self.controller.limit
        budget = self._kv_budget_tokens()
        cur_len = self.kv_tokens // len(self.running) if self.running else 0
        admitted: list[Request] = []
        prefill = 0
        with self._lock:
            while self.waiting and len(self.running) + len(admitted) < limit:
                req = self.waiting[0]
                n = len(req.all_ids)
                if admitted and prefill + n > self.cfg.max_prefill_tokens_per_step:
                    break
                if budget is not None:
                    rows = len(self.running) + len(admitted) + 1
                    projected = rows * (max(cur_len, n) + max(1, req.remaining // 2))
                    if projected > budget and (self.running or admitted):
                        break
                self.waiting.popleft()
                admitted.append(req)
                prefill += n
                cur_len = max(cur_len, n)
        return admitted

    def _admit_and_prefill(self) -> int:
        new = self._admit()
        if not new:
            return 0
        now = self.clock()
        for r in new:
            if r.admitted_at is None:
                r.admitted_at = now
                self._recent_len.append(r.num_prompt + r.params.max_tokens // 2)
            if r.detok is None:
                r.detok = IncrementalDetokenizer(self.tokenizer, r.params.stop)
                r.generator = make_generator(r.params.seed)
        try:
            state, logits = self.runner.prefill([r.all_ids for r in new])
        except Exception as exc:
            if not is_oom(exc):
                for r in new:
                    self._fail(r, f"prefill failed: {exc}")
                return 0
            self._on_oom()
            if not self.running and len(new) == 1:
                self._fail(new[0], "out of memory during prefill of a single request")
            else:
                with self._lock:
                    self.waiting.extendleft(reversed(new))
            return 0
        tokens = sample(logits, [r.params for r in new], [r.generator for r in new])
        keep = [i for i, (r, t) in enumerate(zip(new, tokens, strict=True)) if self._accept(r, t)]
        if len(keep) < len(new):
            state = self.runner.select(state, keep) if keep else None
            new = [new[i] for i in keep]
        if new:
            self.state = state if self.state is None else self.runner.merge(self.state, state)
            self.running.extend(new)
        return sum(len(r.all_ids) for r in new)

    def _decode(self) -> None:
        last = [r.generated[-1] for r in self.running]
        try:
            logits = self.runner.decode(self.state, last)
        except Exception as exc:
            if not is_oom(exc):
                for r in list(self.running):
                    self._fail(r, f"decode failed: {exc}")
                self.running, self.state = [], None
                return
            self._on_oom()
            if len(self.running) == 1:
                self._fail(self.running[0], "out of memory with a single running request")
                self.running, self.state = [], None
            else:
                self._preempt(1)
            return
        tokens = sample(logits, [r.params for r in self.running],
                        [r.generator for r in self.running])
        keep = [i for i, (r, t) in enumerate(zip(self.running, tokens, strict=True))
                if self._accept(r, t)]
        if len(keep) < len(self.running):
            self._keep(keep)

    def _keep(self, keep: list[int]) -> None:
        if not keep:
            self.running, self.state = [], None
            return
        self.state = self.runner.select(self.state, keep)
        self.running = [self.running[i] for i in keep]

    def _preempt(self, n: int) -> None:
        n = min(n, len(self.running))
        victims = self.running[len(self.running) - n:]
        self._keep(list(range(len(self.running) - n)))
        for r in victims:
            r.preemptions += 1
            self.preemptions += 1
            self.metrics.preempted()
        with self._lock:
            self.waiting.extendleft(reversed(victims))

    def _on_oom(self) -> None:
        self.ooms += 1
        self.metrics.oom()
        self.controller.report_oom(self.clock())

    def _accept(self, r: Request, token: int) -> bool:
        """Record one generated token. Returns False if the request is finished."""
        now = self.clock()
        r.generated.append(token)
        if r.first_token_at is None:
            r.first_token_at = now
            self.ttfts.append((now, now - r.arrival))
        r.last_token_at = now
        self.tokens.append((now, 1))
        self.metrics.token(1)
        if token in self.runner.eos_ids:
            self._finish(r, "stop")
            return False
        text = r.detok.push(token)
        if text:
            r.on_event(TokenEvent(text=text, token_id=token, index=len(r.generated) - 1))
        if r.detok.stopped:
            self._finish(r, "stop")
            return False
        if r.remaining <= 0 or len(r.all_ids) >= self.cfg.max_context:
            self._finish(r, "length")
            return False
        return True

    def _finish(self, r: Request, reason: FinishReason) -> None:
        if r.finish_reason is not None:
            return
        r.finish_reason = reason
        if reason != "cancelled" and r.detok is not None:
            tail = r.detok.flush()
            if tail:
                r.on_event(TokenEvent(text=tail, token_id=-1, index=len(r.generated) - 1))
        now = self.clock()
        n = len(r.generated)
        tpot = None
        if n > 1 and r.first_token_at is not None and r.last_token_at is not None:
            tpot = (r.last_token_at - r.first_token_at) / (n - 1)
        done = DoneEvent(
            finish_reason=reason,
            prompt_tokens=r.num_prompt,
            completion_tokens=n,
            ttft_s=None if r.first_token_at is None else r.first_token_at - r.arrival,
            tpot_s=tpot,
            e2e_s=now - r.arrival,
            queue_s=(r.admitted_at or now) - r.arrival,
        )
        self.metrics.finished(r, done)
        r.on_event(done)

    def _fail(self, r: Request, message: str) -> None:
        if r.finish_reason is not None:
            return
        r.finish_reason = "error"
        self.metrics.finished(r, DoneEvent("error", r.num_prompt, len(r.generated), None, None,
                                           self.clock() - r.arrival, 0.0))
        r.on_event(ErrorEvent(message))

    def fail_all(self, message: str) -> None:
        with self._lock:
            pending = list(self.waiting)
            self.waiting.clear()
        for r in pending + self.running:
            self._fail(r, message)
        self.running, self.state = [], None

    # --- telemetry ---------------------------------------------------------------------------

    def stats(self) -> dict:
        """Safe to call from any thread: returns the snapshot taken at the last control tick,
        with the cheap counters refreshed."""
        return {**self._stats, "running": len(self.running), "queued": len(self.waiting),
                "batch_limit": self.controller.limit}

    def _compute_stats(self) -> dict:
        tpots = sorted(s[1] for s in self.steps)
        ttfts = sorted(s[1] for s in self.ttfts)

        def pct(xs: list[float], q: float) -> float | None:
            return xs[min(len(xs) - 1, int(q * len(xs)))] * 1e3 if xs else None

        span = self.cfg.stats_window_s
        return {
            "running": len(self.running),
            "queued": len(self.waiting),
            "batch_limit": self.controller.limit,
            "kv_tokens": self.kv_tokens,
            "kv_ceiling": self.kv_ceiling,
            "tpot_p50_ms": pct(tpots, 0.5),
            "tpot_p95_ms": pct(tpots, 0.95),
            "ttft_p95_ms": pct(ttfts, 0.95),
            "tokens_per_s": sum(n for _, n in self.tokens) / span,
            "busy_ratio": self.busy_ratio,
            "preemptions": self.preemptions,
            "ooms": self.ooms,
        }
