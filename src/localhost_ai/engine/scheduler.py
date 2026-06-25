"""Iteration-level continuous batching.

One `step()` is one scheduler iteration:
1. drop cancelled requests; run the controller if a control interval has passed;
2. admit waiting requests while the batch is under the controller's limit L, the prefill token
   budget and the KV budget;
3. prefill the admitted requests together and sample their first token;
4. merge them into the running batch (left-pad + concat) and run one decode step for every row;
5. filter out rows that finished.

The controller is fed decode-step time only, measured with the batch size at decode time.
Prefill time depends on prompt lengths and how many requests arrived, not on batch size, and
mixing it in drags L toward a value set by output length rather than by capacity. Users do see
prefill stalls in their inter-token latency; `max_prefill_tokens_per_step` bounds them.

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
from .controller import Controller, Decision, Observation, Sample, kv_ceiling, percentile
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
    def step(self, iter_s: float, prefill_s: float, decode_s: float, batch: int,
             prefill_tokens: int) -> None: ...
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
        self.busy_s = 0.0
        self.busy_ratio = 0.0
        self._busy_at_tick = 0.0
        self.steps: deque[tuple[float, float, int]] = deque()  # (t, decode_s, batch)
        self.iters: deque[tuple[float, float]] = deque()  # (t, iteration_s)
        self._kv_at_tick = 0
        # After an OOM, don't admit anything new until a running row finishes; otherwise the
        # preempted row goes straight back in and hits the same OOM (thrash).
        self._hold_admission = False
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
        prefill_tokens = self._admit_and_prefill()
        t_mid = self.clock()
        decode_s = 0.0
        batch = len(self.running)
        if self.running:
            epoch = self.controller.epoch
            ok = self._decode()
            t_end = self.clock()
            decode_s = t_end - t_mid
            if ok:
                self.controller.observe(Sample(t=t_end, tpot_s=decode_s, epoch=epoch,
                                               batch=batch))
                self.steps.append((t_end, decode_s, batch))
        t_end = self.clock()
        dur = t_end - t_start
        self.busy_s += dur
        self.iters.append((t_end, dur))
        self.metrics.step(dur, t_mid - t_start, decode_s, batch, prefill_tokens)
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
        self._kv_at_tick = kv_in_use
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
            self._release()
        self._stats = self._compute_stats()
        self.metrics.decision(d)
        self.metrics.gauges(self)
        return d

    @property
    def kv_tokens(self) -> int:
        return self.runner.padded_tokens(self.state) if self.state is not None else 0

    @property
    def est_seq_len(self) -> int:
        """Per-row KV length the ceiling divides by, estimated the same way admission projects
        a join: the padded length of the batch plus the requests admission would take next,
        plus half of those rows' remaining max_tokens on average. The batch is left-padded, so every
        row costs the padded length. When the estimate is wrong, recompute preemption catches
        the OOM."""
        cur = self.kv_tokens // len(self.running) if self.running else 0
        room = max(1, self.controller.limit - len(self.running))
        with self._lock:
            nxt = list(self.waiting)[:room]  # the requests admission would take next
        width = max([cur, *(len(r.all_ids) for r in nxt)])
        rows = list(self.running) + nxt
        if not rows:
            return 256
        growth = sum(max(1, r.remaining // 2) for r in rows) / len(rows)
        return max(1, int(width + growth))

    def _trim_stats(self, now: float) -> None:
        horizon = now - self.cfg.stats_window_s
        for dq in (self.steps, self.iters, self.ttfts, self.tokens):
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
        """KV tokens the batch may hold, from the reading taken at the last tick: that tick's
        headroom minus the reserve, plus the KV that was already held then."""
        if self.mem is None:
            return None
        budget = self.mem.headroom - self.cfg.mem_reserve * self.mem.limit + self._kv_at_tick
        return int(budget // self.runner.kv_bytes_per_token)

    def _admit(self) -> list[Request]:
        if self._hold_admission:
            if self.running:
                return []
            self._hold_admission = False
        limit = self.controller.limit
        budget = self._kv_budget_tokens()
        cur_len = self.kv_tokens // len(self.running) if self.running else 0
        admitted: list[Request] = []
        longest = 0
        with self._lock:
            while self.waiting and len(self.running) + len(admitted) < limit:
                req = self.waiting[0]
                n = len(req.all_ids)
                # the prefill batch is left-padded, so its cost is rows x longest prompt
                padded = (len(admitted) + 1) * max(longest, n)
                if admitted and padded > self.cfg.max_prefill_tokens_per_step:
                    break
                if budget is not None:
                    rows = len(self.running) + len(admitted) + 1
                    projected = rows * (max(cur_len, n) + max(1, req.remaining // 2))
                    if projected > budget and (self.running or admitted):
                        break
                self.waiting.popleft()
                admitted.append(req)
                longest = max(longest, n)
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
                self._requeue(new)
            return 0
        padded = len(new) * max(len(r.all_ids) for r in new)
        tokens = sample(logits, [r.params for r in new], [r.generator for r in new])
        keep = [i for i, (r, t) in enumerate(zip(new, tokens, strict=True)) if self._accept(r, t)]
        try:
            if len(keep) < len(new):
                state = self.runner.select(state, keep) if keep else None
                new = [new[i] for i in keep]
            if new:
                self.state = state if self.state is None else self.runner.merge(self.state, state)
                self.running.extend(new)
        except Exception as exc:
            if not is_oom(exc):
                raise
            # merge consumes both caches, so the whole batch is recomputed
            self._on_oom()
            self._reset_batch([*self.running, *new])
        return padded

    def _decode(self) -> bool:
        """One decode step for every running row. False if it didn't produce tokens."""
        last = [r.generated[-1] for r in self.running]
        try:
            logits = self.runner.decode(self.state, last)
        except Exception as exc:
            if not is_oom(exc):
                for r in list(self.running):
                    self._fail(r, f"decode failed: {exc}")
                self.running, self.state = [], None
                return False
            self._on_oom()
            if len(self.running) == 1:
                self._fail(self.running[0], "out of memory with a single running request")
                self.running, self.state = [], None
            else:
                self._preempt(1)
            self._release()
            return False
        tokens = sample(logits, [r.params for r in self.running],
                        [r.generator for r in self.running])
        keep = [i for i, (r, t) in enumerate(zip(self.running, tokens, strict=True))
                if self._accept(r, t)]
        if len(keep) < len(self.running):
            self._keep(keep)
        return True

    def _keep(self, keep: list[int]) -> None:
        if len(keep) < len(self.running):
            self._hold_admission = False  # a row left, so there's room again
        if not keep:
            self.running, self.state = [], None
            return
        try:
            self.state = self.runner.select(self.state, keep)
        except Exception as exc:
            if not is_oom(exc):
                raise
            self._on_oom()
            self._reset_batch([self.running[i] for i in keep])
            return
        self.running = [self.running[i] for i in keep]

    def _requeue(self, reqs: list[Request]) -> None:
        with self._lock:
            self.waiting.extendleft(reversed(reqs))

    def _reset_batch(self, reqs: list[Request]) -> None:
        """Drop the whole running batch's KV and requeue its rows for recompute."""
        self.state = None
        self.running = []
        alive = [r for r in reqs if r.finish_reason is None]
        for r in alive:
            r.preemptions += 1
            self.preemptions += 1
            self.metrics.preempted()
        self._requeue(alive)
        self._release()

    def _release(self) -> None:
        release = getattr(self.runner, "release", None)
        if release is not None:
            release()

    def _preempt(self, n: int) -> None:
        n = min(n, len(self.running))
        victims = self.running[len(self.running) - n:]
        keep = list(range(len(self.running) - n))
        if not keep:
            self.running, self.state = [], None
        else:
            try:
                self.state = self.runner.select(self.state, keep)
                self.running = [self.running[i] for i in keep]
            except Exception as exc:
                if not is_oom(exc):
                    raise
                self._reset_batch(list(self.running))
                return
        for r in victims:
            r.preemptions += 1
            self.preemptions += 1
            self.metrics.preempted()
        self._requeue(victims)

    def _on_oom(self) -> None:
        self.ooms += 1
        self.metrics.oom()
        self._hold_admission = True
        d = self.controller.report_oom(self.clock())
        if d is not None:
            self.last_decision = d
            self.metrics.decision(d)

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
        """Latency names used everywhere (telemetry, metrics, docs):
        decode_step_*: wall time of decode steps in the last 5 s (what the controller measures);
        controller_p95_ms: p95 over the fresh samples behind the last controller decision;
        iteration_p95_ms: whole scheduler iterations, prefill included (what streams feel);
        per-request TPOT lives in the request's own timings and the lhai_tpot_seconds histogram."""
        decode = [s[1] for s in self.steps]
        iters = [s[1] for s in self.iters]
        ttfts = [s[1] for s in self.ttfts]

        def ms(xs: list[float], q: float) -> float | None:
            return round(percentile(xs, q) * 1e3, 3) if xs else None

        d = self.last_decision
        span = self.cfg.stats_window_s
        return {
            "running": len(self.running),
            "queued": len(self.waiting),
            "batch_limit": self.controller.limit,
            "kv_tokens": self.kv_tokens,
            "kv_ceiling": self.kv_ceiling,
            "decode_step_p50_ms": ms(decode, 0.5),
            "decode_step_p95_ms": ms(decode, 0.95),
            "controller_p95_ms": None if d is None or d.p95_ms is None else round(d.p95_ms, 3),
            "iteration_p95_ms": ms(iters, 0.95),
            "ttft_p95_ms": ms(ttfts, 0.95),
            "tokens_per_s": sum(n for _, n in self.tokens) / span,
            "busy_ratio": self.busy_ratio,
            "preemptions": self.preemptions,
            "ooms": self.ooms,
        }
