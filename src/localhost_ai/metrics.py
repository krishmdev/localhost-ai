"""Prometheus metrics, all prefixed `lhai_`. Each engine gets its own registry so tests can build
several apps in one process."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info

if TYPE_CHECKING:
    from .engine.controller import Decision
    from .engine.request import DoneEvent, Request
    from .engine.scheduler import Scheduler

LATENCY_BUCKETS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0,
                   2.0, 5.0)
TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0)
E2E_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120)
BATCH_BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)


class EngineMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        r = self.registry = registry or CollectorRegistry()
        p = "lhai_"
        self.ttft = Histogram(p + "ttft_seconds", "Time to first token", registry=r,
                              buckets=TTFT_BUCKETS)
        self.tpot = Histogram(p + "tpot_seconds", "Mean time per output token, per request",
                              registry=r, buckets=LATENCY_BUCKETS)
        self.step_s = Histogram(p + "step_seconds", "Wall time of one scheduler iteration",
                                registry=r, buckets=LATENCY_BUCKETS)
        self.e2e = Histogram(p + "e2e_latency_seconds", "Request latency, arrival to done",
                             registry=r, buckets=E2E_BUCKETS)
        self.batch = Histogram(p + "batch_size", "Running rows per iteration", registry=r,
                               buckets=BATCH_BUCKETS)
        self.prompt_tokens = Counter(p + "prompt_tokens", "Prompt tokens processed", registry=r)
        self.gen_tokens = Counter(p + "generated_tokens", "Tokens generated", registry=r)
        self.requests = Counter(p + "requests", "Finished requests", ["status"], registry=r)
        self.decisions = Counter(p + "controller_decisions", "Controller decisions", ["action"],
                                 registry=r)
        self.preemptions = Counter(p + "preemptions", "Rows preempted (recompute)", registry=r)
        self.ooms = Counter(p + "oom_events", "Out-of-memory errors caught", registry=r)
        self.queue = Gauge(p + "queue_depth", "Requests waiting", registry=r)
        self.running = Gauge(p + "running_requests", "Rows in the running batch", registry=r)
        self.limit = Gauge(p + "batch_limit", "Controller batch limit L", registry=r)
        self.slo = Gauge(p + "slo_tpot_seconds", "TPOT SLO the controller targets", registry=r)
        self.p95 = Gauge(p + "tpot_p95_seconds", "p95 step latency over the last 5 s", registry=r)
        self.mem_used = Gauge(p + "device_memory_used_bytes", "Device memory in use", registry=r)
        self.mem_limit = Gauge(p + "device_memory_limit_bytes", "Device memory limit",
                               registry=r)
        self.headroom = Gauge(p + "memory_headroom_ratio", "Headroom / limit", registry=r)
        self.kv_tokens = Gauge(p + "kv_cache_tokens", "Padded tokens held in the KV cache",
                               registry=r)
        self.kv_ceiling = Gauge(p + "kv_ceiling_rows", "Rows that fit in memory (estimate)",
                                registry=r)
        self.busy = Gauge(p + "engine_busy_ratio", "Fraction of wall time spent computing",
                          registry=r)
        self.gpu_util = Gauge(p + "gpu_utilization_ratio", "GPU utilization from NVML",
                              registry=r)
        self.model = Info(p + "model", "Loaded model", registry=r)
        for action in ("oom_backoff", "mem_decrease", "slo_decrease", "increase", "hold",
                       "clamp"):
            self.decisions.labels(action)
        for status in ("stop", "length", "cancelled", "error", "rejected"):
            self.requests.labels(status)

    # scheduler hooks (compute thread)
    def step(self, duration_s: float, batch: int, prefill_tokens: int) -> None:
        self.step_s.observe(duration_s)
        if batch:
            self.batch.observe(batch)
        if prefill_tokens:
            self.prompt_tokens.inc(prefill_tokens)

    def token(self, n: int) -> None:
        self.gen_tokens.inc(n)

    def finished(self, req: Request, done: DoneEvent) -> None:
        self.requests.labels(done.finish_reason).inc()
        self.e2e.observe(done.e2e_s)
        if done.ttft_s is not None:
            self.ttft.observe(done.ttft_s)
        if done.tpot_s is not None:
            self.tpot.observe(done.tpot_s)

    def rejected(self) -> None:
        self.requests.labels("rejected").inc()

    def decision(self, d: Decision) -> None:
        self.decisions.labels(d.action).inc()

    def preempted(self) -> None:
        self.preemptions.inc()

    def oom(self) -> None:
        self.ooms.inc()

    def gauges(self, s: Scheduler) -> None:
        st = s.stats()
        self.queue.set(st["queued"])
        self.running.set(st["running"])
        self.limit.set(st["batch_limit"])
        self.kv_tokens.set(st["kv_tokens"])
        self.busy.set(st["busy_ratio"])
        if st["tpot_p95_ms"] is not None:
            self.p95.set(st["tpot_p95_ms"] / 1e3)
        if st["kv_ceiling"] is not None:
            self.kv_ceiling.set(st["kv_ceiling"])
        slo = getattr(getattr(s.controller, "cfg", None), "slo_tpot_s", None)
        if slo is not None:
            self.slo.set(slo)
        if s.mem is not None:
            self.mem_used.set(s.mem.used)
            self.mem_limit.set(s.mem.limit)
            self.headroom.set(s.mem.headroom_frac)
            if s.mem.util_pct is not None:
                self.gpu_util.set(s.mem.util_pct / 100)
