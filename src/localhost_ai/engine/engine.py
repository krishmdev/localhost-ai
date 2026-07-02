"""AsyncEngine: the scheduler runs on its own compute thread; the API runs on asyncio.

Events for a request are pushed back to the event loop with `call_soon_threadsafe` into a
per-request asyncio.Queue, so the compute thread never blocks on a slow client."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .controller import AIMDConfig, AIMDController, Controller, FixedController
from .request import DoneEvent, ErrorEvent, Event, Request, SamplingParams
from .scheduler import Scheduler

log = logging.getLogger("localhost_ai.engine")


@dataclass
class Handle:
    request: Request
    queue_position: int
    _queue: asyncio.Queue = field(repr=False)
    _engine: AsyncEngine = field(repr=False)

    async def events(self) -> AsyncIterator[Event]:
        try:
            while True:
                ev = await self._queue.get()
                yield ev
                if isinstance(ev, (DoneEvent, ErrorEvent)):
                    return
        finally:
            if self.request.finish_reason is None:
                self.cancel()

    def cancel(self) -> None:
        self.request.cancel()
        self._engine.wake()


class AsyncEngine:
    def __init__(self, scheduler: Scheduler, info: dict[str, Any] | None = None,
                 aimd_cfg: AIMDConfig | None = None) -> None:
        self.scheduler = scheduler
        self.info = info or {}
        self._aimd_cfg = aimd_cfg or AIMDConfig()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    # --- lifecycle ---------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="lhai-compute", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> bool:
        """Ask the compute thread to stop and wait up to `timeout` seconds (None: until it
        has). Returns True once it has exited. If it is still inside a step, it stays
        referenced, so `alive` keeps reporting it and `start` won't launch a second one."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.warning("compute thread still running after %.1fs", timeout or 0.0)
                return False
            self._thread = None
        return True

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def wake(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        interval = self.scheduler.cfg.control_interval_s
        while not self._stop.is_set():
            try:
                worked = self.scheduler.step()
            except Exception as exc:  # keep serving; fail what was in flight
                log.exception("scheduler step failed")
                self.error = repr(exc)
                self.scheduler.fail_all(f"engine error: {exc}")
                worked = False
            if not worked:
                self._wake.wait(timeout=interval / 4)
                self._wake.clear()

    # --- requests ----------------------------------------------------------------------------

    def submit(self, prompt_ids: list[int], params: SamplingParams) -> Handle:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        req: Request

        def push(ev: Event) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, ev)
            except RuntimeError:  # the event loop is gone; nobody is listening
                req.cancel()

        req = Request(prompt_ids=list(prompt_ids), params=params, on_event=push)
        pos = self.scheduler.add(req)  # raises QueueFull
        self._wake.set()
        return Handle(req, pos, queue, self)

    # --- control -----------------------------------------------------------------------------

    @property
    def controller(self) -> Controller:
        return self.scheduler._next_controller or self.scheduler.controller

    def set_mode(self, mode: str, batch: int | None = None) -> Controller:
        ctl: Controller
        if mode == "fixed":
            ctl = FixedController(batch or self.controller.limit)
        elif mode == "aimd":
            cfg = AIMDConfig(**{**self._aimd_cfg.__dict__})
            if batch:
                cfg.initial = batch
            ctl = AIMDController(cfg)
        else:
            raise ValueError(f"unknown controller mode {mode!r}")
        self.scheduler.swap_controller(ctl)
        self._wake.set()
        return ctl

    def set_slo(self, tpot_ms: float) -> None:
        self._aimd_cfg.slo_tpot_s = tpot_ms / 1e3
        ctl = self.controller
        if isinstance(ctl, AIMDController):
            ctl.set_slo(tpot_ms / 1e3)

    @property
    def slo_tpot_ms(self) -> float:
        return self._aimd_cfg.slo_tpot_s * 1e3

    def telemetry(self) -> dict[str, Any]:
        s = self.scheduler
        mem = s.mem
        d = s.last_decision
        return {
            **self.info,
            "mode": self.controller.mode,
            "slo_tpot_ms": self.slo_tpot_ms,
            "memory": None if mem is None else {
                "used_bytes": mem.used,
                "limit_bytes": mem.limit,
                "headroom_bytes": mem.headroom,
                "headroom_frac": round(mem.headroom_frac, 4),
                "source": mem.source,
            },
            "gpu_util_pct": None if mem is None else mem.util_pct,
            **s.stats(),
            "last_decision": None if d is None else {
                "action": d.action, "reason": d.reason, "limit": d.limit,
                "limit_before": d.limit_before, "epoch": d.epoch,
            },
        }
