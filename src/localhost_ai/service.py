"""Wires settings, device, model, probe, controller, scheduler and engine together."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .engine.controller import AIMDConfig, AIMDController, Controller, FixedController
from .engine.engine import AsyncEngine
from .engine.scheduler import Scheduler, SchedulerConfig
from .memory import MemoryProbe
from .metrics import EngineMetrics

log = logging.getLogger("localhost_ai.service")


@dataclass
class ModelParts:
    name: str
    runner: Any
    tokenizer: Any
    encode_chat: Callable[[list[dict[str, str]]], list[int]]
    info: dict[str, str]
    default_max_tokens: int = 256


def aimd_config(s: Settings) -> AIMDConfig:
    return AIMDConfig(
        slo_tpot_s=s.slo_tpot_ms / 1e3, min_batch=s.min_batch, max_batch=s.max_batch,
        initial=s.initial_batch, interval_s=s.control_interval_s, window_s=s.window_s,
        n_min=s.n_min, mem_low_wm=s.mem_low_wm, mem_high_wm=s.mem_high_wm,
    )


def make_controller(s: Settings, cfg: AIMDConfig) -> Controller:
    if s.controller == "fixed":
        return FixedController(s.fixed_batch)
    return AIMDController(AIMDConfig(**cfg.__dict__))


@dataclass
class Service:
    settings: Settings
    parts: ModelParts
    probe: MemoryProbe | None
    metrics: EngineMetrics
    model_names: list[str]
    loader: Callable[[str], ModelParts] | None = None
    engine: AsyncEngine = field(init=False)
    swapping: bool = False
    _swap_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._aimd = aimd_config(self.settings)
        self.engine = self._build_engine(make_controller(self.settings, self._aimd))

    def _build_engine(self, controller: Controller) -> AsyncEngine:
        s = self.settings
        sched = Scheduler(
            self.parts.runner, self.parts.tokenizer, controller, probe=self.probe,
            cfg=SchedulerConfig(max_queue=s.max_queue,
                                max_prefill_tokens_per_step=s.max_prefill_tokens_per_step,
                                max_context=s.max_context,
                                control_interval_s=s.control_interval_s,
                                mem_reserve=s.mem_reserve),
            metrics=self.metrics,
        )
        self.metrics.model.info(self.parts.info)
        self.metrics.slo.set(self._aimd.slo_tpot_s)
        return AsyncEngine(sched, info=dict(self.parts.info), aimd_cfg=self._aimd)

    @property
    def model_name(self) -> str:
        return self.parts.name if self.parts is not None else "none"

    def start(self) -> None:
        self.engine.start()

    def stop(self) -> None:
        self.engine.stop()

    async def swap_model(self, name: str, drain_timeout_s: float = 120.0) -> None:
        """Drain, unload, load, resume. New requests get 503 while this runs. If the new model
        fails to load, the old one is loaded back and serving resumes with it. Swaps are
        serialized."""
        if self.loader is None:
            raise RuntimeError("model hot-swap is not available in this configuration")
        async with self._swap_lock:
            await self._swap(name, drain_timeout_s)

    def _restart(self, controller: Controller) -> None:
        self.engine = self._build_engine(controller)
        self.engine.start()

    async def _swap(self, name: str, drain_timeout_s: float) -> None:
        self.swapping = True
        try:
            waited = 0.0
            while self.engine.scheduler.has_work() and waited < drain_timeout_s:
                await asyncio.sleep(0.1)
                waited += 0.1
            controller = self.engine.controller
            self.engine.stop()
            self.engine.scheduler.fail_all("model is being replaced")
            # None after an earlier swap failed twice; then there is nothing to fall back to.
            old_name = self.parts.name if self.parts is not None else None
            self.parts = None  # type: ignore[assignment]
            _free_device_cache()
            try:
                self.parts = await asyncio.to_thread(self.loader, name)
            except Exception as first:
                if old_name is None:
                    raise
                log.exception("loading %s failed; reloading %s", name, old_name)
                try:
                    self.parts = await asyncio.to_thread(self.loader, old_name)
                except Exception as second:
                    # Nothing to serve with; keep the engine stopped (readyz reports it) and
                    # surface the original failure with the reload failure attached.
                    raise RuntimeError(f"loading {name!r} failed ({first}) and reloading "
                                       f"{old_name!r} also failed ({second})") from first
                self._restart(controller)
                raise
            self._restart(controller)
            log.info("now serving %s", name)
        finally:
            self.swapping = False


def _free_device_cache() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def build_from_settings(s: Settings) -> Service:
    from . import device
    from .memory import probe_for
    from .models.loader import load
    from .models.registry import Registry

    registry = Registry(s.models_file)
    dev = device.configure(s.device, s.dtype, s.threads)

    def loader(name: str) -> ModelParts:
        spec = registry.get(name)
        m = load(spec, dev, s.models_dir, s.quantization, s.dtype)
        return ModelParts(
            name=spec.name, runner=m.runner, tokenizer=m.tokenizer, encode_chat=m.encode_chat,
            default_max_tokens=spec.max_new_tokens,
            info={"model": spec.name, "repo": spec.repo, "revision": spec.revision,
                  "device": dev.kind, "dtype": str(m.dtype).removeprefix("torch."),
                  "quant": s.quantization, "threads": str(dev.threads),
                  "kv_bytes_per_token": str(m.runner.kv_bytes_per_token)},
        )

    parts = loader(s.model)
    return Service(settings=s, parts=parts, probe=probe_for(dev.kind, s.mem_limit_bytes),
                   metrics=EngineMetrics(), model_names=list(registry.specs), loader=loader)
