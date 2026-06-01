"""Device memory probes. Every probe answers the same question: how much memory is in use, how
much is the limit, and how much is left before we get OOM-killed or hit an allocator error."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import psutil


@dataclass(frozen=True)
class MemSnapshot:
    used: int
    limit: int
    headroom: int
    util_pct: float | None = None  # GPU compute utilization, only where NVML exists
    source: str = ""

    @property
    def headroom_frac(self) -> float:
        return self.headroom / self.limit if self.limit > 0 else 0.0


class MemoryProbe(Protocol):
    name: str

    def snapshot(self) -> MemSnapshot: ...


def _snap(used: int, limit: int, source: str, util: float | None = None,
          headroom: int | None = None) -> MemSnapshot:
    hr = limit - used if headroom is None else headroom
    return MemSnapshot(used=used, limit=limit, headroom=max(0, hr), util_pct=util, source=source)


class CudaProbe:
    name = "cuda"

    def __init__(self, index: int = 0) -> None:
        import torch

        self._torch = torch
        self._index = index
        self._nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:  # noqa: BLE001 - NVML is optional
            self._nvml = None

    def snapshot(self) -> MemSnapshot:
        free, total = self._torch.cuda.mem_get_info(self._index)
        util = None
        if self._nvml is not None:
            util = float(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
        return _snap(total - free, total, "cuda.mem_get_info", util)


class MpsProbe:
    """Apple silicon. The Metal limit is `recommended_max_memory`, but memory is unified with the
    CPU, so headroom is also capped by what the OS says is available."""

    name = "mps"

    def __init__(self) -> None:
        import torch

        self._mps = torch.mps

    def snapshot(self) -> MemSnapshot:
        limit = int(self._mps.recommended_max_memory())
        used = int(self._mps.driver_allocated_memory())
        headroom = min(limit - used, psutil.virtual_memory().available)
        return _snap(used, limit, "mps.recommended_max_memory", headroom=headroom)


class CpuProbe:
    """cgroup v2 limits when running in a container, psutil otherwise."""

    name = "cpu"

    def __init__(self, cgroup_root: Path = Path("/sys/fs/cgroup")) -> None:
        self._root = cgroup_root

    def _cgroup(self) -> tuple[int, int] | None:
        try:
            raw_max = (self._root / "memory.max").read_text().strip()
            current = int((self._root / "memory.current").read_text())
        except (OSError, ValueError):
            return None
        if raw_max == "max":
            return None
        inactive = 0
        try:
            for line in (self._root / "memory.stat").read_text().splitlines():
                key, _, val = line.partition(" ")
                if key == "inactive_file":
                    inactive = int(val)
                    break
        except (OSError, ValueError):
            pass
        # Page cache that the kernel can drop doesn't count against us.
        return max(0, current - inactive), int(raw_max)

    def snapshot(self) -> MemSnapshot:
        cg = self._cgroup()
        if cg is not None:
            used, limit = cg
            return _snap(used, limit, "cgroup.memory")
        vm = psutil.virtual_memory()
        return _snap(vm.total - vm.available, vm.total, "psutil.virtual_memory")


class BudgetProbe:
    """Pretend the server may only use `budget` bytes: used = this process's RSS plus whatever
    the wrapped device probe reports as device-allocated (MPS/CUDA)."""

    def __init__(self, inner: MemoryProbe, budget: int) -> None:
        self.name = f"{inner.name}+budget"
        self._inner = inner
        self._budget = budget
        self._proc = psutil.Process()

    def snapshot(self) -> MemSnapshot:
        inner = self._inner.snapshot()
        used = self._proc.memory_info().rss
        if not isinstance(self._inner, CpuProbe):
            used += inner.used
        return _snap(used, self._budget, f"budget({inner.source})", inner.util_pct)


class FakeProbe:
    name = "fake"

    def __init__(self, limit: int = 1 << 30, used: int = 0) -> None:
        self.limit = limit
        self.used = used

    def snapshot(self) -> MemSnapshot:
        return _snap(self.used, self.limit, "fake")


def probe_for(device_kind: str, budget: int = 0) -> MemoryProbe:
    inner: MemoryProbe
    if device_kind == "cuda":
        inner = CudaProbe()
    elif device_kind == "mps":
        inner = MpsProbe()
    else:
        inner = CpuProbe()
    return BudgetProbe(inner, budget) if budget > 0 else inner
