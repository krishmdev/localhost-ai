from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass(frozen=True)
class DeviceConfig:
    device: torch.device
    dtype: torch.dtype
    threads: int

    @property
    def kind(self) -> str:
        return self.device.type

    @property
    def dtype_name(self) -> str:
        return {v: k for k, v in _DTYPES.items()}[self.dtype]


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("LHAI_DEVICE=cuda but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if requested == "mps" or (requested == "auto" and torch.backends.mps.is_available()):
        if not torch.backends.mps.is_available():
            raise RuntimeError("LHAI_DEVICE=mps but the MPS backend is not available")
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, requested: str = "auto") -> torch.dtype:
    if requested != "auto":
        return _DTYPES[requested]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def cgroup_cpu_limit(root: Path = Path("/sys/fs/cgroup")) -> float | None:
    """CPU quota from cgroup v2 `cpu.max`, e.g. "200000 100000" -> 2.0. None when unlimited."""
    try:
        quota, period = (root / "cpu.max").read_text().split()[:2]
    except (OSError, ValueError):
        return None
    if quota == "max":
        return None
    return int(quota) / int(period)


def default_threads() -> int:
    limit = cgroup_cpu_limit()
    count = os.cpu_count() or 1
    if limit is not None:
        count = min(count, max(1, math.floor(limit)))
    return count


def configure(requested_device: str = "auto", requested_dtype: str = "auto",
              threads: int = 0) -> DeviceConfig:
    device = pick_device(requested_device)
    dtype = pick_dtype(device, requested_dtype)
    n = threads or default_threads()
    torch.set_num_threads(n)
    return DeviceConfig(device=device, dtype=dtype, threads=n)
