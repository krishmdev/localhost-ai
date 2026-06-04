from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LHAI_", env_file=".env", extra="ignore")

    model: str = "smollm2-135m"
    models_file: Path = REPO_ROOT / "models.yaml"
    models_dir: Path = REPO_ROOT / ".models"
    models_lock: Path = REPO_ROOT / "models.lock"
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    dtype: Literal["auto", "fp32", "fp16", "bf16"] = "auto"
    quantization: Literal["none", "bnb8", "bnb4"] = "none"
    threads: int = 0  # 0 = derive from cgroup quota / cpu count

    controller: Literal["aimd", "fixed"] = "aimd"
    fixed_batch: int = 8
    initial_batch: int = 16
    min_batch: int = 1
    max_batch: int = 64
    slo_tpot_ms: float = 100.0
    slo_ttft_ms: float = 2000.0
    control_interval_s: float = 1.0
    window_s: float = 5.0
    n_min: int = 20
    mem_low_wm: float = 0.10
    mem_high_wm: float = 0.20
    mem_reserve: float = 0.15
    # Optional cap on the memory the probe reports as available (bytes). Lets you run the
    # memory-pressure scenario natively without a container limit.
    mem_limit_bytes: int = 0

    max_queue: int = 256
    max_prefill_tokens_per_step: int = 2048
    max_context: int = 2048
    default_max_tokens: int = 256

    admin_token: str = ""
    host: str = "127.0.0.1"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
