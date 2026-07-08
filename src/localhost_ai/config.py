from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
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
    fixed_batch: int = Field(8, ge=1)
    initial_batch: int = Field(16, ge=1)
    min_batch: int = Field(1, ge=1)
    max_batch: int = Field(64, ge=1)
    slo_tpot_ms: float = Field(100.0, gt=0)
    control_interval_s: float = Field(1.0, gt=0)
    window_s: float = Field(5.0, gt=0)
    n_min: int = Field(20, ge=1)
    mem_low_wm: float = Field(0.10, ge=0, lt=1)
    mem_high_wm: float = Field(0.20, ge=0, lt=1)
    mem_reserve: float = Field(0.15, ge=0, lt=1)
    # Optional cap on the memory the probe reports as available (bytes). Lets you run the
    # memory-pressure scenario natively without a container limit.
    mem_limit_bytes: int = 0

    max_queue: int = Field(256, ge=1)
    # /v1/score requests waiting for the compute thread; one runs per scheduler iteration
    max_score_jobs: int = Field(16, ge=1)
    # HTTP request bodies over this are refused with 413 before they are parsed
    max_request_bytes: int = Field(1 << 20, ge=1024)
    max_prefill_tokens_per_step: int = Field(2048, ge=1)
    max_context: int = Field(2048, ge=2)
    default_max_tokens: int = 256
    # Prefix caching (engine/prefix.py): reuse the prefill of a prompt prefix that recent
    # requests share, e.g. a long system prompt. Off by default: rows that start from a stored
    # prefix are computed in a different order, so on low-precision backends their logits can
    # differ in the last bits from a full prefill.
    prefix_cache: bool = False
    prefix_cache_mb: int = Field(512, ge=1)
    prefix_min_tokens: int = Field(32, ge=1)
    # LoRA adapters for the startup model (MLX presets): "name=path,name2=path2", each path an
    # mlx-lm adapter directory or one .safetensors checkpoint inside one.
    adapters: str = ""

    # The server build /v1/score reports. Empty: read it from git at startup (a checkout);
    # set it where there is no .git, e.g. in the container image.
    commit: str = ""

    admin_token: str = ""
    # Host headers the server answers to (DNS-rebinding guard), comma-separated; "server" is
    # the compose service name Prometheus scrapes. IPv6 literals can't be listed: Starlette's
    # check splits the Host header on ":".
    allowed_hosts: str = "localhost,127.0.0.1,server"
    # Full origins (scheme://host:port) whose pages may open WebSockets besides this server's
    # own, comma-separated, e.g. a dev frontend on another port
    allowed_origins: str = ""
    host: str = "127.0.0.1"
    port: int = 8000


def parse_adapters(value: str) -> list[tuple[str, str]]:
    """LHAI_ADAPTERS, `name=path,name2=path2`, as (name, path) pairs."""
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, path = item.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"adapter entry {item!r} should look like name=path")
        out.append((name.strip(), path.strip()))
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        raise ValueError(f"adapter names must be unique: {names}")
    return out


def code_commit(s: Settings) -> tuple[str | None, bool]:
    """(commit, dirty) of the running code: LHAI_COMMIT if set, else the git checkout's HEAD
    and whether tracked files differ from it. (None, False) outside a checkout."""
    if s.commit:
        return s.commit, False
    import subprocess

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                 timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    if not head:
        return None, False
    return head, bool(git("status", "--porcelain", "--untracked-files=no"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
