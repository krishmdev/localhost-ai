from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

FinishReason = Literal["stop", "length", "cancelled", "error"]

_ids = itertools.count()


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 256
    stop: tuple[str, ...] = ()
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


@dataclass(frozen=True)
class TokenEvent:
    text: str
    token_id: int
    index: int


@dataclass(frozen=True)
class DoneEvent:
    finish_reason: FinishReason
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float | None
    tpot_s: float | None
    e2e_s: float
    queue_s: float


@dataclass(frozen=True)
class ErrorEvent:
    message: str
    code: str = "engine_error"


Event = TokenEvent | DoneEvent | ErrorEvent


@dataclass(eq=False)
class Request:
    prompt_ids: list[int]
    params: SamplingParams
    on_event: Callable[[Event], None] = lambda _e: None
    rid: str = field(default_factory=lambda: f"req-{next(_ids)}")
    arrival: float = field(default_factory=time.perf_counter)

    generated: list[int] = field(default_factory=list)
    cancelled: bool = False
    finish_reason: FinishReason | None = None
    admitted_at: float | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    preemptions: int = 0
    # Set by the scheduler; opaque to everyone else.
    detok: object | None = None
    generator: object | None = None
    # response_format constraint (engine/constrain.py), built at submit time; advanced by
    # sampling, so it stays in step with `generated` across preemption and recompute.
    constraint: object | None = None
    # LoRA adapter name (engine/lora.py); None runs the base model.
    adapter: str | None = None

    @property
    def num_prompt(self) -> int:
        return len(self.prompt_ids)

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.generated

    @property
    def remaining(self) -> int:
        return self.params.max_tokens - len(self.generated)

    def cancel(self) -> None:
        # Read by the compute thread at the next iteration; a plain bool write is atomic enough.
        self.cancelled = True
