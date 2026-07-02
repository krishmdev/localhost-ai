"""A deterministic stand-in for the model. The next token depends only on the row's own token
history, so any mix of batching, joining, filtering or preemption must produce exactly the
same output as running each request alone."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

VOCAB = 40
EOS = 39
LETTERS = "abcdefghijklmnopqrstuvwxyz0123456789 .!"


def next_token(history: list[int], eos_after: int | None = None) -> int:
    if eos_after is not None and len(history) >= eos_after:
        return EOS
    h = 0
    for t in history[-8:]:
        h = (h * 31 + t + 7) % 1_000_003
    return (h + len(history)) % (VOCAB - 1)  # never EOS on its own


class FakeTokenizer:
    eos_token_id = EOS

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(LETTERS[i] for i in ids if not (skip_special_tokens and i == EOS))

    def encode(self, text: str) -> list[int]:
        return [LETTERS.index(c) for c in text if c in LETTERS and c != "!"]


class VirtualClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@dataclass
class FakeState:
    rows: list[list[int]]
    width: int  # padded length, like a real left-padded cache


@dataclass
class FakeRunner:
    kv_bytes_per_token: int = 1000
    eos_ids: frozenset[int] = frozenset({EOS})
    t0: float = 0.0  # seconds per decode step
    k: float = 0.0  # extra seconds per row
    oom_above_tokens: int | None = None  # padded tokens that trigger an OOM
    eos_after: int | None = None
    # With a VirtualClock, costs advance simulated time instead of sleeping.
    clock: VirtualClock | None = None
    prefill_s: float = 0.0  # fixed cost of a prefill call
    prefill_per_token: float = 0.0  # plus this per padded prompt token
    merge_ooms: int = 0  # raise OOM from this many merge calls
    calls: list[tuple[str, int]] = field(default_factory=list)

    def _spend(self, seconds: float) -> None:
        if self.clock is not None:
            self.clock.t += seconds
        elif seconds:
            time.sleep(seconds)

    def _logits(self, rows: list[list[int]]) -> torch.Tensor:
        out = torch.full((len(rows), VOCAB), -10.0)
        for i, r in enumerate(rows):
            t = next_token(r, self.eos_after)
            out[i] = -torch.abs(torch.arange(VOCAB) - t).float()
            out[i, t] = 5.0
        return out

    def _check_oom(self, rows: int, width: int) -> None:
        if self.oom_above_tokens is not None and rows * width > self.oom_above_tokens:
            raise torch.OutOfMemoryError("fake: out of memory")

    def prefill(self, seqs: list[list[int]]):
        width = max(len(s) for s in seqs)
        self._check_oom(len(seqs), width)
        self._spend(self.prefill_s + self.prefill_per_token * width * len(seqs))
        self.calls.append(("prefill", len(seqs)))
        return FakeState([list(s) for s in seqs], width), self._logits(seqs)

    def decode(self, state: FakeState, tokens: list[int]):
        self._check_oom(len(state.rows), state.width + 1)
        self._spend(self.t0 + self.k * len(state.rows))
        self.calls.append(("decode", len(state.rows)))
        for r, t in zip(state.rows, tokens, strict=True):
            r.append(t)
        state.width += 1
        return self._logits(state.rows)

    def merge(self, a: FakeState, b: FakeState) -> FakeState:
        if self.merge_ooms > 0:
            self.merge_ooms -= 1
            raise torch.OutOfMemoryError("fake: out of memory in merge")
        return FakeState(a.rows + b.rows, max(a.width, b.width))

    def select(self, state: FakeState, keep: list[int]) -> FakeState:
        rows = [state.rows[i] for i in keep]
        return FakeState(rows, max(len(r) for r in rows))

    def padded_tokens(self, state: FakeState) -> int:
        return len(state.rows) * state.width


def reference(prompt: list[int], max_tokens: int, eos_after: int | None = None) -> list[int]:
    hist = list(prompt)
    out = []
    for _ in range(max_tokens):
        t = next_token(hist, eos_after)
        out.append(t)
        hist.append(t)
        if t == EOS:
            break
    return out


def fake_service(runner: FakeRunner | None = None, probe=None, **settings):
    """A Service around the fake runner, for API tests."""
    from localhost_ai.config import Settings
    from localhost_ai.memory import FakeProbe
    from localhost_ai.metrics import EngineMetrics
    from localhost_ai.service import ModelParts, Service

    tok = FakeTokenizer()

    def encode_chat(messages):
        return tok.encode(" ".join(m["content"] for m in messages))[:64] or [0]

    parts = ModelParts(name="fake-model", runner=runner or FakeRunner(t0=0.002), tokenizer=tok,
                       encode_chat=encode_chat, default_max_tokens=16,
                       info={"model": "fake-model", "repo": "test/fake", "device": "cpu",
                             "dtype": "fp32", "quant": "none"})
    defaults = {"control_interval_s": 0.2, "max_context": 256}
    return Service(settings=Settings(**{**defaults, **settings}), parts=parts,
                   probe=probe or FakeProbe(limit=1 << 30, used=1 << 28),
                   metrics=EngineMetrics(), model_names=["fake-model"])


class FakeCacheLost(MemoryError):
    pass


@dataclass
class LosingFakeRunner(FakeRunner):
    """Behaves like MLXModelRunner after a failed decode: the batch cache is gone, and every
    later decode/select/merge on it raises a MemoryError, so the scheduler has to recompute the
    whole batch. `oom_at_rows` makes any decode with that many rows fail."""

    oom_at_rows: int | None = None

    def decode(self, state: FakeState, tokens: list[int]):
        if getattr(state, "lost", False):
            raise FakeCacheLost("lost")
        if self.oom_at_rows is not None and len(state.rows) >= self.oom_at_rows:
            state.lost = True
            raise RuntimeError("[malloc] Unable to allocate 1073741824 bytes.")
        return super().decode(state, tokens)

    def select(self, state: FakeState, keep: list[int]) -> FakeState:
        if getattr(state, "lost", False):
            raise FakeCacheLost("lost")
        return super().select(state, keep)

    def merge(self, a: FakeState, b: FakeState) -> FakeState:
        if getattr(a, "lost", False) or getattr(b, "lost", False):
            raise FakeCacheLost("lost")
        return super().merge(a, b)
