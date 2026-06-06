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
    calls: list[tuple[str, int]] = field(default_factory=list)

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
        self.calls.append(("prefill", len(seqs)))
        return FakeState([list(s) for s in seqs], width), self._logits(seqs)

    def decode(self, state: FakeState, tokens: list[int]):
        self._check_oom(len(state.rows), state.width + 1)
        if self.t0 or self.k:
            time.sleep(self.t0 + self.k * len(state.rows))
        self.calls.append(("decode", len(state.rows)))
        for r, t in zip(state.rows, tokens, strict=True):
            r.append(t)
        state.width += 1
        return self._logits(state.rows)

    def merge(self, a: FakeState, b: FakeState) -> FakeState:
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
