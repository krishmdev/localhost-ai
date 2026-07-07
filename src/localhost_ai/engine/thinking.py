"""Thinking blocks: splitting reasoning from the answer, and a per-request thinking budget.

Qwen3.5 writes its reasoning between `<think>` and `</think>` (with thinking on, the chat
template already ends the prompt with `<think>`, so generation starts inside the block). Gemma 4
opens a thought channel itself, `<|channel>thought` ... `<channel|>`, when the system turn
carries `<|think|>` (enable_thinking). Which markers a model uses is read off its vocabulary, the
same way mlx-lm's tokenizer wrapper does.

`Thinking` sits where a response_format constraint sits (Request.constraint) and keeps the same
interface, so the scheduler and the sampler treat it like any other constraint. Per token it
records whether the token was reasoning, a marker, or answer text, which the scheduler uses to
send reasoning and content as separate streams. With a budget (`max_thinking_tokens`), once the
block holds that many reasoning tokens the next tokens are forced: a newline and the closing
marker. The model then answers as usual. A JSON constraint, if the request has one, only starts
masking once the block is closed, so the reasoning is free text and the answer is valid JSON."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import torch


@dataclass(frozen=True)
class ThinkSpec:
    open_ids: tuple[int, ...]  # the opening marker; the first token starts the block
    close_id: int
    force_ids: tuple[int, ...]  # forced at the budget: a newline, then close_id
    is_space: Callable[[int], bool] = field(compare=False, repr=False)

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> ThinkSpec | None:
        vocab = tokenizer.get_vocab()

        def enc(text: str) -> tuple[int, ...]:
            return tuple(tokenizer.encode(text, add_special_tokens=False))

        if "<think>" in vocab and "</think>" in vocab:
            opened, close = (vocab["<think>"],), "</think>"
        elif "<|channel>" in vocab and "<channel|>" in vocab:
            opened, close = enc("<|channel>thought"), "<channel|>"
        else:
            return None
        force = enc("\n" + close)
        if not opened or force[-1:] != (vocab[close],):
            return None

        @lru_cache(maxsize=4096)
        def is_space(t: int) -> bool:
            return tokenizer.decode([t], skip_special_tokens=False).strip() == ""

        return cls(opened, vocab[close], force, is_space)

    def inside_at_start(self, prompt_ids: list[int]) -> bool:
        """True when the prompt leaves a block open (Qwen3.5's template ends with `<think>`
        when thinking is on, and with `<think></think>` when it is off)."""
        last_open = max((i for i, t in enumerate(prompt_ids) if t == self.open_ids[0]),
                        default=-1)
        last_close = max((i for i, t in enumerate(prompt_ids) if t == self.close_id),
                         default=-1)
        return last_open > last_close


def _only(token: int, width: int) -> torch.Tensor:
    mask = torch.zeros(width, dtype=torch.bool)
    mask[token] = True
    return mask


class Thinking:
    """Thinking state for one request, used as its constraint. `inner` is the response_format
    constraint for the answer, or None."""

    def __init__(self, spec: ThinkSpec, budget: int | None, inside: bool, may_open: bool,
                 inner: Any = None, eos_ids: frozenset[int] = frozenset()) -> None:
        self.spec = spec
        self.eos_ids = sorted(eos_ids)
        self.budget = budget
        self.inside = inside
        self.may_open = may_open and not inside  # the first token may open a block
        self.inner = inner
        self.tokens = 0  # reasoning tokens (markers and forced tokens not counted)
        self.forced = False  # the budget closed the block
        self.kind = "content"  # what the last token was: "think", "marker" or "content"
        self._forcing: list[int] = []
        self._marker: list[int] = []  # rest of an opening marker being written
        self._after_close = False

    # the constraint interface (engine/constrain.py)
    @property
    def complete(self) -> bool:
        return not self.inside and self.inner is not None and self.inner.complete

    @property
    def broken(self) -> str | None:
        return None if self.inside or self.inner is None else self.inner.broken

    def allowed(self, width: int) -> torch.Tensor | None:
        if self._forcing:
            return _only(self._forcing[0], width)
        if self.inside:
            if not self._marker and self.budget is not None and self.tokens >= self.budget:
                self._forcing = list(self.spec.force_ids)
                self.forced = True
                return _only(self._forcing[0], width)
            if self.inner is None or not self.eos_ids:
                return None
            # with a JSON answer to come, the row may not end inside the block: a "stop"
            # with no JSON would break the guarantee that a stopped JSON answer parses
            mask = torch.ones(width, dtype=torch.bool)
            mask[[t for t in self.eos_ids if t < width]] = False
            return mask
        if self.inner is None:
            return None
        mask = self.inner.allowed(width)
        if self.may_open:
            mask = mask.clone()
            mask[self.spec.open_ids[0]] = True
        return mask

    def advance(self, token: int) -> None:
        if self._forcing:
            self._forcing.pop(0)
            self.kind = "marker"
            if not self._forcing:
                self._close()
            return
        if self.inside:
            if self._marker and token == self._marker[0]:
                self._marker.pop(0)
                self.kind = "marker"
                return
            self._marker = []
            if token == self.spec.close_id:
                self.kind = "marker"
                self._close()
                return
            self.tokens += 1
            self.kind = "think"
            return
        if self.may_open and token == self.spec.open_ids[0]:
            self.inside, self.may_open = True, False
            self._marker = list(self.spec.open_ids[1:])
            self.kind = "marker"
            return
        self.may_open = False
        if self.inner is not None:
            self.inner.advance(token)
        if self._after_close and self.spec.is_space(token):
            self.kind = "marker"  # the blank line between the block and the answer
            return
        self._after_close = False
        self.kind = "content"

    def _close(self) -> None:
        self.inside = False
        self._after_close = True


def wrap(spec: ThinkSpec | None, prompt_ids: list[int], template_kwargs: dict,
         budget: int | None, inner: Any, eos_ids: frozenset[int] = frozenset()) -> Any:
    """The request's constraint: a Thinking around `inner` when this prompt can produce a
    thinking block, else `inner` unchanged."""
    if spec is None:
        return inner
    inside = spec.inside_at_start(prompt_ids)
    may_open = bool(template_kwargs.get("enable_thinking"))
    if not inside and not may_open:
        return inner
    return Thinking(spec, budget, inside, may_open, inner, eos_ids)
