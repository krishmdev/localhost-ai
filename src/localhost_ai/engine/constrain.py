"""Constrained decoding for `response_format` (JSON object or JSON schema), with llguidance.

Each constrained request gets its own matcher. Before a token is sampled, the matcher gives the
set of tokens that keep the output a valid prefix of the grammar, and `sampling.sample` sets
every other logit to -inf, so greedy and sampled rows can only pick allowed tokens. Once the
grammar is complete, only end-of-sequence tokens are allowed. Both backends hand their logits
to the same `sample`, so this works the same on torch and MLX.

The llguidance tokenizer (the vocabulary as bytes, plus its lookup structures) takes about a
second to build for a 150k vocabulary, so it is built once per loaded model, on first use."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

# Whitespace between JSON tokens: newlines and indentation are allowed so a model can
# pretty-print, but a run is bounded, so it can't pad forever instead of closing the object.
WHITESPACE = r"[\x20\x0A\x0D\x09]{0,24}"


class GrammarError(ValueError):
    """The response_format can't be compiled (a bad schema, or unsupported keywords)."""


class JSONConstraint:
    """Token mask and state for one request. Not thread-safe; only the compute thread uses it
    once the request is submitted."""

    def __init__(self, ll_tokenizer: Any, grammar: str, eos_ids: frozenset[int]) -> None:
        from llguidance import LLMatcher

        self.matcher = LLMatcher(ll_tokenizer, grammar, log_level=0)
        if self.matcher.is_error():
            raise GrammarError(self.matcher.get_error())
        self.vocab = ll_tokenizer.vocab_size
        self.eos_ids = sorted(eos_ids)
        self._words = np.zeros((self.vocab + 31) // 32, dtype=np.int32)
        self.failed = False

    def allowed(self, width: int) -> torch.Tensor:
        """Bool mask over the model's logits (`width` may be larger than the tokenizer's vocab;
        the extra padding rows are never allowed)."""
        mask = np.zeros(width, dtype=bool)
        if self.failed or self.matcher.is_error():
            eos = [i for i in self.eos_ids if i < width] or [0]
            mask[eos] = True
            return torch.from_numpy(mask)
        self.matcher.unsafe_compute_mask_ptr(self._words.ctypes.data, self._words.nbytes)
        bits = np.unpackbits(self._words.view(np.uint8), bitorder="little")
        n = min(width, self.vocab)
        mask[:n] = bits[:n]
        if not mask.any():  # can't happen for a valid grammar; end the row rather than crash
            eos = [i for i in self.eos_ids if i < width] or [0]
            mask[eos] = True
        return torch.from_numpy(mask)

    def advance(self, token: int) -> None:
        if not self.failed and not self.matcher.consume_token(token):
            self.failed = True  # from here on only EOS is allowed

    @property
    def done(self) -> bool:
        return self.matcher.is_stopped()


def grammar_for(response_format: dict) -> str | None:
    """llguidance grammar for an OpenAI response_format dict, None for plain text."""
    from llguidance import LLMatcher

    kind = response_format.get("type", "text")
    if kind == "text":
        return None
    opts = {"whitespace_pattern": WHITESPACE}
    if kind == "json_object":
        schema: Any = {"type": "object"}
    elif kind == "json_schema":
        spec = response_format.get("json_schema") or {}
        schema = spec.get("schema")
        if schema is None:
            schema = {"type": "object"}
        if not isinstance(schema, dict):
            raise GrammarError("json_schema.schema must be an object")
        # strict schemas fail on keywords llguidance doesn't implement; others ignore them
        opts["lenient"] = not spec.get("strict", False)
    else:
        raise GrammarError(f"unsupported response_format type {kind!r}")
    try:
        grammar = LLMatcher.grammar_from_json_schema(json.dumps(schema), overrides=opts)
    except ValueError as exc:
        raise GrammarError(str(exc)) from exc
    problem = LLMatcher.validate_grammar(grammar)
    if problem:
        raise GrammarError(problem)
    return grammar


class Grammars:
    """Per-model factory for constraints. `ll_tokenizer` builds the llguidance tokenizer; it's
    called once, lazily, under a lock (the first constrained request pays for it)."""

    def __init__(self, build: Callable[[], Any], eos_ids: frozenset[int]) -> None:
        self._build = build
        self._ll: Any = None
        self._lock = threading.Lock()
        self.eos_ids = eos_ids

    @classmethod
    def from_hf(cls, tokenizer: Any, eos_ids: frozenset[int]) -> Grammars:
        def build() -> Any:
            import llguidance.hf

            return llguidance.hf.from_tokenizer(tokenizer, eos_token=sorted(eos_ids))

        return cls(build, eos_ids)

    @property
    def ll_tokenizer(self) -> Any:
        with self._lock:
            if self._ll is None:
                self._ll = self._build()
            return self._ll

    def constraint(self, response_format: dict | None) -> JSONConstraint | None:
        if not response_format:
            return None
        grammar = grammar_for(response_format)
        if grammar is None:
            return None
        return JSONConstraint(self.ll_tokenizer, grammar, self.eos_ids)
