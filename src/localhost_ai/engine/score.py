"""Teacher-forced scoring for POST /v1/score: the log-probability of each candidate string at
given character offsets of a fixed continuation, with no sampling.

The chat template is applied with add_generation_prompt=True, the continuation is appended as
text with no end-of-turn token, and prompt plus continuation are tokenized together, so the
tokens are the ones the model would see if it had written the continuation itself. Each site's
char_offset (into the continuation) is mapped to a token through the tokenizer's offset mapping:

- a site where a token starts, with a single-token candidate: read in place from the one forward
  pass over the whole sequence (the distribution at the previous position);
- a site inside a token (or a multi-token candidate): back off to that token's start and
  teacher-force the text from there, log P(piece + candidate | head) - log P(piece | head).

This follows Vizor's MLXScorer step for step, so the two agree to float rounding. The sequence is
never truncated: if it (or a forced tail) doesn't fit the context, the request is rejected.

Every teacher-forced candidate costs a full forward pass, and the passes run on the compute
thread between generation steps, so a request is bounded before any of them runs: at most
MAX_CANDIDATE_TOKENS tokens per candidate and MAX_FORWARDS forward passes in all. `stop` is
checked before each pass, so a request whose client went away stops early."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

MAX_CANDIDATES = 64  # per site (the request schema enforces it)
MAX_CANDIDATE_TOKENS = 32
MAX_FORWARDS = 256  # per request, the in-place pass and every teacher-forced one

# (ids, rows) -> float32 log-probabilities [len(rows), vocab] for the next token after each row
LogprobRows = Callable[[list[int], list[int]], torch.Tensor]


class ScoreError(ValueError):
    """A request that can't be scored as asked (400)."""

    code = "invalid_request"


class ContextOverflow(ScoreError):
    code = "context_length_exceeded"


class TooMuchWork(ScoreError):
    code = "score_too_large"


class Stopped(RuntimeError):
    """The caller went away while its request was being scored."""


@dataclass(frozen=True)
class Site:
    char_offset: int
    candidates: tuple[str, ...]


@dataclass
class SiteResult:
    char_offset: int
    token_index: int
    logprobs: dict[str, float]
    forced: bool

    def renorm(self) -> dict[str, float]:
        if not self.logprobs:
            return {}
        m = max(self.logprobs.values())
        ex = {k: math.exp(v - m) for k, v in self.logprobs.items()}
        z = sum(ex.values())
        return {k: v / z for k, v in ex.items()}


def encode_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids and character spans of `text`, special tokens parsed from the text (the chat
    template writes them out) and none added."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]


def encode_plain(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else []


def score(tokenizer: Any, logprob_rows: LogprobRows, head: str, continuation: str,
          sites: list[Site], max_context: int,
          stop: Callable[[], bool] = lambda: False) -> tuple[list[SiteResult], int]:
    """Score every site. Returns the results in request order and the joint token count.
    Raises ScoreError before any forward pass for a request that is malformed or over the
    limits, and Stopped between passes once `stop()` is true."""
    full = head + continuation
    ids, offs = encode_offsets(tokenizer, full)
    if len(ids) > max_context:
        raise ContextOverflow(f"prompt plus continuation is {len(ids)} tokens; context is "
                              f"{max_context}")
    plans = []
    for s in sites:
        if not 0 <= s.char_offset <= len(continuation):
            raise ScoreError(f"char_offset {s.char_offset} is outside the continuation "
                             f"({len(continuation)} characters)")
        if not s.candidates or any(not c for c in s.candidates):
            raise ScoreError("every site needs at least one non-empty candidate")
        if len(s.candidates) > MAX_CANDIDATES:
            raise ScoreError(f"a site has {len(s.candidates)} candidates; the limit is "
                             f"{MAX_CANDIDATES}")
        a = len(head) + s.char_offset
        t = next((i for i, (_, en) in enumerate(offs) if en > a), len(ids))
        start = offs[t][0] if t < len(ids) else len(full)
        c_ids = {c: encode_plain(tokenizer, c) for c in s.candidates}
        if any(len(v) > MAX_CANDIDATE_TOKENS for v in c_ids.values()):
            raise ScoreError(f"a candidate is longer than {MAX_CANDIDATE_TOKENS} tokens")
        plans.append((s, a, t, start, c_ids))
    if any(t == 0 for _, _, t, _, _ in plans):
        raise ScoreError("a site maps to the first token, which has no context to score from")

    in_place = sorted({t - 1 for s, a, t, start, _ in plans if start == a})
    forwards = 1 if in_place else 0
    for s, a, _, start, c_ids in plans:
        forced = sum(1 for c in s.candidates if not (start == a and len(c_ids[c]) == 1))
        forwards += forced + (1 if forced and start < a else 0)  # + the shared log P(piece)
    if forwards > MAX_FORWARDS:
        raise TooMuchWork(f"this request needs {forwards} forward passes; the limit is "
                          f"{MAX_FORWARDS} (fewer sites, fewer or single-token candidates)")

    def rows_of(seq_ids: list[int], rows: list[int]) -> torch.Tensor:
        if stop():
            raise Stopped("the client went away")
        return logprob_rows(seq_ids, rows)

    rows = dict(zip(in_place, rows_of(ids, in_place), strict=True)) if in_place else {}

    def seq(head_ids: list[int], tail: list[int]) -> float:
        if not tail:
            return 0.0
        if len(head_ids) + len(tail) > max_context:
            raise ContextOverflow(f"a teacher-forced candidate needs {len(head_ids) + len(tail)}"
                                  f" tokens; context is {max_context}")
        n = len(head_ids)
        lp = rows_of(head_ids + tail, [n + i - 1 for i in range(len(tail))])
        return float(sum(lp[i, tok].item() for i, tok in enumerate(tail)))

    out = []
    for s, a, t, start, c_ids in plans:
        lp: dict[str, float] = {}
        forced = False
        base: float | None = None  # log P(piece | head), shared by the site's candidates
        for c in s.candidates:
            if start == a and len(c_ids[c]) == 1:
                lp[c] = float(rows[t - 1][c_ids[c][0]].item())
                continue
            forced = True
            piece = full[start:a]
            if base is None:
                base = seq(ids[:t], encode_plain(tokenizer, piece))
            lp[c] = seq(ids[:t], encode_plain(tokenizer, piece + c)) - base
        out.append(SiteResult(s.char_offset, t, lp, forced))
    return out, len(ids)
