"""Prefix caching: reuse the KV (and, for hybrid models, the recurrent state) of a prompt prefix
that many requests share, typically a long system prompt sent with every request.

Entries are found automatically. Each new prompt is compared with the last few prompts; when
two share at least `min_tokens` leading tokens, that common prefix is computed once in a
single-sequence cache and stored. Later prompts that start with a stored prefix only prefill
their own suffix: the runner starts their batch cache from copies of the stored one. Rows still
own their copy, so this saves prefill compute, not KV memory per row; the stored entries
themselves are extra memory, bounded by `budget_bytes` and dropped (LRU) when over it or when
the scheduler needs memory back.

The runner supplies two backend-specific operations, used by `prefill_with_prefixes`:
- `_build_prefix(tokens) -> (state, nbytes)`: run the prefix alone and keep its cache;
- `_prefill(seqs, entry) -> (batch, logits)`: prefill rows that all start with `entry.tokens`
  (or, with entry None, rows with no cached prefix)."""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PrefixEntry:
    tokens: tuple[int, ...]
    state: Any
    nbytes: int
    hits: int = 0
    created: float = field(default_factory=time.monotonic)


def common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class PrefixCache:
    def __init__(self, budget_bytes: int, max_entries: int = 8, min_tokens: int = 32,
                 recent: int = 16) -> None:
        self.budget_bytes = budget_bytes
        self.max_entries = max_entries
        self.min_tokens = min_tokens
        self.entries: OrderedDict[tuple[int, ...], PrefixEntry] = OrderedDict()  # LRU order
        self._recent: deque[list[int]] = deque(maxlen=recent)
        self.hits = 0
        self.misses = 0
        self.hit_tokens = 0
        self.evictions = 0

    @property
    def nbytes(self) -> int:
        return sum(e.nbytes for e in self.entries.values())

    def lookup(self, seq: list[int]) -> PrefixEntry | None:
        """The longest stored prefix of `seq` that still leaves at least one token to run."""
        best = None
        for key, e in self.entries.items():
            n = len(key)
            if n < len(seq) and (best is None or n > len(best.tokens)) and tuple(seq[:n]) == key:
                best = e
        return best

    def propose(self, seq: list[int], have: int = 0) -> int:
        """Length of a new prefix worth storing for `seq`, or 0: the longest common prefix with
        a recent prompt, if it has at least `min_tokens` and beats the `have`-token prefix
        already matched by a clear margin. Records `seq` as recent either way."""
        best = max((common_prefix(seq, r) for r in self._recent), default=0)
        self._recent.append(list(seq))
        best = min(best, len(seq) - 1)
        if best < self.min_tokens or (have and best < have + max(16, have // 4)):
            return 0
        return best

    def add(self, tokens: list[int], state: Any, nbytes: int) -> PrefixEntry | None:
        if nbytes > self.budget_bytes:
            return None
        e = PrefixEntry(tuple(tokens), state, nbytes)
        self.entries[e.tokens] = e
        self._fit()
        return e if e.tokens in self.entries else None

    def touch(self, e: PrefixEntry) -> None:
        e.hits += 1
        self.hits += 1
        self.hit_tokens += len(e.tokens)
        if e.tokens in self.entries:
            self.entries.move_to_end(e.tokens)

    def _fit(self) -> None:
        while self.entries and (len(self.entries) > self.max_entries
                                or self.nbytes > self.budget_bytes):
            self.entries.popitem(last=False)
            self.evictions += 1

    def clear(self) -> int:
        """Drop every entry (the scheduler does this before shedding rows or after an OOM).
        Returns the bytes released."""
        freed = self.nbytes
        self.evictions += len(self.entries)
        self.entries.clear()
        return freed

    def stats(self) -> dict:
        return {"prefix_entries": len(self.entries), "prefix_bytes": self.nbytes,
                "prefix_hits": self.hits, "prefix_misses": self.misses,
                "prefix_hit_tokens": self.hit_tokens, "prefix_evictions": self.evictions}


def prefill_with_prefixes(runner: Any, cache: PrefixCache,
                          seqs: list[list[int]]) -> tuple[Any, torch.Tensor]:
    """Group the rows by the stored prefix they start with (building one if a new shared prefix
    shows up), prefill each group, and merge the groups back in the original row order."""
    groups: dict[tuple[int, ...] | None, list[int]] = {}
    found: dict[tuple[int, ...], PrefixEntry] = {}
    for i, s in enumerate(seqs):
        e = cache.lookup(s)
        n = cache.propose(s, have=len(e.tokens) if e else 0)
        if n:
            state, nbytes = runner._build_prefix(s[:n])
            e = cache.add(s[:n], state, nbytes) or e
        if e is None:
            cache.misses += 1
            groups.setdefault(None, []).append(i)
        else:
            cache.touch(e)
            found[e.tokens] = e
            groups.setdefault(e.tokens, []).append(i)
    if list(groups) == [None]:
        return runner._prefill(seqs, None)

    order: list[int] = []
    state = None
    logits = []
    for key, idx in groups.items():
        part, lg = runner._prefill([seqs[i] for i in idx], found[key] if key else None)
        state = part if state is None else runner.merge(state, part)
        logits.append(lg)
        order += idx
    out = torch.cat(logits)
    if order != list(range(len(seqs))):
        back = sorted(range(len(order)), key=order.__getitem__)  # position of row i in order
        state = runner.select(state, back)
        out = out[back]
    return state, out
