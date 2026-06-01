"""Incremental detokenization with stop strings.

Decoding one token at a time is wrong for BPE byte-fallback tokens and for tokenizers that merge
leading spaces, so we decode a short window (prefix_offset..end) and emit only the part past what
was already emitted, the same trick TGI and vLLM use. Text that could be the start of a stop
string is held back until we know it isn't."""

from __future__ import annotations

from typing import Protocol


class Decoder(Protocol):
    def decode(self, ids: list[int], skip_special_tokens: bool = ...) -> str: ...


class IncrementalDetokenizer:
    def __init__(self, tokenizer: Decoder, stop: tuple[str, ...] = ()) -> None:
        self._tok = tokenizer
        self._stop = tuple(s for s in stop if s)
        self._ids: list[int] = []
        self._prefix = 0
        self._read = 0
        self._held = ""
        self.text = ""
        self.stopped = False

    def _decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def push(self, token_id: int) -> str:
        """Add one token; return the newly visible text (possibly empty)."""
        if self.stopped:
            return ""
        self._ids.append(token_id)
        prefix_text = self._decode(self._ids[self._prefix:self._read])
        full = self._decode(self._ids[self._prefix:])
        if len(full) <= len(prefix_text) or full.endswith("�"):
            return ""
        delta = full[len(prefix_text):]
        self._prefix, self._read = self._read, len(self._ids)
        return self._filter(delta)

    def _filter(self, delta: str) -> str:
        pending = self._held + delta
        if not self._stop:
            self.text += pending
            self._held = ""
            return pending
        cut = min((i for i in (pending.find(s) for s in self._stop) if i >= 0), default=-1)
        if cut >= 0:
            self.stopped = True
            out = pending[:cut]
            self._held = ""
            self.text += out
            return out
        keep = 0
        for s in self._stop:
            for n in range(min(len(s) - 1, len(pending)), 0, -1):
                if pending.endswith(s[:n]):
                    keep = max(keep, n)
                    break
        out = pending[: len(pending) - keep]
        self._held = pending[len(pending) - keep:]
        self.text += out
        return out

    def flush(self) -> str:
        """Release held-back text once generation ended for another reason."""
        out, self._held = self._held, ""
        if self.stopped:
            return ""
        self.text += out
        return out
