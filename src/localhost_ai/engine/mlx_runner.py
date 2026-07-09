"""ModelRunner on Apple MLX, for 4- and 8-bit checkpoints (the mlx-community conversions).

Same contract as HFModelRunner: the batch state is opaque to the scheduler, rows join by
left-padding and concatenating, rows leave by index-select. The per-layer caches are mlx-lm's
batch caches (BatchKVCache, plus the rotating and recurrent variants that sliding-window and
linear-attention layers use). They already track per-row left padding, so this file only wires
them to the scheduler. Logits come back as CPU torch tensors, so sampling is shared with the torch
path and a seeded request samples the same way on either backend.

The batch caches are built here from mlx-lm's public cache classes rather than its private
batch-generator helpers, but they are still internals that change between releases, so mlx-lm is
pinned exactly in pyproject.toml."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
import torch
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
    make_prompt_cache,
)

from .prefix import PrefixCache, PrefixEntry, prefill_with_prefixes


class CacheLost(MemoryError):
    """The last forward pass failed after some layers had already advanced their caches. MLX
    caches can't be cropped back like kv.crop does for transformers (recurrent layers have no
    history to cut), so the whole batch has to be recomputed. Subclassing MemoryError makes the
    scheduler treat it like an OOM in select: drop the batch's KV and requeue every row."""


@dataclass
class MLXBatch:
    cache: list[Any]
    lengths: list[int]  # real tokens per row; the batch is left-padded to the longest
    lost: bool = False


def _leaves(cache: list[Any]) -> list[Any]:
    out = []
    for c in cache:
        out.extend(_leaves(c.caches) if hasattr(c, "caches") else [c])  # CacheList
    return out


def batch_cache(model: Any, rows: int) -> list[Any]:
    """Empty batch-aware caches for `rows` sequences, one per layer, in the model's own layout.

    Recurrent layers (ArraysCache) must NOT get a left_padding array here. ArraysCache.make_mask
    checks left_padding before lengths, so a zero left_padding would make the right-padded
    prefill feed the pad tokens into the recurrent state of every shorter row. mlx-lm 0.31.3's
    `_make_cache` and `ArraysCache.merge` of empty caches both set it; this builder leaves it
    unset, so `prepare(lengths=...)` masks the padding as intended."""

    def convert(c: Any) -> Any:
        if type(c) is KVCache:
            return BatchKVCache([0] * rows)
        if isinstance(c, ArraysCache):
            return c  # fresh from make_prompt_cache: left_padding and lengths are None
        if isinstance(c, RotatingKVCache):
            if c.keep > 0:
                raise ValueError("sliding-window caches that keep leading tokens can't batch")
            return BatchRotatingKVCache(c.max_size, [0] * rows)
        if isinstance(c, CacheList):
            return CacheList(*(convert(sub) for sub in c.caches))
        raise ValueError(f"{type(c).__name__} does not support batching")

    return [convert(c) for c in make_prompt_cache(model)]


def kv_bytes_per_token(cache: list[Any]) -> int:
    """Bytes one more token adds across all attention layers: K and V, kv_heads x head_dim each,
    read off the live cache tensors ([B, kv_heads, T, head_dim]). Layers that share another
    layer's KV have no cache of their own and are not counted twice. Sliding-window layers are
    counted as if they never wrap, which overstates long rows. Recurrent (linear-attention) state
    is a fixed cost per row, not per token, so it isn't in this number; see row_state_bytes."""
    total = 0
    for c in _leaves(cache):
        k, v = getattr(c, "keys", None), getattr(c, "values", None)
        if k is None or v is None:
            continue
        total += k.shape[1] * k.shape[3] * k.itemsize + v.shape[1] * v.shape[3] * v.itemsize
    return total


def kv_layers(cache: list[Any]) -> list[tuple[int, int | None, int]]:
    """Per attention layer with a cache of its own: (bytes per token per row, window or None
    for full attention, allocation step). Read off a live cache like kv_bytes_per_token."""
    out = []
    for c in _leaves(cache):
        k, v = getattr(c, "keys", None), getattr(c, "values", None)
        if k is None or v is None:
            continue
        per = k.shape[1] * k.shape[3] * k.itemsize + v.shape[1] * v.shape[3] * v.itemsize
        out.append((per, getattr(c, "max_size", None), getattr(c, "step", 1)))
    return out


def row_state_bytes(cache: list[Any], rows: int) -> int:
    """Per-row bytes held by recurrent layers (conv and SSM/delta-rule state), 0 for pure
    attention models."""
    total = sum(c.nbytes for c in _leaves(cache) if getattr(c, "keys", None) is None
                and hasattr(c, "nbytes"))
    return total // max(1, rows)


def _repeat(c: Any, rows: int) -> Any:
    """A batch cache of `rows` copies of one single-sequence cache (a stored prefix)."""
    if isinstance(c, CacheList):
        return CacheList(*(_repeat(sub, rows) for sub in c.caches))
    return type(c).merge([c] * rows)


# mlx-lm model classes whose __call__ (checked against the pinned mlx-lm) is exactly
# `head(self.model(inputs))`, where self.model ends in the final norm and head is lm_head, or the
# tied embedding's as_linear, then Gemma 4's final logit softcap. Wrappers that only forward to a
# `language_model` (gemma4.Model, qwen3_5.Model) are unwrapped first.
_HEADS = frozenset({"mlx_lm.models.llama", "mlx_lm.models.qwen2", "mlx_lm.models.qwen3_5",
                    "mlx_lm.models.gemma4_text"})


def _split_head(model: Any) -> tuple[Any, Any] | None:
    """(body, head) with head(body(x)) == model(x), or None for an architecture not in _HEADS."""
    lm = getattr(model, "language_model", model)
    if type(lm).__module__ not in _HEADS:
        return None
    tied = getattr(lm, "tie_word_embeddings", None)
    if tied is None:
        tied = lm.args.tie_word_embeddings
    project = lm.model.embed_tokens.as_linear if tied else lm.lm_head
    cap = getattr(lm, "final_logit_softcapping", None)
    if cap is None:
        return lm.model, project
    softcap = sys.modules[type(lm).__module__].logit_softcap  # the model's own function
    return lm.model, lambda h: softcap(cap, project(h))


class MLXModelRunner:
    def __init__(self, model: Any, eos_ids: frozenset[int], prefill_step: int = 512,
                 prefix_cache: PrefixCache | None = None) -> None:
        self.model = model
        self.eos_ids = eos_ids
        self.prefill_step = prefill_step
        self.head_min_rows = 64  # see logprob_rows
        self.prefix: PrefixCache | None = None
        self.kv_bytes_per_token = 0
        self.row_state_bytes = 0
        # Measure on a real two-row cache (this also serves as the warmup).
        state, logits = self.prefill([[1, 2, 3, 4], [5, 6]])
        self.decode(state, logits.argmax(-1).tolist())
        self.kv_bytes_per_token = kv_bytes_per_token(state.cache)
        self.row_state_bytes = row_state_bytes(state.cache, 2)
        self._kv_layers = kv_layers(state.cache)
        del state
        mx.clear_cache()
        self.prefix = prefix_cache

    def _forward(self, cache: list[Any], tokens: list[int]) -> torch.Tensor:
        out = self.model(mx.array(tokens, dtype=mx.int32)[:, None], cache=cache)[:, -1, :]
        out = out.astype(mx.float32)
        mx.eval(out, [c.state for c in cache])
        return torch.from_numpy(np.array(out))

    def prefill(self, seqs: list[list[int]]) -> tuple[MLXBatch, torch.Tensor]:
        if self.prefix is None:
            return self._prefill(seqs, None)
        return prefill_with_prefixes(self, self.prefix, seqs)

    def _build_prefix(self, tokens: list[int]) -> tuple[list[Any], int]:
        """Run a shared prefix alone through the model's own single-sequence caches."""
        cache = make_prompt_cache(self.model)
        ids = mx.array([tokens], dtype=mx.int32)
        for i in range(0, len(tokens), self.prefill_step):
            self.model(ids[:, i:i + self.prefill_step], cache=cache)
            mx.eval([c.state for c in cache])
        return cache, sum(c.nbytes for c in _leaves(cache))

    def _prefill(self, seqs: list[list[int]],
                 entry: PrefixEntry | None) -> tuple[MLXBatch, torch.Tensor]:
        """Everything but the last token of each prompt goes through the model right-padded;
        `finalize` then rolls each row so the batch ends up left-padded (mlx-lm's own batch
        generator does the same). The last tokens run as one decode step, so every row's logits
        come from the same position.

        With a cached prefix, every row's cache starts as a copy of the prefix's (mlx-lm's
        batch `merge` of the single-sequence caches), and only the rest of each prompt is
        prefilled, exactly as if the prefix had been the first prefill chunk."""
        start = 0
        if entry is None:
            cache = batch_cache(self.model, len(seqs))
        else:
            start = len(entry.tokens)
            cache = [_repeat(c, len(seqs)) for c in entry.state]
        heads = [s[start:-1] for s in seqs]
        width = max(len(h) for h in heads)
        if width:
            lengths = [len(h) for h in heads]
            if min(lengths) < width and width % self.prefill_step == 1:
                # A 1-token chunk takes BatchRotatingKVCache's decode path, which refuses to run
                # while right padding is set. One more pad column makes the last chunk 2 wide.
                width += 1
            pad = [width - n for n in lengths]
            ids = mx.array([h + [0] * p for h, p in zip(heads, pad, strict=True)], dtype=mx.int32)
            if max(pad):
                for c in cache:
                    c.prepare(lengths=lengths, right_padding=pad)
            for i in range(0, width, self.prefill_step):
                self.model(ids[:, i:i + self.prefill_step], cache=cache)
                mx.eval([c.state for c in cache])
            if max(pad):
                for c in cache:
                    c.finalize()
        logits = self._forward(cache, [s[-1] for s in seqs])
        return MLXBatch(cache, [len(s) for s in seqs]), logits

    def decode(self, state: MLXBatch, tokens: list[int]) -> torch.Tensor:
        if state.lost:
            raise CacheLost("batch cache was lost in an earlier failed step")
        try:
            logits = self._forward(state.cache, tokens)
        except BaseException:
            state.lost = True
            raise
        state.lengths = [n + 1 for n in state.lengths]
        return logits

    def merge(self, a: MLXBatch, b: MLXBatch) -> MLXBatch:
        """Append b's rows to a in place; the shorter side is left-padded by mlx-lm's extend."""
        if a.lost or b.lost:
            raise CacheLost("cannot merge a lost batch cache")
        for ca, cb in zip(a.cache, b.cache, strict=True):
            ca.extend(cb)
        return MLXBatch(a.cache, a.lengths + b.lengths)

    def select(self, state: MLXBatch, keep: list[int]) -> MLXBatch:
        if state.lost:
            raise CacheLost("cannot select rows from a lost batch cache")
        idx = mx.array(keep, dtype=mx.int32)
        for c in state.cache:
            c.filter(idx)
        return MLXBatch(state.cache, [state.lengths[i] for i in keep])

    def logprob_rows(self, ids: list[int], rows: list[int]) -> torch.Tensor:
        """Next-token log-probabilities after each of `rows` of one sequence, from one forward
        pass with no cache (for /v1/score), normalized in float32.

        The whole sequence goes through the transformer layers at once on purpose: the client
        checks the result against a plain mlx-lm forward, and in bf16 on the GPU the kernels'
        arithmetic depends on the sequence length, so running the same tokens in prefill_step
        chunks moved Gemma 4 E4B's log-probabilities by up to 0.74 past 512 tokens.

        Only the output head is cut down: for the architectures in _HEADS it runs on the
        requested rows' hidden states (after the model's final norm), so the logits of every
        position (0.5 MB per token for Gemma's 262k vocabulary) never exist. The rows are padded
        to at least head_min_rows by repeating the last one, so the head's matmul has the same
        kernel shape class as in the full forward. Measured on the GPU at about 1,470 tokens, the
        padded rows equal the full forward's bit for bit, while 4 unpadded rows moved Gemma 4
        12B's log-probabilities by up to 0.44. Sequences shorter than that take the plain
        forward."""
        x = mx.array([ids], dtype=mx.int32)
        split = _split_head(self.model)
        if split is None or not rows or len(ids) <= self.head_min_rows:
            logits = self.model(x)[0][mx.array(rows, dtype=mx.int32)]
        else:
            body, head = split
            take = rows + [rows[-1]] * (self.head_min_rows - len(rows))
            h = body(x)[:, mx.array(take, dtype=mx.int32)]
            logits = head(h)[0, : len(rows)]
        sel = logits.astype(mx.float32)
        out = sel - mx.logsumexp(sel, axis=-1, keepdims=True)
        mx.eval(out)
        return torch.from_numpy(np.array(out))

    def row_bytes(self, tokens: int) -> int:
        """What one row of a batch padded to `tokens` holds: full-attention KV for every token,
        sliding-window KV for at most the window, both rounded up to mlx-lm's allocation step
        (256 tokens), plus the fixed recurrent state."""
        total = self.row_state_bytes
        for per, window, step in self._kv_layers:
            alloc = -(-max(1, tokens) // step) * step
            total += per * (alloc if window is None else min(alloc, window))
        return total

    def kv_bytes(self, state: MLXBatch) -> int:
        """Bytes the batch's caches hold right now, read off the arrays (allocated KV buffers
        and recurrent state). After `select` trims padding columns the KV arrays are views,
        so this can be a little below what is still allocated until the buffer is replaced."""
        return sum(c.nbytes for c in _leaves(state.cache) if hasattr(c, "nbytes"))

    def padded_tokens(self, state: MLXBatch) -> int:
        # filter trims columns that are padding in every row, so the width is the longest row
        return len(state.lengths) * max(state.lengths) if state.lengths else 0

    def release(self) -> None:
        mx.clear_cache()

    def drop_prefixes(self) -> int:
        freed = self.prefix.clear() if self.prefix is not None else 0
        mx.clear_cache()
        return freed
