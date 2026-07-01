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


def row_state_bytes(cache: list[Any], rows: int) -> int:
    """Per-row bytes held by recurrent layers (conv and SSM/delta-rule state), 0 for pure
    attention models."""
    total = sum(c.nbytes for c in _leaves(cache) if getattr(c, "keys", None) is None
                and hasattr(c, "nbytes"))
    return total // max(1, rows)


class MLXModelRunner:
    def __init__(self, model: Any, eos_ids: frozenset[int], prefill_step: int = 512) -> None:
        self.model = model
        self.eos_ids = eos_ids
        self.prefill_step = prefill_step
        self.kv_bytes_per_token = 0
        self.row_state_bytes = 0
        # Measure on a real two-row cache (this also serves as the warmup).
        state, logits = self.prefill([[1, 2, 3, 4], [5, 6]])
        self.decode(state, logits.argmax(-1).tolist())
        self.kv_bytes_per_token = kv_bytes_per_token(state.cache)
        self.row_state_bytes = row_state_bytes(state.cache, 2)
        del state
        mx.clear_cache()

    def _forward(self, cache: list[Any], tokens: list[int]) -> torch.Tensor:
        out = self.model(mx.array(tokens, dtype=mx.int32)[:, None], cache=cache)[:, -1, :]
        out = out.astype(mx.float32)
        mx.eval(out, [c.state for c in cache])
        return torch.from_numpy(np.array(out))

    def prefill(self, seqs: list[list[int]]) -> tuple[MLXBatch, torch.Tensor]:
        """Everything but the last token of each prompt goes through the model right-padded;
        `finalize` then rolls each row so the batch ends up left-padded (mlx-lm's own batch
        generator does the same). The last tokens run as one decode step, so every row's logits
        come from the same position."""
        cache = batch_cache(self.model, len(seqs))
        heads = [s[:-1] for s in seqs]
        width = max(len(h) for h in heads)
        if width:
            lengths = [len(h) for h in heads]
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

    def padded_tokens(self, state: MLXBatch) -> int:
        # filter trims columns that are padding in every row, so the width is the longest row
        return len(state.lengths) * max(state.lengths) if state.lengths else 0

    def release(self) -> None:
        mx.clear_cache()
