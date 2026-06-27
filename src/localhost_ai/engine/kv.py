"""Everything that touches transformers' KV-cache internals lives here.

A running batch is a `DynamicCache` whose layers hold [batch, kv_heads, seq, head_dim] tensors,
plus a [batch, seq] attention mask where 0 marks left padding. Rows join by left-padding the
shorter side and concatenating on the batch dim (TGI v1 style); rows leave by index-select, after
which columns that are padding in every row are trimmed off the left."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import DynamicCache


def build(layers: list[tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    """Wrap existing tensors in a DynamicCache without copying them. (Passing tensors through
    `ddp_cache_data` goes through `update`, which concatenates onto an empty tensor: a copy.)"""
    cache = DynamicCache(ddp_cache_data=[(None, None)] * len(layers))
    for layer, (k, v) in zip(cache.layers, layers, strict=True):
        _set(layer, k, v)
    return cache


def _set(layer, k: torch.Tensor, v: torch.Tensor) -> None:
    if not layer.is_initialized:
        layer.lazy_initialization(k, v)
    layer.keys, layer.values = k, v


def layers(cache: DynamicCache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(layer.keys, layer.values) for layer in cache.layers]


def seq_len(cache: DynamicCache) -> int:
    return cache.get_seq_length()


def nbytes(cache: DynamicCache) -> int:
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size()
               for k, v in layers(cache))


def _left_pad(t: torch.Tensor, n: int) -> torch.Tensor:
    # t is [B, H, T, D]; pad spec is (D_left, D_right, T_left, T_right)
    return F.pad(t, (0, 0, n, 0)) if n else t


def merge(cache_a: DynamicCache, mask_a: torch.Tensor,
          cache_b: DynamicCache, mask_b: torch.Tensor) -> tuple[DynamicCache, torch.Tensor]:
    """Append b's rows to a, in place, one layer at a time. Each old layer tensor is released
    as soon as its merged replacement exists, so the peak is about the merged size plus one
    layer rather than a + b + merged. Both caches are consumed; if this raises (OOM), they are
    left half-merged and the caller must drop them."""
    ta, tb = mask_a.shape[1], mask_b.shape[1]
    t = max(ta, tb)
    for la, lb in zip(cache_a.layers, cache_b.layers, strict=True):
        k = torch.cat([_left_pad(la.keys, t - ta), _left_pad(lb.keys, t - tb)], dim=0)
        v = torch.cat([_left_pad(la.values, t - ta), _left_pad(lb.values, t - tb)], dim=0)
        _set(la, k, v)
        lb.keys = lb.values = None
    mask = torch.cat([F.pad(mask_a, (t - ta, 0)), F.pad(mask_b, (t - tb, 0))], dim=0)
    return cache_a, mask


def select_rows(cache: DynamicCache, mask: torch.Tensor,
                keep: list[int]) -> tuple[DynamicCache, torch.Tensor]:
    """Keep only `keep` rows, in place and layer by layer (same peak argument as merge)."""
    idx = torch.tensor(keep, dtype=torch.long, device=mask.device)
    mask = mask.index_select(0, idx)
    real_cols = mask.sum(dim=0).nonzero()
    drop = int(real_cols[0]) if real_cols.numel() else mask.shape[1]
    for layer in cache.layers:
        k = layer.keys.index_select(0, idx)[:, :, drop:]
        v = layer.values.index_select(0, idx)[:, :, drop:]
        _set(layer, k, v)
    return cache, mask[:, drop:]


def crop(cache: DynamicCache, length: int) -> None:
    """Undo a partially applied forward pass (e.g. OOM in layer 17 of 30): every layer that
    already appended the new step is cut back to `length`."""
    for layer in cache.layers:
        if layer.keys.shape[2] > length:
            layer.keys = layer.keys[:, :, :length]
            layer.values = layer.values[:, :, :length]


def prefill_positions(mask: torch.Tensor) -> torch.Tensor:
    return (mask.cumsum(dim=1) - 1).clamp(min=0)


def next_positions(mask: torch.Tensor) -> torch.Tensor:
    """Position id of the next token in each row = number of real tokens so far."""
    return mask.sum(dim=1, keepdim=True)
