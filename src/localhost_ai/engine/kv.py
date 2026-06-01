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
    return DynamicCache(ddp_cache_data=layers)


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
    ta, tb = mask_a.shape[1], mask_b.shape[1]
    t = max(ta, tb)
    merged = []
    for (ka, va), (kb, vb) in zip(layers(cache_a), layers(cache_b), strict=True):
        k = torch.cat([_left_pad(ka, t - ta), _left_pad(kb, t - tb)], dim=0)
        v = torch.cat([_left_pad(va, t - ta), _left_pad(vb, t - tb)], dim=0)
        merged.append((k, v))
    mask = torch.cat([F.pad(mask_a, (t - ta, 0)), F.pad(mask_b, (t - tb, 0))], dim=0)
    return build(merged), mask


def select_rows(cache: DynamicCache, mask: torch.Tensor,
                keep: list[int]) -> tuple[DynamicCache, torch.Tensor]:
    idx = torch.tensor(keep, dtype=torch.long, device=mask.device)
    kept = [(k.index_select(0, idx), v.index_select(0, idx)) for k, v in layers(cache)]
    return trim_left(build(kept), mask.index_select(0, idx))


def trim_left(cache: DynamicCache, mask: torch.Tensor) -> tuple[DynamicCache, torch.Tensor]:
    real_cols = mask.sum(dim=0).nonzero()
    drop = int(real_cols[0]) if real_cols.numel() else mask.shape[1]
    if drop == 0:
        return cache, mask
    for layer in cache.layers:
        layer.keys = layer.keys[:, :, drop:]
        layer.values = layer.values[:, :, drop:]
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
