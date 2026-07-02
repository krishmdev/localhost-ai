from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from transformers import DynamicCache

from . import kv


class ModelRunner(Protocol):
    """What the scheduler needs from a model. Batch state is opaque to the scheduler."""

    kv_bytes_per_token: int
    eos_ids: frozenset[int]

    def prefill(self, seqs: list[list[int]]) -> tuple[Any, torch.Tensor]: ...
    def decode(self, state: Any, tokens: list[int]) -> torch.Tensor: ...
    def merge(self, a: Any, b: Any) -> Any: ...
    def select(self, state: Any, keep: list[int]) -> Any: ...
    def padded_tokens(self, state: Any) -> int: ...


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, (torch.OutOfMemoryError, MemoryError)):
        return True
    msg = str(exc).lower()
    return isinstance(exc, RuntimeError) and (
        "out of memory" in msg or "can't allocate memory" in msg or "mps backend out of" in msg
        # MLX: a buffer over Metal's size limit, too many buffers, or newBuffer returning nil
        or "[metal::malloc]" in msg or "[malloc] unable to allocate" in msg
    )


def kv_bytes_per_token(config: Any, dtype: torch.dtype) -> int:
    """2 (K and V) * layers * kv_heads * head_dim * bytes. SmolLM2-135M in fp32:
    2 * 30 * 3 * 64 * 4 = 46,080 bytes = 45 KiB per token."""
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    heads = text.num_attention_heads
    kv_heads = getattr(text, "num_key_value_heads", None) or heads
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // heads
    itemsize = torch.tensor([], dtype=dtype).element_size()
    return 2 * text.num_hidden_layers * kv_heads * head_dim * itemsize


@dataclass
class HFBatch:
    cache: DynamicCache
    mask: torch.Tensor  # [B, T] long, 0 = left padding


class HFModelRunner:
    def __init__(self, model: Any, pad_id: int, eos_ids: frozenset[int],
                 device: torch.device, dtype: torch.dtype) -> None:
        self.model = model
        self.pad_id = pad_id
        self.eos_ids = eos_ids
        self.device = device
        self.kv_bytes_per_token = kv_bytes_per_token(model.config, dtype)

    @torch.inference_mode()
    def prefill(self, seqs: list[list[int]]) -> tuple[HFBatch, torch.Tensor]:
        t = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), t), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), t), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, t - len(s):] = torch.tensor(s, dtype=torch.long)
            mask[i, t - len(s):] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        out = self.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=kv.prefill_positions(mask),
            past_key_values=DynamicCache(),
            use_cache=True,
            logits_to_keep=1,
        )
        return HFBatch(out.past_key_values, mask), out.logits[:, -1, :]

    @torch.inference_mode()
    def decode(self, state: HFBatch, tokens: list[int]) -> torch.Tensor:
        ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        positions = kv.next_positions(state.mask)
        mask = torch.cat([state.mask, torch.ones_like(ids)], dim=1)
        before = kv.seq_len(state.cache)
        try:
            out = self.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=state.cache,
                use_cache=True,
                logits_to_keep=1,
            )
        except BaseException:
            kv.crop(state.cache, before)
            raise
        state.mask = mask
        return out.logits[:, -1, :]

    def merge(self, a: HFBatch, b: HFBatch) -> HFBatch:
        cache, mask = kv.merge(a.cache, a.mask, b.cache, b.mask)
        return HFBatch(cache, mask)

    def select(self, state: HFBatch, keep: list[int]) -> HFBatch:
        cache, mask = kv.select_rows(state.cache, state.mask, keep)
        return HFBatch(cache, mask)

    def padded_tokens(self, state: HFBatch) -> int:
        return int(state.mask.numel())

    def release(self) -> None:
        """Hand cached allocator blocks back after shedding rows, so the memory probe (and the
        OS) see the drop."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
