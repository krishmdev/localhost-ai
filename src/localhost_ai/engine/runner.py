from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from transformers import DynamicCache

from . import kv
from .prefix import PrefixCache, PrefixEntry, prefill_with_prefixes


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
                 device: torch.device, dtype: torch.dtype,
                 prefix_cache: PrefixCache | None = None) -> None:
        self.model = model
        self.pad_id = pad_id
        self.eos_ids = eos_ids
        self.device = device
        self.kv_bytes_per_token = kv_bytes_per_token(model.config, dtype)
        self.prefix: PrefixCache | None = prefix_cache

    def prefill(self, seqs: list[list[int]]) -> tuple[HFBatch, torch.Tensor]:
        if self.prefix is None:
            return self._prefill(seqs, None)
        return prefill_with_prefixes(self, self.prefix, seqs)

    @torch.inference_mode()
    def _build_prefix(self, tokens: list[int]) -> tuple[list[tuple[torch.Tensor, ...]], int]:
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        out = self.model(input_ids=ids, past_key_values=DynamicCache(), use_cache=True,
                         logits_to_keep=1)
        return kv.layers(out.past_key_values), kv.nbytes(out.past_key_values)

    @torch.inference_mode()
    def _prefill(self, seqs: list[list[int]],
                 entry: PrefixEntry | None) -> tuple[HFBatch, torch.Tensor]:
        """Left-padded prefill. With a cached prefix, each row's cache starts with the prefix's
        KV and only the suffixes run, left-padded after the prefix: the mask then has a hole
        of padding between prefix and suffix, which the attention mask and the cumsum position
        ids handle the same way as leading padding."""
        n = len(entry.tokens) if entry is not None else 0
        tails = [s[n:] for s in seqs]
        t = max(len(s) for s in tails)
        ids = torch.full((len(seqs), t), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), t), dtype=torch.long)
        for i, s in enumerate(tails):
            ids[i, t - len(s):] = torch.tensor(s, dtype=torch.long)
            mask[i, t - len(s):] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        cache = DynamicCache()
        if entry is not None:
            b = len(seqs)
            cache = kv.build([(k.expand(b, -1, -1, -1), v.expand(b, -1, -1, -1))
                              for k, v in entry.state])
            mask = torch.cat([torch.ones((b, n), dtype=torch.long, device=mask.device), mask],
                             dim=1)
        out = self.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=kv.prefill_positions(mask)[:, n:],
            past_key_values=cache,
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

    @torch.inference_mode()
    def logprob_rows(self, ids: list[int], rows: list[int]) -> torch.Tensor:
        """Next-token log-probabilities after each of `rows` of one sequence (for /v1/score).
        Only the requested rows go through the LM head (logits_to_keep), so the logits held
        are rows x vocab, not sequence x vocab."""
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        keep = torch.tensor(rows, dtype=torch.long, device=self.device)
        logits = self.model(input_ids=x, use_cache=False, logits_to_keep=keep).logits[0].float()
        return torch.log_softmax(logits, dim=-1).cpu()

    def merge(self, a: HFBatch, b: HFBatch) -> HFBatch:
        cache, mask = kv.merge(a.cache, a.mask, b.cache, b.mask)
        return HFBatch(cache, mask)

    def select(self, state: HFBatch, keep: list[int]) -> HFBatch:
        cache, mask = kv.select_rows(state.cache, state.mask, keep)
        return HFBatch(cache, mask)

    def padded_tokens(self, state: HFBatch) -> int:
        return int(state.mask.numel())

    def drop_prefixes(self) -> int:
        freed = self.prefix.clear() if self.prefix is not None else 0
        self.release()
        return freed

    def release(self) -> None:
        """Hand cached allocator blocks back after shedding rows, so the memory probe (and the
        OS) see the drop."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
