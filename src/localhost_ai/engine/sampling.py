"""Per-row sampling for a mixed batch: every row has its own temperature/top-k/top-p and its own
seeded generator, so a request's output doesn't depend on who else is in the batch."""

from __future__ import annotations

import torch

from .request import SamplingParams


def make_generator(seed: int | None) -> torch.Generator:
    g = torch.Generator(device="cpu")
    if seed is None:
        g.seed()
    else:
        g.manual_seed(seed)
    return g


def sample(logits: torch.Tensor, params: list[SamplingParams],
           generators: list[torch.Generator]) -> list[int]:
    logits = logits.detach().to("cpu", torch.float32)
    greedy = logits.argmax(dim=-1)
    out = greedy.tolist()
    rows = [i for i, p in enumerate(params) if not p.greedy]
    if not rows:
        return out

    sub = logits[rows]
    temps = torch.tensor([params[i].temperature for i in rows]).unsqueeze(1)
    sub = sub / temps
    sorted_logits, sorted_idx = sub.sort(dim=-1, descending=True)
    vocab = sub.shape[-1]
    ranks = torch.arange(vocab).unsqueeze(0)

    top_k = torch.tensor([params[i].top_k if params[i].top_k > 0 else vocab for i in rows])
    mask = ranks >= top_k.unsqueeze(1)

    probs = sorted_logits.masked_fill(mask, float("-inf")).softmax(dim=-1)
    top_p = torch.tensor([params[i].top_p for i in rows]).unsqueeze(1)
    # Keep the smallest prefix whose mass reaches top_p (the first token always survives).
    cum_before = probs.cumsum(dim=-1) - probs
    mask |= cum_before >= top_p

    probs = sorted_logits.masked_fill(mask, float("-inf")).softmax(dim=-1)
    for j, i in enumerate(rows):
        pick = torch.multinomial(probs[j], 1, generator=generators[i])
        out[i] = int(sorted_idx[j, pick])
    return out
