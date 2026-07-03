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


def constrain(logits: torch.Tensor, constraints: list | None) -> torch.Tensor:
    """Set the logits a row's constraint (engine/constrain.py) forbids to -inf. Rows without
    one, and batches without any, are returned untouched."""
    if not constraints or all(c is None for c in constraints):
        return logits
    logits = logits.clone()
    width = logits.shape[-1]
    for i, c in enumerate(constraints):
        if c is not None:
            logits[i].masked_fill_(~c.allowed(width), float("-inf"))
    return logits


def sample(logits: torch.Tensor, params: list[SamplingParams],
           generators: list[torch.Generator], constraints: list | None = None) -> list[int]:
    """One token per row. `constraints`, if given, has one entry per row (None for
    unconstrained rows); each constrained row picks only allowed tokens and its constraint is
    advanced by the pick."""
    logits = constrain(logits.detach().to("cpu", torch.float32), constraints)
    out = _pick(logits, params, generators)
    for c, t in zip(constraints or (), out, strict=False):
        if c is not None:
            c.advance(t)
    return out


def _pick(logits: torch.Tensor, params: list[SamplingParams],
          generators: list[torch.Generator]) -> list[int]:
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
