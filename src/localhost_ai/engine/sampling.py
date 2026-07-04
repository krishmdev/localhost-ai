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


# Candidates kept per sampled row before top-k/top-p. Sorting a full 150k vocabulary on the
# CPU took 13 ms per row per step (Qwen2.5-3B), about as long as the model's decode step, so the
# nucleus is looked for among the top CANDIDATES logits first. Rows whose nucleus doesn't fit
# there (a very flat distribution) fall back to the full sort, so the result is exact either way.
CANDIDATES = 256


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
    for j, (probs, ids) in enumerate(nucleus(sub, [params[i] for i in rows])):
        # torch.multinomial's seeded draw depends on vector length, even when the extra
        # probabilities are all zero. Keep the original full-vocabulary length and sorted
        # rank positions while avoiding a full sort of the logits on the candidate path.
        if ids is not None and probs.numel() < sub.shape[-1]:
            full = torch.zeros(sub.shape[-1], dtype=probs.dtype)
            full[:probs.numel()] = probs
            draw = full
        else:
            draw = probs
        pick = int(torch.multinomial(draw, 1, generator=generators[rows[j]]))
        out[rows[j]] = pick if ids is None else int(ids[pick])
    return out


def nucleus(scaled: torch.Tensor, params: list[SamplingParams],
            candidates: int = CANDIDATES) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    """Per row, the probabilities left after top-k and top-p (renormalized, 0 where cut) and
    the token ids they belong to; ids None means the probabilities are over the whole
    vocabulary in id order. `scaled` is already divided by the temperature."""
    vocab = scaled.shape[-1]
    k = min(vocab, candidates)
    top_k = torch.tensor([p.top_k if 0 < p.top_k < vocab else vocab for p in params])
    top_p = torch.tensor([p.top_p for p in params])
    # Ask for one extra value so a tie at the candidate boundary is visible. torch.topk and
    # torch.sort can order equal logits differently, which changes a seeded multinomial pick.
    ranked, ranked_idx = scaled.topk(min(vocab, k + 1), dim=-1)
    vals, idx = ranked[:, :k], ranked_idx[:, :k]
    tied = (ranked[:, 1:] == ranked[:, :-1]).any(dim=-1)
    kmask = torch.arange(k).unsqueeze(0) >= top_k.clamp(max=k).unsqueeze(1)
    cut = top_k <= k  # top-k ends inside the candidates: renormalize over the top-k
    within = vals.masked_fill(kmask, float("-inf")).softmax(dim=-1)
    whole = (vals - scaled.logsumexp(dim=-1, keepdim=True)).exp()  # share of the full vocab
    probs = torch.where(cut.unsqueeze(1), within, whole)
    # Keep the smallest prefix whose mass reaches top_p (the first token always survives).
    mask = kmask | ((probs.cumsum(dim=-1) - probs) >= top_p.unsqueeze(1))
    final = vals.masked_fill(mask, float("-inf")).softmax(dim=-1)
    # Without top-k, the nucleus ends inside the candidates if they hold top_p of the mass.
    fits = cut | ((top_p < 1.0) & (whole.sum(dim=-1) >= top_p))
    out: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    for j in range(len(params)):
        if tied[j] or (top_k[j] == vocab and top_p[j] >= 1.0):
            # Plain temperature sampling and ties need the original full-sort order to
            # preserve the exact token sequence for an existing seed.
            out.append(_full_sort(scaled[j], int(top_k[j]), float(top_p[j])))
        elif fits[j]:
            out.append((final[j], idx[j]))
        else:
            out.append(_full_sort(scaled[j], int(top_k[j]), float(top_p[j])))
    return out


def _full_sort(row: torch.Tensor, top_k: int, top_p: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k then top-p over the whole sorted vocabulary, for rows the candidates can't cover."""
    sorted_logits, sorted_idx = row.sort(descending=True)
    mask = torch.arange(row.shape[-1]) >= top_k
    probs = sorted_logits.masked_fill(mask, float("-inf")).softmax(dim=-1)
    mask |= (probs.cumsum(dim=-1) - probs) >= top_p
    return sorted_logits.masked_fill(mask, float("-inf")).softmax(dim=-1), sorted_idx
