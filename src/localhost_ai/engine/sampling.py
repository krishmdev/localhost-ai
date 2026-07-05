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
# CPU took 10-13 ms per row per step (Qwen2.5-3B), about as long as the model's decode step, so
# the nucleus is looked for among the top CANDIDATES logits first.
#
# Seeded rows must give the same tokens as the full-sort sampler this replaced (d2717ca), for
# the same seed. That holds when (1) the nucleus ends inside the candidates, (2) no two logits
# in the nucleus, or at its edge, are equal (torch.sort orders ties differently from
# torch.topk, which would put a different token at a position), and (3) no cumulative sum is
# within top_p_tol(vocab) of top_p (the candidate sums are normalized by logsumexp, the full
# sort's by a softmax over the sorted row, and the two round apart by more as the vocabulary
# grows, so a token right at the cut could flip). The draw then uses the very vector the full
# sort built: the kept logits in rank order, -inf to the full vocabulary length, softmax,
# multinomial, which also consumes the row's generator exactly as before. Rows that fail a
# check are sorted in full. Unseeded rows have no sequence to reproduce, so they draw from the
# candidates directly, which is the same distribution and much cheaper.
CANDIDATES = 256


def top_p_tol(vocab: int) -> float:
    """How close a cumulative sum may come to top_p before a seeded row is sorted in full.
    Measured drift between the two sums on fp16-valued logits (tests/test_sampling.py): up to
    about 2e-4 at 151,936 tokens and 3.4e-4 at 262,144, so this keeps roughly 3x headroom."""
    return max(1e-4, 4e-9 * vocab)


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
        pick = int(torch.multinomial(probs, 1, generator=generators[rows[j]]))
        out[rows[j]] = pick if ids is None else int(ids[pick])
    return out


def nucleus(scaled: torch.Tensor, params: list[SamplingParams],
            candidates: int = CANDIDATES) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    """Per row, the probabilities left after top-k and top-p (renormalized, 0 where cut) and
    the token id at each position; ids None means the probabilities are in id order. `scaled`
    is already divided by the temperature. Drawing position `torch.multinomial(probs, 1)` with
    the row's generator gives the pick (see the note on CANDIDATES for seeded rows)."""
    n, vocab = scaled.shape
    tol = top_p_tol(vocab)
    k = min(vocab, candidates)
    top_k = torch.tensor([p.top_k if 0 < p.top_k < vocab else vocab for p in params])
    top_p = torch.tensor([p.top_p for p in params])
    seeded = [p.seed is not None for p in params]
    ranked, ranked_idx = scaled.topk(min(vocab, k + 1), dim=-1)  # sorted, descending
    vals, idx = ranked[:, :k], ranked_idx[:, :k]
    pos = torch.arange(k).unsqueeze(0)
    kmask = pos >= top_k.clamp(max=k).unsqueeze(1)
    cut = top_k <= k  # top-k ends inside the candidates: renormalize over the top-k
    within = vals.masked_fill(kmask, float("-inf")).softmax(dim=-1)
    whole = (vals - scaled.logsumexp(dim=-1, keepdim=True)).exp()  # share of the full vocab
    probs = torch.where(cut.unsqueeze(1), within, whole)
    # Keep the smallest prefix whose mass reaches top_p (the first token always survives).
    cum_before = probs.cumsum(dim=-1) - probs
    mask = kmask | (cum_before >= top_p.unsqueeze(1))
    final = vals.masked_fill(mask, float("-inf")).softmax(dim=-1)
    kept = (~mask).sum(dim=-1)
    # The nucleus ends inside the candidates: top-k cuts there, or (no top-k) they carry top_p.
    fits = cut | ((top_k == vocab) & (top_p < 1.0) & (whole.sum(dim=-1) >= top_p + tol))
    # Only for seeded rows: equal logits at positions 0..kept (the nucleus and its first
    # excluded neighbour), and cumulative sums too close to top_p to be sure of the cut.
    if len(ranked[0]) > k:
        eq = ranked[:, 1:] == ranked[:, :-1]
    else:  # the candidates are the whole vocabulary; nothing past the last one
        eq = torch.cat([ranked[:, 1:] == ranked[:, :-1], torch.zeros(n, 1, dtype=torch.bool)], 1)
    tied = (eq & (pos < kept.unsqueeze(1))).any(dim=-1)
    near = ((cum_before - top_p.unsqueeze(1)).abs() <= tol).any(dim=-1)

    out: list[tuple[torch.Tensor, torch.Tensor | None] | None] = [None] * n
    full_sort: list[int] = []
    for j in range(n):
        if not seeded[j]:
            if fits[j]:
                out[j] = (final[j], idx[j])
            elif top_k[j] == vocab and top_p[j] >= 1.0:
                out[j] = (scaled[j].softmax(dim=-1), None)  # plain temperature sampling
            else:
                full_sort.append(j)
        elif fits[j] and not tied[j] and not near[j]:
            out[j] = None  # filled below, as the full sort would have built it
        else:
            full_sort.append(j)
    exact = [j for j in range(n) if seeded[j] and out[j] is None and j not in full_sort]
    if exact:
        # The full sort's final vector: kept logits in rank order, -inf to full length.
        rows = torch.full((len(exact), vocab), float("-inf"))
        rows[:, :k] = vals[exact].masked_fill(mask[exact], float("-inf"))
        drawn = rows.softmax(dim=-1)
        for r, j in enumerate(exact):
            out[j] = (drawn[r], idx[j])
    if full_sort:
        for j, pair in zip(full_sort, _full_sort(scaled[full_sort], top_k[full_sort],
                                                 top_p[full_sort]), strict=True):
            out[j] = pair
    return out  # type: ignore[return-value]


def _full_sort(rows: torch.Tensor, top_k: torch.Tensor,
               top_p: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Top-k then top-p over the whole sorted vocabulary, batched, exactly as the sampler
    before the candidate set did it."""
    sorted_logits, sorted_idx = rows.sort(dim=-1, descending=True)
    ranks = torch.arange(rows.shape[-1]).unsqueeze(0)
    mask = ranks >= top_k.unsqueeze(1)
    probs = sorted_logits.masked_fill(mask, float("-inf")).softmax(dim=-1)
    mask |= (probs.cumsum(dim=-1) - probs) >= top_p.unsqueeze(1)
    probs = sorted_logits.masked_fill(mask, float("-inf")).softmax(dim=-1)
    return [(probs[j], sorted_idx[j]) for j in range(len(rows))]
