import pytest
import torch

from localhost_ai.engine.request import SamplingParams
from localhost_ai.engine.sampling import make_generator, sample


def test_greedy_rows_take_argmax():
    logits = torch.tensor([[0.1, 3.0, 0.2], [5.0, 0.0, 1.0]])
    ps = [SamplingParams(temperature=0.0)] * 2
    assert sample(logits, ps, [make_generator(0)] * 2) == [1, 0]


def test_top_k_one_is_greedy():
    logits = torch.randn(4, 50)
    ps = [SamplingParams(temperature=1.3, top_k=1)] * 4
    assert sample(logits, ps, [make_generator(i) for i in range(4)]) == logits.argmax(-1).tolist()


def test_top_p_restricts_support():
    logits = torch.tensor([[10.0, 9.0, -5.0, -5.0, -5.0]])
    p = [SamplingParams(temperature=1.0, top_p=0.9)]
    picks = {sample(logits, p, [make_generator(s)])[0] for s in range(200)}
    assert picks <= {0, 1}


def test_seeded_rows_reproduce_and_ignore_neighbours():
    logits = torch.randn(3, 30)
    ps = [SamplingParams(temperature=1.0, seed=s) for s in (1, 2, 3)]
    a = sample(logits, ps, [make_generator(p.seed) for p in ps])
    b = sample(logits[1:2], ps[1:2], [make_generator(2)])
    assert a[1] == b[0]
    assert a == sample(logits, ps, [make_generator(p.seed) for p in ps])


def test_mixed_batch():
    logits = torch.tensor([[0.0, 9.0], [9.0, 0.0]])
    ps = [SamplingParams(temperature=0.0), SamplingParams(temperature=0.5, top_k=1)]
    assert sample(logits, ps, [make_generator(0), make_generator(0)]) == [1, 0]


def reference_probs(scaled, top_k, top_p):
    """The original sampler: sort the whole row, top-k, then top-p; probabilities by token id."""
    sorted_logits, sorted_idx = scaled.sort(descending=True)
    vocab = scaled.shape[-1]
    mask = torch.arange(vocab) >= (top_k if top_k > 0 else vocab)
    probs = sorted_logits.masked_fill(mask, float("-inf")).softmax(-1)
    mask |= (probs.cumsum(-1) - probs) >= top_p
    probs = sorted_logits.masked_fill(mask, float("-inf")).softmax(-1)
    return torch.zeros(vocab).index_put_((sorted_idx,), probs)


def by_id(probs, ids, vocab):
    return probs if ids is None else torch.zeros(vocab).index_put_((ids,), probs)


@pytest.mark.parametrize("scale", [0.5, 3.0, 12.0])  # flat (falls back) to peaked (fits)
def test_nucleus_matches_the_full_sort(scale):
    from localhost_ai.engine.sampling import nucleus

    g = torch.Generator().manual_seed(int(scale * 10))
    vocab = 5000
    logits = torch.randn(12, vocab, generator=g) * scale
    grid = [(0, 0.9), (0, 0.5), (0, 0.99), (0, 1.0), (5, 1.0), (40, 0.8), (300, 0.95),
            (1, 0.3), (vocab + 7, 0.9), (0, 0.97), (128, 0.999), (0, 0.2)]
    params = [SamplingParams(temperature=0.8, top_k=k, top_p=p) for k, p in grid]
    got = nucleus(logits, params, candidates=64)
    for row, ((k, p), (probs, ids)) in enumerate(zip(grid, got, strict=True)):
        want = reference_probs(logits[row], k, p)
        assert torch.allclose(by_id(probs, ids, vocab), want, atol=1e-6), (scale, k, p)


@pytest.mark.parametrize("top_k,top_p", [(0, 1.0), (0, 0.9), (5, 1.0), (5, 0.9),
                                          (300, 0.95)])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("vocab,scale", [(300, 2.0), (1000, 3.0)])
def test_nucleus_preserves_full_sort_seeded_tokens(top_k, top_p, ties, vocab, scale):
    g = torch.Generator().manual_seed(481)
    logits = torch.randn(12, vocab, generator=g) * scale
    if ties:
        logits = logits.round()
    for seed in (0, 1, 7, 42):
        # the scheduler builds each row's generator from params.seed
        params = [SamplingParams(temperature=0.8, top_k=top_k, top_p=top_p, seed=seed + i)
                  for i in range(len(logits))]
        got = sample(logits, params, [make_generator(seed + i) for i in range(len(logits))])
        want = []
        for i, row in enumerate(logits):
            values, ids = (row / params[i].temperature).sort(descending=True)
            rank = torch.arange(len(values))
            mask = rank >= (top_k if top_k else len(values))
            probs = values.masked_fill(mask, float("-inf")).softmax(-1)
            mask |= (probs.cumsum(-1) - probs) >= top_p
            probs = values.masked_fill(mask, float("-inf")).softmax(-1)
            pick = torch.multinomial(probs, 1, generator=make_generator(seed + i))
            want.append(int(ids[pick]))
        assert got == want, (top_k, top_p, ties, seed)


def old_sampler(logits, params, generators):
    """The sampler before the candidate set, verbatim from d2717ca: sort the whole batch."""
    logits = logits.detach().to("cpu", torch.float32)
    out = logits.argmax(dim=-1).tolist()
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
    cum_before = probs.cumsum(dim=-1) - probs
    mask |= cum_before >= top_p
    probs = sorted_logits.masked_fill(mask, float("-inf")).softmax(dim=-1)
    for j, i in enumerate(rows):
        pick = torch.multinomial(probs[j], 1, generator=generators[i])
        out[i] = int(sorted_idx[j, pick])
    return out


def lm_like(g, rows, vocab, peak):
    """fp16-valued logits (so exact ties are common, as with 4-bit models) with a few strong
    candidates on a noisy floor."""
    x = torch.randn(rows, vocab, generator=g) * 2.0
    top = torch.randint(0, vocab, (rows, 24), generator=g)
    x.scatter_(1, top, torch.randn(rows, 24, generator=g) * 2.0 + peak)
    return x.half().float()


SEEDED = [dict(temperature=0.7, top_p=0.95), dict(temperature=1.0, top_p=0.9),
          dict(temperature=0.7, top_p=0.5), dict(temperature=1.2, top_p=1.0, top_k=40),
          dict(temperature=0.8, top_p=0.95, top_k=1), dict(temperature=0.6, top_p=0.9, top_k=300),
          dict(temperature=1.0, top_p=1.0), dict(temperature=0.7, top_p=0.99, top_k=250)]


@pytest.mark.parametrize("peak", [4.0, 9.0, 14.0])  # flat (mostly falls back) to peaked
@pytest.mark.parametrize("masked", [False, True])  # response_format-style -inf masks
def test_seeded_sequences_match_the_old_full_sort_sampler(peak, masked, monkeypatch):
    from localhost_ai.engine import sampling

    sorted_rows = []
    orig = sampling._full_sort
    monkeypatch.setattr(sampling, "_full_sort",
                        lambda r, k, p: sorted_rows.append(len(r)) or orig(r, k, p))
    g = torch.Generator().manual_seed(int(peak * 10) + masked)
    vocab, steps = 32000, 24
    params = [SamplingParams(seed=100 + i, **kw) for i, kw in enumerate(SEEDED)]
    # unseeded neighbours in the same batch must not disturb the seeded rows
    params += [SamplingParams(temperature=0.9, top_p=0.9), SamplingParams(temperature=0.0)]
    old_g = [make_generator(p.seed) for p in params]
    new_g = [make_generator(p.seed) for p in params]
    old, new = [], []
    for _ in range(steps):
        logits = lm_like(g, len(params), vocab, peak)
        if masked:
            logits[::2, : vocab // 2] = float("-inf")
        old.append(old_sampler(logits, params, old_g)[:len(SEEDED)])
        new.append(sample(logits, params, new_g)[:len(SEEDED)])
    assert new == old
    # Two seeded configs always sort (top_k past the candidates; top_p 1 with no top_k). The
    # rest must mostly take the candidate path on peaked logits, or this proves little.
    by_design = 2 * steps
    if peak >= 14.0:
        assert sum(sorted_rows) - by_design < 0.25 * steps * (len(params) - 1 - 2)
