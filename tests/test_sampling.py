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
