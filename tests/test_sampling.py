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
