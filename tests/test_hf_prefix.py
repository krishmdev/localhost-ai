"""Prefix caching on the torch runner with a tiny random-weight Llama built in-process (CPU,
fp32, no download): rows that start from a stored prefix give the same logits as a plain
prefill of the whole prompt, through decode steps, a join and a row selection."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from localhost_ai.engine.prefix import PrefixCache
from localhost_ai.engine.runner import HFModelRunner

SYSTEM = list(range(40, 60))
SUFFIXES = [[70, 71, 72], [73], [74, 75, 76, 77, 78, 79], [80, 81]]
FORCED = [[5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=256)
    return LlamaForCausalLM(cfg).eval()


def runner(model, prefix=None):
    return HFModelRunner(model, pad_id=0, eos_ids=frozenset({1}), device=torch.device("cpu"),
                         dtype=torch.float32, prefix_cache=prefix)


def alone(r, prompt, forced):
    state, logits = r.prefill([prompt])
    out = [logits[0]]
    for t in forced:
        out.append(r.decode(state, [t])[0])
    return out


def close(a, b):
    assert torch.allclose(a, b, atol=1e-4), f"max diff {(a - b).abs().max():.2e}"


def test_prefix_rows_match_plain_prefill(model):
    plain = runner(model)
    pc = PrefixCache(budget_bytes=1 << 30, min_tokens=8)
    cached = runner(model, pc)
    prompts = [SYSTEM + s for s in SUFFIXES]
    ref = [alone(plain, p, f) for p, f in zip(prompts, FORCED, strict=True)]

    state, logits = cached.prefill(prompts)  # row 0 misses, rows 1-3 hit the new entry
    assert list(pc.entries) == [tuple(SYSTEM)] and pc.hits == 3
    steps = [logits] + [cached.decode(state, [f[i] for f in FORCED]) for i in range(3)]
    for i, got in enumerate(steps):
        for row in range(len(prompts)):
            close(got[row], ref[row][i])


def test_prefix_rows_join_and_leave(model):
    plain = runner(model)
    pc = PrefixCache(budget_bytes=1 << 30, min_tokens=8)
    cached = runner(model, pc)
    prompts = [SYSTEM + s for s in SUFFIXES]
    ref = [alone(plain, p, f) for p, f in zip(prompts, FORCED, strict=True)]
    first, _ = cached.prefill(prompts[:2])
    cached.decode(first, [FORCED[0][0], FORCED[1][0]])
    late, _ = cached.prefill(prompts[2:])  # both hit
    merged = cached.merge(first, late)
    got = cached.decode(merged, [FORCED[0][1], FORCED[1][1], FORCED[2][0], FORCED[3][0]])
    for row, step in enumerate([2, 2, 1, 1]):
        close(got[row], ref[row][step])
    kept = cached.select(merged, [1, 3])
    got = cached.decode(kept, [FORCED[1][2], FORCED[3][1]])
    close(got[0], ref[1][3])
    close(got[1], ref[3][2])


def test_drop_prefixes(model):
    pc = PrefixCache(budget_bytes=1 << 30, min_tokens=8)
    cached = runner(model, pc)
    cached.prefill([SYSTEM + s for s in SUFFIXES[:2]])
    # 20 tokens x 2 layers x (K + V) x 2 kv heads x head_dim 16 x fp32
    size = pc.nbytes
    assert size == 20 * 2 * 2 * 2 * 16 * 4
    assert cached.drop_prefixes() == size and not pc.entries
