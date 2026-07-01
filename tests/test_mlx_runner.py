"""MLXModelRunner against tiny random-weight mlx-lm models built in-process: no download, but
it needs the mlx extra (Apple silicon), so it skips elsewhere.

Rows are compared under teacher forcing (every row is fed the same next tokens batched and
alone) rather than by greedy decoding, so near-tied logits in a random model can't make the
test flaky."""

import pytest
import torch

mx = pytest.importorskip("mlx.core")
llama = pytest.importorskip("mlx_lm.models.llama")

from localhost_ai.engine.mlx_runner import CacheLost, MLXModelRunner  # noqa: E402
from localhost_ai.engine.runner import is_oom  # noqa: E402

VOCAB = 128
PROMPTS = [[3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14], [15], [16, 17, 18, 19, 20]]
FORCED = [[21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]]
ATOL = 1e-4


def tiny_llama():
    mx.random.seed(0)
    args = llama.ModelArgs(model_type="llama", hidden_size=64, num_hidden_layers=2,
                           intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                           rms_norm_eps=1e-5, vocab_size=VOCAB)
    model = llama.Model(args)
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def runner():
    return MLXModelRunner(tiny_llama(), eos_ids=frozenset({0}))


def alone(r, prompt, forced):
    """Logits of one sequence run by itself: after prefill, then after each forced token."""
    state, logits = r.prefill([prompt])
    out = [logits[0]]
    for t in forced:
        out.append(r.decode(state, [t])[0])
    return out


def assert_close(a, b):
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=ATOL), f"max diff {(a - b).abs().max():.2e}"


def test_measures_kv_bytes_from_the_live_cache(runner):
    # 2 layers x (K + V) x 2 kv heads x head_dim 16 x fp32
    assert runner.kv_bytes_per_token == 2 * 2 * 2 * 16 * 4
    assert runner.row_state_bytes == 0  # no recurrent layers


def test_logits_come_back_as_cpu_float32(runner):
    _, logits = runner.prefill([[1, 2, 3]])
    assert logits.dtype == torch.float32 and logits.device.type == "cpu"
    assert logits.shape == (1, VOCAB)


def test_mixed_length_batch_matches_each_row_alone(runner):
    ref = [alone(runner, p, f) for p, f in zip(PROMPTS, FORCED, strict=True)]
    state, logits = runner.prefill(PROMPTS)
    steps = [logits]
    for i in range(len(FORCED[0])):
        steps.append(runner.decode(state, [f[i] for f in FORCED]))
    for step, got in enumerate(steps):
        for row in range(len(PROMPTS)):
            assert_close(got[row], ref[row][step])
    assert state.lengths == [len(p) + len(FORCED[0]) for p in PROMPTS]


def test_rows_joining_mid_stream_match_alone(runner):
    ref = [alone(runner, p, f) for p, f in zip(PROMPTS, FORCED, strict=True)]
    first, _ = runner.prefill(PROMPTS[:2])
    runner.decode(first, [FORCED[0][0], FORCED[1][0]])
    runner.decode(first, [FORCED[0][1], FORCED[1][1]])
    late, late_logits = runner.prefill(PROMPTS[2:])
    for row in range(2):
        assert_close(late_logits[row], ref[2 + row][0])
    merged = runner.merge(first, late)
    got = runner.decode(merged, [FORCED[0][2], FORCED[1][2], FORCED[2][0], FORCED[3][0]])
    assert_close(got[0], ref[0][3])
    assert_close(got[1], ref[1][3])
    assert_close(got[2], ref[2][1])
    assert_close(got[3], ref[3][1])


def test_select_keeps_the_right_rows(runner):
    ref = [alone(runner, p, f) for p, f in zip(PROMPTS, FORCED, strict=True)]
    state, _ = runner.prefill(PROMPTS)
    runner.decode(state, [f[0] for f in FORCED])
    kept = runner.select(state, [1, 3])  # drops the longest prompt, so padding shrinks too
    assert kept.lengths == [len(PROMPTS[1]) + 1, len(PROMPTS[3]) + 1]
    assert runner.padded_tokens(kept) == 2 * (len(PROMPTS[3]) + 1)
    got = runner.decode(kept, [FORCED[1][1], FORCED[3][1]])
    assert_close(got[0], ref[1][2])
    assert_close(got[1], ref[3][2])


def test_failed_step_loses_the_cache_and_reads_as_oom(runner, monkeypatch):
    state, logits = runner.prefill(PROMPTS[:2])
    other, _ = runner.prefill(PROMPTS[2:3])

    def boom(cache, tokens):
        raise RuntimeError("[metal::malloc] Attempting to allocate 1 bytes")

    monkeypatch.setattr(runner, "_forward", boom)
    with pytest.raises(RuntimeError):
        runner.decode(state, [1, 2])
    monkeypatch.undo()
    assert state.lost
    # some layers may have advanced before the failure, so nothing may touch this cache again;
    # the scheduler sees an OOM and recomputes the batch
    for call in (lambda: runner.decode(state, [1, 2]), lambda: runner.select(state, [0]),
                 lambda: runner.merge(state, other)):
        with pytest.raises(CacheLost) as exc:
            call()
        assert is_oom(exc.value)
