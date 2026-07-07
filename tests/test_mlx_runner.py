"""MLXModelRunner against tiny random-weight mlx-lm models built in-process: no download, but
it needs the mlx extra (Apple silicon), so it skips elsewhere.

Rows are compared under teacher forcing (every row is fed the same next tokens batched and
alone) rather than by greedy decoding, so near-tied logits in a random model can't make the
test flaky."""

import numpy as np
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
    """Reference logits for one sequence, independent of the runner and of mlx-lm's batch
    caches: the model's own single-sequence caches (make_prompt_cache), fed one token at a
    time. Entry 0 is after the prompt, then one entry after each forced token."""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(r.model)
    out = []
    for i, t in enumerate(prompt + forced):
        logits = r.model(mx.array([[t]], dtype=mx.int32), cache=cache)[:, -1, :]
        if i >= len(prompt) - 1:
            out.append(torch.from_numpy(np.array(logits.astype(mx.float32)))[0])
    return out


def test_runner_alone_matches_the_plain_cache_reference(runner, hybrid, windowed):
    for r in (runner, hybrid, windowed):
        for p, f in zip(PROMPTS, FORCED, strict=True):
            state, logits = r.prefill([p])
            got = [logits[0]] + [r.decode(state, [t])[0] for t in f]
            for a, b in zip(got, alone(r, p, f), strict=True):
                assert_close(a, b)


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


def tiny_qwen3_5():
    """Qwen3.5's text stack: gated-delta linear attention (ArraysCache: conv + recurrent state)
    every layer except each second one, which is full attention (KVCache)."""
    from mlx_lm.models import qwen3_5

    mx.random.seed(0)
    args = qwen3_5.TextModelArgs(model_type="qwen3_5_text", hidden_size=64, intermediate_size=128,
                                 num_hidden_layers=4, num_attention_heads=4,
                                 num_key_value_heads=2, head_dim=16, vocab_size=VOCAB,
                                 linear_num_value_heads=4, linear_num_key_heads=2,
                                 linear_key_head_dim=16, linear_value_head_dim=16,
                                 full_attention_interval=2)
    model = qwen3_5.TextModel(args)
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def hybrid():
    return MLXModelRunner(tiny_qwen3_5(), eos_ids=frozenset({0}))


def test_hybrid_caches_leave_recurrent_left_padding_unset(hybrid):
    from mlx_lm.models.cache import ArraysCache, BatchKVCache

    from localhost_ai.engine.mlx_runner import batch_cache

    cache = batch_cache(hybrid.model, 3)
    assert [type(c) for c in cache] == [ArraysCache, BatchKVCache] * 2
    assert all(c.left_padding is None for c in cache if isinstance(c, ArraysCache))


def test_hybrid_right_padded_prefill_keeps_pads_out_of_recurrent_state(hybrid):
    # Regression: with a zero left_padding on the recurrent caches, ArraysCache.make_mask
    # ignored the prefill lengths and the pad tokens after each shorter prompt ran through the
    # gated-delta state, so every row but the longest came out wrong (max logit diff 0.07).
    ref = [alone(hybrid, p, f) for p, f in zip(PROMPTS, FORCED, strict=True)]
    state, logits = hybrid.prefill(PROMPTS)
    steps = [logits]
    for i in range(2):
        steps.append(hybrid.decode(state, [f[i] for f in FORCED]))
    for step, got in enumerate(steps):
        for row in range(len(PROMPTS)):
            assert_close(got[row], ref[row][step])


def test_hybrid_join_and_select_match_alone(hybrid):
    ref = [alone(hybrid, p, f) for p, f in zip(PROMPTS, FORCED, strict=True)]
    first, _ = hybrid.prefill(PROMPTS[:2])
    hybrid.decode(first, [FORCED[0][0], FORCED[1][0]])
    late, _ = hybrid.prefill(PROMPTS[2:])
    merged = hybrid.merge(first, late)
    got = hybrid.decode(merged, [FORCED[0][1], FORCED[1][1], FORCED[2][0], FORCED[3][0]])
    for row, step in enumerate([2, 2, 1, 1]):
        assert_close(got[row], ref[row][step])
    kept = hybrid.select(merged, [0, 3])
    got = hybrid.decode(kept, [FORCED[0][2], FORCED[3][1]])
    assert_close(got[0], ref[0][3])
    assert_close(got[1], ref[3][2])


def test_hybrid_reports_fixed_recurrent_state_per_row(hybrid):
    # 2 linear-attention layers, each: fp32 delta state 4 v-heads x 16 x 16, plus the conv
    # state (kernel 4 - 1) x conv_dim (2 x 2 x 16 + 4 x 16 = 128) in the activation dtype
    delta = 4 * 16 * 16 * 4
    conv = 3 * 128 * 4
    assert hybrid.row_state_bytes == 2 * (delta + conv)
    assert hybrid.kv_bytes_per_token == 2 * 2 * 2 * 16 * 4  # the 2 full-attention layers


def tiny_gemma4():
    """Gemma 4's text stack: sliding-window layers (BatchRotatingKVCache) with a full-attention
    layer every third, and the last two layers reuse earlier layers' KV (no cache of their own).
    The window is 4 tokens, so the longest prompt and the decode steps both wrap it."""
    from mlx_lm.models import gemma4_text

    mx.random.seed(0)
    args = gemma4_text.ModelArgs(hidden_size=64, num_hidden_layers=6, intermediate_size=128,
                                 num_attention_heads=4, head_dim=16, global_head_dim=32,
                                 num_key_value_heads=2, num_global_key_value_heads=2,
                                 num_kv_shared_layers=2, hidden_size_per_layer_input=8,
                                 vocab_size=VOCAB, vocab_size_per_layer_input=VOCAB,
                                 sliding_window=4, sliding_window_pattern=3,
                                 use_double_wide_mlp=False)
    model = gemma4_text.Model(args)
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def windowed():
    return MLXModelRunner(tiny_gemma4(), eos_ids=frozenset({0}))


def test_windowed_caches_and_sizes(windowed):
    from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache

    from localhost_ai.engine.mlx_runner import batch_cache

    cache = batch_cache(windowed.model, 2)
    assert [type(c) for c in cache] == [BatchRotatingKVCache] * 2 + [BatchKVCache,
                                                                      BatchRotatingKVCache]
    # 3 sliding layers (2 kv heads x 16) + 1 full layer (2 kv heads x 32), K and V, fp32; the
    # two shared layers add nothing
    assert windowed.kv_bytes_per_token == 3 * 2 * 2 * 16 * 4 + 2 * 2 * 32 * 4
    assert windowed.row_state_bytes == 0


def test_windowed_batch_past_the_window_matches_alone(windowed):
    ref = [alone(windowed, p, f) for p, f in zip(PROMPTS, FORCED, strict=True)]
    state, logits = windowed.prefill(PROMPTS)
    steps = [logits]
    for i in range(len(FORCED[0])):
        steps.append(windowed.decode(state, [f[i] for f in FORCED]))
    for step, got in enumerate(steps):
        for row in range(len(PROMPTS)):
            assert_close(got[row], ref[row][step])


def test_windowed_join_and_select_match_alone(windowed):
    ref = [alone(windowed, p, f) for p, f in zip(PROMPTS, FORCED, strict=True)]
    first, _ = windowed.prefill(PROMPTS[:2])
    for i in range(2):
        windowed.decode(first, [FORCED[0][i], FORCED[1][i]])
    late, _ = windowed.prefill(PROMPTS[2:])
    merged = windowed.merge(first, late)
    got = windowed.decode(merged, [FORCED[0][2], FORCED[1][2], FORCED[2][0], FORCED[3][0]])
    for row, step in enumerate([3, 3, 1, 1]):
        assert_close(got[row], ref[row][step])
    kept = windowed.select(merged, [1, 2])
    got = windowed.decode(kept, [FORCED[1][3], FORCED[2][1]])
    assert_close(got[0], ref[1][4])
    assert_close(got[1], ref[2][2])


@pytest.mark.parametrize("step", [2, 3, 5])
@pytest.mark.parametrize("model", ["llama", "hybrid", "windowed"])
def test_chunked_prefill_matches_alone(model, step, request):
    # Every architecture, with chunk sizes that leave a 1-token last chunk for some head widths
    # (the 9-token prompt has an 8-token head: 8 % 3 == 2, 8 % 5 == 3, 5 % 2 == 1 for the batch
    # of the shorter prompts below). Sliding-window caches used to fail on a 1-token chunk while
    # the right padding was still active.
    base = request.getfixturevalue({"llama": "runner", "hybrid": "hybrid",
                                    "windowed": "windowed"}[model])
    chunked = MLXModelRunner(base.model, eos_ids=frozenset({0}), prefill_step=step)
    for prompts, forced in ((PROMPTS, FORCED), (PROMPTS[1:], FORCED[1:])):
        ref = [alone(base, p, f) for p, f in zip(prompts, forced, strict=True)]
        state, logits = chunked.prefill(prompts)
        steps = [logits, chunked.decode(state, [f[0] for f in forced])]
        for i, got in enumerate(steps):
            for row in range(len(prompts)):
                assert_close(got[row], ref[row][i])


@pytest.mark.parametrize("model", ["llama", "hybrid", "windowed"])
def test_one_token_head_with_padding(model, request):
    # head widths 1 and 0: the whole padded prefill is a single token
    r = request.getfixturevalue({"llama": "runner", "hybrid": "hybrid",
                                 "windowed": "windowed"}[model])
    ref = [alone(r, p, [9]) for p in ([5, 6], [7])]
    state, logits = r.prefill([[5, 6], [7]])
    got = [logits, r.decode(state, [9, 9])]
    for i in range(2):
        for row in range(2):
            assert_close(got[i][row], ref[row][i])


def test_row_bytes_follows_the_cache_layout(runner, hybrid, windowed):
    # KV buffers grow in 256-token steps, so a 5-token row already holds 256 tokens of KV
    assert runner.row_bytes(5) == runner.row_bytes(256) == 256 * runner.kv_bytes_per_token
    assert runner.row_bytes(257) == 512 * runner.kv_bytes_per_token
    # recurrent state is a fixed cost on top
    assert hybrid.row_bytes(5) == 256 * hybrid.kv_bytes_per_token + hybrid.row_state_bytes
    # sliding-window layers stop at the window (4 tokens here); only the full layer grows
    sliding, full = 3 * 2 * 2 * 16 * 4, 2 * 2 * 32 * 4
    assert windowed.row_bytes(5) == 4 * sliding + 256 * full
    assert windowed.row_bytes(2000) == 4 * sliding + 2048 * full


@pytest.mark.parametrize("model", ["llama", "hybrid", "windowed"])
def test_kv_bytes_reads_the_live_cache(model, request):
    r = request.getfixturevalue({"llama": "runner", "hybrid": "hybrid",
                                 "windowed": "windowed"}[model])
    state, _ = r.prefill(PROMPTS)
    assert r.kv_bytes(state) == len(PROMPTS) * r.row_bytes(max(map(len, PROMPTS)))
    kept = r.select(state, [1, 2])
    assert r.kv_bytes(kept) <= 2 * r.row_bytes(max(map(len, PROMPTS)))


SYSTEM = [40, 41, 42, 43, 44, 45, 46]  # a shared "system prompt", longer than Gemma's window
SUFFIXES = [[50, 51, 52], [53], [54, 55, 56, 57, 58, 59], [60, 61]]


@pytest.mark.parametrize("step", [3, 512])
@pytest.mark.parametrize("model", ["llama", "hybrid", "windowed"])
def test_prefix_cache_matches_alone(model, step, request):
    from localhost_ai.engine.prefix import PrefixCache

    base = request.getfixturevalue({"llama": "runner", "hybrid": "hybrid",
                                    "windowed": "windowed"}[model])
    pc = PrefixCache(budget_bytes=1 << 30, min_tokens=4)
    r = MLXModelRunner(base.model, eos_ids=frozenset({0}), prefill_step=step, prefix_cache=pc)
    prompts = [SYSTEM + s for s in SUFFIXES]
    ref = [alone(base, p, f) for p, f in zip(prompts, FORCED, strict=True)]

    # first batch: the second row finds the shared prefix with the first and stores it, so the
    # batch is prefilled as two groups (no prefix / cached prefix) and merged back in order
    state, logits = r.prefill(prompts)
    assert list(pc.entries) == [tuple(SYSTEM)] and pc.misses == 1 and pc.hits == 3
    steps = [logits] + [r.decode(state, [f[i] for f in FORCED]) for i in range(len(FORCED[0]))]
    for i, got in enumerate(steps):
        for row in range(len(prompts)):
            assert_close(got[row], ref[row][i])

    # a later batch hits the stored prefix for every row
    state, logits = r.prefill(prompts[::-1])
    assert pc.hits == 7 and pc.misses == 1
    got = r.decode(state, [f[0] for f in FORCED[::-1]])
    for row, orig in enumerate(range(len(prompts) - 1, -1, -1)):
        assert_close(logits[row], ref[orig][0])
        assert_close(got[row], ref[orig][1])


def test_prefix_cache_budget_and_drop(runner):
    from localhost_ai.engine.prefix import PrefixCache

    pc = PrefixCache(budget_bytes=1 << 30, min_tokens=4)
    r = MLXModelRunner(runner.model, eos_ids=frozenset({0}), prefix_cache=pc)
    r.prefill([SYSTEM + s for s in SUFFIXES[:2]])
    size = pc.nbytes
    assert size > 0 and size == pc.entries[tuple(SYSTEM)].nbytes
    assert r.drop_prefixes() == size and pc.nbytes == 0 and not pc.entries
    small = PrefixCache(budget_bytes=size - 1, min_tokens=4)  # an entry that can't fit
    r2 = MLXModelRunner(runner.model, eos_ids=frozenset({0}), prefix_cache=small)
    r2.prefill([SYSTEM + s for s in SUFFIXES[:2]])
    assert not small.entries and small.misses == 2


@pytest.mark.parametrize("name", ["llama", "hybrid", "windowed"])
def test_logprob_rows_matches_a_plain_forward(request, name):
    """/v1/score's forward: the rows asked for, log-softmaxed, equal the model's own full
    forward (no cache) at the same positions, for plain, recurrent and sliding-window layers."""
    r = request.getfixturevalue({"llama": "runner", "hybrid": "hybrid",
                                 "windowed": "windowed"}[name])
    ids = PROMPTS[0] + FORCED[0] + PROMPTS[3]
    rows = [0, 5, len(ids) - 1]
    logits = r.model(mx.array([ids], dtype=mx.int32))[0].astype(mx.float32)
    ref = torch.from_numpy(np.array(logits - mx.logsumexp(logits, axis=-1, keepdims=True)))
    got = r.logprob_rows(ids, rows)
    assert got.dtype == torch.float32 and got.shape == (3, ref.shape[1])
    assert torch.allclose(got, ref[rows], atol=1e-5)
