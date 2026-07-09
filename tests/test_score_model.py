"""/v1/score's scoring on the real 4-bit MLX checkpoints, against a direct mlx-lm forward that
follows a client-side MLX scorer step by step (joint tokenization, in-place reads, back-off and
teacher forcing inside a token). Presets that aren't downloaded are skipped.

    uv run pytest -q -m mlx_model tests/test_score_model.py
    LHAI_MLX_PRESETS=qwen2.5-3b-mlx4,gemma-4-e4b-mlx4 uv run pytest -q -m mlx_model \
        tests/test_score_model.py"""

import os

import numpy as np
import pytest

from localhost_ai.config import get_settings
from localhost_ai.engine.score import Site, score

pytestmark = pytest.mark.mlx_model

PRESETS = os.environ.get("LHAI_MLX_PRESETS", "qwen2.5-0.5b-mlx4").split(",")
TOL = 1e-4
MSG = [{"role": "system", "content": "Cite sources as [n]."},
       {"role": "user", "content": "Sources: [1] Paris guide. [2] Rome guide. [3] Oslo guide. "
                                   "Where is the Eiffel Tower?"}]
CONT = "The Eiffel Tower is in Paris [1], not Rome [2]."


@pytest.fixture(scope="module", params=PRESETS)
def loaded(request):
    pytest.importorskip("mlx_lm")
    import torch

    from localhost_ai.device import DeviceConfig
    from localhost_ai.models.loader import load
    from localhost_ai.models.registry import Registry, local_path

    s = get_settings()
    spec = Registry(s.models_file).get(request.param)
    try:
        local_path(spec, s.models_dir)
    except Exception as exc:  # noqa: BLE001 - only a missing snapshot skips
        pytest.skip(f"{request.param} not downloaded ({type(exc).__name__})")
    m = load(spec, DeviceConfig(torch.device("mps"), torch.float16, threads=1), s.models_dir)
    yield m
    m.runner.release()


def mlx_scorer_reference(m, messages, continuation, sites):
    """A client-side MLX scorer's score(), on the same weights, with mlx-lm's tokenizer wrapper."""
    import mlx.core as mx
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    tok = load_tokenizer(m.path)
    enc = tok._tokenizer

    def logprobs(ids, rows):
        """A plain forward over the whole sequence. Only `rows` are cast to float32 and kept (the
        cast and the log-softmax are per row), and the full logits are freed before returning:
        at 1,470 tokens they are 0.77 GB in bf16 for Gemma's 262k vocabulary, and holding them in
        float32 ran the 16 GB GPU out of memory."""
        logits = m.runner.model(mx.array([ids]))[0]
        sel = logits[mx.array(rows)].astype(mx.float32)
        out = sel - mx.logsumexp(sel, axis=-1, keepdims=True)
        mx.eval(out)
        del logits, sel
        out = np.array(out)
        mx.clear_cache()
        return out

    def forced(head_ids, piece, cand):
        a = enc(piece + cand, add_special_tokens=False)["input_ids"]
        b = enc(piece, add_special_tokens=False)["input_ids"] if piece else []

        def seq(tail):
            if not tail:
                return 0.0
            n = len(head_ids)
            lp = logprobs(head_ids + tail, [n + i - 1 for i in range(len(tail))])
            return float(sum(float(lp[i, t]) for i, t in enumerate(tail)))

        return seq(a) - seq(b)

    # mlx-lm's wrapper turns enable_thinking on by default for models that can think (Gemma 4,
    # Qwen3.5); the server's default is the preset's, which is off, so say so explicitly
    kwargs = {"enable_thinking": False, **(m.spec.chat_template_kwargs or {})}
    head = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                   **kwargs)
    full = head + continuation
    e = enc(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = e["input_ids"], e["offset_mapping"]
    at = [next(i for i, (_, en) in enumerate(offs) if en > len(head) + s.char_offset)
          for s in sites]
    reads = sorted({t - 1 for t in at if t > 0})
    lp_all = dict(zip(reads, logprobs(ids, reads), strict=True)) if reads else {}
    out = []
    for s, t in zip(sites, at, strict=True):
        a = len(head) + s.char_offset
        start = offs[t][0]
        lp = {}
        for c in s.candidates:
            c_ids = enc(c, add_special_tokens=False)["input_ids"]
            if start == a and len(c_ids) == 1 and t > 0:
                lp[c] = float(lp_all[t - 1][c_ids[0]])
            else:
                lp[c] = forced(ids[:t], full[start:a], c)
        out.append((t, lp))
    return out


def sites():
    digits = [Site(CONT.index("[1") + 1, ("1", "2", "3")),
              Site(CONT.index("[2") + 1, ("1", "2", "3"))]
    inside = Site(CONT.index("Paris") + 2, ("ris", "x"))  # inside the "Paris" token
    multi = Site(CONT.index("Paris"), ("Paris", "Rome"))  # multi-token or not, per tokenizer
    return [*digits, inside, multi]


def run(m, s):
    head = m.chat_text(MSG)
    return score(m.tokenizer, m.runner.logprob_rows, head, CONT, s, 4096)[0]


def test_score_matches_a_direct_mlx_lm_forward(loaded):
    s = sites()
    got = run(loaded, s)
    ref = mlx_scorer_reference(loaded, MSG, CONT, s)
    assert got[2].forced  # the site inside "Paris" backed off to the token start
    for r, (t, lp) in zip(got, ref, strict=True):
        assert r.token_index == t
        assert r.logprobs.keys() == lp.keys()
        for c in lp:
            assert abs(r.logprobs[c] - lp[c]) < TOL, (r.char_offset, c, r.logprobs[c], lp[c])


# Past 1,000 tokens, well beyond Gemma 4's 512-token sliding window and mlx-lm's 512-token
# prefill step. Scoring once ran long sequences in chunks and drifted by up to 0.74 here.
CITIES = ["Paris", "Rome", "Oslo", "Lisbon", "Vienna", "Prague", "Dublin", "Madrid"]
LONG_MSG = [MSG[0], {"role": "user", "content": " ".join(
    f"[{i + 1}] {CITIES[i % len(CITIES)]} guide, part {i // len(CITIES) + 1}: opening hours, "
    f"ticket prices, the nearest metro stations and a short history of the old town."
    for i in range(48)) + " Where is the Eiffel Tower?"}]


def test_long_prompt_score_matches_a_direct_mlx_lm_forward(loaded):
    s = sites()
    head = loaded.chat_text(LONG_MSG)
    assert len(loaded.tokenizer.encode(head + CONT)) > 1000
    ref = mlx_scorer_reference(loaded, LONG_MSG, CONT, s)  # frees its logits as it goes
    got = score(loaded.tokenizer, loaded.runner.logprob_rows, head, CONT, s, 4096)[0]
    for r, (t, lp) in zip(got, ref, strict=True):
        assert r.token_index == t
        for c in lp:
            assert abs(r.logprobs[c] - lp[c]) < TOL, (r.char_offset, c, r.logprobs[c], lp[c])


def test_long_logprob_rows_equal_the_full_forward_rows(loaded):
    """logprob_rows runs the output head on the requested rows only, padded to head_min_rows.
    On the GPU in bf16 that must give the full forward's rows bit for bit. The same rows with
    the head on exactly those four rows (no padding) are measured and reported as a warning."""
    import warnings

    import mlx.core as mx
    import torch

    r = loaded.runner
    ids = loaded.tokenizer.encode(loaded.chat_text(LONG_MSG) + CONT)
    assert len(ids) > r.head_min_rows
    rows = [0, len(ids) // 3, len(ids) - 2, len(ids) - 1]
    logits = r.model(mx.array([ids]))[0]
    sel = logits[mx.array(rows)].astype(mx.float32)
    ref = sel - mx.logsumexp(sel, axis=-1, keepdims=True)
    mx.eval(ref)
    del logits, sel
    ref = torch.from_numpy(np.array(ref))
    mx.clear_cache()
    got = r.logprob_rows(ids, rows)
    mx.clear_cache()
    keep, r.head_min_rows = r.head_min_rows, 0
    try:
        unpadded = r.logprob_rows(ids, rows)
    finally:
        r.head_min_rows = keep
        mx.clear_cache()
    loose = (unpadded - ref).abs().max().item()
    warnings.warn(f"{loaded.spec.name} {len(ids)} tokens: head on {len(rows)} unpadded rows, "
                  f"max |logprob - full forward| = {loose:.3g}", stacklevel=1)
    assert torch.equal(got, ref), (got - ref).abs().max().item()


def test_sites_scored_together_equal_sites_scored_alone(loaded):
    s = sites()
    together = run(loaded, s)
    for one, r in zip(s, together, strict=True):
        (alone,) = run(loaded, [one])
        assert alone.token_index == r.token_index
        for c in r.logprobs:
            assert abs(alone.logprobs[c] - r.logprobs[c]) < TOL


def test_the_cited_source_is_the_likely_one(loaded):
    first = run(loaded, sites()[:1])[0]
    assert max(first.logprobs, key=first.logprobs.get) == "1"
