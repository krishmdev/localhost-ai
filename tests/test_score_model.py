"""/v1/score's scoring on the real 4-bit MLX checkpoints, against a direct mlx-lm forward that
follows Vizor's MLXScorer step by step (joint tokenization, in-place reads, back-off and teacher
forcing inside a token). Presets that aren't downloaded are skipped.

    uv run pytest -q -m mlx_model tests/test_score_model.py
    LHAI_MLX_PRESETS=qwen2.5-3b-mlx4,gemma-4-e4b-mlx4 uv run pytest -q -m mlx_model \
        tests/test_score_model.py"""

import os

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
    """Vizor's MLXScorer.score, on the same weights, with mlx-lm's own tokenizer wrapper."""
    import mlx.core as mx
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    tok = load_tokenizer(m.path)
    enc = tok._tokenizer

    def logprobs(ids):
        logits = m.runner.model(mx.array([ids]))[0].astype(mx.float32)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def forced(head_ids, piece, cand):
        a = enc(piece + cand, add_special_tokens=False)["input_ids"]
        b = enc(piece, add_special_tokens=False)["input_ids"] if piece else []

        def seq(tail):
            if not tail:
                return 0.0
            lp = logprobs(head_ids + tail)
            n = len(head_ids)
            return float(sum(lp[n + i - 1, t].item() for i, t in enumerate(tail)))

        return seq(a) - seq(b)

    # mlx-lm's wrapper turns enable_thinking on by default for models that can think (Gemma 4,
    # Qwen3.5); the server's default is the preset's, which is off, so say so explicitly
    kwargs = {"enable_thinking": False, **(m.spec.chat_template_kwargs or {})}
    head = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                   **kwargs)
    full = head + continuation
    e = enc(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = e["input_ids"], e["offset_mapping"]
    lp_all = logprobs(ids)
    out = []
    for s in sites:
        a = len(head) + s.char_offset
        t = next(i for i, (_, en) in enumerate(offs) if en > a)
        start = offs[t][0]
        lp = {}
        for c in s.candidates:
            c_ids = enc(c, add_special_tokens=False)["input_ids"]
            if start == a and len(c_ids) == 1 and t > 0:
                lp[c] = float(lp_all[t - 1, c_ids[0]].item())
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
