"""response_format constraints against a fake model with a small JSON-capable vocabulary.

The fake model's next-token scores come from a hash of the row's history (like fakes.FakeRunner),
with a small bias toward closing brackets and end-of-sequence so most runs finish inside the
token budget. Without the mask it produces garbage; with it, every output that stops must parse."""

import json

import pytest
import torch
from fakes import FakeRunner
from test_scheduler import Clock, done, drain, req

from localhost_ai.engine.constrain import GrammarError, Grammars, grammar_for
from localhost_ai.engine.controller import FixedController
from localhost_ai.engine.request import SamplingParams
from localhost_ai.engine.sampling import make_generator, sample
from localhost_ai.engine.scheduler import Scheduler, SchedulerConfig

llguidance = pytest.importorskip("llguidance")

WORDS = ['{', '}', '[', ']', '"', ':', ',', ' ', '\n', '-', '.', *'0123456789',
         *'abcdefghijklmnopqrstuvwxyz', 'true', 'false', 'null', '"name"', '": ', '", "',
         '{"', '"}', 'age', 'ok']
EOS = len(WORDS)
TOKENS = [w.encode() for w in WORDS] + [b"<eos>"]
V = len(TOKENS)
BIAS = {WORDS.index('}'): 1.5, WORDS.index(']'): 1.5, WORDS.index('"'): 0.5, EOS: 3.0}


class JSONTokenizer:
    """Greedy longest-match tokenizer over TOKENS; also what llguidance's TokenizerWrapper
    wants (tokens as bytes, eos id, call on bytes)."""

    eos_token_id = EOS
    bos_token_id = None
    tokens = TOKENS
    special_token_ids = [EOS]

    def __call__(self, text: bytes) -> list[int]:
        out, i = [], 0
        while i < len(text):
            best = max((j for j, t in enumerate(TOKENS[:-1]) if text.startswith(t, i)),
                       key=lambda j: len(TOKENS[j]))
            out.append(best)
            i += len(TOKENS[best])
        return out

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(WORDS[i] for i in ids if i != EOS)


def fake_grammars() -> Grammars:
    def build():
        return llguidance.LLTokenizer(llguidance.TokenizerWrapper(JSONTokenizer()))

    return Grammars(build, frozenset({EOS}))


class JSONRunner(FakeRunner):
    def _logits(self, rows):
        out = torch.zeros((len(rows), V + 3))  # wider than the vocab, like padded embeddings
        for i, r in enumerate(rows):
            g = torch.Generator().manual_seed(hash(tuple(r[-6:])) % (1 << 31) + len(r))
            out[i, :V] = torch.randn(V, generator=g)
            for t, b in BIAS.items():
                out[i, t] += b
            out[i, V:] = -5.0  # padding rows past the tokenizer's vocab
        return out


GRAMMARS = fake_grammars()
PERSON = {"type": "json_schema", "json_schema": {"name": "person", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["name", "age"],
    "properties": {"name": {"type": "string", "maxLength": 12},
                   "age": {"type": "integer", "minimum": 0, "maximum": 120}}}}}


def sched(limit=8, runner=None):
    return Scheduler(runner or JSONRunner(eos_ids=frozenset({EOS})), JSONTokenizer(),
                     FixedController(limit), cfg=SchedulerConfig(), clock=Clock())


def run(fmt, n=16, params=None, runner=None, limit=8):
    s = sched(limit, runner)
    rs = []
    for i in range(n):
        p = params or SamplingParams(temperature=1.0, seed=i, max_tokens=96)
        r = req([1 + i % 5, 2, 3], p)
        r.constraint = GRAMMARS.constraint(fmt)
        rs.append(r)
        s.add(r)
    drain(s)
    return rs


def text(r):
    return JSONTokenizer().decode(r.generated)


def test_unconstrained_fake_model_does_not_produce_json():
    rs = run({"type": "text"}, n=8)
    parsed = 0
    for r in rs:
        try:
            json.loads(text(r))
            parsed += 1
        except ValueError:
            pass
    assert parsed < len(rs)


def test_json_object_outputs_that_stop_always_parse():
    rs = run({"type": "json_object"}, n=24)
    stopped = [r for r in rs if done(r).finish_reason == "stop"]
    assert len(stopped) >= 18  # the EOS bias ends most rows well inside 96 tokens
    for r in stopped:
        assert isinstance(json.loads(text(r)), dict), text(r)
        assert r.generated[-1] == EOS and r.constraint.done


def test_json_schema_outputs_match_the_schema():
    rs = run(PERSON, n=24)
    stopped = [r for r in rs if done(r).finish_reason == "stop"]
    assert len(stopped) >= 18
    for r in stopped:
        obj = json.loads(text(r))
        assert set(obj) == {"name", "age"}, obj
        assert isinstance(obj["name"], str) and len(obj["name"]) <= 12
        assert isinstance(obj["age"], int) and 0 <= obj["age"] <= 120


def test_greedy_constrained_rows_are_deterministic_and_unconstrained_rows_untouched():
    greedy = SamplingParams(temperature=0.0, max_tokens=40)
    s = sched(limit=4)
    plain = [req([7, 8, 9], greedy) for _ in range(2)]
    shaped = [req([7, 8, 9], greedy) for _ in range(2)]
    for r in shaped:
        r.constraint = GRAMMARS.constraint({"type": "json_object"})
    for a, b in zip(plain, shaped, strict=True):
        s.add(a)
        s.add(b)
    drain(s)
    alone = sched(limit=1)
    ref = req([7, 8, 9], greedy)
    alone.add(ref)
    drain(alone)
    assert [r.generated for r in plain] == [ref.generated] * 2
    assert shaped[0].generated == shaped[1].generated != ref.generated
    json.loads(text(shaped[0]))


def test_constraint_survives_preemption_by_recompute():
    greedy = SamplingParams(temperature=0.0, max_tokens=40)
    calm = run(PERSON, n=4, params=greedy, limit=4)
    # four rows of ~3 prompt tokens: OOM once the padded batch passes 60 tokens
    squeezed = run(PERSON, n=4, params=greedy, limit=4,
                   runner=JSONRunner(eos_ids=frozenset({EOS}), oom_above_tokens=60))
    assert any(r.preemptions for r in squeezed)
    assert [r.generated for r in squeezed] == [r.generated for r in calm]
    for r in squeezed:
        if done(r).finish_reason == "stop":
            json.loads(text(r))


def test_mask_overrides_the_top_logit_in_greedy_and_sampled_rows():
    logits = torch.zeros(2, V + 3)
    logits[:, WORDS.index('a')] = 100.0  # the model "wants" a bare letter
    logits[:, V:] = 200.0  # or a padding id past the tokenizer's vocab
    cs = [GRAMMARS.constraint({"type": "json_object"}) for _ in range(2)]
    ps = [SamplingParams(temperature=0.0), SamplingParams(temperature=1.5, seed=3)]
    picks = sample(logits, ps, [make_generator(0), make_generator(3)], cs)
    first = {WORDS[p] for p in picks}
    assert first <= {'{', '{"', ' ', '\n'}, first


def test_rows_without_constraints_skip_the_mask():
    logits = torch.randn(3, V)
    ps = [SamplingParams(temperature=0.0)] * 3
    gens = [make_generator(0)] * 3
    assert sample(logits, ps, gens, [None, None, None]) == logits.argmax(-1).tolist()


def test_text_format_means_no_constraint():
    assert grammar_for({"type": "text"}) is None
    assert GRAMMARS.constraint(None) is None
    assert GRAMMARS.constraint({"type": "text"}) is None


def test_bad_schemas_raise_grammar_error():
    with pytest.raises(GrammarError):
        GRAMMARS.constraint({"type": "json_schema", "json_schema": {"schema": {"type": "nope"}}})
    with pytest.raises(GrammarError):
        GRAMMARS.constraint({"type": "xml"})
    with pytest.raises(GrammarError):
        GRAMMARS.constraint({"type": "json_schema", "json_schema": {"schema": [1, 2]}})

