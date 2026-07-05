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
from localhost_ai.engine.request import ErrorEvent, SamplingParams
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



class NoEOSRunner(JSONRunner):
    """Logits one column short, so the end-of-sequence token never fits the mask."""

    def _logits(self, rows):
        return super()._logits(rows)[:, :EOS]


def test_complete_json_with_no_eos_in_the_logits_finishes_with_stop():
    rs = run({"type": "json_object"}, n=12, runner=NoEOSRunner(eos_ids=frozenset({EOS})))
    for r in rs:
        d = done(r)
        if d.finish_reason == "stop":  # the grammar completed; nothing forced after it
            assert isinstance(json.loads(text(r)), dict), text(r)
            assert EOS not in r.generated and r.constraint.complete
        else:
            assert d.finish_reason == "length" and not r.constraint.complete
    assert sum(done(r).finish_reason == "stop" for r in rs) >= 6


def test_a_failed_matcher_ends_the_row_with_an_error(monkeypatch):
    s = sched()
    r = req([1, 2, 3], SamplingParams(temperature=1.0, seed=3, max_tokens=96))
    r.constraint = GRAMMARS.constraint({"type": "json_object"})
    calls = []
    real = r.constraint.matcher.consume_token
    r.constraint.matcher = type("M", (), {
        "consume_token": lambda self, t: calls.append(t) or (len(calls) < 3 and real(t)),
        "__getattr__": lambda self, name: getattr(real.__self__, name)})()
    s.add(r)
    drain(s)
    assert isinstance(done(r), ErrorEvent) and "rejected" in done(r).message
    assert r.finish_reason == "error" and len(r.generated) == 2  # the rejected pick is dropped


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


# --- the OpenAI API ---------------------------------------------------------------------------

def json_service():
    from localhost_ai.config import Settings
    from localhost_ai.memory import FakeProbe
    from localhost_ai.metrics import EngineMetrics
    from localhost_ai.service import ModelParts, Service

    parts = ModelParts(name="fake-json", runner=JSONRunner(eos_ids=frozenset({EOS}), t0=0.001),
                       tokenizer=JSONTokenizer(), encode_chat=lambda msgs: [1, 2, 3],
                       default_max_tokens=96, grammars=GRAMMARS,
                       info={"model": "fake-json", "repo": "test/fake-json", "device": "cpu",
                             "dtype": "fp32", "quant": "none"})
    return Service(settings=Settings(control_interval_s=0.2, max_context=256), parts=parts,
                   probe=FakeProbe(limit=1 << 30, used=1 << 28), metrics=EngineMetrics(),
                   model_names=["fake-json"])


@pytest.fixture
async def json_sdk():
    import httpx
    from openai import AsyncOpenAI

    from localhost_ai.api.app import create_app

    svc = json_service()
    app = create_app(svc)
    svc.start()
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost")
    yield AsyncOpenAI(api_key="unused", base_url="http://localhost/v1", http_client=http)
    await http.aclose()
    svc.stop()


MSG = [{"role": "user", "content": "a person as json"}]


async def test_sdk_json_schema_non_streaming(json_sdk):
    stopped = 0
    for seed in range(4):
        r = await json_sdk.chat.completions.create(
            model="fake-json", messages=MSG, seed=seed, temperature=1.0,
            response_format=PERSON)
        if r.choices[0].finish_reason == "stop":
            stopped += 1
            obj = json.loads(r.choices[0].message.content)
            assert set(obj) == {"name", "age"}
    assert stopped >= 2


async def test_sdk_json_object_streaming(json_sdk):
    stream = await json_sdk.chat.completions.create(
        model="fake-json", messages=MSG, seed=1, temperature=1.0, stream=True,
        response_format={"type": "json_object"})
    chunks = [c async for c in stream]
    body = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert isinstance(json.loads(body), dict)


async def test_sdk_rejects_a_bad_schema(json_sdk):
    import openai

    with pytest.raises(openai.BadRequestError, match="invalid response_format"):
        await json_sdk.chat.completions.create(
            model="fake-json", messages=MSG,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "x", "schema": {"type": "nope"}}})


def test_models_without_grammar_support_get_a_400():
    from fakes import fake_service
    from fastapi.testclient import TestClient

    from localhost_ai.api.app import create_app

    with TestClient(create_app(fake_service())) as c:
        r = c.post("/v1/chat/completions", json={"messages": MSG,
                                                 "response_format": {"type": "json_object"}})
        assert r.status_code == 400
        assert "not supported" in r.json()["error"]["message"]
        r = c.post("/v1/chat/completions", json={"messages": MSG, "max_tokens": 2,
                                                 "response_format": {"type": "text"}})
        assert r.status_code == 200


def test_websocket_generate_with_response_format():
    from fastapi.testclient import TestClient

    from localhost_ai.api.app import create_app

    with TestClient(create_app(json_service())) as c, c.websocket_connect("/v1/ws/generate") as ws:
        ws.send_json({"type": "generate", "id": "a", "messages": MSG, "seed": 1,
                      "temperature": 1.0, "response_format": {"type": "json_object"}})
        ws.send_json({"type": "generate", "id": "b", "messages": MSG, "response_format":
                      {"type": "json_schema", "json_schema": {"schema": {"type": "nope"}}}})
        text, done, errors = "", None, []
        while done is None:
            m = ws.receive_json()
            if m["type"] == "token":
                text += m["text"]
            elif m["type"] == "error":
                errors.append(m)
            elif m["type"] == "done":
                done = m
    assert done["finish_reason"] == "stop" and isinstance(json.loads(text), dict)
    assert errors[0]["id"] == "b" and errors[0]["code"] == "400"


async def test_grammar_prepared_before_model_swap_cannot_reach_new_engine(monkeypatch):
    import asyncio
    from dataclasses import replace

    from fastapi import HTTPException

    from localhost_ai.api import openai_routes
    from localhost_ai.api.schemas import ChatCompletionRequest

    svc = json_service()
    new_parts = replace(svc.parts, name="new-json", runner=JSONRunner(eos_ids=frozenset({EOS})),
                        info={**svc.parts.info, "model": "new-json"})
    svc.loader = lambda name: new_parts
    svc.start()
    entered, release = asyncio.Event(), asyncio.Event()

    async def paused_constraint(_grammars, _fmt):
        entered.set()
        await release.wait()
        return object()

    monkeypatch.setattr(openai_routes, "build_constraint", paused_constraint)
    body = ChatCompletionRequest(messages=MSG, response_format={"type": "json_object"})
    try:
        preparing = asyncio.create_task(openai_routes.prepare_request(svc, body, MSG))
        await entered.wait()
        await svc.swap_model("new-json")
        release.set()
        with pytest.raises(HTTPException) as exc:
            await preparing
        assert exc.value.status_code == 503
        assert svc.engine.scheduler.waiting == [] or not svc.engine.scheduler.waiting
    finally:
        release.set()
        svc.stop()
