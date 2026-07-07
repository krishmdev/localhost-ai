"""Thinking blocks on fake models: reasoning split from the answer, the max_thinking_tokens budget
forcing the block closed, batched rows equal to solo rows, and a JSON response_format that only
applies after the block closes."""

import json

import pytest
import torch
from fakes import FakeRunner, fake_service
from fastapi.testclient import TestClient
from test_scheduler import Clock, drain, req

from localhost_ai.api.app import create_app
from localhost_ai.engine.controller import FixedController
from localhost_ai.engine.request import DoneEvent, SamplingParams, TokenEvent
from localhost_ai.engine.scheduler import Scheduler, SchedulerConfig
from localhost_ai.engine.thinking import Thinking, ThinkSpec, wrap

OPEN, CLOSE, NL, EOS = 30, 31, 32, 39
PIECES = {**{i: chr(ord("a") + i) for i in range(26)}, OPEN: "<think>", CLOSE: "</think>",
          NL: "\n"}
SPEC = ThinkSpec((OPEN,), CLOSE, (NL, CLOSE), is_space=lambda t: t == NL)


class ThinkTokenizer:
    eos_token_id = EOS

    def decode(self, ids, skip_special_tokens=True):
        return "".join(PIECES.get(i, "") for i in ids)


class ThinkRunner(FakeRunner):
    """Thinks for `natural` tokens (letters, varying with the prompt), closes the block, writes
    a blank line and an answer of `answer` letters, then EOS. Every choice depends only on the
    row's own history, so batching can't change a row's output."""

    def __init__(self, natural=12, answer=5, **kw):
        super().__init__(eos_ids=frozenset({EOS}), **kw)
        self.natural, self.answer = natural, answer

    def next(self, h):
        last_open = max((i for i, t in enumerate(h) if t == OPEN), default=-1)
        last_close = max((i for i, t in enumerate(h) if t == CLOSE), default=-1)
        if last_open > last_close:
            n = len(h) - last_open - 1
            return CLOSE if n >= self.natural else (n + h[0]) % 26
        since = len(h) - last_close - 1 if last_close >= 0 else len(h)
        if since == 0:
            return NL
        return EOS if since > self.answer else 20 + (since - 1) % 5

    def _logits(self, rows):
        out = torch.full((len(rows), 40), -10.0)
        for i, r in enumerate(rows):
            out[i, self.next(r)] = 5.0
        return out


def run(prompts, budgets, limit, runner=None, inner=None):
    s = Scheduler(runner or ThinkRunner(), ThinkTokenizer(), FixedController(limit),
                  cfg=SchedulerConfig(), clock=Clock())
    rs = []
    for p, b in zip(prompts, budgets, strict=True):
        r = req(p, SamplingParams(temperature=0, max_tokens=64))
        r.constraint = wrap(SPEC, p, {}, b, inner() if inner else None)
        events = []
        r.on_event = events.append
        r.events = events
        rs.append(r)
        s.add(r)
    drain(s)
    return rs


def split(r):
    think = "".join(e.text for e in r.events if isinstance(e, TokenEvent) and e.reasoning)
    content = "".join(e.text for e in r.events if isinstance(e, TokenEvent) and not e.reasoning)
    done = next(e for e in r.events if isinstance(e, DoneEvent))
    return think, content, done


PROMPT = [3, 4, OPEN]  # the template left the block open (Qwen3.5 with thinking on)


def test_natural_close_splits_reasoning_from_the_answer():
    (r,) = run([PROMPT], [None], 1)
    think, content, done = split(r)
    assert len(think) == 12 and "<think>" not in think + content
    assert content == "uvwxy"  # the blank line after </think> is not part of the answer
    assert done.thinking_tokens == 12 and done.finish_reason == "stop"
    assert r.generated[12] == CLOSE


@pytest.mark.parametrize("budget", [0, 1, 5, 11])
def test_budget_forces_the_block_closed(budget):
    (r,) = run([PROMPT], [budget], 1)
    think, content, done = split(r)
    assert done.thinking_tokens == budget and len(think) == budget
    assert r.generated[budget:budget + 2] == [NL, CLOSE]  # forced
    assert r.constraint.forced
    assert content == "uvwxy" and done.finish_reason == "stop"


def test_budget_above_the_natural_length_changes_nothing():
    (a,) = run([PROMPT], [None], 1)
    (b,) = run([PROMPT], [50], 1)
    assert a.generated == b.generated and not b.constraint.forced


def test_batched_rows_equal_solo_rows():
    prompts = [[i + 1, 4, OPEN] for i in range(6)]
    budgets = [None, 0, 3, 7, 30, 2]
    solo = [run([p], [b], 1)[0] for p, b in zip(prompts, budgets, strict=True)]
    batched = run(prompts, budgets, 6)
    for a, b in zip(solo, batched, strict=True):
        assert a.generated == b.generated
        assert split(a)[:2] == split(b)[:2]
        assert split(a)[2].thinking_tokens == split(b)[2].thinking_tokens


def test_the_model_may_open_a_block_itself_when_thinking_is_enabled():
    """Gemma 4 style: the prompt has no open block; enable_thinking lets the first token open
    one, and the budget still applies."""
    class Opener(ThinkRunner):
        def next(self, h):
            return OPEN if len(h) == 3 else super().next(h)

    s = Scheduler(Opener(), ThinkTokenizer(), FixedController(1), cfg=SchedulerConfig(),
                  clock=Clock())
    r = req([5, 6, 7], SamplingParams(temperature=0, max_tokens=64))
    r.constraint = wrap(SPEC, r.prompt_ids, {"enable_thinking": True}, 4, None)
    assert isinstance(r.constraint, Thinking) and not r.constraint.inside
    events = []
    r.on_event = events.append
    r.events = events
    s.add(r)
    drain(s)
    think, content, done = split(r)
    assert r.generated[0] == OPEN and done.thinking_tokens == 4 and len(think) == 4
    assert content == "uvwxy"


def test_no_block_means_no_wrapper():
    assert wrap(SPEC, [3, 4, OPEN, CLOSE], {}, 10, None) is None  # thinking off in the template
    assert wrap(SPEC, [3, 4], {"enable_thinking": False}, 10, "inner") == "inner"
    assert wrap(None, PROMPT, {}, 10, "inner") == "inner"


class OnlyLetters:
    """A stand-in answer constraint: allows letters u..y and EOS, records what it saw."""

    def __init__(self):
        self.seen, self.masks = [], 0
        self.complete, self.broken = False, None

    def allowed(self, width):
        self.masks += 1
        m = torch.zeros(width, dtype=torch.bool)
        m[20:25] = True
        m[EOS] = True
        return m

    def advance(self, t):
        self.seen.append(t)


def test_answer_constraint_applies_only_after_the_block():
    inner = OnlyLetters()
    (r,) = run([PROMPT], [3], 1, inner=lambda: inner)
    _, content, done = split(r)
    # the blank line the model wants after </think> is masked out (the first allowed token
    # stands in), the rest of the answer isn't
    assert content == "uuvwxy"
    assert inner.seen == r.generated[5:]  # 3 thoughts + forced "\n" + "</think>"
    assert inner.masks == len(r.generated) - 5


# --- JSON after thinking, with llguidance ------------------------------------------------------

def test_json_response_format_applies_after_a_forced_close():
    pytest.importorskip("llguidance")
    from test_constrain import EOS as JEOS
    from test_constrain import GRAMMARS, WORDS, JSONRunner, JSONTokenizer

    spec = ThinkSpec((WORDS.index("ok"),), WORDS.index("null"),
                     (WORDS.index("\n"), WORDS.index("null")),
                     is_space=lambda t: WORDS[t].strip() == "")
    s = Scheduler(JSONRunner(eos_ids=frozenset({JEOS})), JSONTokenizer(), FixedController(8),
                  cfg=SchedulerConfig(), clock=Clock())
    rs = []
    for i in range(16):
        prompt = [1 + i % 5, 2, WORDS.index("ok")]
        r = req(prompt, SamplingParams(temperature=1.0, seed=i, max_tokens=96))
        r.constraint = wrap(spec, prompt, {}, 4, GRAMMARS.constraint({"type": "json_object"}),
                            frozenset({JEOS}))
        events = []
        r.on_event = events.append
        r.events = events
        rs.append(r)
        s.add(r)
    drain(s)
    stopped = 0
    for r in rs:
        think, content, done = split(r)
        assert done.thinking_tokens <= 4
        if done.finish_reason == "stop":
            stopped += 1
            assert isinstance(json.loads(content), dict)
    assert stopped >= 12


# --- the API -----------------------------------------------------------------------------------

def think_service(**settings):
    seen = {}
    svc = fake_service(ThinkRunner(t0=0.001), **settings)
    svc.parts.tokenizer = ThinkTokenizer()

    def encode_chat(messages, overrides=None):
        seen["kwargs"] = overrides
        on = {**svc.parts.template_defaults, **(overrides or {})}.get("enable_thinking")
        return [3, 4, OPEN] if on else [3, 4, OPEN, CLOSE]

    svc.parts.encode_chat = encode_chat
    svc.parts.template_defaults = {"enable_thinking": False}
    svc.parts.think = SPEC
    svc.engine = svc._build_engine(svc.engine.controller)
    return svc, seen


MSG = [{"role": "user", "content": "hi"}]


def test_api_reasoning_content_and_usage():
    svc, seen = think_service()
    with TestClient(create_app(svc)) as c:
        r = c.post("/v1/chat/completions", json={
            "messages": MSG, "max_tokens": 64, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": True}, "max_thinking_tokens": 4})
        off = c.post("/v1/chat/completions", json={"messages": MSG, "max_tokens": 64,
                                                   "temperature": 0})
    body = r.json()
    msg = body["choices"][0]["message"]
    assert seen["kwargs"] is None  # the last request sent none
    assert msg["reasoning_content"] == "defg"
    assert msg["content"] == "uvwxy"
    assert body["usage"]["thinking_tokens"] == 4
    assert body["usage"]["completion_tokens_details"] == {"reasoning_tokens": 4}
    # thoughts, the forced newline and close, the blank line, the answer, EOS
    assert body["usage"]["completion_tokens"] == 4 + 2 + 1 + 5 + 1
    plain = off.json()
    assert plain["choices"][0]["message"].get("reasoning_content") is None
    assert plain["usage"].get("thinking_tokens") is None


def test_api_streams_reasoning_as_its_own_delta():
    svc, _ = think_service()
    with TestClient(create_app(svc)) as c, c.stream("POST", "/v1/chat/completions", json={
            "messages": MSG, "max_tokens": 64, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": True},
            "max_thinking_tokens": 3}) as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data: {")]
    deltas = [json.loads(ln[6:])["choices"][0]["delta"] for ln in lines
              if json.loads(ln[6:]).get("choices")]
    thought = "".join(d.get("reasoning_content") or "" for d in deltas)
    answer = "".join(d.get("content") or "" for d in deltas)
    assert len(thought) == 3 and answer == "uvwxy"
    first_answer = next(i for i, d in enumerate(deltas) if d.get("content"))
    assert all(not d.get("reasoning_content") for d in deltas[first_answer:])


def test_api_budget_on_a_model_without_thinking_is_400():
    with TestClient(create_app(fake_service())) as c:
        r = c.post("/v1/chat/completions", json={"messages": MSG, "max_thinking_tokens": 8})
    assert r.status_code == 400
