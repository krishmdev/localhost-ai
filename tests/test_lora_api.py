"""Adapter selection through the OpenAI API and the WebSocket API, with a fake runner whose
"adapters" shift every next token by a fixed offset, so the output shows which adapter a row
ran with. No mlx needed."""

import pytest
import torch
from fakes import VOCAB, FakeRunner, FakeTokenizer, next_token
from fastapi.testclient import TestClient

from localhost_ai.api.app import create_app
from localhost_ai.config import Settings
from localhost_ai.engine.controller import FixedController
from localhost_ai.engine.request import Request, SamplingParams
from localhost_ai.engine.scheduler import Scheduler
from localhost_ai.memory import FakeProbe
from localhost_ai.metrics import EngineMetrics
from localhost_ai.service import ModelParts, Service

OFFSET = {None: 0, "tutor": 5, "terse": 11}


class AdapterRunner(FakeRunner):
    def _adapter_logits(self, state):
        out = torch.full((len(state.rows), VOCAB), -10.0)
        for i, (r, a) in enumerate(zip(state.rows, state.slots, strict=True)):
            t = (next_token(r) + OFFSET[a]) % (VOCAB - 1)
            out[i, t] = 5.0
        return out

    def prefill(self, seqs, adapters=None):
        state, _ = super().prefill(seqs)
        state.slots = list(adapters) if adapters else [None] * len(seqs)
        self.calls.append(("adapters", tuple(state.slots)))
        return state, self._adapter_logits(state)

    def decode(self, state, tokens):
        super().decode(state, tokens)
        self.calls.append(("slots", tuple(state.slots)))
        return self._adapter_logits(state)

    def merge(self, a, b):
        out = super().merge(a, b)
        out.slots = a.slots + b.slots
        return out

    def select(self, state, keep):
        out = super().select(state, keep)
        out.slots = [state.slots[i] for i in keep]
        return out


def expected(prompt, adapter, n):
    hist, out = list(prompt), []
    for _ in range(n):
        t = (next_token(hist) + OFFSET[adapter]) % (VOCAB - 1)
        out.append(t)
        hist.append(t)
    return FakeTokenizer().decode(out)


@pytest.fixture
def client():
    tok = FakeTokenizer()
    parts = ModelParts(name="base-model", runner=AdapterRunner(t0=0.001), tokenizer=tok,
                       encode_chat=lambda m: tok.encode(m[-1]["content"]) or [0],
                       default_max_tokens=8, adapters=["tutor", "terse"],
                       info={"model": "base-model", "repo": "org/base", "device": "cpu",
                             "dtype": "fp32", "quant": "none", "adapters": "tutor,terse"})
    svc = Service(settings=Settings(control_interval_s=0.2, max_context=256), parts=parts,
                  probe=FakeProbe(limit=1 << 30, used=1 << 28), metrics=EngineMetrics(),
                  model_names=["base-model"])
    with TestClient(create_app(svc)) as c:
        yield c


PROMPT = "hello there"


def chat(client, **kw):
    body = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 8,
            "temperature": 0, **kw}
    return client.post("/v1/chat/completions", json=body)


def test_models_lists_adapters_with_their_base(client):
    data = client.get("/v1/models").json()["data"]
    assert [(m["id"], m.get("parent")) for m in data] == [
        ("base-model", None), ("tutor", "base-model"), ("terse", "base-model")]


@pytest.mark.parametrize("kw,adapter", [
    ({}, None), ({"model": "base-model"}, None), ({"model": "org/base"}, None),
    ({"model": "tutor"}, "tutor"), ({"adapter": "terse"}, "terse"),
    ({"model": "base-model", "adapter": "tutor"}, "tutor"),
    ({"model": "terse", "adapter": "terse"}, "terse")])
def test_requests_run_on_the_adapter_they_name(client, kw, adapter):
    r = chat(client, **kw)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == (adapter or "base-model")
    ids = FakeTokenizer().encode(PROMPT)
    assert body["choices"][0]["message"]["content"] == expected(ids, adapter, 8)


def test_unknown_names_and_conflicts(client):
    r = chat(client, model="nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
    r = chat(client, adapter="nope")
    assert r.status_code == 404 and "tutor, terse" in r.json()["error"]["message"]
    r = chat(client, model="tutor", adapter="terse")
    assert r.status_code == 400


def test_streaming_reports_the_adapter_as_model(client):
    with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 4,
            "temperature": 0, "stream": True, "model": "tutor"}) as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data: {")]
    assert lines and all('"model":"tutor"' in ln for ln in lines)


def test_websocket_generate_with_an_adapter(client):
    with client.websocket_connect("/v1/ws/generate") as ws:
        for rid, extra in (("a", {"adapter": "terse"}), ("b", {"model": "nope"})):
            ws.send_json({"type": "generate", "id": rid, "max_tokens": 4, "temperature": 0,
                          "messages": [{"role": "user", "content": PROMPT}], **extra})
        text, errors = "", []
        while True:
            m = ws.receive_json()
            if m["type"] == "token":
                text += m["text"]
            elif m["type"] == "error":
                errors.append(m)
            elif m["type"] == "done":
                break
    assert text == expected(FakeTokenizer().encode(PROMPT), "terse", 4)
    assert errors and errors[0]["id"] == "b" and errors[0]["code"] == "404"


def test_one_batch_mixes_adapters_and_base_rows():
    runner = AdapterRunner()
    sched = Scheduler(runner, FakeTokenizer(), FixedController(8))
    params = SamplingParams(temperature=0.0, max_tokens=6)
    route = [None, "tutor", "terse", "tutor", None]
    prompts = [[1, 2, 3], [4, 5], [6], [7, 8, 9, 10], [11, 12]]
    reqs = [Request(p, params, adapter=a) for p, a in zip(prompts, route, strict=True)]
    for r in reqs[:3]:
        sched.add(r)
    sched.step()
    for r in reqs[3:]:  # join a running batch
        sched.add(r)
    while sched.has_work():
        sched.step()
    for p, a, r in zip(prompts, route, reqs, strict=True):
        assert FakeTokenizer().decode(r.generated) == expected(p, a, 6)
    assert tuple(route) in [slots for kind, slots in runner.calls if kind == "slots"]
