import json
import time

import httpx
import pytest
from fakes import FakeRunner, fake_service
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from localhost_ai.api.app import create_app
from localhost_ai.engine.request import Request, SamplingParams


@pytest.fixture
def client():
    app = create_app(fake_service())
    with TestClient(app) as c:
        yield c


@pytest.fixture
async def sdk():
    svc = fake_service()
    app = create_app(svc)
    svc.start()
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    yield AsyncOpenAI(api_key="unused", base_url="http://test/v1", http_client=http)
    await http.aclose()
    svc.stop()


MSG = [{"role": "user", "content": "hello there"}]


async def test_openai_sdk_non_streaming(sdk):
    r = await sdk.chat.completions.create(model="fake-model", messages=MSG, max_tokens=8,
                                          temperature=0)
    assert r.object == "chat.completion"
    assert r.choices[0].finish_reason == "length"
    assert r.usage.completion_tokens == 8
    assert r.usage.total_tokens == r.usage.prompt_tokens + 8
    assert len(r.choices[0].message.content) == 8


async def test_openai_sdk_streaming_with_usage(sdk):
    stream = await sdk.chat.completions.create(model="fake-model", messages=MSG, max_tokens=6,
                                               temperature=0, stream=True,
                                               stream_options={"include_usage": True})
    chunks = [c async for c in stream]
    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert len(text) == 6
    finals = [c for c in chunks if c.choices and c.choices[0].finish_reason]
    assert finals[-1].choices[0].finish_reason == "length"
    assert chunks[-1].choices == [] and chunks[-1].usage.completion_tokens == 6


async def test_openai_sdk_models(sdk):
    models = await sdk.models.list()
    assert [m.id for m in models.data] == ["fake-model"]


def test_sse_framing(client):
    with client.stream("POST", "/v1/chat/completions",
                       json={"messages": MSG, "max_tokens": 4, "stream": True}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = "".join(r.iter_text())
    events = [e for e in raw.split("\n\n") if e]
    assert all(e.startswith("data: ") for e in events)
    assert events[-1] == "data: [DONE]"
    bodies = [json.loads(e[6:]) for e in events[:-1]]
    assert all(b["object"] == "chat.completion.chunk" for b in bodies)
    assert all("finish_reason" in b["choices"][0] for b in bodies)
    assert bodies[-1]["choices"][0]["finish_reason"] == "length"


def test_same_seed_same_output(client):
    body = {"messages": MSG, "max_tokens": 12, "temperature": 0.8, "seed": 7}
    a = client.post("/v1/chat/completions", json=body).json()
    b = client.post("/v1/chat/completions", json=body).json()
    assert a["choices"][0]["message"]["content"] == b["choices"][0]["message"]["content"]


def test_unknown_model_and_bad_request(client):
    r = client.post("/v1/chat/completions", json={"model": "gpt-9", "messages": MSG})
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
    r = client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"


def test_queue_full_returns_429():
    svc = fake_service(runner=FakeRunner(t0=0.05), max_queue=1, fixed_batch=1,
                       controller="fixed")
    sched = svc.engine.scheduler
    with TestClient(create_app(svc)) as c:
        sched.add(Request([1, 2], SamplingParams(max_tokens=50)))
        deadline = time.time() + 5
        while not sched.running and time.time() < deadline:
            time.sleep(0.01)
        sched.add(Request([3, 4], SamplingParams(max_tokens=50)))  # the one queue slot
        r = c.post("/v1/chat/completions", json={"messages": MSG, "max_tokens": 2})
        assert r.status_code == 429
        assert r.headers["retry-after"] == "1"
        assert r.json()["error"]["code"] == "queue_full"
        metrics = c.get("/metrics/").text
        assert 'lhai_requests_total{status="rejected"} 1.0' in metrics


def test_admin_controller_and_auth():
    app = create_app(fake_service(admin_token="s3cret"))
    with TestClient(app) as c:
        assert c.get("/v1/admin/controller").status_code == 401
        h = {"Authorization": "Bearer s3cret"}
        r = c.put("/v1/admin/controller", json={"mode": "fixed", "batch": 3}, headers=h)
        assert r.status_code == 200
        assert c.get("/v1/admin/controller", headers=h).json()["mode"] == "fixed"
        r = c.put("/v1/admin/controller", json={"mode": "aimd", "slo_tpot_ms": 40}, headers=h)
        assert r.json()["mode"] == "aimd" and r.json()["slo_tpot_ms"] == 40
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/readyz").status_code == 200


def test_egress_route_reports_blocked_and_open(monkeypatch):
    from localhost_ai import egress

    app = create_app(fake_service())
    with TestClient(app) as c:
        monkeypatch.setattr(egress, "probe", lambda timeout: {"1.1.1.1:443": "blocked (x)"})
        assert c.get("/v1/admin/egress").json()["ok"] is True
        assert c.get("/v1/admin/egress?expect=open").status_code == 409
        monkeypatch.setattr(egress, "probe", lambda timeout: {"1.1.1.1:443": "open"})
        assert c.get("/v1/admin/egress").status_code == 409
