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
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost")
    yield AsyncOpenAI(api_key="unused", base_url="http://localhost/v1", http_client=http)
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


def test_failed_model_swap_keeps_serving():
    svc = fake_service()
    good = svc.parts
    svc.model_names = ["fake-model", "broken"]

    def loader(name):
        if name == "broken":
            raise OSError("weights missing")
        return good

    svc.loader = loader
    with TestClient(create_app(svc)) as c:
        r = c.post("/v1/admin/models/load", json={"model": "broken"})
        assert r.status_code == 500 and "still serving fake-model" in r.json()["error"]["message"]
        assert c.get("/readyz").status_code == 200
        r = c.post("/v1/chat/completions", json={"messages": MSG, "max_tokens": 3})
        assert r.status_code == 200


def test_double_swap_failure_reports_the_original_error():
    svc = fake_service()
    svc.model_names = ["fake-model", "broken"]

    def loader(name):
        raise OSError(f"weights missing for {name}")

    svc.loader = loader
    with TestClient(create_app(svc)) as c:
        r = c.post("/v1/admin/models/load", json={"model": "broken"})
        msg = r.json()["error"]["message"]
        assert r.status_code == 500 and "weights missing for broken" in msg
        assert "reloading 'fake-model' also failed" in msg
        assert c.get("/readyz").status_code == 503
        r = c.post("/v1/chat/completions", json={"messages": MSG, "max_tokens": 3})
        assert r.status_code == 503


def test_server_recovers_after_double_swap_failure():
    svc = fake_service()
    good = svc.parts
    svc.model_names = ["fake-model", "broken"]
    calls = {"n": 0}

    def loader(name):
        calls["n"] += 1
        if calls["n"] <= 2:  # the swap fails, and so does reloading the old model
            raise OSError("disk unavailable")
        return good

    svc.loader = loader
    with TestClient(create_app(svc)) as c:
        assert c.post("/v1/admin/models/load", json={"model": "broken"}).status_code == 500
        assert c.get("/readyz").status_code == 503
        r = c.post("/v1/admin/models/load", json={"model": "fake-model"})
        assert r.status_code == 200 and r.json()["model"] == "fake-model"
        assert c.get("/readyz").status_code == 200


def test_swap_uses_the_new_models_memory_probe():
    # An MLX model's memory is invisible to torch.mps, so a swap between a torch and an MLX
    # preset has to switch the probe the scheduler reads along with the runner.
    from dataclasses import replace

    from localhost_ai.memory import FakeProbe

    svc = fake_service()
    service_probe = svc.probe
    assert svc.engine.scheduler.probe is service_probe  # parts.probe None: the service's
    mlx_probe = FakeProbe(limit=1 << 32, used=1 << 30)
    mlx_parts = replace(svc.parts, name="fake-mlx", probe=mlx_probe,
                        info={**svc.parts.info, "model": "fake-mlx", "backend": "mlx"})
    svc.model_names = ["fake-model", "fake-mlx"]
    svc.loader = lambda name: mlx_parts
    with TestClient(create_app(svc)) as c:
        r = c.post("/v1/admin/models/load", json={"model": "fake-mlx"})
        assert r.status_code == 200
        assert svc.engine.scheduler.probe is mlx_probe
        assert svc.engine.scheduler.mem.limit == 1 << 32
