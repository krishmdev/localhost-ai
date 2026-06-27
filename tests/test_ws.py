import pytest
from fakes import FakeRunner, fake_service
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from localhost_ai.api.app import create_app


def gen(rid, n=8, **kw):
    return {"type": "generate", "id": rid, "messages": [{"role": "user", "content": "hi " + rid}],
            "max_tokens": n, "temperature": 0, **kw}


def collect(ws, want_done):
    by_id, done = {}, {}
    while len(done) < want_done:
        m = ws.receive_json()
        if m["type"] == "token":
            by_id.setdefault(m["id"], []).append(m["text"])
        elif m["type"] in ("done", "error"):
            done[m["id"]] = m
    return by_id, done


def test_generate_multiplexes_requests():
    with TestClient(create_app(fake_service())) as c, c.websocket_connect("/v1/ws/generate") as ws:
        ws.send_json(gen("a", 6))
        ws.send_json(gen("b", 9))
        ws.send_json(gen("c", 3))
        tokens, done = collect(ws, 3)
        assert {k: len("".join(v)) for k, v in tokens.items()} == {"a": 6, "b": 9, "c": 3}
        assert all(d["type"] == "done" and d["finish_reason"] == "length" for d in done.values())
        assert done["b"]["usage"]["completion_tokens"] == 9
        assert done["a"]["timings"]["ttft_ms"] is not None


def test_generate_cancel():
    svc = fake_service(runner=FakeRunner(t0=0.02))
    with TestClient(create_app(svc)) as c, c.websocket_connect("/v1/ws/generate") as ws:
        ws.send_json(gen("long", 200))
        ws.send_json(gen("short", 5))
        seen = 0
        while seen < 3:
            if ws.receive_json()["type"] == "token":
                seen += 1
        ws.send_json({"type": "cancel", "id": "long"})
        _, done = collect(ws, 2)
        assert done["long"]["finish_reason"] == "cancelled"
        assert done["long"]["usage"]["completion_tokens"] < 200
        assert done["short"]["finish_reason"] == "length"


def test_generate_rejects_bad_messages():
    with TestClient(create_app(fake_service())) as c, c.websocket_connect("/v1/ws/generate") as ws:
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "generate", "id": "x", "messages": []})
        m = ws.receive_json()
        assert m["type"] == "error" and m["id"] == "x"
        ws.send_json({"type": "generate", "messages": [{"role": "user", "content": "hi"}]})
        assert ws.receive_json()["type"] == "error"


def test_disconnect_cancels_in_flight():
    svc = fake_service(runner=FakeRunner(t0=0.02))
    with TestClient(create_app(svc)) as c:
        with c.websocket_connect("/v1/ws/generate") as ws:
            ws.send_json(gen("a", 500))
            assert ws.receive_json()["type"] == "accepted"
            ws.receive_json()
        import time

        deadline = time.time() + 5
        while svc.engine.scheduler.has_work() and time.time() < deadline:
            time.sleep(0.02)
        assert not svc.engine.scheduler.has_work()


def test_telemetry_pushes_snapshots():
    with TestClient(create_app(fake_service())) as c, \
            c.websocket_connect("/v1/ws/telemetry?interval=0.1") as ws:
        a, b = ws.receive_json(), ws.receive_json()
        for m in (a, b):
            assert m["type"] == "telemetry"
            assert m["mode"] == "aimd" and m["batch_limit"] == 16
            assert m["memory"]["limit_bytes"] == 1 << 30
            assert "decode_step_p95_ms" in m and "controller_p95_ms" in m and "busy_ratio" in m
        assert b["t"] > a["t"]


def test_telemetry_controls_need_token():
    app = create_app(fake_service(admin_token="tok"))
    with TestClient(app) as c:
        with c.websocket_connect("/v1/ws/telemetry") as ws:
            ws.send_json({"type": "set_slo", "tpot_ms": 25})
            m = ws.receive_json()
            while m["type"] == "telemetry":
                m = ws.receive_json()
            assert m["type"] == "error" and m["code"] == "forbidden"
        with c.websocket_connect("/v1/ws/telemetry?token=tok") as ws:
            ws.send_json({"type": "set_slo", "tpot_ms": 25})
            ws.send_json({"type": "set_mode", "mode": "fixed", "batch": 4})
            acks = []
            while len(acks) < 2:
                m = ws.receive_json()
                if m["type"] == "ack":
                    acks.append(m)
            assert acks[0]["slo_tpot_ms"] == 25
            assert acks[1]["mode"] == "fixed" and acks[1]["batch_limit"] == 4
            m = ws.receive_json()
            while m["type"] != "telemetry" or m["mode"] != "fixed":
                m = ws.receive_json()
            assert m["slo_tpot_ms"] == 25 and m["batch_limit"] == 4


def test_unknown_ws_path():
    with (TestClient(create_app(fake_service())) as c, pytest.raises(WebSocketDisconnect),
          c.websocket_connect("/v1/ws/nope") as ws):
        ws.receive_json()


def test_foreign_origin_rejected_without_token():
    with TestClient(create_app(fake_service())) as c:
        with pytest.raises(WebSocketDisconnect), \
                c.websocket_connect("/v1/ws/generate",
                                    headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()
        with c.websocket_connect("/v1/ws/telemetry",
                                 headers={"origin": "http://localhost:3000"}) as ws:
            assert ws.receive_json()["type"] == "telemetry"


def test_bad_frames_and_params_do_not_kill_the_socket():
    with TestClient(create_app(fake_service())) as c:
        with c.websocket_connect("/v1/ws/generate") as ws:
            ws.send_text("[1, 2]")
            assert ws.receive_json()["message"] == "expected an object"
            ws.send_json(gen("ok", 2))
            _, done = collect(ws, 1)
            assert done["ok"]["type"] == "done"
        with c.websocket_connect("/v1/ws/telemetry") as ws:
            ws.send_json({"type": "set_slo", "tpot_ms": float("inf")})
            m = ws.receive_json()
            while m["type"] == "telemetry":
                m = ws.receive_json()
            assert m["type"] == "error"
        with pytest.raises(WebSocketDisconnect), \
                c.websocket_connect("/v1/ws/telemetry?interval=abc") as ws:
            ws.receive_json()
