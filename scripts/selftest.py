"""Load the real model in-process and do one REST and one WebSocket round trip. Used by CI
inside a `--network none` container, where nothing may be downloaded."""

from fastapi.testclient import TestClient

from localhost_ai.api.app import create_app
from localhost_ai.config import Settings
from localhost_ai.egress import probe
from localhost_ai.service import build_from_settings

leaks = [k for k, v in probe(2.0).items() if v == "open"]
assert not leaks, f"egress open: {leaks}"

svc = build_from_settings(Settings())
with TestClient(create_app(svc)) as c:
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Name three planets."}],
        "max_tokens": 16, "temperature": 0})
    r.raise_for_status()
    print("rest:", r.json()["choices"][0]["message"]["content"][:60])
    with c.websocket_connect("/v1/ws/generate") as ws:
        ws.send_json({"type": "generate", "id": "a", "max_tokens": 8,
                      "messages": [{"role": "user", "content": "Say hi."}]})
        while (m := ws.receive_json())["type"] != "done":
            pass
    print("ws:", m["finish_reason"], m["usage"])
    assert "lhai_generated_tokens_total" in c.get("/metrics/").text
print("selftest ok")
