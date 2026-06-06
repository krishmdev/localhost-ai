from fakes import fake_service
from fastapi.testclient import TestClient

from localhost_ai.api.app import create_app


def sample(text, name):
    for line in text.splitlines():
        if line.startswith(name + " ") or line.startswith(name + "{"):
            return float(line.rsplit(" ", 1)[1])
    raise KeyError(name)


def test_metrics_populated_after_requests():
    svc = fake_service(control_interval_s=0.05)
    with TestClient(create_app(svc)) as c:
        for _ in range(3):
            r = c.post("/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "abc"}], "max_tokens": 5})
            assert r.status_code == 200
        import time

        time.sleep(0.2)  # let a control tick refresh the gauges
        text = c.get("/metrics/").text
    assert sample(text, "lhai_generated_tokens_total") == 15
    assert sample(text, "lhai_ttft_seconds_count") == 3
    assert sample(text, "lhai_tpot_seconds_count") == 3
    assert sample(text, "lhai_e2e_latency_seconds_count") == 3
    assert sample(text, 'lhai_requests_total{status="length"}') == 3
    assert sample(text, "lhai_batch_limit") == 16
    assert sample(text, "lhai_device_memory_limit_bytes") == 1 << 30
    assert sample(text, "lhai_slo_tpot_seconds") == 0.1
    assert sample(text, 'lhai_controller_decisions_total{action="hold"}') >= 1
    assert 'lhai_model_info{' in text and 'model="fake-model"' in text
    assert "lhai_engine_busy_ratio" in text and "lhai_kv_cache_tokens" in text
