# Verification log

What was actually run on the build machine (Apple M1 Pro, 16 GB, macOS 26, Docker Desktop 28.2
arm64). This project calls no paid APIs and uses no keys.

## 2026-09-23

- **Unit tests** (`make test`): the fake-model suite passes locally. It covers the controller
  rules, the simulator bounds over 20 seeds, scheduler, KV, sampling, detok, probes, OpenAI SDK
  and SSE shape, WebSockets and metrics.
- **Model tests** (`make test-model`, SmolLM2-135M-Instruct @ `12fd25f7`, CPU fp32): 4 passed.
  - Batched and one-at-a-time greedy outputs are identical for all 32 tokens of five
    mixed-length prompts, including a mid-stream join.
  - Merge logits match a solo run to 1e-4.
  - A preempted request's tokens match an uninterrupted run.
  - An error injected in layer 17 is cropped and the next step matches.
- **Pinned model files**: `lhai models pull` downloads the pinned revision into `.models/` and
  all eight files match `models.lock`.
- **Native smoke** (`scripts/smoke.sh`, MPS): readyz, `/v1/models`, a JSON completion, SSE with
  usage and `[DONE]`, two multiplexed WebSocket generations, one `ws_top.py` frame, and
  `/metrics` all worked.
- **Compose stack**:
  - `docker compose -p lhai up` came up; Prometheus reported the server target `up`, and Grafana
    listed the provisioned dashboard.
  - `lhai egress-check` inside the server container: all three targets blocked ("Network is
    unreachable" / name resolution failure). `wget` to 1.1.1.1 from the Prometheus and Grafana
    containers failed too.
  - Stack torn down afterwards.
- **Simulator**: `scripts/sim_report.py` wrote `docs/results/sim.json` and the S1-S5 plots. S3's
  floor bound (L >= 0.5 b*) holds on 19 of 20 seeds; the worst seed reaches 0.47 b*. This is
  reported, not tuned.
- **MPS benchmark** (`bench/results/mps-native.json`): ran under the compute lease with other
  agents' workloads on the machine. CPU idle was 55-85% before each point, and the memory probe
  saw a 3.27 GiB MPS limit because swap was nearly full. Both are recorded in the file.
