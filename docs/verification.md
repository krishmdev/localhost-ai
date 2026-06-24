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
- **Docker CPU sweep** (`bench/results/cpu-docker.json`): the full 3 modes x 6 concurrencies ran
  under the lease on a contended host (CPU idle 35-69%). The calibrated SLO (1883 ms) came out
  loose, so every point meets it. Treat the run as rough throughput-scaling data only. The
  Grafana screenshot `docs/grafana.png` was captured at the end of it.
- **Memory pressure** (`bench/results/cpu-mempressure.json`, 1.5 GB cgroup, 32 clients, 512
  tokens): neither mode was OOM-killed.
  - fixed:32 completed 41 requests with 0 errors. The KV admission budget kept it to 9 or fewer
    rows.
  - aimd completed 28 requests with 12 read timeouts. Its KV ceiling clamped L to 1-2.
  - The expected "fixed OOMs, AIMD survives" result did not happen. See the README limitations.
- **Offline check** (`make offline-check`, under `.tools/offline-run`):
  - `models verify` passed.
  - The egress canary was blocked (EPERM) in the CLI process and inside the running server
    (`/v1/admin/egress`).
  - A REST completion and a WebSocket generation both worked offline.
  - Companion check, unsandboxed: `lhai egress-check --expect open` connects to all three
    targets.
