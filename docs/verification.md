# Verification log

Everything below was run on the build machine: an Apple M1 Pro with 16 GB running macOS 26, and
Docker Desktop 28.2 on arm64. The project calls no paid APIs and needs no keys. Other workloads
were running on the machine during the benchmarks. Each results file records the host CPU idle
before every point and a coarse host manifest, and each benchmark ran as the only heavy job on
the machine at the time.

## Tests

| check | what it covers | result |
|---|---|---|
| `make test` | controller rules and simulator bounds (20 seeds), scheduler, KV, sampling, detok, probes, OpenAI SDK and SSE shape, WebSockets, metrics | pass |
| `make test-model` | SmolLM2-135M-Instruct @ `12fd25f7`, CPU fp32 | 4 pass |
| `lhai models pull` | pinned revision into `.models/`, 8 files against `models.lock` | match |

The model tests check the following:

- Batched and one-at-a-time greedy decoding give identical output for all 32 tokens of five
  mixed-length prompts, including a request that joins mid-stream.
- Logits after a merge match a solo run to 1e-4.
- A preempted request's tokens match an uninterrupted run.
- An error injected in layer 17 is cropped, and the next step matches.

## Offline and isolation

`make offline-check` was run with `LHAI_OFFLINE_RUN` set to a sandbox-exec wrapper that denies
outbound network except to localhost.

- `models verify` passed.
- The egress canary saw EPERM for 1.1.1.1, api.openai.com and huggingface.co, both from the CLI
  process and from inside the running server (`/v1/admin/egress`).
- A REST completion and a WebSocket generation both worked.
- Without the wrapper, the same canary connects to all three hosts, so the check isn't vacuous.

The compose stack came up with Prometheus reporting the server target `up` and Grafana listing
the provisioned dashboard. Inside the server container, `lhai egress-check` found every target
unreachable, and `wget` to 1.1.1.1 failed from the Prometheus and Grafana containers. The stack
was torn down after each use.

## Benchmarks

The Apple GPU (MPS) sweep in `bench/results/mps-native.json` is a rerun made after the
throughput counting was fixed: throughput is now the server's generated-token counter over the
measurement window. The memory probe saw about a 3 GiB MPS limit, because other programs held
unified memory and swap. The preflight warning is saved in the file.

The Docker CPU sweep in `bench/results/cpu-docker.json` is a trimmed rerun (fixed:1, fixed:8 and
aimd at 1, 4 and 8 clients, 30 s each). It followed the fix that keeps n_min samples of an epoch
when decode steps are slow. The earlier full sweep had a controller that never acted on CPU,
because its 5 s window never held 20 steps of 300+ ms. That sweep is superseded. In the rerun, AIMD kept L at 16 on every point. With at most 8 clients,
the batch was never saturated and latency stayed under the SLO, so the CPU sweep shows
throughput scaling with batch size. It says nothing about the controller.

## Memory pressure

Both memory-pressure runs used a 1.5 GB cgroup, 32 clients and 512 tokens per request. Neither
controller mode was OOM-killed in either run.

- **Before the KV-ceiling fix** (`bench/results/cpu-mempressure-before-ceiling-fix.json`, 90 s per
  mode): fixed:32 completed 41 requests with no errors. aimd completed 28, and 12 requests timed
  out. AIMD's batch limit went 13 -> 5 -> 1 in the first seconds and stayed at 1-2. The ceiling
  assumed every row was as long as the longest recent request.
- **After the fix** (`bench/results/cpu-mempressure.json`, 60 s per mode): fixed:32 completed 38
  requests with no errors. aimd completed 28, and 9 timed out. AIMD's limit dropped to 1 within
  3 s and stayed between 1 and 11. Headroom sat under the 20% high watermark most of the time.

AIMD shows no advantage in this scenario. The fixed ceiling estimate is covered by
`test_kv_ceiling_does_not_undercut_admission`.

## Simulator

`scripts/sim_report.py` writes `docs/results/sim.json` and the S1-S5 plots. The S3 floor bound
(L >= 0.5 b*) holds on 19 of 20 seeds, and the worst seed reaches 0.469 b*. That is reported,
not tuned.
