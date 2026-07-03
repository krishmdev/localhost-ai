"""Turn bench/results/*.json into bench/RESULTS.md and bench/figures/*.png. Every number in
RESULTS.md comes from those files; nothing is typed in by hand.

    uv run python bench/report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
import features_report  # noqa: E402
import plotstyle  # noqa: E402

plt = plotstyle.plt
FIG = HERE / "figures"
TARGET_NAMES = {"cpu-docker": "CPU, Docker (linux/arm64 VM; contended shared host, rough)",
                "mps-native": "Apple GPU (MPS), native",
                "mlx-qwen2.5-0.5b": "Apple GPU via MLX, Qwen2.5-0.5B-Instruct 4-bit",
                "mlx-gemma-4-e4b": "Apple GPU via MLX, Gemma 4 E4B-it 4-bit",
                "mlx-qwen3.5-9b": "Apple GPU via MLX, Qwen3.5-9B 4-bit"}
SWEEPS = ("mps-native", "mlx-qwen2.5-0.5b", "mlx-gemma-4-e4b", "mlx-qwen3.5-9b", "cpu-docker")


def load(name: str) -> dict | None:
    p = HERE / "results" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, nd=1, suffix=""):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}{suffix}"
    return f"{v}{suffix}"


def host_line(m: dict) -> str:
    h = m.get("host", {})
    parts = [h.get("chip") or h.get("machine"), f"{h.get('ram_gb')} GB" if h.get("ram_gb")
             else None, h.get("os"), f"power: {h['power']}" if h.get("power") else None]
    return ", ".join(p for p in parts if p)


def sweep_table(d: dict) -> list[str]:
    rows = ["| controller | clients | completed | req/s | output tok/s | TTFT p50 / p95 (ms) | "
            "request TPOT p50 / p95 (ms) | SLO attainment | errors | host CPU idle before |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in d["runs"]:
        if r.get("skipped"):
            rows.append(f"| {r['mode']} | {r['concurrency']} | skipped: {r['skipped']} |"
                        " | | | | | | |")
            continue
        slo = "n/a" if r["slo_attainment"] is None else f"{r['slo_attainment'] * 100:.0f}%"
        idle = r.get("host_cpu_idle_before")
        load = "n/a" if idle is None else f"{idle:.0f}%"
        rows.append(
            f"| {r['mode']} | {r['concurrency']} | {r['completed']} | {r['req_per_s']:.2f} | "
            f"{r['out_tok_per_s']:.0f} | {fmt(r['ttft_p50_ms'], 0)} / {fmt(r['ttft_p95_ms'], 0)} | "
            f"{fmt(r['req_tpot_p50_ms'])} / {fmt(r['req_tpot_p95_ms'])} | {slo} | "
            f"{r['errors']} | {load} |")
    return rows


def describe_range(vals: list[int]) -> str:
    if not vals:
        return "no trace"
    return f"stayed at {vals[0]}" if len(set(vals)) == 1 else f"{min(vals)} to {max(vals)}"


def best(d: dict, mode: str, key: str) -> dict | None:
    rs = [r for r in d["runs"] if r.get("mode") == mode and not r.get("skipped")]
    return max(rs, key=lambda r: r[key]) if rs else None


def plot_sweep(d: dict, label: str) -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for mode in plotstyle.modes_in(d["runs"]):
        color = plotstyle.mode_color(mode)
        rs = [r for r in d["runs"] if r.get("mode") == mode and not r.get("skipped")]
        if not rs:
            continue
        xs = [r["concurrency"] for r in rs]
        a1.plot(xs, [r["out_tok_per_s"] for r in rs], color=color, lw=2, marker="o", ms=5,
                label=mode)
        a2.plot(xs, [r["req_tpot_p95_ms"] for r in rs], color=color, lw=2, marker="o", ms=5,
                label=mode)
    slo = d.get("slo_tpot_ms")
    if slo:
        a2.axhline(slo, color=plotstyle.INK2, lw=1)
        a2.text(a2.get_xlim()[0], slo, f" SLO {slo:.0f} ms", color=plotstyle.INK2, fontsize=8,
                va="bottom")
    for a in (a1, a2):
        a.set_xscale("log", base=2)
        a.set_xlabel("concurrent clients")
        a.set_ylim(bottom=0)
        a.xaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    a1.set_ylabel("output tokens / s")
    a1.set_title("Throughput", loc="left")
    a2.set_ylabel("per-request TPOT p95 (ms)")
    a2.set_title("Latency per output token", loc="left")
    a1.legend(frameon=False, fontsize=9)
    fig.suptitle(TARGET_NAMES.get(label, label), x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    name = f"sweep_{label}.png"
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    return name


def plot_trace(d: dict, label: str) -> str | None:
    rs = [r for r in d["runs"] if r.get("mode") == "aimd" and r.get("trace")]
    if not rs:
        return None
    r = max(rs, key=lambda r: r["concurrency"])
    tr = r["trace"]
    ts = [p["t"] for p in tr]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(ts, [p["batch_limit"] for p in tr], color=plotstyle.BLUE, lw=2,
            drawstyle="steps-post", label="batch limit L")
    a1.plot(ts, [p["running"] for p in tr], color=plotstyle.ORANGE, lw=1,
            drawstyle="steps-post", label="running rows")
    a1.set_ylabel("rows")
    a1.set_ylim(bottom=0)
    a1.legend(frameon=False, fontsize=9, loc="lower right", ncols=2)
    a2.plot(ts, [p.get("decode_step_p95_ms") for p in tr], color=plotstyle.BLUE, lw=1.5,
            label="p95 decode step")
    a2.axhline(r["slo_tpot_ms"], color=plotstyle.INK2, lw=1)
    a2.text(ts[0] if ts else 0, r["slo_tpot_ms"], " SLO", color=plotstyle.INK2, fontsize=8,
            va="bottom")
    a2.set_ylabel("ms")
    a2.set_ylim(bottom=0)
    a2.legend(frameon=False, fontsize=9, loc="lower right")
    a2.set_xlabel("seconds into the run (clients stop at the end; the batch drains)")
    a1.set_title(f"AIMD, {TARGET_NAMES.get(label, label)}, {r['concurrency']} clients",
                 loc="left")
    fig.tight_layout()
    name = f"aimd_trace_{label}.png"
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    return name


def plot_mem(d: dict, name: str = "mempressure.png", title: str = "") -> str:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for r in d["runs"]:
        tr = r.get("trace") or []
        color = plotstyle.mode_color(r["mode"])
        ax.plot([p["t"] for p in tr], [(p["mem_used_bytes"] or 0) / 2**30 for p in tr],
                color=color, lw=2, label=r["mode"] + (" (OOM-killed)" if r["container"].get(
                    "oom_killed") else ""))
    lim = d["config"]["mem_limit"]
    limit = float(lim[:-1]) / 1024 if lim.endswith("m") else float(lim[:-1])
    ax.axhline(limit, color=plotstyle.INK2, lw=1)
    ax.text(0, limit, f" container limit {lim}", color=plotstyle.INK2, fontsize=8, va="bottom")
    ax.set_ylim(0, limit * 1.15)
    ax.set_ylabel("container memory (GiB)")
    ax.set_xlabel("seconds into the run")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Memory pressure: same load, two controllers" + title, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    return name


def main() -> None:
    FIG.mkdir(exist_ok=True)
    out = ["# Benchmark results", "",
           "Generated by `bench/report.py` from `bench/results/*.json`; don't edit by hand.", "",
           "Method: `bench/loadgen.py` runs a closed loop of N streaming clients for a fixed "
           "duration after a warm-up, per controller mode. Prompts come from "
           "`bench/prompts.jsonl`, sampled with temperature 0.7 and a per-request seed. "
           "TTFT is the time to the first content chunk as seen by the client. Request TPOT is "
           "client-side: (last token time - first token time) / (tokens - 1) for each request, "
           "so it includes prefill stalls from other requests joining the batch. SLO attainment "
           "is the share of requests whose request TPOT is at or under the SLO. The SLO is "
           "calibrated per target before the sweep as a fixed multiple of the median request "
           "TPOT of a single client with batch size 1.", ""]
    for label in SWEEPS:
        d = load(label)
        if d is None and label.startswith("mlx-"):
            continue
        out += [f"## {TARGET_NAMES[label]}", ""]
        if d is None:
            out += ["Not run yet.", ""]
            continue
        cal = d.get("calibration") or {}
        pre = d.get("preflight") or {}
        mem = pre.get("memory") or {}
        out += [f"Model `{d['model']['id']}` ({d['model'].get('root')}), "
                f"max_tokens {d['config']['max_tokens']}, {d['config']['duration']:.0f} s per "
                f"point after {d['config']['warmup']:.0f} s warm-up. "
                f"SLO {d['slo_tpot_ms']} ms = {cal.get('factor')} x "
                f"{cal.get('baseline_tpot_p50_ms')} ms single-client median.",
                "",
                "Memory probe before the run: limit "
                f"{fmt((mem.get('limit_bytes') or 0) / 2**30, 2)} GiB, headroom "
                f"{fmt(mem.get('headroom_frac'), 2)} ({mem.get('source')}). "
                f"Host swap: {pre.get('host_swap')}."
                + (" Warnings: " + "; ".join(pre["warnings"]) if pre.get("warnings") else ""),
                "", f"Host: {host_line(d['manifest'])}", ""]
        if pre.get("backend") == "mlx":
            out += [mlx_memory_note(d), ""]
        if label == "cpu-docker":
            idles = [r["host_cpu_idle_before"] for r in d["runs"]
                     if r.get("host_cpu_idle_before") is not None]
            ok = [r for r in d["runs"] if not r.get("skipped")]
            measured = [r for r in ok if r["slo_attainment"] is not None]
            meets = sum(r["slo_attainment"] >= 0.99 for r in measured)
            unmeasured = len(ok) - len(measured)
            aimd_ok = all(r["slo_attainment"] >= 0.99 for r in measured if r["mode"] == "aimd")
            aimd_ls = {p["batch_limit"] for r in ok if r["mode"] == "aimd"
                       for p in r.get("trace", []) if p.get("batch_limit") is not None}
            gap = (f" ({unmeasured} point(s) completed no request inside the window)"
                   if unmeasured else "")
            inert = (len(aimd_ls) == 1 and aimd_ok
                     and max(r["concurrency"] for r in ok) <= min(aimd_ls))
            text = ("These numbers are rough. Other workloads were running on the machine and "
                    "in the Docker VM during this sweep (host CPU idle before each point ranged "
                    f"{min(idles):.0f}-{max(idles):.0f}%). The single-client baseline used for "
                    "calibration was measured under that load, which sets the SLO "
                    f"({d['slo_tpot_ms']} ms); {meets} of {len(measured)} points with completed "
                    f"requests meet it for 99%+ of them{gap}. Across the AIMD runs the batch "
                    f"limit took {len(aimd_ls)} distinct value(s): "
                    f"{describe_range(sorted(aimd_ls))}.")
            if inert:
                text += (" The sweep never had more clients than AIMD's starting limit, so its "
                         "batch was never saturated and every AIMD point stayed under the SLO: "
                         "there was nothing to adapt to, and AIMD behaved like a fixed batch of "
                         f"{min(aimd_ls)} here. This sweep is not evidence for or against the "
                         "controller.")
            out += [text, ""]
        out += sweep_table(d)
        out += ["", f"![{label} sweep](figures/{plot_sweep(d, label)})", ""]
        tr = plot_trace(d, label)
        if tr:
            out += [f"![{label} AIMD trace](figures/{tr})", ""]
        for mode in plotstyle.modes_in(d["runs"]):
            b = best(d, mode, "out_tok_per_s")
            if b:
                slo = "n/a" if b["slo_attainment"] is None else f"{b['slo_attainment']:.0%}"
                out.append(f"- {mode}: peak throughput was {b['out_tok_per_s']:.0f} tok/s "
                           f"at {b['concurrency']} clients; {slo} met the SLO at that point.")
        out.append("")
    out += mlx_direct_section()
    out += prefix_section()
    out += ["## NVIDIA CUDA", "", "Not measured. This machine has no NVIDIA GPU. The CUDA "
            "probe, the cu126 image and `docker-compose.gpu.yml` are untested.", ""]
    out += ["## Memory pressure (CPU, Docker)", ""]
    for key, heading, fig in (
            ("cpu-mempressure-guard-pair",
             "Same-session pair with the active-row guard (current code)",
             "mempressure_guard_pair.png"),
            ("cpu-mempressure", "After the KV-ceiling fix, before the guard", "mempressure.png"),
            ("cpu-mempressure-controller-guard",
             "AIMD only, with the active-row guard (separate session, no fixed baseline)",
             "mempressure_controller_guard.png"),
            ("cpu-mempressure-before-ceiling-fix", "Before the KV-ceiling fix (history)",
             "mempressure_before_fix.png")):
        mp = load(key)
        out += [f"### {heading}", ""]
        if mp is None:
            out += ["Not run yet.", ""]
            continue
        c = mp["config"]
        if len(mp["runs"]) == 1:
            r0 = mp["runs"][0]
            out += [f"Only `{r0['mode']}` ran in this session, so there is no same-session "
                    "baseline; the other runs in this section used different host windows and "
                    "aren't comparable to it. Requests could keep draining after the load window "
                    f"ends, so TTFT p95 ({fmt(r0['ttft_p95_ms'] / 1e3, 0)} s) includes long queue "
                    "waits.", ""]
        if mp.get("throughput_source") == "server_counter":
            out += ["Throughput is the server's token counter over the load window. "
                    "\"Completed\" and TTFT only count requests that finished inside that "
                    "window; with 512-token requests on a slow CPU most were still running at "
                    "the deadline, so those samples are small.", ""]
        else:
            out += ["Throughput in this run is the older client-side count (tokens of requests "
                    "that started after the warm-up), not the server counter used above.", ""]
        out += [f"Server container limited to {c['mem_limit']} (cgroup), {c['concurrency']} "
                f"clients, max_tokens {c['max_tokens']}, {c['duration']:.0f} s, SLO set loose "
                f"({c['slo_tpot_ms']:.0f} ms) so only memory matters. The server is recreated "
                "before each mode.", "",
                "| controller | completed | errors | container after the run | peak memory seen "
                "by the probe | output tok/s | TTFT p95 |", "|---|---:|---:|---|---:|---:|---:|"]
        for r in mp["runs"]:
            st = r["container"]
            state = "OOM-killed" if st.get("oom_killed") else "not OOM-killed"
            peak = r.get("peak_mem_used_bytes")
            out.append(f"| {r['mode']} | {r['completed']} | {r['errors']} | {state} "
                       "| "
                       f"{fmt(None if peak is None else peak / 2**30, 2, ' GiB')} | "
                       f"{r['out_tok_per_s']:.0f} | "
                       f"{fmt(r['ttft_p95_ms'] / 1e3 if r['ttft_p95_ms'] else None, 0, ' s')} |")
        out.append("")
        for r in mp["runs"]:
            tr = r.get("trace") or []
            ls = [p["batch_limit"] for p in tr if p.get("batch_limit") is not None]
            rn = [p["running"] for p in tr if p.get("running") is not None]
            hr = [p["headroom_frac"] for p in tr if p.get("headroom_frac") is not None]
            acts: dict[str, int] = {}
            for p in tr:
                if p.get("action") and p["action"] != "hold":
                    acts[p["action"]] = acts.get(p["action"], 0) + 1
            out.append(f"- {r['mode']}: batch limit {describe_range(ls)}, running rows at most "
                       f"{max(rn)}, headroom {min(hr):.0%}-{max(hr):.0%}, non-hold telemetry "
                       f"samples {acts or 'none'}, error kinds {r['error_kinds'] or 'none'}.")
        png = plot_mem(mp, fig, f" ({heading.lower()})")
        out += ["", f"![memory pressure](figures/{png})", "",
                f"Host: {host_line(mp['manifest'])}", ""]
    out += features_report.sections(host_line)  # other servers, sampler, response_format
    (HERE / "RESULTS.md").write_text("\n".join(out))
    traces = [f"![AIMD trace, {TARGET_NAMES[lbl]}](../bench/figures/aimd_trace_{lbl}.png)"
              for lbl in ("mps-native", "cpu-docker") if (FIG / f"aimd_trace_{lbl}.png").exists()]
    write_section(ROOT / "docs" / "controller.md", "trace", "\n".join(traces))
    write_section(ROOT / "README.md", "results", "\n".join(readme_summary()))
    print("wrote bench/RESULTS.md")


def mlx_memory_note(d: dict) -> str:
    """What the MLX probe saw across the sweep: the model's footprint and how close the served
    system came to the limit."""
    pre = d["preflight"]
    mem = pre.get("memory") or {}
    pts = [(p["mem_used_bytes"], p.get("headroom_frac")) for r in d["runs"]
           for p in r.get("trace", []) if p.get("mem_used_bytes") is not None]
    peak = max(u for u, _ in pts) if pts else None
    low = min(h for _, h in pts if h is not None) if pts else None
    kv = int(pre.get("kv_bytes_per_token") or 0)
    row = int(pre.get("row_state_bytes") or 0)
    text = (f"MLX backend, {pre.get('quant')} weights, {pre.get('dtype')} activations. KV "
            f"{kv / 1024:.0f} KiB per token")
    text += (f", plus {row / 2**20:.0f} MiB of recurrent state per row" if row else "") + ". "
    text += (f"MLX active memory before the run: {fmt((mem.get('used_bytes') or 0) / 2**30, 2)} "
             "GiB.")
    if peak is not None:
        text += (f" Highest active memory in any telemetry sample during the sweep: "
                 f"{peak / 2**30:.2f} GiB; lowest headroom {low:.0%} of the probe's limit.")
    return text


def mlx_direct_section() -> list[str]:
    d = load("mlx-direct")
    if not d:
        return []
    out = ["## MLX presets, runner only (no HTTP, no controller)", "",
           "`bench/mlx_direct.py` loads the preset, then prefills a fixed batch of prompts from "
           "`bench/prompts.jsonl` and runs 128 greedy decode steps with EOS ignored. Memory is "
           "MLX's allocator (active = live arrays including weights; peak = high-water mark "
           "since the last reset).", "",
           "| preset | load | active after load | peak during load | KV per token | "
           "recurrent state per row | Metal working set | answer to \"capital of France\" |",
           "|---|---:|---:|---:|---:|---:|---:|---|"]
    for e in d.values():
        ld = e["load"]
        row = e["row_state_bytes"]
        out.append(f"| {e['preset']} | {ld['load_s']:.1f} s | {ld['active_gib']:.2f} GiB | "
                   f"{ld['peak_gib']:.2f} GiB | {e['kv_bytes_per_token'] / 1024:.0f} KiB | "
                   f"{f'{row / 2**20:.0f} MiB' if row else '0'} | "
                   f"{e['metal_recommended_working_set_gib']:.2f} GiB | "
                   f"{e['answer'].strip()!r} |")
    out += ["", "| preset | batch | prefill | decode step | step vs batch 1 | "
            "decode tok/s (all rows) | peak memory |", "|---|---:|---:|---:|---:|---:|---:|"]
    for e in d.values():
        one = next((r["decode_step_ms"] for r in e["runs"] if r["batch"] == 1), None)
        for r in e["runs"]:
            ratio = f"{r['decode_step_ms'] / one:.1f}x" if one else "n/a"
            out.append(f"| {e['preset']} | {r['batch']} | {r['prefill_s']:.2f} s | "
                       f"{r['decode_step_ms']:.1f} ms | {ratio} | {r['decode_tok_per_s']:.0f} | "
                       f"{r['peak_gib']:.2f} GiB |")
    hosts = {host_line(e["manifest"]) for e in d.values() if e.get("manifest")}
    out += ["", "Host: " + "; ".join(sorted(hosts)), ""]
    return out + mlx_qmm_section()


def mlx_qmm_section() -> list[str]:
    q = load("mlx-qmm")
    if not q:
        return []
    runs = q["runs"]
    one = runs[0]["ms"]
    out = ["### 4-bit matmul cost against rows", "",
           f"`bench/mlx_qmm.py` times `mx.quantized_matmul` (MLX {q['mlx']}, {q['bits']}-bit, "
           f"group size {q['group_size']}) on {q['mats']} distinct {q['n']} x {q['k']} weight "
           f"matrices ({q['weight_bytes'] / 1e9:.2f} GB, more than the GPU caches hold), "
           "with one token per row as in a decode step. \"Weight GB/s\" is the weight bytes "
           "divided by the time, so it counts each weight once however many rows share it.", "",
           "| rows | time for all matrices | vs 1 row | weight GB/s |", "|---:|---:|---:|---:|"]
    for r in runs:
        out.append(f"| {r['rows']} | {r['ms']:.2f} ms | {r['ms'] / one:.1f}x | "
                   f"{r['weight_gb_per_s']:.0f} |")
    # the longest run of row counts whose cost is still about proportional to the rows
    lin = 0
    for r in runs[1:]:
        if r["ms"] / one < 0.8 * r["rows"]:
            break
        lin = r["rows"]
    rest = [r for r in runs if r["rows"] > lin]
    if lin:
        out += ["", f"Up to {lin} rows the time grows about in proportion to the row count, so "
                "each extra row costs close to another full read of the weights"
                + (f"; past that it stays at {min(r['ms'] for r in rest):.0f}-"
                   f"{max(r['ms'] for r in rest):.0f} ms" if rest else "")
                + ". The decode step of a large preset follows the same shape in the table "
                "above, so on this machine batching those pays off only past that point."]
    out += ["", f"Host: {host_line(q['manifest'])}", ""]
    return out


PREFIX_RUNS = (("smollm2-135m", "SmolLM2-135M on the Apple GPU (MPS, torch runner)"),
               ("qwen3.5-9b", "Qwen3.5-9B 4-bit on the Apple GPU (MLX runner)"))


def prefix_section() -> list[str]:
    """Prefix caching off vs on, same sweep, every request carrying the same system prompt."""
    pairs = [(lbl, name, load(f"prefix-{lbl}-off"), load(f"prefix-{lbl}-on"))
             for lbl, name in PREFIX_RUNS]
    pairs = [p for p in pairs if p[2] and p[3]]
    if not pairs:
        return []
    out = ["## Prefix caching, shared system prompt", "",
           "`bench/prefix.sh` runs the same sweep twice, with `LHAI_PREFIX_CACHE` off and on. "
           "Every request sends `bench/system_prompt.txt` as its system message and one of the "
           "usual prompts as the user message, so the prompts share a long common prefix. The "
           "server finds it after the first two requests and stores it once; later rows "
           "prefill only their own suffix. TTFT here includes the wait in the queue.", ""]
    for _, name, off, on in pairs:
        c = on["config"]
        tr = [p for r in on["runs"] for p in r.get("trace", []) if "prefix_bytes" in p]
        size = max((p["prefix_bytes"] for p in tr), default=None)
        hit = max((p.get("prefix_hit_tokens") or 0 for p in tr), default=None)
        out += [f"### {name}", "",
                f"Model `{on['model']['id']}`, max_tokens {c['max_tokens']}, "
                f"{c['duration']:.0f} s per point, modes {c['modes']}."
                + (f" Stored prefixes held up to {size / 2**20:.1f} MiB; by the end "
                   f"{hit:,} prompt tokens had been served from them." if size else ""), "",
                "| controller | clients | TTFT p50 off / on (ms) | TTFT p95 off / on (ms) | "
                "output tok/s off / on | completed off / on |", "|---|---:|---:|---:|---:|---:|"]
        key = {(r["mode"], r["concurrency"]): r for r in on["runs"] if not r.get("skipped")}
        for r0 in off["runs"]:
            r1 = key.get((r0.get("mode"), r0.get("concurrency")))
            if r0.get("skipped") or r1 is None:
                continue
            out.append(f"| {r0['mode']} | {r0['concurrency']} | "
                       f"{fmt(r0['ttft_p50_ms'], 0)} / {fmt(r1['ttft_p50_ms'], 0)} | "
                       f"{fmt(r0['ttft_p95_ms'], 0)} / {fmt(r1['ttft_p95_ms'], 0)} | "
                       f"{r0['out_tok_per_s']:.0f} / {r1['out_tok_per_s']:.0f} | "
                       f"{r0['completed']} / {r1['completed']} |")
        out += ["", f"Host: {host_line(on['manifest'])}", ""]
    return out


def write_section(path: Path, name: str, body: str) -> None:
    text = path.read_text()
    begin, end = f"<!-- {name}:begin -->", f"<!-- {name}:end -->"
    if begin not in text:
        return
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{begin}\n{body}\n{end}{tail}")


SWEEP_OF = {"qwen2.5-0.5b-mlx4": "mlx-qwen2.5-0.5b", "gemma-4-e4b-mlx4": "mlx-gemma-4-e4b",
            "qwen3.5-9b-mlx4": "mlx-qwen3.5-9b"}


def readme_mlx() -> list[str]:
    """One line per MLX preset: footprint and decode speed from mlx_direct, and the best served
    point (AIMD) from its sweep."""
    d = load("mlx-direct")
    if not d:
        return []
    rows = ["", "MLX 4-bit presets on the same Mac (runner-only decode, 128 greedy steps; served "
            "numbers from each preset's sweep in RESULTS.md):", "",
            "| preset | memory after load | decode tok/s, 1 row | decode tok/s, 16 rows | "
            "served AIMD peak |", "|---|---:|---:|---:|---|"]
    for e in d.values():
        by = {r["batch"]: r for r in e["runs"]}
        one, sixteen = by.get(1), by.get(16)
        sw = load(SWEEP_OF.get(e["preset"], ""))
        served = "not run"
        if sw:
            b = best(sw, "aimd", "out_tok_per_s")
            if b:
                att = "n/a" if b["slo_attainment"] is None else f"{b['slo_attainment']:.0%}"
                served = (f"{b['out_tok_per_s']:.0f} tok/s at {b['concurrency']} clients, "
                          f"{att} within the {sw['slo_tpot_ms']} ms SLO")
        rows.append(f"| {e['preset']} | {e['load']['active_gib']:.2f} GiB | "
                    f"{fmt(one and one['decode_tok_per_s'], 0)} | "
                    f"{fmt(sixteen and sixteen['decode_tok_per_s'], 0)} | {served} |")
    return rows


def readme_summary() -> list[str]:
    """Headline for the README: the Apple GPU sweep at its highest concurrency. The contended
    CPU sweep stays in RESULTS.md only."""
    rows = []
    d = load("mps-native")
    if d is not None:
        top = max(r["concurrency"] for r in d["runs"] if not r.get("skipped"))
        rows += [f"{TARGET_NAMES['mps-native']}, {top} clients, SLO {d['slo_tpot_ms']} ms per "
                 "token:", "",
                 "| controller | output tok/s | requests completed | request TPOT p95 | "
                 "SLO attainment* | TTFT p95 |", "|---|---:|---:|---:|---:|---:|"]
        at = {r["mode"]: r for r in d["runs"]
              if r.get("concurrency") == top and not r.get("skipped")}
        for mode in plotstyle.modes_in(d["runs"]):
            r = at.get(mode)
            if r is None:
                continue
            att = "n/a" if r["slo_attainment"] is None else f"{r['slo_attainment']:.0%}"
            rows.append(f"| {mode} | {r['out_tok_per_s']:.0f} | {r['completed']} | "
                        f"{fmt(r['req_tpot_p95_ms'])} ms | {att} | "
                        f"{fmt(r['ttft_p95_ms'] / 1e3 if r['ttft_p95_ms'] else None, 1)} s |")
        a = at.get("aimd")
        if a:
            ls = sorted({p["batch_limit"] for p in a.get("trace", [])
                         if p.get("batch_limit") is not None})
            rows += ["", f"AIMD's batch limit during that run: {describe_range(ls)}."]
        rows += ["", "*Share of completed requests whose per-token latency (TPOT) met the SLO. It "
                 "ignores time to first token, which grows with queueing when the batch is "
                 "capped; that's the TTFT column. Throughput is the server's generated-token "
                 "count over the measurement window."]
    rows += readme_mlx()
    rows += ["", "NVIDIA CUDA: not measured (no NVIDIA GPU here). The Docker CPU sweep ran on a "
             "contended host; its rough numbers are in RESULTS.md only."]
    mp = load("cpu-mempressure-guard-pair")
    if mp:
        rows += ["", "Memory pressure, same-session pair on the current code (CPU container "
                 f"capped at {mp['config']['mem_limit']}, {mp['config']['concurrency']} clients, "
                 f"{mp['config']['max_tokens']} tokens each, {mp['config']['duration']:.0f} s):",
                 ""]
        for r in mp["runs"]:
            st = r["container"]
            outcome = "OOM-killed by the kernel" if st.get("oom_killed") else "not OOM-killed"
            rows.append(f"- {r['mode']}: {outcome}, {r['out_tok_per_s']:.0f} tok/s, "
                        f"{r['completed']} requests finished inside the window, "
                        f"{r['errors']} failed.")
        rows += ["", "Earlier pressure runs (before the guard, and an AIMD-only run) are in "
                 "RESULTS.md; they come from different host windows and aren't compared here."]
    return rows

if __name__ == "__main__":
    main()
