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
import plotstyle  # noqa: E402

plt = plotstyle.plt
FIG = HERE / "figures"
TARGET_NAMES = {"cpu-docker": "CPU, Docker (linux/arm64 VM)",
                "mps-native": "Apple GPU (MPS), native"}


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
    lease = m.get("compute_lease", {})
    others = [line.split("\t")[0] for line in m.get("docker_ps", []) if line]
    return (f"{h.get('chip')}, {h.get('ram_gb')} GB, {h.get('os')}; power: {h.get('power')}; "
            f"lease holder: {lease.get('holder') or lease.get('holder_env')}; "
            f"containers running at record time: {', '.join(others) or 'none'}")


def sweep_table(d: dict) -> list[str]:
    rows = ["| controller | clients | req/s | output tok/s | TTFT p50 / p95 (ms) | "
            "request TPOT p50 / p95 (ms) | SLO attainment | errors | host load (1 min) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in d["runs"]:
        if r.get("skipped"):
            rows.append(f"| {r['mode']} | {r['concurrency']} | skipped: {r['skipped']} |"
                        " | | | | | |")
            continue
        slo = "n/a" if r["slo_attainment"] is None else f"{r['slo_attainment'] * 100:.0f}%"
        load = "/".join(str(x) for x in r.get("host_load_1m", [])) or "n/a"
        rows.append(
            f"| {r['mode']} | {r['concurrency']} | {r['req_per_s']:.2f} | "
            f"{r['out_tok_per_s']:.0f} | {fmt(r['ttft_p50_ms'], 0)} / {fmt(r['ttft_p95_ms'], 0)} | "
            f"{fmt(r['req_tpot_p50_ms'])} / {fmt(r['req_tpot_p95_ms'])} | {slo} | "
            f"{r['errors']} | {load} |")
    return rows


def best(d: dict, mode: str, key: str) -> dict | None:
    rs = [r for r in d["runs"] if r.get("mode") == mode and not r.get("skipped")]
    return max(rs, key=lambda r: r[key]) if rs else None


def plot_sweep(d: dict, label: str) -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for mode, color in plotstyle.MODE_COLORS.items():
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
    a2.set_xlabel("seconds into the run")
    a1.set_title(f"AIMD, {TARGET_NAMES.get(label, label)}, {r['concurrency']} clients",
                 loc="left")
    fig.tight_layout()
    name = f"aimd_trace_{label}.png"
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    return name


def plot_mem(d: dict) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for r in d["runs"]:
        tr = r.get("trace") or []
        color = plotstyle.MODE_COLORS.get(r["mode"], plotstyle.INK2)
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
    ax.set_title("Memory pressure: same load, two controllers", loc="left")
    fig.tight_layout()
    name = "mempressure.png"
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
    for label in ("mps-native", "cpu-docker"):
        d = load(label)
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
        out += sweep_table(d)
        out += ["", f"![{label} sweep](figures/{plot_sweep(d, label)})", ""]
        tr = plot_trace(d, label)
        if tr:
            out += [f"![{label} AIMD trace](figures/{tr})", ""]
        for mode in ("fixed:1", "fixed:32", "aimd"):
            b = best(d, mode, "out_tok_per_s")
            if b:
                slo = "n/a" if b["slo_attainment"] is None else f"{b['slo_attainment']:.0%}"
                out.append(f"- {mode}: peak {b['out_tok_per_s']:.0f} tok/s at "
                           f"{b['concurrency']} clients, SLO attainment {slo} there.")
        out.append("")
    out += ["## NVIDIA CUDA", "", "Not measured. This machine has no NVIDIA GPU. The CUDA "
            "probe, the cu126 image and `docker-compose.gpu.yml` are untested.", ""]
    mp = load("cpu-mempressure")
    out += ["## Memory pressure (CPU, Docker)", ""]
    if mp is None:
        out += ["Not run yet.", ""]
    else:
        c = mp["config"]
        out += [f"Server container limited to {c['mem_limit']} (cgroup), {c['concurrency']} "
                f"clients, max_tokens {c['max_tokens']}, {c['duration']:.0f} s, SLO set loose "
                f"({c['slo_tpot_ms']:.0f} ms) so only memory matters. The server is recreated "
                "before each mode.", "",
                "| controller | completed | errors | container after the run | peak memory seen "
                "by the probe | output tok/s |", "|---|---:|---:|---|---:|---:|"]
        for r in mp["runs"]:
            st = r["container"]
            state = ("OOM-killed" if st.get("oom_killed") else st.get("status", "?"))
            peak = r.get("peak_mem_used_bytes")
            out.append(f"| {r['mode']} | {r['completed']} | {r['errors']} | {state} "
                       f"(exit {st.get('exit_code')}) | "
                       f"{fmt(None if peak is None else peak / 2**30, 2, ' GiB')} | "
                       f"{r['out_tok_per_s']:.0f} |")
        out += ["", f"![memory pressure](figures/{plot_mem(mp)})", "",
                f"Host: {host_line(mp['manifest'])}", ""]
    (HERE / "RESULTS.md").write_text("\n".join(out))
    print("wrote bench/RESULTS.md")


if __name__ == "__main__":
    main()
