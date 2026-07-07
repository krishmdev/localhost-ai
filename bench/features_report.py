"""RESULTS.md sections for the baseline comparison against other local servers, the sampler
microbenchmark and the response_format overhead. Called from report.py; every number comes
from bench/results/*.json."""

from __future__ import annotations

import json
from pathlib import Path

import plotstyle

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
plt = plotstyle.plt

ENGINE_NAMES = {"lhai-aimd": "localhost-ai (AIMD)", "lhai-fixed": "localhost-ai (fixed:32)",
                "mlx_lm": "mlx_lm.server", "llama.cpp": "llama.cpp llama-server",
                "ollama": "Ollama"}
ENGINE_STYLE = {"lhai-aimd": (plotstyle.BLUE, "o"), "lhai-fixed": (plotstyle.ORANGE, "s"),
                "mlx_lm": (plotstyle.AQUA, "^"), "llama.cpp": (plotstyle.YELLOW, "D"),
                "ollama": (plotstyle.MAGENTA, "v")}


def load(name: str) -> dict | None:
    p = HERE / "results" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, nd=1):
    return "n/a" if v is None else f"{v:.{nd}f}"


def gib(b) -> str:
    return "n/a" if not b else f"{b / 2**30:.2f}"


def plot_baseline(d: dict) -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for e in d["engines"]:
        color, marker = ENGINE_STYLE.get(e["name"], (plotstyle.INK2, "o"))
        xs = [p["concurrency"] for p in e["points"]]
        label = ENGINE_NAMES.get(e["name"], e["name"])
        a1.plot(xs, [p["out_tok_per_s"] for p in e["points"]], color=color, lw=2,
                marker=marker, ms=6, label=label)
        a2.plot(xs, [p["tpot_p50_ms"] for p in e["points"]], color=color, lw=2,
                marker=marker, ms=6, label=label)
    for a in (a1, a2):
        a.set_xscale("log", base=2)
        a.set_xlabel("concurrent clients")
        a.set_ylim(bottom=0)
        a.xaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    a1.set_ylabel("output tokens / s")
    a1.set_title("Throughput", loc="left")
    a2.set_ylabel("per-request TPOT p50 (ms)")
    a2.set_title("Latency per output token", loc="left")
    a1.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Qwen2.5-3B-Instruct, 4-bit, same Mac, same client", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    name = "baseline_qwen2.5-3b.png"
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    return name


def baseline_section(host_line) -> list[str]:
    d = load("baseline-qwen2.5-3b")
    if not d:
        return []
    c = d["config"]
    out = ["## Against other local servers (Qwen2.5-3B-Instruct, 4-bit)", "",
           "`bench/baseline.py` starts each server in turn on the same Mac, warms it up, "
           f"and drives it with the same closed loop: {', '.join(map(str, c['concurrency']))} "
           f"streaming clients, {c['duration']:.0f} s per point after a {c['warmup']:.0f} s "
           f"warm-up, max_tokens {c['max_tokens']}, temperature 0.7, top-p 0.95, no seed, "
           "prompts from `bench/prompts.jsonl` in the same order for every engine. Each engine "
           f"is configured for {c['slots']} parallel sequences of {c['ctx']} tokens. "
           "Everything is measured by the client: throughput counts completion tokens whose "
           "chunk arrived inside the window, TTFT is the first content chunk, TPOT is per "
           "request. Memory is the server's process tree, sampled every 250 ms.", ""]
    m = d.get("manifest") or {}
    design = m.get("design") or (m.get("extra") or {}).get("design")
    if design:
        out += [f"Run design: {design}.", ""]
    out += ["Engines and weights:", ""]
    for e in d["engines"]:
        out.append(f"- {ENGINE_NAMES.get(e['name'], e['name'])}: {e['version']}, "
                   f"{e['weights']}, {e['quant']}.")
    out += ["", "What isn't equal:", "",
            "- Quantization. The MLX engines run mlx-community's 4-bit conversion (affine, "
            "group size 64, a 1.74 GB file); llama.cpp and Ollama run the file behind Ollama's "
            "qwen2.5:3b-instruct tag, Q4_K_M (1.93 GB, some tensors kept at higher precision). "
            "They are different quantizations of the same model, so neither output quality "
            "nor speed is strictly like for like.",
            "- Kernels. llama.cpp and Ollama use ggml's Metal kernels, the other three use "
            "MLX's. llama.cpp here is the Homebrew build; Ollama bundles its own ggml-based "
            "runner.",
            "- Prompt caching. llama.cpp, Ollama and mlx_lm.server reuse the KV of a repeated "
            "prompt prefix by default; this server's prefix cache is off by default and was off "
            "here. With 32 prompts repeating, that helps their TTFT.",
            "- Output lengths differ per engine (different samplers and quantizations stop at "
            "different places); the mean completion length is in the table.",
            "- Memory. `footprint` is macOS's phys_footprint (what Activity Monitor shows), "
            "which includes Metal buffers. `rss` also counts resident file-backed pages; for "
            "llama.cpp and Ollama, which mmap the GGUF, it comes out about one weights-file "
            "larger than footprint, most likely the mapped file counted on top of the weights "
            "the GPU uses. Footprint is the column to compare.", "",
            "| engine | clients | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | "
            "completed | mean tokens | peak footprint (GiB) | peak rss (GiB) | errors |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in d["engines"]:
        for p in e["points"]:
            out.append(f"| {ENGINE_NAMES.get(e['name'], e['name'])} | {p['concurrency']} | "
                       f"{p['out_tok_per_s']:.0f} | {fmt(p['ttft_p50_ms'], 0)} / "
                       f"{fmt(p['ttft_p99_ms'], 0)} | {fmt(p['tpot_p50_ms'])} / "
                       f"{fmt(p['tpot_p99_ms'])} | {p['completed']} | "
                       f"{fmt(p['mean_completion_tokens'], 0)} | "
                       f"{gib(p['peak_footprint_bytes'])} | {gib(p['peak_rss_bytes'])} | "
                       f"{p['errors']} |")
    out += ["", f"![baseline comparison](figures/{plot_baseline(d)})", ""]
    out += baseline_summary(d)
    out += ["", f"Host: {host_line(d['manifest'])}", ""]
    return out


def baseline_summary(d: dict) -> list[str]:
    """Plain comparisons at one client and at the highest concurrency, computed here."""
    by = {e["name"]: {p["concurrency"]: p for p in e["points"]} for e in d["engines"]}
    concs = sorted({p["concurrency"] for e in d["engines"] for p in e["points"]})
    lo, hi = concs[0], concs[-1]
    out = []
    for c in (lo, hi):
        pts = {n: v[c] for n, v in by.items() if c in v}
        if not pts:
            continue
        fastest = max(pts, key=lambda n: pts[n]["out_tok_per_s"])
        ranking = ", ".join(f"{ENGINE_NAMES.get(n, n)} {pts[n]['out_tok_per_s']:.0f}"
                            for n in sorted(pts, key=lambda n: -pts[n]["out_tok_per_s"]))
        out.append(f"- {c} client{'s' if c > 1 else ''}: {ranking} tok/s. Fastest: "
                   f"{ENGINE_NAMES.get(fastest, fastest)}.")
    return out


def sampler_section(host_line) -> list[str]:
    d = load("sampler")
    if not d:
        return []
    st = d["timing_setting"]
    out = ["## Sampler CPU time", "",
           "Found while setting up the comparison above: at temperature 0.7 the sampler sorted "
           "the whole vocabulary on the CPU for every sampled row, every step. "
           "`bench/sampler_bench.py` collects real next-token logits from "
           f"`{d['preset']}` ({d['logit_rows']} rows, vocabulary {d['vocab']:,}) and times the "
           f"old full sort against the current sampler, which looks for the top-p nucleus "
           f"among the top {d['candidates']} logits (the median nucleus held "
           f"{d['nucleus_size_p50']} tokens, the largest {d['nucleus_size_max']}). Unseeded "
           "rows draw from the candidates directly. Seeded rows rebuild the full sort's "
           "probability vector from the candidates, so a seed gives the same tokens as before, "
           "and fall back to the full sort when the nucleus doesn't fit, has tied logits or has "
           "a cumulative sum too close to top_p. Timing at temperature "
           f"{st['temperature']}, top_p {st['top_p']}:", "",
           "| rows sampled | full sort (ms) | seeded (ms) | unseeded (ms) |",
           "|---:|---:|---:|---:|"]
    for r in d["runs"]:
        out.append(f"| {r['batch']} | {r['full_sort_ms']:.2f} | {r['seeded_ms']:.2f} | "
                   f"{r['unseeded_ms']:.2f} |")
    out += ["", "Same-seed check: the tokens a seeded row draws, against the full sort, over "
            "the same logits, and how many rows took the full-sort fallback:", "",
            "| setting | rows sampled | tokens compared | mismatches | full-sort rows |",
            "|---|---:|---:|---:|---:|"]
    for name, by_batch in d["same_seed_check"].items():
        for b, c in by_batch.items():
            share = c["full_sorted_rows"] / max(1, c["compared_tokens"])
            out.append(f"| {name} | {b} | {c['compared_tokens']} | {c['mismatches']} | "
                       f"{c['full_sorted_rows']} ({share:.0%}) |")
    out += ["", "With top_p 1.0 and no top-k there is no nucleus to look for, so every "
            "seeded row is sorted in full; unseeded rows at that setting sample the softmax "
            "directly."]
    return out + ["", f"Host: {host_line(d['manifest'])}", ""]


def json_section(host_line) -> list[str]:
    d = load("json-overhead")
    if not d:
        return []
    out = ["## response_format overhead", "",
           "`bench/json_overhead.py` decodes the same prompts through the scheduler with and "
           f"without a JSON-schema constraint on `{d['preset']}` (greedy, up to "
           f"{d['max_tokens']} tokens). The mask column is CPU time spent in the constraint "
           "per row per token: llguidance computing the allowed-token bitmask, unpacking it, "
           "masking the logits and advancing the matcher. Step time is the whole scheduler "
           f"iteration. Building the llguidance tokenizer took {d['first_constraint_s']:.2f} s "
           f"on the first constrained request; compiling the schema for a request takes "
           f"{d['constraint_compile_ms']:.1f} ms after that.", "",
           "| rows | constrained | decode steps | step (ms) | mask per row-token (ms) | "
           "finished with stop | parsed |", "|---:|---|---:|---:|---:|---:|---:|"]
    for r in d["runs"]:
        stops = sum(f == "stop" for f in r["finish"])
        out.append(f"| {r['batch']} | {'yes' if r['constrained'] else 'no'} | "
                   f"{r['decode_steps']} | {r['step_ms']:.2f} | "
                   f"{fmt(r['mask_ms_per_row_token'], 3)} | {stops} / {len(r['finish'])} | "
                   f"{'n/a' if r['parsed'] is None else r['parsed']} |")
    return out + ["", f"Host: {host_line(d['manifest'])}", ""]


def readme_lines() -> list[str]:
    """README block: the comparison at one client and at the highest concurrency, plus the
    response_format cost, pointing at RESULTS.md for the rest."""
    d = load("baseline-qwen2.5-3b")
    out: list[str] = []
    if d:
        concs = sorted({p["concurrency"] for e in d["engines"] for p in e["points"]})
        lo, hi = concs[0], concs[-1]
        out += ["Against other local servers on the same Mac, Qwen2.5-3B-Instruct 4-bit (MLX "
                "4-bit for the MLX engines, Q4_K_M GGUF for llama.cpp and Ollama), same "
                "client and prompts:", "",
                f"| engine | tok/s, {lo} client | TPOT p50, {lo} client | tok/s, {hi} clients "
                f"| TPOT p50, {hi} clients | TTFT p50, {hi} clients | peak footprint |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for e in d["engines"]:
            by = {p["concurrency"]: p for p in e["points"]}
            a, b = by.get(lo), by.get(hi)
            if not a or not b:
                continue
            peak = max(p["peak_footprint_bytes"] for p in e["points"])
            out.append(f"| {ENGINE_NAMES.get(e['name'], e['name'])} | "
                       f"{a['out_tok_per_s']:.0f} | {fmt(a['tpot_p50_ms'])} ms | "
                       f"{b['out_tok_per_s']:.0f} | {fmt(b['tpot_p50_ms'])} ms | "
                       f"{fmt(b['ttft_p50_ms'], 0)} ms | {gib(peak)} GiB |")
        out += ["", "The quantizations differ and the other servers cache repeated prompt "
                "prefixes by default; the full table and caveats are in RESULTS.md."]
    j = load("json-overhead")
    if j:
        masks = [r["mask_ms_per_row_token"] for r in j["runs"] if r["mask_ms_per_row_token"]]
        if masks:
            out += ["", f"`response_format` costs {min(masks):.2f}-{max(masks):.2f} ms of CPU "
                    f"per constrained row per token on `{j['preset']}` "
                    "(RESULTS.md has step times with and without it)."]
    return out


def seedcheck_section(host_line) -> list[str]:
    d = load("seedcheck-qwen3.5-9b")
    if not d:
        return []
    out = ["## Seeded and greedy determinism, Qwen3.5-9B 4-bit", "",
           "`bench/seedcheck.py` runs the same prompts through the scheduler one at a time "
           "twice, then together at each width, once with every row admitted at the start and "
           "once with half the rows joining after four steps. A row counts as identical when "
           f"its text matches the first solo run byte for byte ({d['max_tokens']} tokens, "
           f"top_p {d['top_p']}, seed 1000 + row).", "",
           "| temperature | run | rows | identical to solo |", "|---:|---|---:|---:|"]
    names = {"alone-repeat": "solo again", "batched": "batched",
             "batched-join": "batched, half join late"}
    for c in d["cases"]:
        out.append(f"| {c['temperature']:g} | {names.get(c['mode'], c['mode'])} | "
                   f"{c['of'] if c['mode'] == 'alone-repeat' else c['width']} | "
                   f"{c['identical']} of {c['of']} |")
    out += ["", "Solo runs repeat exactly, seeded or greedy. Batched rows do not, from two "
            "rows up. The two-row run where the second row joined late, so each row was "
            "prefilled alone, stayed identical, which points at the padded multi-row prefill; "
            "the 4-bit matmul also changes kernels with the row count (above). Once one token "
            "differs, the rest of the text follows it. Where byte-identical output matters, "
            "send requests one at a time or run the server with a fixed batch of 1.", "",
            f"Host: {host_line(d['manifest'])}", ""]
    return out


def sections(host_line) -> list[str]:
    return (baseline_section(host_line) + sampler_section(host_line) + json_section(host_line)
            + seedcheck_section(host_line))
