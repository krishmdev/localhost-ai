"""Run the controller simulator (tests/sim_controller.py) for S1-S6 over 20 fixed seeds, write
docs/results/sim.json with the acceptance numbers, and plot seed 0 of S1-S5 to docs/figures/.

    uv run python scripts/sim_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

import plotstyle  # noqa: E402
from sim_controller import SCENARIOS, Scenario, band, run  # noqa: E402

WARMUP = 30
SEEDS = range(20)


def seeded(key: str, seed: int) -> Scenario:
    return Scenario(**{**SCENARIOS[key].__dict__, "seed": seed})


def summarize(key: str) -> dict:
    sc0 = SCENARIOS[key]
    b = sc0.b_star()
    lo, hi = band(b)
    per_seed = []
    for seed in SEEDS:
        sc = seeded(key, seed)
        tr = run(sc)
        ls = tr.limit[WARMUP:]
        ps = tr.p95_true[WARMUP:]
        row = {
            "seed": seed,
            "in_band": round(sum(lo <= x <= hi for x in ls) / len(ls), 4),
            "min_L": min(ls),
            "max_L": max(ls),
            "min_L_over_b_star": round(min(ls) / b, 3),
            "slo_violation_frac": round(sum(p > sc.slo for p in ps) / len(ps), 4),
            "max_used_frac": round(max(tr.used), 4),
            "L_le_ceiling_always": all(L <= max(1, c) for L, c in zip(tr.limit, tr.ceiling,
                                                                         strict=True)),
        }
        if sc.k_step:
            step = sc.k_step[0]
            nlo, nhi = band(sc.b_star(sc.k_step[1]))
            back = next((i for i in range(step, sc.intervals) if nlo <= tr.limit[i] <= nhi), None)
            row["intervals_to_new_band"] = None if back is None else back - step
        if sc.mem_shock_at is not None:
            s = sc.mem_shock_at
            ok = next((i for i in range(s, sc.intervals) if tr.headroom[i] > 0.10), None)
            row["intervals_to_headroom_above_low_wm"] = None if ok is None else ok - s
        if sc.oom_at is not None:
            t = sc.oom_at
            row["L_before_oom"], row["L_after_oom"] = tr.limit[t - 1], tr.limit[t]
        per_seed.append(row)
    return {"name": sc0.name, "b_star": b, "band": [lo, hi], "per_seed": per_seed}


def plot(key: str, out: Path) -> None:
    plt = plotstyle.plt
    sc = SCENARIOS[key]
    tr = run(sc)
    xs = list(range(sc.intervals))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    lo, hi = band(sc.b_star())
    if key == "S4":
        step = sc.k_step[0]
        nlo, nhi = band(sc.b_star(sc.k_step[1]))
        ax1.fill_between([0, step], lo, hi, color=plotstyle.BAND, lw=0, step="post")
        ax1.fill_between([step, sc.intervals], nlo, nhi, color=plotstyle.BAND, lw=0)
        ax1.plot(xs, tr.b_star, color=plotstyle.INK2, lw=1, drawstyle="steps-post")
        ax1.text(sc.intervals - 1, tr.b_star[-1] + 0.8, "b*", color=plotstyle.INK2,
                 ha="right", va="bottom", fontsize=9)
    elif key != "S5":
        ax1.axhspan(lo, hi, color=plotstyle.BAND, lw=0)
        ax1.axhline(sc.b_star(), color=plotstyle.INK2, lw=1)
        ax1.text(sc.intervals - 1, sc.b_star() + 0.6, "b*", color=plotstyle.INK2, ha="right",
                 va="bottom", fontsize=9)
    ax1.plot(xs, tr.limit, color=plotstyle.BLUE, lw=2, drawstyle="steps-post",
             label="batch limit L")
    ax1.plot(xs, tr.running, color=plotstyle.ORANGE, lw=1, drawstyle="steps-post",
             label="running rows")
    top = max(max(tr.limit), hi) * 1.15
    ax1.axvline(WARMUP, color=plotstyle.MUTED, lw=0.8)
    ax1.text(WARMUP + 1, top * 0.97, "warm-up ends", color=plotstyle.MUTED, fontsize=8, va="top")
    ax1.set_ylabel("rows")
    ax1.set_ylim(0, top)
    handles, labels = ax1.get_legend_handles_labels()
    if key != "S5":
        handles.append(plotstyle.matplotlib.patches.Patch(color=plotstyle.BAND))
        labels.append("S1/S2 band" if key == "S3" else "acceptance band")
    ax1.legend(handles, labels, loc="lower right", frameon=False, fontsize=9, ncols=3)
    if key == "S5":
        ax2.plot(xs, [h * 100 for h in tr.headroom], color=plotstyle.BLUE, lw=2)
        for y, lbl in ((10, "low watermark"), (20, "high watermark")):
            ax2.axhline(y, color=plotstyle.INK2, lw=1)
            ax2.text(1, y + 1, lbl, color=plotstyle.INK2, fontsize=8, va="bottom")
        ax2.set_ylabel("headroom (%)")
        ax2.set_ylim(0, 60)
    else:
        ax2.plot(xs, [p * 1e3 for p in tr.p95_true], color=plotstyle.BLUE, lw=1.5)
        ax2.axhline(sc.slo * 1e3, color=plotstyle.INK2, lw=1)
        ax2.text(1, sc.slo * 1e3 + 1, "SLO", color=plotstyle.INK2, fontsize=8, va="bottom")
        ax2.set_ylabel("p95 step (ms)")
        ax2.set_ylim(0, max(sc.slo * 1e3 * 1.6, max(tr.p95_true) * 1e3 * 1.1))
    ax2.set_xlabel("control interval (1 s each)")
    ax1.set_title(f"{sc.name}  (seed {sc.seed})", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    results = {key: summarize(key) for key in SCENARIOS}
    agg = {}
    for key, r in results.items():
        rows = r["per_seed"]
        b = r["b_star"]
        agg[key] = {
            "min_in_band": min(x["in_band"] for x in rows),
            "min_L": min(x["min_L"] for x in rows),
            "min_L_over_b_star": round(min(x["min_L"] for x in rows) / b, 3),
            "seeds_L_ge_half_b_star": sum(x["min_L"] >= 0.5 * b for x in rows),
            "seeds_L_ge_0_64_b_star": sum(x["min_L"] >= 0.64 * b for x in rows),
            "max_slo_violation_frac": max(x["slo_violation_frac"] for x in rows),
            "max_used_frac": max(x["max_used_frac"] for x in rows),
            "L_le_ceiling_all_seeds": all(x["L_le_ceiling_always"] for x in rows),
        }
        for k in ("intervals_to_new_band", "intervals_to_headroom_above_low_wm"):
            if k in rows[0]:
                agg[key][f"max_{k}"] = max(x[k] for x in rows)
        if "L_after_oom" in rows[0]:
            agg[key]["oom_halved_all_seeds"] = all(
                x["L_after_oom"] <= x["L_before_oom"] // 2 for x in rows)
    sys.path.insert(0, str(ROOT / "bench"))
    import loadgen

    out = {"seeds": len(SEEDS), "warmup_intervals": WARMUP, "summary": agg,
           "scenarios": results, "manifest": loadgen.manifest(None, {"kind": "simulator"})}
    (ROOT / "docs" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "results" / "sim.json").write_text(json.dumps(out, indent=1) + "\n")
    for key in ("S1", "S2", "S3", "S4", "S5"):
        plot(key, ROOT / "docs" / "figures" / f"sim_{key}.png")
    write_section(ROOT / "docs" / "controller.md", "sim", sim_markdown(results, agg))
    for key, a in agg.items():
        print(key, json.dumps(a))


def write_section(path: Path, name: str, body: str) -> None:
    text = path.read_text()
    begin, end = f"<!-- {name}:begin -->", f"<!-- {name}:end -->"
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{begin}\n{body}\n{end}{tail}")


def sim_markdown(results: dict, agg: dict) -> str:
    def pct(x: float) -> str:
        return f"{x * 100:.0f}%"

    n = len(SEEDS)
    s1, s2, s3, s4, s5 = (agg[k] for k in ("S1", "S2", "S3", "S4", "S5"))
    b = results["S1"]["b_star"]
    lo, hi = results["S1"]["band"]
    s6_ok = agg["S6"]["oom_halved_all_seeds"]
    s2_ok = (s2["min_in_band"] >= .95 and s2["seeds_L_ge_0_64_b_star"] == n
             and s2["max_slo_violation_frac"] <= .1)
    s3_verdict = "yes" if s3["seeds_L_ge_half_b_star"] == n else (
        ("violations yes" if s3["max_slo_violation_frac"] <= .2 else "violations no")
        + ", floor no (reported, not tuned)")
    rows = [
        f"Generated from `docs/results/sim.json` ({n} seeds, {WARMUP}-interval warm-up, "
        f"b* = {b}, band [{lo}, {hi}]).",
        "",
        "| scenario | plan bound | measured (worst seed) | met |",
        "|---|---|---|---|",
        f"| S1 no noise | L in band 100% of intervals | {pct(s1['min_in_band'])} in band, "
        f"L min {s1['min_L']} | {'yes' if s1['min_in_band'] == 1 else 'no'} |",
        f"| S2 5% noise | band >= 95%, L >= 0.64 b*, violations <= 10% | "
        f"{pct(s2['min_in_band'])} in band, L min {s2['min_L']} "
        f"({s2['min_L_over_b_star']} b*), violations {pct(s2['max_slo_violation_frac'])} | "
        f"{'yes' if s2_ok else 'no'} |",
        f"| S3 15% noise | violations <= 20%, L >= 0.5 b* | violations "
        f"{pct(s3['max_slo_violation_frac'])}; L >= 0.5 b* on {s3['seeds_L_ge_half_b_star']}/{n}"
        f" seeds, worst {s3['min_L']} ({s3['min_L_over_b_star']} b*) | "
        f"{s3_verdict} |",
        f"| S4 b* halves | back in new band within 6 intervals | "
        f"{s4['max_intervals_to_new_band']} intervals | "
        f"{'yes' if s4['max_intervals_to_new_band'] <= 6 else 'no'} |",
        f"| S5 headroom halves | above low watermark within 3 intervals | "
        f"{s5['max_intervals_to_headroom_above_low_wm']} intervals | "
        f"{'yes' if s5['max_intervals_to_headroom_above_low_wm'] <= 3 else 'no'} |",
        f"| S6 OOM | next interval L <= floor(L/2) | "
        f"{'halved on every seed' if s6_ok else 'not halved on some seed'} | "
        f"{'yes' if s6_ok else 'no'} |",
        f"| all | L <= KV ceiling, memory <= limit | peak memory "
        f"{max(a['max_used_frac'] for a in agg.values()):.2f} of limit | "
        f"{'yes' if all(a['L_le_ceiling_all_seeds'] for a in agg.values()) else 'no'} |",
        "",
    ]
    rows += [f"![{k}](figures/sim_{k}.png)" for k in ("S1", "S2", "S3", "S4", "S5")]
    return "\n".join(rows)


if __name__ == "__main__":
    main()
