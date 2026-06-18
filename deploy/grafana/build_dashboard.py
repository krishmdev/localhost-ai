"""Generates dashboards/localhost-ai.json. Colors follow one categorical order (blue, orange,
aqua) validated for CVD separation on the dark surface; reference lines use neutral ink."""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "lhai-prom"}
BLUE, ORANGE, AQUA, INK2, MUTED = "#3987e5", "#d95926", "#199e70", "#c3c2b7", "#898781"
panels = []
pid = [0]
def nid():
    pid[0] += 1; return pid[0]
def target(expr, legend, ref):
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref, "range": True}
def stat(title, expr, x, unit, desc, decimals=None, w=4):
    p = {"id": nid(), "type": "stat", "title": title, "description": desc, "datasource": DS,
         "gridPos": {"h": 4, "w": w, "x": x, "y": 0},
         "targets": [target(expr, title, "A")],
         "options": {"colorMode": "none", "graphMode": "area", "justifyMode": "center",
                     "textMode": "value", "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                     "wideLayout": True, "showPercentChange": False},
         "fieldConfig": {"defaults": {"unit": unit, "color": {"mode": "fixed", "fixedColor": BLUE},
                                      "noValue": "no data"}, "overrides": []}}
    if decimals is not None:
        p["fieldConfig"]["defaults"]["decimals"] = decimals
    return p
def series_override(name, color, width=2, style=None, step=False):
    props = [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}},
             {"id": "custom.lineWidth", "value": width}]
    if style:
        props.append({"id": "custom.lineStyle", "value": style})
    if step:
        props.append({"id": "custom.lineInterpolation", "value": "stepAfter"})
    return {"matcher": {"id": "byName", "options": name}, "properties": props}
def ts(title, targets, pos, unit, desc, overrides, mn=0, mx=None, decimals=None, extra=None):
    d = {"unit": unit, "min": mn, "custom": {"lineWidth": 2, "fillOpacity": 0, "showPoints": "never",
         "spanNulls": True, "axisSoftMin": 0, "gradientMode": "none", "drawStyle": "line",
         "axisBorderShow": False, "thresholdsStyle": {"mode": "off"}},
         "color": {"mode": "fixed", "fixedColor": BLUE}}
    if mx is not None: d["max"] = mx
    if decimals is not None: d["decimals"] = decimals
    if extra: d.update(extra)
    return {"id": nid(), "type": "timeseries", "title": title, "description": desc, "datasource": DS,
            "gridPos": pos, "targets": targets,
            "fieldConfig": {"defaults": d, "overrides": overrides},
            "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                        "tooltip": {"mode": "multi", "sort": "none"}}}
x = 0
for t, e, u, desc, dec in [
    ("Batch limit L", "lhai_batch_limit", "none", "Current controller limit on running rows", 0),
    ("p95 decode step (5 s)", "lhai_decode_step_p95_seconds", "s", "p95 wall time of decode steps over the last 5 s, the latency the controller holds under the SLO", None),
    ("TPOT SLO", "lhai_slo_tpot_seconds", "s", "The per-token latency target the AIMD controller holds", None),
    ("Output tokens/s", "sum(rate(lhai_generated_tokens_total[15s]))", "short", "Generated tokens per second, 15 s rate", 1),
    ("Memory headroom", "lhai_memory_headroom_ratio", "percentunit", "Headroom / limit from the device memory probe", 1),
    ("Queue depth", "lhai_queue_depth", "none", "Requests waiting for a batch slot", 0),
]:
    panels.append(stat(t, e, x, u, desc, dec)); x += 4
panels.append(ts("p95 decode-step time vs TPOT SLO",
    [target("lhai_decode_step_p95_seconds", "p95 decode step", "A"), target("lhai_slo_tpot_seconds", "SLO", "B"), target("histogram_quantile(0.95, sum by (le) (rate(lhai_step_seconds_bucket[30s])))", "p95 iteration (incl. prefill)", "C")],
    {"h": 8, "w": 24, "x": 0, "y": 4}, "s",
    "The controller watches decode-step time: above the SLO line it shrinks L by 20%; below 90% of it (with a saturated batch and memory headroom) it grows L by 10%. The iteration line adds prefill stalls, which is what a stream feels.",
    [series_override("p95 decode step", BLUE), series_override("SLO", INK2, 1), series_override("p95 iteration (incl. prefill)", ORANGE, 1)]))
panels.append(ts("Batch limit and running rows",
    [target("lhai_batch_limit", "batch limit L", "A"), target("lhai_running_requests", "running rows", "B")],
    {"h": 8, "w": 24, "x": 0, "y": 12}, "none",
    "L is the controller's output; running rows is how many requests were actually in the batch at the last tick.",
    [series_override("batch limit L", BLUE, 2, step=True), series_override("running rows", ORANGE, 1)], decimals=0))
panels.append(ts("Output tokens per second",
    [target("sum(rate(lhai_generated_tokens_total[15s]))", "tokens/s", "A")],
    {"h": 8, "w": 12, "x": 0, "y": 20}, "short", "Generated tokens per second (15 s rate).",
    [series_override("tokens/s", BLUE)]))
panels.append(ts("Memory headroom",
    [target("lhai_memory_headroom_ratio", "headroom", "A")],
    {"h": 8, "w": 12, "x": 12, "y": 20}, "percentunit",
    "Below the 10% line the controller cuts L and preempts rows; it only grows L above the 20% line.",
    [series_override("headroom", BLUE)], mn=0, mx=1,
    extra={"thresholds": {"mode": "absolute", "steps": [
        {"color": MUTED, "value": None}, {"color": MUTED, "value": 0.10}, {"color": MUTED, "value": 0.20}]},
        "custom": {"lineWidth": 2, "fillOpacity": 0, "showPoints": "never", "spanNulls": True,
                   "drawStyle": "line", "axisBorderShow": False, "gradientMode": "none",
                   "thresholdsStyle": {"mode": "line"}}}))
panels.append(ts("Request latency (p95 from histograms)",
    [target("histogram_quantile(0.95, sum by (le) (rate(lhai_ttft_seconds_bucket[1m])))", "TTFT p95", "A"),
     target("histogram_quantile(0.95, sum by (le) (rate(lhai_e2e_latency_seconds_bucket[1m])))", "end-to-end p95", "B")],
    {"h": 8, "w": 12, "x": 0, "y": 28}, "s", "Bucketed estimates over a 1 min window.",
    [series_override("TTFT p95", BLUE), series_override("end-to-end p95", ORANGE)]))
bar = ts("Controller actions per minute",
    [target('sum by (action) (increase(lhai_controller_decisions_total{action!="hold"}[1m]))', "{{action}}", "A")],
    {"h": 8, "w": 12, "x": 12, "y": 28}, "none", "Every non-hold decision the controller made, by rule.",
    [series_override("increase", BLUE), series_override("slo_decrease", ORANGE),
     series_override("mem_decrease", AQUA), series_override("oom_backoff", "#9085e9"),
     series_override("clamp", "#c98500")], decimals=0)
bar["fieldConfig"]["defaults"]["custom"].update({"drawStyle": "bars", "fillOpacity": 60, "lineWidth": 1, "stacking": {"mode": "normal", "group": "A"}})
panels.append(bar)
panels.append(ts("Engine busy ratio and KV cache",
    [target("lhai_engine_busy_ratio", "busy ratio", "A")],
    {"h": 7, "w": 12, "x": 0, "y": 36}, "percentunit",
    "Share of wall time the compute thread spent in prefill/decode. Device-agnostic utilization; on NVIDIA the GPU utilization panel is the hardware view.",
    [series_override("busy ratio", BLUE)], mn=0, mx=1))
panels.append(ts("Preemptions and OOM events",
    [target("increase(lhai_preemptions_total[1m])", "preemptions / min", "A"),
     target("increase(lhai_oom_events_total[1m])", "OOM events / min", "B")],
    {"h": 7, "w": 12, "x": 12, "y": 36}, "none", "Recompute preemptions (memory shedding or OOM) and caught out-of-memory errors.",
    [series_override("preemptions / min", BLUE), series_override("OOM events / min", ORANGE)], decimals=0))
dash = {"uid": "lhai-overview", "title": "localhost-ai: adaptive batching", "tags": ["localhost-ai"],
        "timezone": "browser", "editable": True, "graphTooltip": 1, "refresh": "5s",
        "schemaVersion": 41, "version": 1, "time": {"from": "now-15m", "to": "now"},
        "timepicker": {"refresh_intervals": ["2s", "5s", "10s", "30s", "1m"]},
        "templating": {"list": []}, "annotations": {"list": []}, "panels": panels, "links": []}
out = Path(__file__).parent / "dashboards" / "localhost-ai.json"
out.write_text(json.dumps(dash, indent=2) + "\n")
print(f"wrote {out} ({len(panels)} panels)")
