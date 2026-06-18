"""A `top`-style view of the telemetry WebSocket: batch limit, latency vs SLO, memory.

    uv run python scripts/ws_top.py --url ws://127.0.0.1:8000
    uv run python scripts/ws_top.py --set-slo 60          # needs --token if LHAI_ADMIN_TOKEN is set
    uv run python scripts/ws_top.py --set-mode fixed:8
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
from collections import deque

import websockets
from rich.console import Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

SPARK = "▁▂▃▄▅▆▇█"
BAR_W = 28


def bar(frac: float | None, style: str = "blue") -> Text:
    t = Text("▕", style="grey50")
    if frac is None:
        t.append(" " * BAR_W)
    else:
        n = round(min(max(frac, 0.0), 1.0) * BAR_W)
        t.append("█" * n, style=style)
        t.append("░" * (BAR_W - n), style="grey30")
    t.append("▏", style="grey50")
    return t


def spark(values: deque, lo: float | None = None, hi: float | None = None) -> str:
    vs = [v for v in values if v is not None]
    if not vs:
        return ""
    lo = min(vs) if lo is None else lo
    hi = max(vs) if hi is None else hi
    span = (hi - lo) or 1.0
    return "".join(" " if v is None else SPARK[min(7, int((v - lo) / span * 7.999))]
                   for v in values)


def gb(n: float | None) -> str:
    return "?" if n is None else f"{n / 2**30:.2f} GB"


def fmt_ms(v: float | None) -> str:
    return "  n/a" if v is None else f"{v:6.1f} ms"


class View:
    def __init__(self, url: str) -> None:
        self.url = url
        self.limits: deque = deque(maxlen=60)
        self.p95s: deque = deque(maxlen=60)
        self.msg: dict = {}
        self.note = "connecting..."

    def update(self, m: dict) -> None:
        self.msg = m
        self.limits.append(m.get("batch_limit"))
        self.p95s.append(m.get("decode_step_p95_ms"))
        self.note = ""

    def render(self) -> Group:
        m = self.msg
        head = Text()
        head.append("localhost-ai  ", style="bold")
        if m:
            head.append(f"{m.get('model', '?')} on {m.get('device', '?')}/{m.get('dtype', '?')}   ")
            head.append(f"mode {m.get('mode')}   SLO {m.get('slo_tpot_ms', 0):.0f} ms   ")
        head.append(self.url, style="grey50")
        if not m:
            return Group(head, Text(self.note, style="grey50"))

        g = Table.grid(padding=(0, 2))
        g.add_column(style="grey70", no_wrap=True)
        g.add_column(justify="right", no_wrap=True)
        g.add_column(no_wrap=True)
        g.add_column(style="grey70", no_wrap=True)

        limit, running = m.get("batch_limit") or 0, m.get("running") or 0
        g.add_row("Batch limit L", f"{limit}", bar(running / limit if limit else None),
                  f"{running} running, {m.get('queued', 0)} queued")

        p95, slo = m.get("decode_step_p95_ms"), m.get("slo_tpot_ms") or 0
        over = p95 is not None and slo and p95 > slo
        g.add_row("p95 decode step", fmt_ms(p95),
                  bar(None if p95 is None or not slo else p95 / slo,
                      "red" if over else "blue"),
                  ("over SLO" if over else "of SLO") + f" {slo:.0f} ms")

        mem = m.get("memory") or {}
        used, lim, hr = mem.get("used_bytes"), mem.get("limit_bytes"), mem.get("headroom_frac")
        g.add_row("Memory", gb(used),
                  bar(None if not lim else 1 - (hr if hr is not None else 0)),
                  f"of {gb(lim)}, headroom {hr:.0%}" if hr is not None else "")
        g.add_row("Throughput", f"{m.get('tokens_per_s', 0):.0f} tok/s",
                  bar(m.get("busy_ratio")),
                  f"busy {m.get('busy_ratio', 0):.0%}, KV {m.get('kv_tokens', 0):,} tokens")
        if m.get("gpu_util_pct") is not None:
            g.add_row("GPU util (NVML)", f"{m['gpu_util_pct']:.0f}%",
                      bar(m["gpu_util_pct"] / 100), "")

        lo = [v for v in self.limits if v is not None]
        g.add_row("L, last 60 s", f"{min(lo)}-{max(lo)}" if lo else "",
                  Text(spark(self.limits, 0), style="blue"), "")
        g.add_row("p95, last 60 s", "", Text(spark(self.p95s, 0, max(slo * 1.5, 1)),
                                             style="blue"), "")
        d = m.get("last_decision") or {}
        tail = Text()
        if d:
            tail.append("Last decision  ", style="grey70")
            tail.append(f"{d['action']}", style="bold")
            tail.append(f"  {d['reason']}  ({d['limit_before']} -> {d['limit']})")
        return Group(head, Text(""), g, Text(""), tail,
                     Text(self.note, style="grey50") if self.note else Text(""))


async def main(args: argparse.Namespace) -> None:
    url = args.url.rstrip("/") + f"/v1/ws/telemetry?interval={args.interval}"
    if args.token:
        url += f"&token={args.token}"
    view = View(args.url)
    with Live(view.render(), refresh_per_second=4, screen=not args.once) as live:
        async with websockets.connect(url) as ws:
            if args.set_slo:
                await ws.send(json.dumps({"type": "set_slo", "tpot_ms": args.set_slo}))
            if args.set_mode:
                mode, _, batch = args.set_mode.partition(":")
                await ws.send(json.dumps({"type": "set_mode", "mode": mode,
                                          "batch": int(batch) if batch else None}))
            async for raw in ws:
                m = json.loads(raw)
                if m.get("type") == "telemetry":
                    view.update(m)
                    if args.once:
                        live.update(view.render())
                        return
                elif m.get("type") in ("ack", "error"):
                    view.note = f"{m['type']}: {m.get('message') or m.get('for')}"
                live.update(view.render())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000")
    ap.add_argument("--token", default="")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--set-slo", type=float)
    ap.add_argument("--set-mode", help="aimd, aimd:16 (initial L) or fixed:8")
    ap.add_argument("--once", action="store_true", help="print one frame and exit")
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main(ap.parse_args()))
