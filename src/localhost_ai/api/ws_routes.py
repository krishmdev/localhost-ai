"""WebSocket APIs.

`/v1/ws/generate` multiplexes many generations over one socket:
  client -> {"type": "generate", "id": "a", "messages": [...], "max_tokens": 64, ...}
            {"type": "cancel", "id": "a"}
  server -> {"type": "accepted", "id", "queue_position"}
            {"type": "token", "id", "text", "index"}
            {"type": "done", "id", "finish_reason", "usage", "timings"}
            {"type": "error", "id", "message", "code"}

`/v1/ws/telemetry` pushes one engine snapshot per control interval (device memory, batch
limit, running/queued, p95 decode step, tokens/s, the controller's last decision) and accepts
{"type": "set_slo", "tpot_ms"} and {"type": "set_mode", "mode", "batch"} when the admin token
matches (`?token=...`; no token configured means local-dev mode, controls open). Both routes
reject browser origins outside LHAI_ALLOWED_HOSTS."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import math
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from ..engine.engine import Handle
from ..engine.request import DoneEvent, ErrorEvent, TokenEvent
from ..engine.scheduler import QueueFull
from ..service import Service
from .openai_routes import pick_adapter, sampling_params, submit, timings, usage
from .schemas import ChatCompletionRequest

router = APIRouter()


def origin_ok(svc: Service, ws: WebSocket) -> bool:
    """Browsers send an Origin header on WebSocket handshakes and CORS doesn't apply to them, so
    a web page could otherwise drive this server. Only non-browser clients (no Origin) and pages
    served from an allowed host are accepted, whether or not an admin token is set."""
    origin = ws.headers.get("origin")
    if not origin:
        return True
    allowed = {h.strip().strip("[]") for h in svc.settings.allowed_hosts.split(",") if h.strip()}
    return "*" in allowed or (urlsplit(origin).hostname or "") in allowed


def token_ok(svc: Service, supplied: str | None) -> bool:
    expected = svc.settings.admin_token
    if not expected:
        return True
    return supplied is not None and hmac.compare_digest(supplied, expected)


class _Conn:
    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self._lock = asyncio.Lock()

    async def send(self, msg: dict[str, Any]) -> None:
        async with self._lock:
            await self.ws.send_text(json.dumps(msg, separators=(",", ":")))


@router.websocket("/v1/ws/generate")
async def ws_generate(ws: WebSocket) -> None:
    svc: Service = ws.app.state.svc
    if not origin_ok(svc, ws):
        await ws.close(code=1008)
        return
    await ws.accept()
    conn = _Conn(ws)
    handles: dict[str, Handle] = {}
    tasks: set[asyncio.Task] = set()

    async def pump(rid: str, handle: Handle) -> None:
        try:
            async for ev in handle.events():
                if isinstance(ev, TokenEvent):
                    await conn.send({"type": "token", "id": rid, "text": ev.text,
                                     "index": ev.index})
                elif isinstance(ev, ErrorEvent):
                    await conn.send({"type": "error", "id": rid, "message": ev.message,
                                     "code": ev.code})
                elif isinstance(ev, DoneEvent):
                    await conn.send({"type": "done", "id": rid,
                                     "finish_reason": ev.finish_reason,
                                     "usage": usage(ev).model_dump(),
                                     "timings": timings(ev).model_dump()})
        except (WebSocketDisconnect, RuntimeError):
            handle.cancel()
        finally:
            handles.pop(rid, None)

    try:
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except json.JSONDecodeError:
                await conn.send({"type": "error", "id": None, "message": "invalid JSON"})
                continue
            if not isinstance(msg, dict):
                await conn.send({"type": "error", "id": None, "message": "expected an object"})
                continue
            kind, rid = msg.get("type"), str(msg.get("id", ""))
            if kind == "cancel":
                h = handles.get(rid)
                if h is not None:
                    h.cancel()
                continue
            if kind != "generate":
                await conn.send({"type": "error", "id": rid or None,
                                 "message": f"unknown message type {kind!r}"})
                continue
            if not rid or rid in handles:
                await conn.send({"type": "error", "id": rid or None,
                                 "message": "each generate needs a unique id"})
                continue
            try:
                body = ChatCompletionRequest(**{k: v for k, v in msg.items()
                                                if k not in ("type", "id")})
                handle = submit(svc, [m.model_dump() for m in body.messages],
                                lambda n, b=body: sampling_params(b, n, svc),
                                body.response_format, pick_adapter(svc, body))
            except ValidationError as exc:
                await conn.send({"type": "error", "id": rid, "code": "invalid_request",
                                 "message": exc.errors()[0]["msg"]})
                continue
            except QueueFull as exc:
                svc.metrics.rejected()
                await conn.send({"type": "error", "id": rid, "code": "queue_full",
                                 "message": str(exc), "retry_after_s": exc.retry_after_s})
                continue
            except HTTPException as exc:
                await conn.send({"type": "error", "id": rid, "code": str(exc.status_code),
                                 "message": exc.detail})
                continue
            handles[rid] = handle
            await conn.send({"type": "accepted", "id": rid,
                             "queue_position": handle.queue_position})
            task = asyncio.create_task(pump(rid, handle))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
    except WebSocketDisconnect:
        pass
    finally:
        for h in list(handles.values()):
            h.cancel()
        for t in list(tasks):
            t.cancel()


@router.websocket("/v1/ws/telemetry")
async def ws_telemetry(ws: WebSocket) -> None:
    svc: Service = ws.app.state.svc
    if not origin_ok(svc, ws):
        await ws.close(code=1008)
        return
    try:
        period = float(ws.query_params.get("interval", svc.settings.control_interval_s))
    except ValueError:
        await ws.close(code=1008, reason="interval must be a number")
        return
    period = min(max(period, 0.1), 10.0) if math.isfinite(period) else 1.0
    await ws.accept()
    conn = _Conn(ws)
    can_control = token_ok(svc, ws.query_params.get("token"))

    async def push() -> None:
        while True:
            await conn.send({"type": "telemetry", "t": asyncio.get_running_loop().time(),
                             **svc.engine.telemetry()})
            await asyncio.sleep(period)

    pusher = asyncio.create_task(push())
    try:
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except json.JSONDecodeError:
                await conn.send({"type": "error", "message": "invalid JSON"})
                continue
            if not isinstance(msg, dict):
                await conn.send({"type": "error", "message": "expected an object"})
                continue
            kind = msg.get("type")
            if kind not in ("set_slo", "set_mode"):
                await conn.send({"type": "error", "message": f"unknown message type {kind!r}"})
                continue
            if not can_control:
                await conn.send({"type": "error", "code": "forbidden",
                                 "message": "controls need ?token=<LHAI_ADMIN_TOKEN>"})
                continue
            try:
                if kind == "set_slo":
                    tpot = float(msg["tpot_ms"])
                    if not math.isfinite(tpot) or tpot <= 0:
                        raise ValueError("tpot_ms must be a finite number > 0")
                    svc.engine.set_slo(tpot)
                    svc.metrics.slo.set(tpot / 1e3)
                    await conn.send({"type": "ack", "for": kind, "slo_tpot_ms": tpot})
                else:
                    batch = msg.get("batch")
                    ctl = svc.engine.set_mode(str(msg["mode"]), int(batch) if batch else None)
                    await conn.send({"type": "ack", "for": kind, "mode": ctl.mode,
                                     "batch_limit": ctl.limit})
            except (KeyError, ValueError, TypeError) as exc:
                await conn.send({"type": "error", "code": "invalid_request", "message": str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        pusher.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await pusher
