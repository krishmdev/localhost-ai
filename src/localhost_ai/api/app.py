from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..service import Service
from . import admin_routes, openai_routes, score_routes, ws_routes


class BodyLimit:
    """Refuse HTTP bodies over `limit` bytes with 413, whether or not they declare a
    Content-Length: the body is read (up to the limit) before the app sees it, then replayed.
    WebSocket frames are bounded by the ASGI server instead."""

    def __init__(self, app: ASGIApp, limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.limit:
            return await self._too_large(scope, receive, send)
        held: list[Message] = []
        size = 0
        while True:
            msg = await receive()
            held.append(msg)
            if msg["type"] != "http.request":  # the client went away mid-body
                break
            size += len(msg.get("body", b""))
            if size > self.limit:
                return await self._too_large(scope, receive, send)
            if not msg.get("more_body", False):
                break

        async def replay() -> Message:
            return held.pop(0) if held else await receive()

        await self.app(scope, replay, send)

    async def _too_large(self, scope: Scope, receive: Receive, send: Send) -> None:
        r = JSONResponse({"error": {"message": f"request body is over {self.limit} bytes",
                                    "type": "invalid_request_error", "code": "request_too_large"}},
                         status_code=413)
        await r(scope, receive, send)


def create_app(svc: Service) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc.start()
        try:
            yield
        finally:
            svc.stop()

    app = FastAPI(title="localhost-ai", version="0.1.0", lifespan=lifespan)
    app.state.svc = svc
    app.add_middleware(BodyLimit, limit=svc.settings.max_request_bytes)
    hosts = [h.strip() for h in svc.settings.allowed_hosts.split(",") if h.strip()]
    if hosts and "*" not in hosts:  # added last, so it runs first
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
    app.include_router(openai_routes.router)
    app.include_router(score_routes.router)
    app.include_router(ws_routes.router)
    app.include_router(admin_routes.router)
    app.mount("/metrics", make_asgi_app(registry=svc.metrics.registry))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse({"error": {"message": str(exc.detail), "type": "invalid_request_error",
                                       "code": str(exc.status_code)}},
                            status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {"msg": "invalid request"}
        loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return JSONResponse({"error": {"message": f"{loc}: {first['msg']}" if loc else first["msg"],
                                       "type": "invalid_request_error", "code": "invalid_request"}},
                            status_code=400)

    return app
