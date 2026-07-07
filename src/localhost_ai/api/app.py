from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..service import Service
from . import admin_routes, openai_routes, score_routes, ws_routes


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
    hosts = [h.strip() for h in svc.settings.allowed_hosts.split(",") if h.strip()]
    if hosts and "*" not in hosts:
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
