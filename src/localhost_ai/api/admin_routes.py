from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ..service import Service
from .schemas import ControllerUpdate, ModelLoad

router = APIRouter()


def require_admin(request: Request) -> Service:
    svc: Service = request.app.state.svc
    expected = svc.settings.admin_token
    if expected:
        got = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(got, expected):
            raise HTTPException(401, "admin routes need 'Authorization: Bearer <LHAI_ADMIN_TOKEN>'")
    return svc


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request):
    svc: Service = request.app.state.svc
    ok = svc.engine.alive and not svc.swapping
    body = {"ready": ok, "model": svc.model_name, "engine_error": svc.engine.error}
    return JSONResponse(body, status_code=200 if ok else 503)


def controller_view(svc: Service) -> dict:
    ctl = svc.engine.controller
    return {**ctl.state(), "slo_tpot_ms": svc.engine.slo_tpot_ms}


@router.get("/v1/admin/controller")
async def get_controller(svc: Service = Depends(require_admin)) -> dict:
    return controller_view(svc)


@router.put("/v1/admin/controller")
async def put_controller(update: ControllerUpdate, svc: Service = Depends(require_admin)) -> dict:
    if update.slo_tpot_ms is not None:
        svc.engine.set_slo(update.slo_tpot_ms)
        svc.metrics.slo.set(update.slo_tpot_ms / 1e3)
    if update.mode is not None:
        svc.engine.set_mode(update.mode, update.batch)
    elif update.batch is not None:
        raise HTTPException(400, "batch needs a mode ('fixed' or 'aimd' initial limit)")
    return controller_view(svc)


@router.get("/v1/admin/decisions")
async def decisions(limit: int = 100, svc: Service = Depends(require_admin)) -> dict:
    log = list(getattr(svc.engine.controller, "log", []))[-limit:]
    return {"decisions": [d.to_dict() for d in log]}


@router.get("/v1/admin/egress")
async def egress(expect: str = "blocked", svc: Service = Depends(require_admin)):
    """Run the egress canary inside the server process. 200 when the result matches `expect`
    (blocked: every connect failed; open: every connect worked), 409 otherwise."""
    import asyncio

    from ..egress import probe

    results = await asyncio.to_thread(probe, 2.0)
    opened = [k for k, v in results.items() if v == "open"]
    ok = not opened if expect == "blocked" else len(opened) == len(results)
    return JSONResponse({"expect": expect, "ok": ok, "results": results},
                        status_code=200 if ok else 409)


@router.post("/v1/admin/models/load")
async def load_model(body: ModelLoad, svc: Service = Depends(require_admin)) -> dict:
    if body.model not in svc.model_names:
        raise HTTPException(404, f"unknown model {body.model!r}; known: {svc.model_names}")
    await svc.swap_model(body.model)
    return {"model": svc.model_name, "info": svc.parts.info}
