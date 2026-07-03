from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..engine.constrain import GrammarError
from ..engine.engine import Handle
from ..engine.request import DoneEvent, ErrorEvent, SamplingParams, TokenEvent
from ..engine.scheduler import QueueFull
from ..service import Service
from .schemas import (
    AssistantMessage,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChunkChoice,
    Delta,
    ModelCard,
    ModelList,
    ResponseFormat,
    Timings,
    Usage,
)

router = APIRouter()


def error(status: int, message: str, type_: str, code: str | None = None,
          headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": type_, "code": code}},
                        status_code=status, headers=headers)


def sampling_params(body: ChatCompletionRequest, prompt_len: int, svc: Service) -> SamplingParams:
    s = svc.settings
    room = s.max_context - prompt_len
    if room < 1:
        raise HTTPException(400, f"prompt is {prompt_len} tokens; context is {s.max_context}")
    wanted = body.max_completion_tokens or body.max_tokens or svc.parts.default_max_tokens
    stop = (body.stop,) if isinstance(body.stop, str) else tuple(body.stop or ())
    return SamplingParams(temperature=body.temperature, top_p=body.top_p, top_k=body.top_k,
                          max_tokens=min(wanted, room), stop=stop, seed=body.seed)


def submit(svc: Service, messages: list[dict[str, str]], params_for,
           response_format: ResponseFormat | None = None) -> Handle:
    """Shared by REST and WebSocket. Raises QueueFull / HTTPException."""
    if svc.swapping:
        raise HTTPException(503, "model is being replaced; retry shortly")
    if svc.parts is None:
        raise HTTPException(503, "no model is loaded")
    ids = svc.parts.encode_chat(messages)
    params = params_for(len(ids))
    return svc.engine.submit(ids, params, constraint_for(svc, response_format))


def constraint_for(svc: Service, fmt: ResponseFormat | None):
    """The token-level constraint for a response_format, None for plain text."""
    spec = fmt.as_dict() if fmt is not None else None
    if spec is None:
        return None
    if svc.parts.grammars is None:
        raise HTTPException(400, "response_format is not supported for this model")
    try:
        return svc.parts.grammars.constraint(spec)
    except GrammarError as exc:
        raise HTTPException(400, f"invalid response_format: {exc}") from exc


def usage(done: DoneEvent) -> Usage:
    return Usage(prompt_tokens=done.prompt_tokens, completion_tokens=done.completion_tokens,
                 total_tokens=done.prompt_tokens + done.completion_tokens)


def timings(done: DoneEvent) -> Timings:
    ms = lambda v: None if v is None else round(v * 1e3, 3)  # noqa: E731
    return Timings(queue_ms=ms(done.queue_s), ttft_ms=ms(done.ttft_s), tpot_ms=ms(done.tpot_s),
                   e2e_ms=ms(done.e2e_s))


@router.get("/v1/models")
async def list_models(request: Request) -> ModelList:
    svc: Service = request.app.state.svc
    if svc.parts is None:
        return ModelList(data=[])
    return ModelList(data=[ModelCard(id=svc.model_name, root=svc.parts.info.get("repo"))])


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(body: ChatCompletionRequest, request: Request):
    svc: Service = request.app.state.svc
    repo = svc.parts.info.get("repo") if svc.parts is not None else None
    if svc.parts is not None and body.model and body.model not in (svc.model_name, repo):
        return error(404, f"model {body.model!r} is not loaded (serving {svc.model_name})",
                     "invalid_request_error", "model_not_found")
    messages = [m.model_dump() for m in body.messages]
    try:
        handle = submit(svc, messages, lambda n: sampling_params(body, n, svc),
                        body.response_format)
    except QueueFull as exc:
        svc.metrics.rejected()
        return error(429, str(exc), "rate_limit_error", "queue_full",
                     {"Retry-After": str(max(1, round(exc.retry_after_s)))})
    except HTTPException as exc:
        return error(exc.status_code, exc.detail, "invalid_request_error")

    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    if body.stream:
        include_usage = bool(body.stream_options and body.stream_options.include_usage)
        return StreamingResponse(
            stream(handle, cid, created, svc.model_name, include_usage),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def watch_disconnect() -> None:
        while not await request.is_disconnected():
            await asyncio.sleep(0.5)
        handle.cancel()

    watcher = asyncio.create_task(watch_disconnect())
    parts: list[str] = []
    try:
        async for ev in handle.events():
            if isinstance(ev, TokenEvent):
                parts.append(ev.text)
            elif isinstance(ev, ErrorEvent):
                return error(500, ev.message, "server_error", ev.code)
            elif isinstance(ev, DoneEvent):
                return ChatCompletion(
                    id=cid, created=created, model=svc.model_name,
                    choices=[Choice(message=AssistantMessage(content="".join(parts)),
                                    finish_reason=ev.finish_reason)],
                    usage=usage(ev), timings=timings(ev),
                )
    finally:
        watcher.cancel()
        if handle.request.finish_reason is None:
            handle.cancel()
    return error(500, "generation ended without a result", "server_error")


def _sse(obj: ChatCompletionChunk | dict) -> str:
    data = obj if isinstance(obj, dict) else obj.model_dump(exclude_none=True)
    for c in data.get("choices", []):
        c.setdefault("finish_reason", None)  # OpenAI sends an explicit null mid-stream
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


async def stream(handle: Handle, cid: str, created: int, model: str,
                 include_usage: bool) -> AsyncIterator[str]:
    def chunk(delta: Delta, finish: str | None = None) -> ChatCompletionChunk:
        return ChatCompletionChunk(id=cid, created=created, model=model,
                                   choices=[ChunkChoice(delta=delta, finish_reason=finish)])

    yield _sse(chunk(Delta(role="assistant", content="")))
    # Leaving this generator early (client disconnect) runs Handle.events()' finally, which
    # cancels the request so the batch slot is freed on the next iteration.
    async for ev in handle.events():
        if isinstance(ev, TokenEvent):
            yield _sse(chunk(Delta(content=ev.text)))
        elif isinstance(ev, ErrorEvent):
            yield _sse({"error": {"message": ev.message, "type": "server_error",
                                  "code": ev.code}})
        elif isinstance(ev, DoneEvent):
            final = chunk(Delta(), ev.finish_reason).model_dump(exclude_none=True)
            final["timings"] = timings(ev).model_dump()
            yield _sse(final)
            if include_usage:
                u = ChatCompletionChunk(id=cid, created=created, model=model, choices=[],
                                        usage=usage(ev))
                yield _sse(u.model_dump(exclude_none=True))
    yield "data: [DONE]\n\n"
