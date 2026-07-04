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
           constraint: object | None = None, adapter: str | None = None) -> Handle:
    """Shared by REST and WebSocket. Raises QueueFull / HTTPException."""
    if svc.swapping:
        raise HTTPException(503, "model is being replaced; retry shortly")
    if svc.parts is None:
        raise HTTPException(503, "no model is loaded")
    ids = svc.parts.encode_chat(messages)
    params = params_for(len(ids))
    return svc.engine.submit(ids, params, constraint, adapter)


def pick_adapter(svc: Service, body: ChatCompletionRequest) -> str | None:
    """`model` may name the base model (or its Hub repo) or a loaded LoRA adapter; `adapter`
    names an adapter explicitly. Returns the adapter to run, None for the base model. Raises
    HTTPException 404 for names that aren't loaded, 400 if the two fields disagree."""
    if svc.parts is None:
        return None
    names = svc.parts.adapters
    base = (svc.model_name, svc.parts.info.get("repo"))
    by_model = body.model if body.model in names else None
    if body.model and body.model not in base and by_model is None:
        raise HTTPException(404, f"model {body.model!r} is not loaded (serving "
                            f"{', '.join([svc.model_name, *names])})")
    if body.adapter is None:
        return by_model
    if body.adapter not in names:
        raise HTTPException(404, f"adapter {body.adapter!r} is not loaded"
                            + (f" (loaded: {', '.join(names)})" if names else ""))
    if by_model is not None and by_model != body.adapter:
        raise HTTPException(400, f"model names adapter {by_model!r} but adapter is "
                            f"{body.adapter!r}")
    return body.adapter


async def build_constraint(grammars: object | None, fmt: ResponseFormat | None):
    """The token-level constraint for a response_format, None for plain text. Built on a worker
    thread: the first one on a model also builds llguidance's view of the vocabulary, which
    takes a second or two and would otherwise stall every stream on the event loop."""
    if fmt is None or fmt.type == "text":
        return None
    return await asyncio.to_thread(constraint_for, grammars, fmt)


def constraint_for(grammars: object | None, fmt: ResponseFormat | None):
    """The token-level constraint for a response_format, None for plain text."""
    spec = fmt.as_dict() if fmt is not None else None
    if spec is None:
        return None
    if grammars is None:
        raise HTTPException(400, "response_format is not supported for this model")
    try:
        return grammars.constraint(spec)
    except GrammarError as exc:
        raise HTTPException(400, f"invalid response_format: {exc}") from exc


async def prepare_request(svc: Service, body: ChatCompletionRequest,
                          messages: list[dict[str, str]]) -> tuple[Handle, str]:
    """Prepare a request and submit it only to the model whose grammar was used.

    Grammar construction runs in a thread and may finish after a hot-swap. Hold only its
    tokenizer wrapper across that await, not the old model or engine, so a swap can free the
    old weights before loading the new ones.
    """
    if svc.swapping or svc.parts is None:
        raise HTTPException(503, "model is being replaced; retry shortly")
    generation = svc.generation
    adapter = pick_adapter(svc, body)
    model_name = svc.model_name
    grammars = svc.parts.grammars
    constraint = await build_constraint(grammars, body.response_format)
    if svc.swapping or svc.parts is None or svc.generation != generation:
        raise HTTPException(503, "model changed while preparing the request; retry shortly")
    handle = submit(svc, messages, lambda n: sampling_params(body, n, svc),
                    constraint, adapter)
    return handle, adapter or model_name


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
    base = ModelCard(id=svc.model_name, root=svc.parts.info.get("repo"))
    return ModelList(data=[base, *(ModelCard(id=a, parent=svc.model_name)
                                   for a in svc.parts.adapters)])


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(body: ChatCompletionRequest, request: Request):
    svc: Service = request.app.state.svc
    messages = [m.model_dump() for m in body.messages]
    try:
        handle, served = await prepare_request(svc, body, messages)
    except QueueFull as exc:
        svc.metrics.rejected()
        return error(429, str(exc), "rate_limit_error", "queue_full",
                     {"Retry-After": str(max(1, round(exc.retry_after_s)))})
    except HTTPException as exc:
        return error(exc.status_code, exc.detail, "invalid_request_error",
                     "model_not_found" if exc.status_code == 404 else None)

    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    if body.stream:
        include_usage = bool(body.stream_options and body.stream_options.include_usage)
        return StreamingResponse(
            stream(handle, cid, created, served, include_usage),
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
                    id=cid, created=created, model=served,
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
