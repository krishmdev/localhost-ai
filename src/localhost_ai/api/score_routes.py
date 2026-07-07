"""POST /v1/score (engine/score.py): candidate log-probabilities at sites of a fixed
continuation. The forward passes run on the compute thread between scheduler iterations."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..engine.engine import JobAborted
from ..engine.runner import is_oom
from ..engine.score import ScoreError, Site, score
from ..service import Service
from .openai_routes import error
from .schemas import ScoreRequest, ScoreResponse, ScoreSiteResult

router = APIRouter()


@router.post("/v1/score", response_model=None)
async def score_route(body: ScoreRequest, request: Request):
    svc: Service = request.app.state.svc
    if svc.swapping or svc.parts is None:
        return error(503, "model is being replaced; retry shortly", "server_error")
    parts = svc.parts
    if body.model and body.model not in (svc.model_name, parts.info.get("repo")):
        return error(404, f"model {body.model!r} is not loaded (serving {svc.model_name})",
                     "invalid_request_error", "model_not_found")
    if parts.chat_text is None or not hasattr(parts.runner, "logprob_rows"):
        return error(400, f"{svc.model_name} can't be scored", "invalid_request_error")
    messages = [m.model_dump() for m in body.messages]
    try:
        head = parts.chat_text(messages, body.chat_template_kwargs)
    except Exception as exc:  # a template that rejects the messages or kwargs
        return error(400, f"chat template failed: {exc}", "invalid_request_error")
    sites = [Site(s.char_offset, tuple(s.candidates)) for s in body.sites]
    generation = svc.generation

    def job():
        if svc.generation != generation or svc.parts is not parts:
            raise JobAborted("model changed while the request was queued")
        return score(parts.tokenizer, parts.runner.logprob_rows, head, body.continuation, sites,
                     svc.settings.max_context)

    try:
        results, n = await svc.engine.run(job)
    except ScoreError as exc:
        return error(400, str(exc), "invalid_request_error", exc.code)
    except JobAborted as exc:
        return error(503, str(exc), "server_error")
    except Exception as exc:
        if is_oom(exc):
            return error(503, f"out of memory while scoring: {exc}", "server_error",
                         "out_of_memory")
        raise
    commit, dirty = svc.commit
    return ScoreResponse(
        model=svc.model_name, revision=parts.info.get("revision"), commit=commit, dirty=dirty,
        tokenizer_sha=parts.tokenizer_sha, prompt_tokens=n,
        sites=[ScoreSiteResult(char_offset=r.char_offset, token_index=r.token_index,
                               candidates=r.logprobs, renorm=r.renorm(), forced=r.forced)
               for r in results])
