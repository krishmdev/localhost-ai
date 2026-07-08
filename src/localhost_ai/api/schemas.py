"""OpenAI-shaped request/response models (the subset this server implements)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str


class StreamOptions(BaseModel):
    include_usage: bool = False


class JSONSchemaFormat(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = "response"
    description: str | None = None
    schema_: dict | None = Field(None, alias="schema")
    strict: bool | None = False


class ResponseFormat(BaseModel):
    """`text` (the default), `json_object` (any JSON object) or `json_schema`. The JSON types are
    enforced token by token, so a response that finishes with `stop` parses. They can't be
    combined with `stop` strings (400), and a row whose grammar matcher fails ends with an
    error instead of `stop`."""

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: JSONSchemaFormat | None = None

    def as_dict(self) -> dict | None:
        if self.type == "text":
            return None
        out: dict = {"type": self.type}
        if self.json_schema is not None:
            out["json_schema"] = self.json_schema.model_dump(by_alias=True)
        return out


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = Field(0, ge=0)  # extension; 0 = off
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int = Field(1, ge=1, le=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    user: str | None = None
    response_format: ResponseFormat | None = None
    # extension: a loaded LoRA adapter by name (the same as sending its name as `model`)
    adapter: str | None = None
    # extensions (as in vLLM): chat template variables on top of the preset's, e.g.
    # {"enable_thinking": true}; and a cap on the tokens inside a thinking block, after which
    # the block is closed and the model answers
    chat_template_kwargs: dict | None = None
    max_thinking_tokens: int | None = Field(None, ge=0)


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int  # includes the thinking block
    total_tokens: int
    # requests that could think: the reasoning tokens (block markers and forced tokens not
    # counted), also in OpenAI's completion_tokens_details
    thinking_tokens: int | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


class Timings(BaseModel):
    """Non-standard, but handy for benchmarks: server-side latency of this request."""

    queue_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    e2e_ms: float


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    reasoning_content: str | None = None  # the thinking block's text, when there was one


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: str | None


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    timings: Timings | None = None


class Delta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning_content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Usage | None = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "localhost-ai"
    root: str | None = None
    parent: str | None = None  # for a LoRA adapter, the base model it runs on


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class ErrorBody(BaseModel):
    message: str
    type: str
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class ControllerUpdate(BaseModel):
    mode: Literal["aimd", "fixed"] | None = None
    batch: int | None = Field(None, ge=1, le=1024)
    slo_tpot_ms: float | None = Field(None, gt=0)


class ModelLoad(BaseModel):
    model: str


class ScoreSite(BaseModel):
    char_offset: int = Field(ge=0)  # into the continuation
    candidates: list[Annotated[str, Field(max_length=256)]] = Field(min_length=1,
                                                                    max_length=64)


class ScoreRequest(BaseModel):
    """POST /v1/score: log-probabilities of candidate strings at sites of a fixed continuation
    of the assistant turn (teacher forcing, no sampling)."""

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    continuation: str
    sites: list[ScoreSite] = Field(min_length=1, max_length=1024)
    chat_template_kwargs: dict | None = None


class ScoreSiteResult(BaseModel):
    char_offset: int
    token_index: int
    candidates: dict[str, float]  # log-probability of each candidate at the site
    renorm: dict[str, float]  # the same, as probabilities renormalized over the candidates
    forced: bool  # teacher-forced from a token start instead of read in place


class ScoreResponse(BaseModel):
    model: str
    revision: str | None
    commit: str | None
    dirty: bool
    tokenizer_sha: str | None
    prompt_tokens: int  # prompt plus continuation, tokenized together
    sites: list[ScoreSiteResult]
