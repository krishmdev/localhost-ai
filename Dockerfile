# CPU image by default; docker-compose.gpu.yml rebuilds it with TORCH_EXTRA=cu126.
FROM ghcr.io/astral-sh/uv:0.9.28 AS uv

FROM python:3.11.13-slim-bookworm
COPY --from=uv /uv /uvx /usr/local/bin/
ARG TORCH_EXTRA=cpu
ARG EXTRA_FLAGS=""
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ${TORCH_EXTRA} ${EXTRA_FLAGS} --no-install-project
COPY src ./src
COPY models.yaml models.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ${TORCH_EXTRA} ${EXTRA_FLAGS} --no-editable

RUN useradd --create-home --uid 1000 lhai
USER lhai
# Weights are mounted read-only from the host's .models/ (filled by `make models`), and the
# container never needs the network at runtime.
ENV PATH=/app/.venv/bin:$PATH \
    LHAI_HOST=0.0.0.0 LHAI_PORT=8000 \
    LHAI_MODELS_DIR=/models LHAI_MODELS_FILE=/app/models.yaml LHAI_MODELS_LOCK=/app/models.lock \
    HF_HOME=/tmp/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2).status == 200 else 1)"]
CMD ["lhai", "serve"]
