PORT ?= 8000
# Optional host wrappers, empty by default:
#   LHAI_OFFLINE_RUN  command that runs its arguments with outbound network denied
#   LHAI_LEASE        command prefix that serializes heavy jobs on a shared machine
LHAI_OFFLINE_RUN ?=
LHAI_LEASE ?=

.PHONY: setup setup-mlx models test test-model test-mlx lint serve demo offline-check up down bench bench-mlx report sim

setup:  ## install locked deps (CPU/MPS torch) and fetch the pinned model
	uv sync --frozen --extra cpu --extra dev
	$(MAKE) models

setup-mlx:  ## apple silicon: also install mlx and fetch the 4-bit MLX preset
	uv sync --frozen --extra cpu --extra dev --extra mlx
	$(MAKE) models
	LHAI_MODEL=qwen2.5-0.5b-mlx4 uv run lhai models pull

models:  ## download pinned revisions into .models/ and check sha256 against models.lock
	uv run lhai models pull

test:
	uv run pytest -q

test-model:  ## batched-vs-sequential equivalence on the real model (CPU, fp32)
	uv run pytest -q -m model

test-mlx:  ## batched-vs-sequential equivalence on the 4-bit MLX checkpoints (apple silicon)
	uv run pytest -q -m mlx_model

lint:
	uv run ruff check .

serve:
	uv run lhai serve --port $(PORT)

demo:  ## offline: server + REST/SSE/WebSocket client round trip, no network needed
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PORT=$(PORT) scripts/smoke.sh

offline-check:  ## the demo plus egress canaries; set LHAI_OFFLINE_RUN to a network-denying wrapper
	@test -n "$(LHAI_OFFLINE_RUN)" || { echo "set LHAI_OFFLINE_RUN (e.g. a sandbox-exec wrapper)"; exit 2; }
	$(LHAI_OFFLINE_RUN) scripts/offline_check.sh

up:
	docker compose -p lhai up --build -d

down:
	docker compose -p lhai down

bench:  ## every measurement in bench/RESULTS.md (prefix with LHAI_LEASE on a shared machine)
	$(LHAI_LEASE) bench/all.sh
	$(MAKE) report

bench-mlx:  ## the mlx sections of bench/RESULTS.md (apple silicon, mlx extra)
	$(LHAI_LEASE) bench/mlx.sh
	$(MAKE) report

report:
	uv run python bench/report.py

sim:
	uv run python scripts/sim_report.py
