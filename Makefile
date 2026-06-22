TOOLS ?= ../.tools
PORT ?= 8000

.PHONY: setup models test test-model lint serve demo offline-check up down bench report sim

setup:  ## install locked deps (CPU/MPS torch) and fetch the pinned model
	uv sync --frozen --extra cpu --extra dev
	$(MAKE) models

models:  ## download pinned revisions into .models/ and check sha256 against models.lock
	uv run lhai models pull

test:
	uv run pytest -q

test-model:  ## batched-vs-sequential equivalence on the real model (CPU, fp32)
	uv run pytest -q -m model

lint:
	uv run ruff check .

serve:
	uv run lhai serve --port $(PORT)

demo:  ## offline: server + REST/SSE/WebSocket client round trip, no network needed
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PORT=$(PORT) scripts/smoke.sh

offline-check:  ## the demo plus egress canaries, with outbound network denied for the whole tree
	$(TOOLS)/offline-run scripts/offline_check.sh

up:
	docker compose -p lhai up --build -d

down:
	docker compose -p lhai down

bench:  ## every measurement in bench/RESULTS.md (takes the shared compute lease)
	$(TOOLS)/compute_lease.py run localhost-ai-bench -- bench/all.sh
	$(MAKE) report

report:
	uv run python bench/report.py

sim:
	uv run python scripts/sim_report.py
