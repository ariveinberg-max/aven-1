# Single entry point for common tasks (WSL2/Linux, macOS, CI).
.PHONY: setup setup-ml demo-model lock check lint format typecheck layers test cov smoke guard api web web-check clean

UV ?= uv

setup:            ## Install core + dev + api deps and git hooks
	$(UV) sync --extra api
	$(UV) run pre-commit install

setup-ml:         ## Full ML stack (MNE, MOABB, PyTorch, MLflow): use on the GPU PC
	$(UV) sync --extra api --extra neuro --extra dl --extra tracking

lock:             ## Re-resolve uv.lock after dependency changes
	$(UV) lock

check: lint typecheck layers test guard  ## Everything CI runs for Python

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck:
	$(UV) run mypy

layers:           ## Architecture layering contracts (import-linter)
	$(UV) run lint-imports

test:
	$(UV) run pytest

cov:
	$(UV) run pytest --cov --cov-report=term-missing

smoke:            ## End-to-end CAP-1 harness run on synthetic data
	$(UV) run neurolayer smoke

guard:            ## No data/weights tracked in git
	$(UV) run python scripts/check_no_data_files.py --all

demo-model:       ## Train the synthetic demo bundle used by the local API and dashboard
	$(UV) run neurolayer train configs/experiments/models/nl_synthetic_demo.yaml --version 0.1.0

api:              ## Local API (auth disabled: development only)
	NEUROLAYER_API_DOCS=1 NEUROLAYER_AUTH_DISABLED=1 $(UV) run uvicorn neurolayer_api.app:app --reload --port 8000

web:
	cd apps/web && npm install && npm run dev

web-check:
	cd apps/web && npm ci && npm run lint && npm run typecheck && npm run build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis .coverage htmlcov dist build
