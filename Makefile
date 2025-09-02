SHELL := /bin/bash

# Python
PY := python
PIP := pip
VENV := .venv

# Rust
CARGO := cargo

.PHONY: help bootstrap run backend tui test fast-test cov fmt lint type mypy rust-fmt rust-clippy openapi-golden openapi-test build clean docs

help:
	@echo "Targets: bootstrap run backend tui test fast-test cov fmt lint type mypy rust-fmt rust-clippy openapi-golden openapi-test build clean docs"

bootstrap:
	@echo "[bootstrap] Creating virtualenv and installing Python deps"
	@if [ ! -d $(VENV) ]; then $(PY) -m venv $(VENV); fi
	. $(VENV)/bin/activate && $(PIP) install -U pip && $(PIP) install -r requirements.txt
	@echo "[bootstrap] Building Rust workspace"
	$(CARGO) build

run: backend

backend:
	@echo "[backend] Starting FastAPI server on :8000"
	$(PY) api_server.py

tui:
	@echo "[tui] Running Rust TUI"
	$(CARGO) run

test:
	@echo "[pytest] Running full test suite"
	. $(VENV)/bin/activate && pytest -v

fast-test:
	@echo "[pytest] Running fast tests"
	. $(VENV)/bin/activate && pytest -m "not integration and not slow" -q

cov:
	@echo "[pytest] Coverage run"
	. $(VENV)/bin/activate && pytest -v --cov --cov-report=term-missing

fmt:
	@echo "[fmt] Python format: black + isort"
	- . $(VENV)/bin/activate && black . && isort .
	@echo "[fmt] Rust format"
	- $(CARGO) fmt

lint:
	@echo "[lint] flake8"
	- . $(VENV)/bin/activate && flake8 . || true

type mypy:
	@echo "[type] mypy"
	- . $(VENV)/bin/activate && mypy . || true

rust-fmt:
	$(CARGO) fmt

rust-clippy:
	$(CARGO) clippy --all-targets --all-features -D warnings || true

openapi-golden:
	@echo "[openapi] Writing golden to tests/golden/openapi_v1.json"
	. $(VENV)/bin/activate && $(PY) - <<'PY'
import json, api_server
from pathlib import Path
Path('tests/golden').mkdir(parents=True, exist_ok=True)
with open('tests/golden/openapi_v1.json','w') as f: json.dump(api_server.app.openapi(), f, indent=2, sort_keys=True)
print('wrote golden')
PY

openapi-test:
	. $(VENV)/bin/activate && pytest -q tests/backend/test_openapi_contract.py -q

build:
	@echo "[build] Rust release + PyInstaller"
	$(CARGO) build --release
	. $(VENV)/bin/activate && $(PY) build_binary.py

docs:
	@echo "[docs] Quickstart at docs/QUICKSTART.md; runbook at docs/RUNBOOK.md"

clean:
	@echo "[clean] Removing build artifacts"
	rm -rf dist build target __pycache__ .pytest_cache htmlcov
