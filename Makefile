.PHONY: check setup-hooks test typecheck lint build clean help export

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "Available targets:"
	@echo "  make check         Pre-commit gate: fmt-check + clippy + cargo test + pytest (everything)"
	@echo "  make setup-hooks   Install .git/hooks/pre-commit (run once after clone)"
	@echo "  make test          Run the test suite only (Rust + Python, everything)"
	@echo "  make test-py-fast       Python tests that don't need a real checkpoint download"
	@echo "  make test-py-parity     Python tests that download a real checkpoint + run ONNX-vs-PyTorch parity"
	@echo "  make test-py-parity-ci  Same, minus heavy_build tests too large to export on a CI runner"
	@echo "  make typecheck     Run cargo check only"
	@echo "  make lint          Run clippy only"
	@echo "  make build         Build the project"
	@echo "  make export        Generate models/*.onnx + *.npy (ARGS=--with-voice-cloning for --voice support)"
	@echo "  make clean         Remove build artifacts"

# ── Python interpreter selection ──────────────────────────────────────────────
# The export venv lives at .venv/bin/python on POSIX, .venv/Scripts/python.exe on
# Windows. Pick the right layout by OS, then prefer that interpreter if it exists
# (checked with make's own $(wildcard) so no POSIX shell is required — Windows
# runs recipes through cmd.exe, which can't parse `if [ -x ... ]; then ... fi`).
ifeq ($(OS),Windows_NT)
  # cmd.exe needs backslashes for the leading executable path; $(wildcard) needs
  # forward slashes. Keep one of each.
  VENV_PY := .venv\Scripts\python.exe
  VENV_PY_GLOB := export/.venv/Scripts/python.exe
else
  VENV_PY := .venv/bin/python
  VENV_PY_GLOB := export/.venv/bin/python
endif

PYTEST := $(if $(wildcard $(VENV_PY_GLOB)),$(VENV_PY) -m pytest,pytest)

# ── Pre-commit gate ──────────────────────────────────────────────────────────

check: fmt-check clippy test-rs test-py

# ── Hook installation ─────────────────────────────────────────────────────────

setup-hooks:
	bash scripts/setup-hooks.sh

# ── Language-specific targets ─────────────────────────────────────────────────

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

typecheck:
	cargo check --workspace

test-rs:
	cargo test --workspace

test-py:
	cd export && $(PYTEST)

# `-m "not parity"`/`-m parity` split the suite along the same line as
# .github/workflows/ci.yml vs parity.yml — see export/pytest.ini for the marker
# definition and docs/decisions/0006-split-ci-into-fast-and-parity-workflows.md
# for why they're two separate jobs.
test-py-fast:
	cd export && $(PYTEST) -m "not parity"

test-py-parity:
	cd export && $(PYTEST) -m parity

# What CI's parity.yml actually runs: same as test-py-parity minus `heavy_build`
# tests (currently just T3 — its first-time ONNX export peaks at ~9GB, more than
# a free-tier CI runner has; see docs/decisions/0007-exclude-t3-parity-from-ci.md).
# `heavy_build` tests still run locally via plain `test-py-parity`/`check`.
test-py-parity-ci:
	cd export && $(PYTEST) -m "parity and not heavy_build"

test: test-rs test-py

lint: clippy

build:
	cargo build --workspace --release

# ── Model artifact generation ─────────────────────────────────────────────────
# Runs export/'s scripts in the order docs/dev-setup.md §11.1 documents, writing
# to <repo>/models/ (git-ignored, never committed — see CLAUDE.md §1). Pass
# ARGS=--with-voice-cloning to also export the two models `--voice` needs.

export:
ifeq ($(OS),Windows_NT)
	powershell -ExecutionPolicy Bypass -File scripts\export-all.ps1 $(ARGS)
else
	bash scripts/export-all.sh $(ARGS)
endif

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	rm -rf target/ export/__pycache__ export/.pytest_cache export/**/__pycache__
