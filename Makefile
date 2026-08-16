.PHONY: check setup-hooks test typecheck lint build clean help

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "Available targets:"
	@echo "  make check         Pre-commit gate: fmt-check + clippy + cargo test + pytest"
	@echo "  make setup-hooks   Install .git/hooks/pre-commit (run once after clone)"
	@echo "  make test          Run the test suite only (Rust + Python)"
	@echo "  make typecheck     Run cargo check only"
	@echo "  make lint          Run clippy only"
	@echo "  make build         Build the project"
	@echo "  make clean         Remove build artifacts"

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
	cd export && if [ -x .venv/bin/python ]; then .venv/bin/python -m pytest; else pytest; fi

test: test-rs test-py

lint: clippy

build:
	cargo build --workspace --release

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	rm -rf target/ export/__pycache__ export/.pytest_cache export/**/__pycache__
