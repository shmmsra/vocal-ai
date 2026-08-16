# vocal-ai — Changelog

> Chronological log of what changed in this repo and *why*. The "why" matters more than the "what" — the diff already shows the what.
>
> Update at the end of every session. Newest entries at the top.

---

## 2026-08-16 — Architecture overview doc + diagram

**What changed**: Added `docs/architecture.md` (plain-language companion to `docs/phase1-onnx-rust-cli-plan.md`, written for a reader new to ML systems) and `docs/architecture-diagram.drawio.xml` (visualizes the dev-time Python export pipeline vs. the runtime Rust inference pipeline, including the optional voice-cloning branch and `session.rs`'s cross-cutting EP-selection role).

**Why**: The plan doc is dense and assumes ML background (ONNX, autoregressive decoding, flow matching). This doc explains the module map (`crates/vocalai-core/src/*.rs` responsibilities + milestone status), what Chatterbox's 4 sub-networks do, why ONNX export + Rust over shipping Python (links ADR-002/003), and the end-to-end request flow — so a newcomer can follow the architecture without reverse-engineering it from the plan or the code.

**What was rejected**: Re-deriving or restating the Milestone 2-7 technical decisions already in the plan doc — this is a companion, not a replacement; where the two disagree, the plan doc wins.

**What's next**: Keep this doc in sync as modules move from "planned" to "done" in the module map table (§5).

---

## 2026-08-16 — VAI-001: Cargo workspace + `ort` EP scaffold, export toolchain pins

**What changed**: Pinned `ort = "=2.0.0-rc.13"` in `vocalai-core`, with `coreml`/`cuda` Cargo features that pass through `vocalai-cli` and map to `ort`'s matching features (selected per release artifact via `--features`, per plan §2.3). Added `crates/vocalai-core/src/session.rs`: builds the execution-provider list in explicit fallback order (hardware EPs first, CPU always last, `.fail_silently()` made explicit in code) with real unit tests covering the default (CPU-only) and `coreml`-enabled builds. Pinned `export/requirements.txt` to real, verified versions (`chatterbox-tts==0.1.7`, `onnx==1.22.0`, `onnxruntime==1.28.0`, `pytest`) without installing them yet. Replaced both throwaway placeholder tests (`crates/vocalai-core`'s `#[ignore]`d Rust test, `export/tests/test_scaffold.py`) with real ones. Added ADR-002 (ONNX + Rust runtime, no Python-wrapper interim) and ADR-003 (Rust over C++), transcribing the already-made decisions from `docs/phase1-onnx-rust-cli-plan.md` §2.1/§2.2.

**Why**: Milestone 1 proves the toolchain (workspace, EP feature-gating, export env pins) before any model export work starts, so Milestones 2+ build on a working scaffold instead of discovering Cargo/feature issues mid-export. The EP ordering is a hard constraint (`CLAUDE.md` §1) and needed a real, tested implementation rather than a placeholder.

**What was rejected**: Actually pip-installing `export/requirements.txt` now (defers the multi-GB torch/chatterbox download to when export scripts are first run in Milestone 2). A `session.rs` unit test that detects *runtime* silent CPU fallback (needs a live session against a loaded model — deferred to Milestone 6).

**What's next**: Milestone 2 — export HiFiGAN, voice encoder, and S3 tokenizer to ONNX; stand up `export/parity_check.py` (`docs/issues.md` VAI-002).

---

## 2026-08-16 — ai-sdlc-bootstrap scaffold

**What changed**: Bootstrapped the AI-driven SDLC workflow on this repo via the `ai-sdlc-bootstrap` skill. Added agent-config layer (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`), `docs/agents/` triad, `CONTRIBUTING.md`, `docs/issues.md`, ADR template, pre-commit gate (`make check`), CI workflow, repo hygiene files (README, LICENSE, CODEOWNERS, `.editorconfig`, VS Code settings), and a throwaway placeholder Cargo workspace + Python `export/` stub carrying one intentionally-failing test each as a TDD seed.

**Why**: This project will be developed by humans + multiple AI agents across many sessions. Without the agent-config layer and a strict plan/test/commit workflow, every session would start from zero. The scaffold installs the contract.

**What was rejected**: Commit co-authorship trailer and the trailer-log/pre-commit-block commit-tracking modes — this is a solo-developer repo, convention-only tracking (no hook, no `docs/commit-log.md`) was chosen instead. PR-required merge policy was also rejected in favor of direct-merge-to-main.

**What's next**: Begin Phase 1 Milestone 1 (Cargo workspace scaffold, real `ort` wiring) as tracked in `docs/issues.md` (`VAI-001`).

---

*Add new entries above this line. Format: `## YYYY-MM-DD — Short title`, followed by `What / Why / Rejected / Next` sub-headings.*
