# vocal-ai — Project Overview

> Canonical reference for project context, architecture, tech stack, and build commands.
> Read before asking architecture questions or evaluating new libraries/frameworks.

---

## What is this project?

**vocal-ai** — a fully self-contained, cross-platform command-line TTS tool built around the
Chatterbox TTS model, using ONNX + native Rust inference (no bundled Python/PyTorch, no
multi-GB runtime download).

**Owner**: Shivam Mishra
**AI-first SDLC**: Designed to be built by humans and multiple AI agents across many sessions.
Every significant decision and status change is committed to this repo so agents never need
manual context transfer.

---

## Architecture in 30 seconds

```
text ──► [tokenizer] ──► [T3 backbone: KV-cache decode loop] ──► speech tokens
                                                                       │
ref .wav ──► [voice encoder + S3 tokenizer] ──────────────────────────┤
                                                                       ▼
                                            [S3Gen flow estimator: Euler ODE loop]
                                                                       │
                                                                       ▼
                                            [HiFiGAN vocoder] ──► [PerthNet watermark]
                                                                       │
                                                                       ▼
                                                              24kHz mono WAV
```

Every static per-step network (T3 backbone, S3Gen flow estimator, HiFiGAN, voice encoder, S3
tokenizer, PerthNet) is exported to ONNX. The autoregressive/iterative loops (T3's token decode,
S3Gen's Euler ODE solver) are hand-driven from Rust against the `ort` crate — they are not
in-graph control flow.

For the full design see [`docs/phase1-onnx-rust-cli-plan.md`](../phase1-onnx-rust-cli-plan.md).

---

## Key decisions (quick reference)

Full ADRs in [`docs/decisions/`](../decisions/). Read the ADR before changing anything related
to that decision.

| # | Decision | Short rationale |
|---|----------|-----------------|
| [ADR-001](../decisions/0001-adopt-ai-sdlc.md) | Adopt the ai-sdlc-bootstrap workflow | Plan-first, TDD-enforced, docs-as-done, multi-agent compatible |

*Add new rows as ADRs accumulate — the ONNX-vs-Python-wrapper and Rust-vs-C++ decisions from
the Phase 1 plan (§2.1, §2.2) are good ADR-002/003 candidates once Milestone 1 starts.*

---

## Tech stack

| Layer | Tech | Key files |
|-------|------|-----------|
| Language(s) | Rust (`crates/`, product code), Python (`export/`, dev-time only) | — |
| Testing | `cargo test` (Rust) + `pytest` (Python `export/`) | `crates/*/tests/`, `export/tests/` |
| CI | GitHub Actions | `.github/workflows/ci.yml` |
| Build | Make | `Makefile` |
| Inference runtime | `ort` (ONNX Runtime Rust bindings) — CoreML / CUDA / CPU execution providers | `crates/vocalai-core/src/session.rs` (Milestone 1) |

---

## Build and run

```bash
make check          # pre-commit gate — run before every commit
make setup-hooks     # one-time: installs the pre-commit hook
```

**First-time setup**: see [`docs/dev-setup.md`](../dev-setup.md) for the full bootstrap (Rust
toolchain, `ort`/onnxruntime, Python `export/` environment, VS Code extensions). The dev-setup
doc is the single source of truth for onboarding — if it's out of date, fix it in the same
commit as whatever broke it.

Credentials (if any): see `.env.example` for the required variables. **Never** commit `.env`.

---

## Repository layout

```
.
├── README.md                          # Project description + dev-setup pointer
├── LICENSE                            # MIT
├── CODEOWNERS                         # Default ownership
├── .gitignore                         # Rust + Python aware ignores
├── .editorconfig
├── .vscode/                           # Recommended settings + extensions
├── docs/
│   ├── agents/                        # AGENT-CRITICAL: OVERVIEW.md, CONVENTIONS.md, STATUS.md
│   ├── decisions/                     # ADRs
│   ├── CHANGELOG.md
│   ├── requirements.md
│   ├── issues.md
│   ├── manual-testing.md
│   ├── dev-setup.md
│   └── phase1-onnx-rust-cli-plan.md   # Approved Phase 1 implementation plan
├── scripts/
│   └── setup-hooks.sh                 # Installs the pre-commit gate
├── CLAUDE.md / AGENTS.md / GEMINI.md  # Agent entry points
├── CONTRIBUTING.md                    # Workflow + merge policy + hygiene rules
├── Makefile                           # `make check`, `make setup-hooks`
├── .github/workflows/ci.yml
├── Cargo.toml                         # Rust workspace (crates/vocalai-cli, crates/vocalai-core)
├── crates/                            # Rust product code (placeholder pending Milestone 1)
└── export/                            # Python ONNX export scripts, dev-time only
```

*Update this tree as the project grows. Agents read it to navigate.*

---

## Further reading

External docs / wikis that were used to inform this scaffold (read these for deeper context):

- [Phase 1 Plan: Standalone ONNX + Rust TTS CLI](../phase1-onnx-rust-cli-plan.md) — approved
  implementation plan; source of the architecture, tech stack, and domain rules in this scaffold.
