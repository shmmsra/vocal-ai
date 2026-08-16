# vocal-ai — Current Status & Backlog

> Updated: 2026-08-16
> For the full feature history see [`docs/CHANGELOG.md`](../CHANGELOG.md).
> For per-ticket detail see [`docs/issues.md`](../issues.md).
> For the full phase breakdown see [`docs/requirements.md`](../requirements.md).

---

## Phase status

| Phase | Status |
|-------|--------|
| 0 — Project bootstrap (AI SDLC) | ✅ Complete |
| 1 — Milestone 1: Cargo workspace scaffold + export toolchain | ✅ Complete |
| 1 — Milestone 2: Export HiFiGAN/voice-encoder/S3-tokenizer + parity harness | ✅ Complete |
| 1 — Milestones 3–7: ONNX export + Rust runtime (see `docs/phase1-onnx-rust-cli-plan.md` §7) | 📋 Planned |

*Update this table as phases progress. Use ✅ Complete / 🔄 In progress / 📋 Planned / 🚫 Blocked.*

**Current test counts**: 3 real Rust tests (`crates/vocalai-core/src/session.rs`, EP-ordering
coverage; 2 under default features, +1 more under `--features coreml`) + 7 real Python tests
in `export/` (`test_requirements.py` checks `export/requirements.txt` pins; `test_parity_check.py`
covers the `_common.allclose_report` comparison helper plus end-to-end ONNX-vs-PyTorch parity
for HiFiGAN, the voice encoder, and the S3 tokenizer — these three download the Chatterbox
checkpoint from HuggingFace on first run and require `export/requirements.txt` installed, see
`docs/dev-setup.md`). `make check` passes cleanly. No placeholder/ignored tests remain.

---

## What's next

The next logical work, in priority order. Update at the end of every session.

1. Milestone 3 — export the S3Gen flow estimator, implement the Euler ODE loop in
   `vocalai-core/src/s3gen.rs`, chain into HiFiGAN (see `docs/phase1-onnx-rust-cli-plan.md` §7,
   milestone 3).
2. See `docs/issues.md` for the tracked ticket (`VAI-003`).

---

## Recently closed

| Date | Ticket | Summary | Commit |
|------|--------|---------|--------|
| 2026-08-16 | VAI-002 | Export HiFiGAN/voice-encoder/S3-tokenizer to ONNX + `parity_check.py` | `820ff9a` |
| 2026-08-16 | VAI-001 | Cargo workspace + `ort` EP scaffold, export toolchain pins | `5b4815c` |
| 2026-08-16 | — | ai-sdlc-bootstrap scaffold | `ba0f453` |

---

## Deferred — pull only when a specific need surfaces

*Add tickets here when they're explicitly de-prioritized rather than open.*
