# vocal-ai — Current Status & Backlog

> Updated: 2026-08-17
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
| 1 — Milestone 3: Export S3Gen flow estimator + Euler ODE loop, chain into HiFiGAN | ✅ Complete |
| 1 — Milestone 4: Export T3 as decoder-with-past, KV-cache decode loop + sampling | ✅ Complete |
| 1 — Milestones 5–7: ONNX export + Rust runtime (see `docs/phase1-onnx-rust-cli-plan.md` §7) | 📋 Planned |

*Update this table as phases progress. Use ✅ Complete / 🔄 In progress / 📋 Planned / 🚫 Blocked.*

**Current test counts**: 22 real Rust tests (`crates/vocalai-core/src/session.rs`'s EP-ordering
coverage — 2 under default features, +1 more under `--features coreml` — `s3gen.rs`'s 5
Euler-loop/CFG-math tests, and `t3.rs`'s 14 KV-cache-loop/sampling-math tests (CFG combine,
repetition penalty, temperature, min-p, top-p, greedy/multinomial selection, embedding-table
lookup, `.npy` round-trip, end-to-end synthetic decode loop), all pure `ndarray` math with no ONNX
Runtime session or model file needed) + 9 real Python tests in `export/` (`test_requirements.py`
checks `export/requirements.txt` pins; `test_parity_check.py` covers the `_common.allclose_report`
comparison helper plus end-to-end ONNX-vs-PyTorch parity for HiFiGAN, the voice encoder, the S3
tokenizer, the S3Gen flow estimator (mel→waveform, chained through the Milestone-2 HiFiGAN export),
and now T3 (greedy-decode token-sequence + per-step logits parity against the real
`transformers`-backed reference, `parity_check.py::check_t3`) — these download the Chatterbox
checkpoint from
HuggingFace on first run and require `export/requirements.txt` installed, see `docs/dev-setup.md`).
`make check` passes cleanly. No placeholder/ignored tests remain.

---

## What's next

The next logical work, in priority order. Update at the end of every session.

1. Milestone 5 — export PerthNet (from external `resemble-perth`), wire watermarking into the
   output pipeline (see `docs/phase1-onnx-rust-cli-plan.md` §7, milestone 5). Confirm license
   permits redistribution of exported weights (plan §9 Open Items).
2. See `docs/issues.md` for the tracked ticket (`VAI-005`).

---

## Recently closed

| Date | Ticket | Summary | Commit |
|------|--------|---------|--------|
| 2026-08-17 | VAI-004 | Export T3 as decoder-with-past (hand-rolled Llama, ADR-0005); KV-cache decode loop + sampling | `2e13c33` |
| 2026-08-16 | VAI-003 | Export S3Gen flow estimator + Euler ODE loop, chain into HiFiGAN | `1bc9095` |
| 2026-08-16 | VAI-002 | Export HiFiGAN/voice-encoder/S3-tokenizer to ONNX + `parity_check.py` | `820ff9a` |
| 2026-08-16 | VAI-001 | Cargo workspace + `ort` EP scaffold, export toolchain pins | `5b4815c` |
| 2026-08-16 | — | ai-sdlc-bootstrap scaffold | `ba0f453` |

---

## Deferred — pull only when a specific need surfaces

*Add tickets here when they're explicitly de-prioritized rather than open.*
