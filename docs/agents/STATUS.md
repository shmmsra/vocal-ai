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
| 1 — Milestone 1: Cargo workspace scaffold + export toolchain | 📋 Planned |
| 1 — Milestones 2–7: ONNX export + Rust runtime (see `docs/phase1-onnx-rust-cli-plan.md` §7) | 📋 Planned |

*Update this table as phases progress. Use ✅ Complete / 🔄 In progress / 📋 Planned / 🚫 Blocked.*

**Current test counts**: 1 throwaway placeholder Rust test (`crates/vocalai-core`, `#[ignore]`d)
+ 1 throwaway placeholder Python test (`export/tests/test_scaffold.py`, `@pytest.mark.skip`ped).
Both are ignored/skipped (rather than left failing) so `make check` stays green on the scaffold
commit — `make check` passes cleanly today. Both placeholders are TDD seeds only — remove the
ignore/skip marker (or delete them outright) once real Milestone 1 work starts (see
`docs/phase1-onnx-rust-cli-plan.md` §7).

---

## What's next

The next logical work, in priority order. Update at the end of every session.

1. Milestone 1 — scaffold the real Cargo workspace (`vocalai-cli`, `vocalai-core`), pin `ort`
   with `coreml`/`cuda` features, set up `export/requirements.txt` with a working
   chatterbox + onnx env (see `docs/phase1-onnx-rust-cli-plan.md` §7, milestone 1).
2. See `docs/issues.md` for the tracked ticket (`VAI-001`).

---

## Recently closed

| Date | Ticket | Summary | Commit |
|------|--------|---------|--------|
| 2026-08-16 | — | ai-sdlc-bootstrap scaffold | `ba0f453` |

---

## Deferred — pull only when a specific need surfaces

*Add tickets here when they're explicitly de-prioritized rather than open.*
