# vocal-ai — Requirements

> What this project must do, broken into phases. Use this to plan; use `docs/issues.md` to track individual tickets.
> Tick items as complete. Add new items as scope emerges.

---

## Phase 0 — Bootstrap (✅ done 2026-08-16)

- [x] Agent-config layer in place (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`)
- [x] `docs/agents/` triad written (OVERVIEW, CONVENTIONS, STATUS)
- [x] `CONTRIBUTING.md` defines the plan/test/commit workflow
- [x] `docs/issues.md` ticket tracker initialised
- [x] `docs/decisions/` ADR archive seeded with ADR-001
- [x] Pre-commit gate (`make check`) wired up
- [x] CI workflow mirrors the local gate — since 2026-08-17 split into two workflows
  (`ci.yml` fast, `parity.yml` real-checkpoint), same triggers, together covering
  everything `make check` covers locally; see ADR-0006
- [x] `docs/manual-testing.md` runbook initialised
- [x] Repo hygiene files scaffolded (README, LICENSE, CODEOWNERS, `.editorconfig`, VS Code settings)

---

## Phase 1 — ONNX + Rust TTS CLI (see `docs/phase1-onnx-rust-cli-plan.md`)

- [x] Milestone 1: Scaffold the real Cargo workspace (`vocalai-cli`, `vocalai-core`); pin `ort` with `coreml`/`cuda` features; set up `export/requirements.txt` with a working chatterbox + onnx env
- [x] Milestone 2: Export HiFiGAN → voice encoder → S3 tokenizer; stand up `export/parity_check.py`
- [x] Milestone 3: Export S3Gen flow estimator; implement the Euler ODE loop; chain into HiFiGAN
- [x] Milestone 4: Export T3 as decoder-with-past; implement the KV-cache decode loop + sampling
- [x] Milestone 5: Export PerthNet; wire watermarking into output
- [x] Milestone 6, part A (VAI-008): Export S3Gen's flow-encoder (bucketed, `TOKEN_BUCKETS`) +
  CAMPPlus (fixed 400-frame window) to ONNX, closing a gap found while starting Milestone 6 —
  only the CFM estimator (Milestone 3) had been exported; nothing produced real `mu`/`spks`. See
  ADR-0009.
- [x] Milestone 6, part B (VAI-006): Wire the full pipeline + `clap` CLI; support `--voice`
  zero-shot cloning — part B.1 (default-voice pipeline + CLI) verified end-to-end with real,
  arbitrary-length text (VAI-009 fixed HiFiGAN's ONNX export, which was the last blocker); part
  B.2 (`--voice` cloning: `voice_encoder.rs`, `s3tokenizer.rs`, `campplus.rs`, `mel.rs`) verified
  end-to-end producing genuine non-silent audio from a live reference wav
- [ ] Milestone 7: Per-platform packaging (macOS/CoreML, Windows/Linux CUDA, CPU fallback)

---

## Future phases

- [ ] Phase 2 — Tauri frontend wrapping the CLI as an `externalBin` sidecar
- [ ] Phase 3 — Claude skill wrapping the CLI
- [ ] Backlog — turbo/multilingual model variants; further ONNX optimization if profiling shows a bottleneck
