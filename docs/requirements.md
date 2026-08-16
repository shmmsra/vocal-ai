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
- [x] CI workflow mirrors the local gate
- [x] `docs/manual-testing.md` runbook initialised
- [x] Repo hygiene files scaffolded (README, LICENSE, CODEOWNERS, `.editorconfig`, VS Code settings)

---

## Phase 1 — ONNX + Rust TTS CLI (see `docs/phase1-onnx-rust-cli-plan.md`)

- [x] Milestone 1: Scaffold the real Cargo workspace (`vocalai-cli`, `vocalai-core`); pin `ort` with `coreml`/`cuda` features; set up `export/requirements.txt` with a working chatterbox + onnx env
- [x] Milestone 2: Export HiFiGAN → voice encoder → S3 tokenizer; stand up `export/parity_check.py`
- [ ] Milestone 3: Export S3Gen flow estimator; implement the Euler ODE loop; chain into HiFiGAN
- [ ] Milestone 4: Export T3 as decoder-with-past; implement the KV-cache decode loop + sampling
- [ ] Milestone 5: Export PerthNet; wire watermarking into output
- [ ] Milestone 6: Wire the full pipeline + `clap` CLI; support `--voice` zero-shot cloning
- [ ] Milestone 7: Per-platform packaging (macOS/CoreML, Windows/Linux CUDA, CPU fallback)

---

## Future phases

- [ ] Phase 2 — Tauri frontend wrapping the CLI as an `externalBin` sidecar
- [ ] Phase 3 — Claude skill wrapping the CLI
- [ ] Backlog — turbo/multilingual model variants; further ONNX optimization if profiling shows a bottleneck
