# vocal-ai — Feature & Issue Tracker

> **Single source of truth for all planned, in-progress, and recently completed work.**
>
> Use this instead of GitHub Issues or JIRA. It lives in the repo so AI agents can read it without
> any external system access, and every status change is committed alongside the code that caused it.

---

## How to use this file

**Human**: Add new issues at the bottom of the Open section. Set priority, write acceptance criteria. No need to assign — just set status to `IN PROGRESS` when a session starts on it.

**AI Agent**: Before starting a session, scan this file for the highest-priority `OPEN` issue that matches the session goal. Update the status to `IN PROGRESS` (with the session date) when you begin. Mark `DONE` and move to "Recently closed" when complete. Add any new issues you discover (bugs, missing tests, follow-up work) during the session.

---

## Status legend

| Status | Meaning |
|--------|---------|
| `OPEN` | Ready to work on, not yet started |
| `IN PROGRESS` | Actively being worked on — note session date |
| `BLOCKED` | Cannot proceed — reason and blocker recorded |
| `DONE` | Complete and committed — note commit hash |
| `REJECTED` | Will not implement — reason recorded |

## Priority legend

| Priority | Meaning |
|----------|---------|
| **P0** | Blocking — nothing else should be worked on until resolved |
| **P1** | High — next logical thing to do in the current phase |
| **P2** | Medium — important but not urgent; can wait one session |
| **P3** | Low — nice to have; do it when there's slack |

---

## Ticket ID convention

Tickets use the prefix `VAI-NNN`, numbered sequentially (e.g. `VAI-001`, `VAI-002`). When closing, reference the ticket ID in the commit message: `feat(scope): vai-042 add retry logic`.

---

## Open Issues

### VAI-006 · P2 · IN PROGRESS (session 2026-08-18) · Milestone 6
**Wire full pipeline in vocalai-core + clap CLI in vocalai-cli**

**Acceptance criteria**:
- [x] Full pipeline wired in `vocalai-core` (tokenizer → T3 → S3Gen → HiFiGAN → watermark → WAV) — default-voice path (part B.1)
- [x] `vocalai-cli` clap CLI implements flags per plan §3 (`--text`, `--voice`, `--exaggeration`, `--cfg-weight`, `--temperature`, `--repetition-penalty`, `--min-p`, `--top-p`, `--max-new-tokens`, `--out`)
- [ ] `--voice` zero-shot cloning: 16 kHz resample + mel + speaker embedding preprocessing (part B.2, not started — flag exists, returns a clear "not implemented" error)
- [x] Built-in default voice used when `--voice` is omitted
- [x] `vocalai --text "hello world" --out out.wav` produces audible, correct 24 kHz speech —
      unblocked by `VAI-009` (2026-08-18): confirmed non-silent, plausible-duration audio for both
      the acceptance-criterion command and a much longer sentence (~7.4s), see `docs/CHANGELOG.md`
- [x] Docs updated (CHANGELOG, STATUS, manual-testing, this ticket)

**Notes**: Depends on `VAI-005` and `VAI-008` (closed). Part B.1 (default-voice pipeline + CLI) is
code-complete, `make check`-clean, and now verified end-to-end for real, arbitrary-length text as
of 2026-08-18 (`VAI-009` closed). Part B.2 (`--voice` cloning: `voice_encoder.rs`, `s3tokenizer.rs`,
`campplus.rs`, the mel-filterbank builder) has not been started — this is the only remaining item
before VAI-006 itself can close. See `docs/phase1-onnx-rust-cli-plan.md` §7 Milestone 6 and §8
Verification/Exit Criteria (end-to-end, voice cloning).

---

### VAI-007 · P2 · OPEN · Milestone 7
**Per-platform packaging: artifact matrix, bundling, smoke tests**

**Acceptance criteria**:
- [ ] Build artifact matrix per §2.3: `vocalai-macos` (CoreML→CPU), `vocalai-windows-cuda`, `vocalai-linux-cuda`, `vocalai-{win,linux}-cpu`
- [ ] GPU artifacts bundle required CUDA/cuDNN libs; macOS/CPU artifacts stay lean
- [ ] Model weights bundled into each release artifact
- [ ] Each artifact smoke-tested; CPU-fallback EP forcing produces equivalent output
- [ ] Memory/swap measured against the PyTorch/MPS baseline (§8) on the same hardware
- [ ] Docs updated (CHANGELOG, STATUS, manual-testing)

**Notes**: Depends on `VAI-006`. See `docs/phase1-onnx-rust-cli-plan.md` §7 Milestone 7 and §8 Verification/Exit Criteria (full list).

---

*Add new tickets below this line. Use the same format: heading with ID · priority · status · brief category; then bold one-line title; then acceptance criteria as checkboxes; then notes.*

---

## Recently closed

| Date | Ticket | Title | Commit |
|------|--------|-------|--------|
| 2026-08-18 | VAI-009 | Fix two trace-baking bugs in `export_hifigan.py` (envelope `.repeat(int)`, noise-buffer size) so `speech_feat` is a genuine dynamic ONNX axis; extend `check_hifigan` to 3 frame counts — unblocks VAI-006's end-to-end acceptance criterion | `9ff4b4e` |
| 2026-08-18 | VAI-008 | Export S3Gen's flow-encoder (bucketed) + CAMPPlus (fixed window) to ONNX, closing the `mu`/`spks` export gap found while starting VAI-006 | `65b1642` |
| 2026-08-18 | VAI-005 | Export PerthNet encoder; implement STFT/ISTFT/resample watermarking pipeline in `watermark.rs` | `91c92ea` |
| 2026-08-17 | VAI-004 | Export T3 as decoder-with-past (hand-rolled Llama, ADR-0005); implement KV-cache decode loop + sampling (greedy-decode parity, see notes) | `2e13c33` |
| 2026-08-16 | VAI-003 | Export S3Gen flow estimator + Euler ODE loop, chain into HiFiGAN | `1bc9095` |
| 2026-08-16 | VAI-002 | Export HiFiGAN/voice-encoder/S3-tokenizer to ONNX + `parity_check.py` | `820ff9a` |
| 2026-08-16 | VAI-001 | Scaffold the real Cargo workspace + export toolchain | `5b4815c` |
| 2026-08-16 | — | ai-sdlc-bootstrap scaffold | `ba0f453` |

*When a ticket is closed: move it to this table, set the commit hash, and remove it from the Open section. Keep the last ~20 closures here; archive older ones to `docs/CHANGELOG.md`.*
