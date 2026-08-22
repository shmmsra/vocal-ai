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

### VAI-015 · P3 · OPEN · CI/Release
**Windows/Linux CUDA/cuDNN-bundled GPU release artifacts**

**Acceptance criteria**:
- [ ] `vocalai-windows-cuda`/`vocalai-linux-cuda` artifacts in `release.yml`: built with
  `--features vocalai-cli/cuda`, bundling the required CUDA runtime + cuDNN libs (plan §2.3)
- [ ] NVIDIA's redistribution terms for the bundled CUDA runtime + cuDNN libs checked and
  recorded (an ADR), same rigor as ADR-0008's model-weight license check — not yet done
- [ ] Real GPU-hardware smoke test (structural-only CI smoke test doesn't cover actual CUDA
  inference — per the repo owner's instruction, that stays manual; see `docs/manual-testing.md`)
- [ ] Docs updated (CHANGELOG, STATUS, manual-testing, dev-setup)

**Notes**: Split out of `VAI-007` (see `docs/decisions/0013-hf-hub-model-distribution-and-release-packaging.md`)
because it needs real Windows/Linux+GPU hardware to verify on and an unresolved licensing
question, neither of which blocked shipping the CoreML+CPU-only artifact matrix.

---

### VAI-014 · P3 · OPEN · Export/ONNX
**Bucket `s3gen_estimator.onnx`'s time dimension so CoreML can run the full pipeline**

**Acceptance criteria**:
- [ ] `export/export_s3gen.py` exports the flow estimator with a bucketed/padded time dimension (mirroring the flow-encoder's existing `TOKEN_BUCKETS` trick, ADR-0009), instead of the current genuinely dynamic `2 * token_len`
- [ ] `export/parity_check.py::check_s3gen` extended to cover the bucketed shapes, matching `check_s3gen_flow_encoder`'s padding-invariance property
- [ ] Once bucketed, remove the CPU pin added in VAI-011/`pipeline.rs::ModelBundle::load` (the `ResolvedProvider::Gpu("CoreML")` special case) since CoreML should then handle the estimator like every other session
- [ ] Manually verified CoreML runs the full pipeline (all sessions) with no CPU carve-out

**Notes**: The *real* fix for the CoreML crash worked around in VAI-011. Root-caused by manual bisection during VAI-011: `s3gen_estimator.onnx`'s UNet-style (`down_blocks`/attention) architecture combined with a truly dynamic (non-bucketed) sequence length reliably crashes ONNX Runtime's CoreML EP at inference time (`Unable to compute the prediction using a neural network model`), even though CoreML's `GetCapability` accepts a partition of the graph. Every other session in the pipeline — including T3's full up-to-1000-iteration KV-cache decode loop — already runs on CoreML successfully, so this is scoped narrowly to the estimator, not a general CoreML-viability problem.

---

### VAI-010 · P3 · OPEN · Docs
**Split `docs/dev-setup.md` into a dev-setup doc and a usage doc, once there's a real end-user usage story**

**Acceptance criteria**:
- [ ] `docs/dev-setup.md` keeps §1–8 (toolchain, clone, hooks, `make check`, IDE, agent-workflow
      verification, troubleshooting) — pure contributor onboarding.
- [ ] A new doc (e.g. `docs/usage.md`) takes §9 (generate model artifacts, build the CLI,
      synthesize speech, tuning flags) and anything else that's about *running* `vocalai` rather
      than *contributing to* it.
- [ ] Cross-links updated in both directions; `CLAUDE.md` §2 item 7's dev-setup pointer and any
      other doc referencing dev-setup.md's old section numbers get checked for staleness.

**Notes**: Deliberately deferred, not started now — today's §9 is still dev-audience work (no
prebuilt binary yet, same Rust/Python toolchain as §1–2 either way), so the setup/usage line is
blurry. Revisit once Milestone 7 (`VAI-007`) ships prebuilt binaries + bundled models, when a real
end-user usage doc (no Rust/Python toolchain, no `make`) becomes a genuinely different document
for a genuinely different reader. Depends on `VAI-007`.

---

*Add new tickets below this line. Use the same format: heading with ID · priority · status · brief category; then bold one-line title; then acceptance criteria as checkboxes; then notes.*

---

## Recently closed

| Date | Ticket | Title | Commit |
|------|--------|-------|--------|
| 2026-08-22 | VAI-012 | `--show-progress` flag: `ProgressEvent`/`PipelinePhase` threaded through `pipeline::synthesize` (wraps the existing `decoder_step` closure, no `t3.rs` changes), rendered via `indicatif` in `vocalai-cli` only. Plus, at the repo owner's request while approving the plan (not original ticket scope): `scripts/install.sh`/`install.ps1` track installed CLI/model versions (`.vocalai_version`, reused `MODELS_VERSION`) and skip re-downloading whichever half is already up to date — see ADR-0015 for the three real design bugs review/testing caught (a shared-IP `api.github.com` rate limit; a premature version-marker write on an interrupted download; and a since-corrected design that silently fell back to downloading on *any* version-check failure, including a confirmed rate limit — now fails fast with a specific error instead, on explicit repo-owner instruction). `install.ps1`'s half, and a full completed model download for `install.sh`, not yet verified live (no Windows machine; full download intentionally not run to completion this session, see `docs/manual-testing.md`). | (pending commit) |
| 2026-08-22 | VAI-007 | Per-platform packaging (artifact matrix, HF Hub model publish, one-line installers) closed `DONE` by repo owner decision. Real end-to-end install+synthesis verified on macOS (`v0.1.2`) and Windows (`v0.1.3`, this session — both default-EP and `--use-cpu` produced non-silent audio). **Not live-tested on Linux** — no Linux machine available, closed anyway per repo owner's call rather than left open pending hardware access. The PyTorch/MPS memory/swap-benchmark comparison (§8) was also not attempted (macOS-specific baseline, no Windows/Linux equivalent). CUDA/cuDNN GPU artifacts remain split out to `VAI-015`. | — |
| 2026-08-22 | VAI-013 | Closed as `REJECTED`/superseded by `VAI-007`'s `.github/workflows/release.yml` (macos-latest/windows-latest/ubuntu-latest matrix, build-only verified without GPU hardware, per-OS artifacts uploaded, GPU inference explicitly out of scope) — confirmed after `VAI-007`'s workflows actually ran live (see VAI-016's closing runs `32571043242`/`32571043262`). Note: win/linux legs build CPU-only, not `--features cuda` as VAI-013's literal wording specified — CUDA artifacts are intentionally deferred to `VAI-015`, not a gap in this closure. Repo owner confirmed. | — |
| 2026-08-22 | VAI-016 | Version-bump-driven `push`+`paths` triggers for `models-export.yml`/`release.yml` (replacing manual dispatch/tag-push as the primary flow, ADR-0014); closed after a real push to `main` (`e9f4825`) produced a genuine `v0.1.3` release (run `32571043242`) and `models-v0.1.1` HF Hub publish (run `32571043262`), both fired via `push` not `workflow_dispatch` | `d4c64ad`, `e9f4825` |
| 2026-08-19 | VAI-011 | `--use-gpu`/`--use-cpu` execution-provider selection (CPU by default, logged); `error_on_failure()` instead of silent fallback for forced/probed hardware attempts; `Makefile` OS-based feature auto-detection (`coreml` on macOS, `cuda` elsewhere); benchmarked and fixed a 30-40% CoreML slowdown vs CPU (`CPUAndGPU`+`FastPrediction`+`RequireStaticInputShapes`); S3Gen flow estimator pinned to CPU on CoreML (see `VAI-014`) | `f1c74af` |
| 2026-08-18 | VAI-006 | Wire full pipeline in `vocalai-core` + clap CLI in `vocalai-cli`, including part B.2 `--voice` zero-shot cloning (`voice_encoder.rs`, `s3tokenizer.rs`, `campplus.rs`, `mel.rs`'s hand-rolled mel/Kaldi-fbank DSP) | `fbcf484`, `4d97b0d` |
| 2026-08-18 | VAI-009 | Fix two trace-baking bugs in `export_hifigan.py` (envelope `.repeat(int)`, noise-buffer size) so `speech_feat` is a genuine dynamic ONNX axis; extend `check_hifigan` to 3 frame counts — unblocks VAI-006's end-to-end acceptance criterion | `9ff4b4e` |
| 2026-08-18 | VAI-008 | Export S3Gen's flow-encoder (bucketed) + CAMPPlus (fixed window) to ONNX, closing the `mu`/`spks` export gap found while starting VAI-006 | `65b1642` |
| 2026-08-18 | VAI-005 | Export PerthNet encoder; implement STFT/ISTFT/resample watermarking pipeline in `watermark.rs` | `91c92ea` |
| 2026-08-17 | VAI-004 | Export T3 as decoder-with-past (hand-rolled Llama, ADR-0005); implement KV-cache decode loop + sampling (greedy-decode parity, see notes) | `2e13c33` |
| 2026-08-16 | VAI-003 | Export S3Gen flow estimator + Euler ODE loop, chain into HiFiGAN | `1bc9095` |
| 2026-08-16 | VAI-002 | Export HiFiGAN/voice-encoder/S3-tokenizer to ONNX + `parity_check.py` | `820ff9a` |
| 2026-08-16 | VAI-001 | Scaffold the real Cargo workspace + export toolchain | `5b4815c` |
| 2026-08-16 | — | ai-sdlc-bootstrap scaffold | `ba0f453` |

*When a ticket is closed: move it to this table, set the commit hash, and remove it from the Open section. Keep the last ~20 closures here; archive older ones to `docs/CHANGELOG.md`.*
