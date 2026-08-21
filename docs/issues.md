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

### VAI-007 · P2 · IN PROGRESS (2026-08-21) · Milestone 7
**Per-platform packaging: artifact matrix, bundling, smoke tests**

**Acceptance criteria** (narrowed scope — CUDA-bundled GPU artifacts split out to VAI-015; see
ADR-0013 for the full rationale):
- [x] `.github/workflows/models-export.yml`: manual-trigger export → full `make test-py-parity`
  gate (including T3) → structural smoke test → publish to public HF Hub repo
  (`shmmsra/vocal-ai-models`)
- [x] `.github/workflows/release.yml`: build artifact matrix — `vocalai-macos` (CoreML→CPU),
  `vocalai-windows-cpu`, `vocalai-linux-cpu` — on a `v*` tag or manual dispatch
- [x] Model weights downloaded from HF Hub inside the release job and structurally validated;
  **not** archived into the release asset (see ADR-0013's 2026-08-21 addendum — GitHub caps
  release assets at 2GiB/file, the ~4GB model set doesn't fit). `THIRD_PARTY_LICENSES` is still
  bundled with the binary (fulfills ADR-0008).
- [x] `scripts/install.sh`/`install.ps1` + `README.md` "Install" section: one-line installer that
  fetches the release binary from GitHub and every model file from the public HF repo
  (anonymously, no token) into `./vocalai/` — the effective replacement for "bundle everything
  into one artifact"
- [x] Repo owner: flipped GitHub visibility to public, added `HF_TOKEN` repo secret
- [x] A real `models-export.yml` run against the live public HF repo — verified (26 model files
  present, `THIRD_PARTY_LICENSES` byte-identical, structural smoke test passes on a fresh
  download)
- [ ] A real `v*` tag / `release.yml` run producing a real, downloadable release asset (in
  progress — `v0.1.0`/`v0.1.1` hit the permissions and asset-size bugs above; retry pending)
- [ ] Manual per-platform validation (`docs/manual-testing.md`): real end-to-end audio,
  CPU-fallback EP forcing produces equivalent output, memory/swap measured against the
  PyTorch/MPS baseline (§8) on the same hardware
- [ ] Docs updated (CHANGELOG, STATUS, manual-testing) — done for the workflow/tooling landing;
  revisit once the above manual steps complete

**Notes**: Depends on `VAI-006`. See `docs/phase1-onnx-rust-cli-plan.md` §7 Milestone 7 and §8
Verification/Exit Criteria (full list), and `docs/decisions/0013-hf-hub-model-distribution-and-release-packaging.md`
for the distribution-design decisions (bundle-at-build, CI-driven export/publish, structural-only
smoke tests, corrected GitHub-hosted-runner specs). Windows/Linux CUDA/cuDNN bundling split out
as `VAI-015`. `VAI-013` (GitHub Actions cross-platform build matrix) is likely superseded by
`release.yml` — not closed here, pending confirmation.

---

### VAI-012 · P2 · OPEN · CLI UX
**Console progress indicator behind `--show-progress`**

**Acceptance criteria**:
- [ ] `--show-progress` flag on `vocalai` (default off, no behavior/output change without it)
- [ ] Progress reported for T3's autoregressive decode loop (dominant runtime cost, up to `--max-new-tokens` steps) via a callback threaded through `pipeline::synthesize`, plus coarse phase labels for voice conditioning / S3Gen vocoding / watermarking
- [ ] Rendering (e.g. `indicatif`) lives in `vocalai-cli` only; `vocalai-core` stays UI-agnostic (a plain callback/event type, no UI crate dependency)
- [ ] Manual test steps added to `docs/manual-testing.md`

**Notes**: Split out of `VAI-011` at the user's request so the execution-provider work could ship independently. See the approved plan in the VAI-011 session for the exact `ProgressEvent`/callback design (phase boundaries + per-token count, wrapped around the existing `decoder_step` closure in `pipeline.rs::synthesize` — no changes needed to `t3.rs` itself).

---

### VAI-013 · P3 · OPEN · CI/Release
**GitHub Actions cross-platform release-build matrix (Windows/macOS/Linux)**

**Acceptance criteria**:
- [ ] Matrix workflow building `vocalai-cli` on `macos-latest` (`--features coreml`), `windows-latest`/`ubuntu-latest` (`--features cuda`), mirroring `Makefile`'s `HW_FEATURE` auto-detection (VAI-011)
- [ ] Confirm build-only jobs succeed without GPU hardware present on the runner (verified: `ort-sys` downloads prebuilt ONNX Runtime binaries per target-triple+feature combo, no local CUDA toolkit/GPU needed to compile)
- [ ] Release artifacts uploaded per OS
- [ ] Explicitly out of scope: running/verifying real GPU-path inference in CI (would need a GPU-enabled runner, self-hosted or paid)

**Notes**: Raised by the user while reviewing VAI-011; deferred to its own ticket rather than bundled in.

**Update (2026-08-21)**: `VAI-007`'s `.github/workflows/release.yml` already delivers this
ticket's acceptance criteria (matrix build on macos-latest/windows-latest/ubuntu-latest,
build-only verified without GPU hardware, artifacts uploaded per OS, GPU inference explicitly
out of scope). Likely superseded — not closed here pending the repo owner's confirmation once
VAI-007's workflows have actually run.

---

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

### VAI-016 · P3 · OPEN · CI/Release
**Version-bump-driven triggers for the model-publish + release pipelines (replace manual dispatch)**

**Acceptance criteria**:
- [ ] Promote the duplicated `version = "0.1.0"` in `vocalai-cli`/`vocalai-core`'s `Cargo.toml` to
      a single `[workspace.package] version`, inherited via `version.workspace = true` — one
      source of truth for the CLI/package version.
- [ ] A new root-level `MODELS_VERSION` file (plain text) as the model artifacts' version — no
      existing home for this since `models/` is git-ignored.
- [ ] A workflow step/job that, on push to `main`, compares each version against its last
      matching git tag (`v*` for the Cargo version, `models-v*` for `MODELS_VERSION`) and only
      proceeds if it actually changed — avoids firing on unrelated edits to either file.
- [ ] On a genuine model-version bump: auto-create/push a `models-vN` tag and trigger
      `models-export.yml` (no more manual `workflow_dispatch`); tag the published HF Hub revision
      to match.
- [ ] On a genuine package-version bump: auto-create/push a `vX.Y.Z` tag, which triggers the
      existing tag-triggered path in `release.yml` unchanged.
- [ ] Release notes standardized via GitHub's built-in `generate_release_notes: true` (already
      supported by `softprops/action-gh-release`) instead of hand-rolled commit-range diffing.
- [ ] `docs/dev-setup.md` §10 updated to describe "bump the version file, push" instead of manual
      `gh workflow run`/tag commands; `docs/manual-testing.md` updated to match.

**Notes**: Raised by the repo owner while reviewing `VAI-007`; deliberately split out rather than
bundled in, so `VAI-007`'s already-tested manual-trigger pipeline could ship and get a first real
run before adding an automatic-triggering layer on top. See the `VAI-007` session (2026-08-21) for
the design discussion this ticket summarizes.

---

*Add new tickets below this line. Use the same format: heading with ID · priority · status · brief category; then bold one-line title; then acceptance criteria as checkboxes; then notes.*

---

## Recently closed

| Date | Ticket | Title | Commit |
|------|--------|-------|--------|
| 2026-08-19 | VAI-011 | `--use-gpu`/`--use-cpu` execution-provider selection (CPU by default, logged); `error_on_failure()` instead of silent fallback for forced/probed hardware attempts; `Makefile` OS-based feature auto-detection (`coreml` on macOS, `cuda` elsewhere); benchmarked and fixed a 30-40% CoreML slowdown vs CPU (`CPUAndGPU`+`FastPrediction`+`RequireStaticInputShapes`); S3Gen flow estimator pinned to CPU on CoreML (see `VAI-014`) | _pending_ |
| 2026-08-18 | VAI-006 | Wire full pipeline in `vocalai-core` + clap CLI in `vocalai-cli`, including part B.2 `--voice` zero-shot cloning (`voice_encoder.rs`, `s3tokenizer.rs`, `campplus.rs`, `mel.rs`'s hand-rolled mel/Kaldi-fbank DSP) | _pending_ |
| 2026-08-18 | VAI-009 | Fix two trace-baking bugs in `export_hifigan.py` (envelope `.repeat(int)`, noise-buffer size) so `speech_feat` is a genuine dynamic ONNX axis; extend `check_hifigan` to 3 frame counts — unblocks VAI-006's end-to-end acceptance criterion | `9ff4b4e` |
| 2026-08-18 | VAI-008 | Export S3Gen's flow-encoder (bucketed) + CAMPPlus (fixed window) to ONNX, closing the `mu`/`spks` export gap found while starting VAI-006 | `65b1642` |
| 2026-08-18 | VAI-005 | Export PerthNet encoder; implement STFT/ISTFT/resample watermarking pipeline in `watermark.rs` | `91c92ea` |
| 2026-08-17 | VAI-004 | Export T3 as decoder-with-past (hand-rolled Llama, ADR-0005); implement KV-cache decode loop + sampling (greedy-decode parity, see notes) | `2e13c33` |
| 2026-08-16 | VAI-003 | Export S3Gen flow estimator + Euler ODE loop, chain into HiFiGAN | `1bc9095` |
| 2026-08-16 | VAI-002 | Export HiFiGAN/voice-encoder/S3-tokenizer to ONNX + `parity_check.py` | `820ff9a` |
| 2026-08-16 | VAI-001 | Scaffold the real Cargo workspace + export toolchain | `5b4815c` |
| 2026-08-16 | — | ai-sdlc-bootstrap scaffold | `ba0f453` |

*When a ticket is closed: move it to this table, set the commit hash, and remove it from the Open section. Keep the last ~20 closures here; archive older ones to `docs/CHANGELOG.md`.*
