# vocal-ai — Changelog

> Chronological log of what changed in this repo and *why*. The "why" matters more than the "what" — the diff already shows the what.
>
> Update at the end of every session. Newest entries at the top.

---

## 2026-08-22 — VAI-013: closed as superseded by VAI-007's release matrix

**What changed**: no code — `docs/issues.md`/`docs/agents/STATUS.md` updated to close VAI-013
(`REJECTED`, superseded), moved to Recently closed.

**Why**: VAI-013 asked for a GitHub Actions matrix building `vocalai-cli` on
macos-latest/windows-latest/ubuntu-latest without requiring GPU hardware on the runner, with
per-OS artifacts uploaded and GPU inference explicitly out of scope. `VAI-007`'s
`.github/workflows/release.yml` already does exactly this, and — as of VAI-016's closure this
same session — has now actually run live via a real push to `main` (runs `32571043242`/
`32571043262`), which is the confirmation this ticket's "likely superseded, pending confirmation"
note was waiting on.

**What was rejected**: keeping VAI-013 open as a separate ticket once its acceptance criteria are
already met by shipped, verified work — that would just be duplicate tracking.

**Note**: VAI-013's literal wording asked for `windows-latest`/`ubuntu-latest` to build with
`--features cuda`; `release.yml` builds those two CPU-only. That gap is intentional, not a hole in
this closure — CUDA-bundled artifacts are `VAI-015`'s explicit scope, split out for the same
GPU-hardware/licensing reasons documented there.

---

## 2026-08-22 — VAI-016: version-bump-driven triggers for model-publish + release pipelines

**What changed**: consolidated `Cargo.toml`'s duplicated `version = "0.1.0"` into a single
`[workspace.package] version = "0.1.2"` (matching the already-released `v0.1.2` tag), with
`vocalai-cli`/`vocalai-core` switching to `version.workspace = true`. Added a root
`MODELS_VERSION` file (seeded `0.1.0`). `models-export.yml` now also triggers on a push to
`main` that touches `MODELS_VERSION`; `release.yml` now also triggers on a push to `main` that
touches `Cargo.toml`. Each gained a leading guard step that skips the rest of the job if a tag
for that exact version already exists (`models-vN` / `vX.Y.Z`), so a no-op edit or re-run is
safe. On a genuine bump, `models-export.yml` publishes then pushes a matching `models-vN` tag
and tags the HF Hub revision to match (`publish_models.py`'s new `--hf-tag`); `release.yml`
passes `tag_name: vX.Y.Z` straight to `softprops/action-gh-release`, which creates the tag as
part of publishing — no separate tag-push step needed there. Also: `generate_release_notes:
true` replaces having no release-body content at all; `publish_models.py` now also copies
`MODELS_VERSION` into the published HF repo folder so the version is visible on the Hub itself,
not just in git. Manual fallbacks (`workflow_dispatch`, direct `git tag vX.Y.Z && git push`)
are unchanged and still work.

**Why**: manual publishing (`gh workflow run models-export.yml`) and releasing (`git tag &&
git push`) required remembering the right command every time. The natural implementation —
a CI job auto-creating and pushing the triggering tag — runs into a real GitHub Actions
constraint: a git push made with the default `GITHUB_TOKEN` does not fire other workflows'
`on: push: tags` triggers (anti-recursion safeguard). The repo owner explicitly ruled out the
standard workarounds (a PAT secret, a GitHub App token, or an explicit `gh workflow run`
dispatch) as more ongoing management overhead than the ticket warranted. See
`docs/decisions/0014-version-bump-driven-release-triggers.md` for the full reasoning.

**What was rejected**: PAT-based tag pushing (closest to VAI-016's literal wording, but needs a
human to mint/rotate a token); a GitHub App token (disproportionate setup for a P3 ticket);
`gh workflow run` dispatch (would still need a new `release.yml` input to fake the
tag-triggered path); lexicographic "compare against the last matching tag" (an exact
tag-existence check is simpler and equally effective). Full design in ADR-0014.

**Verified this session**: `cargo metadata` confirms both crates resolve `version = "0.1.2"`
from the workspace; both edited workflow YAML files parse cleanly; `make check` passes (77 Rust
tests, 12 export parity tests, 16 script tests including 2 new ones for the `MODELS_VERSION`
publish guard).

**Verified live (2026-08-22, closing VAI-016)**: two real pushes to `main` exercised the new
triggers end-to-end. `d4c64ad` (this change) landed first — no version actually changed yet, so
both guards correctly no-op'd. `e9f4825` then bumped `Cargo.toml` to `0.1.3` and
`MODELS_VERSION` to `0.1.1` in one commit: `release.yml`'s `push`+`paths` trigger fired and
produced a real `v0.1.3` tag + GitHub release (3-platform matrix, all green, run `32571043242`);
`models-export.yml`'s trigger fired and produced a real `models-v0.1.1` tag plus a fresh HF Hub
publish (full parity gate including T3, structural smoke test, tag step all green, run
`32571043262`). Both fired via `push`, not `workflow_dispatch` — confirming the anti-recursion
workaround (direct `paths`-filtered push trigger instead of a tag-push chain) works as designed.
VAI-016 is closed on this evidence.

**What's next**: VAI-007's remaining Windows/Linux manual-validation checklist is still
separately open; nothing further needed for VAI-016 itself.

---

## 2026-08-21 — VAI-007: install script hardening (retries + progress visibility) + real e2e verification

**What changed**: `scripts/install.sh`/`install.ps1` now retry transient download failures
(`curl --retry 3 --retry-delay 2 --connect-timeout 10` / a PowerShell retry loop around
`Invoke-WebRequest`) instead of failing on the first blip, and print visible progress: a
`--progress-bar` per file in bash (previously fully silent, `-s`) plus an `[i/N] filename`
counter in both scripts, so a slow download (HF Hub's anonymous tier can be rate-limited) is
visibly progressing rather than looking hung.

**Why**: the repo owner hit a `curl: (56) ... 504` on the release-binary download; re-testing the
same URL from this session succeeded immediately, pointing to a transient network/proxy blip
rather than a real server-side issue -- but the script had no retry, so any transient failure
killed the whole install. Separately, the repo owner reported the install "seems very slow" with
no visibility into whether it was progressing or stuck, since every curl call used `-s` (silent).

**What was rejected**: nothing structural -- straightforward hardening, no design change.

**Verified for real this session**: ran `scripts/install.sh` end-to-end from a completely fresh
directory against the live `v0.1.2` GitHub release + the public `shmmsra/vocal-ai-models` HF
repo -- downloaded the binary, all 26 model files (`~4.2GB` total, confirmed via `du -sh`),
correctly laid out as `./vocalai/vocalai` + `./vocalai/models/`, then ran
`./vocalai/vocalai --text "hello world" --out out.wav --models-dir ./vocalai/models` for real,
producing a genuine mono 24kHz WAV, 0.88s duration -- matching `docs/dev-setup.md` §9.3's
expected output for that phrase exactly. Full install-to-audio pipeline confirmed working.

**What's next**: none outstanding for the installer itself; VAI-007's remaining open item is the
manual per-platform validation checklist in `docs/manual-testing.md` (CPU-fallback equivalence,
memory/swap benchmark) on Windows/Linux, not yet run.

---

## 2026-08-21 — VAI-007 correction: GitHub's 2GiB/asset cap rules out bundling models into releases

**What changed**: `release.yml` no longer archives `models/` into the uploaded release asset —
the asset is now binary + `THIRD_PARTY_LICENSES` + `LICENSE` only (tens of MB). The workflow still
downloads the pinned HF Hub revision inside the job to structurally validate it, just doesn't
package it into what gets uploaded. Added `scripts/install.sh`/`install.ps1`: a one-line installer
(`curl -fsSL .../install.sh | bash` / `irm .../install.ps1 | iex`) that downloads the release
binary from GitHub *and* every model file from the public HF repo (anonymously, no token needed)
into `./vocalai/`, ready to run. Added a "Install" section to `README.md` with these commands,
and renamed the existing contributor-facing section to "Contributor setup" to disambiguate.

**Why**: the first real tag-triggered `release.yml` run built successfully but failed uploading
the release asset on all 3 platforms: GitHub Releases caps assets at 2GiB/file, and the model set
is ~4GB (a single file, `t3_decoder.onnx`, is already ~1.9GiB on its own) — no realistic
compression gets the full bundle under 2GiB. Neither this session nor the original Phase 1 plan
accounted for this hard platform limit when deciding to bundle everything into one artifact.

**What was rejected**: publishing the full bundle to a HF Hub repo instead of a GitHub Release
asset (HF Hub has no comparable cap) — more moving parts, splits "the official binary" away from
GitHub's Releases UI; revisit if the installer-script approach proves fragile in practice. Also
rejected: adding a download-on-first-run path inside `vocalai-core`/`vocalai-cli` itself — the
installer script is a one-time *external* step; the CLI stays unchanged and runs fully offline
once installed, preserving the original "no bundled Python, no in-tool multi-GB download" goal.

**What's next**: re-tag and re-run `release.yml` with this fix; verify `scripts/install.sh`
actually works end-to-end once a real binary-only release exists to download from. Full story in
`docs/decisions/0013-*.md`'s 2026-08-21 addendum.

---

## 2026-08-21 — VAI-007 fix: release.yml's GITHUB_TOKEN lacked permission to create a release

**What changed**: added an explicit `permissions: contents: write` block to
`.github/workflows/release.yml`. Also simplified the artifact-upload/release-asset `files:`/
`path:` args from two explicit `.zip`/`.tar.gz` lines to a single `dist/<artifact>.*` glob (each
OS only ever produces one of the two; the unconditional second line was logging a harmless but
noisy "Pattern ... does not match any files" warning), and changed `upload-artifact`'s
`if-no-files-found` from `ignore` to `error` now that the glob makes "nothing matched" a real bug
worth failing loud on.

**Why**: the first real tag-triggered run of `release.yml` failed all 3 platform jobs at the
"Upload release asset" step with a 403 ("Resource not accessible by integration") from
`softprops/action-gh-release`. The auto-provided `GITHUB_TOKEN` only gets the permissions a
workflow explicitly declares (or a repo-wide default, which is read-only on many repos) --
`contents: write` was never declared, so release creation failed.

**What was rejected**: changing the repo's Settings -> Actions -> General -> Workflow
permissions default instead -- rejected in favor of declaring the permission explicitly in the
workflow file itself: least-privilege, portable across repos/orgs, and doesn't depend on a
manual UI setting a future contributor could miss.

**What's next**: re-push the `v0.1.0` tag (delete + re-push, or bump to `v0.1.1`) to retry.

---

## 2026-08-21 — VAI-007 fix: THIRD_PARTY_LICENSES was missing from the published HF Hub repo

**What changed**: `scripts/publish_models.py` now also uploads `THIRD_PARTY_LICENSES` into the
public HF Hub model repo (same temp-write-then-cleanup pattern already used for the generated
`README.md` model card). Added a regression test
(`test_publish_rejects_missing_third_party_licenses`) and a guard clause that fails fast if the
source file is missing.

**Why**: the repo owner noticed, after the first real `models-export.yml` run, that
`https://huggingface.co/shmmsra/vocal-ai-models` had no license file, only a README whose
`license: mit` front matter is metadata, not the actual notice text MIT requires. The original
design (ADR-0013) only copied `THIRD_PARTY_LICENSES` into the CLI release bundle, missing that
the HF Hub repo is itself a separate redistribution of the same weights and needs the same
notice.

**What was rejected**: nothing structural — this is a straightforward gap-fill within the
already-approved ADR-0013 design, not a new decision; ADR-0013 §8 amended in place with an
addendum rather than superseded.

**What's next**: re-run `models-export.yml` (or `make publish-models` locally with `HF_TOKEN`
set — HF Hub's content-addressed storage means this won't re-upload the unchanged multi-GB
model files, just add the one new file) to backfill the already-published repo.

---

## 2026-08-21 — VAI-007 (part 1): CI-driven model publish + release-build pipeline

**What changed**: Added two manual-trigger-only GitHub Actions workflows —
`.github/workflows/models-export.yml` (runs `make export`, gates on the *full*
`make test-py-parity` including T3, structurally validates the result with no inference
(`scripts/smoke_test_artifact.py`), then publishes to the public HuggingFace Hub repo
`shmmsra/vocal-ai-models`) and `.github/workflows/release.yml` (matrix build of `vocalai-cli` for
macOS/CoreML + Windows/Linux CPU-only, downloads the current HF Hub revision, stages a bundle
with the binary + `models/` + `THIRD_PARTY_LICENSES` + `LICENSE`, structurally smoke-tests it,
uploads as a release asset on a `v*` tag). Added `scripts/publish_models.py`,
`scripts/smoke_test_artifact.py`, and 14 new pytest tests for both (`make test-scripts`, wired
into `make check` and `ci.yml`'s fast gate). Added `THIRD_PARTY_LICENSES` (verbatim MIT notices
for `resemble-perth` + `ResembleAI/chatterbox`, fulfilling ADR-0008's standing commitment). Added
`make smoke-test`/`make publish-models` for local debugging. Also added `license = "MIT"` to
`vocalai-cli`/`vocalai-core`'s `Cargo.toml` (cosmetic — `cargo metadata` previously reported them
as unlicensed even though the repo-level `LICENSE` already covered them; found while auditing the
repo for anything that would block making it public — no legal blockers found: no secrets/keys
ever committed, no accidentally-committed weights/audio, and every dependency in both the Rust
crate tree and the Python `export/` toolchain is permissively licensed). See ADR-0013 for the
full design.

**Why**: The repo owner is making the GitHub repo public and wants generated `.onnx`/`.npy`
artifacts published to HF Hub, with export/publish running in CI rather than locally, and with
an explicit constraint that automated checks must never execute real model inference (CPU or
GPU). Along the way, this session's — and the repo owner's — initial assumptions about
GitHub-hosted runner specs turned out to be wrong (verified against GitHub's current docs):
`macos-latest` is 7GB RAM, not 14GB (that figure is the legacy `-intel` label); going public
gives `ubuntu-latest` 16GB (double the 8GB private cap), which is what actually makes running
T3's ~9GB-peak export in CI viable — and is why `models-export.yml` runs on `ubuntu-latest`, not
macOS.

**What was rejected**: a download-on-first-run CLI (bundle-at-build was chosen, keeping the
original offline-first design goal); running export/publish on `macos-latest` for its RAM (the
real spec is worse than public `ubuntu-latest`, once checked); auto-triggering
`models-export.yml` on `export/**` pushes (repo owner wants manual-only — publishing to a public
model repo should be deliberate); a git-tracked `MODEL_REVISION` pin file auto-committed by CI
(HF Hub is queried directly at build time instead, avoiding a bot-commits-to-main pattern);
building the full Windows/Linux CUDA/cuDNN-bundled GPU artifacts now (split out to a new ticket,
`VAI-015` — needs real GPU hardware and an unresolved NVIDIA redistribution-license question);
retiring ADR-0007's `parity.yml` T3 exclusion (left as a flagged, optional follow-up per the repo
owner's call, not done in this pass).

**What's next**: this is packaging/tooling only — nothing has actually run yet. The repo is now
public and `HF_TOKEN` is set; still need to trigger a real `models-export.yml` publish and cut a
first real `v*` tag, then run the manual per-platform validation in `docs/manual-testing.md`
(real audio, CPU-fallback equivalence, memory/swap benchmark) before Milestone 7 can be called
done. `VAI-013` is likely superseded by `release.yml`'s build matrix, pending confirmation.
`VAI-016` (new, split out of this session) tracks replacing the manual `workflow_dispatch`/
tag-push triggers with version-bump-driven ones (a `MODELS_VERSION` file + a `Cargo.toml`
workspace version, plus standardized GitHub-native release notes) — deliberately deferred until
the manual-trigger pipeline gets a first real run.

---

## 2026-08-20 — Correction: VAI-011's "CoreML tuning reaches CPU parity" claim was overstated

**What changed**: no code changes. Corrected `docs/decisions/0012-*.md`, `docs/CHANGELOG.md`
(2026-08-19 entry, below), `docs/agents/STATUS.md`, and `docs/manual-testing.md`, which had all
claimed the tuned CoreML config (`CPUAndGPU`+`FastPrediction`+`RequireStaticInputShapes`) restored
GPU to "CPU parity or better." That claim doesn't hold up: the small-scale benchmark it was based on
was within RNG-driven run-to-run noise (each `vocalai` invocation gets a fresh RNG, so the number of
tokens generated before EOS varies run to run even at a fixed `--max-new-tokens` cap), and a
full-scale (default 1000-max-tokens) run that was already on hand at the time showed the tuned
config **~14% slower than CPU**, not faster — a contradiction that should have been surfaced instead
of dismissed as noise. The repo owner's own real-world re-test after the fix landed also found no
improvement, which is what prompted this correction.

**Why**: accuracy — an agent asserting a performance win that the agent's own data already
contradicted, without flagging the contradiction, is worse than not measuring at all.

**What's actually true, kept from VAI-011**: the tuning is a real, worthwhile fix for the *severe*
regression (naive CoreML config was 30-40% slower than CPU) — it brings GPU to roughly
tied-or-somewhat-worse, not clearly worse across the board. `--use-gpu` stays available with the
tuned config (no reason to remove an opt-in path just because it isn't a proven win), CPU stays the
default, and the `.error_on_failure()`/`s3gen_estimator` CPU-pin fixes from VAI-011 are unaffected by
this correction.

**What's next**: if GPU speed is worth pursuing further, it needs a properly controlled benchmark
(deterministic/fixed token count, many repeated trials) before any claim is made — not attempted in
this session.

---

## 2026-08-19 — VAI-011: `--use-gpu`/`--use-cpu` execution-provider selection (CPU by default)

**What changed**: `vocalai` now takes `--use-gpu`/`--use-cpu` (mutually exclusive); neither flag
defaults to CPU. `--use-gpu` requires a hardware EP (CoreML/CUDA, gated by Cargo features) and
errors out (`SessionError::GpuUnavailable`) instead of silently falling back to CPU if none is
usable. The resolved provider is printed (`Using GPU execution provider (CoreML)` / `Using CPU
execution provider`) and decided once per `ModelBundle`, not once per session
(`session::resolve_and_build_session`/`build_session`, `pipeline.rs::ModelBundle`). Registration
uses `ort`'s `.error_on_failure()` (opposite of the previous `.fail_silently()`) so a failed
hardware-EP registration is a catchable `ort::Error`, not a silent continuation. `Makefile`'s
`build:` target auto-detects the right feature per OS (`coreml` on macOS, `cuda` elsewhere) so
`make build` compiles in hardware-EP support regardless of default — confirmed this never requires
local GPU hardware/CUDA toolchain to *compile*, only to actually *use* the resulting EP at runtime
(`ort-sys` downloads a prebuilt ONNX Runtime binary per target+feature).

Manual testing on real Apple Silicon hardware surfaced two things the original plan hadn't
anticipated:

1. `s3gen_estimator.onnx` (S3Gen's flow-matching Euler ODE estimator) reliably crashes ONNX
   Runtime's CoreML EP mid-inference. Bisected by manually forcing just that one session to CPU
   while leaving everything else (including T3's full up-to-1000-step KV-cache decode loop) on
   CoreML — confirmed that isolates the fix. Root cause: the estimator is the only session in the
   pipeline whose sequence length is genuinely dynamic (`2 * token_len`) rather than bucketed like
   the flow-encoder (ADR-0009). `ModelBundle::load` now pins this one session to CPU whenever
   CoreML is resolved (CUDA untested, left alone; real fix tracked as VAI-014).
2. Once that crash was fixed, CoreML's **default configuration measured 30-40% slower than CPU**
   wall-clock, and subjectively made the whole machine feel less responsive during a run — directly
   contradicting the original assumption that "T3's decode loop is where most of the GPU speedup
   is." Root cause: T3's decode loop calls `session.run()` roughly once per generated token (up to
   ~1000 times); CoreML's default config pays a fixed per-call dispatch/specialization cost that
   dominates across many tiny sequential calls (compounded by the decoder graph only being
   partially covered by CoreML, so every call also pays a CPU↔CoreML marshaling cost). Benchmarked
   three `ort` CoreML config knobs individually and combined (`ComputeUnits::CPUAndGPU`,
   `SpecializationStrategy::FastPrediction`, `RequireStaticInputShapes`, `ModelFormat::MLProgram`);
   the combination of the first three fixed the severe regression, while `MLProgram` failed outright
   for this graph (a clean error, thanks to `.error_on_failure()`). **See the 2026-08-20 correction
   below** — the initial "parity or better" reading of this benchmark did not hold up. `--use-gpu`
   (with the tuned config) stays opt-in and CPU stays the default regardless. Full benchmark table
   in `docs/decisions/0012-*.md`.

The CPU-retry-on-mid-inference-failure safety net an earlier iteration of this session added for
`Auto` mode was removed once `Auto` stopped being CLI-reachable (dead code keyed on a state that
could no longer occur) — `session::ExecutionProviderPreference::Auto` remains valid, tested library
API, just not wired to any current CLI flag.

**Why**: the user wanted a hardware EP failure to be loud rather than a silent, potentially
much-slower CPU fallback, and also asked "why is GPU mode slower, is that expected, and can we fix
the root cause?" rather than settling for "GPU is just worse here" — leading to the CoreML
config-tuning investigation above.

**What was rejected**: re-exporting `s3gen_estimator.onnx` with a bucketed time dimension (the
"real" fix for the crash, tracked as VAI-014) — separate export-pipeline work, not a CLI-flag
change. `ModelFormat::MLProgram` — fails to build an execution plan for this graph outright. Keeping
`Auto`/GPU as the CLI default — the tuned config only reaches parity, not a clear win, which isn't
enough evidence to default away from CPU. A GitHub Actions cross-platform release-build matrix
(VAI-013) and the `--show-progress` console progress indicator (VAI-012) were both raised during
this session and deliberately split into their own tickets rather than bundled in.

**What's next**: VAI-014 (bucket the estimator, remove the CPU pin), VAI-012 (progress indicator),
VAI-013 (GH Actions release matrix).

---

## 2026-08-18 — `make export` wrapper script

**What changed**: Added `scripts/export-all.sh` / `scripts/export-all.ps1` and a `make export`
target (dispatches to whichever script matches `$(OS)`) that runs the eight `export/` scripts
`docs/dev-setup.md` §11.1 already documented, in the required order, stopping at the first
failure. `ARGS=--with-voice-cloning` appends `export_ve.py`/`export_s3tokenizer.py` for users who
want `--voice` zero-shot cloning too. `docs/dev-setup.md` §11.1 now leads with `make export` and
keeps the old manual per-script commands as a documented fallback for re-running one script in
isolation.

**Why**: A user who just wants to run `vocalai` locally previously had to copy 8-10 commands by
hand out of the docs, in the right order, picking the right one of two platform-specific blocks.
This came up while discussing whether pre-built ONNX/npy artifacts could be published to an
artifactory for others to use (they technically could be — no machine-identifying content in
either format — but that's gated on this repo's own "ask before bundling/redistributing model
weights" rule and a not-yet-written `THIRD_PARTY_LICENSES` file per ADR-0008); the lower-effort
near-term answer is to make self-generation trivially easy instead of publishing anything.

**What was rejected**: Folding the two voice-cloning-only exports into the default run — kept them
opt-in behind `--with-voice-cloning` since most users only need default-voice synthesis and those
two exports add real time/bandwidth (another HuggingFace-cached model each). A CI job to run this
— out of scope, this is a local dev-convenience script, not a packaging pipeline (that's
Milestone 7 territory).

**What's next**: Milestone 7 (VAI-007) packaging; a real Windows run of `scripts/export-all.ps1`
to confirm parity with the POSIX script (read-reviewed but not executed as of this entry).

---

## 2026-08-18 — VAI-006, part B.2: `--voice` zero-shot cloning

**What changed**: wired `--voice` zero-shot voice cloning into the pipeline, closing VAI-006's last
open item. All three ONNX networks this needed (voice encoder, S3 tokenizer, CAMPPlus) were already
exported and parity-checked in earlier milestones (Milestone 2, VAI-008) — this was purely
host-side DSP/wiring work, no new export step. Full reasoning in
[ADR-0011](decisions/0011-voice-cloning-dsp-front-ends-hand-rolled-not-parity-checked.md).

- **`crates/vocalai-core/src/mel.rs`** (new) — four mel/fbank "flavors" ported from four different
  Python modules: the voice encoder's unscaled power-mel (`ve_mel_spectrogram`), the S3-tokenizer's
  Whisper-style log-mel (`whisper_log_mel`, shared by T3's cond-prompt tokens and S3Gen's prompt
  token), S3Gen's natural-log 24kHz mel (`s3gen_log_mel`), and CAMPPlus's Kaldi-style log-fbank
  (`kaldi_fbank`, Povey window + preemphasis + snip-edges framing). Plus the shared
  `slaney_mel_filterbank`/`kaldi_mel_filterbank` filterbank builders.
- **`crates/vocalai-core/src/voice_encoder.rs`** (new) — `librosa.effects.trim`'s frame-RMS silence
  trim, the overlapping partial-utterance windowing (`get_frame_step`/`get_num_wins`/
  `stride_as_partials`), and `ve.onnx` wiring to produce T3's speaker embedding.
- **`crates/vocalai-core/src/s3tokenizer.rs`** (new) — wraps `s3tokenizer.onnx`; discovered its
  `code` output is traced as int32 (not int64, despite `mel_len`'s int64 input dtype) via a runtime
  extraction error, confirmed against the ONNX graph's own declared output types.
- **`crates/vocalai-core/src/campplus.rs`** (new) — Kaldi-fbank extraction, per-utterance mean
  subtraction, and the trim-or-cyclically-repeat-to-400-frames windowing ADR-0009 already mandated
  (never zero-pad — would corrupt CAMPPlus's internal statistics pooling).
- **`crates/vocalai-core/src/audio.rs`** — added `read_wav` (mono `f32`, arbitrary bit depth,
  multi-channel downmix by averaging) and a `Result`-returning, arbitrary-sample-rate-pair
  `resample` (kept separate from `watermark.rs`'s fixed-ratio/`.expect()`-based twin — different
  error contract, accepted small duplication).
- **`crates/vocalai-core/src/pipeline.rs`** — `DefaultVoice` renamed `VoiceConditioning`, gains
  `from_reference(bundle, wav_path)` alongside the existing `load_default(dir)`; both produce the
  same six-tensor shape so `synthesize`'s T3/S3Gen wiring is unchanged downstream of voice
  selection. `ModelBundle` lazily loads `ve`/`s3tokenizer`/`campplus` sessions on first `--voice`
  use, so the default-voice-only path pays no extra load cost. `PipelineError` gains `Audio`/
  `Resample` variants; the old `VoiceCloningNotImplemented` variant is gone.
- **`crates/vocalai-core/src/s3gen.rs`**/**`t3.rs`** — added `S3GEN_SR`/`SPEECH_COND_PROMPT_LEN`
  constants (previously only implicit in the default-voice `.npy` dump) for the new preprocessing
  code to reference.
- **`crates/vocalai-cli/src/main.rs`** — doc-comment only; `--voice` was already a wired-up flag.

**What was rejected** (see ADR-0011 for full reasoning): an automated `parity_check.py`-style gate
for the new DSP (disproportionate machinery for classical signal processing with no ONNX graph to
diff against — same treatment already given to `watermark.rs`'s STFT/ISTFT/resample and CAMPPlus's
fbank input); sharing `hann_window`/`reflect_pad` between `watermark.rs` and `mel.rs` (small,
independently-documented duplication preferred over cross-module coupling); a single generic
mel-spectrogram function covering all four flavors (they differ in too many dimensions for a shared
signature to be simpler than four small functions).

**A real bug shipped in the first pass and was caught by the repo owner's manual test, not by CI or
unit tests**: `synthesize`'s token-assembly step (`pipeline.rs`) read the S3Gen prompt-token
*values* from `bundle.default_voice.s3gen_prompt_token` unconditionally — a leftover from the
`DefaultVoice` → `VoiceConditioning` rename's mechanical find-replace, which only matched
single-line `bundle.default_voice.<field>` occurrences and missed this one because it spanned
multiple lines. `prompt_token_len` (used for bucket/slice-length arithmetic) was correctly read
from the *cloned* voice, but the token values themselves silently came from the *default* voice —
a length/data mismatch that only surfaced as a panic
(`prompt_feat's 500 frames exceed total_mel_len=416` in `s3gen::build_cond`) when the two
diverged enough. My own pre-commit end-to-end check used a reference clip
(`tmp/out.wav`, itself default-voice output) short enough that the mismatch happened not to trip
the assert, so it passed without exposing the bug — the repo owner's test, using a different
(real, longer) reference clip, did trip it. Fixed by reading `voice.s3gen_prompt_token` instead of
`bundle.default_voice.s3gen_prompt_token`.

**How this was verified**: `make check` (fmt, clippy, 76 Rust unit tests including numeric
spot-checks against real librosa/torchaudio output on synthetic tones, 12 Python parity tests) is
clean. End-to-end, after the fix above: `vocalai --text "This is a voice cloning test." --voice
tmp/sample-voice-1.wav --out tmp/cloned.wav --models-dir models` produced a 52800-frame (~2.2s)
mono 24kHz WAV, peak amplitude 32767/32767, RMS ~3937 — genuine non-silent audio. The default-voice
command was re-run immediately after with no regression. Audible confirmation that the cloned
voice actually resembles the reference speaker is still the repo owner's to make (see
`docs/manual-testing.md`'s new section) — not yet done as of this entry.

**Residual risk**: the new DSP (mel.rs, voice_encoder.rs's trim/striding) has no automated
cross-language parity gate — see ADR-0011. Tracked alongside the existing watermark-resampler
(VAI-005) and CAMPPlus-fbank-input (ADR-0009) residual risks.

**What's next**: VAI-006 is fully closed. Milestone 7 (per-platform packaging) is next —
`docs/agents/STATUS.md`'s "Deferred" section already flags T3's ~9GB export-time memory as a real
resourcing question for that milestone's build pipeline.

## 2026-08-18 — Fix `make check` on Windows (MSVC CRT link mismatch + non-portable pytest recipe)

**What changed**: two independent Windows-only breakages in the `make check` gate, neither
of which reproduces on macOS/Linux (so CI stayed green and both were invisible until the gate
was run on Windows). Full reasoning in [ADR-0010](decisions/0010-windows-build-and-make-portability.md).

- **`crates/vocalai-core/Cargo.toml`** — `tokenizers` is now pulled with
  `default-features = false, features = ["onig", "progressbar"]`, dropping the default
  `esaxx_fast` feature. `esaxx_fast` enables `esaxx-rs/cpp`, whose `build.rs` hardcodes
  `.static_crt(true)` and compiles its C++ against the static CRT (`/MT`); ONNX Runtime's
  prebuilt (`ort`) and Rust std use the dynamic CRT (`/MD`), so the MSVC linker aborted with
  `LNK2038: RuntimeLibrary mismatch (MD_DynamicRelease vs MT_StaticRelease)`. `esaxx_fast` is a
  suffix-automaton used only to *train* Unigram tokenizers — never touched by inference-time
  tokenizer loading/encoding — so dropping it changes no runtime behavior. Applied on all
  platforms, not just Windows, since the C++ path is unused everywhere.
- **`Makefile`** — the `test-py*` recipes no longer use a POSIX-shell conditional
  (`if [ -x .venv/bin/python ]; then ...; fi`), which `cmd.exe` can't parse (`-x was unexpected
  at this time`) since there's no `sh` on a stock Windows PATH. The interpreter path is now
  chosen by `$(OS)` (`.venv\Scripts\python.exe` on Windows with backslashes for `cmd.exe`'s
  leading-exe rule, `.venv/bin/python` elsewhere) and existence is probed with make's own
  `$(wildcard)` instead of a shell test. Same "prefer venv else PATH pytest" behavior, now
  valid under both `cmd.exe` and POSIX `sh`.

**How this was verified**: ran the full `make check` on Windows 11 / MSVC after each fix.
`cargo fmt` + `clippy` clean, `cargo test --workspace` links and passes all 53 Rust tests
(previously failed at link), and `test-py` runs the export venv's interpreter and passes all
12 Python parity tests (previously errored before pytest even started). Gate is fully green.

**What was rejected** (see ADR-0010): forcing `+crt-static` globally (breaks `ort`'s
dynamic-CRT prebuilt), patching/forking `esaxx-rs`, and `SHELL := bash` (the box's only bash is
WSL, which can't run the Windows-native venv interpreter).

**Also in this session — docs**: added `docs/dev-setup.md` §11 ("Generate model artifacts + run the
app"), a single authoritative, cross-platform (macOS/Linux + Windows PowerShell) runbook mapping each
`export/` script to the `models/` files it produces, plus the build and synthesis commands. Previously
this was scattered across the per-milestone `docs/manual-testing.md` sections with no Windows commands,
so running the app end-to-end required re-researching the export order every time. The
manual-testing CLI section now points at §11 as the source of truth instead of duplicating it.

**What's next**: if `tokenizers` is version-bumped, re-check its default feature list for a
re-introduced C++/CRT-sensitive default.

## 2026-08-18 — VAI-009: fix dynamic-length HiFiGAN export, unblocking VAI-006's end-to-end acceptance criterion

**What changed**: `export/export_hifigan.py`'s `speech_feat` input is now a genuine ONNX dynamic
axis instead of hard-fixed at 50 mel frames. Two independent trace-baking bugs had to be fixed
(same category ADR-0009 already documented for the flow-encoder/CAMPPlus exports — a Python-int
read off a tensor's `.shape` gets baked as an ONNX literal, so the graph only works at the exact
length it was traced at):
- `_istft_onnx`'s overlap-add envelope built its window via `.repeat(1, 1, num_frames)`, baking the
  traced frame count into the envelope regardless of the real (dynamic) input length. Fixed by
  broadcasting against `frames_ct`'s own dynamic time dim (`window_sq_col * torch.ones_like(...)`)
  instead of `.repeat(int)`.
- `_sine_gen_deterministic`'s source noise drew `torch.tensor(rng.randn(*sine_waves.shape))` —
  `.shape` resolves to plain Python ints at trace time, and the exporter registers the resulting
  array as a literal ONNX constant (confirmed via the exporter's own `TracerWarning`), baking the
  traced sample count. Fixed by precomputing one fixed-size deterministic noise buffer (same
  `_SOURCE_NOISE_SEED`, registered as a module buffer in `HiFiGANExportWrapper.__init__`) and
  dynamically slicing it to the real length at forward time.
`export/parity_check.py::check_hifigan` now exercises three frame counts (17/50/123), not just the
one the original fixed-shape export happened to use — the single-fixed-shape convention is exactly
what let both bugs ship unnoticed the first time (same lesson ADR-0009 already drew for the other
two exports).

**How this was found to actually work, not just "look shape-agnostic"**: prototyped in scratch
space before touching any repo file — a naive fix (add `dynamic_axes=...` alone, no code change)
exports without error but fails at ONNX Runtime with a broadcast-shape error at any frame count
other than the traced one, confirming the two bugs above are real, not theoretical. After both
fixes, verified bit-consistent (eager-vs-ONNX, within `atol=1e-4`) output across frame counts
1/17/30/50/80/123/200, then ran the real `vocalai --text "hello world" --out out.wav` CLI command
(VAI-006/VAI-009's own acceptance criterion — previously failed with a shape-mismatch error) and a
much longer sentence (~7.4s of audio), both producing genuinely non-silent, plausible-duration WAV
output — manually confirmed audible by the repo owner.

**What was rejected**: a bucketed/fixed-length export (ADR-0009's approach for the flow-encoder/
CAMPPlus) — unlike those two, HiFiGAN's dynamic-shape bugs are narrowly in this repo's own wrapper
code (not third-party relative-position-encoding/pooling math), so a real dynamic fix was tractable
and strictly better (no per-length Rust-side bucket-selection logic needed downstream).

**What's next**: Milestone 6 part B.2 (`--voice` zero-shot cloning) is the only remaining item
before VAI-006 itself can close. Also worth a closer listen to `watermark.rs`'s resampler-fidelity
residual risk (`docs/agents/STATUS.md`) now that real, arbitrary-length audio exists to check it
against.

---

## 2026-08-18 — fix: stage `tokenizer.json` reproducibly via `export/fetch_tokenizer.py`

**What changed**: Added `export/fetch_tokenizer.py`, a small script that downloads
`tokenizer.json` from `ResembleAI/chatterbox` on the HuggingFace Hub (via `hf_hub_download`, same
pattern as every other `_common.py` loader) and copies it to `models/tokenizer.json`. Found while
manually testing VAI-006 part B.1: every other artifact under `models/` (the `.onnx` exports, the
default-voice `.npy` dump) is produced by some `export/*.py` script, but nothing fetched
`tokenizer.json` — it needs no ONNX conversion (the Rust `tokenizers` crate reads the HuggingFace
file format directly), so it was missing from the export toolchain entirely. A from-scratch
`models/` build was silently missing this file; the gap was masked this session by manually
copying it out of the local `huggingface_hub` cache, which isn't reproducible from a clean clone.

**Why not folded into `export_default_voice.py` or `_common.py`**: kept as its own script/command
(`python fetch_tokenizer.py`) rather than a side effect of an unrelated export, since it isn't a
model export and has no parity check to run — matches `export_default_voice.py`'s own precedent of
being a separate, purpose-named script for a non-ONNX asset.

**What's next**: none — this was a small, mechanical toolchain gap, not deferred work.

---

## 2026-08-18 — VAI-006 part B.1: wire the default-voice pipeline + CLI (blocked on a newly-found HiFiGAN export gap)

**What changed**: Implemented the default-voice half of Milestone 6's full pipeline (part B.1 of
VAI-006; `--voice` zero-shot cloning is B.2, not started). New: `tokenizer.rs` (`tokenizers`-crate
wrapper + `punc_norm`/CFG-doubling/sot-eot padding matching `chatterbox/tts.py::generate` exactly,
verified against the live Python reference including its double-space punctuation quirk),
`audio.rs` (WAV writing only, `hound`), `pipeline.rs` (`ModelBundle` + `synthesize()` orchestrator
wiring tokenizer → T3 → S3Gen → HiFiGAN → watermark), and real `session::build_session` (registers
`execution_providers()` on a live session; `ort`'s `tracing` feature + a minimal
`tracing-subscriber` make its internal EP-registration warnings visible instead of silently
compiled out — the practical form "log a silent CPU fallback" can take given `ort`
2.0.0-rc.13 has no queryable "which EP actually ran" API). `s3gen.rs` gained the *upstream* wiring
VAI-008 exported but nothing yet drove: bucket selection/padding over `TOKEN_BUCKETS`, the
mandatory post-encoder slice to the real valid prefix (ADR-0009), `cond`/mask/noise-init tensor
assembly matching `flow.py::CausalMaskedDiffWithXvec.inference`, and the speaker-embedding
normalize+affine hand-roll. `t3.rs` gained one small filter helper
(`filter_valid_speech_tokens`, matching `tts.py`'s `speech_tokens[speech_tokens < 6561]`) — no loop
math changes. `vocalai-cli` is a real `clap` CLI now (all plan §3 flags); `--voice` exists but
returns a clear "not implemented yet" error rather than silently using the default voice. New
Cargo deps: `tokenizers`, `hound`, `rand_distr` (0.4, pinned for `rand` 0.8 compatibility — 0.6
pulls in `rand` 0.9), `clap`, `tracing`/`tracing-subscriber`.

**Blocking discovery (new, not yet fixed — see `docs/issues.md` VAI-009)**: running the real CLI
against the already-exported `models/` directory surfaced that `hifigan.onnx` (Milestone 2,
`export_hifigan.py`) has a **hard fixed input shape** — `speech_feat` is `(1, 80, 50)` with no
`dynamic_axes` at all, not just an internal `output_size` assumption as that script's own docstring
flagged ("KNOWN LIMITATION ... follow-up work before Milestone 6"). Every `check_hifigan`/
`check_s3gen` parity test happens to use exactly 50 frames, so this never surfaced until real
variable-length generated mel (this session's new code) tried to call it. Confirmed via
`onnx.load(...).graph.input` (static dims, no `dim_param`) and by running the CLI: `--text "hello
world"` (the plan's own acceptance-criterion command) fails with `Got invalid dimensions ... Got:
44 Expected: 50`. Forcing exactly 25 generated tokens (`--max-new-tokens 25` against text long
enough not to hit EOS first, so `mel_len2 = 2*25 = 50` lands exactly on the fixed shape) *does* run
the full chain successfully end-to-end — T3 decode, flow-encoder bucket call, Euler loop, HiFiGAN,
watermark, WAV write — producing a real 1.0s 24kHz waveform (peak ~26654/32767, RMS ~4023, not
silence/NaN/garbage). This confirms every piece built this session is wired correctly; the sole
blocker is HiFiGAN's fixed-50-frame export. Fixing it (a dynamic-length re-export + a parity check
that exercises more than one frame count, the same category of fix ADR-0009 already made for the
flow-encoder/CAMPPlus) is `export/export_hifigan.py` work — an ONNX-export-boundary change needing
its own plan and approval, out of this session's authorized B.1 file list, so left for a follow-up
session (VAI-009) rather than done opportunistically here.

**What was rejected**: fixing `export_hifigan.py` inline to unblock the manual test — rejected
because it's an unplanned, unapproved change to the ONNX export boundary (`CLAUDE.md` §0), and
`docs/agents/CONVENTIONS.md` §3 requires a passing parity check before any export change ships,
which a same-session drive-by fix couldn't responsibly clear.

**What's next**: VAI-009 (re-export HiFiGAN with a genuinely dynamic-length overlap-add + a parity
check exercising multiple frame counts), then VAI-006 part B.1 can be manually verified end-to-end
for real (arbitrary-length) text before B.2 (`--voice` cloning) starts.

---

## 2026-08-18 — VAI-008: Export S3Gen's flow encoder + CAMPPlus to ONNX (closes a gap found starting Milestone 6)

**What changed**: Starting Milestone 6 (VAI-006 — wire the full pipeline), a source read of
`chatterbox/models/s3gen/{s3gen,flow}.py` found that Milestone 3's "S3Gen flow estimator" export
was only the *downstream* CFM diffusion network (`x,mu,spks,cond -> dxdt`); nothing had ever
exported the *upstream* piece that produces real `mu`/`spks` from speech tokens + a reference
voice. `parity_check.py::check_s3gen` had always fed random synthetic `mu`/`spks`/`cond` into the
estimator — nothing mechanically proved the real token→mel chain worked.

Two new exports close the gap: `export/export_s3gen_flow_encoder.py` (wraps `flow`'s real
`input_embedding`/`encoder`/`encoder_proj` — the token→`mu` path) and `export/export_campplus.py`
(wraps `CAMPPlus`, S3Gen's x-vector speaker encoder — the `spks` path; also dumps
`spk_embed_affine_layer`'s weights for a hand-rolled Rust matmul, same treatment T3's embedding
table got under ADR-0005). `export/export_default_voice.py` dumps the bundled `conds.pt`'s tensor
fields to `.npy` so the built-in default voice works without a `--voice` flag (no parity check —
it's a data copy, not a model export). `export/parity_check.py` gained
`check_s3gen_flow_encoder`/`check_campplus`; `export/tests/test_parity_check.py` gained the
matching pytest wrappers (now 8 `@pytest.mark.parity` tests, 12 total Python tests).

**What was rejected / what changed mid-flight**: both new networks were *first* exported with a
single dynamic-length axis (the convention every other export in this repo uses) — and both were
found broken by a manual sanity check at a shape other than the tracing example (a check the
existing same-shape-only parity convention doesn't catch). `EspnetRelPositionalEncoding` bakes its
Python-`int` `size` argument into the flow-encoder's graph as a constant; `CAMLayer.seg_pooling`'s
pool→expand→trim pattern only round-trips correctly through ONNX when the trim is a no-op. Rather
than hand-roll a reimplementation (T3/ADR-0005's more expensive fallback), both are now
fixed-length/bucketed exports: the flow encoder ships six static buckets
(`TOKEN_BUCKETS = 200..1200`) selected by real token count and padded via `token_len`-driven
masking (parity-checked for *padding invariance*, not just same-shape match); CAMPPlus ships one
static 400-frame graph (any multiple of 200 is safe — enforced by an `assert`), and Rust must
always feed real (not zero-padded) content at that exact length. Full diagnosis, bisection
results, and the decision rationale are in ADR-0009.

**What's next**: Milestone 6's Rust wiring (VAI-006) — `audio.rs` (WAV I/O + the four DSP
front-ends: VE's 16kHz/40-mel, S3-tokenizer's 16kHz/128-mel, S3Gen's 24kHz/80-mel reference mel,
and CAMPPlus's Kaldi-style fbank), bucket-selection/padding logic for the flow encoder, the
default-voice loader, and the `clap` CLI.

---

## 2026-08-18 — VAI-005: Export PerthNet encoder, implement STFT/ISTFT/resample watermarking (`watermark.rs`)

**What changed**: `export/export_perthnet.py` exports `PerthNet.encoder` (the Conv1d
residual-encoder submodule inside the `resemble-perth` package — the sole learned piece;
`Encoder.forward` internally crops to the 128-bin `subband` below `max_wmark_freq=2000Hz`,
applies the residual, and masks it, so the exported graph already does the full mask/residual
logic, not just the raw conv stack) to `models/perthnet_encoder.onnx`
(`export/parity_check.py::check_perthnet`, tight-tolerance synthetic-magspec parity, same
pattern as `check_hifigan`/`check_ve` — satisfies `CLAUDE.md` §1's parity hard constraint for the
one actual ONNX-exported piece). `export/_common.py` gained `load_perthnet()`.

Loading PerthNet needed a real fix, not a workaround: `perth/perth_net/__init__.py` does
`from pkg_resources import resource_filename` to locate its bundled checkpoint, and
`setuptools>=81` has started dropping `pkg_resources` entirely (a real, in-progress upstream
removal, not specific to this repo) — an unpinned `pip install setuptools` in this venv resolved
to 84.0.0, which lacks it. `export/requirements.txt` now pins `setuptools<81`.

`crates/vocalai-core/src/watermark.rs` (new) reimplements
`PerthImplicitWatermarker.apply_watermark`: STFT (reflect-pad, Hann window, real FFT per frame via
`realfft`, matching `torch.stft(center=True, pad_mode="reflect", normalized=False)`) → dB-scale
log-magnitude normalize → the ONNX encoder call → denormalize → ISTFT (inverse real FFT per
frame, Hann synthesis window, COLA-normalized overlap-add — the same recipe already proven
correct in `export_hifigan.py`'s `_istft_onnx`, expressed with a real inverse FFT instead of a
precomputed DFT matrix) → 24kHz↔32kHz resample (`rubato`, FFT-based synchronous resampler; the
ratio is a simple 4/3). Structured like `s3gen.rs`/`t3.rs`: the DSP pipeline is generic over the
encoder-step call, so it's unit-tested (7 tests: Hann-window values, reflect-pad convention,
normalize/denormalize round-trip, a synthetic-signal STFT→ISTFT round trip, resample duration
preservation, an identity-encoder end-to-end round trip, encoder-error propagation) without a
live ONNX session; `run_encoder` provides the real `ort`-backed wiring.

`stft_magphase` was manually spot-checked once (not a repeatable test) against a live
`AudioProcessor.signal_to_magphase` call on a synthetic 220Hz tone: the signal-carrying bin
matched to ~1e-7 across all frames. The only disagreements larger than float32 rounding were at
near-silent bins (DC leakage, near-Nyquist), where different FFT implementations' summation order
disagrees by orders of magnitude in *relative* terms after log compression while both sides are
inaudibly close to zero in *absolute* terms — expected floating-point behavior, not a
framing/windowing bug.

**Why**: Milestone 5 (plan §7) — VAI-005. The licensing question that blocked this was already
resolved by ADR-0008 (both `resemble-perth` and the Chatterbox weights are MIT).

**What was rejected**: bit-exact resampler parity between `rubato` and librosa's default
`soxr_hq` — classical DSP isn't ONNX-exported, so it isn't gated by `CLAUDE.md` §1's parity
constraint the way the exported networks are; chasing that now, with no live CLI to listen to the
result on yet (Milestone 6), would be effort spent on an unverifiable claim. Documented as an
accepted residual risk (`docs/agents/STATUS.md`) instead of silently assumed away. Also rejected:
hand-rolling the subband crop/mask/residual-add math in Rust — the exported `Encoder.forward`
already does this internally, so re-deriving it in Rust would just be redundant, divergent logic.

**What's next**: Milestone 6 — wire the full pipeline (tokenizer → T3 → S3Gen → HiFiGAN →
watermark → WAV) in `vocalai-core`, plus the `clap` CLI in `vocalai-cli`. This is also the first
point real end-to-end audio exists to manually verify the resampler-fidelity residual risk above.

---

## 2026-08-18 — docs: add ADR-0008 resolving PerthNet/Chatterbox license question (VAI-005)

**What changed**: Verified `resemble-perth` (package + bundled weights) and
`ResembleAI/chatterbox` (HuggingFace model card) are both MIT-licensed, closing the two open
licensing questions plan §9 flagged for VAI-005. Recorded as ADR-0008, including a new
commitment: Milestone 7's bundled release artifacts must ship a `THIRD_PARTY_LICENSES`/`NOTICE`
file with both MIT notices, to satisfy MIT's copyright-notice-retention condition. Also backfilled
the stale ADR index in `docs/decisions/README.md` (0005–0007 were missing).

**Why**: `CLAUDE.md`'s universal rule requires an ADR for decisions another agent might wonder
about; a licensing question blocking a ticket, resolved by direct verification rather than
assumption, qualifies. Unblocks VAI-005 with no licensing gate.

**What was rejected**: assuming a package's code license (MIT) automatically covers its model
weights without checking the weights' own source — weights are often licensed separately from
surrounding code in ML packages, so each was verified independently.

**What's next**: VAI-005 itself (export PerthNet, wire watermarking into the output pipeline).

---

## 2026-08-18 — ci: exclude T3's parity check from CI (still local-only), add `heavy_build` marker

**What changed**: The previous entry's `check_t3` memory fix didn't actually fix CI — a
second real run still failed (`Terminated`, cancelled), because that fix addressed the
*wrong phase*. Root-caused with real measurements (`/usr/bin/time -l`, forcing a genuinely
fresh export by clearing the locally-cached `models/*.onnx` first — the earlier "fix"
had been silently verified against a stale local cache, the exact trap the VAI-002
postmortem already flagged): `torch.onnx.export` tracing/serializing `t3_decoder.onnx`
from scratch peaks at **~9GB**, independent of `do_constant_folding`; loading an
already-built `.onnx` and running inference on it peaks at only ~2.3GB. `external_data=True`
(the parameter's stated default) is silently ignored on this torch version's legacy
(non-`dynamo`) export path — confirmed empirically (same 9GB peak, same single-file
output, whether passed explicitly or not). This repo is a private GitHub repo (confirmed:
unauthenticated API check returns 404), meaning the free-tier hosted runner — nowhere near
9GB of headroom.

Since the expensive part is *building* the ONNX graph, not verifying an already-built one,
and CI has no way to obtain a pre-built `.onnx` without either committing model artifacts
(violates the no-binary-artifacts constraint) or fetching from a persistent store (out of
scope here), T3's parity test now gets a second marker, `@pytest.mark.heavy_build`
(`export/pytest.ini`), and CI's `parity.yml` runs a new `make test-py-parity-ci`
(`pytest -m "parity and not heavy_build"`) instead of the full `test-py-parity`. T3's
parity check still runs locally (`make test-py-parity`/`make check`, unchanged) and is now
a **local-only, developer-run gate**: must be run manually before committing changes to
`export/export_t3.py` or `crates/vocalai-core/src/t3.rs`. See ADR-0007.

**Why**: The hard parity-check constraint (`CLAUDE.md` §1) can't be satisfied by CI for a
component whose build step needs more memory than the runner has — no amount of in-process
memory-lifecycle tuning changes that, since the ~9GB is the *building* cost, which any CI
run on a fresh checkout must pay. Rather than keep CI red or silently skip the hard
constraint entirely, the enforcement mechanism for this one component moves from
"automatic, every commit" to "manual, before committing T3-affecting changes" — a
deliberate, documented exception, not a silent gap.

**What was rejected**: Caching the built `.onnx` across CI runs (the *first* cache-populating
run still needs the same ~9GB peak — doesn't fix the actual failure). Switching to the
`dynamo=True` exporter or splitting the decoder into per-layer ONNX graphs (both plausible
future fixes, both substantial redesigns disproportionate to an urgent CI fix — the latter
would reopen ADR-0005's approved single-graph decision). Paying for a larger runner or
self-hosting one (a legitimate option, but a billing/infra decision for the repo owner to
make deliberately, not something to reach for while fixing a test).

**What's next**: Milestone 5 — export PerthNet, wire watermarking into output (`docs/issues.md`
`VAI-005`); mark its parity test `@pytest.mark.heavy_build` too if its export step turns out
to need similarly large memory. Separately: Milestone 7's release-build pipeline will hit
this same ~9GB ceiling for T3 (building the release artifact requires the same
`torch.onnx.export` call) — left open per the repo owner's request, revisit when that
milestone starts.

---

## 2026-08-17 — ci: split CI into fast + parity workflows, fix `check_t3` OOM

**What changed**: VAI-004's `check_t3` (previous entry) OOM-killed CI (`Killed`, exit 143):
the ~2GB PyTorch T3 model, the in-memory ~1.9GB ONNX protobuf built during export, and a
loaded ~1.9GB `onnxruntime` session on the same graph could all be resident at once. Fixed
at the source: `check_t3` now extracts the (tiny) reference greedy-decode outputs, then
explicitly frees the torch model (`del t3`, `_common.load_t3.cache_clear()`, `gc.collect()`)
*before* loading the ONNX Runtime sessions for the second half of the comparison.

Separately, split `.github/workflows/ci.yml` (previously the single "mirrors `make check`
exactly" job, per ADR-0001) into two workflows on the **same triggers as before** (every
push/PR to `main` — no schedule, no path filter): `ci.yml` now runs only the fast,
fully-offline checks (fmt/clippy/`cargo test`/`pytest -m "not parity"`), and new
`parity.yml` runs the 5 tests that download a real HuggingFace checkpoint and validate
ONNX-vs-PyTorch numerical parity (`pytest -m parity`, keeping the disk-space-reclaim step
from the earlier CI-hang fix). New `export/pytest.ini` registers the `parity` marker; new
`Makefile` targets `test-py-fast`/`test-py-parity` mirror the split. Local `make
check`/`test-py` are unchanged and still run everything. See ADR-0006.

**Why**: The parity checks are a hard project constraint (`CLAUDE.md` §1 — no exported
component ships until `parity_check.py` confirms numerical parity), so they can't be
skipped or deferred to a schedule; but coupling them to the same fast per-commit job as
lint/unit-tests means one growing multi-GB checkpoint (T3's is 2GB, the largest by far)
drags down a signal every contributor wants fast. Splitting into two jobs on identical
triggers preserves exactly the same enforcement while isolating each job's resourcing.

**What was rejected**: A scheduled (nightly/weekly) or `workflow_dispatch`-only parity
trigger — explicitly rejected by the repo owner ("everything should be cause and effect");
would also mean a broken export could land on `main` without the hard parity constraint
being checked on that commit at all. A path-filtered trigger (`export/**`/
`crates/vocalai-core/src/**` only) — reasonable, documented as a fallback in ADR-0006 if
checkpoint growth later makes every-commit parity runs impractical, but not adopted now to
keep the "runs on every commit, same as before" property simple and legible.

**What's next**: Milestone 5 — export PerthNet, wire watermarking into output
(`docs/issues.md` `VAI-005`); mark its parity test `@pytest.mark.parity` in the same
commit, per ADR-0006's new commitment.

---

## 2026-08-17 — VAI-004: Export T3 as decoder-with-past, implement KV-cache decode loop + sampling

**What changed**: Added `export/export_t3.py`, which exports T3's Llama-style backbone as two ONNX
graphs plus two raw embedding-table `.npy` files. `transformers==5.2.0`'s `LlamaModel` is built
entirely around `Cache`/`DynamicCache` objects and `masking_utils.create_causal_mask`, which don't
trace through the legacy `torch.onnx.export` tracer this repo already uses (ADR-0002) — so
`T3DecoderExport`/`_ExportDecoderLayer` hand-roll the same Llama math (RMSNorm, RoPE reusing the
model's own precomputed llama3-scaled `inv_freq` buffer, SwiGLU MLP, no GQA since
`num_key_value_heads == num_attention_heads`) directly against `T3.tfmr`'s real submodules — no
weight copying, no Cache object. See ADR-0005 for the full rationale, including why the KV-cache is
one stacked `(layers, k/v, batch, heads, seq, head_dim)` tensor rather than 60 per-layer-named
tensors. `T3CondPrefillExport` separately reproduces `T3.prepare_input_embeds()` plus
`T3.inference()`'s double-BOS-embedding construction (a real quirk in the reference — two
numerically-identical BOS embeddings get concatenated back to back before the first decoder
forward). `export/_common.py` gained `load_t3()`.

Added `crates/vocalai-core/src/t3.rs`: the sampling math (CFG combine, repetition penalty,
temperature, min-p, top-p, greedy/multinomial selection) is generic over the decoder-step call,
mirroring `s3gen::solve_euler`'s pattern (ADR-0004) — 22 new Rust unit tests cover it with synthetic
decoders/logits, no ONNX Runtime session needed. Per-step new-token embedding is a plain
`speech_emb`/`speech_pos_emb` weight-table row lookup in Rust (`embed_speech_token`,
`load_embedding_table` via the new `ndarray-npy` dependency), not an extra ONNX call per generated
token. Added `rand` for multinomial sampling.

Added `export/parity_check.py::check_t3`. `T3.inference()`'s `do_sample` parameter is accepted but
never actually read in the reference — sampling is always stochastic (`torch.multinomial`), and
PyTorch's/Rust's RNGs are unrelated, so comparing free-running sampled token sequences across
languages would be meaningless (one divergent token cascades into a different sequence). Instead,
`check_t3` runs a **greedy** (argmax) replica of the real reference forward pass
(`_greedy_reference_t3` — same `Cache`, same weights, same RoPE, just non-stochastic selection)
alongside a free-running greedy loop driving the exported ONNX graphs (`_greedy_onnx_t3`), and
compares both the resulting token sequences (must match exactly) and the per-step processed logits
(within tolerance). Passed on first real run against the downloaded checkpoint:
`max_abs_diff=4.768e-05` (well within `atol=1e-4`), 6/6 greedy tokens matching.

**Why**: Milestone 4 is the main technical risk of Phase 1 (plan §9) — T3 is the only exported
component with real autoregressive control flow. The hand-rolled decoder was necessary because
tracing HF's own `LlamaModel.forward` would mean monkeypatching private `transformers` internals
(`cache_utils`, `masking_utils`) at a specific minor version — a worse maintenance bet than
re-implementing the small, stable, public Llama math directly (ADR-0005). Greedy-decode parity was
chosen over the plan's literal "fixed seed" wording because a "fixed seed" doesn't make stochastic
sampling comparable across two unrelated RNG implementations; greedy removes randomness from the
comparison while still exercising the real end-to-end forward pass.

**What was rejected**: Tracing `T3HuggingfaceBackend.forward` directly with `Cache`/mask internals
monkeypatched (fragile against `transformers` version churn). Per-layer-named
`past_key_values.N.key/.value` ONNX I/O matching the `optimum`/HF convention (no external consumer
in this pipeline needs it; one stacked tensor is less Rust-side bookkeeping). A third ONNX graph for
the per-step new-token embedding (a trivial lookup, not worth an ONNX Runtime call per generated
token). See ADR-0005 for the full list.

**What's next**: Milestone 5 — export PerthNet, wire watermarking into output
(`docs/issues.md` `VAI-005`).

---

## 2026-08-16 — fix: S3 tokenizer export corrupted `freqs_cis` on a fresh `models/` dir

**What changed**: `export_s3tokenizer.py::build_wrapper()` mutates `encoder.freqs_cis` in place on
the `@lru_cache`d shared `s3gen` object returned by `_common.load_s3gen()` (replacing the original
complex rotary buffer with the real-valued equivalent needed for ONNX export — see the Milestone 2
CHANGELOG entry below). `check_s3tokenizer()` calls `build_wrapper()` once directly (to get the
"PyTorch reference" module), then again — unguarded — via `export()` when `models/s3tokenizer.onnx`
doesn't already exist. The second call read the *already-mutated* buffer's shape (whose last dim is
now `2`, the real/imaginary pair, not the original head dim) and computed a corrupted replacement
from it, causing a hard shape-mismatch `RuntimeError` during tracing (`"size of tensor a (64) must
match the size of tensor b (2)"`). Fixed by guarding the mutation with `torch.is_complex(...)`, so
a second call on an already-converted buffer is a no-op.

**Why**: This was a **pre-existing bug from VAI-002**, invisible in every local `make check` run
because the developer's `export/`-adjacent `models/` directory (git-ignored, dev-time only) had a
stale `s3tokenizer.onnx` cached since the day VAI-002 landed — so `check_s3tokenizer()`'s
`export()` branch was always skipped locally, and `build_wrapper()` only ever ran once per process.
CI clones fresh (no `models/` directory at all) and was the first environment to actually exercise
the `export()` branch, surfacing the bug as CI's very first pre-commit-gate failure (reported by
the human after pushing VAI-003). Reproduced locally by clearing `models/*.onnx` and reinstalling
`export/requirements.txt` into a throwaway venv to match a clean-checkout CI run exactly; confirmed
the fix resolves it in that same reproduction before applying it for real.

**What was rejected**: Ruled out transitive-dependency drift (a common risk with unpinned
sub-dependencies like `transformers`/`diffusers`/`s3tokenizer` in `chatterbox-tts`'s dependency
tree) by installing `requirements.txt` fresh and confirming it resolved to the *same* versions
already in the local dev venv — the bug reproduced regardless, isolating it to the shared-mutation
logic, not a version mismatch.

**What's next**: No open follow-up — `build_wrapper()` is now idempotent under repeated calls in
the same process, which is the property every `export_*.py`'s model-loading helper needs (see
`export_hifigan.py::_fuse_weight_norm`'s pre-existing idempotency comment for the established
pattern).

---

## 2026-08-16 — VAI-003: Export S3Gen flow estimator + Euler ODE loop, chain into HiFiGAN

**What changed**: Added `export/export_s3gen.py`, which exports the S3Gen flow-matching estimator
(`ConditionalDecoder`, accessed via `s3gen.flow.decoder.estimator`) as a static per-step ONNX graph
— the same per-step call `ConditionalCFM.solve_euler` makes inside its Python loop
(`x, mask, mu, t, spks, cond` in; `dxdt` out, batch pre-doubled for CFG). Added
`crates/vocalai-core/src/s3gen.rs`: `cosine_t_span()` (the `t_scheduler='cosine'` schedule),
`solve_euler()` (the CFG-doubled fixed-step Euler loop, generic over the per-step estimator call —
see ADR-0004 for why), `run_estimator()`/`mel_to_waveform()` (real `ort::Session`-backed adapters
for the estimator and the Milestone-2 HiFiGAN session), and `generate_waveform()` (chains both).
5 new Rust unit tests cover the cosine schedule and the Euler/CFG math against a synthetic linear
estimator (`dxdt = mu - x`) with hand-computed expected outputs — no ONNX Runtime session or model
file needed. Added `export/parity_check.py::check_s3gen`, which replicates the identical
CFG-doubled loop in Python (`_solve_euler_onnx`) driving the exported `s3gen_estimator.onnx` +
`hifigan.onnx`, and compares both the intermediate mel and the final waveform against the real
PyTorch `ConditionalCFM.solve_euler` + HiFiGAN wrapper (mel max_abs_diff ~4e-5, waveform ~1e-5;
well within atol=1e-4/rtol=1e-3). Added `ndarray = "0.17"` + `ort`'s `ndarray` feature to
`vocalai-core`'s `Cargo.toml` (version pinned to match `ort` 2.0.0-rc.13's own `ndarray`
dependency, so the two share one type in the dependency graph).

**Why**: Milestone 3 is the first Rust code to actually drive an ONNX Runtime session (Milestones
1-2 only built the EP-selection list). Model weights/`.onnx` files are git-ignored build artifacts
(`CLAUDE.md` §1) and don't exist in a fresh clone, so the Euler loop's real correctness risk — the
CFG batch-doubling and combination formula, the Euler update — needed to be testable without a
real model file or network access; making `solve_euler` generic over the estimator call achieves
that (ADR-0004) while the numerically-fragile ONNX-vs-PyTorch check stays in `parity_check.py`
alongside every other component's parity check, following the same pattern Milestone 2 established.

**What was rejected**: Writing `solve_euler` directly against `&mut ort::Session` with no Rust-side
math tests (would violate TDD, `CONVENTIONS.md` §1, and leave the CFG/Euler math uncovered in
Rust). Bundling a fixture `.onnx` for Rust tests (violates the no-binary-artifacts constraint and
wouldn't test anything the synthetic-closure test doesn't already cover). See ADR-0004 for the
full rationale.

**What's next**: Milestone 4 — export T3 as decoder-with-past, implement the KV-cache decode loop
+ sampling in `vocalai-core/src/t3.rs` (`docs/issues.md` `VAI-004`) — the main technical risk of
Phase 1 (plan §9 Open Items).

---

## 2026-08-16 — VAI-002: Export HiFiGAN/voice-encoder/S3-tokenizer to ONNX + `parity_check.py`

**What changed**: Set up the export venv (`export/.venv`, Python 3.12 — chatterbox-tts==0.1.7
requires >=3.10) and installed the real toolchain. Added `export/_common.py` (shared model
loading, ONNX export helper, comparison helper), `export/export_hifigan.py`,
`export/export_ve.py`, `export/export_s3tokenizer.py`, and `export/parity_check.py`, plus
`export/tests/test_parity_check.py` (7 tests total, all real — no mocks). All three components
export and pass parity against the PyTorch reference on a fixed input (HiFiGAN
max_abs_diff=5.9e-5, voice encoder 2.1e-7, S3 tokenizer exact-match on discrete tokens; default
tolerance atol=1e-4/rtol=1e-3).

Model loading bypasses `ChatterboxTTS.from_pretrained()` — it unconditionally constructs a
`PerthImplicitWatermarker`, which errors in this chatterbox-tts/resemble-perth combo (perth's
`PerthNet` import silently no-ops on a missing `pkg_resources`/`setuptools`, but
`ChatterboxTTS.__init__` calls it unguarded). Milestone 2 doesn't need T3 or PerthNet anyway, so
`_common.py` downloads and loads only `ve.safetensors`/`s3gen.safetensors` directly.

HiFiGAN's vocoder (`HiFTGenerator.decode()`) calls `torch.stft`/`torch.istft` internally; neither
exports cleanly in this torch version: `return_complex=True` has no ONNX symbolic at all, and
`torch.istft` has none full stop. `export_hifigan.py`'s wrapper reimplements both directions as
ONNX-exportable primitives — manual reflect-pad + `torch.stft(..., return_complex=False)` for
the forward direction (native ONNX `STFT` op), and a precomputed inverse-DFT matrix +
`conv_transpose1d`-as-overlap-add (an identity kernel scatters each windowed frame back to its
hop offset, summing overlaps — the standard iSTFTNet trick) for the inverse, with COLA
window-envelope normalization matching `torch.istft`'s default. It also reimplements
`SourceModuleHnNSF`/`SineGen`'s noise injection using `numpy.random.RandomState`-seeded
constants instead of live `torch.manual_seed` + `Uniform.sample`/`randn_like` — empirically,
the latter does NOT reproduce identically between an eager call and the same call replayed
through `torch.jit.trace` (confirmed with a minimal repro), which was the actual source of an
initial ~1.4e-2 parity failure; numpy's RNG isn't touched by JIT tracing's tensor-op
interception, so it reproduces bit-for-bit in both. `HiFTGenerator.remove_weight_norm()` also
had to be bypassed — chatterbox-tts applies the new parametrize-based `weight_norm` API but
calls the old function-based removal API on it, which raises; `_fuse_weight_norm()` walks the
module tree and removes via the matching (`torch.nn.utils.parametrize`) API instead.

S3 tokenizer's `AudioEncoderV2` precomputes rotary-embedding angles as a complex buffer
(`torch.polar`) and calls `torch.view_as_real` on it every forward — ONNX has no complex dtype,
so tracing fails the moment that buffer is embedded as a graph constant. Fixed by precomputing
the real-valued equivalent (`_real_freqs_cis`, identical math, no complex intermediate) and
patching `torch.view_as_real` to pass through already-real input (falls back to the real
implementation for genuinely complex tensors elsewhere, so this is behavior-preserving outside
this one path).

Added `Makefile`'s `test-py` target preferring `export/.venv/bin/python -m pytest` when that
venv exists (falls back to plain `pytest` otherwise, e.g. in CI, which installs
`export/requirements.txt` into the runner's system Python directly) — otherwise `make check`
would silently use system Python (3.9 here, too old for chatterbox-tts) instead of the export
venv. Documented per-platform (macOS/Linux/Windows) venv setup in `docs/dev-setup.md` — no
wrapper script, per explicit instruction, to keep it trivially auditable/cross-platform without
a bash/PowerShell fork.

**Why**: Milestone 2 proves the export + parity toolchain end-to-end on the three easy/static
components before Milestone 3's Euler ODE loop and Milestone 4's KV-cache decode loop, where a
broken toolchain would be a much more expensive place to discover issues (plan §7 sequencing
rationale). The `ChatterboxTTS.from_pretrained()` bypass and the STFT/ISTFT/RNG/weight-norm
workarounds were all necessary correctness fixes, not stylistic choices — each was verified by
making the failure reproduce, understanding the root cause, and confirming the fix against the
PyTorch reference via `parity_check.py`, not just silencing the export-time error.

**What was rejected**: Making the exported HiFiGAN graph handle arbitrary/dynamic input length —
`F.fold`'s ONNX symbolic (`aten::col2im`) errors on the traced dynamic `output_size` value
(worked around via `conv_transpose1d` instead, which incidentally may also be more dynamic-shape
friendly, but that's untested); the acceptance criteria only require a fixed-input parity check,
so proving full dynamic-length support is deferred to Milestone 6 when real variable-length CLI
audio is wired up. The dynamo-based (`torch.onnx.export(..., dynamo=True)`) exporter was tried
and rejected — it fails on a `torch.no_grad()`/random-sampling interaction inside `SineGen`
unrelated to any of the above ("cannot mutate tensors with frozen storage"), and the legacy
exporter path was already far enough along to be worth finishing instead of switching horses.
Also rejected: a bash-only setup script for the export venv (not cross-platform) and later a
Python setup-script wrapper too, in favor of plain documented per-platform commands in
`docs/dev-setup.md` (explicit user preference).

**What's next**: Milestone 3 — export the S3Gen flow estimator, implement the Euler ODE loop in
`vocalai-core/src/s3gen.rs`, chain into HiFiGAN (`docs/issues.md` `VAI-003`).

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
