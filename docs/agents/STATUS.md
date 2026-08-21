# vocal-ai — Current Status & Backlog

> Updated: 2026-08-20
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
| 1 — Milestone 5: Export PerthNet encoder; STFT/ISTFT/resample watermarking in `watermark.rs` | ✅ Complete |
| 1 — Milestone 6, part A (VAI-008): Export S3Gen's flow-encoder (bucketed) + CAMPPlus to ONNX | ✅ Complete |
| 1 — Milestone 6, part B.1 (VAI-006): wire the default-voice pipeline + CLI | ✅ Complete (real end-to-end audio verified for arbitrary text, VAI-009 unblocked it) |
| 1 — Milestone 6, part B.2 (VAI-006): `--voice` zero-shot cloning | ✅ Complete (see ADR-0011; produces non-silent audio end-to-end after a bug fix caught by manual testing, see `docs/CHANGELOG.md`; audible speaker-resemblance confirmation still pending) |
| 1 — VAI-009: re-export HiFiGAN with a dynamic-length `speech_feat` input | ✅ Complete |
| 1 — Milestone 7 (VAI-007): per-platform packaging — macOS/CoreML + Windows/Linux CPU (see `docs/phase1-onnx-rust-cli-plan.md` §7, ADR-0013) | 🔄 In progress |
| 1 — VAI-015: Windows/Linux CUDA/cuDNN-bundled GPU artifacts (split out of VAI-007) | 📋 Planned |

*Update this table as phases progress. Use ✅ Complete / 🔄 In progress / 📋 Planned / 🚫 Blocked.*

**Current test counts**: 77 real Rust tests (`crates/vocalai-core/src/session.rs`'s
execution-provider-selection coverage, VAI-011 — 3 under default features, +1 more under
`--features coreml` — `s3gen.rs`'s 5
Euler-loop/CFG-math tests plus 10 Milestone-6 tests (bucket selection, token padding, valid-
prefix slicing, `cond` assembly, noise-shape sampling, speaker-embedding normalize+affine),
`t3.rs`'s 14 KV-cache-loop/sampling-math tests (CFG combine, repetition penalty, temperature,
min-p, top-p, greedy/multinomial selection, embedding-table lookup, `.npy` round-trip, end-to-end
synthetic decode loop) plus 2 speech-token-filter tests, `tokenizer.rs`'s 10 tests
(`punc_norm` verified line-for-line against the live Python reference including its double-space
punctuation quirk, plus CFG-doubling/sot-eot padding), `audio.rs`'s 6 WAV round-trip/downmix/
resample tests (VAI-006 part B.2 added `read_wav` + arbitrary-ratio `resample` on top of B.1's 2
`write_wav` tests), `watermark.rs`'s 7 STFT/ISTFT/resample tests (Hann-window values, reflect-pad
convention, dB normalize/denormalize round-trip, a synthetic-signal STFT→ISTFT round trip, resample
duration preservation, an identity-encoder end-to-end `apply_watermark` round trip, and
encoder-error propagation), and (new in VAI-006 part B.2, see ADR-0011) `mel.rs`'s 10 tests, half
hand-derived (window shapes) and half spot-checked against real librosa/torchaudio output on
synthetic 440Hz tones (VE/Whisper/S3Gen mel flavors, Kaldi fbank, two Slaney-filterbank configs),
`voice_encoder.rs`'s 8 tests (silence-trim indices matched against a live `librosa.effects.trim`
call, partial-utterance striding/coverage math), and `campplus.rs`'s 3 fixed-frame-count
trim/repeat tests — all pure `ndarray`/DSP/string math with no ONNX Runtime session or model file
needed except `tokenizer.rs`'s `TextTokenizer::from_file` path, not unit-tested against a live
`tokenizer.json`) + 12 real Python tests in
`export/` (`test_requirements.py` checks `export/requirements.txt` pins; `test_parity_check.py`
covers the `_common.allclose_report` comparison helper plus end-to-end ONNX-vs-PyTorch parity for
HiFiGAN, the voice encoder, the S3 tokenizer, the S3Gen flow estimator (mel→waveform, chained
through the Milestone-2 HiFiGAN export), T3 (greedy-decode token-sequence + per-step logits parity
against the real `transformers`-backed reference, `parity_check.py::check_t3`), PerthNet's
encoder (`parity_check.py::check_perthnet`, synthetic-magspec parity against the real `Encoder`
submodule), and now the S3Gen flow-encoder (`check_s3gen_flow_encoder`, all 6 `TOKEN_BUCKETS`
checked for both ONNX-vs-eager match and padding invariance) and CAMPPlus
(`check_campplus`, single fixed 400-frame graph) — these download the Chatterbox checkpoint from
HuggingFace on first run (PerthNet's weights ship inside the `resemble-perth` package itself, no
download needed) and require `export/requirements.txt` installed, see `docs/dev-setup.md`) + 14 real Python tests
in `scripts/tests/` (VAI-007: `test_smoke_test_artifact.py` covers pass/fail behavior of the
structural ONNX/npy/tokenizer-json/binary-`--version`/extra-file checks against synthetic fixtures
— including a real minimal valid ONNX model built with `onnx.helper` — with no real model file or
checkpoint needed; `test_publish_models.py` covers `HF_TOKEN`-presence validation and the
missing-dir/no-`.onnx`-files guard clauses, without touching the network or HfApi).
`make check` passes cleanly — verified on both POSIX and Windows/MSVC as of 2026-08-18 (see
ADR-0010 for the two Windows-only breakages that were fixed: a `tokenizers`/`esaxx-rs` static-CRT
link mismatch, and a non-portable `test-py` recipe). No placeholder/ignored tests remain.

**Residual risk (VAI-005)**: `watermark.rs`'s STFT/ISTFT/resample math has no PyTorch-reference
parity check — classical DSP isn't ONNX-exported, so `CLAUDE.md` §1's hard constraint doesn't gate
it. `stft_magphase` was manually spot-checked once against a live `AudioProcessor` call (see the
module's doc comment) and matched to ~1e-7 on signal-carrying content; `rubato`'s resampler is not
bit-exact with librosa's default `soxr_hq`. Correctness rests on round-trip unit tests, not
cross-language numerical parity — flagged, not silently assumed, and worth revisiting once
Milestone 6 makes real end-to-end audio available to listen to.

**Residual risk (VAI-008)**: CAMPPlus's ONNX input is precomputed Kaldi-style fbank features
(`torchaudio.compliance.kaldi.fbank`); the Rust-side port of that feature extraction is not
guaranteed bit-exact, same category of risk as the watermark resampler gap above — see ADR-0009.
Separately (not a residual risk, but a real constraint downstream code must respect): both new
exports are fixed-length/bucketed, not dynamic — the S3Gen flow-encoder ships 6 static buckets
(`TOKEN_BUCKETS = 200..1200`) and CAMPPlus ships one static 400-frame graph, because a first
dynamic-length attempt at each was found broken (see ADR-0009 for the full diagnosis: relative-
position-encoding and `seg_pooling` export bugs in the third-party reference code, not this repo's
own math). Milestone 6's Rust wiring must pick the right bucket / assemble exactly 400 real frames
for CAMPPlus — this is load-bearing correctness logic.

**Residual risk (VAI-006 part B.2, ADR-0011)**: the `--voice` zero-shot cloning DSP front ends
(`mel.rs`'s four mel/Kaldi-fbank flavors, `voice_encoder.rs`'s `librosa.effects.trim` port and
partial-utterance striding) have no automated cross-language parity gate, same category as the two
residual risks above. Confidence rests on unit tests hand-derived from the governing formulas plus
one-time spot-checks against real librosa/torchaudio output on synthetic tones — not a repeatable
`parity_check.py`-style harness. Revisit if cloned-voice audio is found to sound subtly wrong (as
opposed to silent/crashing, which the unit tests would already catch).

**Resolved (VAI-009, closed 2026-08-18)**: `hifigan.onnx`'s `speech_feat` input used to be **hard
fixed** at `(1, 80, 50)`, no `dynamic_axes` at all — not merely an internal assumption as
`export_hifigan.py`'s own docstring used to flag. Two trace-baking bugs (a Python-int read off a
tensor's `.shape` getting registered as an ONNX literal — same category ADR-0009 already documented
for the flow-encoder/CAMPPlus exports) were the real cause, not just a missing `dynamic_axes`
declaration: the overlap-add envelope's `.repeat(1, 1, num_frames)` and the deterministic source
noise's `torch.tensor(rng.randn(*shape))` both baked the traced length. Both are now built from a
fixed-size buffer sliced dynamically. `vocalai --text "hello world" --out out.wav` (VAI-006's own
acceptance-criterion command) now succeeds end-to-end, verified for both short and much-longer
(~7.4s) text, both producing genuine non-silent audio confirmed audible by the repo owner. See
`docs/issues.md` VAI-009 and `docs/CHANGELOG.md`'s 2026-08-18 VAI-009 entry for the full diagnostic
trail.

**Resolved (VAI-011, closed 2026-08-19)**: added `--use-gpu`/`--use-cpu` CLI flags
(`session::ExecutionProviderPreference`); **neither flag defaults to CPU** (not auto-select — see
the performance note below for why), resolved once per `ModelBundle` and logged (`Using GPU
execution provider (CoreML)` / `Using CPU execution provider`). `--use-gpu` uses `ort`'s
`.error_on_failure()` instead of `.fail_silently()`, so a hardware EP that can't register errors out
rather than silently running CPU. `Makefile`'s `build:` target auto-detects the right feature per OS
(`coreml` on macOS, `cuda` elsewhere) — compiling with a hardware feature never requires local GPU
hardware or a CUDA toolchain (`ort-sys` downloads a prebuilt ONNX Runtime binary per target+
feature), only *using* the resulting EP at runtime does.

**Resolved (VAI-011, see VAI-014)**: manual testing on real Apple Silicon hardware found that
`s3gen_estimator.onnx` (S3Gen's flow-matching Euler ODE estimator, a UNet-style network) reliably
crashes ONNX Runtime's CoreML EP at inference time — root-caused by bisection to its genuinely
dynamic (non-bucketed) `2 * token_len` sequence length, the one session in the pipeline never given
the bucketing treatment ADR-0009 already gave the flow-encoder. Every other session, including T3's
full up-to-1000-iteration KV-cache decode loop, runs on CoreML successfully. Worked around:
`ModelBundle::load` pins the estimator to CPU whenever CoreML is resolved (CUDA untested, left
alone). `VAI-014` tracks the real fix: bucketing the estimator's export like the flow-encoder, so
CoreML can cover 100% of the pipeline with no CPU carve-out.

**Resolved (VAI-011) — why CPU is the default, not GPU**: once the crash above was fixed, CoreML's
*default* configuration measured 30-40% slower than CPU wall-clock (T3's decode loop calls
`session.run()` ~1000 times; CoreML's fixed per-call dispatch/specialization cost dominates across
many tiny sequential calls, worse since the decoder graph is only partially CoreML-covered so every
call also pays a CPU↔CoreML marshaling cost) — directly contradicting this session's original
assumption that T3's loop is where the GPU speedup lives. Benchmarked `ort`'s CoreML config knobs;
`ComputeUnits::CPUAndGPU` (excludes the Neural Engine) + `SpecializationStrategy::FastPrediction` +
`RequireStaticInputShapes` together fixed that severe regression (`ModelFormat::MLProgram` failed
outright for this graph — a clean error via `.error_on_failure()`, not a hang). This tuned config is
now permanent in `hardware_execution_providers()`.

**Correction (2026-08-20)**: this session initially read the tuned-config benchmark as "CPU parity
or better" — that doesn't hold up. The small-scale numbers were within RNG-driven run-to-run noise
(token count before EOS varies run to run), and a full-scale run already on hand at the time actually
showed the tuned config ~14% *slower* than CPU — a contradiction that should have been flagged
instead of dismissed. The repo owner's own real-world re-test confirmed no improvement. Honest
takeaway: the tuning fixes the severe regression down to roughly tied-or-somewhat-worse; it is not a
demonstrated speed win. CPU stays the default on this evidence; `--use-gpu` stays available (still a
real improvement over the naive config, and not worth removing) but should not be recommended for
speed. See `docs/decisions/0012-*.md` and `docs/CHANGELOG.md`'s 2026-08-20 entry for the full story.

**In progress (VAI-007, ADR-0013, started 2026-08-21)**: per-platform packaging. Two new
manual-trigger-only GitHub Actions workflows: `models-export.yml` (`ubuntu-latest` — exports
every `.onnx`/`.npy` via `make export`, gates on the **full** `make test-py-parity` including T3,
structurally validates the result with no inference (`scripts/smoke_test_artifact.py`), then
publishes to the public HF Hub repo `shmmsra/vocal-ai-models`) and `release.yml` (matrix of
`macos-latest`/CoreML, `windows-latest`/CPU, `ubuntu-latest`/CPU — builds `vocalai-cli`, downloads
the current HF Hub revision, stages a bundle with the binary + `models/` + `THIRD_PARTY_LICENSES`
+ `LICENSE`, structural-smoke-tests it, uploads as a release asset on a `v*` tag). Both smoke
tests are deliberately structural-only (`onnx.checker.check_model`, `.npy` load, `tokenizer.json`
parse, `<binary> --version`) — no ONNX Runtime session, no inference, no audio, per explicit
instruction; real end-to-end audio/CPU-fallback/memory validation stays a manual per-platform step
(`docs/manual-testing.md`). Windows/Linux CUDA/cuDNN-bundled GPU artifacts are deferred to VAI-015
— this pass only ships CoreML + CPU-only artifacts, all buildable/verifiable on standard public
GitHub-hosted runners with no new licensing research. Also added: `THIRD_PARTY_LICENSES` (verbatim
MIT notices, fulfilling ADR-0008's standing commitment), `scripts/publish_models.py` +
`scripts/smoke_test_artifact.py` + their tests (`make test-scripts`, wired into `make check`/
`ci.yml`), `make publish-models`/`make smoke-test` for local debugging. Corrected a runner-spec
assumption along the way (see ADR-0013): `macos-latest` is 7GB RAM, not 14GB (the 14GB figure is
the legacy `-intel` label); going public gives `ubuntu-latest` 16GB (double the 8GB private cap),
which is why model export (including T3's ~9GB peak) runs there, not on macOS. Not yet done:
actually running these workflows against the real public repo/secret (requires the repo owner to
flip GitHub visibility to public and add the `HF_TOKEN` secret — both explicitly left as
human-only steps, not run by the agent), and a first real tagged release.

**CI**: split into two workflows since 2026-08-17 — `.github/workflows/ci.yml` (fast:
fmt/clippy/`cargo test`/`pytest -m "not parity"`) and `.github/workflows/parity.yml` (real-
checkpoint ONNX-vs-PyTorch tests, `pytest -m "parity and not heavy_build"` — 7 of the 8:
hifigan/ve/s3tokenizer/s3gen/perthnet/s3gen_flow_encoder/campplus). Same triggers as before (every
push/PR to `main`); see ADR-0006.
T3's parity test (`@pytest.mark.heavy_build`) is excluded from CI as of 2026-08-18 — its
from-scratch ONNX export measured ~9GB peak memory, more than this free-tier private-repo
runner has (verified independent of `do_constant_folding`; `external_data=True` is silently
ignored on the legacy, non-`dynamo` export path). It still runs locally (`make
test-py-parity`/`make check`) and must be run manually before committing changes to
`export/export_t3.py` or `crates/vocalai-core/src/t3.rs` — see ADR-0007.

---

## What's next

The next logical work, in priority order. Update at the end of every session.

1. VAI-007 — finish per-platform packaging: the workflows/scripts landed this session
   (`models-export.yml`, `release.yml`, `THIRD_PARTY_LICENSES`, smoke-test tooling — see the
   "In progress" note above and ADR-0013), but nothing has actually *run* yet. Remaining:
   repo owner flips GitHub visibility to public + adds the `HF_TOKEN` secret (both human-only),
   then trigger `models-export.yml` for a real publish and cut a first real `v*` tag, then run
   the manual per-platform validation in `docs/manual-testing.md` (real audio, CPU-fallback
   equivalence, memory/swap benchmark) before calling Milestone 7 done.
2. VAI-015 — Windows/Linux CUDA/cuDNN-bundled GPU release artifacts (split out of VAI-007,
   ADR-0013): needs real GPU hardware to smoke-test and an NVIDIA redistribution-license check
   that hasn't been done yet (only the model weights were checked, in ADR-0008).
3. VAI-014 — bucket `s3gen_estimator.onnx`'s time dimension so CoreML covers the full pipeline
   with no CPU carve-out (see the VAI-011 residual-risk note above for the diagnosis).
4. VAI-012 — `--show-progress` console progress indicator (split out of VAI-011).
5. VAI-013 — likely superseded by VAI-007's `release.yml` build matrix; confirm and close once
   VAI-007 lands (not closed unilaterally — see `docs/issues.md`).
6. VAI-016 — version-bump-driven triggers for `models-export.yml`/`release.yml` (replace manual
   `workflow_dispatch`/tag-push with a `MODELS_VERSION` file + `Cargo.toml` workspace version bump
   detection, plus standardized GitHub-native release notes). Deliberately split out of VAI-007 so
   the manual-trigger pipeline could ship and get a first real run first.
7. Candidate, not yet scoped: revisit ADR-0007's `parity.yml` T3 exclusion now that public
   `ubuntu-latest` has 16GB RAM (see ADR-0013) — newly viable, not required.
8. See `docs/issues.md` for the full tracked-ticket list.

---

## Recently closed

| Date | Ticket | Summary | Commit |
|------|--------|---------|--------|
| 2026-08-20 | — | Correction: walked back VAI-011's "CoreML tuning reaches CPU parity" claim after the repo owner's real-world re-test found no improvement — docs-only, no code change | _pending_ |
| 2026-08-19 | VAI-011 | `--use-gpu`/`--use-cpu` execution-provider selection (CPU by default), `error_on_failure()` instead of silent hardware-EP fallback, `Makefile` OS-based feature auto-detection, CoreML tuned (`CPUAndGPU`+`FastPrediction`+`RequireStaticInputShapes`) to fix a measured 30-40% slowdown vs CPU, S3Gen flow estimator pinned to CPU on CoreML (real fix tracked as VAI-014) | _pending_ |
| 2026-08-18 | — | Add `make export` (+ `scripts/export-all.{sh,ps1}`) wrapping the 8-10 `export/` scripts `docs/dev-setup.md` §11.1 documents into one command; `--with-voice-cloning` opt-in for the two extra `--voice`-only exports | _pending_ |
| 2026-08-18 | VAI-006 | `--voice` zero-shot cloning (part B.2, ADR-0011): `mel.rs`, `voice_encoder.rs`, `s3tokenizer.rs`, `campplus.rs`; `pipeline.rs`'s `DefaultVoice` → `VoiceConditioning` with a `from_reference` constructor. Closes VAI-006. | _pending_ |
| 2026-08-18 | — | Fix `make check` on Windows/MSVC (ADR-0010): drop `tokenizers`' static-CRT `esaxx_fast` to resolve the `LNK2038` CRT mismatch; make the `test-py*` Makefile recipes `cmd.exe`-portable via `$(OS)`/`$(wildcard)` | _pending_ |
| 2026-08-18 | VAI-009 | Fix two trace-baking bugs in `export_hifigan.py` so `speech_feat` is a genuine dynamic ONNX axis; extend `check_hifigan` to 3 frame counts — unblocks VAI-006 part B.1's end-to-end acceptance criterion | `9ff4b4e` |
| 2026-08-18 | VAI-008 | Export S3Gen's flow-encoder (bucketed, ADR-0009) + CAMPPlus (fixed 400-frame window) to ONNX, closing the `mu`/`spks` gap found while starting Milestone 6 | `65b1642` |
| 2026-08-18 | VAI-005 | Export PerthNet encoder; implement STFT/ISTFT/resample watermarking pipeline (`watermark.rs`); pin `setuptools<81` (real fix for a `pkg_resources` removal breaking `resemble-perth`) | `91c92ea` |
| 2026-08-18 | — | Add ADR-0008 resolving PerthNet/Chatterbox license question (both MIT) | `1f29052` |
| 2026-08-18 | — | Exclude T3 parity from CI, `heavy_build` marker (ADR-0007) — the ~9GB export-time memory is a real resource ceiling, not what the `d536fac` fix (below) addressed | `98e549b` |
| 2026-08-17 | — | Split CI into fast + parity workflows (ADR-0006); fix `check_t3` OOM | `d536fac` |
| 2026-08-17 | VAI-004 | Export T3 as decoder-with-past (hand-rolled Llama, ADR-0005); KV-cache decode loop + sampling | `2e13c33` |
| 2026-08-16 | VAI-003 | Export S3Gen flow estimator + Euler ODE loop, chain into HiFiGAN | `1bc9095` |
| 2026-08-16 | VAI-002 | Export HiFiGAN/voice-encoder/S3-tokenizer to ONNX + `parity_check.py` | `820ff9a` |
| 2026-08-16 | VAI-001 | Cargo workspace + `ort` EP scaffold, export toolchain pins | `5b4815c` |
| 2026-08-16 | — | ai-sdlc-bootstrap scaffold | `ba0f453` |

---

## Deferred — pull only when a specific need surfaces

*Add tickets here when they're explicitly de-prioritized rather than open.*

- **Milestone 7's build-generation resourcing**: the release-build/bundling pipeline
  (plan §7 item 7) will need to run `torch.onnx.export` on T3 at least once — the
  same ~9GB-peak operation ADR-0007 excluded from CI. It cannot run on this repo's
  free-tier hosted GitHub Actions runner. Repo owner is deciding the approach
  (local build + upload release assets, a paid larger runner, or self-hosted) —
  revisit when Milestone 7 actually starts, not before.
