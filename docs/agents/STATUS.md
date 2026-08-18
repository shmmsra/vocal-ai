# vocal-ai — Current Status & Backlog

> Updated: 2026-08-18
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
| 1 — Milestone 6, part B.2 (VAI-006): `--voice` zero-shot cloning | 📋 Planned |
| 1 — VAI-009: re-export HiFiGAN with a dynamic-length `speech_feat` input | ✅ Complete |
| 1 — Milestone 7: per-platform packaging (see `docs/phase1-onnx-rust-cli-plan.md` §7) | 📋 Planned |

*Update this table as phases progress. Use ✅ Complete / 🔄 In progress / 📋 Planned / 🚫 Blocked.*

**Current test counts**: 53 real Rust tests (`crates/vocalai-core/src/session.rs`'s EP-ordering
coverage — 2 under default features, +1 more under `--features coreml` — `s3gen.rs`'s 5
Euler-loop/CFG-math tests plus 10 new Milestone-6 tests (bucket selection, token padding, valid-
prefix slicing, `cond` assembly, noise-shape sampling, speaker-embedding normalize+affine),
`t3.rs`'s 14 KV-cache-loop/sampling-math tests (CFG combine, repetition penalty, temperature,
min-p, top-p, greedy/multinomial selection, embedding-table lookup, `.npy` round-trip, end-to-end
synthetic decode loop) plus 2 new speech-token-filter tests, `tokenizer.rs`'s 10 new tests
(`punc_norm` verified line-for-line against the live Python reference including its double-space
punctuation quirk, plus CFG-doubling/sot-eot padding), `audio.rs`'s 2 new WAV round-trip tests, and
`watermark.rs`'s 7 STFT/ISTFT/resample tests (Hann-window values, reflect-pad convention, dB
normalize/denormalize round-trip, a synthetic-signal STFT→ISTFT round trip, resample duration
preservation, an identity-encoder end-to-end `apply_watermark` round trip, and encoder-error
propagation), all pure `ndarray`/DSP/string math with no ONNX Runtime session or model file needed
except `tokenizer.rs`'s `TextTokenizer::from_file` path, not unit-tested against a live
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
download needed) and require `export/requirements.txt` installed, see `docs/dev-setup.md`).
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

1. Milestone 6, part B.2 (VAI-006) — `--voice` zero-shot cloning: `voice_encoder.rs`,
   `s3tokenizer.rs`, `campplus.rs`, the mel-filterbank builder, 16 kHz resample/mel/speaker-
   embedding preprocessing. See `docs/phase1-onnx-rust-cli-plan.md` §7 Milestone 6. This is the
   only remaining item before VAI-006 itself can close.
2. See `docs/issues.md` for the tracked tickets (`VAI-006`, `VAI-007`).

---

## Recently closed

| Date | Ticket | Summary | Commit |
|------|--------|---------|--------|
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
