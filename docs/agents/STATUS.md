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
| 1 — Milestones 6–7: full pipeline wiring + CLI, per-platform packaging (see `docs/phase1-onnx-rust-cli-plan.md` §7) | 📋 Planned |

*Update this table as phases progress. Use ✅ Complete / 🔄 In progress / 📋 Planned / 🚫 Blocked.*

**Current test counts**: 29 real Rust tests (`crates/vocalai-core/src/session.rs`'s EP-ordering
coverage — 2 under default features, +1 more under `--features coreml` — `s3gen.rs`'s 5
Euler-loop/CFG-math tests, `t3.rs`'s 14 KV-cache-loop/sampling-math tests (CFG combine,
repetition penalty, temperature, min-p, top-p, greedy/multinomial selection, embedding-table
lookup, `.npy` round-trip, end-to-end synthetic decode loop), and `watermark.rs`'s 7
STFT/ISTFT/resample tests (Hann-window values, reflect-pad convention, dB normalize/denormalize
round-trip, a synthetic-signal STFT→ISTFT round trip, resample duration preservation, an
identity-encoder end-to-end `apply_watermark` round trip, and encoder-error propagation), all pure
`ndarray`/DSP math with no ONNX Runtime session or model file needed) + 10 real Python tests in
`export/` (`test_requirements.py` checks `export/requirements.txt` pins; `test_parity_check.py`
covers the `_common.allclose_report` comparison helper plus end-to-end ONNX-vs-PyTorch parity for
HiFiGAN, the voice encoder, the S3 tokenizer, the S3Gen flow estimator (mel→waveform, chained
through the Milestone-2 HiFiGAN export), T3 (greedy-decode token-sequence + per-step logits parity
against the real `transformers`-backed reference, `parity_check.py::check_t3`), and now PerthNet's
encoder (`parity_check.py::check_perthnet`, synthetic-magspec parity against the real `Encoder`
submodule) — these download the Chatterbox checkpoint from HuggingFace on first run (PerthNet's
weights ship inside the `resemble-perth` package itself, no download needed) and require
`export/requirements.txt` installed, see `docs/dev-setup.md`).
`make check` passes cleanly. No placeholder/ignored tests remain.

**Residual risk (VAI-005)**: `watermark.rs`'s STFT/ISTFT/resample math has no PyTorch-reference
parity check — classical DSP isn't ONNX-exported, so `CLAUDE.md` §1's hard constraint doesn't gate
it. `stft_magphase` was manually spot-checked once against a live `AudioProcessor` call (see the
module's doc comment) and matched to ~1e-7 on signal-carrying content; `rubato`'s resampler is not
bit-exact with librosa's default `soxr_hq`. Correctness rests on round-trip unit tests, not
cross-language numerical parity — flagged, not silently assumed, and worth revisiting once
Milestone 6 makes real end-to-end audio available to listen to.

**CI**: split into two workflows since 2026-08-17 — `.github/workflows/ci.yml` (fast:
fmt/clippy/`cargo test`/`pytest -m "not parity"`) and `.github/workflows/parity.yml` (real-
checkpoint ONNX-vs-PyTorch tests, `pytest -m "parity and not heavy_build"` — 4 of the 5:
hifigan/ve/s3tokenizer/s3gen). Same triggers as before (every push/PR to `main`); see ADR-0006.
T3's parity test (`@pytest.mark.heavy_build`) is excluded from CI as of 2026-08-18 — its
from-scratch ONNX export measured ~9GB peak memory, more than this free-tier private-repo
runner has (verified independent of `do_constant_folding`; `external_data=True` is silently
ignored on the legacy, non-`dynamo` export path). It still runs locally (`make
test-py-parity`/`make check`) and must be run manually before committing changes to
`export/export_t3.py` or `crates/vocalai-core/src/t3.rs` — see ADR-0007.

---

## What's next

The next logical work, in priority order. Update at the end of every session.

1. Milestone 6 — wire the full pipeline (tokenizer → T3 → S3Gen → HiFiGAN → watermark → WAV) in
   `vocalai-core`, plus the `clap` CLI in `vocalai-cli` (see `docs/phase1-onnx-rust-cli-plan.md`
   §7, milestone 6). This is also the first point real end-to-end audio exists to manually verify
   `watermark.rs`'s resampler-fidelity residual risk (see STATUS test-counts section above).
2. See `docs/issues.md` for the tracked ticket (`VAI-006`).

---

## Recently closed

| Date | Ticket | Summary | Commit |
|------|--------|---------|--------|
| 2026-08-18 | VAI-005 | Export PerthNet encoder; implement STFT/ISTFT/resample watermarking pipeline (`watermark.rs`); pin `setuptools<81` (real fix for a `pkg_resources` removal breaking `resemble-perth`) | _pending_ |
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
