# ADR-0011: `--voice` zero-shot cloning's DSP front ends are hand-rolled Rust with no automated parity gate

**Date**: 2026-08-18
**Status**: Accepted
**Decider**: Shivam Mishra + AI agent (session 2026-08-18)

## Context

Milestone 6, part B.2 (`docs/issues.md` VAI-006) needed to wire `--voice` zero-shot cloning: turn a
live reference WAV into the same conditioning tensors the default-voice path already loads from
`models/default_voice/*.npy`. All three ONNX networks this needs -- the voice encoder (`ve.onnx`),
the S3 tokenizer (`s3tokenizer.onnx`), and CAMPPlus (`campplus.onnx`) -- were already exported and
parity-checked in earlier milestones (Milestone 2, VAI-008). What was missing was entirely
host-side preprocessing: four distinct mel/fbank feature-extraction "flavors", each copied from a
different Python module (`chatterbox/models/voice_encoder/melspec.py`,
`chatterbox/models/s3tokenizer/s3tokenizer.py`, `chatterbox/models/s3gen/utils/mel.py`,
`torchaudio.compliance.kaldi.fbank`), plus `librosa.effects.trim` and the voice encoder's
overlapping partial-utterance windowing (`voice_encoder.py`'s `get_frame_step`/`get_num_wins`/
`stride_as_partials`).

None of this DSP is part of any ONNX graph -- it is classical signal processing (STFT framing,
triangular mel filterbanks, RMS-based silence trimming) that runs host-side in the Python
reference too, feeding precomputed features into each network exactly like `watermark.rs`'s
STFT/ISTFT/resample already does for PerthNet (VAI-005) and like CAMPPlus's own fbank input already
does (ADR-0009). `CLAUDE.md` §1's hard constraint ("no exported component ships into
`vocalai-core` without a passing parity check") only gates *exported* components; it does not, and
was never intended to, gate hand-rolled DSP that has no PyTorch/ONNX counterpart to diff against in
`export/parity_check.py`'s harness.

## Decision

Four new modules implement this preprocessing directly in Rust, each documenting its own math and
provenance rather than sharing a generic abstraction across mel "flavors" that differ in every
parameter (n_fft, hop, window, power-vs-magnitude, log-vs-linear, sample rate):

- **`mel.rs`**: the shared low-level STFT/filterbank primitives (`stft_complex`,
  `slaney_mel_filterbank` for librosa's HTK=False/`norm="slaney"` convention,
  `kaldi_mel_filterbank`/`kaldi_fbank` for Kaldi's distinct mel-scale-and-windowing convention) and
  the four public "flavor" functions (`ve_mel_spectrogram`, `whisper_log_mel`, `s3gen_log_mel`,
  `kaldi_fbank`).
- **`voice_encoder.rs`**: `librosa.effects.trim`'s frame-RMS-vs-`top_db` silence trim, plus the
  partial-utterance striding/aggregation around `ve.onnx`.
- **`s3tokenizer.rs`**: wraps `s3tokenizer.onnx`, shared by both real callers (T3's cond-prompt
  tokens, truncated; S3Gen's prompt token, untruncated).
- **`campplus.rs`**: Kaldi-fbank extraction, per-utterance mean subtraction, and the
  trim-or-cyclically-repeat-to-exactly-400-frames windowing ADR-0009 already mandated.

`pipeline.rs`'s `DefaultVoice` struct is renamed `VoiceConditioning` and gains a second constructor,
`from_reference(bundle, wav_path)`, alongside the existing `load_default(dir)`. Both produce the
same six-tensor shape, so `synthesize`'s T3/S3Gen wiring downstream of voice selection needed no
changes. The three new ONNX sessions (`ve`, `s3tokenizer`, `campplus`) are lazily loaded on
`ModelBundle`, mirroring the flow-encoder buckets' lazy-load pattern (ADR/`docs/issues.md` VAI-008),
so the default-voice-only fast path pays no extra load cost.

**No automated cross-language parity gate was added for this new DSP.** Confidence instead comes
from two sources: (1) unit tests that hand-derive expected values from the governing formulas
(window shapes, filterbank normalization) the same way `watermark.rs`'s `hann_window`/`reflect_pad`
tests already do, and (2) unit tests that spot-check specific output values against the *real*
librosa/torchaudio implementations, computed once via the `export/.venv` Python environment on
synthetic tone signals and hardcoded as expected values with a tolerance band (see `mel.rs`'s
`*_matches_python_reference_on_a_440hz_tone` tests, `voice_encoder.rs`'s
`trim_silence_matches_python_reference_indices`). This is spot-checking, not an automated
`parity_check.py`-style gate that re-derives the reference on every CI run.

## Rationale

- Every hand-rolled-DSP precedent already in this repo (`watermark.rs`'s STFT/ISTFT/resample,
  `docs/issues.md` VAI-005; CAMPPlus's fbank-input gap, ADR-0009) treats this category of code the
  same way: unit-test-verified, explicitly flagged as a residual risk, not blocked on an automated
  parity harness that has nothing ONNX-side to compare against. Extending that same treatment here
  is consistency, not a new precedent.
- A real `parity_check.py`-style gate would require standing up a second, independent Python
  reference implementation call path for four different mel flavors plus `librosa.effects.trim` and
  re-running it in CI on every change -- disproportionate machinery for classical DSP whose formulas
  are public, stable, and unlikely to drift, versus the ML network weights `parity_check.py` exists
  to guard against silent numerical divergence in.
- Spot-checking against real librosa/torchaudio output (rather than only hand-derived values) still
  catches the most likely class of bug -- an off-by-one in framing, a wrong window convention, a
  transposed axis -- without building permanent CI infrastructure for it.

## Alternatives rejected

- **Add a `check_voice_cloning_dsp` function to `export/parity_check.py`**: rejected as
  disproportionate machinery for classical DSP with no ONNX graph on the other side of the diff (see
  Rationale). Revisit if real generated cloned audio is later found to sound wrong in a way unit
  tests didn't catch — the same trigger condition ADR-0009 already set for CAMPPlus's fbank-input
  risk and VAI-005 set for the watermark resampler.
- **Share `hann_window`/`reflect_pad` between `watermark.rs` and `mel.rs` via `pub(crate)`**:
  rejected in favor of small, explicitly-duplicated, independently-documented copies — each DSP
  module stays fully self-contained and reviewable without cross-module coupling for a handful of
  lines, the same call already made for `audio::resample` vs. `watermark::resample` (see
  `audio.rs`'s module doc).
- **A single generic mel-spectrogram function parameterized by every knob (power vs. magnitude,
  log vs. linear, center-pad vs. manual-pad, drop-last-frame vs. not)**: rejected — the four flavors
  differ in enough dimensions that a fully generic function would need as many parameters as there
  are call sites, trading four small, individually-documented-and-tested functions for one large,
  harder-to-verify one for no real reuse benefit beyond the shared `stft_complex` framing loop
  (which *is* factored out).

## Consequences

**Easier**: `--voice` zero-shot cloning is fully wired with no new ONNX exports required — all
three networks it needs were already parity-checked in earlier milestones.

**Harder**: three more lazily-loaded ONNX sessions in `ModelBundle`; a fourth (fifth, counting
CAMPPlus's) hand-rolled DSP front end with no automated parity gate, growing the surface area
`docs/agents/STATUS.md`'s "residual risk" note already covers for `watermark.rs` and CAMPPlus.

**New commitments / residual risk**: correctness of `mel.rs`'s four mel flavors and
`voice_encoder.rs`'s trim/striding rests on unit tests and one-time manual spot-checks against real
librosa/torchaudio output, not a repeatable cross-language parity gate. If cloned-voice audio is
later found to sound subtly wrong (as opposed to silent/crashing, which unit tests would already
catch), this is the first place to suspect — same treatment, same trigger condition, as the
watermark resampler (VAI-005) and CAMPPlus's fbank input (ADR-0009).
