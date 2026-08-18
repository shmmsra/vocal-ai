//! PerthNet audio watermarking: STFT -> ONNX encoder -> ISTFT, wrapped around a
//! 24kHz<->32kHz resample.
//!
//! Reimplements `PerthImplicitWatermarker.apply_watermark`
//! (`perth/perth_net/perth_net_implicit/perth_watermarker.py`, from the external
//! `resemble-perth` package) against the exported encoder-only ONNX graph
//! (`export/export_perthnet.py`). PerthNet operates internally at 32kHz
//! (`PerthConfig.sample_rate`) while the rest of this pipeline runs at 24kHz
//! (`S3GEN_SR`), so `apply_watermark` always resamples up before encoding and
//! back down after -- matching the reference's `change_rate` branch, which is
//! unconditionally true for this pipeline's fixed 24kHz output.
//!
//! The STFT/log-magnitude normalization/ISTFT/resample around the network call
//! is classical DSP, not exported to ONNX -- reimplemented directly here, the
//! same treatment as T3's sampling math and S3Gen's Euler loop being hand-rolled
//! around their exported network forwards (docs/phase1-onnx-rust-cli-plan.md §5).
//! As with `s3gen::solve_euler`/`t3::run_decoder`, `apply_watermark` is generic
//! over the encoder-step call so the DSP pipeline is unit-testable without a
//! live ONNX session; `run_encoder` provides the real `ort`-backed wiring.
//!
//! Unlike the exported networks (T3, S3Gen, HiFiGAN, VE, S3 tokenizer), this
//! module's STFT/ISTFT/resample math has no PyTorch-reference parity check --
//! classical DSP isn't ONNX-exported, so `CLAUDE.md` §1's parity hard constraint
//! doesn't gate it. Correctness here rests on the round-trip property tests
//! below, not cross-language numerical parity; `rubato`'s resampler will not
//! bit-match librosa's default `soxr_hq`, and that's an accepted, documented gap
//! (see docs/issues.md VAI-005), not a silently-assumed one.
//!
//! [`stft_magphase`] was, however, manually spot-checked once against a live
//! `AudioProcessor.signal_to_magphase` call (a synthetic 220Hz tone, dumped to
//! `.npy` and compared against this module's output) -- not a repeatable test,
//! since it isn't wired into CI, but worth recording: the energetic bin (14,
//! where the tone's energy sits) matched to ~1e-7 across all 21 frames. The
//! only bins that disagreed by more than float32 rounding were near-silent ones
//! (DC leakage, near-Nyquist) where the true magnitude sits at or below
//! `stft_magnitude_min` -- there, different FFT implementations' summation
//! order disagrees by orders of magnitude in *relative* terms after log
//! compression, while both sides are inaudibly close to zero in *absolute*
//! terms. That's expected floating-point behavior at a near-zero signal, not a
//! framing/windowing bug -- the actual signal-carrying content matched almost
//! exactly.
//!
//! Detection (`get_watermark`) is out of scope: `chatterbox/tts.py` only ever
//! calls `apply_watermark` (`self.watermarker.apply_watermark(wav, ...)`).
//!
//! See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 5).

use ndarray::{Array1, Array2, Axis, Ix3};
use ort::session::Session;
use ort::value::Tensor;
use realfft::num_complex::Complex32;
use realfft::RealFftPlanner;
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Fft as FftResampler, FixedSync, Resampler};

/// PerthNet's internal sample rate (`PerthConfig.sample_rate`).
pub const SAMPLE_RATE: u32 = 32_000;
/// This pipeline's output sample rate (`S3GEN_SR`, plan §3) -- what
/// `apply_watermark` is always called with in practice.
pub const PIPELINE_SAMPLE_RATE: u32 = 24_000;
/// `PerthConfig.n_fft` / `window_size` (they're equal, so no separate win-length
/// zero-padding is needed, unlike HiFiGAN's STFT wrapper).
pub const N_FFT: usize = 2048;
/// `PerthConfig.hop_size`.
pub const HOP_SIZE: usize = 320;
/// `n_fft // 2 + 1`.
pub const NFREQ: usize = N_FFT / 2 + 1;
/// `PerthConfig.stft_magnitude_min`.
const STFT_MAGNITUDE_MIN: f32 = 1e-9;
/// `normalize()`'s default `headroom_db`.
const HEADROOM_DB: f32 = 15.0;

fn min_level_db() -> f32 {
    20.0 * STFT_MAGNITUDE_MIN.log10()
}

/// Periodic Hann window, matching `torch.hann_window(len, periodic=True)` (the
/// default `torchaudio.transforms.Spectrogram`/`InverseSpectrogram` use).
fn hann_window(len: usize) -> Vec<f32> {
    (0..len)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / len as f32).cos())
        .collect()
}

/// Reflect-pads `signal` by `pad` samples on each side, matching
/// `torch.nn.functional.pad(..., mode="reflect")` (what `torch.stft(center=True,
/// pad_mode="reflect")` does internally before framing).
fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    let n = signal.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    out.extend((1..=pad).rev().map(|k| signal[k.min(n - 1)]));
    out.extend_from_slice(signal);
    out.extend((0..pad).map(|k| signal[n - 2 - k]));
    out
}

/// `normalize()` (perth `utils.py`): maps a dB-scale magnitude onto roughly
/// `[0, 1]` given `stft_magnitude_min`/`headroom_db`.
fn normalize_db(mag_db: f32) -> f32 {
    (mag_db - min_level_db()) / (-min_level_db() + HEADROOM_DB)
}

/// `denormalize_spectrogram()` (perth `utils.py`): inverse of [`normalize_db`].
fn denormalize_db(normalized: f32) -> f32 {
    normalized * (-min_level_db() + HEADROOM_DB) + min_level_db()
}

/// STFT + log-magnitude normalize + phase, matching
/// `AudioProcessor.signal_to_magphase` (perth `audio_processor.py`) exactly:
/// reflect-pad by `n_fft/2`, frame with `hop_size`, Hann-window, real FFT per
/// frame (unnormalized, matching `torch.stft(normalized=False)`), then
/// `cx_to_magphase` (dB-scale magnitude, clipped to `stft_magnitude_min`,
/// normalized; phase via `atan2`).
///
/// Returns `(normalized_magspec, phase)`, each `(NFREQ, num_frames)`.
pub fn stft_magphase(signal: &[f32]) -> (Array2<f32>, Array2<f32>) {
    let pad = N_FFT / 2;
    let padded = reflect_pad(signal, pad);
    let window = hann_window(N_FFT);
    let num_frames = (padded.len() - N_FFT) / HOP_SIZE + 1;

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(N_FFT);

    let mut magspec = Array2::<f32>::zeros((NFREQ, num_frames));
    let mut phase = Array2::<f32>::zeros((NFREQ, num_frames));
    let mut frame_buf = r2c.make_input_vec();
    let mut spectrum = r2c.make_output_vec();

    for t in 0..num_frames {
        let start = t * HOP_SIZE;
        for i in 0..N_FFT {
            frame_buf[i] = padded[start + i] * window[i];
        }
        r2c.process(&mut frame_buf, &mut spectrum)
            .expect("fixed-size real FFT never fails on correctly-sized buffers");
        for (f, bin) in spectrum.iter().enumerate() {
            let magnitude = bin.norm().max(STFT_MAGNITUDE_MIN);
            magspec[[f, t]] = normalize_db(20.0 * magnitude.log10());
            phase[[f, t]] = bin.im.atan2(bin.re);
        }
    }
    (magspec, phase)
}

/// Inverse of [`stft_magphase`]: denormalize + exponentiate the log-magnitude,
/// recombine with `phase` into a complex spectrum, inverse real FFT per frame,
/// Hann-window (synthesis), overlap-add with COLA (window-energy) normalization,
/// then trim the `n_fft/2` reflect-padding back off -- matching
/// `AudioProcessor.magphase_to_signal` (which delegates to
/// `torch.istft(center=True)`'s default behavior). This is the same
/// window-envelope-normalized overlap-add recipe already proven correct in
/// `export_hifigan.py`'s `_istft_onnx`, expressed with a real inverse FFT per
/// frame instead of a precomputed DFT matrix.
pub fn magphase_to_signal(magspec: &Array2<f32>, phase: &Array2<f32>) -> Array1<f32> {
    let (nfreq, num_frames) = magspec.dim();
    assert_eq!(nfreq, NFREQ, "magspec must have NFREQ frequency bins");
    assert_eq!(phase.dim(), magspec.dim(), "magspec/phase shape mismatch");

    let window = hann_window(N_FFT);
    let mut planner = RealFftPlanner::<f32>::new();
    let c2r = planner.plan_fft_inverse(N_FFT);

    let pad = N_FFT / 2;
    let out_len = if num_frames == 0 {
        0
    } else {
        HOP_SIZE * (num_frames - 1) + N_FFT
    };
    let mut ola = vec![0.0_f32; out_len];
    let mut envelope = vec![0.0_f32; out_len];

    let mut spectrum = c2r.make_input_vec();
    let mut frame_buf = c2r.make_output_vec();

    for t in 0..num_frames {
        for f in 0..NFREQ {
            let normalized = magspec[[f, t]];
            let mag_db = denormalize_db(normalized);
            let magnitude = (10.0_f32).powf((mag_db / 20.0).min(10.0));
            let ph = phase[[f, t]];
            spectrum[f] = Complex32::new(magnitude * ph.cos(), magnitude * ph.sin());
        }
        // DC and Nyquist bins must be purely real for a real-valued inverse FFT
        // (guaranteed by the forward real FFT that produced `phase`); floating
        // point makes `magnitude * phase.sin()` a tiny non-zero epsilon rather
        // than exactly 0 when `phase` is near `+-PI`, which realfft rejects.
        spectrum[0].im = 0.0;
        spectrum[NFREQ - 1].im = 0.0;
        c2r.process(&mut spectrum, &mut frame_buf)
            .expect("fixed-size inverse real FFT never fails on correctly-sized buffers");

        let start = t * HOP_SIZE;
        for i in 0..N_FFT {
            // 1/N_FFT: realfft's inverse transform is unnormalized (see its README).
            let sample = frame_buf[i] / N_FFT as f32 * window[i];
            ola[start + i] += sample;
            envelope[start + i] += window[i] * window[i];
        }
    }

    let length = if num_frames == 0 {
        0
    } else {
        HOP_SIZE * (num_frames - 1)
    };
    let mut out = Array1::<f32>::zeros(length);
    for i in 0..length {
        out[i] = ola[pad + i] / envelope[pad + i].max(1e-11);
    }
    out
}

/// Resamples `signal` from `from_hz` to `to_hz` (mono) via `rubato`'s FFT-based
/// synchronous resampler -- a good fit for the simple 32000/24000 = 4/3 ratio
/// this module always uses. Not bit-exact with librosa's default `soxr_hq`
/// resampler (see module docs) -- correctness here is "reasonable audio
/// fidelity", not cross-language numerical parity.
fn resample(signal: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || signal.is_empty() {
        return signal.to_vec();
    }
    const CHUNK_SIZE: usize = 1024;
    let mut resampler = FftResampler::<f32>::new(
        from_hz as usize,
        to_hz as usize,
        CHUNK_SIZE,
        1,
        FixedSync::Both,
    )
    .expect("32000/24000 is a valid, simple resampling ratio");

    let input_channels = vec![signal.to_vec()];
    let input = SequentialSliceOfVecs::new(&input_channels, 1, signal.len())
        .expect("input buffer sized exactly to signal.len()");
    let output = resampler
        .process_all(&input, signal.len(), None)
        .expect("fixed-ratio FFT resampling of a well-formed buffer never fails");
    output.take_data()
}

/// Full watermarking pipeline, matching `PerthImplicitWatermarker.apply_watermark`:
/// resample `signal` (at [`PIPELINE_SAMPLE_RATE`]) up to PerthNet's internal
/// [`SAMPLE_RATE`], STFT, run the encoder (`encoder_step`), ISTFT, resample back
/// down. Generic over the encoder call so the surrounding DSP is unit-testable
/// without a live ONNX session (see `tests` below); [`run_encoder`] provides the
/// real `ort`-backed wiring.
pub fn apply_watermark<E>(
    signal: &[f32],
    encoder_step: &mut impl FnMut(&Array2<f32>) -> Result<Array2<f32>, E>,
) -> Result<Array1<f32>, E> {
    let upsampled = resample(signal, PIPELINE_SAMPLE_RATE, SAMPLE_RATE);
    let (magspec, phase) = stft_magphase(&upsampled);
    let wmarked_magspec = encoder_step(&magspec)?;
    let watermarked_upsampled = magphase_to_signal(&wmarked_magspec, &phase);
    let watermarked = resample(
        watermarked_upsampled.as_slice().expect("contiguous array"),
        SAMPLE_RATE,
        PIPELINE_SAMPLE_RATE,
    );
    Ok(Array1::from_vec(watermarked))
}

/// Runs the exported PerthNet encoder (`export/export_perthnet.py`,
/// `models/perthnet_encoder.onnx`) on a `(NFREQ, T)` normalized log-magnitude
/// spectrogram, adding/removing the batch axis the ONNX graph expects.
pub fn run_encoder(session: &mut Session, magspec: &Array2<f32>) -> ort::Result<Array2<f32>> {
    let batched = magspec.clone().insert_axis(Axis(0));
    let outputs = session.run(ort::inputs!["magspec" => Tensor::from_array(batched)?])?;
    let wmarked = outputs["wmarked_magspec"].try_extract_array::<f32>()?;
    Ok(wmarked
        .into_dimensionality::<Ix3>()
        .expect("wmarked_magspec is always rank-3")
        .remove_axis(Axis(0))
        .to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_matches_known_values() {
        // torch.hann_window(8, periodic=True): [0, 0.1464, 0.5, 0.8536, 1.0, 0.8536, 0.5, 0.1464]
        let w = hann_window(8);
        let expected = [0.0, 0.1464, 0.5, 0.8536, 1.0, 0.8536, 0.5, 0.1464];
        for (a, b) in w.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn reflect_pad_matches_numpy_reflect_convention() {
        // np.pad([1,2,3,4,5], 2, mode="reflect") == [3,2,1,2,3,4,5,4,3]
        let signal = [1.0, 2.0, 3.0, 4.0, 5.0];
        let padded = reflect_pad(&signal, 2);
        assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn normalize_denormalize_round_trips() {
        for mag_db in [-120.0_f32, -60.0, -20.0, 0.0, 10.0] {
            let round_tripped = denormalize_db(normalize_db(mag_db));
            assert!((round_tripped - mag_db).abs() < 1e-3);
        }
    }

    fn synthetic_signal(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 32_000.0).sin())
            .collect()
    }

    #[test]
    fn stft_then_istft_round_trips_a_synthetic_signal() {
        // A hop-aligned length: like `torch.istft` (what the PyTorch reference
        // uses via `torchaudio.transforms.InverseSpectrogram`), this ISTFT
        // returns `hop_size * (num_frames - 1)` samples with no explicit target
        // length, which is only exactly `signal.len()` when the length is a
        // multiple of `HOP_SIZE` -- otherwise up to `HOP_SIZE - 1` trailing
        // samples are dropped by design, matching the reference's own behavior.
        let len = HOP_SIZE * 20;
        let signal = synthetic_signal(len);
        let (magspec, phase) = stft_magphase(&signal);
        let reconstructed = magphase_to_signal(&magspec, &phase);

        assert_eq!(reconstructed.len(), signal.len());
        // Trim a frame's worth from each end: edge frames see the most
        // reflect-padding/window-taper error.
        let trim = N_FFT / HOP_SIZE + 1;
        for i in trim..(len - trim) {
            assert!(
                (reconstructed[i] - signal[i]).abs() < 5e-3,
                "index {i}: {} vs {}",
                reconstructed[i],
                signal[i]
            );
        }
    }

    #[test]
    fn resample_preserves_duration_within_a_few_samples() {
        let signal = synthetic_signal(24_000); // 1 second at 24kHz
        let up = resample(&signal, PIPELINE_SAMPLE_RATE, SAMPLE_RATE);
        assert!(
            (up.len() as i64 - 32_000).abs() < 32,
            "up.len()={}",
            up.len()
        );

        let back_down = resample(&up, SAMPLE_RATE, PIPELINE_SAMPLE_RATE);
        assert!(
            (back_down.len() as i64 - 24_000).abs() < 32,
            "back_down.len()={}",
            back_down.len()
        );
    }

    #[test]
    fn apply_watermark_with_identity_encoder_approximately_preserves_signal() {
        let signal = synthetic_signal(24_000);
        let mut identity = |m: &Array2<f32>| -> Result<Array2<f32>, &'static str> { Ok(m.clone()) };
        let watermarked = apply_watermark(&signal, &mut identity).unwrap();

        // Round-tripping through resample+STFT+ISTFT+resample isn't lossless;
        // this checks it's a recognizable reconstruction, not a numerically
        // tight one (that's the accepted gap documented in the module docs).
        assert!((watermarked.len() as i64 - signal.len() as i64).abs() < 64);
        let trim = 2_000; // resampler filter ringing lives at the edges
        let compare_len = watermarked.len().min(signal.len());
        let mut max_diff = 0.0_f32;
        for i in trim..(compare_len - trim) {
            max_diff = max_diff.max((watermarked[i] - signal[i]).abs());
        }
        assert!(max_diff < 0.2, "max_diff={max_diff}");
    }

    #[test]
    fn apply_watermark_propagates_encoder_errors() {
        let signal = synthetic_signal(4_000);
        let mut failing =
            |_: &Array2<f32>| -> Result<Array2<f32>, &'static str> { Err("encoder failed") };
        let result = apply_watermark(&signal, &mut failing);
        assert_eq!(result, Err("encoder failed"));
    }
}
