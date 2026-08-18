//! Shared mel-spectrogram / Kaldi-fbank DSP front ends for `--voice` zero-shot
//! cloning (Milestone 6, part B.2, `docs/issues.md` VAI-006).
//!
//! Four distinct feature-extraction "flavors" are needed, one per reference
//! preprocessing step:
//! - [`ve_mel_spectrogram`]: the voice encoder's unscaled power-mel (40 bins,
//!   16kHz) -- `chatterbox/models/voice_encoder/melspec.py::melspectrogram`.
//! - [`whisper_log_mel`]: the S3-tokenizer's log-mel (128 bins, 16kHz), reused
//!   for both T3's cond-prompt tokens and S3Gen's prompt token --
//!   `chatterbox/models/s3tokenizer/s3tokenizer.py::S3Tokenizer.log_mel_spectrogram`.
//! - [`s3gen_log_mel`]: S3Gen's natural-log mel (80 bins, 24kHz) for the
//!   `prompt_feat` conditioning input --
//!   `chatterbox/models/s3gen/utils/mel.py::mel_spectrogram`.
//! - [`kaldi_fbank`]: CAMPPlus's log-fbank front end (16kHz) --
//!   `torchaudio.compliance.kaldi.fbank`.
//!
//! None of this DSP is ONNX-exported (same category as `watermark.rs`'s
//! STFT/ISTFT/resample), so `CLAUDE.md` §1's parity hard constraint doesn't gate
//! it -- correctness rests on unit tests spot-checked against real
//! librosa/torchaudio output (see each function's tests below), not an
//! automated cross-language parity gate. Tracked as a residual risk alongside
//! `watermark.rs`'s resampler gap and CAMPPlus's fbank-input gap (ADR-0009).
//!
//! `hann_window`/`reflect_pad` here duplicate small helpers already private to
//! `watermark.rs` rather than sharing them cross-module -- the same accepted-
//! duplication call the plan made for `audio::resample` vs `watermark`'s: each
//! DSP module stays fully self-contained and independently documented.

use ndarray::Array2;
use realfft::num_complex::Complex32;
use realfft::RealFftPlanner;

/// Periodic Hann window (`torch.hann_window(len, periodic=True)`, also
/// librosa's default STFT window since `librosa.filters.get_window(...,
/// fftbins=True)` is periodic). Used by every flavor in this module except
/// [`povey_window`] (Kaldi's own, symmetric-based, convention).
fn hann_window_periodic(len: usize) -> Vec<f32> {
    (0..len)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / len as f32).cos())
        .collect()
}

/// Symmetric Hann window (`torch.hann_window(len, periodic=False)`), the basis
/// for Kaldi's "povey" window ([`povey_window`]).
fn hann_window_symmetric(len: usize) -> Vec<f32> {
    if len == 1 {
        return vec![1.0];
    }
    (0..len)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / (len - 1) as f32).cos())
        .collect()
}

/// Kaldi's "povey" window: a symmetric Hann window raised to the 0.85 power
/// (`_feature_window_function`, `torchaudio/compliance/kaldi.py`) -- "like
/// hanning but goes to zero at edges" faster.
fn povey_window(len: usize) -> Vec<f32> {
    hann_window_symmetric(len)
        .into_iter()
        .map(|w| w.powf(0.85))
        .collect()
}

/// Reflect-pads `signal` by `pad` samples on each side, matching
/// `torch`/`numpy`'s "reflect" convention (mirrors without repeating the edge
/// sample).
fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    let n = signal.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    out.extend((1..=pad).rev().map(|k| signal[k.min(n - 1)]));
    out.extend_from_slice(signal);
    out.extend((0..pad).map(|k| signal[n - 2 - k]));
    out
}

/// Real FFT per frame, reflect-padded by `pad` on each side, periodic-Hann-
/// windowed, hop-strided -- the shared framing loop behind [`ve_mel_spectrogram`]/
/// [`whisper_log_mel`] (`pad = n_fft/2`, matching `center=True`) and
/// [`s3gen_log_mel`] (`pad = (n_fft-hop)/2`, matching S3Gen's manual
/// `center=False` padding). Returns `(n_fft/2+1, num_frames)` complex spectra.
fn stft_complex(signal: &[f32], pad: usize, n_fft: usize, hop: usize) -> Array2<Complex32> {
    let padded = reflect_pad(signal, pad);
    let window = hann_window_periodic(n_fft);
    let nfreq = n_fft / 2 + 1;
    let num_frames = if padded.len() >= n_fft {
        (padded.len() - n_fft) / hop + 1
    } else {
        0
    };

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let mut frame_buf = r2c.make_input_vec();
    let mut spectrum = r2c.make_output_vec();

    let mut out = Array2::<Complex32>::zeros((nfreq, num_frames));
    for t in 0..num_frames {
        let start = t * hop;
        for i in 0..n_fft {
            frame_buf[i] = padded[start + i] * window[i];
        }
        r2c.process(&mut frame_buf, &mut spectrum)
            .expect("fixed-size real FFT never fails on correctly-sized buffers");
        for (f, bin) in spectrum.iter().enumerate() {
            out[[f, t]] = *bin;
        }
    }
    out
}

/// Slaney mel-scale conversion (`htk=False`), matching `librosa.core.convert.hz_to_mel`.
fn hz_to_mel_slaney(hz: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0_f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f32.ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        hz / f_sp
    }
}

/// Inverse of [`hz_to_mel_slaney`], matching `librosa.core.convert.mel_to_hz`.
fn mel_to_hz_slaney(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0_f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f32.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_sp * mel
    }
}

/// librosa-style ("Slaney") mel filterbank, `htk=False`, `norm="slaney"` --
/// matches `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)` exactly.
/// Returns a `(n_mels, n_fft/2+1)` weight matrix.
pub fn slaney_mel_filterbank(
    sr: f32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Array2<f32> {
    let nfreq = n_fft / 2 + 1;
    let fftfreqs: Vec<f32> = (0..nfreq).map(|i| i as f32 * sr / n_fft as f32).collect();

    let min_mel = hz_to_mel_slaney(fmin);
    let max_mel = hz_to_mel_slaney(fmax);
    let mel_f: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_to_hz_slaney(min_mel + (max_mel - min_mel) * i as f32 / (n_mels + 1) as f32))
        .collect();

    let mut weights = Array2::<f32>::zeros((n_mels, nfreq));
    for i in 0..n_mels {
        let fdiff_lower = mel_f[i + 1] - mel_f[i];
        let fdiff_upper = mel_f[i + 2] - mel_f[i + 1];
        for (j, &f) in fftfreqs.iter().enumerate() {
            let lower = (f - mel_f[i]) / fdiff_lower;
            let upper = (mel_f[i + 2] - f) / fdiff_upper;
            weights[[i, j]] = lower.min(upper).max(0.0);
        }
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for j in 0..nfreq {
            weights[[i, j]] *= enorm;
        }
    }
    weights
}

/// Kaldi mel scale (`1127 * ln(1 + f/700)`), matching
/// `torchaudio.compliance.kaldi.mel_scale_scalar`. Distinct from the Slaney
/// scale above -- Kaldi's `get_mel_banks` always uses this formula, with no
/// `htk`/`slaney` choice.
fn kaldi_mel_scale(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// Kaldi's triangular mel filterbank (`get_mel_banks`, `vtln_warp=1.0` fixed --
/// this pipeline never VTLN-warps), zero-padded by one column on the right to
/// match a real FFT's `padded_window_size/2 + 1` bin count. Returns
/// `(num_bins, padded_window_size/2 + 1)`.
fn kaldi_mel_filterbank(
    num_bins: usize,
    padded_window_size: usize,
    sample_freq: f32,
    low_freq: f32,
    high_freq: f32,
) -> Array2<f32> {
    let num_fft_bins = padded_window_size / 2;
    let nyquist = 0.5 * sample_freq;
    let high_freq = if high_freq <= 0.0 {
        high_freq + nyquist
    } else {
        high_freq
    };
    let fft_bin_width = sample_freq / padded_window_size as f32;
    let mel_low = kaldi_mel_scale(low_freq);
    let mel_high = kaldi_mel_scale(high_freq);
    let mel_delta = (mel_high - mel_low) / (num_bins as f32 + 1.0);

    let mut weights = Array2::<f32>::zeros((num_bins, num_fft_bins + 1));
    for i in 0..num_bins {
        let left = mel_low + i as f32 * mel_delta;
        let center = mel_low + (i as f32 + 1.0) * mel_delta;
        let right = mel_low + (i as f32 + 2.0) * mel_delta;
        for j in 0..num_fft_bins {
            let mel_j = kaldi_mel_scale(fft_bin_width * j as f32);
            let up = (mel_j - left) / (center - left);
            let down = (right - mel_j) / (right - center);
            weights[[i, j]] = up.min(down).max(0.0);
        }
        // Column `num_fft_bins` (the Nyquist bin) stays zero, matching Kaldi's
        // `F.pad(mel_energies, (0, 1))`.
    }
    weights
}

/// The voice encoder's mel front end: `n_fft=400`, `hop=160`, 40 mel bins,
/// `[0, 8000]` Hz, unscaled power (no log/dB, `mel_type="amp"`,
/// `normalized_mels=False`) -- matches
/// `chatterbox/models/voice_encoder/melspec.py::melspectrogram` exactly (with
/// `hp.preemphasis=0`, so no pre-emphasis step). Returns `(T, 40)`, time-major
/// (matching `melspectrogram(...).T`'s own convention, which
/// `voice_encoder.rs`'s partial-utterance striding consumes directly).
pub fn ve_mel_spectrogram(signal: &[f32]) -> Array2<f32> {
    const N_FFT: usize = 400;
    const HOP: usize = 160;
    const N_MELS: usize = 40;

    let spectrum = stft_complex(signal, N_FFT / 2, N_FFT, HOP);
    let power = spectrum.mapv(|c| c.norm_sqr());
    let filterbank = slaney_mel_filterbank(16_000.0, N_FFT, N_MELS, 0.0, 8_000.0);
    filterbank.dot(&power).reversed_axes()
}

/// The S3-tokenizer's log-mel front end: `n_fft=400`, `hop=160`, `n_mels` bins
/// (128 for the real tokenizer), `[0, 8000]` Hz, dropping the final STFT frame
/// (`stft[..., :-1]`) -- matches
/// `chatterbox/models/s3tokenizer/s3tokenizer.py::S3Tokenizer.log_mel_spectrogram`
/// exactly. Returns `(n_mels, T)`, channel-major (matching `s3tokenizer.onnx`'s
/// `mel` input shape directly).
pub fn whisper_log_mel(signal: &[f32], n_mels: usize) -> Array2<f32> {
    const N_FFT: usize = 400;
    const HOP: usize = 160;

    let spectrum = stft_complex(signal, N_FFT / 2, N_FFT, HOP);
    let num_frames = spectrum.shape()[1];
    let power = spectrum
        .slice(ndarray::s![.., ..num_frames.saturating_sub(1)])
        .mapv(|c| c.norm_sqr());
    let filterbank = slaney_mel_filterbank(16_000.0, N_FFT, n_mels, 0.0, 8_000.0);
    let mel = filterbank.dot(&power);

    let mut log_spec = mel.mapv(|v| v.max(1e-10).log10());
    let max = log_spec.iter().cloned().fold(f32::MIN, f32::max);
    log_spec.mapv_inplace(|v| v.max(max - 8.0));
    log_spec.mapv_inplace(|v| (v + 4.0) / 4.0);
    log_spec
}

/// S3Gen's natural-log mel front end (Matcha/CosyVoice convention): `n_fft=1920`,
/// `hop=480`, `win=1920`, 80 mel bins, `[0, 8000]` Hz, `center=False` with a
/// manual `(n_fft-hop)/2`-sample reflect pad, magnitude (not power) via
/// `sqrt(re^2+im^2+1e-9)`, natural-log dynamic-range compression clamped at
/// `1e-5` -- matches `chatterbox/models/s3gen/utils/mel.py::mel_spectrogram`
/// exactly. Returns `(T, 80)`, time-major (matching `prompt_feat`'s existing
/// channel-last convention throughout `s3gen.rs`/`pipeline.rs`).
pub fn s3gen_log_mel(signal: &[f32]) -> Array2<f32> {
    const N_FFT: usize = 1920;
    const HOP: usize = 480;
    const N_MELS: usize = 80;

    let spectrum = stft_complex(signal, (N_FFT - HOP) / 2, N_FFT, HOP);
    let magnitude = spectrum.mapv(|c| (c.norm_sqr() + 1e-9).sqrt());
    let filterbank = slaney_mel_filterbank(24_000.0, N_FFT, N_MELS, 0.0, 8_000.0);
    let mel = filterbank.dot(&magnitude);
    mel.mapv(|v| v.max(1e-5).ln()).reversed_axes()
}

/// CAMPPlus's Kaldi-style log-fbank front end: 25ms/10ms frames at 16kHz
/// (`window_size=400`, `window_shift=160`), zero-padded to the next power of
/// two (512) for the FFT, `snip_edges=True` framing (no signal padding -- only
/// whole frames), DC removal, `0.97` pre-emphasis (edge-replicated), Kaldi's
/// "povey" window, log-power-mel via [`kaldi_mel_filterbank`] -- matches
/// `torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=N)`'s defaults
/// exactly (`dither=0`, `low_freq=20`, `high_freq=0` i.e. Nyquist,
/// `preemphasis_coefficient=0.97`, `window_type="povey"`). Per-utterance mean
/// subtraction (`xvector.py::extract_feature`'s
/// `feature - feature.mean(dim=0)`) is the caller's job (`campplus.rs`), not
/// this function's -- `Kaldi.fbank` itself never mean-subtracts by default
/// (`subtract_mean=False`). Returns `(T, num_mel_bins)`, time-major (matching
/// `torchaudio`'s own convention).
pub fn kaldi_fbank(signal: &[f32], num_mel_bins: usize) -> Array2<f32> {
    const SAMPLE_RATE: f32 = 16_000.0;
    const FRAME_LENGTH_MS: f32 = 25.0;
    const FRAME_SHIFT_MS: f32 = 10.0;
    const PREEMPHASIS: f32 = 0.97;
    const LOW_FREQ: f32 = 20.0;
    const HIGH_FREQ: f32 = 0.0; // <= 0 means "offset from Nyquist" (here, exactly Nyquist)

    let window_size = (SAMPLE_RATE * FRAME_LENGTH_MS / 1000.0).round() as usize;
    let window_shift = (SAMPLE_RATE * FRAME_SHIFT_MS / 1000.0).round() as usize;
    let padded_window_size = window_size.next_power_of_two();

    let num_frames = if signal.len() < window_size {
        0
    } else {
        1 + (signal.len() - window_size) / window_shift
    };

    let window = povey_window(window_size);
    let filterbank = kaldi_mel_filterbank(
        num_mel_bins,
        padded_window_size,
        SAMPLE_RATE,
        LOW_FREQ,
        HIGH_FREQ,
    );

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(padded_window_size);
    let mut frame_buf = r2c.make_input_vec();
    let mut spectrum = r2c.make_output_vec();

    let mut out = Array2::<f32>::zeros((num_frames, num_mel_bins));
    for t in 0..num_frames {
        let start = t * window_shift;
        let raw = &signal[start..start + window_size];
        let mean = raw.iter().sum::<f32>() / window_size as f32;

        let mut prev = raw[0] - mean;
        frame_buf[0] = (prev - PREEMPHASIS * prev) * window[0];
        for i in 1..window_size {
            let centered = raw[i] - mean;
            frame_buf[i] = (centered - PREEMPHASIS * prev) * window[i];
            prev = centered;
        }
        for slot in frame_buf
            .iter_mut()
            .take(padded_window_size)
            .skip(window_size)
        {
            *slot = 0.0;
        }

        r2c.process(&mut frame_buf, &mut spectrum)
            .expect("fixed-size real FFT never fails on correctly-sized buffers");

        for c in 0..num_mel_bins {
            let energy: f32 = spectrum
                .iter()
                .enumerate()
                .map(|(f, bin)| bin.norm_sqr() * filterbank[[c, f]])
                .sum();
            out[[t, c]] = energy.max(f32::EPSILON).ln();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_tone(sr: f32, freq: f32, seconds: f32, amplitude: f32) -> Vec<f32> {
        let n = (sr * seconds) as usize;
        (0..n)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect()
    }

    #[test]
    fn hann_window_periodic_matches_known_values() {
        // torch.hann_window(8, periodic=True): [0, 0.1464, 0.5, 0.8536, 1.0, 0.8536, 0.5, 0.1464]
        let w = hann_window_periodic(8);
        let expected = [0.0, 0.1464, 0.5, 0.8536, 1.0, 0.8536, 0.5, 0.1464];
        for (a, b) in w.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn povey_window_goes_to_zero_at_edges_and_peaks_at_one() {
        let w = povey_window(9);
        assert!((w[0]).abs() < 1e-6);
        assert!((w[8]).abs() < 1e-6);
        assert!((w[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn slaney_mel_filterbank_matches_librosa_ve_config() {
        // librosa.filters.mel(sr=16000, n_fft=400, n_mels=40, fmin=0, fmax=8000)[0]
        // nonzero values: [0.0073902095, 0.012404521, 0.0050143115] at columns 1..4.
        let fb = slaney_mel_filterbank(16_000.0, 400, 40, 0.0, 8_000.0);
        let expected = [0.0073902095_f32, 0.012404521, 0.0050143115];
        for (j, &want) in expected.iter().enumerate() {
            assert!(
                (fb[[0, j + 1]] - want).abs() < 1e-5,
                "col {}: {} vs {}",
                j + 1,
                fb[[0, j + 1]],
                want
            );
        }
        assert!(fb[[0, 0]].abs() < 1e-9);
        assert!(fb[[0, 5]].abs() < 1e-9);
    }

    #[test]
    fn slaney_mel_filterbank_matches_librosa_s3gen_config() {
        // librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=8000)[0]
        // nonzero values start at column 1.
        let fb = slaney_mel_filterbank(24_000.0, 1920, 80, 0.0, 8_000.0);
        let expected = [
            0.009013824_f32,
            0.018027648,
            0.02666536,
            0.017651534,
            0.0086377105,
        ];
        for (j, &want) in expected.iter().enumerate() {
            assert!(
                (fb[[0, j + 1]] - want).abs() < 1e-4,
                "col {}: {} vs {}",
                j + 1,
                fb[[0, j + 1]],
                want
            );
        }
    }

    #[test]
    fn ve_mel_spectrogram_matches_python_reference_on_a_440hz_tone() {
        let sig = synthetic_tone(16_000.0, 440.0, 1.0, 0.5);
        let mel = ve_mel_spectrogram(&sig);
        assert_eq!(mel.dim(), (101, 40));
        // Python reference (melspectrogram(sig, hp).T)[5] / [50]: energy concentrated
        // in bins 4-6, everything else near zero.
        let expected_frame5 = [5.439_242_4_f32, 41.077_72, 4.454_809_7];
        for (j, &want) in expected_frame5.iter().enumerate() {
            let got = mel[[5, j + 4]];
            assert!(
                (got - want).abs() < 0.05,
                "bin {}: {} vs {}",
                j + 4,
                got,
                want
            );
        }
        assert!(mel[[5, 0]] < 1e-6);
    }

    #[test]
    fn whisper_log_mel_matches_python_reference_on_a_440hz_tone() {
        let sig = synthetic_tone(16_000.0, 440.0, 1.0, 0.5);
        let mel = whisper_log_mel(&sig, 128);
        assert_eq!(mel.dim(), (128, 100));
        // Python reference logmel[:, 5][16..21]: energetic bins amid a floor of -0.51464546.
        let expected = [1.3445051_f32, 1.337491, 1.4853542, 1.2752444, 1.2873932];
        for (j, &want) in expected.iter().enumerate() {
            let got = mel[[16 + j, 5]];
            assert!(
                (got - want).abs() < 0.01,
                "bin {}: {} vs {}",
                16 + j,
                got,
                want
            );
        }
        assert!((mel[[0, 5]] - (-0.51464546)).abs() < 1e-4);
    }

    #[test]
    fn s3gen_log_mel_matches_python_reference_on_a_440hz_tone() {
        let sig = synthetic_tone(24_000.0, 440.0, 1.0, 0.5);
        let mel = s3gen_log_mel(&sig);
        assert_eq!(mel.dim(), (50, 80));
        // Python reference s3gen_out[0, :, 5][8..12].
        let expected = [-4.9513745_f32, -3.3712943, 1.1285378, 2.2697933];
        for (j, &want) in expected.iter().enumerate() {
            let got = mel[[5, 8 + j]];
            assert!(
                (got - want).abs() < 0.05,
                "bin {}: {} vs {}",
                8 + j,
                got,
                want
            );
        }
    }

    #[test]
    fn kaldi_fbank_matches_python_reference_on_a_440hz_tone() {
        let sig = synthetic_tone(16_000.0, 440.0, 1.0, 0.5);
        let fb = kaldi_fbank(&sig, 80);
        assert_eq!(fb.dim(), (98, 80));
        // Python reference fbank[5][8..12].
        let expected = [-8.685935_f32, -7.6385646, -4.5738635, -2.6388];
        for (j, &want) in expected.iter().enumerate() {
            let got = fb[[5, 8 + j]];
            assert!(
                (got - want).abs() < 0.1,
                "bin {}: {} vs {}",
                8 + j,
                got,
                want
            );
        }
    }

    #[test]
    fn kaldi_fbank_returns_no_frames_for_signals_shorter_than_one_window() {
        let fb = kaldi_fbank(&[0.0_f32; 100], 80);
        assert_eq!(fb.dim(), (0, 80));
    }
}
