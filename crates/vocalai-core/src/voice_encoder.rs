//! Voice-encoder speaker embedding: silence trim -> mel -> overlapping
//! partial-utterance windows -> `ve.onnx` -> mean + L2-normalize.
//!
//! Reimplements `VoiceEncoder.embeds_from_wavs`/`.inference`
//! (`chatterbox/models/voice_encoder/voice_encoder.py`) against the exported
//! `ve.onnx` graph (`export/export_ve.py`, Milestone 2). [`trim_silence`] ports
//! `librosa.effects.trim`'s frame-RMS-vs-`top_db` decision
//! (`librosa/effects.py::_signal_to_frame_nonsilent`); the partial-utterance
//! striding ([`frame_step`]/[`num_wins`]/[`stride_partials`]) ports
//! `get_frame_step`/`get_num_wins`/`stride_as_partials`. Neither is
//! ONNX-exported, so (like `mel.rs`) correctness rests on unit tests, not an
//! automated parity gate -- see `mel.rs`'s module doc for the shared rationale.
//!
//! See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 6, part B.2).

use ndarray::{s, Array2, Array3, Axis, Ix2};
use ort::session::Session;
use ort::value::Tensor;

use crate::mel;

/// `VoiceEncConfig.sample_rate` -- the voice encoder always operates at 16kHz.
const SAMPLE_RATE: f32 = 16_000.0;
/// `VoiceEncConfig.ve_partial_frames`.
const VE_PARTIAL_FRAMES: usize = 160;
/// `chatterbox/tts.py::ChatterboxTTS.prepare_conditionals`'s call never
/// overrides `embeds_from_wavs`'s `trim_top_db` default.
const TRIM_TOP_DB: f32 = 20.0;
/// Resemble's own default (`embeds_from_wavs`'s `kwargs["rate"] = 1.3`).
const PARTIAL_RATE: f32 = 1.3;
/// `stride_as_partials`'s default.
const MIN_COVERAGE: f32 = 0.8;

/// Port of `librosa.effects.trim(y, top_db=top_db, frame_length=2048,
/// hop_length=512, ref=np.max)`, mono-only (this pipeline never calls it on
/// multi-channel audio, so `aggregate`'s cross-channel handling is omitted).
/// Silence is any frame whose RMS is more than `top_db` below the signal's
/// loudest frame; returns the sub-slice of `signal` spanning the first through
/// last non-silent frame (or an empty slice if the whole signal is silent).
pub fn trim_silence(signal: &[f32], top_db: f32) -> &[f32] {
    const FRAME_LENGTH: usize = 2048;
    const HOP_LENGTH: usize = 512;
    const AMIN: f32 = 1e-5;

    let pad = FRAME_LENGTH / 2;
    let mut padded = vec![0.0_f32; pad];
    padded.extend_from_slice(signal);
    padded.extend(std::iter::repeat_n(0.0_f32, pad));

    if padded.len() < FRAME_LENGTH {
        return signal;
    }
    let num_frames = 1 + (padded.len() - FRAME_LENGTH) / HOP_LENGTH;

    let rms: Vec<f32> = (0..num_frames)
        .map(|t| {
            let start = t * HOP_LENGTH;
            let frame = &padded[start..start + FRAME_LENGTH];
            (frame.iter().map(|v| v * v).sum::<f32>() / FRAME_LENGTH as f32).sqrt()
        })
        .collect();
    let max_rms = rms.iter().cloned().fold(0.0_f32, f32::max).max(AMIN);
    let ref_db = 20.0 * max_rms.log10();
    let non_silent: Vec<bool> = rms
        .iter()
        .map(|&r| 20.0 * r.max(AMIN).log10() - ref_db > -top_db)
        .collect();

    match (
        non_silent.iter().position(|&b| b),
        non_silent.iter().rposition(|&b| b),
    ) {
        (Some(first), Some(last)) => {
            let start = first * HOP_LENGTH;
            let end = signal.len().min((last + 1) * HOP_LENGTH);
            &signal[start..end]
        }
        _ => &signal[0..0],
    }
}

/// `get_frame_step`: how many mel frames separate two partial utterances, given
/// `rate` (partials/sec).
fn frame_step(rate: f32) -> usize {
    ((SAMPLE_RATE / rate) / VE_PARTIAL_FRAMES as f32).round() as usize
}

/// `get_num_wins`: how many overlapping `VE_PARTIAL_FRAMES`-length windows fit
/// in `n_frames` mel frames at the given `step`, and the total (possibly
/// zero-padded) length they span.
fn num_wins(n_frames: usize, step: usize, min_coverage: f32) -> (usize, usize) {
    let win_size = VE_PARTIAL_FRAMES;
    let base = (n_frames as isize - win_size as isize + step as isize).max(0) as usize;
    let mut n_wins = base / step;
    let remainder = base % step;
    if n_wins == 0
        || (remainder as f32 + (win_size as f32 - step as f32)) / win_size as f32 >= min_coverage
    {
        n_wins += 1;
    }
    let target_n = win_size + step * (n_wins - 1);
    (n_wins, target_n)
}

/// `stride_as_partials`: zero-pads/trims `mel` (time-major, `(T, num_mels)`) to
/// the target length [`num_wins`] computes, then slices it into overlapping
/// `(VE_PARTIAL_FRAMES, num_mels)` windows.
fn stride_partials(mel_spec: &Array2<f32>, step: usize, min_coverage: f32) -> Array3<f32> {
    let (n_frames, n_mels) = mel_spec.dim();
    let (n_partials, target_len) = num_wins(n_frames, step, min_coverage);

    let fitted = if target_len > n_frames {
        let mut extended = Array2::<f32>::zeros((target_len, n_mels));
        extended.slice_mut(s![..n_frames, ..]).assign(mel_spec);
        extended
    } else if target_len < n_frames {
        mel_spec.slice(s![..target_len, ..]).to_owned()
    } else {
        mel_spec.clone()
    };

    let mut partials = Array3::<f32>::zeros((n_partials, VE_PARTIAL_FRAMES, n_mels));
    for i in 0..n_partials {
        let start = i * step;
        partials
            .slice_mut(s![i, .., ..])
            .assign(&fitted.slice(s![start..start + VE_PARTIAL_FRAMES, ..]));
    }
    partials
}

/// Runs `ve.onnx` on a batch of partial-utterance mel windows
/// `(N, VE_PARTIAL_FRAMES, num_mels)`, returning their (already L2-normalized,
/// per `VoiceEncoder.forward`) embeddings `(N, 256)`.
pub fn run_ve(session: &mut Session, partials: &Array3<f32>) -> ort::Result<Array2<f32>> {
    let outputs = session.run(ort::inputs!["mels" => Tensor::from_array(partials.clone())?])?;
    let embeds = outputs["speaker_embedding"].try_extract_array::<f32>()?;
    Ok(embeds
        .into_dimensionality::<Ix2>()
        .expect("speaker_embedding is always rank-2")
        .to_owned())
}

/// Full speaker-embedding pipeline for a 16kHz reference clip: trim silence,
/// compute the unscaled power-mel, stride into overlapping partial-utterance
/// windows, run `ve.onnx`, then mean the (already-normalized) partial embeds
/// and L2-normalize the mean -- matching `VoiceEncoder.inference` exactly.
/// Returns `(1, 256)`.
pub fn compute_embedding(session: &mut Session, signal_16k: &[f32]) -> ort::Result<Array2<f32>> {
    let trimmed = trim_silence(signal_16k, TRIM_TOP_DB);
    let mel_spec = mel::ve_mel_spectrogram(trimmed);
    let step = frame_step(PARTIAL_RATE);
    let partials = stride_partials(&mel_spec, step, MIN_COVERAGE);
    let partial_embeds = run_ve(session, &partials)?;

    let mean = partial_embeds
        .mean_axis(Axis(0))
        .expect("run_ve always returns at least one partial");
    let norm = mean.mapv(|v| v * v).sum().sqrt();
    Ok((mean / norm).insert_axis(Axis(0)))
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
    fn trim_silence_matches_python_reference_indices() {
        // librosa.effects.trim(sig_padded, top_db=20) where sig_padded is 4000
        // zeros + a 1s 440Hz*0.5 tone (16kHz) + 4000 zeros -> index [3072, 21504].
        let tone = synthetic_tone(16_000.0, 440.0, 1.0, 0.5);
        let mut padded = vec![0.0_f32; 4000];
        padded.extend_from_slice(&tone);
        padded.extend(std::iter::repeat_n(0.0_f32, 4000));

        let trimmed = trim_silence(&padded, 20.0);
        // Recover the observed slice bounds by pointer arithmetic against `padded`.
        let start = trimmed.as_ptr() as usize - padded.as_ptr() as usize;
        let start = start / std::mem::size_of::<f32>();
        assert_eq!(start, 3072);
        assert_eq!(start + trimmed.len(), 21504);
    }

    #[test]
    fn trim_silence_leaves_a_fully_silent_signal_untrimmed() {
        // Matches librosa.effects.trim's documented behavior: with the default
        // `ref=np.max`, a uniformly (exactly) silent signal has no frame quieter
        // than the reference, so nothing is trimmed.
        let signal = vec![0.0_f32; 8000];
        let trimmed = trim_silence(&signal, 20.0);
        assert_eq!(trimmed.len(), signal.len());
    }

    #[test]
    fn frame_step_matches_hand_computed_value() {
        // (16000/1.3)/160 = 76.923... -> rounds to 77.
        assert_eq!(frame_step(1.3), 77);
    }

    #[test]
    fn num_wins_covers_a_single_partial_worth_of_frames() {
        let (n_wins, target_len) = num_wins(160, 77, 0.8);
        assert_eq!(n_wins, 1);
        assert_eq!(target_len, 160);
    }

    #[test]
    fn num_wins_matches_hand_computed_value_for_multiple_windows() {
        // n_frames=300, step=77, win_size=160: base=300-160+77=217; n_wins=217/77=2
        // (remainder=63); coverage=(63+(160-77))/160=146/160=0.9125>=0.8 -> +1 -> 3.
        // target_len = 160 + 77*(3-1) = 314.
        let (n_wins, target_len) = num_wins(300, 77, 0.8);
        assert_eq!(n_wins, 3);
        assert_eq!(target_len, 314);
    }

    #[test]
    fn stride_partials_produces_the_requested_window_shape() {
        let mel_spec = Array2::<f32>::zeros((160, 40));
        let partials = stride_partials(&mel_spec, 77, 0.8);
        assert_eq!(partials.dim(), (1, 160, 40));
    }

    #[test]
    fn stride_partials_zero_pads_short_input_to_one_full_window() {
        let mel_spec = Array2::<f32>::from_elem((50, 40), 1.0_f32);
        let partials = stride_partials(&mel_spec, 77, 0.8);
        assert_eq!(partials.dim(), (1, 160, 40));
        assert_eq!(partials[[0, 0, 0]], 1.0);
        assert_eq!(partials[[0, 159, 0]], 0.0);
    }
}
