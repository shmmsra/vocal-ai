//! CAMPPlus speaker-embedding front end: `mel.rs`'s Kaldi-style fbank -> a
//! fixed [`CAMPPLUS_FRAMES`]-frame window -> `campplus.onnx`.
//!
//! See `export/export_campplus.py` and
//! `docs/decisions/0009-s3gen-flow-encoder-and-campplus-export.md` for why
//! CAMPPlus is exported at one fixed frame count with no length/mask input at
//! all, and why Rust must always feed exactly that many frames of *real*
//! content (trim a longer clip, cyclically repeat a shorter one -- never
//! zero-pad, which would corrupt `StatsPool`'s statistics with silent frames).
//!
//! See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 6, part B.2).

use ndarray::{s, Array2, Axis, Ix2};
use ort::session::Session;
use ort::value::Tensor;

use crate::mel;

/// `export_campplus.py::FEAT_DIM`.
pub const FEAT_DIM: usize = 80;
/// `export_campplus.py::CAMPPLUS_FRAMES` -- the fixed frame count
/// `campplus.onnx` was traced at.
pub const CAMPPLUS_FRAMES: usize = 400;

/// Kaldi-style fbank, mean-subtracted over the *real* utterance length
/// (matching `xvector.py::extract_feature`'s `feature - feature.mean(dim=0)`,
/// computed before any trim/repeat below -- the mean-subtraction reference
/// behavior operates over whatever length the eager model was given, so this
/// mirrors that rather than mean-subtracting only the fitted window), then
/// trimmed or cyclically repeated to exactly [`CAMPPLUS_FRAMES`] frames.
pub fn extract_features(signal_16k: &[f32]) -> Array2<f32> {
    let fbank = mel::kaldi_fbank(signal_16k, FEAT_DIM);
    let centered = if fbank.shape()[0] == 0 {
        fbank
    } else {
        let mean = fbank.mean_axis(Axis(0)).expect("non-empty fbank");
        &fbank - &mean
    };
    fit_to_frame_count(centered, CAMPPLUS_FRAMES)
}

/// Trims `fbank` (time-major, `(T, FEAT_DIM)`) to `frames` rows if `T >=
/// frames`, or cyclically repeats its rows to reach `frames` if `T < frames`.
/// Never zero-pads (see module docs).
fn fit_to_frame_count(fbank: Array2<f32>, frames: usize) -> Array2<f32> {
    let (t, d) = fbank.dim();
    if t == 0 {
        return Array2::zeros((frames, d));
    }
    if t >= frames {
        return fbank.slice(s![..frames, ..]).to_owned();
    }
    let mut out = Array2::<f32>::zeros((frames, d));
    for i in 0..frames {
        out.row_mut(i).assign(&fbank.row(i % t));
    }
    out
}

/// Runs `campplus.onnx` on a `(CAMPPLUS_FRAMES, FEAT_DIM)` fbank window,
/// returning the raw (pre-normalize, pre-affine) 192-dim x-vector `(1, 192)`.
/// Normalize + `spk_embed_affine_layer` are `s3gen::embed_speaker`'s job
/// (ADR-0009), not this function's.
pub fn run(session: &mut Session, fbank: &Array2<f32>) -> ort::Result<Array2<f32>> {
    let batched = fbank.clone().insert_axis(Axis(0));
    let outputs = session.run(ort::inputs!["fbank" => Tensor::from_array(batched)?])?;
    let embedding = outputs["embedding"].try_extract_array::<f32>()?;
    Ok(embedding
        .into_dimensionality::<Ix2>()
        .expect("embedding is always rank-2")
        .to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_to_frame_count_trims_longer_input() {
        let fbank = Array2::<f32>::from_shape_fn((500, 80), |(t, _)| t as f32);
        let fitted = fit_to_frame_count(fbank, CAMPPLUS_FRAMES);
        assert_eq!(fitted.dim(), (CAMPPLUS_FRAMES, 80));
        assert_eq!(fitted[[0, 0]], 0.0);
        assert_eq!(fitted[[399, 0]], 399.0);
    }

    #[test]
    fn fit_to_frame_count_cyclically_repeats_shorter_input() {
        let fbank = Array2::<f32>::from_shape_fn((150, 80), |(t, _)| t as f32);
        let fitted = fit_to_frame_count(fbank, CAMPPLUS_FRAMES);
        assert_eq!(fitted.dim(), (CAMPPLUS_FRAMES, 80));
        assert_eq!(fitted[[0, 0]], 0.0);
        assert_eq!(fitted[[149, 0]], 149.0);
        assert_eq!(fitted[[150, 0]], 0.0); // wraps back to row 0
        assert_eq!(fitted[[399, 0]], 99.0); // 399 % 150 = 99
    }

    #[test]
    fn fit_to_frame_count_returns_zeros_for_empty_input() {
        let fbank = Array2::<f32>::zeros((0, 80));
        let fitted = fit_to_frame_count(fbank, CAMPPLUS_FRAMES);
        assert_eq!(fitted.dim(), (CAMPPLUS_FRAMES, 80));
        assert!(fitted.iter().all(|&v| v == 0.0));
    }
}
