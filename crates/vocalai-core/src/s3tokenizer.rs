//! S3 (speech) tokenizer front end: `mel.rs`'s Whisper-style log-mel ->
//! `s3tokenizer.onnx`.
//!
//! Reimplements `S3Tokenizer.forward`'s integrated log-mel + quantize path
//! (`chatterbox/models/s3tokenizer/s3tokenizer.py`) against the exported
//! `s3tokenizer.onnx` graph (`export/export_s3tokenizer.py`, Milestone 2). One
//! shared session/function covers both real callers -- `chatterbox/tts.py`'s
//! T3 cond-prompt tokens (`max_tokens = Some(150)`, truncated) and
//! `S3Gen.embed_ref`'s prompt token (`max_tokens = None`, untruncated) both
//! ultimately call the same `S3Tokenizer.forward`/`.__call__`.
//!
//! See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 6, part B.2).

use ndarray::{s, Array1, Axis, Ix2};
use ort::session::Session;
use ort::value::Tensor;

use crate::mel;

/// `S3_SR` (`chatterbox/models/s3tokenizer/s3tokenizer.py`) -- the sample rate
/// every S3-tokenizer caller in this pipeline resamples reference audio to.
pub const S3_SR: u32 = 16_000;

/// Mel bins the real tokenizer (`speech_tokenizer_v2_25hz`) was configured
/// with (`ModelConfig().n_mels`, `export_s3tokenizer.py::N_MELS`).
const N_MELS: usize = 128;

/// Tokenizes `signal_16k` (mono, 16kHz) via `s3tokenizer.onnx`. `max_tokens`,
/// if given, truncates the mel input to `max_tokens * 4` frames first
/// (`S3Tokenizer.forward`'s own `mel[..., :max_len * 4]` -- S3 runs at 25
/// tokens/sec against a 100-frame/sec mel) -- used for T3's cond-prompt tokens
/// (`speech_cond_prompt_len`); `None` tokenizes the full clip (S3Gen's prompt
/// token). Returns the token sequence sliced to the model's own reported
/// `code_len` (always the full length for our always-unpadded, batch-of-one
/// calls, but sliced defensively rather than assumed).
pub fn tokenize(
    session: &mut Session,
    signal_16k: &[f32],
    max_tokens: Option<usize>,
) -> ort::Result<Vec<i64>> {
    let mut mel_spec = mel::whisper_log_mel(signal_16k, N_MELS);
    if let Some(max_tok) = max_tokens {
        let keep = (max_tok * 4).min(mel_spec.shape()[1]);
        mel_spec = mel_spec.slice(s![.., ..keep]).to_owned();
    }
    let mel_len = mel_spec.shape()[1] as i64;
    let batched = mel_spec.insert_axis(Axis(0));

    let outputs = session.run(ort::inputs![
        "mel" => Tensor::from_array(batched)?,
        "mel_len" => Tensor::from_array(Array1::from_elem(1, mel_len))?,
    ])?;
    // `code` is traced as int32 (`quantizer.encode`'s own output dtype) while
    // `code_len` stays int64 (like `mel_len`'s input dtype) -- confirmed via the
    // ONNX graph's declared output types, not assumed.
    let code = outputs["code"]
        .try_extract_array::<i32>()?
        .into_dimensionality::<Ix2>()
        .expect("code is always rank-2")
        .to_owned();
    let code_len = outputs["code_len"].try_extract_array::<i64>()?;
    let valid_len = code_len.iter().next().copied().unwrap_or(0).max(0) as usize;

    Ok(code
        .row(0)
        .iter()
        .map(|&t| t as i64)
        .take(valid_len)
        .collect())
}
