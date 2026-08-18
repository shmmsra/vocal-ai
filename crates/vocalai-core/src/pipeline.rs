//! Full pipeline orchestration: text -> tokenizer -> T3 -> S3Gen -> HiFiGAN ->
//! watermark -> PCM waveform. Wires together the per-component modules that were
//! built (and parity-checked) in earlier milestones; this module owns no DSP/loop
//! math of its own beyond simple tensor assembly already covered by each module's
//! own unit tests.
//!
//! Milestone 6, part B.1 (`docs/issues.md` VAI-006): the *default voice* path only
//! -- `models/default_voice/*.npy` (VAI-008's `export_default_voice.py`) already
//! contains everything T3/S3Gen need for conditioning, so this path needs no voice
//! encoder / S3-tokenizer / CAMPPlus / mel-extraction machinery at all. `--voice`
//! zero-shot cloning is part B.2 (not yet implemented) -- [`synthesize`] returns
//! [`PipelineError::VoiceCloningNotImplemented`] if a voice path is given.
//!
//! See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 6).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use rand::Rng;

use crate::{s3gen, session, t3, tokenizer, watermark};

/// Sampling/conditioning knobs, matching `ChatterboxTTS.generate()`'s CLI-exposed
/// parameters (plan §3).
#[derive(Clone, Debug)]
pub struct SynthesisParams {
    pub text: String,
    /// Reference-audio path for zero-shot cloning. `Some` is B.2 scope --
    /// [`synthesize`] errors out rather than silently using the default voice.
    pub voice: Option<PathBuf>,
    pub exaggeration: f32,
    pub cfg_weight: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub min_p: f32,
    pub top_p: f32,
    pub max_new_tokens: usize,
}

#[derive(Debug)]
pub enum PipelineError {
    Session(ort::Error),
    Npy(ndarray_npy::ReadNpyError),
    Tokenizer(Box<dyn std::error::Error + Send + Sync>),
    /// `--voice` was given but zero-shot cloning (Milestone 6, part B.2) isn't
    /// implemented yet.
    VoiceCloningNotImplemented,
    /// T3's decode loop produced no speech tokens after filtering (an immediate
    /// EOS) -- nothing to synthesize.
    NoSpeechGenerated,
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Session(e) => write!(f, "ONNX Runtime session error: {e}"),
            PipelineError::Npy(e) => write!(f, "failed to load a .npy model weight: {e}"),
            PipelineError::Tokenizer(e) => write!(f, "text tokenizer error: {e}"),
            PipelineError::VoiceCloningNotImplemented => write!(
                f,
                "--voice zero-shot cloning is not yet implemented (Milestone 6, part B.2) -- omit --voice to use the built-in default voice"
            ),
            PipelineError::NoSpeechGenerated => {
                write!(f, "T3 generated no speech tokens for this text")
            }
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<ort::Error> for PipelineError {
    fn from(e: ort::Error) -> Self {
        PipelineError::Session(e)
    }
}

impl From<ndarray_npy::ReadNpyError> for PipelineError {
    fn from(e: ndarray_npy::ReadNpyError) -> Self {
        PipelineError::Npy(e)
    }
}

/// The built-in default voice's conditioning tensors, dumped from `conds.pt` by
/// `export/export_default_voice.py` (VAI-008) -- see that script's module doc for
/// field provenance.
pub struct DefaultVoice {
    pub t3_speaker_emb: Array2<f32>,
    pub t3_cond_prompt_speech_tokens: Array2<i64>,
    pub s3gen_prompt_token: Array2<i64>,
    pub s3gen_prompt_token_len: Array1<i64>,
    pub s3gen_prompt_feat: Array3<f32>,
    pub s3gen_embedding: Array2<f32>,
}

impl DefaultVoice {
    fn load(dir: &Path) -> Result<Self, PipelineError> {
        Ok(Self {
            t3_speaker_emb: ndarray_npy::read_npy(dir.join("t3_speaker_emb.npy"))?,
            t3_cond_prompt_speech_tokens: ndarray_npy::read_npy(
                dir.join("t3_cond_prompt_speech_tokens.npy"),
            )?,
            s3gen_prompt_token: ndarray_npy::read_npy(dir.join("s3gen_prompt_token.npy"))?,
            s3gen_prompt_token_len: ndarray_npy::read_npy(dir.join("s3gen_prompt_token_len.npy"))?,
            s3gen_prompt_feat: ndarray_npy::read_npy(dir.join("s3gen_prompt_feat.npy"))?,
            s3gen_embedding: ndarray_npy::read_npy(dir.join("s3gen_embedding.npy"))?,
        })
    }
}

/// Owns every live ONNX session and loaded weight array the default-voice pipeline
/// needs. S3Gen's flow-encoder buckets (`TOKEN_BUCKETS`) are loaded lazily, one at a
/// time as real token counts demand them -- eagerly loading all six would hold
/// ~1.2GB of sessions never used in a single synthesis call.
pub struct ModelBundle {
    models_dir: PathBuf,
    pub tokenizer: tokenizer::TextTokenizer,
    pub t3_cond_prefill_session: Session,
    pub t3_decoder_session: Session,
    pub t3_speech_emb: Array2<f32>,
    pub t3_speech_pos_emb: Array2<f32>,
    flow_encoder_sessions: HashMap<usize, Session>,
    pub s3gen_estimator_session: Session,
    pub hifigan_session: Session,
    pub perthnet_session: Session,
    pub s3gen_spk_embed_affine_weight: Array2<f32>,
    pub s3gen_spk_embed_affine_bias: Array1<f32>,
    pub default_voice: DefaultVoice,
}

impl ModelBundle {
    pub fn load(models_dir: &Path) -> Result<Self, PipelineError> {
        let session_at = |name: &str| session::build_session(&models_dir.join(name));
        Ok(Self {
            models_dir: models_dir.to_path_buf(),
            tokenizer: tokenizer::TextTokenizer::from_file(&models_dir.join("tokenizer.json"))
                .map_err(PipelineError::Tokenizer)?,
            t3_cond_prefill_session: session_at("t3_cond_prefill.onnx")?,
            t3_decoder_session: session_at("t3_decoder.onnx")?,
            t3_speech_emb: t3::load_embedding_table(&models_dir.join("t3_speech_emb.npy"))?,
            t3_speech_pos_emb: t3::load_embedding_table(&models_dir.join("t3_speech_pos_emb.npy"))?,
            flow_encoder_sessions: HashMap::new(),
            s3gen_estimator_session: session_at("s3gen_estimator.onnx")?,
            hifigan_session: session_at("hifigan.onnx")?,
            perthnet_session: session_at("perthnet_encoder.onnx")?,
            s3gen_spk_embed_affine_weight: ndarray_npy::read_npy(
                models_dir.join("s3gen_spk_embed_affine_weight.npy"),
            )?,
            s3gen_spk_embed_affine_bias: ndarray_npy::read_npy(
                models_dir.join("s3gen_spk_embed_affine_bias.npy"),
            )?,
            default_voice: DefaultVoice::load(&models_dir.join("default_voice"))?,
        })
    }

    /// Ensures the flow-encoder session for `bucket` is loaded (from
    /// `models/s3gen_flow_encoder_{bucket}.onnx`, on first use), without borrowing
    /// the whole `ModelBundle` -- so callers can still separately borrow
    /// `s3gen_estimator_session`/`hifigan_session` afterwards. An associated
    /// function taking the two fields it needs directly, rather than a `&mut self`
    /// method, so the borrow checker sees disjoint field borrows at the call site.
    fn ensure_flow_encoder_session(
        sessions: &mut HashMap<usize, Session>,
        models_dir: &Path,
        bucket: usize,
    ) -> ort::Result<()> {
        if let std::collections::hash_map::Entry::Vacant(entry) = sessions.entry(bucket) {
            let path = models_dir.join(format!("s3gen_flow_encoder_{bucket}.onnx"));
            entry.insert(session::build_session(&path)?);
        }
        Ok(())
    }
}

/// Runs the full default-voice pipeline for `params.text`, returning a mono 24kHz
/// (`S3GEN_SR`) `f32` waveform in `[-1.0, 1.0]`, watermarked.
pub fn synthesize(
    bundle: &mut ModelBundle,
    params: &SynthesisParams,
    rng: &mut impl Rng,
) -> Result<Vec<f32>, PipelineError> {
    if params.voice.is_some() {
        return Err(PipelineError::VoiceCloningNotImplemented);
    }

    // 1. Text -> tokens. T3's decode loop (t3.rs, unchanged since VAI-004) always
    // indexes a 2-row (cond, uncond) batch, so the text-token batch is always
    // CFG-doubled here regardless of `cfg_weight` -- `cfg_weight` still controls the
    // actual CFG *strength* via `combine_cfg_logits`, and at `cfg_weight == 0.0` that
    // reduces exactly to the conditional branch, so this is not a behavior change,
    // just an unconditional extra (cheap) unconditional-branch forward pass.
    let normalized = tokenizer::punc_norm(&params.text);
    let ids = bundle
        .tokenizer
        .encode_ids(&normalized)
        .map_err(PipelineError::Tokenizer)?;
    let text_tokens = tokenizer::build_text_tokens(&ids, true);

    let cfg_uncond_mask = if params.cfg_weight > 0.0 {
        Array3::from_shape_vec((2, 1, 1), vec![1.0_f32, 0.0]).expect("fixed shape")
    } else {
        Array3::from_elem((2, 1, 1), 1.0_f32)
    };
    let emotion_adv = Array3::from_elem((1, 1, 1), params.exaggeration);

    // 2. T3 cond-prefill + KV-cache decode loop (default-voice conditioning).
    let cond_prefill_embeds = t3::run_cond_prefill(
        &mut bundle.t3_cond_prefill_session,
        &bundle.default_voice.t3_speaker_emb,
        &bundle.default_voice.t3_cond_prompt_speech_tokens,
        &emotion_adv,
        &text_tokens,
        &cfg_uncond_mask,
    )?;

    let sampling_config = t3::SamplingConfig {
        max_new_tokens: params.max_new_tokens,
        temperature: params.temperature,
        top_p: params.top_p,
        min_p: params.min_p,
        repetition_penalty: params.repetition_penalty,
        cfg_weight: params.cfg_weight,
    };

    let t3_decoder_session = &mut bundle.t3_decoder_session;
    let t3_speech_emb = &bundle.t3_speech_emb;
    let t3_speech_pos_emb = &bundle.t3_speech_pos_emb;
    let raw_tokens = t3::generate_speech_tokens(
        cond_prefill_embeds,
        &sampling_config,
        |embeds, past_kv| t3::run_decoder(t3_decoder_session, embeds, past_kv),
        |tok, pos| t3::embed_speech_token(t3_speech_emb, t3_speech_pos_emb, tok, pos),
        |logits| t3::sample_token(logits, rng),
    )?;

    let generated = t3::filter_valid_speech_tokens(&raw_tokens);
    if generated.is_empty() {
        return Err(PipelineError::NoSpeechGenerated);
    }

    // 3. Assemble S3Gen's token sequence: default voice's prompt_token + T3-generated
    // tokens (`flow.py`'s own `torch.concat`, done host-side here per ADR-0009).
    let prompt_token_len = bundle.default_voice.s3gen_prompt_token_len[0] as usize;
    let prompt_token: Vec<i64> = bundle
        .default_voice
        .s3gen_prompt_token
        .iter()
        .copied()
        .collect();

    let mut full_token = prompt_token.clone();
    full_token.extend(generated.iter().copied());

    let final_tokens = if s3gen::select_bucket(full_token.len()).is_some() {
        full_token
    } else {
        // Decision (approved plan): truncate rather than error when the generated
        // token count would need a bucket larger than TOKEN_BUCKETS's largest.
        let max_bucket = *s3gen::TOKEN_BUCKETS.last().expect("non-empty");
        let keep = max_bucket.saturating_sub(prompt_token_len);
        tracing::warn!(
            total_len = full_token.len(),
            max_bucket,
            "generated speech-token count exceeds the largest S3Gen flow-encoder bucket; truncating"
        );
        let mut truncated = prompt_token.clone();
        truncated.extend(generated.iter().take(keep).copied());
        truncated
    };
    let token_len = final_tokens.len();

    // 4. Speaker embedding (already-exported CAMPPlus output, normalize + affine).
    let spks = s3gen::embed_speaker(
        &bundle.default_voice.s3gen_embedding,
        &bundle.s3gen_spk_embed_affine_weight,
        &bundle.s3gen_spk_embed_affine_bias,
    );

    // 5. S3Gen: flow-encoder bucket call -> Euler ODE loop -> HiFiGAN.
    let bucket =
        s3gen::select_bucket(token_len).expect("truncated above to fit the largest bucket");
    ModelBundle::ensure_flow_encoder_session(
        &mut bundle.flow_encoder_sessions,
        &bundle.models_dir,
        bucket,
    )?;
    let flow_encoder_session = bundle
        .flow_encoder_sessions
        .get_mut(&bucket)
        .expect("just ensured");
    let waveform_2d = s3gen::token_to_waveform(
        flow_encoder_session,
        &mut bundle.s3gen_estimator_session,
        &mut bundle.hifigan_session,
        &final_tokens,
        prompt_token_len,
        &bundle.default_voice.s3gen_prompt_feat,
        &spks,
        rng,
    )?;
    let waveform: Vec<f32> = waveform_2d.row(0).to_vec();

    // 6. Watermark.
    let perthnet_session = &mut bundle.perthnet_session;
    let mut encoder_step =
        |magspec: &Array2<f32>| watermark::run_encoder(perthnet_session, magspec);
    let watermarked = watermark::apply_watermark(&waveform, &mut encoder_step)?;

    Ok(watermarked.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_error_display_is_human_readable() {
        assert!(PipelineError::VoiceCloningNotImplemented
            .to_string()
            .contains("not yet implemented"));
        assert!(PipelineError::NoSpeechGenerated
            .to_string()
            .contains("no speech tokens"));
    }
}
