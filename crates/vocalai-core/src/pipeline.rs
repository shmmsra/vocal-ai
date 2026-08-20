//! Full pipeline orchestration: text -> tokenizer -> T3 -> S3Gen -> HiFiGAN ->
//! watermark -> PCM waveform. Wires together the per-component modules that were
//! built (and parity-checked) in earlier milestones; this module owns no DSP/loop
//! math of its own beyond simple tensor assembly already covered by each module's
//! own unit tests.
//!
//! Milestone 6, part B.1 (`docs/issues.md` VAI-006): the *default voice* path --
//! `models/default_voice/*.npy` (VAI-008's `export_default_voice.py`) already
//! contains everything T3/S3Gen need for conditioning, so this path needs no voice
//! encoder / S3-tokenizer / CAMPPlus / mel-extraction machinery at all.
//!
//! Part B.2 adds `--voice` zero-shot cloning ([`VoiceConditioning::from_reference`]),
//! reimplementing `ChatterboxTTS.prepare_conditionals` + `S3Gen.embed_ref`
//! (`chatterbox/tts.py`, `chatterbox/models/s3gen/s3gen.py`) against the voice
//! encoder / S3-tokenizer / CAMPPlus ONNX sessions (all already exported and
//! parity-checked -- Milestones 2 and VAI-008) plus `mel.rs`'s hand-rolled DSP
//! front ends. Both voice paths produce the same [`VoiceConditioning`] shape, so
//! [`synthesize`]'s T3/S3Gen wiring downstream of voice selection is unchanged
//! from B.1.
//!
//! See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 6).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array3, Axis};
use ort::session::Session;
use rand::Rng;

use crate::{
    audio, campplus, mel, s3gen, s3tokenizer, session, t3, tokenizer, voice_encoder, watermark,
};

/// Sampling/conditioning knobs, matching `ChatterboxTTS.generate()`'s CLI-exposed
/// parameters (plan §3).
#[derive(Clone, Debug)]
pub struct SynthesisParams {
    pub text: String,
    /// Reference-audio (WAV) path for zero-shot voice cloning
    /// ([`VoiceConditioning::from_reference`]); `None` uses the built-in
    /// default voice.
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
    /// `--use-gpu` was passed but no hardware execution provider is usable
    /// (VAI-011, see `session::SessionError`).
    GpuUnavailable(session::SessionError),
    Npy(ndarray_npy::ReadNpyError),
    Tokenizer(Box<dyn std::error::Error + Send + Sync>),
    /// Failed to read a `--voice` reference WAV file.
    Audio(hound::Error),
    /// Failed to resample reference audio (`--voice`) to a rate a model expects.
    Resample(audio::ResampleError),
    /// T3's decode loop produced no speech tokens after filtering (an immediate
    /// EOS) -- nothing to synthesize.
    NoSpeechGenerated,
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Session(e) => write!(f, "ONNX Runtime session error: {e}"),
            PipelineError::GpuUnavailable(e) => write!(f, "{e}"),
            PipelineError::Npy(e) => write!(f, "failed to load a .npy model weight: {e}"),
            PipelineError::Tokenizer(e) => write!(f, "text tokenizer error: {e}"),
            PipelineError::Audio(e) => write!(f, "failed to read --voice reference audio: {e}"),
            PipelineError::Resample(e) => {
                write!(f, "failed to resample --voice reference audio: {e}")
            }
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

impl From<session::SessionError> for PipelineError {
    fn from(e: session::SessionError) -> Self {
        match e {
            session::SessionError::Ort(inner) => PipelineError::Session(inner),
            gpu_err @ session::SessionError::GpuUnavailable(_) => {
                PipelineError::GpuUnavailable(gpu_err)
            }
        }
    }
}

impl From<ndarray_npy::ReadNpyError> for PipelineError {
    fn from(e: ndarray_npy::ReadNpyError) -> Self {
        PipelineError::Npy(e)
    }
}

impl From<hound::Error> for PipelineError {
    fn from(e: hound::Error) -> Self {
        PipelineError::Audio(e)
    }
}

impl From<audio::ResampleError> for PipelineError {
    fn from(e: audio::ResampleError) -> Self {
        PipelineError::Resample(e)
    }
}

/// The conditioning tensors T3/S3Gen need, either loaded from the built-in
/// default voice ([`VoiceConditioning::load_default`], dumped from `conds.pt` by
/// `export/export_default_voice.py`, VAI-008) or computed live from a `--voice`
/// reference wav ([`VoiceConditioning::from_reference`], Milestone 6 part B.2).
#[derive(Clone)]
pub struct VoiceConditioning {
    pub t3_speaker_emb: Array2<f32>,
    pub t3_cond_prompt_speech_tokens: Array2<i64>,
    pub s3gen_prompt_token: Array2<i64>,
    pub s3gen_prompt_token_len: Array1<i64>,
    pub s3gen_prompt_feat: Array3<f32>,
    pub s3gen_embedding: Array2<f32>,
}

/// `S3_SR` (`chatterbox/models/s3tokenizer/s3tokenizer.py`) -- see
/// `s3tokenizer::S3_SR`'s doc for provenance; re-exported as a `usize` here for
/// the sample-count arithmetic below.
const S3_SR: usize = s3tokenizer::S3_SR as usize;
/// `S3GEN_SR` -- see `s3gen::S3GEN_SR`'s doc for provenance.
const S3GEN_SR: usize = s3gen::S3GEN_SR as usize;
/// `ChatterboxTTS.ENC_COND_LEN` (`chatterbox/tts.py`): T3's cond-prompt tokens
/// are computed from at most the first 6s of the (full-length) 16kHz resample.
const ENC_COND_LEN: usize = 6 * S3_SR;
/// `ChatterboxTTS.DEC_COND_LEN`: S3Gen's conditioning is computed from at most
/// the first 10s of the 24kHz reference.
const DEC_COND_LEN: usize = 10 * S3GEN_SR;

impl VoiceConditioning {
    fn load_default(dir: &Path) -> Result<Self, PipelineError> {
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

    /// Zero-shot voice cloning: builds the same conditioning tensors
    /// [`load_default`] loads from disk, but computed live from `wav_path`.
    /// Matches `ChatterboxTTS.prepare_conditionals` + `S3Gen.embed_ref`
    /// (`chatterbox/tts.py`, `chatterbox/models/s3gen/s3gen.py`) exactly:
    /// - `t3_speaker_emb`: the voice encoder's embedding of the *full-length*
    ///   16kHz resample (never truncated -- `embeds_from_wavs` runs on the
    ///   whole clip).
    /// - `t3_cond_prompt_speech_tokens`: the S3-tokenizer's tokens for at most
    ///   [`ENC_COND_LEN`] samples of that same 16kHz resample, truncated to
    ///   `speech_cond_prompt_len` (150) tokens.
    /// - `s3gen_prompt_feat`: S3Gen's own 24kHz mel of at most [`DEC_COND_LEN`]
    ///   samples of the reference.
    /// - `s3gen_prompt_token`/`s3gen_embedding`: the S3-tokenizer/CAMPPlus
    ///   outputs for that same (truncated-to-10s) clip resampled to 16kHz.
    fn from_reference(bundle: &mut ModelBundle, wav_path: &Path) -> Result<Self, PipelineError> {
        let (native_samples, native_sr) = audio::read_wav(wav_path)?;
        let ref_24k = audio::resample(&native_samples, native_sr, S3GEN_SR as u32)?;
        let ref_16k_full = audio::resample(&ref_24k, S3GEN_SR as u32, S3_SR as u32)?;

        let enc_cond_end = ref_16k_full.len().min(ENC_COND_LEN);
        let s3tok_session = bundle.ensure_s3tokenizer_session()?;
        let cond_tokens = s3tokenizer::tokenize(
            s3tok_session,
            &ref_16k_full[..enc_cond_end],
            Some(t3::SPEECH_COND_PROMPT_LEN),
        )?;
        let t3_cond_prompt_speech_tokens =
            Array2::from_shape_vec((1, cond_tokens.len()), cond_tokens).expect("row-vector shape");

        let ve_session = bundle.ensure_ve_session()?;
        let t3_speaker_emb = voice_encoder::compute_embedding(ve_session, &ref_16k_full)?;

        let dec_cond_end = ref_24k.len().min(DEC_COND_LEN);
        let s3gen_ref_24k = &ref_24k[..dec_cond_end];
        let prompt_feat_2d = mel::s3gen_log_mel(s3gen_ref_24k);
        let mel_len1 = prompt_feat_2d.shape()[0];
        let s3gen_prompt_feat = prompt_feat_2d.insert_axis(Axis(0));

        let s3gen_ref_16k = audio::resample(s3gen_ref_24k, S3GEN_SR as u32, S3_SR as u32)?;
        let s3tok_session = bundle.ensure_s3tokenizer_session()?;
        let mut prompt_token_vec = s3tokenizer::tokenize(s3tok_session, &s3gen_ref_16k, None)?;

        // `embed_ref`'s own consistency check: `ref_mels_24.shape[1]` must equal
        // `2 * ref_speech_tokens.shape[1]` (`token_mel_ratio=2`); truncate the
        // tokens to match if not, exactly like the Python reference.
        if mel_len1 != 2 * prompt_token_vec.len() {
            tracing::warn!(
                mel_len1,
                token_len = prompt_token_vec.len(),
                "reference mel length is not 2x token length; truncating tokens"
            );
            prompt_token_vec.truncate(mel_len1 / 2);
        }
        let prompt_token_len = prompt_token_vec.len();
        let s3gen_prompt_token = Array2::from_shape_vec((1, prompt_token_len), prompt_token_vec)
            .expect("row-vector shape");
        let s3gen_prompt_token_len = Array1::from_elem(1, prompt_token_len as i64);

        let campplus_session = bundle.ensure_campplus_session()?;
        let fbank = campplus::extract_features(&s3gen_ref_16k);
        let s3gen_embedding = campplus::run(campplus_session, &fbank)?;

        Ok(Self {
            t3_speaker_emb,
            t3_cond_prompt_speech_tokens,
            s3gen_prompt_token,
            s3gen_prompt_token_len,
            s3gen_prompt_feat,
            s3gen_embedding,
        })
    }
}

/// Owns every live ONNX session and loaded weight array the default-voice pipeline
/// needs. S3Gen's flow-encoder buckets (`TOKEN_BUCKETS`) are loaded lazily, one at a
/// time as real token counts demand them -- eagerly loading all six would hold
/// ~1.2GB of sessions never used in a single synthesis call.
pub struct ModelBundle {
    models_dir: PathBuf,
    /// The execution-provider decision made once, against the first session
    /// built in [`ModelBundle::load`] (VAI-011), and reused for every other
    /// session this bundle builds (including the lazily loaded ones below).
    resolved_provider: session::ResolvedProvider,
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
    pub default_voice: VoiceConditioning,
    /// Lazily loaded (Milestone 6, part B.2): only needed when `--voice` is
    /// given, so the default-voice-only fast path never pays their load cost.
    ve_session: Option<Session>,
    s3tokenizer_session: Option<Session>,
    campplus_session: Option<Session>,
}

impl ModelBundle {
    pub fn load(
        models_dir: &Path,
        ep_pref: session::ExecutionProviderPreference,
    ) -> Result<Self, PipelineError> {
        let (t3_cond_prefill_session, resolved_provider) =
            session::resolve_and_build_session(&models_dir.join("t3_cond_prefill.onnx"), ep_pref)?;
        let session_at =
            |name: &str| session::build_session(&models_dir.join(name), resolved_provider);

        // VAI-011 follow-up: `s3gen_estimator.onnx`'s Euler ODE loop runs on a
        // genuinely dynamic (non-bucketed) sequence length (`2 * token_len`)
        // through a UNet-style attention architecture -- confirmed (manually) to
        // reliably crash CoreML at inference time, unlike every other session in
        // this bundle (including T3's full decode loop), which run on CoreML fine.
        // Pinned to CPU whenever CoreML was resolved; CUDA is untested and left
        // alone. See `docs/issues.md` VAI-014 for the real fix (bucketing this
        // model's time dimension like the flow-encoder already is, ADR-0009).
        let estimator_provider = match resolved_provider {
            session::ResolvedProvider::Gpu("CoreML") => {
                tracing::warn!(
                    "forcing the S3Gen flow estimator to CPU (known CoreML incompatibility, \
                     VAI-014); every other session still uses {resolved_provider}"
                );
                session::ResolvedProvider::Cpu
            }
            other => other,
        };

        Ok(Self {
            models_dir: models_dir.to_path_buf(),
            resolved_provider,
            tokenizer: tokenizer::TextTokenizer::from_file(&models_dir.join("tokenizer.json"))
                .map_err(PipelineError::Tokenizer)?,
            t3_cond_prefill_session,
            t3_decoder_session: session_at("t3_decoder.onnx")?,
            t3_speech_emb: t3::load_embedding_table(&models_dir.join("t3_speech_emb.npy"))?,
            t3_speech_pos_emb: t3::load_embedding_table(&models_dir.join("t3_speech_pos_emb.npy"))?,
            flow_encoder_sessions: HashMap::new(),
            s3gen_estimator_session: session::build_session(
                &models_dir.join("s3gen_estimator.onnx"),
                estimator_provider,
            )?,
            hifigan_session: session_at("hifigan.onnx")?,
            perthnet_session: session_at("perthnet_encoder.onnx")?,
            s3gen_spk_embed_affine_weight: ndarray_npy::read_npy(
                models_dir.join("s3gen_spk_embed_affine_weight.npy"),
            )?,
            s3gen_spk_embed_affine_bias: ndarray_npy::read_npy(
                models_dir.join("s3gen_spk_embed_affine_bias.npy"),
            )?,
            default_voice: VoiceConditioning::load_default(&models_dir.join("default_voice"))?,
            ve_session: None,
            s3tokenizer_session: None,
            campplus_session: None,
        })
    }

    /// Lazily loads (on first `--voice` use) and returns `models/ve.onnx`'s session.
    fn ensure_ve_session(&mut self) -> Result<&mut Session, session::SessionError> {
        if self.ve_session.is_none() {
            self.ve_session = Some(session::build_session(
                &self.models_dir.join("ve.onnx"),
                self.resolved_provider,
            )?);
        }
        Ok(self.ve_session.as_mut().expect("just ensured"))
    }

    /// Lazily loads (on first `--voice` use) and returns `models/s3tokenizer.onnx`'s
    /// session -- shared by both T3's cond-prompt tokens and S3Gen's prompt token.
    fn ensure_s3tokenizer_session(&mut self) -> Result<&mut Session, session::SessionError> {
        if self.s3tokenizer_session.is_none() {
            self.s3tokenizer_session = Some(session::build_session(
                &self.models_dir.join("s3tokenizer.onnx"),
                self.resolved_provider,
            )?);
        }
        Ok(self.s3tokenizer_session.as_mut().expect("just ensured"))
    }

    /// Lazily loads (on first `--voice` use) and returns `models/campplus.onnx`'s
    /// session.
    fn ensure_campplus_session(&mut self) -> Result<&mut Session, session::SessionError> {
        if self.campplus_session.is_none() {
            self.campplus_session = Some(session::build_session(
                &self.models_dir.join("campplus.onnx"),
                self.resolved_provider,
            )?);
        }
        Ok(self.campplus_session.as_mut().expect("just ensured"))
    }

    /// The execution provider [`ModelBundle::load`] resolved (once) for this
    /// bundle's sessions -- for the CLI to report which one is in use.
    pub fn execution_provider(&self) -> session::ResolvedProvider {
        self.resolved_provider
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
        resolved_provider: session::ResolvedProvider,
    ) -> Result<(), session::SessionError> {
        if let std::collections::hash_map::Entry::Vacant(entry) = sessions.entry(bucket) {
            let path = models_dir.join(format!("s3gen_flow_encoder_{bucket}.onnx"));
            entry.insert(session::build_session(&path, resolved_provider)?);
        }
        Ok(())
    }
}

/// Runs the full pipeline for `params.text` against either the built-in default
/// voice or a `--voice` reference clip, returning a mono 24kHz (`S3GEN_SR`)
/// `f32` waveform in `[-1.0, 1.0]`, watermarked.
pub fn synthesize(
    bundle: &mut ModelBundle,
    params: &SynthesisParams,
    rng: &mut impl Rng,
) -> Result<Vec<f32>, PipelineError> {
    let voice = match &params.voice {
        Some(wav_path) => VoiceConditioning::from_reference(bundle, wav_path)?,
        None => bundle.default_voice.clone(),
    };

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
        &voice.t3_speaker_emb,
        &voice.t3_cond_prompt_speech_tokens,
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

    // 3. Assemble S3Gen's token sequence: the selected voice's prompt_token + T3-generated
    // tokens (`flow.py`'s own `torch.concat`, done host-side here per ADR-0009).
    let prompt_token_len = voice.s3gen_prompt_token_len[0] as usize;
    let prompt_token: Vec<i64> = voice.s3gen_prompt_token.iter().copied().collect();

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
        &voice.s3gen_embedding,
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
        bundle.resolved_provider,
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
        &voice.s3gen_prompt_feat,
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
        assert!(PipelineError::NoSpeechGenerated
            .to_string()
            .contains("no speech tokens"));
    }
}
