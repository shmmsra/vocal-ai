//! `vocalai` CLI: standalone Chatterbox TTS, per docs/phase1-onnx-rust-cli-plan.md
//! §3. Milestone 6, part B.1 (`docs/issues.md` VAI-006): default-voice synthesis
//! only -- `--voice` zero-shot cloning is part B.2 (not yet implemented) and errors
//! out clearly rather than being silently ignored.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use vocalai_core::pipeline::{synthesize, ModelBundle, SynthesisParams};

/// Standalone Chatterbox TTS: `vocalai --text "hello world" --out out.wav`.
#[derive(Parser, Debug)]
#[command(name = "vocalai", version, about)]
struct Args {
    /// Text to synthesize.
    #[arg(long)]
    text: String,

    /// Reference audio for zero-shot voice cloning. Not yet implemented (Milestone
    /// 6, part B.2) -- omit to use the built-in default voice.
    #[arg(long)]
    voice: Option<PathBuf>,

    /// Emotion exaggeration, matching `ChatterboxTTS.generate(exaggeration=...)`.
    #[arg(long, default_value_t = 0.5)]
    exaggeration: f32,

    /// Classifier-free-guidance weight.
    #[arg(long = "cfg-weight", default_value_t = 0.5)]
    cfg_weight: f32,

    /// Sampling temperature.
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Repetition penalty applied to previously generated speech tokens.
    #[arg(long = "repetition-penalty", default_value_t = 1.2)]
    repetition_penalty: f32,

    /// Min-p sampling threshold.
    #[arg(long = "min-p", default_value_t = 0.05)]
    min_p: f32,

    /// Top-p (nucleus) sampling threshold.
    #[arg(long = "top-p", default_value_t = 1.0)]
    top_p: f32,

    /// Maximum number of speech tokens T3 may generate.
    #[arg(long = "max-new-tokens", default_value_t = 1000)]
    max_new_tokens: usize,

    /// Output WAV path.
    #[arg(long, default_value = "out.wav")]
    out: PathBuf,

    /// Directory containing the exported `.onnx`/`.npy` model files (see
    /// `export/`). Defaults to `./models`, matching that directory's own
    /// convention.
    #[arg(long = "models-dir", default_value = "models")]
    models_dir: PathBuf,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let mut bundle = match ModelBundle::load(&args.models_dir) {
        Ok(bundle) => bundle,
        Err(e) => {
            eprintln!(
                "error: failed to load models from {}: {e}",
                args.models_dir.display()
            );
            return ExitCode::FAILURE;
        }
    };

    let params = SynthesisParams {
        text: args.text,
        voice: args.voice,
        exaggeration: args.exaggeration,
        cfg_weight: args.cfg_weight,
        temperature: args.temperature,
        repetition_penalty: args.repetition_penalty,
        min_p: args.min_p,
        top_p: args.top_p,
        max_new_tokens: args.max_new_tokens,
    };

    let mut rng = rand::thread_rng();
    let waveform = match synthesize(&mut bundle, &params, &mut rng) {
        Ok(waveform) => waveform,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = vocalai_core::audio::write_wav(
        &args.out,
        &waveform,
        vocalai_core::watermark::PIPELINE_SAMPLE_RATE,
    ) {
        eprintln!("error: failed to write {}: {e}", args.out.display());
        return ExitCode::FAILURE;
    }

    println!("Wrote {}", args.out.display());
    ExitCode::SUCCESS
}
