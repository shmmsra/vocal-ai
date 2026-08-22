//! `vocalai` CLI: standalone Chatterbox TTS, per docs/phase1-onnx-rust-cli-plan.md
//! §3. Supports both the built-in default voice and `--voice` zero-shot cloning
//! from a WAV reference clip (Milestone 6, part B, `docs/issues.md` VAI-006).
//!
//! `--use-gpu`/`--use-cpu` (`docs/issues.md` VAI-011) select the execution
//! provider; neither flag defaults to CPU (see `docs/decisions/0012-*.md` --
//! CoreML's naive default config measured slower than CPU for T3's
//! many-tiny-sequential-calls decode loop; `--use-gpu` applies a tuned CoreML
//! config that fixes that severe regression, but is not a demonstrated speed
//! win over CPU -- see the ADR's 2026-08-20 correction -- so it stays opt-in).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use vocalai_core::pipeline::{
    synthesize, ModelBundle, PipelinePhase, ProgressEvent, SynthesisParams,
};
use vocalai_core::session::ExecutionProviderPreference;

/// Standalone Chatterbox TTS: `vocalai --text "hello world" --out out.wav`.
#[derive(Parser, Debug)]
#[command(name = "vocalai", version, about)]
struct Args {
    /// Text to synthesize.
    #[arg(long)]
    text: String,

    /// Reference audio (WAV) for zero-shot voice cloning. Omit to use the
    /// built-in default voice.
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

    /// Require a hardware execution provider (CoreML/CUDA); error out rather
    /// than falling back to CPU if none is usable on this device.
    #[arg(long = "use-gpu", action = clap::ArgAction::SetTrue, conflicts_with = "use_cpu")]
    use_gpu: bool,

    /// Use the CPU execution provider only; never attempt a hardware EP.
    #[arg(long = "use-cpu", action = clap::ArgAction::SetTrue, conflicts_with = "use_gpu")]
    use_cpu: bool,

    /// Print phase labels and a decode-loop progress bar to stderr while
    /// synthesizing (default off, no output change without it -- VAI-012).
    #[arg(long = "show-progress", action = clap::ArgAction::SetTrue)]
    show_progress: bool,
}

fn phase_label(phase: PipelinePhase) -> &'static str {
    match phase {
        PipelinePhase::VoiceConditioning => "Preparing voice conditioning...",
        PipelinePhase::Decoding => "Decoding speech tokens...",
        PipelinePhase::Vocoding => "Vocoding...",
        PipelinePhase::Watermarking => "Watermarking...",
    }
}

fn main() -> ExitCode {
    let args = Args::parse();

    // Default (also `--use-cpu`, currently identical): CPU. VAI-011 originally
    // defaulted to `Auto` (try a hardware EP, fall back to CPU); switched to
    // CPU-by-default after benchmarking found CoreML's naive config measurably
    // slower than CPU for T3's decode loop -- see `docs/decisions/0012-*.md`.
    // `--use-gpu` applies a tuned CoreML config that fixes that severe
    // regression, but is NOT a demonstrated speed win over CPU (an earlier
    // "CPU parity or better" reading of the benchmark didn't hold up -- see the
    // ADR's 2026-08-20 correction), so it stays opt-in. `--use-cpu` stays an
    // explicit flag (rather than folding it away) so it keeps meaning something
    // if a future change reintroduces a non-CPU default.
    let ep_pref = match (args.use_gpu, args.use_cpu) {
        (true, _) => ExecutionProviderPreference::Gpu,
        (false, _) => ExecutionProviderPreference::Cpu,
    };

    let mut bundle = match ModelBundle::load(&args.models_dir, ep_pref) {
        Ok(bundle) => bundle,
        Err(e) => {
            eprintln!(
                "error: failed to load models from {}: {e}",
                args.models_dir.display()
            );
            return ExitCode::FAILURE;
        }
    };
    eprintln!("Using {}", bundle.execution_provider());

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

    let mut noop = |_event: ProgressEvent| {};
    let mut decode_bar: Option<ProgressBar> = None;
    let mut render = |event: ProgressEvent| match event {
        ProgressEvent::Phase(phase) => {
            if let Some(bar) = decode_bar.take() {
                bar.finish_and_clear();
            }
            if phase == PipelinePhase::Decoding {
                let bar = ProgressBar::new(args.max_new_tokens as u64);
                bar.set_style(
                    ProgressStyle::with_template("{msg} [{bar:40}] {pos}/{len}")
                        .expect("static template")
                        .progress_chars("=> "),
                );
                bar.set_message(phase_label(phase));
                decode_bar = Some(bar);
            } else {
                eprintln!("==> {}", phase_label(phase));
            }
        }
        ProgressEvent::DecodeStep { step, max } => {
            if let Some(bar) = &decode_bar {
                bar.set_position(step.min(max) as u64);
            }
        }
    };
    let on_progress: &mut dyn FnMut(ProgressEvent) = if args.show_progress {
        &mut render
    } else {
        &mut noop
    };

    let waveform = match synthesize(&mut bundle, &params, &mut rng, on_progress) {
        Ok(waveform) => waveform,
        Err(e) => {
            if let Some(bar) = decode_bar.take() {
                bar.finish_and_clear();
            }
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
