//! ONNX Runtime session setup: execution-provider selection.
//!
//! Three selection modes ([`ExecutionProviderPreference`]), CLI-exposed as
//! `--use-gpu`/`--use-cpu`/neither (VAI-011, `docs/issues.md`):
//!
//! - `Gpu`: a hardware EP (CoreML/CUDA, gated by Cargo features) is registered
//!   with `.error_on_failure()` -- if it fails to initialize on this device (no
//!   capable hardware, missing driver, insufficient VRAM, ...), session creation
//!   returns a real `ort::Error` instead of silently continuing on CPU. `CPU` is
//!   never registered in this mode.
//! - `Cpu`: only `CPU` is registered. Hardware EPs are never attempted.
//! - `Auto` (default, neither flag passed): hardware EP(s) are tried the same way
//!   as `Gpu`, but a failure falls back to `Cpu` instead of erroring.
//!
//! [`resolve_and_build_session`] makes this decision once, against the *first*
//! model a caller loads, and returns the [`ResolvedProvider`] alongside the
//! session so the caller (`pipeline.rs::ModelBundle::load`) can reuse the same
//! decision for the rest of its ~9 session builds via [`build_session`], rather
//! than re-probing hardware availability on every single model file.
//!
//! Caveat (documented in `docs/decisions/0012-*.md`): registering (or not
//! registering) `CPUExecutionProvider` only controls whether CPU can be the
//! *primary/sole* provider. ONNX Runtime's CPU kernels are intrinsic to the
//! runtime and may still execute individual ops a hardware EP doesn't cover, even
//! in `Gpu` mode -- this module cannot guarantee literally zero CPU instructions,
//! only that CPU cannot be silently substituted as the *only* thing running.

#[cfg(feature = "coreml")]
use ort::ep::coreml::CoreML;
use ort::ep::cpu::CPU;
#[cfg(feature = "cuda")]
use ort::ep::cuda::CUDA;
use ort::ep::ExecutionProviderDispatch;
use ort::session::Session;
use std::path::Path;
use std::sync::Once;

/// Which execution provider(s) a caller wants `resolve_and_build_session` to try.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProviderPreference {
    /// Try hardware EP(s) first; fall back to CPU (logged) if none succeed.
    Auto,
    /// Require a hardware EP; error out rather than falling back to CPU.
    Gpu,
    /// Use CPU only; never attempt a hardware EP.
    Cpu,
}

/// Which execution provider a [`resolve_and_build_session`]/[`build_session`]
/// call actually used. Carries the hardware EP's name (e.g. `"CoreML"`) so
/// callers can log a specific, human-readable label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedProvider {
    Gpu(&'static str),
    Cpu,
}

impl std::fmt::Display for ResolvedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolvedProvider::Gpu(name) => write!(f, "GPU execution provider ({name})"),
            ResolvedProvider::Cpu => write!(f, "CPU execution provider"),
        }
    }
}

#[derive(Debug)]
pub enum SessionError {
    /// `Gpu` was requested but no hardware EP is usable: either none was
    /// compiled in (`coreml`/`cuda` Cargo features off, `None`) or every
    /// compiled-in candidate failed to register on this device (`Some`, the
    /// last candidate's error).
    GpuUnavailable(Option<ort::Error>),
    /// Any other ONNX Runtime session-build failure.
    Ort(ort::Error),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::GpuUnavailable(None) => write!(
                f,
                "--use-gpu requires a hardware execution provider, but this build has none \
                 compiled in (rebuild with --features coreml/cuda), or pass --use-cpu / omit \
                 --use-gpu to auto-select"
            ),
            SessionError::GpuUnavailable(Some(e)) => write!(
                f,
                "--use-gpu was passed but no compatible GPU execution provider could be \
                 initialized on this device: {e}; pass --use-cpu, or omit --use-gpu to auto-select"
            ),
            SessionError::Ort(e) => write!(f, "ONNX Runtime session error: {e}"),
        }
    }
}

impl std::error::Error for SessionError {}

impl From<ort::Error> for SessionError {
    fn from(e: ort::Error) -> Self {
        SessionError::Ort(e)
    }
}

/// Hardware EP candidates compiled into this binary (0-2, depending on Cargo
/// features), each registered with `.error_on_failure()` so a caller can detect
/// (rather than silently swallow) a failed hardware-EP registration attempt.
/// Paired with a human-readable name for [`ResolvedProvider::Gpu`] and for
/// re-finding the matching dispatch in [`build_session`].
#[allow(clippy::vec_init_then_push, unused_mut)]
fn hardware_execution_providers() -> Vec<(ExecutionProviderDispatch, &'static str)> {
    let mut eps = Vec::new();

    // VAI-011: T3's decode loop calls `session.run()` roughly once per generated
    // token (up to `--max-new-tokens`, e.g. ~1000) -- CoreML's default config pays
    // a fixed per-call dispatch/specialization cost that dominates for many tiny
    // sequential calls, making the naive default config measurably *slower* than
    // CPU (manually benchmarked: ~30-40% slower wall-clock, lower CPU utilization
    // consistent with time lost to dispatch/synchronization rather than compute).
    // These three tuned options fix that severe regression, bringing CoreML to
    // roughly tied-or-somewhat-worse than CPU -- NOT a demonstrated speed win (an
    // earlier reading of this benchmark claimed "parity or better"; a full-scale
    // run and the repo owner's own real-world re-test both contradicted that --
    // see `docs/decisions/0012-*.md`'s 2026-08-20 correction). Kept anyway because
    // it's still better than the naive default and `--use-gpu` stays opt-in:
    // `CPUAndGPU` (excludes the Neural Engine -- ANE's fixed per-call dispatch
    // latency is worse than GPU/Metal's for this many-tiny-calls pattern),
    // `FastPrediction` (trades extra one-time specialization cost at session-load
    // for lower per-call prediction latency -- exactly the right trade for a loop
    // that pays that cost once and calls `run()` ~1000 times), and
    // `RequireStaticInputShapes` (keeps CoreML from taking nodes whose shape
    // depends on the growing KV-cache, avoiding dynamic-shape handling overhead
    // inside the compiled program).
    #[cfg(feature = "coreml")]
    eps.push((
        CoreML::default()
            .with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndGPU)
            .with_specialization_strategy(ort::ep::coreml::SpecializationStrategy::FastPrediction)
            .with_static_input_shapes(true)
            .build()
            .error_on_failure(),
        "CoreML",
    ));

    #[cfg(feature = "cuda")]
    eps.push((CUDA::default().build().error_on_failure(), "CUDA"));

    eps
}

fn build_with_ep(model_path: &Path, ep: ExecutionProviderDispatch) -> ort::Result<Session> {
    Session::builder()?
        .with_execution_providers([ep])?
        .commit_from_file(model_path)
}

fn build_cpu_session(model_path: &Path) -> ort::Result<Session> {
    build_with_ep(model_path, CPU::default().build().fail_silently())
}

static INIT_LOGGING: Once = Once::new();

/// Installs a process-wide `tracing-subscriber` (stderr, `warn`-level by default,
/// overridable via `RUST_LOG`) exactly once, so `ort`'s internal EP-registration
/// warnings are visible instead of silently compiled out.
fn init_logging() {
    INIT_LOGGING.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
    });
}

/// Decides which execution provider to use (per `pref`) and builds `model_path`'s
/// session against that decision. Intended for the *first* model a caller loads;
/// pass the returned [`ResolvedProvider`] to [`build_session`] for every
/// subsequent model so the decision is made once, not once per file.
pub fn resolve_and_build_session(
    model_path: &Path,
    pref: ExecutionProviderPreference,
) -> Result<(Session, ResolvedProvider), SessionError> {
    init_logging();
    let resolved = match pref {
        ExecutionProviderPreference::Cpu => {
            let session = build_cpu_session(model_path)?;
            (session, ResolvedProvider::Cpu)
        }
        ExecutionProviderPreference::Gpu => {
            let candidates = hardware_execution_providers();
            if candidates.is_empty() {
                return Err(SessionError::GpuUnavailable(None));
            }
            let mut last_err = None;
            let mut resolved = None;
            for (ep, name) in candidates {
                match build_with_ep(model_path, ep) {
                    Ok(session) => {
                        resolved = Some((session, ResolvedProvider::Gpu(name)));
                        break;
                    }
                    Err(e) => last_err = Some(e),
                }
            }
            match resolved {
                Some(r) => r,
                None => return Err(SessionError::GpuUnavailable(last_err)),
            }
        }
        ExecutionProviderPreference::Auto => {
            let mut resolved = None;
            for (ep, name) in hardware_execution_providers() {
                if let Ok(session) = build_with_ep(model_path, ep) {
                    resolved = Some((session, ResolvedProvider::Gpu(name)));
                    break;
                }
            }
            match resolved {
                Some(r) => r,
                None => {
                    let session = build_cpu_session(model_path)?;
                    (session, ResolvedProvider::Cpu)
                }
            }
        }
    };
    tracing::info!(provider = %resolved.1, "selected execution provider");
    Ok(resolved)
}

/// Builds `model_path`'s session against an already-[`resolve_and_build_session`]-d
/// decision, without re-probing hardware availability.
pub fn build_session(
    model_path: &Path,
    resolved: ResolvedProvider,
) -> Result<Session, SessionError> {
    init_logging();
    match resolved {
        ResolvedProvider::Cpu => Ok(build_cpu_session(model_path)?),
        ResolvedProvider::Gpu(name) => {
            let ep = hardware_execution_providers()
                .into_iter()
                .find(|(_, n)| *n == name)
                .map(|(ep, _)| ep)
                .expect("resolved GPU provider name must match a compiled-in hardware EP");
            Ok(build_with_ep(model_path, ep)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardware_providers_empty_without_features() {
        if cfg!(not(any(feature = "coreml", feature = "cuda"))) {
            assert!(hardware_execution_providers().is_empty());
        }
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn coreml_is_a_hardware_candidate() {
        let eps = hardware_execution_providers();
        assert!(eps
            .iter()
            .any(|(ep, name)| *name == "CoreML" && ep.downcast_ref::<CoreML>().is_some()));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_is_a_hardware_candidate() {
        let eps = hardware_execution_providers();
        assert!(eps
            .iter()
            .any(|(ep, name)| *name == "CUDA" && ep.downcast_ref::<CUDA>().is_some()));
    }

    #[test]
    fn resolved_provider_display_labels() {
        assert_eq!(ResolvedProvider::Cpu.to_string(), "CPU execution provider");
        assert_eq!(
            ResolvedProvider::Gpu("CoreML").to_string(),
            "GPU execution provider (CoreML)"
        );
    }

    #[test]
    fn gpu_forced_without_any_hardware_feature_errors_before_touching_ort() {
        if cfg!(not(any(feature = "coreml", feature = "cuda"))) {
            let result = resolve_and_build_session(
                Path::new("/nonexistent/model.onnx"),
                ExecutionProviderPreference::Gpu,
            );
            assert!(matches!(result, Err(SessionError::GpuUnavailable(None))));
        }
    }
}
