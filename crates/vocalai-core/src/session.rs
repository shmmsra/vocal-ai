//! ONNX Runtime session setup: execution-provider selection.
//!
//! Hardware EPs (CoreML, CUDA) are always registered before CPU, per the hard
//! constraint in `CLAUDE.md` §1. Each EP is registered with `.fail_silently()`
//! (ORT's documented default, made explicit here so the fallback behavior is
//! visible in code rather than implied) — if a hardware EP fails to initialize,
//! `ort::session::builder::SessionBuilder` moves on to the next entry in the
//! list instead of erroring out.
//!
//! Milestone 6 (`docs/issues.md` VAI-006) adds [`build_session`], the live-session
//! half: it registers [`execution_providers`]'s list on a real `SessionBuilder` and
//! commits a model file. Detecting *which* EP a session actually used isn't exposed
//! as a queryable API by `ort` 2.0.0-rc.13 -- `ExecutionProviderDispatch::fail_silently`
//! (the hard constraint this module enforces) causes EP registration failures to be
//! swallowed at the Rust level, only surfacing via `ort`'s own internal `tracing`
//! calls (`ort::ep::apply_execution_providers`'s `crate::error!`/`crate::warning!` in
//! its source). [`build_session`] enables `ort`'s `tracing` Cargo feature (see
//! `Cargo.toml`) and installs a minimal `tracing-subscriber` (once, process-wide) so
//! those warnings actually reach stderr instead of being compiled out -- the
//! practical form "log any silent CPU fallback" can take without a queryable
//! provider-introspection API to build on.

#[cfg(feature = "coreml")]
use ort::ep::coreml::CoreML;
use ort::ep::cpu::CPU;
#[cfg(feature = "cuda")]
use ort::ep::cuda::CUDA;
use ort::ep::ExecutionProviderDispatch;
use ort::session::Session;
use std::path::Path;
use std::sync::Once;

/// Execution providers to register on a session, in fallback order:
/// hardware EPs first (as enabled by Cargo features), CPU always last.
// The number of `push`es is feature-dependent (0-2 hardware EPs + CPU), so a
// `vec![]` literal can't express it without duplicating the `#[cfg]` guards.
#[allow(clippy::vec_init_then_push)]
pub fn execution_providers() -> Vec<ExecutionProviderDispatch> {
    let mut eps = Vec::new();

    #[cfg(feature = "coreml")]
    eps.push(CoreML::default().build().fail_silently());

    #[cfg(feature = "cuda")]
    eps.push(CUDA::default().build().fail_silently());

    eps.push(CPU::default().build().fail_silently());

    eps
}

static INIT_LOGGING: Once = Once::new();

/// Installs a process-wide `tracing-subscriber` (stderr, `warn`-level by default,
/// overridable via `RUST_LOG`) exactly once, so `ort`'s internal EP-registration
/// warnings (see module docs) are actually visible instead of silently compiled out.
fn init_logging() {
    INIT_LOGGING.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
    });
}

/// Builds a live `ort::Session` from an ONNX model file, registering
/// [`execution_providers`]'s hardware-EPs-before-CPU fallback list. See module docs
/// for how (and the limits of how) a silent CPU fallback gets logged.
pub fn build_session(model_path: &Path) -> ort::Result<Session> {
    init_logging();
    Session::builder()?
        .with_execution_providers(execution_providers())?
        .commit_from_file(model_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_always_last() {
        let eps = execution_providers();
        assert!(!eps.is_empty());
        assert!(
            eps.last().unwrap().downcast_ref::<CPU>().is_some(),
            "CPU must be the last execution provider in the fallback list"
        );
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn coreml_precedes_cpu_when_enabled() {
        let eps = execution_providers();
        assert!(
            eps[0].downcast_ref::<CoreML>().is_some(),
            "CoreML must be registered before CPU when the `coreml` feature is enabled"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_precedes_cpu_when_enabled() {
        let eps = execution_providers();
        assert!(
            eps[0].downcast_ref::<CUDA>().is_some(),
            "CUDA must be registered before CPU when the `cuda` feature is enabled"
        );
    }

    #[test]
    fn default_build_is_cpu_only() {
        if cfg!(not(any(feature = "coreml", feature = "cuda"))) {
            assert_eq!(execution_providers().len(), 1);
        }
    }
}
