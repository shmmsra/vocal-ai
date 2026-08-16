//! ONNX Runtime session setup: execution-provider selection.
//!
//! Hardware EPs (CoreML, CUDA) are always registered before CPU, per the hard
//! constraint in `CLAUDE.md` §1. Each EP is registered with `.fail_silently()`
//! (ORT's documented default, made explicit here so the fallback behavior is
//! visible in code rather than implied) — if a hardware EP fails to initialize,
//! `ort::session::builder::SessionBuilder` moves on to the next entry in the
//! list instead of erroring out.
//!
//! This module only builds the ordered EP list; it does not yet create or run
//! a session. Detecting *which* EP a session actually used (to log a silent
//! CPU fallback) requires a live session against a loaded model, which lands
//! in Milestone 6 (`docs/issues.md` VAI-006) when the full pipeline is wired up.

#[cfg(feature = "coreml")]
use ort::ep::coreml::CoreML;
use ort::ep::cpu::CPU;
#[cfg(feature = "cuda")]
use ort::ep::cuda::CUDA;
use ort::ep::ExecutionProviderDispatch;

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
