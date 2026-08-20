# ADR-0012: Execution-provider selection (`--use-gpu`/`--use-cpu`, CPU by default), tuning CoreML for T3's decode loop, and pinning `s3gen_estimator.onnx` to CPU

**Date**: 2026-08-19
**Status**: Accepted
**Decider**: repo owner + Claude (session 2026-08-19)

## Context

Before this change, `crates/vocalai-core/src/session.rs::execution_providers()` always registered
hardware EPs (CoreML/CUDA, gated by Cargo features) with `.fail_silently()`, then always registered
`CPU` last. `coreml`/`cuda` are opt-in Cargo features (`default = []`), and no documented build
command (`make build`, `cargo build --release -p vocalai-cli`) ever passed `--features coreml`/
`cuda` — so in practice every `vocalai` run to date had used CPU only, and CoreML/CUDA had never
actually been attempted.

The user asked for two things: (1) a hardware-EP registration failure should be a loud error, not a
silent CPU fallback that could be much slower or (in the PyTorch-eager-mode case they'd previously
hit) hang a device under memory pressure; (2) their own capable Mac should still "just work" without
needing to pass a flag. Both required a real decision about what "no flag" means, plus (once
`coreml` was actually compiled in for the first time and exercised) two things manual testing
surfaced that the original plan hadn't anticipated:

1. `s3gen_estimator.onnx` reliably crashes ONNX Runtime's CoreML EP mid-inference.
2. Once that crash was fixed, CoreML's **default** configuration measured 30-40% *slower* than CPU
   wall-clock (and subjectively made the whole device feel less responsive during a run) — the
   opposite of the assumed benefit. This directly contradicted the original plan's assumption that
   "T3's decode loop is where most of the GPU speedup is."

## Decision

**No flag (default) → CPU.** `--use-gpu` requires a hardware EP with a tuned CoreML configuration
(below) and errors out (`SessionError::GpuUnavailable`) rather than falling back to CPU if none is
usable. `--use-cpu` is kept as an explicit, separate flag (currently behaves identically to the
default) so it keeps meaning something if a future change reintroduces a non-CPU default.
`session::ExecutionProviderPreference::Auto` (try hardware, fall back to CPU on registration
failure) remains in `session.rs`'s API and test coverage but is not reachable from any current CLI
flag combination.

**Root cause of the CoreML slowdown, and the fix**: T3's autoregressive decode loop calls
`session.run()` roughly once per generated token (up to `--max-new-tokens`, e.g. ~1000 for a long
input). CoreML's default config pays a fixed per-call dispatch/specialization cost that's negligible
for one large batched call but dominates across ~1000 tiny sequential ones — compounded further
because the decoder graph is only partially covered by CoreML (1150 of 2378 nodes measured), so
every one of those ~1000 calls also pays a CPU↔CoreML data-marshaling cost for the uncovered
portion. Manually benchmarked on a fixed workload (same text, same `--max-new-tokens`):

| Config | Wall-clock (250-token workload) | vs. CPU |
|---|---|---|
| CPU (`--use-cpu`) | ~28-32s | baseline |
| CoreML, default config | ~42s | ~30-40% slower |
| + `SpecializationStrategy::FastPrediction` | ~39s | still slower |
| + `ComputeUnits::CPUAndGPU` (excludes ANE) | ~37s | still slower |
| + `ModelFormat::MLProgram` | — | **fails outright** (error code -14) for this graph |
| `CPUAndGPU` + `FastPrediction` + `RequireStaticInputShapes` | ~26-33s | roughly tied |

**Correction (2026-08-20)**: the "roughly tied" reading above does not hold up. Each `vocalai`
invocation uses a fresh RNG, so the number of tokens generated before EOS varies run to run (output
duration ranged 8.28-9.52s across nominally-identical 250-token trials) — the small-scale numbers
above are within that noise, not a real signal either way. A separate full-scale run (the default
1000-max-tokens case, closer to real usage) measured the tuned config at **~14% slower than CPU**
(117s vs 102s), and the repo owner's own real-world re-test after this fix landed found no
improvement either. **The honest conclusion: this tuning fixes the severe regression (30-40%
slower) down to roughly-tied-or-somewhat-worse — it is not a demonstrated speed win, and CPU remains
the better-or-equal choice on this hardware for this workload.** The tuned config is still kept (it
is a strict improvement over the naive default, and doesn't regress anything `--use-gpu` was already
opt-in for), but nothing in this repo should claim GPU is faster without a more rigorous benchmark
(fixed/deterministic token count across many repeated trials) than what was actually done here.

The tuned config (`CoreML::default().with_compute_units(CPUAndGPU)
.with_specialization_strategy(FastPrediction).with_static_input_shapes(true)`) is the permanent
CoreML config in `hardware_execution_providers()`, kept for the reasons below even though it isn't a
proven speed win:
- `CPUAndGPU` excludes the Apple Neural Engine, whose fixed per-call dispatch latency is worse than
  GPU/Metal's for a many-tiny-calls access pattern like this.
- `FastPrediction` trades a larger one-time specialization cost at session load for lower per-call
  prediction latency — exactly the right trade when a session pays that cost once and then calls
  `run()` ~1000 times.
- `RequireStaticInputShapes` keeps CoreML from taking nodes whose shape depends on the growing
  KV-cache, avoiding dynamic-shape handling overhead inside the compiled program.
- `MLProgram` format was tried and rejected — it fails to even build an execution plan for this
  graph on this ONNX Runtime/macOS combination (a real, clean `SessionError::GpuUnavailable` thanks
  to `.error_on_failure()`, not a hang or silent fallback).

Separately (unaffected by the above): manual testing found that `s3gen_estimator.onnx` — S3Gen's
flow-matching Euler ODE estimator, a UNet-style network — reliably crashes CoreML mid-inference
(`Unable to compute the prediction using a neural network model`), even though `GetCapability`
accepts a partition of its graph. Bisected by manually forcing just that one session to CPU while
leaving every other session on CoreML: the run completed successfully. Root cause: the estimator is
the only session in the pipeline whose sequence length (`2 * token_len`) is genuinely dynamic rather
than bucketed like the flow-encoder already is (ADR-0009). `ModelBundle::load` pins this one session
to CPU whenever CoreML is the resolved provider (CUDA is untested and left alone).

`Makefile`'s `build:` target auto-detects the right hardware feature per OS (`coreml` on Darwin,
`cuda` elsewhere) so `make build` compiles in hardware-EP support without the caller needing to know
which Cargo feature applies to their platform — confirmed via `ort-sys`'s build script that this
never requires local GPU hardware or a CUDA toolchain to *compile*, only to *use* the resulting EP.

## Rationale

- `.error_on_failure()` vs `.fail_silently()` is the exact `ort` primitive for "a hardware EP that
  can't register should be a catchable error" — this is what turned the `MLProgram` incompatibility
  above into a clean error instead of a silent fallback or a hang, and is why `--use-gpu` can be
  trusted to mean what it says.
- Defaulting to CPU rather than `Auto`/GPU is the direct, honest consequence of the evidence: the
  naive CoreML config was measurably worse across the board, including subjective device
  responsiveness; the tuned config fixes the severe regression but, per the correction above, does
  not reliably beat CPU either. There is no evidence here that justifies defaulting away from the
  simple, well-understood CPU path.
- Removing the CPU-retry-on-inference-failure logic (present in an earlier iteration of this
  session, tied to `Auto` mode) followed directly from removing `Auto`'s CLI reachability — dead code
  keyed on a state (`ep_pref == Auto`) that can no longer occur from any flag combination.
- Pinning only the one identified session (the estimator) to CPU, rather than disabling CoreML
  entirely, preserves the CoreML path for the rest of the pipeline as an option for `--use-gpu`
  users, without over-scoping the fix to a component that was never the reported problem.
- The Makefile's OS-based feature detection is build-time only and platform-exclusive in practice
  (CoreML is Apple-only, CUDA is everywhere else) — no runtime GPU probing needed to decide which
  Cargo feature to compile with, and it costs nothing on platforms where the resulting hardware EP
  never actually gets used (`--use-gpu` opt-in only).

## Alternatives rejected

- **Keep `Auto` (try GPU, fall back to CPU) as the default**: rejected — before the config tuning,
  GPU was a clear regression; after tuning, it's roughly tied-or-worse (see the correction above),
  which is no evidence at all for defaulting away from CPU. Revisit only if a rigorous benchmark
  ever shows GPU ahead.
- **Keep `.fail_silently()` everywhere, just gate CPU registration behind a flag**: doesn't satisfy
  "a hardware failure should be loud" — `ort`'s intrinsic CPU-kernel fallback for unsupported ops
  happens regardless of whether `CPUExecutionProvider` is explicitly registered.
- **`ModelFormat::MLProgram`**: tried; fails to build an execution plan for this graph outright.
- **Re-export `s3gen_estimator.onnx` with a bucketed time dimension now, instead of pinning to CPU**:
  the real, complete fix (tracked as VAI-014) — rejected for *this* ticket because it's
  export-pipeline work (`export/export_s3gen.py` + re-running `parity_check.py`), not a CLI-flag or
  EP-config change.
- **Disable CoreML entirely until VAI-014 lands**: rejected — `--use-gpu` stays opt-in and is now at
  least not severely regressed; there's no reason to remove it just because it isn't a proven win.
- **Retry-on-inference-failure for `--use-gpu`**: rejected — it's an explicit hardware requirement;
  silently falling back on any failure would make the flag meaningless. (This also applied to
  `Auto` in an earlier iteration of this session, but that mode is no longer CLI-reachable.)

## Consequences

- `session::build_session`/`resolve_and_build_session` and `pipeline::ModelBundle::load` all take an
  explicit provider argument now — any future new session in `ModelBundle` must remember to pass
  `resolved_provider` (or a CPU-pinned override, if a session is later found to need one).
- The CoreML tuning knobs (`CPUAndGPU`/`FastPrediction`/`RequireStaticInputShapes`) are specific to
  this pipeline's access pattern (many tiny sequential calls). If T3's export ever changes to a
  batched or non-autoregressive decode strategy, this configuration should be re-benchmarked — it is
  not a universally-correct CoreML config.
- The CPU pin on `s3gen_estimator_session` is a known, temporary special case, not a general
  mechanism — VAI-014 should remove it once the estimator is bucketed.
- Cannot literally guarantee zero CPU instructions execute in `Gpu` mode: ONNX Runtime's CPU kernels
  are intrinsic to the runtime and may still run individual ops a hardware EP doesn't cover. This
  module only controls whether CPU can be the *primary/sole* provider.
- `session::ExecutionProviderPreference::Auto` and its associated fallback logic remain valid,
  tested library API, but nothing in the CLI currently constructs it — a future change wanting
  "try GPU, silently fall back to CPU" behavior from the CLI would need to re-wire a flag to it (and
  likely re-add a CPU-retry safety net for the mid-inference-failure case this ADR's earlier
  iteration had and then removed).
