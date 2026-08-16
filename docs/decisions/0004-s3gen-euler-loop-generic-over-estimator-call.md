# ADR-0004: S3Gen's Euler ODE loop is generic over the estimator call, not `ort`-coupled

**Date**: 2026-08-16
**Status**: Accepted
**Decider**: Shivam Mishra + AI agent (session 2026-08-16)

## Context

Milestone 3 (`docs/issues.md` VAI-003) reimplements `ConditionalCFM.solve_euler`'s CFG-doubled
Euler ODE loop (`chatterbox/models/s3gen/flow_matching.py`) in `crates/vocalai-core/src/s3gen.rs`,
driving the exported flow-estimator ONNX graph. Per `docs/agents/CONVENTIONS.md` §1, TDD is
mandatory — the loop needs real unit tests in the same commit.

Model weights and `.onnx` exports are git-ignored build/release artifacts (hard constraint,
`CLAUDE.md` §1) — they don't exist in a fresh clone and aren't downloaded as part of `cargo test`.
`crates/vocalai-core` had no live `ort::Session` usage before this milestone (Milestones 1-2 only
built the EP-selection list in `session.rs`). A Euler-loop implementation written directly against
`ort::Session`/`ort::Value` would have no way to be unit-tested in `cargo test --workspace` without
either bundling a real ONNX file (violates the git-ignore constraint) or downloading one at test
time (network dependency, slow, breaks a fresh-clone `make check`).

## Decision

`solve_euler()` takes the CFG-doubled per-step estimator call as a generic closure parameter
(`impl FnMut(EstimatorStep<'_>) -> Result<Array3<f32>, E>`), not an `&ort::Session`. The loop's
tensor assembly (CFG batch-doubling, `dxdt` combination, Euler update) is plain `ndarray` math with
zero `ort` types in its signature. A separate, thin function (`run_estimator`) adapts a real
`&mut ort::Session` to that closure signature for production use (`generate_waveform`).

## Rationale

- Lets the loop's actual math — the part with real correctness risk (CFG combination formula,
  batch assembly, Euler update) — be unit-tested with a synthetic closure (`dxdt = mu - x`) and
  hand-computed expected outputs, with no ONNX Runtime session, no model file, no network access.
  See `crates/vocalai-core/src/s3gen.rs`'s `tests` module.
- Keeps the numerically-fragile, weights-dependent part of the validation (does the *exported ONNX
  graph* numerically match the *PyTorch reference*) where it already lives for every other
  component: `export/parity_check.py` (`check_s3gen`), which replicates this exact loop in Python
  against the real exported files and the real PyTorch model. Rust and Python thus implement the
  identical algorithm, cross-checked by inspection/parity rather than by sharing a test fixture.
- Costs one small abstraction (a struct + generic closure param) in exchange for keeping
  `cargo test --workspace` fast, offline, and independent of `models/`'s existence — consistent
  with how Milestone 1-2's `session.rs` tests never needed a live model either.

## Alternatives rejected

- **Write `solve_euler` directly against `&mut ort::Session`, skip Rust-level math tests**: would
  leave the CFG/Euler math with zero automated coverage in Rust (violates TDD, `CONVENTIONS.md`
  §1), relying entirely on the Python parity check to catch a Rust-side transcription bug.
- **Bundle a tiny fixture `.onnx` file for Rust tests**: violates the no-binary-artifacts
  constraint (`CLAUDE.md` §1) even for a small file, and still wouldn't validate anything beyond
  what the synthetic-closure test already proves (session wiring correctness, not math
  correctness) — the trace-and-export machinery is exactly what's *not* being unit-tested here.
- **Skip the estimator-call abstraction, inline everything in `generate_waveform`**: same result as
  the first rejected option, just without even the `run_estimator`/`solve_euler` seam for reuse.

## Consequences

**Easier**:
- `solve_euler`'s tests run in milliseconds, offline, in every `cargo test --workspace` — no
  network, no `models/` directory required.
- Any future estimator backend (e.g. a different runtime, or a mocked estimator for
  `vocalai-cli` integration tests) can reuse `solve_euler` by supplying a different closure.

**Harder**:
- Two places implement the same loop (Rust `solve_euler` + Python `_solve_euler_onnx` in
  `export/parity_check.py`) that must be kept in sync by hand if the reference algorithm ever
  changes — there's no shared source of truth beyond the PyTorch reference itself.

**New commitments**:
- If `ConditionalCFM.solve_euler`'s CFG/Euler math changes upstream (e.g. a different scheduler),
  both `crates/vocalai-core/src/s3gen.rs::solve_euler` and
  `export/parity_check.py::_solve_euler_onnx` need updating together, plus this ADR's Context.
