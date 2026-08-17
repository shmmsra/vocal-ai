# ADR-0006: Split CI into a fast gate and a separate real-checkpoint parity gate

**Date**: 2026-08-17
**Status**: Accepted
**Decider**: Shivam Mishra + AI agent (session 2026-08-17)

## Context

ADR-0001 (§Decision item 4) decided: *"`make check` runs `cargo fmt --check`/`clippy`/
`cargo test` + `pytest` (for `export/`); ... CI mirrors it."* `.github/workflows/ci.yml`'s
header made this explicit: *"mirrors `make check` exactly... don't let them diverge."*
Until VAI-004, this was a single job running the whole `pytest` suite, including
`export/parity_check.py`'s real-checkpoint-vs-ONNX numerical parity checks
(`check_hifigan`/`check_ve`/`check_s3tokenizer`/`check_s3gen`).

VAI-004 added `check_t3`, which downloads `t3_cfg.safetensors` (2.0GB — by far the
largest checkpoint; `s3gen.safetensors` is 1.0GB, `ve.safetensors` 5.4MB) and holds
the loaded ~2GB PyTorch model in memory alongside an in-memory ~1.9GB ONNX protobuf
(built during export) and a live ~1.9GB `onnxruntime` session. On the shared
GitHub-hosted `ubuntu-latest` runner, this first hit a disk-space exhaustion hang
(fixed by reclaiming preinstalled-toolchain disk, see the VAI-004 CI-hang commit),
then an out-of-memory kill (`Killed`, exit 143) on the same test. The per-commit-gate
job was accumulating GB-scale, RAM-heavy work that only grows as more components
(PerthNet next, turbo/multilingual variants later) get exported.

Two things this repo's owner explicitly does **not** want, raised while discussing
the fix: (1) parity checks running on a time-based schedule instead of being
triggered by an actual relevant change ("everything should be cause and effect"),
and (2) any change that makes CI *weaker* — the parity check is a hard project
constraint (`CLAUDE.md` §1: *"No exported component ships into `vocalai-core` until
`export/parity_check.py` confirms numerical parity with the PyTorch reference"*),
and the Rust-side unit tests are deliberately ONNX-free by design (ADR-0004,
ADR-0005) — nothing else in this repo actually proves an exported `.onnx` graph is
numerically correct.

## Decision

Split the single CI workflow into two, **on the same triggers as before** (push/PR
to `main` — no schedule, no path filter, no reduction in what runs on a given
commit):

- **`.github/workflows/ci.yml`** ("Pre-commit gate (fast)"): `fmt-check`, `clippy`,
  `cargo test --workspace`, and `make test-py-fast` (`pytest -m "not parity"` —
  `test_requirements.py` plus `_common.allclose_report`'s 3 pure tests). Fully
  offline beyond installing `export/requirements.txt`; no checkpoint download.
- **`.github/workflows/parity.yml`** ("ONNX-vs-PyTorch parity gate"): `make
  test-py-parity` (`pytest -m parity` — the 5 tests that call `load_t3`/
  `load_s3gen`/`load_voice_encoder` and a `check_*` function). Keeps the
  disk-space-reclaiming step from the VAI-004 CI-hang fix.

New `export/pytest.ini` registers the `parity` marker; the 5 heavy tests in
`export/tests/test_parity_check.py` are decorated `@pytest.mark.parity`. New
`Makefile` targets `test-py-fast`/`test-py-parity` mirror the marker split;
`test-py`/`check`/`test` are unchanged and still run everything (a local dev
machine has checkpoints cached across runs and much more RAM/disk than a CI
runner, so the strict "everything must pass" local pre-commit gate is preserved).

Also fixed in the same change: `check_t3` now explicitly frees the ~2GB PyTorch
model (`del t3`, `load_t3.cache_clear()`, `gc.collect()`) after extracting the
reference greedy-decode outputs and before loading the ONNX Runtime sessions for
the second half of the comparison — this reduces peak memory regardless of which
workflow runs the test, and was the actual proximate cause of the OOM kill.

## Rationale

- **Same triggers, same total coverage, just reorganized**: together the two
  workflows run everything `make check` runs locally, on every push/PR exactly as
  before — this is a reorganization for blast-radius/resource isolation, not a
  weakening of the gate. A future agent finding two workflow files instead of one
  should read this ADR and see nothing was made optional or delayed.
- **No schedule, no path filter**: explicitly rejected per the repo owner's "cause
  and effect" principle (see Context) — the parity gate runs *because* a commit
  landed, same as the fast gate, not on a timer that might fire with nothing to
  check or miss a relevant change made outside its filter.
- **Separating the jobs makes each easier to reason about and resource
  independently**: the fast job can stay a simple, fully-offline lint/test gate
  indefinitely as more components are exported; the parity job is where
  checkpoint-size growth, memory ceilings, and (later) larger-runner decisions
  belong, without dragging the fast job's simplicity down with it.
- **The `check_t3` memory fix is orthogonal but necessary regardless of the
  workflow split**: even in a single combined workflow, the same OOM would recur
  every time `check_t3` runs, so it's fixed at the source (the test's own memory
  lifecycle), not worked around by moving it to a different job.

## Alternatives rejected

- **Keep the parity checks in the fast, per-commit job and just fix the memory
  bug**: still couples a fast, cheap, always-green-fast gate to a slow,
  multi-GB-download, memory-heavy one — doesn't scale as more components (PerthNet,
  turbo/multilingual) are added, and one flaky/slow checkpoint download blocks the
  fast signal every contributor actually wants quickly.
- **Move the parity gate to a schedule (nightly/weekly) or manual-only
  (`workflow_dispatch`)**: considered and explicitly rejected — violates "cause and
  effect" (Context) and would mean a broken export could land on `main` without the
  hard parity-check constraint (`CLAUDE.md` §1) being enforced on that commit at
  all, only whenever the schedule/manual trigger next happens to run.
- **Path-filter the parity workflow to only `export/**`/`crates/vocalai-core/src/**`
  changes**: a reasonable middle ground, but still rejected for now in favor of
  running on every commit — kept as a documented option here if checkpoint growth
  or runner cost later makes "every commit" impractical.

## Consequences

**Easier**:
- The fast gate stays fast and simple as a matter of structure, not discipline —
  a future contributor adding a 6th component's parity check can't accidentally
  slow down the fast job just by adding a test function; they have to explicitly
  mark it `@pytest.mark.parity` or it runs in the wrong job.
- The parity job's memory/disk/runner-size tuning is isolated from the fast job's
  configuration.

**Harder**:
- Two workflow files (and two Makefile targets) to keep in sync with `pytest.ini`'s
  marker registration — a new heavy test that forgets `@pytest.mark.parity` would
  silently run in (and slow down) the fast job instead of failing loudly.

**New commitments**:
- Milestone 5 (PerthNet export) and any future component's parity check must be
  marked `@pytest.mark.parity` in the same commit that adds it.
- If checkpoint sizes keep growing (turbo/multilingual variants, per the plan's
  backlog) and per-commit parity runs become impractical even on their own runner,
  revisit the path-filter alternative above rather than reaching for a schedule.
- Milestone 7 (packaging/release-artifact builds, plan §7 item 7 — the actual
  "build generation" pipeline bundling weights + CUDA/cuDNN into release
  artifacts) is a further, heavier, separate pipeline from both of these — expect
  a third workflow (e.g. `release.yml`) when that milestone starts, following the
  same "own job, own resourcing" pattern established here.
