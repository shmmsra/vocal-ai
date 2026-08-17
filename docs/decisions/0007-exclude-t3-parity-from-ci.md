# ADR-0007: Exclude T3's parity check from CI; run it locally instead

**Date**: 2026-08-18
**Status**: Accepted
**Decider**: Shivam Mishra + AI agent (session 2026-08-18)

## Context

ADR-0006 split CI into a fast gate (`ci.yml`) and a real-checkpoint parity gate
(`parity.yml`), on the same every-commit triggers as before, specifically so the
hard parity-check constraint (`CLAUDE.md` §1) stayed enforced on every commit. In
practice, `parity.yml`'s T3 test still failed on a real run — first with a
disk-exhaustion hang (fixed), then with an out-of-memory kill after a first
memory-lifecycle fix in `check_t3` (that fix addressed the *wrong* phase).

Measuring directly (`/usr/bin/time -l` on a clean `models/` directory, forcing a
real fresh export rather than reusing a locally-cached `.onnx`) isolated the actual
cost:

| Step | Peak memory (measured) |
|---|---|
| `torch.onnx.export` tracing/serializing `t3_decoder.onnx` from scratch | **~9GB** |
| Loading an already-built `t3_decoder.onnx` into `onnxruntime` + running inference | ~2.3GB |

The ~9GB is inherent to the legacy (non-`dynamo`) `torch.onnx.export` tracer
building a ~2GB-parameter (510M param, fp32) model's ONNX graph — confirmed
independent of `do_constant_folding`, and confirmed that `external_data=True`
(the parameter's stated default) is silently ignored on this torch version's
legacy export path (verified: still writes one single 1.9GB file, same ~9GB peak,
whether the flag is passed explicitly or not — it only takes effect on the newer
`dynamo=True` exporter). This repo's confirmed a **private** GitHub repo (a clean
unauthenticated API check returns 404), which puts it on GitHub's free-tier hosted
Linux runner — nowhere near enough headroom for a 9GB peak alongside everything
else the job needs (OS, Python, torch/onnxruntime import overhead).

Every other exported component's checkpoint/graph is far smaller (`s3gen`'s
`s3gen_estimator.onnx` is 273MB, the largest of the rest) and fits comfortably;
T3 alone is exceptional by roughly an order of magnitude.

## Decision

`export/tests/test_parity_check.py`'s T3 test keeps its `@pytest.mark.parity`
marker (still runs locally via `make test-py-parity`/`make check`) and gains a new
`@pytest.mark.heavy_build` marker. CI's `parity.yml` now runs a new Makefile target,
`test-py-parity-ci` (`pytest -m "parity and not heavy_build"`), instead of the full
`test-py-parity` — so CI verifies hifigan/ve/s3tokenizer/s3gen on every commit, and
T3's parity check becomes a **local-only, developer-run gate**: it must be run
manually (`make test-py-parity` or `python parity_check.py --component t3`) before
committing changes to `export/export_t3.py` or `crates/vocalai-core/src/t3.rs`.

The `heavy_build` marker is deliberately generic (not `t3_only`) — any future
component whose *build* step (not just verification) needs more memory than the
CI runner has should get the same marker, not a bespoke exclusion.

## Rationale

- **The expensive part is building the ONNX graph, not verifying it.** Once a
  `.onnx` file exists, checking it against PyTorch is cheap (~2.3GB, measured).
  There's no way to get a pre-built file into a fresh CI checkout without either
  committing model artifacts (violates the no-binary-artifacts hard constraint,
  `CLAUDE.md` §1) or fetching them from some other persistent store — out of scope
  for this fix.
- **This is a resource ceiling, not a bug to patch away.** Disabling constant
  folding didn't change peak memory; the `external_data` flag doesn't apply to the
  legacy exporter. The realistic ways to actually reduce the ~9GB figure — switching
  to the newer `dynamo=True` exporter, or splitting the 30-layer decoder into
  per-layer ONNX graphs — are both substantial redesigns (the latter reopens
  ADR-0005's approved single-graph decision) disproportionate to fixing a CI gap for
  one already-shipped, already-tested component.
- **The hard parity constraint isn't dropped, just its enforcement mechanism for
  this one component changes from automatic to manual.** `check_t3` still exists,
  still gets exercised (locally, and by whoever changes T3-adjacent code), and still
  must pass before a T3-affecting change ships — CI just isn't the thing checking
  it.

## Alternatives rejected

- **Keep re-exporting T3 in CI on every commit**: this is what failed twice already
  (disk, then memory) — not viable on this runner tier.
- **Cache the built `.onnx` across CI runs** (`actions/cache`): the *first*
  cache-populating run still needs the same ~9GB peak, so this doesn't fix the
  underlying problem, only amortizes it after one success — and that first success
  is exactly what's failing.
- **Switch to the `dynamo=True` exporter for T3** (or all components, for
  consistency): plausible future fix if the memory profile is genuinely better, but
  unproven for this repo's hand-rolled decoder (ADR-0005) and every other export
  script; too large a change to bundle into an urgent CI fix.
- **Split the decoder into per-layer ONNX graphs**: would likely reduce peak memory
  roughly proportionally, but reopens ADR-0005's approved single-graph design and
  adds real Rust-side complexity (30 chained sessions per decode step) — a
  deliberate future option, not an urgent one.
- **Pay for a larger GitHub-hosted runner, or self-host one**: viable, but a
  billing/infrastructure decision for the repo owner to make deliberately, not
  something to reach for silently while fixing a test failure.

## Consequences

**Easier**: CI stays fast and green for every commit that doesn't touch T3-adjacent
code; the fast/parity split (ADR-0006) keeps working as designed for every other
component.

**Harder**: T3's parity check has no automatic enforcement — a change to
`export/export_t3.py` or `crates/vocalai-core/src/t3.rs` that breaks numerical
parity could land on `main` if the contributor forgets to run it locally first.

**New commitments**:
- Any future component whose real, from-scratch export needs more memory than the
  CI runner offers gets `@pytest.mark.heavy_build`, following this same pattern —
  checked explicitly before assuming a new `check_*` test can just run in CI.
- If this repo's resourcing changes later (a paid larger runner, a self-hosted
  runner, or a switch to the `dynamo` exporter with a verified lower memory
  profile), revisit whether `heavy_build` tests can move back into
  `test-py-parity-ci`.
- Milestone 7 (release-build/bundling, plan §7 item 7 — the "build generation"
  pipeline) will very likely hit this exact same resource ceiling, since it also
  has to run `torch.onnx.export` on T3 at least once to produce the release
  artifact. Left open for now (repo owner is still deciding the approach) but
  worth remembering when that milestone starts: it cannot run on this same
  free-tier hosted runner either.
