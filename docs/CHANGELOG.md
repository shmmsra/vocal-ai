# vocal-ai — Changelog

> Chronological log of what changed in this repo and *why*. The "why" matters more than the "what" — the diff already shows the what.
>
> Update at the end of every session. Newest entries at the top.

---

## 2026-08-17 — ci: split CI into fast + parity workflows, fix `check_t3` OOM

**What changed**: VAI-004's `check_t3` (previous entry) OOM-killed CI (`Killed`, exit 143):
the ~2GB PyTorch T3 model, the in-memory ~1.9GB ONNX protobuf built during export, and a
loaded ~1.9GB `onnxruntime` session on the same graph could all be resident at once. Fixed
at the source: `check_t3` now extracts the (tiny) reference greedy-decode outputs, then
explicitly frees the torch model (`del t3`, `_common.load_t3.cache_clear()`, `gc.collect()`)
*before* loading the ONNX Runtime sessions for the second half of the comparison.

Separately, split `.github/workflows/ci.yml` (previously the single "mirrors `make check`
exactly" job, per ADR-0001) into two workflows on the **same triggers as before** (every
push/PR to `main` — no schedule, no path filter): `ci.yml` now runs only the fast,
fully-offline checks (fmt/clippy/`cargo test`/`pytest -m "not parity"`), and new
`parity.yml` runs the 5 tests that download a real HuggingFace checkpoint and validate
ONNX-vs-PyTorch numerical parity (`pytest -m parity`, keeping the disk-space-reclaim step
from the earlier CI-hang fix). New `export/pytest.ini` registers the `parity` marker; new
`Makefile` targets `test-py-fast`/`test-py-parity` mirror the split. Local `make
check`/`test-py` are unchanged and still run everything. See ADR-0006.

**Why**: The parity checks are a hard project constraint (`CLAUDE.md` §1 — no exported
component ships until `parity_check.py` confirms numerical parity), so they can't be
skipped or deferred to a schedule; but coupling them to the same fast per-commit job as
lint/unit-tests means one growing multi-GB checkpoint (T3's is 2GB, the largest by far)
drags down a signal every contributor wants fast. Splitting into two jobs on identical
triggers preserves exactly the same enforcement while isolating each job's resourcing.

**What was rejected**: A scheduled (nightly/weekly) or `workflow_dispatch`-only parity
trigger — explicitly rejected by the repo owner ("everything should be cause and effect");
would also mean a broken export could land on `main` without the hard parity constraint
being checked on that commit at all. A path-filtered trigger (`export/**`/
`crates/vocalai-core/src/**` only) — reasonable, documented as a fallback in ADR-0006 if
checkpoint growth later makes every-commit parity runs impractical, but not adopted now to
keep the "runs on every commit, same as before" property simple and legible.

**What's next**: Milestone 5 — export PerthNet, wire watermarking into output
(`docs/issues.md` `VAI-005`); mark its parity test `@pytest.mark.parity` in the same
commit, per ADR-0006's new commitment.

---

## 2026-08-17 — VAI-004: Export T3 as decoder-with-past, implement KV-cache decode loop + sampling

**What changed**: Added `export/export_t3.py`, which exports T3's Llama-style backbone as two ONNX
graphs plus two raw embedding-table `.npy` files. `transformers==5.2.0`'s `LlamaModel` is built
entirely around `Cache`/`DynamicCache` objects and `masking_utils.create_causal_mask`, which don't
trace through the legacy `torch.onnx.export` tracer this repo already uses (ADR-0002) — so
`T3DecoderExport`/`_ExportDecoderLayer` hand-roll the same Llama math (RMSNorm, RoPE reusing the
model's own precomputed llama3-scaled `inv_freq` buffer, SwiGLU MLP, no GQA since
`num_key_value_heads == num_attention_heads`) directly against `T3.tfmr`'s real submodules — no
weight copying, no Cache object. See ADR-0005 for the full rationale, including why the KV-cache is
one stacked `(layers, k/v, batch, heads, seq, head_dim)` tensor rather than 60 per-layer-named
tensors. `T3CondPrefillExport` separately reproduces `T3.prepare_input_embeds()` plus
`T3.inference()`'s double-BOS-embedding construction (a real quirk in the reference — two
numerically-identical BOS embeddings get concatenated back to back before the first decoder
forward). `export/_common.py` gained `load_t3()`.

Added `crates/vocalai-core/src/t3.rs`: the sampling math (CFG combine, repetition penalty,
temperature, min-p, top-p, greedy/multinomial selection) is generic over the decoder-step call,
mirroring `s3gen::solve_euler`'s pattern (ADR-0004) — 22 new Rust unit tests cover it with synthetic
decoders/logits, no ONNX Runtime session needed. Per-step new-token embedding is a plain
`speech_emb`/`speech_pos_emb` weight-table row lookup in Rust (`embed_speech_token`,
`load_embedding_table` via the new `ndarray-npy` dependency), not an extra ONNX call per generated
token. Added `rand` for multinomial sampling.

Added `export/parity_check.py::check_t3`. `T3.inference()`'s `do_sample` parameter is accepted but
never actually read in the reference — sampling is always stochastic (`torch.multinomial`), and
PyTorch's/Rust's RNGs are unrelated, so comparing free-running sampled token sequences across
languages would be meaningless (one divergent token cascades into a different sequence). Instead,
`check_t3` runs a **greedy** (argmax) replica of the real reference forward pass
(`_greedy_reference_t3` — same `Cache`, same weights, same RoPE, just non-stochastic selection)
alongside a free-running greedy loop driving the exported ONNX graphs (`_greedy_onnx_t3`), and
compares both the resulting token sequences (must match exactly) and the per-step processed logits
(within tolerance). Passed on first real run against the downloaded checkpoint:
`max_abs_diff=4.768e-05` (well within `atol=1e-4`), 6/6 greedy tokens matching.

**Why**: Milestone 4 is the main technical risk of Phase 1 (plan §9) — T3 is the only exported
component with real autoregressive control flow. The hand-rolled decoder was necessary because
tracing HF's own `LlamaModel.forward` would mean monkeypatching private `transformers` internals
(`cache_utils`, `masking_utils`) at a specific minor version — a worse maintenance bet than
re-implementing the small, stable, public Llama math directly (ADR-0005). Greedy-decode parity was
chosen over the plan's literal "fixed seed" wording because a "fixed seed" doesn't make stochastic
sampling comparable across two unrelated RNG implementations; greedy removes randomness from the
comparison while still exercising the real end-to-end forward pass.

**What was rejected**: Tracing `T3HuggingfaceBackend.forward` directly with `Cache`/mask internals
monkeypatched (fragile against `transformers` version churn). Per-layer-named
`past_key_values.N.key/.value` ONNX I/O matching the `optimum`/HF convention (no external consumer
in this pipeline needs it; one stacked tensor is less Rust-side bookkeeping). A third ONNX graph for
the per-step new-token embedding (a trivial lookup, not worth an ONNX Runtime call per generated
token). See ADR-0005 for the full list.

**What's next**: Milestone 5 — export PerthNet, wire watermarking into output
(`docs/issues.md` `VAI-005`).

---

## 2026-08-16 — fix: S3 tokenizer export corrupted `freqs_cis` on a fresh `models/` dir

**What changed**: `export_s3tokenizer.py::build_wrapper()` mutates `encoder.freqs_cis` in place on
the `@lru_cache`d shared `s3gen` object returned by `_common.load_s3gen()` (replacing the original
complex rotary buffer with the real-valued equivalent needed for ONNX export — see the Milestone 2
CHANGELOG entry below). `check_s3tokenizer()` calls `build_wrapper()` once directly (to get the
"PyTorch reference" module), then again — unguarded — via `export()` when `models/s3tokenizer.onnx`
doesn't already exist. The second call read the *already-mutated* buffer's shape (whose last dim is
now `2`, the real/imaginary pair, not the original head dim) and computed a corrupted replacement
from it, causing a hard shape-mismatch `RuntimeError` during tracing (`"size of tensor a (64) must
match the size of tensor b (2)"`). Fixed by guarding the mutation with `torch.is_complex(...)`, so
a second call on an already-converted buffer is a no-op.

**Why**: This was a **pre-existing bug from VAI-002**, invisible in every local `make check` run
because the developer's `export/`-adjacent `models/` directory (git-ignored, dev-time only) had a
stale `s3tokenizer.onnx` cached since the day VAI-002 landed — so `check_s3tokenizer()`'s
`export()` branch was always skipped locally, and `build_wrapper()` only ever ran once per process.
CI clones fresh (no `models/` directory at all) and was the first environment to actually exercise
the `export()` branch, surfacing the bug as CI's very first pre-commit-gate failure (reported by
the human after pushing VAI-003). Reproduced locally by clearing `models/*.onnx` and reinstalling
`export/requirements.txt` into a throwaway venv to match a clean-checkout CI run exactly; confirmed
the fix resolves it in that same reproduction before applying it for real.

**What was rejected**: Ruled out transitive-dependency drift (a common risk with unpinned
sub-dependencies like `transformers`/`diffusers`/`s3tokenizer` in `chatterbox-tts`'s dependency
tree) by installing `requirements.txt` fresh and confirming it resolved to the *same* versions
already in the local dev venv — the bug reproduced regardless, isolating it to the shared-mutation
logic, not a version mismatch.

**What's next**: No open follow-up — `build_wrapper()` is now idempotent under repeated calls in
the same process, which is the property every `export_*.py`'s model-loading helper needs (see
`export_hifigan.py::_fuse_weight_norm`'s pre-existing idempotency comment for the established
pattern).

---

## 2026-08-16 — VAI-003: Export S3Gen flow estimator + Euler ODE loop, chain into HiFiGAN

**What changed**: Added `export/export_s3gen.py`, which exports the S3Gen flow-matching estimator
(`ConditionalDecoder`, accessed via `s3gen.flow.decoder.estimator`) as a static per-step ONNX graph
— the same per-step call `ConditionalCFM.solve_euler` makes inside its Python loop
(`x, mask, mu, t, spks, cond` in; `dxdt` out, batch pre-doubled for CFG). Added
`crates/vocalai-core/src/s3gen.rs`: `cosine_t_span()` (the `t_scheduler='cosine'` schedule),
`solve_euler()` (the CFG-doubled fixed-step Euler loop, generic over the per-step estimator call —
see ADR-0004 for why), `run_estimator()`/`mel_to_waveform()` (real `ort::Session`-backed adapters
for the estimator and the Milestone-2 HiFiGAN session), and `generate_waveform()` (chains both).
5 new Rust unit tests cover the cosine schedule and the Euler/CFG math against a synthetic linear
estimator (`dxdt = mu - x`) with hand-computed expected outputs — no ONNX Runtime session or model
file needed. Added `export/parity_check.py::check_s3gen`, which replicates the identical
CFG-doubled loop in Python (`_solve_euler_onnx`) driving the exported `s3gen_estimator.onnx` +
`hifigan.onnx`, and compares both the intermediate mel and the final waveform against the real
PyTorch `ConditionalCFM.solve_euler` + HiFiGAN wrapper (mel max_abs_diff ~4e-5, waveform ~1e-5;
well within atol=1e-4/rtol=1e-3). Added `ndarray = "0.17"` + `ort`'s `ndarray` feature to
`vocalai-core`'s `Cargo.toml` (version pinned to match `ort` 2.0.0-rc.13's own `ndarray`
dependency, so the two share one type in the dependency graph).

**Why**: Milestone 3 is the first Rust code to actually drive an ONNX Runtime session (Milestones
1-2 only built the EP-selection list). Model weights/`.onnx` files are git-ignored build artifacts
(`CLAUDE.md` §1) and don't exist in a fresh clone, so the Euler loop's real correctness risk — the
CFG batch-doubling and combination formula, the Euler update — needed to be testable without a
real model file or network access; making `solve_euler` generic over the estimator call achieves
that (ADR-0004) while the numerically-fragile ONNX-vs-PyTorch check stays in `parity_check.py`
alongside every other component's parity check, following the same pattern Milestone 2 established.

**What was rejected**: Writing `solve_euler` directly against `&mut ort::Session` with no Rust-side
math tests (would violate TDD, `CONVENTIONS.md` §1, and leave the CFG/Euler math uncovered in
Rust). Bundling a fixture `.onnx` for Rust tests (violates the no-binary-artifacts constraint and
wouldn't test anything the synthetic-closure test doesn't already cover). See ADR-0004 for the
full rationale.

**What's next**: Milestone 4 — export T3 as decoder-with-past, implement the KV-cache decode loop
+ sampling in `vocalai-core/src/t3.rs` (`docs/issues.md` `VAI-004`) — the main technical risk of
Phase 1 (plan §9 Open Items).

---

## 2026-08-16 — VAI-002: Export HiFiGAN/voice-encoder/S3-tokenizer to ONNX + `parity_check.py`

**What changed**: Set up the export venv (`export/.venv`, Python 3.12 — chatterbox-tts==0.1.7
requires >=3.10) and installed the real toolchain. Added `export/_common.py` (shared model
loading, ONNX export helper, comparison helper), `export/export_hifigan.py`,
`export/export_ve.py`, `export/export_s3tokenizer.py`, and `export/parity_check.py`, plus
`export/tests/test_parity_check.py` (7 tests total, all real — no mocks). All three components
export and pass parity against the PyTorch reference on a fixed input (HiFiGAN
max_abs_diff=5.9e-5, voice encoder 2.1e-7, S3 tokenizer exact-match on discrete tokens; default
tolerance atol=1e-4/rtol=1e-3).

Model loading bypasses `ChatterboxTTS.from_pretrained()` — it unconditionally constructs a
`PerthImplicitWatermarker`, which errors in this chatterbox-tts/resemble-perth combo (perth's
`PerthNet` import silently no-ops on a missing `pkg_resources`/`setuptools`, but
`ChatterboxTTS.__init__` calls it unguarded). Milestone 2 doesn't need T3 or PerthNet anyway, so
`_common.py` downloads and loads only `ve.safetensors`/`s3gen.safetensors` directly.

HiFiGAN's vocoder (`HiFTGenerator.decode()`) calls `torch.stft`/`torch.istft` internally; neither
exports cleanly in this torch version: `return_complex=True` has no ONNX symbolic at all, and
`torch.istft` has none full stop. `export_hifigan.py`'s wrapper reimplements both directions as
ONNX-exportable primitives — manual reflect-pad + `torch.stft(..., return_complex=False)` for
the forward direction (native ONNX `STFT` op), and a precomputed inverse-DFT matrix +
`conv_transpose1d`-as-overlap-add (an identity kernel scatters each windowed frame back to its
hop offset, summing overlaps — the standard iSTFTNet trick) for the inverse, with COLA
window-envelope normalization matching `torch.istft`'s default. It also reimplements
`SourceModuleHnNSF`/`SineGen`'s noise injection using `numpy.random.RandomState`-seeded
constants instead of live `torch.manual_seed` + `Uniform.sample`/`randn_like` — empirically,
the latter does NOT reproduce identically between an eager call and the same call replayed
through `torch.jit.trace` (confirmed with a minimal repro), which was the actual source of an
initial ~1.4e-2 parity failure; numpy's RNG isn't touched by JIT tracing's tensor-op
interception, so it reproduces bit-for-bit in both. `HiFTGenerator.remove_weight_norm()` also
had to be bypassed — chatterbox-tts applies the new parametrize-based `weight_norm` API but
calls the old function-based removal API on it, which raises; `_fuse_weight_norm()` walks the
module tree and removes via the matching (`torch.nn.utils.parametrize`) API instead.

S3 tokenizer's `AudioEncoderV2` precomputes rotary-embedding angles as a complex buffer
(`torch.polar`) and calls `torch.view_as_real` on it every forward — ONNX has no complex dtype,
so tracing fails the moment that buffer is embedded as a graph constant. Fixed by precomputing
the real-valued equivalent (`_real_freqs_cis`, identical math, no complex intermediate) and
patching `torch.view_as_real` to pass through already-real input (falls back to the real
implementation for genuinely complex tensors elsewhere, so this is behavior-preserving outside
this one path).

Added `Makefile`'s `test-py` target preferring `export/.venv/bin/python -m pytest` when that
venv exists (falls back to plain `pytest` otherwise, e.g. in CI, which installs
`export/requirements.txt` into the runner's system Python directly) — otherwise `make check`
would silently use system Python (3.9 here, too old for chatterbox-tts) instead of the export
venv. Documented per-platform (macOS/Linux/Windows) venv setup in `docs/dev-setup.md` — no
wrapper script, per explicit instruction, to keep it trivially auditable/cross-platform without
a bash/PowerShell fork.

**Why**: Milestone 2 proves the export + parity toolchain end-to-end on the three easy/static
components before Milestone 3's Euler ODE loop and Milestone 4's KV-cache decode loop, where a
broken toolchain would be a much more expensive place to discover issues (plan §7 sequencing
rationale). The `ChatterboxTTS.from_pretrained()` bypass and the STFT/ISTFT/RNG/weight-norm
workarounds were all necessary correctness fixes, not stylistic choices — each was verified by
making the failure reproduce, understanding the root cause, and confirming the fix against the
PyTorch reference via `parity_check.py`, not just silencing the export-time error.

**What was rejected**: Making the exported HiFiGAN graph handle arbitrary/dynamic input length —
`F.fold`'s ONNX symbolic (`aten::col2im`) errors on the traced dynamic `output_size` value
(worked around via `conv_transpose1d` instead, which incidentally may also be more dynamic-shape
friendly, but that's untested); the acceptance criteria only require a fixed-input parity check,
so proving full dynamic-length support is deferred to Milestone 6 when real variable-length CLI
audio is wired up. The dynamo-based (`torch.onnx.export(..., dynamo=True)`) exporter was tried
and rejected — it fails on a `torch.no_grad()`/random-sampling interaction inside `SineGen`
unrelated to any of the above ("cannot mutate tensors with frozen storage"), and the legacy
exporter path was already far enough along to be worth finishing instead of switching horses.
Also rejected: a bash-only setup script for the export venv (not cross-platform) and later a
Python setup-script wrapper too, in favor of plain documented per-platform commands in
`docs/dev-setup.md` (explicit user preference).

**What's next**: Milestone 3 — export the S3Gen flow estimator, implement the Euler ODE loop in
`vocalai-core/src/s3gen.rs`, chain into HiFiGAN (`docs/issues.md` `VAI-003`).

---

## 2026-08-16 — Architecture overview doc + diagram

**What changed**: Added `docs/architecture.md` (plain-language companion to `docs/phase1-onnx-rust-cli-plan.md`, written for a reader new to ML systems) and `docs/architecture-diagram.drawio.xml` (visualizes the dev-time Python export pipeline vs. the runtime Rust inference pipeline, including the optional voice-cloning branch and `session.rs`'s cross-cutting EP-selection role).

**Why**: The plan doc is dense and assumes ML background (ONNX, autoregressive decoding, flow matching). This doc explains the module map (`crates/vocalai-core/src/*.rs` responsibilities + milestone status), what Chatterbox's 4 sub-networks do, why ONNX export + Rust over shipping Python (links ADR-002/003), and the end-to-end request flow — so a newcomer can follow the architecture without reverse-engineering it from the plan or the code.

**What was rejected**: Re-deriving or restating the Milestone 2-7 technical decisions already in the plan doc — this is a companion, not a replacement; where the two disagree, the plan doc wins.

**What's next**: Keep this doc in sync as modules move from "planned" to "done" in the module map table (§5).

---

## 2026-08-16 — VAI-001: Cargo workspace + `ort` EP scaffold, export toolchain pins

**What changed**: Pinned `ort = "=2.0.0-rc.13"` in `vocalai-core`, with `coreml`/`cuda` Cargo features that pass through `vocalai-cli` and map to `ort`'s matching features (selected per release artifact via `--features`, per plan §2.3). Added `crates/vocalai-core/src/session.rs`: builds the execution-provider list in explicit fallback order (hardware EPs first, CPU always last, `.fail_silently()` made explicit in code) with real unit tests covering the default (CPU-only) and `coreml`-enabled builds. Pinned `export/requirements.txt` to real, verified versions (`chatterbox-tts==0.1.7`, `onnx==1.22.0`, `onnxruntime==1.28.0`, `pytest`) without installing them yet. Replaced both throwaway placeholder tests (`crates/vocalai-core`'s `#[ignore]`d Rust test, `export/tests/test_scaffold.py`) with real ones. Added ADR-002 (ONNX + Rust runtime, no Python-wrapper interim) and ADR-003 (Rust over C++), transcribing the already-made decisions from `docs/phase1-onnx-rust-cli-plan.md` §2.1/§2.2.

**Why**: Milestone 1 proves the toolchain (workspace, EP feature-gating, export env pins) before any model export work starts, so Milestones 2+ build on a working scaffold instead of discovering Cargo/feature issues mid-export. The EP ordering is a hard constraint (`CLAUDE.md` §1) and needed a real, tested implementation rather than a placeholder.

**What was rejected**: Actually pip-installing `export/requirements.txt` now (defers the multi-GB torch/chatterbox download to when export scripts are first run in Milestone 2). A `session.rs` unit test that detects *runtime* silent CPU fallback (needs a live session against a loaded model — deferred to Milestone 6).

**What's next**: Milestone 2 — export HiFiGAN, voice encoder, and S3 tokenizer to ONNX; stand up `export/parity_check.py` (`docs/issues.md` VAI-002).

---

## 2026-08-16 — ai-sdlc-bootstrap scaffold

**What changed**: Bootstrapped the AI-driven SDLC workflow on this repo via the `ai-sdlc-bootstrap` skill. Added agent-config layer (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`), `docs/agents/` triad, `CONTRIBUTING.md`, `docs/issues.md`, ADR template, pre-commit gate (`make check`), CI workflow, repo hygiene files (README, LICENSE, CODEOWNERS, `.editorconfig`, VS Code settings), and a throwaway placeholder Cargo workspace + Python `export/` stub carrying one intentionally-failing test each as a TDD seed.

**Why**: This project will be developed by humans + multiple AI agents across many sessions. Without the agent-config layer and a strict plan/test/commit workflow, every session would start from zero. The scaffold installs the contract.

**What was rejected**: Commit co-authorship trailer and the trailer-log/pre-commit-block commit-tracking modes — this is a solo-developer repo, convention-only tracking (no hook, no `docs/commit-log.md`) was chosen instead. PR-required merge policy was also rejected in favor of direct-merge-to-main.

**What's next**: Begin Phase 1 Milestone 1 (Cargo workspace scaffold, real `ort` wiring) as tracked in `docs/issues.md` (`VAI-001`).

---

*Add new entries above this line. Format: `## YYYY-MM-DD — Short title`, followed by `What / Why / Rejected / Next` sub-headings.*
