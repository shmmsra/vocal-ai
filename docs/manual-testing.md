# vocal-ai — Manual Testing Runbook

> Canonical runbook for manual verification before commit. Automated tests prove correctness; manual tests prove the feature works end-to-end.
>
> **When you add a new feature, CLI flag, or export step, add a section here in the same commit.**

---

## How to use this file

For every feature area, this file lists:

1. **The exact command(s)** to run
2. **The setup** (env vars, fixtures, preconditions)
3. **What to observe** (log lines, CLI output, WAV output, parity-check numbers)
4. **Pass criteria** (concrete, observable conditions)
5. **Fail indicators** (symptoms that mean the feature is broken or a regression has occurred)

The agent writes these for every new runtime-affecting change. The human runs them before commit approval.

---

## Test template (copy this when adding a new section)

```markdown
### <feature name>

**Test command(s)**:
  <exact shell command(s) to run>

**Setup** (if any):
  <env vars, flags, or preconditions>

**What to observe**:
  <exact log lines, CLI output, WAV output, or parity-check numbers to inspect>

**Pass criteria**:
  <concrete, observable — not "it should work" but "CLI prints X", "WAV output is audible", "parity check reports max delta < tolerance">

**Fail indicators**:
  <symptoms that mean the feature is broken or another feature has regressed>
```

---

## CI split: fast gate vs parity gate vs local-only heavy builds

**Test command(s)**:
```bash
make test-py-fast        # mirrors .github/workflows/ci.yml's Python step
make test-py-parity-ci   # mirrors .github/workflows/parity.yml's Python step (excl. T3)
make test-py-parity      # everything test-py-parity-ci runs, PLUS T3 (local-only, see below)
make check                # everything (fast + parity, incl. T3, plus Rust) — full local gate
```

**Setup**: Same as the export toolchain requirements below. Any `parity`-marked target
downloads real HuggingFace checkpoints on first run.

**What to observe**: `test-py-fast` collects/runs only `test_requirements.py` +
`_common.allclose_report`'s 3 pure tests (4 total, "8 deselected"). `test-py-parity-ci`
runs 7 of the 8 `@pytest.mark.parity` tests — hifigan/ve/s3tokenizer/s3gen/perthnet/
s3gen_flow_encoder/campplus ("5 deselected", since the T3 test carries *two* markers
and gets excluded by `-m "parity and not heavy_build"`). `test-py-parity` runs all 8
(including T3, "4 deselected"). `make check` still runs all 12.

**Pass criteria**:
- `make test-py-fast` / `make test-py-parity-ci` exit 0 in seconds to ~15s, no T3 checkpoint
  download needed for either.
- `make test-py-parity` exits 0 (downloads `t3_cfg.safetensors` too, ~2GB, on a clean
  `~/.cache/huggingface`; the export step alone measures ~9GB peak memory — fine on a real
  dev machine, not on this repo's free-tier CI runner, see ADR-0007).
- `make check` exits 0 and runs all 12 Python tests + all Rust tests (unchanged from before
  the CI split — see ADR-0006).

**Fail indicators**:
- A new parity-style test (downloads a checkpoint, calls a `check_*` function) that isn't
  decorated `@pytest.mark.parity` — it'll silently run in `ci.yml`'s fast job instead of
  `parity.yml`, slowing down/breaking the fast gate (see ADR-0006's "New commitments").
- A new component whose *export* (not just verification) needs more memory than CI has,
  added to `parity.yml` without a `@pytest.mark.heavy_build` marker — will fail CI the same
  way T3 did (see ADR-0007's "New commitments": check this *before* assuming a new
  `check_*` test can run in CI unmodified).
- **Before trusting a local `test-py-parity`/`check_t3` result as proof an export script is
  correct after touching `export/export_t3.py`, clear `models/t3_*.onnx`/`models/t3_*.npy`
  first** — same stale-cache trap as the Milestone 2 warning above: a cached `.onnx` file
  silently skips the real export path, which is exactly where the ~9GB memory cost (and any
  export-time bug) actually lives.
- `test-py-fast` and `test-py-parity` together don't add up to all 12 collected tests —
  means `export/pytest.ini`'s marker registration or a test's decorator is wrong.

---

## Bootstrap sanity check

**Test command(s)**:
```bash
make check
```

**Setup**: Fresh clone; Rust toolchain + Python installed; `make setup-hooks` has been run once.

**What to observe**: Full output of the check pipeline (`cargo fmt --check`, `cargo clippy`, `cargo test --workspace`, `pytest` in `export/`).

**Pass criteria**:
- Exit code 0
- All tests pass (the throwaway placeholder tests in `crates/vocalai-core` and `export/tests/` are `#[ignore]`d / `@pytest.mark.skip`ped, so they don't count against this — remove the marker and replace them with real tests when Milestone 1 starts)
- No clippy warnings, no fmt diffs

**Fail indicators**:
- Exit code non-zero for reasons other than the known placeholder tests
- Pre-commit hook missing or not executable (`.git/hooks/pre-commit` should exist after `make setup-hooks`)

---

## Execution-provider selection (Milestone 1 / VAI-011)

**Test command(s)**:
```bash
cargo test -p vocalai-core session::
cargo test -p vocalai-core --features coreml session::
```

**Setup**: Rust toolchain installed. Second command only meaningful on macOS (CoreML feature).

**What to observe**: Test names and pass/fail output from `crates/vocalai-core/src/session.rs`'s `session::tests` module.

**Pass criteria**:
- Default build (no features): `hardware_execution_providers()` returns an empty list; `resolve_and_build_session(_, ExecutionProviderPreference::Gpu)` returns `Err(SessionError::GpuUnavailable(None))` immediately, without touching the filesystem/ORT.
- `--features coreml` build: `hardware_execution_providers()` includes an entry named `"CoreML"` that downcasts to `CoreML`.
- `ResolvedProvider::Cpu`/`Gpu("CoreML")` `Display` output matches `"CPU execution provider"`/`"GPU execution provider (CoreML)"`.
- All tests pass, no clippy warnings under either feature combination (`cargo clippy --workspace --all-targets -- -D warnings`, then again with `--features vocalai-cli/coreml`).

**Fail indicators**:
- `Gpu` mode ever silently falls back to CPU (must error instead — see `SessionError::GpuUnavailable`).
- A hardware EP is missing when its feature is enabled, or present when it isn't.

---

## CLI: `--use-gpu`/`--use-cpu` execution-provider selection, CPU by default (VAI-011)

**Test command(s)** (from a machine with `models/` already populated — see "CLI: default-voice
end-to-end synthesis" below):
```bash
make build   # auto-detects --features vocalai-cli/coreml (macOS) or vocalai-cli/cuda (else)

./target/release/vocalai --text "Testing default (CPU)." --out /tmp/def.wav --models-dir models
./target/release/vocalai --text "Testing forced GPU." --out /tmp/gpu.wav --models-dir models --use-gpu
./target/release/vocalai --text "Testing forced CPU." --out /tmp/cpu.wav --models-dir models --use-cpu
./target/release/vocalai --text "test" --use-gpu --use-cpu   # expect a clap conflict, no models needed
```

**What to observe**: the `Using <...> execution provider` line printed to stderr right after models
load, and whether each command exits 0 with `Wrote <path>`.

**Pass criteria**:
- No flag: prints `Using CPU execution provider`, succeeds. Same as `--use-cpu` (see
  `docs/decisions/0012-*.md` for why CPU is the default rather than trying GPU first).
- `--use-gpu` on a capable machine: prints `Using GPU execution provider (CoreML)` (or `CUDA`),
  succeeds. **Do not expect it to be faster than CPU** — the tuned CoreML config
  (`CPUAndGPU`/`FastPrediction`/`RequireStaticInputShapes`) only fixes the *severe* naive-config
  regression (was 30-40% slower); real-world testing found no speed improvement over CPU, and one
  full-scale benchmark even measured it ~14% slower (see `docs/decisions/0012-*.md`'s 2026-08-20
  correction). `--use-gpu` on a build with no `coreml`/`cuda` feature compiled in, or on hardware
  that can't register the EP: exits 1 with an actionable `SessionError::GpuUnavailable` message
  (build without the hardware feature to reproduce this deterministically).
- `--use-cpu`: always prints `Using CPU execution provider` and succeeds, regardless of what
  hardware is present.
- `--use-gpu --use-cpu` together: clap rejects it before any model loads (`error: the argument
  '--use-gpu' cannot be used with '--use-cpu'`), exit code 2.
- On CoreML specifically: no `error: ONNX Runtime session error: ... CoreML ...` mid-run — this
  would indicate the `s3gen_estimator.onnx` CPU pin (VAI-011/ADR-0012) regressed; check
  `pipeline.rs::ModelBundle::load`'s `estimator_provider` special-case is still present until
  VAI-014 removes it.

**Fail indicators**:
- `--use-gpu` silently produces `Using CPU execution provider` — the whole point of the flag is to
  never do this.
- `--use-gpu` dramatically slower than `--use-cpu` (i.e. back to the original ~30-40% naive-config
  regression, not just "not obviously faster") — would indicate the CoreML config tuning regressed
  (check `session.rs::hardware_execution_providers`'s `.with_compute_units`/
  `.with_specialization_strategy`/`.with_static_input_shapes` calls are still present). A small,
  inconclusive difference either way is expected — see the pass criteria above.

---

## Export toolchain requirements (Milestone 1)

**Test command(s)**:
```bash
cd export && pytest
```

**Setup**: Python 3 installed. Does **not** require the packages in `requirements.txt` to actually be installed — the test only parses the file.

**What to observe**: `export/tests/test_requirements.py` pass/fail output.

**Pass criteria**: Test confirms `chatterbox-tts`, `onnx`, and `onnxruntime` are each pinned to an exact version in `export/requirements.txt`.

**Fail indicators**: Any of the three packages missing or unpinned (no `==version`).

---

---

## ONNX export + parity check: HiFiGAN, voice encoder, S3 tokenizer (Milestone 2)

**Test command(s)**:
```bash
cd export
python3 -m venv .venv   # if not already created — see docs/dev-setup.md §2
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python export_hifigan.py
python export_ve.py
python export_s3tokenizer.py
python parity_check.py
```

**Setup**: Python 3.10+ (chatterbox-tts requires it). First run downloads
`ve.safetensors`/`s3gen.safetensors` (~1-2 GB) from the HuggingFace Hub (`ResembleAI/chatterbox`)
— needs network access. `models/` is git-ignored; exported `.onnx` files land there.

**What to observe**:
- Each `export_*.py` prints `Exported <component> to models/<name>.onnx` with no traceback.
- `parity_check.py` prints one `[PASS]`/`[FAIL]` line per component with `max_abs_diff`.

**Pass criteria**:
- All three `export_*.py` scripts exit 0 and produce a `.onnx` file under `models/`.
- `parity_check.py` exits 0 and prints `[PASS]` for `hifigan`, `ve`, and `s3tokenizer` — expect
  `max_abs_diff` around `~5e-5`, `~2e-7`, and `0.0` (exact match — discrete tokens) respectively.
- `python -m pytest` in `export/` (or `make check` from the repo root) passes
  `export/tests/test_parity_check.py`'s 6 tests + `test_requirements.py`'s 1.

**Fail indicators**:
- Any `[FAIL]` line, or `max_abs_diff` far above `1e-4`/`1e-3` (atol/rtol) — check whether
  `export_hifigan.py`'s deterministic-noise reimplementation (`_sine_gen_deterministic`) or the
  `_stft_onnx`/`_istft_onnx` primitives were touched; both are numerically fragile (see
  `docs/CHANGELOG.md`'s VAI-002 entry for what makes each one work).
- Import errors for `torch`/`chatterbox`/`onnx`/`onnxruntime` — the venv isn't activated or
  `requirements.txt` wasn't (re-)installed after a change.
- `PerthImplicitWatermarker`/`pkg_resources` errors — means something started calling
  `ChatterboxTTS.from_pretrained()` again instead of `_common.py`'s narrower
  `load_voice_encoder()`/`load_s3gen()` loaders (Milestone 2 doesn't need T3 or PerthNet).

**Important**: `models/` is git-ignored, so a locally-cached `.onnx` file silently skips each
`check_*()`'s `export()` call (`if not onnx_path.exists(): export_*.export(...)`) — masking any bug
in the export path itself (a stale-cache-masked bug like this bit CI once; see the "fix: S3
tokenizer export corrupted `freqs_cis`" CHANGELOG entry). **Before trusting a green
`parity_check.py`/`pytest` run as proof an export script is correct, clear `models/*.onnx` first**
to force every component through a real fresh export, not just a parity check against a
already-exported (possibly stale) file.

---

## ONNX export + parity check: S3Gen flow estimator → HiFiGAN (Milestone 3)

**Test command(s)**:
```bash
cd export
source .venv/bin/activate  # if not already active — see docs/dev-setup.md §2

python export_s3gen.py
python parity_check.py --component s3gen
```

**Setup**: Same venv/checkpoint as the Milestone 2 export (`ve.safetensors`/`s3gen.safetensors`
from `ResembleAI/chatterbox`). Requires `models/hifigan.onnx` to already exist (run
`python export_hifigan.py` first if starting from a clean `models/` — `parity_check.py` will
also export it on demand if missing).

**What to observe**:
- `export_s3gen.py` prints `Exported S3Gen flow estimator to models/s3gen_estimator.onnx` with no
  traceback.
- `parity_check.py --component s3gen` prints one `[PASS]`/`[FAIL]` line with `max_abs_diff`.

**Pass criteria**:
- `export_s3gen.py` exits 0 and produces `models/s3gen_estimator.onnx`.
- `parity_check.py --component s3gen` exits 0 and prints `[PASS] s3gen: max_abs_diff=...`
  — expect an order of `~1e-4` or smaller (this check reports the worse of the intermediate-mel
  and final-waveform max_abs_diff, chaining through `models/hifigan.onnx`).
- `python -m pytest` in `export/` (or `make check` from the repo root) passes
  `export/tests/test_parity_check.py::test_s3gen_export_matches_pytorch_reference_mel_to_waveform`.
- `cargo test -p vocalai-core s3gen::` passes all 5 tests (`cosine_t_span_*`, `solve_euler_*`) —
  these are pure `ndarray` math tests against a synthetic linear estimator, so they run offline
  with no `models/` directory required.

**Fail indicators**:
- Any `[FAIL]` line, or `max_abs_diff` far above `1e-4`/`1e-3` (atol/rtol) — `hifigan.onnx`'s
  `speech_feat` input is a dynamic axis as of VAI-009, so `export_s3gen.py`'s `EXAMPLE_FRAMES`
  no longer needs to match any HiFiGAN-side constant; check instead whether
  `parity_check.py::_solve_euler_onnx` still matches `crates/vocalai-core/src/s3gen.rs::solve_euler`
  and `ConditionalCFM.solve_euler`'s CFG-doubling/combination math exactly (see ADR-0004).
- A Rust `s3gen::` test failure with no `models/` directory present indicates an actual math bug
  in `solve_euler`/`cosine_t_span`, not a missing-fixture issue — these tests never touch ONNX
  Runtime.

---

## ONNX export + parity check: T3 decoder-with-past (Milestone 4)

**Test command(s)**:
```bash
cd export
source .venv/bin/activate  # if not already active — see docs/dev-setup.md §2

python export_t3.py
python parity_check.py --component t3
```

**Setup**: Same venv as prior milestones. First run downloads `t3_cfg.safetensors` from
`ResembleAI/chatterbox` (in addition to `ve.safetensors`/`s3gen.safetensors`). No `--voice`
reference audio needed — `check_t3` uses a fixed-seed synthetic conditioning fixture (random
speaker embedding / cond-prompt tokens / emotion value), matching the pattern `check_s3gen` uses.

**What to observe**:
- `export_t3.py` prints four `Exported ...` lines with no traceback: `t3_cond_prefill.onnx`,
  `t3_decoder.onnx`, `t3_speech_emb.npy`, `t3_speech_pos_emb.npy` under `models/`.
- `parity_check.py --component t3` prints one `[PASS]`/`[FAIL]` line with `max_abs_diff`.

**Pass criteria**:
- `export_t3.py` exits 0 and produces all four files under `models/`.
- `parity_check.py --component t3` exits 0 and prints `[PASS] t3: max_abs_diff=...` — expect an
  order of `~5e-5` (well within `atol=1e-4`). This means both: (a) a **greedy** (argmax) 6-token
  free-running decode driving the exported ONNX graphs produced the exact same token sequence as a
  greedy replica of the real reference forward pass, and (b) the per-step processed logits agree
  within tolerance. See `docs/decisions/0005-t3-hand-rolled-decoder-export.md` and the VAI-004
  CHANGELOG entry for why this check uses greedy decoding rather than comparing stochastic samples
  (PyTorch's and any Rust-side RNG are unrelated, so free-running sampled sequences can't be
  compared across languages).
- `python -m pytest` in `export/` (or `make check` from the repo root) passes
  `export/tests/test_parity_check.py::test_t3_export_matches_pytorch_reference_greedy_decode`.
- `cargo test -p vocalai-core t3::` passes all 22 tests — pure `ndarray`/`rand` math tests (CFG
  combine, repetition penalty, temperature, min-p, top-p, greedy/multinomial selection, embedding
  lookup, `.npy` round-trip, a synthetic end-to-end decode loop) — these run offline, no `models/`
  directory or ONNX Runtime session required.

**Fail indicators**:
- Any `[FAIL]` line, or `max_abs_diff` far above `1e-4`/`1e-3` (atol/rtol) — check whether
  `T3DecoderExport`/`_ExportDecoderLayer` (`export/export_t3.py`) still matches
  `transformers.models.llama.modeling_llama`'s RMSNorm/RoPE/attention/MLP math exactly (this repo
  pins `transformers==5.2.0`; a version bump could change internals subtly — see ADR-0005), or
  whether `T3CondPrefillExport` still reproduces `T3.prepare_input_embeds()` + `T3.inference()`'s
  double-BOS-embedding construction faithfully.
- A token-sequence mismatch (not just a logits tolerance miss) between the greedy PyTorch and
  greedy ONNX runs points at a KV-cache bookkeeping bug (wrong layer/key-vs-value ordering in the
  stacked `past_kv`/`present_kv` tensor, or a RoPE position-id off-by-one across the prefill→decode
  boundary) rather than a small numerical-precision issue.
- A Rust `t3::` test failure with no `models/` directory present indicates an actual math bug in
  the sampling/decode-loop logic, not a missing-fixture issue — these tests never touch ONNX
  Runtime (synthetic decoder closures and hand-computed expected values only).

---

## ONNX export + parity check: PerthNet watermark encoder (Milestone 5)

**Test command(s)**:
```bash
cd export
source .venv/bin/activate  # if not already active — see docs/dev-setup.md §2

python export_perthnet.py
python parity_check.py --component perthnet
```

**Setup**: Same venv as prior milestones, plus `resemble-perth==1.0.1` and `setuptools<81`
(pinned in `export/requirements.txt` — `setuptools>=81` dropped `pkg_resources`, which
`resemble-perth` needs to locate its bundled checkpoint; without the pin, `perth.PerthImplicitWatermarker`
silently becomes `None` and any real use of it raises `TypeError: 'NoneType' object is not
callable`). No download needed — PerthNet's weights ship inside the `resemble-perth` package
itself (`perth/perth_net/pretrained/implicit/perth_net_250000.pth.tar`), unlike VE/S3Gen/T3, which
pull from `ResembleAI/chatterbox` on HuggingFace.

**What to observe**:
- `export_perthnet.py` prints one `Exported PerthNet encoder to ...` line with no traceback,
  producing `models/perthnet_encoder.onnx`.
- `parity_check.py --component perthnet` prints one `[PASS]`/`[FAIL]` line with `max_abs_diff`.

**Pass criteria**:
- `export_perthnet.py` exits 0 and produces `models/perthnet_encoder.onnx`.
- `parity_check.py --component perthnet` exits 0 and prints `[PASS] perthnet: max_abs_diff=...`
  (expect an order of `1e-5`–`1e-6`, similar to `hifigan`/`ve`) — the exported graph is
  `PerthNet.encoder`'s full `Encoder.forward` (subband crop, Conv1d residual stack, magnitude-mask,
  residual-add all included), so this one check covers the entire learned/exported piece.
- `python -m pytest` in `export/` (or `make check` from the repo root) passes
  `export/tests/test_parity_check.py::test_perthnet_export_matches_pytorch_reference`.
- `cargo test -p vocalai-core watermark::` passes all 7 tests — pure DSP math tests (Hann-window
  values, reflect-pad convention, dB normalize/denormalize round-trip, a synthetic-signal
  STFT→ISTFT round trip, resample duration preservation, an identity-encoder end-to-end
  `apply_watermark` round trip, encoder-error propagation) — these run offline, no `models/`
  directory or ONNX Runtime session required.

**Known gap — no live audio to listen to yet**: unlike HiFiGAN/S3Gen, this module has no
end-to-end "does it sound right" check available yet — `vocalai-cli` is still a placeholder
(Milestone 6 wires the real pipeline). `watermark.rs`'s STFT/ISTFT/resample math also has no
PyTorch-reference parity check the way the exported networks do (classical DSP isn't
ONNX-exported, so `CLAUDE.md` §1's constraint doesn't gate it) — correctness rests on the Rust
round-trip unit tests above, plus a one-time manual spot-check of `stft_magphase` against a live
`AudioProcessor.signal_to_magphase` call (documented in `watermark.rs`'s module doc comment: the
signal-carrying frequency bin matched to ~1e-7). `rubato`'s resampler is not bit-exact with
librosa's default `soxr_hq` — this is an accepted, documented residual risk (see
`docs/agents/STATUS.md`), to be revisited once Milestone 6 produces real audio to listen to.

**Fail indicators**:
- Any `[FAIL]` line, or `max_abs_diff` far above `1e-4`/`1e-3` (atol/rtol) — check whether
  `PerthEncoderWrapper` (`export/export_perthnet.py`) still matches `PerthNet.encoder`'s actual
  submodule (subband/hidden-size mismatch would show up as a shape error at export time, not a
  numerical parity failure).
- A `watermark::` Rust test failure indicates an actual DSP bug (windowing, frame/hop arithmetic,
  dB normalization, or overlap-add/COLA math) — these tests never touch ONNX Runtime.
- If `load_perthnet()` (`export/_common.py`) raises `TypeError: 'NoneType' object is not
  callable` or an `ImportError` mentioning `pkg_resources`, check `setuptools<81` is actually
  installed in the active venv (`pip show setuptools`) — a stale venv or a broadened pin is the
  likely cause, not a code regression.

---

## ONNX export + parity check: S3Gen flow encoder (bucketed) + CAMPPlus (VAI-008)

**Test command(s)**:
```bash
cd export
source .venv/bin/activate  # if not already active — see docs/dev-setup.md §2

python export_s3gen_flow_encoder.py   # writes models/s3gen_flow_encoder_{200,400,600,800,1000,1200}.onnx
python export_campplus.py             # writes models/campplus.onnx + s3gen_spk_embed_affine_{weight,bias}.npy
python export_default_voice.py        # writes models/default_voice/*.npy (no parity check — see below)

python parity_check.py --component s3gen_flow_encoder
python parity_check.py --component campplus
```

**Setup**: Same venv as prior milestones — downloads `s3gen.safetensors` if not already cached
(shared with Milestone 3's `export_s3gen.py`/`export_s3tokenizer.py`; `export_default_voice.py`
additionally downloads `conds.pt`, both from `ResembleAI/chatterbox` on HuggingFace).

**What to observe**:
- `export_s3gen_flow_encoder.py` prints six `Exported S3Gen flow encoder bucket to ...` lines
  (one per `TOKEN_BUCKETS` entry: 200/400/600/800/1000/1200), producing six `.onnx` files.
- `export_campplus.py` prints `Exported CAMPPlus to ...` plus a second line naming the two
  `.npy` affine-layer weight files.
- `export_default_voice.py` prints one `Wrote ...` line per tensor field (7 total: T3's
  `speaker_emb`/`cond_prompt_speech_tokens`/`emotion_adv`, S3Gen's
  `prompt_token`/`prompt_token_len`/`prompt_feat`/`embedding`).
- `parity_check.py --component s3gen_flow_encoder` prints one `[PASS]`/`[FAIL]` line covering
  *all six buckets* (see `check_s3gen_flow_encoder`'s docstring — each bucket is checked both for
  ONNX-vs-eager match and for padding invariance).
- `parity_check.py --component campplus` prints one `[PASS]`/`[FAIL]` line for the single
  400-frame graph.

**Pass criteria**:
- All three export scripts exit 0 and produce the files listed above.
- `parity_check.py --component s3gen_flow_encoder` exits 0, `[PASS]`, `max_abs_diff` on the
  order of `1e-5`–`1e-6`.
- `parity_check.py --component campplus` exits 0, `[PASS]`, `max_abs_diff` on the order of
  `1e-5`–`1e-6`.
- `python -m pytest` in `export/` (or `make check`) passes both
  `test_s3gen_flow_encoder_export_matches_pytorch_reference` and
  `test_campplus_export_matches_pytorch_reference`.

**Known gap — bucketed, not dynamic, by necessity**: both graphs were originally attempted as
single dynamic-length exports and found broken — see ADR-0009 for the full diagnosis
(`EspnetRelPositionalEncoding`'s Python-int `size` argument bakes the tracing length for the flow
encoder; `CAMLayer.seg_pooling`'s trim-to-original-length op only round-trips correctly through
ONNX when it's a no-op, requiring CAMPPlus's frame count to be a multiple of 200). Rust
(Milestone 6, Part B) must pick the right bucket / assemble exactly `CAMPPLUS_FRAMES` real frames
— this is load-bearing correctness logic, not an optimization, and should be exercised by its own
tests when written.
`export_default_voice.py` has no parity check — it only copies tensors out of `conds.pt`
unchanged (`torch.load` isn't reachable from Rust), so there is nothing to compare against.

**Fail indicators**:
- Any `[FAIL]` line, or `max_abs_diff` far above `1e-4`/`1e-3` — re-check
  `export_s3gen_flow_encoder.py`'s `make_pad_mask(..., max_len=...)` calls specifically (an
  earlier version of this wrapper omitted `max_len` and silently computed masks against the
  *padded* length instead of the bucket's physical length, causing a shape-mismatch crash, not a
  quiet numerical drift — see git history if this regresses).
- `export_campplus.py`'s `assert CAMPPLUS_FRAMES % 200 == 0` firing means someone changed the
  constant without re-reading the module docstring's `seg_pooling` explanation.
- Before trusting a local parity result after touching either export script, clear the relevant
  `models/s3gen_flow_encoder_*.onnx` / `models/campplus.onnx` files first — same stale-cache trap
  documented in the Milestone 2/4 sections above.

---

## ONNX export + parity check: dynamic-length HiFiGAN (VAI-009)

**Test command(s)**:
```bash
cd export
source .venv/bin/activate  # if not already active — see docs/dev-setup.md §2

rm -f ../models/hifigan.onnx   # force a real re-export, not a stale cached graph
python export_hifigan.py
python parity_check.py --component hifigan
```

**Setup**: Same venv/checkpoint as Milestone 2 (`s3gen.safetensors` from `ResembleAI/chatterbox`).

**What to observe**:
- `export_hifigan.py` prints `Exported HiFiGAN to models/hifigan.onnx` with no traceback.
- `parity_check.py --component hifigan` prints one `[PASS]`/`[FAIL]` line with `max_abs_diff`.
- `onnx.load("models/hifigan.onnx").graph.input[0].type.tensor_type.shape.dim` shows a
  `dim_param` (not a fixed `dim_value`) on `speech_feat`'s time axis — confirms the export is
  genuinely dynamic, not merely re-baked at a different fixed length.

**Pass criteria**:
- `export_hifigan.py` exits 0 and produces `models/hifigan.onnx` with a dynamic `speech_feat` time
  axis.
- `parity_check.py --component hifigan` exits 0, `[PASS]`, `max_abs_diff` on the order of `5e-5` —
  this check now exercises three frame counts (`HIFIGAN_CHECK_FRAME_COUNTS = (17, 50, 123)` in
  `parity_check.py`), not just the one the original fixed-shape export happened to use.
- `python -m pytest` in `export/` (or `make check`) passes
  `export/tests/test_parity_check.py::test_hifigan_export_matches_pytorch_reference`.
- The CLI's own acceptance-criterion command (see the next section) succeeds for arbitrary text,
  not just text that happens to land on a specific token/frame count.

**Fail indicators**:
- Any `[FAIL]` line, or an ONNX Runtime broadcast/shape error at any frame count other than the one
  the graph happened to be traced at — points at a new instance of the same "Python-int-off-`.shape`
  gets baked as an ONNX constant" bug class this fix addressed (see `export_hifigan.py`'s module
  docstring: the overlap-add envelope's old `.repeat(1, 1, num_frames)` and the deterministic source
  noise's old `torch.tensor(rng.randn(*shape))` were both instances of it — ADR-0009 documents the
  same category of bug in the flow-encoder/CAMPPlus exports).
- Before trusting a local parity result after touching `export_hifigan.py`, clear
  `models/hifigan.onnx` first — same stale-cache trap documented in earlier sections; a cached
  fixed-shape graph from before this fix would otherwise silently mask a regression.

---

## CLI: default-voice end-to-end synthesis (VAI-006, part B.1)

**Prerequisites**:
- The full `models/` directory must be populated. **The single, authoritative, cross-platform
  (macOS/Linux + Windows) command list for generating every required artifact — plus the build and
  run commands — lives in `docs/dev-setup.md` §9.** Follow §9.1 (generate model files) once, then
  §9.2 (build). Don't re-derive the export order from the per-milestone sections above.
- For the record, `ModelBundle::load` (`crates/vocalai-core/src/pipeline.rs`) requires exactly:
  `tokenizer.json`, `t3_cond_prefill.onnx`, `t3_decoder.onnx`, `t3_speech_emb.npy`,
  `t3_speech_pos_emb.npy`, `s3gen_estimator.onnx`, `s3gen_flow_encoder_{200,400,600,800,1000,1200}.onnx`,
  `hifigan.onnx`, `perthnet_encoder.onnx`, `s3gen_spk_embed_affine_{weight,bias}.npy`, and
  `default_voice/*.npy`. `ve.onnx`, `s3tokenizer.onnx`, and `campplus.onnx` are loaded lazily, only
  on the first `--voice` call (see the part B.2 section below) -- the default-voice-only path never
  pays their load cost.

**Test command(s)** (see `docs/dev-setup.md` §9 for the full per-platform export list this assumes
is already done; `--out /tmp/out.wav` shown for POSIX, use `--out out.wav` on Windows):
```bash
cargo build --release -p vocalai-cli
./target/release/vocalai --text "hello world" --out /tmp/out.wav --models-dir models
```

**Fixed by VAI-009**: this exact command used to fail with a shape-mismatch error
(`hifigan.onnx`'s ONNX export was hard-fixed at exactly 50 mel frames; see
`docs/decisions/`/`docs/CHANGELOG.md`'s VAI-009 entry). It now succeeds for arbitrary text.
Also verified with a much longer sentence (~145 words, ~7.4s of audio) to confirm the fix
generalizes across frame counts rather than coincidentally matching one:
```bash
./target/release/vocalai \
  --text "This is a longer sentence to make sure the dynamic length HiFiGAN export truly generalizes across many different mel frame counts, not just one." \
  --out /tmp/out_long.wav --models-dir models
```

**What to observe**:
- Both commands print a `Using GPU execution provider (...)`/`Using CPU execution provider` line
  (VAI-011) to stderr right after models load, then `Wrote <path>` and exit 0 — no `speech_feat`
  shape-mismatch error.
- `/tmp/out.wav` is a mono 24000 Hz 16-bit PCM WAV, ~0.9s long; `/tmp/out_long.wav` is ~7.4s long.
- Inspect programmatically if you can't listen: `python3 -c "import wave; w=wave.open('/tmp/out.wav'); print(w.getnchannels(), w.getframerate(), w.getnframes())"`.
- The waveform should not be silence/NaN: peak amplitude and RMS should both be clearly nonzero
  (verified during implementation: `out.wav` peak ~27933/32767 RMS ~4274; `out_long.wav` peak
  ~32605/32767 RMS ~3962).
- Listen to both — this is also the first real chance to sanity-check `watermark.rs`'s
  resampler-fidelity residual risk by ear (see `docs/agents/STATUS.md`).

**Pass criteria**:
- Both commands (short and long text) exit 0 and produce non-silent, audible speech-like WAVs of
  plausible, different durations.
- `cargo build --release -p vocalai-cli` succeeds with no warnings.

**Fail indicators**:
- `error: failed to load models from ...`: a required `.onnx`/`.npy`/`tokenizer.json` file is
  missing from `--models-dir` -- re-check the prerequisites list above.
- `error: T3 generated no speech tokens for this text`: T3 emitted EOS immediately -- try a longer
  or different input text.
- `Got invalid dimensions for input: speech_feat ...`: a regression in VAI-009's dynamic-length
  fix -- see the "ONNX export + parity check: dynamic-length HiFiGAN (VAI-009)" section above.
- Silence (all-zero samples) or clipped-flat output: a real bug in the T3/S3Gen/HiFiGAN/watermark
  wiring -- investigate `pipeline::synthesize`'s tensor assembly.

---

## CLI: `--voice` zero-shot cloning end-to-end synthesis (VAI-006, part B.2)

**Prerequisites**:
- Same `models/` directory as the default-voice section above, plus `ve.onnx`, `s3tokenizer.onnx`,
  and `campplus.onnx` (all already produced by Milestone 2/VAI-008's export steps -- no new export
  step needed for this ticket).
- A short (a few seconds to ~10s) mono or stereo WAV reference clip. Any real speech recording
  works; a previous `vocalai` output WAV (e.g. `/tmp/out.wav` from the default-voice section) is a
  convenient stand-in if you don't have one handy.

**Test command(s)**:
```bash
cargo build --release -p vocalai-cli
./target/release/vocalai --text "This is a voice cloning test." \
  --voice /tmp/out.wav --out /tmp/cloned.wav --models-dir models
```

**What to observe**:
- Prints `Wrote /tmp/cloned.wav` and exits 0.
- `/tmp/cloned.wav` is a mono 24000 Hz 16-bit PCM WAV whose duration is plausible for the given text
  (a few seconds), not silent: `python3 -c "import wave,struct; w=wave.open('/tmp/cloned.wav'); f=w.readframes(w.getnframes()); s=struct.unpack('<%dh'%w.getnframes(),f); print(w.getparams()); print('peak', max(abs(x) for x in s))"`.
- Listen to it: the voice timbre should audibly differ from the built-in default voice (run the
  default-voice command from the section above for an A/B comparison) and should bear some
  resemblance to the reference clip's speaker, though this repo has no automated speaker-similarity
  metric -- judge by ear.
- First run against a fresh `--models-dir` is slower than the default-voice path (loads three more
  ONNX sessions -- `ve.onnx`, `s3tokenizer.onnx`, `campplus.onnx` -- lazily on this first `--voice`
  call); the default-voice path (no `--voice` flag) should show no such slowdown, confirming the
  lazy-load only triggers when actually needed.

**Pass criteria**:
- Exits 0, produces a non-silent WAV of plausible duration.
- The default-voice command (no `--voice`) still works afterwards with no regression.

**Fail indicators**:
- `error: failed to read --voice reference audio: ...`: the reference path doesn't exist or isn't a
  WAV file (this pipeline is WAV-only, matching `audio.rs`'s existing convention -- no mp3/flac).
- `error: failed to resample --voice reference audio: ...`: an unusual/invalid sample rate in the
  reference file.
- `error: ONNX Runtime session error: ...`: check `models/ve.onnx`/`s3tokenizer.onnx`/`campplus.onnx`
  are present and not corrupted.
- Silence or garbled/noise-only output: a real bug in `pipeline::VoiceConditioning::from_reference`'s
  tensor assembly, `mel.rs`'s DSP, or `voice_encoder.rs`'s partial-utterance striding -- these have
  no automated cross-language parity gate (see the residual-risk note in `mel.rs`'s module doc and
  the new ADR), so a wrong-sounding clone (not merely a nonzero-vs-silent bug) is the kind of issue
  only this manual listen would catch.

**Verified during implementation (2026-08-18)**: the repo owner's first manual run of this exact
test (with a real, longer reference clip, `tmp/sample-voice-1.wav`) caught a real bug --
`synthesize` was silently reading the S3Gen prompt-token *values* from the built-in default voice
regardless of `--voice` (a leftover from the `DefaultVoice` -> `VoiceConditioning` rename's
find-replace missing one multi-line occurrence), causing a `prompt_feat's N frames exceed
total_mel_len=M` panic in `s3gen::build_cond` once the two diverged enough. The
pre-fix self-check happened to use a short reference clip that didn't trip the mismatch, so it
shipped undetected until this manual test. Fixed by reading `voice.s3gen_prompt_token` instead of
`bundle.default_voice.s3gen_prompt_token`; see `docs/CHANGELOG.md`'s VAI-006 part B.2 entry for the
full diagnosis. Post-fix: `vocalai --text "This is a voice cloning test." --voice
tmp/sample-voice-1.wav --out tmp/cloned.wav --models-dir models` produced a 52800-frame (~2.2s)
mono 24kHz WAV, peak amplitude 32767/32767, RMS ~3937 -- genuine non-silent audio, no crash. The
default-voice command was re-run immediately after and still succeeded with no regression.
Audible confirmation that the clone actually resembles the reference speaker is still outstanding.

---

### Cross-platform build gate: `make check` on Windows (ADR-0010)

**Test command(s)** (from repo root):

```bash
make check
```

**Setup**:
- Rust toolchain (`cargo`), GNU `make`, and the `export/.venv` all installed per `docs/dev-setup.md`.
- On Windows, run from a shell whose PATH already includes `cargo` and `make` (open a *new* shell
  after installing them — see `docs/dev-setup.md` §8). No `sh`/Git Bash is required; the recipes
  run under `cmd.exe`.

**What to observe**:
- `cargo test --workspace` **links and runs** — it does not abort at the MSVC linker.
- The `test-py` step launches the export venv interpreter and runs pytest.

**Pass criteria**:
- `fmt-check` + `clippy` clean, all Rust tests pass, all Python tests pass; `make check` exits 0.

**Fail indicators**:
- `LNK2038: mismatch detected for 'RuntimeLibrary' (MD_DynamicRelease vs MT_StaticRelease)`:
  the `esaxx_fast`/static-CRT regression is back — confirm `crates/vocalai-core/Cargo.toml` still
  pins `tokenizers` with `default-features = false` (no `esaxx_fast`). See ADR-0010.
- `-x was unexpected at this time` or `'.venv' is not recognized`: the `test-py*` Makefile recipes
  reverted to POSIX-shell syntax / forward-slash exe paths that `cmd.exe` can't run. See ADR-0010.

---

### Export pipeline: `make export` wrapper script

**Test command(s)** (from repo root):

```bash
make help                              # confirm the new line is listed
make export                            # default-voice path, 8 scripts
make export ARGS=--with-voice-cloning  # + ve.onnx/s3tokenizer.onnx for --voice
```

**Setup**:
- `export/.venv` installed per `docs/dev-setup.md` §2 (the script errors out with a clear message
  if it's missing, rather than falling back to system Python).

**What to observe**:
- `make help` lists `make export`.
- `make export` prints one `→ Running export/<script>.py...` line per script, in the §9.1 order
  (`fetch_tokenizer.py`, `export_t3.py`, `export_s3gen.py`, `export_s3gen_flow_encoder.py`,
  `export_hifigan.py`, `export_perthnet.py`, `export_campplus.py`, `export_default_voice.py`),
  then `✓ All model artifacts generated in <repo>/models/`.
- `make export ARGS=--with-voice-cloning` runs the same 8 plus `export_ve.py`/
  `export_s3tokenizer.py` at the end.
- An unrecognized flag (e.g. `bash scripts/export-all.sh --bogus`) fails fast with
  `error: unknown argument '--bogus' (expected --with-voice-cloning)` and a nonzero exit, before
  running anything.

**Pass criteria**:
- Both invocations exit 0 and populate `models/` with the full artifact set (cross-check against
  the table in `docs/dev-setup.md` §9.1).
- The Windows path (`scripts\export-all.ps1`, dispatched automatically by `make export` when
  `$(OS)` is `Windows_NT`) behaves identically — same flag, same script order, same error text.
  This needs a real Windows run to confirm; it has only been read-reviewed against the POSIX
  script, not executed, as of this writing.

**Fail indicators**:
- `error: ... not found — set up the export venv first`: `export/.venv` doesn't exist yet or was
  built with a different interpreter path than `export/.venv/bin/python` (POSIX) /
  `export\.venv\Scripts\python.exe` (Windows).
- Any script's own failure (e.g. a HuggingFace download error, an ONNX export bug) aborts the
  whole sequence immediately (`set -euo pipefail` / `$ErrorActionPreference = "Stop"`) rather than
  continuing past a broken artifact.

**Verified during implementation (2026-08-18)**: `make export` and
`make export ARGS=--with-voice-cloning` were both run end-to-end on macOS and completed
successfully, regenerating every artifact under `models/` (including a real T3 export). The bad-arg
path (`bash scripts/export-all.sh --bogus`) was also run and failed as expected. The `.ps1` variant
has not yet been executed on Windows — pending a follow-up manual run.

---

### Release packaging pipeline: model publish + per-platform build (VAI-007/VAI-016, ADR-0013/ADR-0014)

Both `models-export.yml` and `release.yml` are structural-only in CI — no real inference, no
audio synthesis (per the repo owner's explicit instruction; see ADR-0013). The steps below are
the manual, human-run counterpart that actually exercises a built artifact end-to-end, and must
be run at least once per platform before announcing a release. Since VAI-016 (ADR-0014), both
workflows are primarily triggered by a version-bump pushed to `main`, not manual dispatch/tags —
the commands below use that path; the manual fallbacks (`gh workflow run`, direct `git tag`) still
work unchanged if you want to test a specific revision without bumping either version file.

**Test command(s)**:

```bash
# 1. Publish models (only if models/ changed since the last publish) -- bump MODELS_VERSION and
#    push to main; skip if the version is unchanged (models-export.yml will no-op):
$EDITOR MODELS_VERSION && git add MODELS_VERSION && git commit -m "models: bump to X.Y.Z" && git push
gh run watch   # wait for it to finish

# 2. Cut (or dry-run) a release build -- bump Cargo.toml's [workspace.package] version and push:
$EDITOR Cargo.toml && git add Cargo.toml && git commit -m "release: bump to vX.Y.Z" && git push
gh run watch
# manual fallback, e.g. to test a one-off tag without a real version bump:
#   git tag v0.0.0-test && git push origin v0.0.0-test   # or: gh workflow run release.yml

# 3. Install exactly the way an end user would (release binary from GitHub, models from the
#    public HF repo -- see ADR-0013's 2026-08-21 addendum for why these are no longer bundled
#    into one release asset: GitHub caps release assets at 2GiB, the ~4GB model set doesn't fit):
bash scripts/install.sh   # or: irm .../install.ps1 | iex on Windows

# 4. Real end-to-end synthesis (the thing CI deliberately does NOT do):
./vocalai/vocalai --text "hello world" --out out.wav --models-dir ./vocalai/models
./vocalai/vocalai --text "hello world" --out out-cpu.wav --models-dir ./vocalai/models --use-cpu
```

**Setup**: `HF_TOKEN` repo secret already configured (`docs/dev-setup.md` §10.1); `gh` CLI
authenticated.

**What to observe**:
- `models-export.yml`'s job summary/logs: `make test-py-parity` passes (including T3), the
  smoke-test step reports `smoke test passed: N files validated`, and the publish step prints
  `published to https://huggingface.co/shmmsra/vocal-ai-models @ <sha>`.
- `release.yml`'s job summary: `Bundling shmmsra/vocal-ai-models@<sha>` for each of the 3
  matrix jobs, and each job's smoke-test step passes.
- Step 4's two WAV files are both audible, non-silent, correct speech — `out.wav` (default EP)
  and `out-cpu.wav` (forced `--use-cpu`) should sound equivalent (plan §8's CPU-fallback
  criterion).
- (Optional, plan §8's memory criterion) run `while true; do sysctl vm.swapusage; sleep 1; done`
  during step 4 on macOS and compare swap growth against the ~5GB→~30GB PyTorch/MPS baseline
  noted in `docs/phase1-onnx-rust-cli-plan.md` §1.

**Pass criteria**: both workflows exit 0; `scripts/install.sh` produces a working
`./vocalai/vocalai` binary + complete `./vocalai/models/`; both WAVs are audible and sound the
same; clean up the test tag afterward (`git push origin :v0.0.0-test`, `gh release delete
v0.0.0-test`).

**Fail indicators**:
- `models-export.yml` fails at the parity step: a real regression in `export/` — do not publish.
- `release.yml`'s smoke-test step fails: a corrupted/incomplete download or build — do not ship
  that asset.
- `scripts/install.sh`/`install.ps1` fails to download the release binary or any model file: check
  the HF repo/GitHub release actually have the expected files (`gh release view v0.0.0-test`,
  `curl https://huggingface.co/api/models/shmmsra/vocal-ai-models`).
- `out.wav`/`out-cpu.wav` silent, crashing, or audibly different: a real runtime bug, independent
  of anything CI can catch given the no-inference-in-CI constraint.

**Verified on Windows (2026-08-22)**: ran `scripts/install.ps1` fresh (no version bump needed —
tested against the already-live `v0.1.3` release + HF repo) into a scratch directory. Binary
reported `vocalai 0.1.3`; both `vocalai.exe --text "hello world" --out out.wav --models-dir models`
and the same with `--use-cpu` printed `Using CPU execution provider` (Windows ships CPU-only, no
`--use-gpu` artifact to compare against) and exited 0. `out.wav`: mono 24kHz 16-bit PCM, 1.12s,
peak 22835/32767, RMS ~4495. `out-cpu.wav`: same format, 0.84s, peak 17459/32767, RMS ~3961. Both
clearly non-silent; the duration difference is expected run-to-run sampling variance in T3's
autoregressive decode loop (see VAI-011's 2026-08-20 correction note), not a regression. Did not
re-run the anonymous-HF-Hub download a second time to isolate its mechanics — that logic is
identical shell-vs-PowerShell to what already passed on macOS. The download itself was slow
(~300KB/s on the ~2GB `t3_decoder.onnx`, several minutes total); alternatives discussed but not
decided (see `docs/CHANGELOG.md`'s 2026-08-22 VAI-007 entry). Linux install+synthesis and the
memory/swap-benchmark step remain unverified.

---

*Add new sections below this line as features land. Group by feature area (e.g. CLI, export pipeline, EP selection, voice cloning).*

---

## `--show-progress` console progress indicator (VAI-012)

**Test command(s)**:
```bash
# Baseline -- no flag, confirm zero output change:
./vocalai --text "hello world" --out out.wav --models-dir models
# With progress:
./vocalai --text "This is a somewhat longer sentence so the decode loop runs long enough to actually see the bar move." \
  --out out2.wav --models-dir models --show-progress
```

**Setup**: run from a real terminal (not through a pipe/log redirect) -- `indicatif`
auto-hides all drawing when stderr isn't a tty (`ProgressDrawTarget::is_hidden()`), by
design, so redirected/piped output won't show the bar even with the flag on; that's
expected, not a bug.

**What to observe**:
- Baseline command's stderr is unchanged from before this change (`Using <EP>`, nothing else
  until `Wrote out.wav`).
- With `--show-progress`: `==> Preparing voice conditioning...`, then a live progress bar
  labeled "Decoding speech tokens..." advancing token-by-token (bounded by
  `--max-new-tokens`, may finish short if EOS is hit first), then `==> Vocoding...`, then
  `==> Watermarking...`, then `Wrote out2.wav`.

**Pass criteria**: baseline output byte-identical to pre-VAI-012 behavior; with the flag,
all four phase labels appear in order and the decode bar visibly advances (verified in this
session on real hardware: `Using CPU execution provider`, phase lines, and a real WAV
produced for both a short 1s and a longer ~7.6s synthesis, `--show-progress` on and off).

**Fail indicators**: bar never advances or gets stuck at 0/max (progress callback not wired
to the decode loop); phase labels print out of order or duplicate; baseline (no-flag) output
differs from before this change (the no-op closure isn't truly free).

---

## Install-script version tracking + skip-if-up-to-date (VAI-012)

**Test command(s)**:
```bash
# Fresh install:
VOCALAI_INSTALL_DIR=./scratch-vocalai bash scripts/install.sh
# Re-run immediately, no version bump:
VOCALAI_INSTALL_DIR=./scratch-vocalai bash scripts/install.sh
```
(Windows: same two runs with `install.ps1`, `$env:VOCALAI_INSTALL_DIR` instead.)

**What to observe**: the first run downloads the binary and lists+downloads every model
file as before, then writes `./scratch-vocalai/.vocalai_version` and
`./scratch-vocalai/models/MODELS_VERSION`. The **second** run prints `==> vocalai binary is
up to date (vX.Y.Z), skipping download` and `==> models are up to date (X.Y.Z), skipping
model download`, and does not re-download the binary archive or list/fetch any model file.

**Pass criteria**: second run completes in a few seconds (two small metadata requests only:
a HEAD against the release-redirect URL, a tiny `MODELS_VERSION` text fetch), both "up to
date" lines appear, and `./scratch-vocalai/vocalai --text "hello world" --out out.wav
--models-dir ./scratch-vocalai/models` still produces a valid, non-silent WAV afterward
(confirming the skip didn't leave anything unusable).

**Fail indicators**: second run re-downloads the binary or any model file (skip logic
broken); second run falsely reports "up to date" against a deliberately-corrupted install
(delete one `.onnx` file from `models/` and rerun -- should NOT report up to date, since at
least the `find ... -name '*.onnx'` presence check would still pass if any other `.onnx`
file remains, so this specific corruption mode is a known gap, not a guaranteed catch -- see
ADR-0015's Consequences); `.vocalai_version`/`MODELS_VERSION` missing after a successful
first run; a rate-limited lookup falls back to attempting a download instead of exiting with
an error (see ADR-0015 -- this must never happen after the fail-fast fix).

**Verified this session (macOS, `install.sh`, live `v0.1.3` release + HF repo)**: the
`fail_if_rate_limited` helper was unit-tested in isolation (429 → exits 1; 403 +
`x-ratelimit-remaining: 0` → exits 1; 403 without that header → passes through; a normal 302 →
passes through) -- all four cases matched expectations. End-to-end: a first install run,
deliberately interrupted mid-download (`timeout`), correctly downloaded the binary + wrote
`.vocalai_version`, then correctly started listing + downloading model files without ever
writing `MODELS_VERSION` (confirming the ordering fix -- an incomplete model set leaves no
version stamp behind). A second run against that exact interrupted state correctly printed
`vocalai binary is up to date`, skipped the binary re-download, and correctly did **not**
claim the models were up to date (since `MODELS_VERSION` was genuinely absent) -- it re-listed
and resumed downloading model files as expected. **Not completed this session**: a full
~4GB/27-file model download run to completion (stopped partway through at the user's request,
twice, to avoid spending session time on HF Hub's slow anonymous-tier transfer) -- so the
"models are up to date" skip path was verified against a genuinely-interrupted state (correctly
NOT skipping) but not against a genuinely-complete one (correctly skipping). The write-after-
full-success logic itself is simple and was reviewed carefully, but hasn't been watched running
end to end. Also not observed live this session: an actual 429/403-rate-limited response from
the real GitHub/HF endpoints during a real install run (the rate-limit-detection *logic* was
verified via the unit test above, not via triggering a live rate limit against the real
service).

**Not verified this session**: `install.ps1`'s equivalent (`[System.Net.HttpWebRequest]`
redirect-header lookup + rate-limit detection, `Get-Content`/`Set-Content` version tracking) --
no Windows machine available. Logic was written symmetrically to the verified `install.sh` and
reviewed by hand,
but needs a real Windows run before being fully trusted (matching this repo's established
`install.ps1` verification gap pattern, see VAI-007's own notes above).
