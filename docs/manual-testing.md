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

## Execution-provider selection (Milestone 1)

**Test command(s)**:
```bash
cargo test -p vocalai-core session:: 
cargo test -p vocalai-core --features coreml session::
```

**Setup**: Rust toolchain installed. Second command only meaningful on macOS (CoreML feature).

**What to observe**: Test names and pass/fail output from `crates/vocalai-core/src/session.rs`'s `session::tests` module.

**Pass criteria**:
- Default build (no features): `execution_providers()` returns exactly one entry, and it downcasts to `CPU`.
- `--features coreml` build: `execution_providers()` returns `[CoreML, CPU]` in that order — CoreML first, CPU last.
- All tests pass, no clippy warnings under either feature combination (`cargo clippy --workspace --all-targets -- -D warnings`, then again with `--features vocalai-core/coreml`).

**Fail indicators**:
- CPU appears anywhere but last in the list.
- A hardware EP is missing when its feature is enabled, or present when it isn't.

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
  run commands — lives in `docs/dev-setup.md` §11.** Follow §11.1 (generate model files) once, then
  §11.2 (build). Don't re-derive the export order from the per-milestone sections above.
- For the record, `ModelBundle::load` (`crates/vocalai-core/src/pipeline.rs`) requires exactly:
  `tokenizer.json`, `t3_cond_prefill.onnx`, `t3_decoder.onnx`, `t3_speech_emb.npy`,
  `t3_speech_pos_emb.npy`, `s3gen_estimator.onnx`, `s3gen_flow_encoder_{200,400,600,800,1000,1200}.onnx`,
  `hifigan.onnx`, `perthnet_encoder.onnx`, `s3gen_spk_embed_affine_{weight,bias}.npy`, and
  `default_voice/*.npy`. It does **not** load `ve.onnx`, `s3tokenizer.onnx`, or `campplus.onnx` —
  those are for the future `--voice` cloning path (part B.2).

**Test command(s)** (see `docs/dev-setup.md` §11 for the full per-platform export list this assumes
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
- Both commands print `Wrote <path>` and exit 0 — no `speech_feat` shape-mismatch error.
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

### Cross-platform build gate: `make check` on Windows (ADR-0010)

**Test command(s)** (from repo root):

```bash
make check
```

**Setup**:
- Rust toolchain (`cargo`), GNU `make`, and the `export/.venv` all installed per `docs/dev-setup.md`.
- On Windows, run from a shell whose PATH already includes `cargo` and `make` (open a *new* shell
  after installing them — see §10). No `sh`/Git Bash is required; the recipes run under `cmd.exe`.

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

*Add new sections below this line as features land. Group by feature area (e.g. CLI, export pipeline, EP selection, voice cloning).*
