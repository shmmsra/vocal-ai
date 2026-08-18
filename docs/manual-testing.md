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
- Any `[FAIL]` line, or `max_abs_diff` far above `1e-4`/`1e-3` (atol/rtol) — check whether
  `export_s3gen.py`'s example shapes still match `export_hifigan.py`'s fixed `EXAMPLE_FRAMES=50`
  (the HiFiGAN export has no dynamic frame axis; see its module docstring), or whether
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

## CLI: default-voice end-to-end synthesis (VAI-006, part B.1)

**Prerequisites**:
- All Milestone 2/3/4/5/8 exports present in `models/` (`hifigan.onnx`, `ve.onnx` -- not used by
  this path but harmless if absent is not true, see below --, `s3gen_estimator.onnx`,
  `s3gen_flow_encoder_{200,400,600,800,1000,1200}.onnx`, `campplus.onnx` -- not used by this path
  either --, `s3gen_spk_embed_affine_{weight,bias}.npy`, `t3_cond_prefill.onnx`, `t3_decoder.onnx`,
  `t3_speech_emb.npy`, `t3_speech_pos_emb.npy`, `perthnet_encoder.onnx`, `models/default_voice/*.npy`
  -- see VAI-008's manual-testing section above for how to produce all of these).
- **`models/tokenizer.json`** -- not a model export (no ONNX graph, no parity check: it's already
  the exact file format the Rust `tokenizers` crate reads directly). Staged by
  `export/fetch_tokenizer.py`, which downloads it from `ResembleAI/chatterbox` on the HuggingFace
  Hub and copies it to `models/tokenizer.json`:
  ```bash
  cd export && python fetch_tokenizer.py
  ```

**Test command(s)**:
```bash
cd export && python fetch_tokenizer.py && cd ..   # stages models/tokenizer.json, see prerequisites
cargo build --release -p vocalai-cli
./target/release/vocalai --text "hello world" --out /tmp/out.wav --models-dir models
```

**Known-blocked result (VAI-009)**: this exact command currently **fails**:
```
error: ONNX Runtime session error: Got invalid dimensions for input: speech_feat for the following indices
 index: 2 Got: 44 Expected: 50
 Please fix either the inputs/outputs or the model.
```
This is `hifigan.onnx`'s fixed-50-mel-frame export limitation (see `docs/issues.md` VAI-009), not a
bug in this session's new code. Until VAI-009 lands, use the diagnostic command below to verify the
rest of the pipeline instead:
```bash
./target/release/vocalai \
  --text "This is a somewhat longer sentence designed to make sure text to speech generation does not stop too early during this quick diagnostic test." \
  --out /tmp/out.wav --models-dir models --max-new-tokens 25
```
(`--max-new-tokens 25` forces exactly 25 generated speech tokens when the text is long enough not
to hit EOS first, so `mel_len2 = 2 * 25 = 50` lands exactly on HiFiGAN's fixed shape.)

**What to observe**:
- The command prints `Wrote /tmp/out.wav` and exits 0.
- `/tmp/out.wav` is a mono 24000 Hz 16-bit PCM WAV, ~1.0s long (25 generated tokens * 40ms/token).
- Inspect programmatically if you can't listen: `python3 -c "import wave; w=wave.open('/tmp/out.wav'); print(w.getnchannels(), w.getframerate(), w.getnframes())"` should print `1 24000 24000`.
- The waveform should not be silence/NaN: peak amplitude and RMS should both be clearly nonzero
  (verified during implementation: peak ~26654/32767, RMS ~4023 on a real run).

**Pass criteria**:
- The diagnostic (`--max-new-tokens 25`, long text) command exits 0 and produces a non-silent WAV
  as above.
- `cargo build --release -p vocalai-cli` succeeds with no warnings.
- Once VAI-009 lands: the plain `--text "hello world" --out /tmp/out.wav` command (no
  `--max-new-tokens` override) should also succeed and produce audible, recognizable speech --
  re-run this section's first command and update this note when that's true.

**Fail indicators**:
- `error: failed to load models from ...`: a required `.onnx`/`.npy`/`tokenizer.json` file is
  missing from `--models-dir` -- re-check the prerequisites list above.
- `error: T3 generated no speech tokens for this text`: T3 emitted EOS immediately -- try a longer
  or different input text.
- `Got invalid dimensions for input: speech_feat ... Expected: 50` at any frame count other than
  the diagnostic's forced 50: expected until VAI-009 lands (see "Known-blocked result" above); if
  it still happens *after* VAI-009 lands, that's a regression.
- Silence (all-zero samples) or clipped-flat output: a real bug in the T3/S3Gen/HiFiGAN/watermark
  wiring, not the known VAI-009 limitation -- investigate `pipeline::synthesize`'s tensor assembly.

---

*Add new sections below this line as features land. Group by feature area (e.g. CLI, export pipeline, EP selection, voice cloning).*
