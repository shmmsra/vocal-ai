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

---

*Add new sections below this line as features land. Group by feature area (e.g. CLI, export pipeline, EP selection, voice cloning).*
