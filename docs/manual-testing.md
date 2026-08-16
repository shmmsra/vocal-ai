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

*Add new sections below this line as features land. Group by feature area (e.g. CLI, export pipeline, EP selection, voice cloning).*
