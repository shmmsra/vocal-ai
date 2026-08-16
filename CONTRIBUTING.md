# Contributing to vocal-ai

> **AI agents read this first.** This file defines the non-negotiable process for making changes to vocal-ai. Treat every rule here as a hard constraint, not a suggestion.

---

## Rule 0 — Plan first, wait for approval, then implement

This rule exists before all others. Before writing a single line of implementation code for any non-trivial change (any new file, OR more than one file modified, OR a change touching architecture/API/IPC/ONNX-export boundaries):

1. Write out the plan: every file you will create or modify, and why.
2. State what you will **not** do.
3. **Stop. Post the plan. Wait.**
4. Do not proceed until the human responds with explicit approval ("lgtm", "go ahead", "approved", or equivalent).

**"implement X" is not pre-approval of your plan.** It is a task assignment. You must still surface the plan and wait for the human to sign off on your specific approach.

**Silence is not approval.** If the human does not respond, do not proceed.

After implementing:

1. **Write out the manual test plan** (see §3 for the required format) — exact command(s), what to observe, pass/fail criteria.
2. **Post the test plan. Wait.** The human runs the test. Do not proceed until they explicitly confirm ("tested, looks good", "manual test passed", or equivalent). **Silence is not confirmation.**
3. **Show the diff.** Summarise every file modified and why.
4. **Wait for commit approval.** Do not run `git commit` until the human says "lgtm", "commit it", or equivalent.

These four steps are sequential. Posting the diff before the human has confirmed the manual test is a violation of this rule — even if `make check` is green.

---

## Rule 1 — All tests and checks must pass before any commit lands. No exceptions.

This is not a style preference. A commit that breaks `make check` breaks the contract this repository runs on. The regression gates exist precisely because a human (or agent) will not always be watching.

---

## 1. Test-Driven Development (TDD)

### The workflow

1. **Write the failing test first** — or alongside the code if the feature is exploratory, but no later than the same commit.
2. **Run the full suite** (`make check`) — the new test must fail before implementation.
3. **Implement** — minimal code to make the test pass.
4. **Confirm green** — `make check` passes with the new test included.
5. **Commit** — only now.

### What must have tests

Every public function, exported module, CLI command, IPC handler, and external integration. Bug fixes must include a regression test that **reproduces the bug** before the fix.

If you can't write a test for a bug, describe why in the commit message. This is not optional — bugs without tests come back.

### Test framework

This project uses **`cargo test` (Rust, `crates/`) + `pytest` (Python, `export/`, dev-time only)**. Run with `cargo test --workspace && (cd export && pytest)`.

### Naming

- Rust tests live in `crates/*/tests/` (integration) or inline `#[cfg(test)]` modules (unit). Python tests live in `export/tests/`.
- Test names should describe the invariant in plain English: *"returns null when the input is missing a required field"*, not *"test null case"*.

---

## 2. Pre-commit gate (`make check`)

Before every commit, run:

```bash
make check
```

This runs `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace`, and `pytest` (in `export/`).

**Set up the automated git hook once** (after cloning):

```bash
make setup-hooks
```

This installs a `.git/hooks/pre-commit` that runs `make check` automatically on every `git commit`. If any check fails, the commit is aborted. Fix the failure, then commit again.

### Hook bypass policy

`--no-verify` is **forbidden** unless the commit is a docs-only or repo-housekeeping change with zero code modifications. If a hook is failing on legitimate code, fix the code — do not skip the hook.

---

## 3. Manual testing before commit (required for runtime-affecting changes)

Automated tests verify correctness at the unit and integration level. Manual testing verifies the feature works end-to-end in the actual runtime environment. Given this project's core risk is numerical/audio parity between the PyTorch reference and the ONNX/Rust runtime, any change touching inference, export, or CLI I/O must be manually verified — running the CLI and inspecting/listening to the output, or running `export/parity_check.py` — before it can be committed.

> **[`docs/manual-testing.md`](docs/manual-testing.md)** is the canonical runbook. It lists exact commands for every feature area. **When you add a new feature, CLI flag, or export step, update that file in the same commit.**

### What the agent must write out

Before showing the diff and requesting commit approval, the agent must produce an explicit manual test plan in this format:

```
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

The human runs the test (or confirms it was run). Only after an explicit confirmation does the commit proceed.

---

## 4. Architecture constraints

These are load-bearing rules. Violating them silently breaks the system in ways that only appear at runtime.

- **Execution-provider fallback order must stay explicit.** Hardware EPs (CoreML/CUDA) before CPU in every `ort::SessionBuilder`; log any silent CPU fallback.
- **No model weights or large binary artifacts in git.** `models/`, `*.onnx`, `*.safetensors`, `*.pt` are build/release artifacts only.
- **ONNX exports require a passing parity check before use.** No exported component ships into `vocalai-core` until `export/parity_check.py` confirms numerical parity with the PyTorch reference.

Full detail in `docs/agents/CONVENTIONS.md §3`.

---

## 5. Documentation as part of done

A feature or fix is **not done** until these files are updated in the same commit (or a follow-up commit in the same PR):

1. **`docs/agents/STATUS.md`** — update phase table, test counts, "What's next" if anything changed.
2. **`docs/CHANGELOG.md`** — add an entry: what changed, why, what was rejected, what's next.
3. **`docs/requirements.md`** — tick completed items, add new planned items.
4. **`docs/issues.md`** — mark the issue `DONE`, add a row to the **Recently closed** table with the commit hash. Use `pending` as the placeholder before committing; immediately after the commit lands, update the row with the real short hash (`git log --oneline -1`). Never leave `pending` in the table.
5. **`docs/manual-testing.md`** — add test steps for every new feature, CLI flag, or export step introduced.
6. **`docs/decisions/`** — if an architectural decision was made, create an ADR.

This is not optional cleanup. The next agent to pick up this repo reads these files first. Stale docs make every subsequent session less accurate.

---

## 6. Branch and merge strategy

**Keep history linear.** This repo uses a strict linear history — no merge commits.

### Code-review and merge policy

**Direct merge after local review.** Every change is committed straight onto `main` after the agent and human have reviewed it locally — no feature branches or PRs required for this solo-developer repo. `git push` remains a manual, human-only action (see below); the agent stops after the local commit.

### When merging a feature branch

If a branch is used for a larger piece of work, merge with **rebase or cherry-pick**, never `git merge --no-ff`. If `git merge --ff-only` fails, the branch needs to be rebased first.

### `git push` is always manual — no exceptions

**AI agents must never run `git push`.** Pushing is a one-way, externally visible action. It is the human's responsibility.

This rule holds even when the human says "merge it" or "land it" (means commit locally, not push). If the human explicitly types `git push` or says "push to remote", run it. Otherwise, stop after the local commit and report the final commit SHA.

---

## 7. Commit message format

```
<type>(<scope>): <short imperative summary>

<body — what changed and why, not a diff summary>
```

**Types**: `feat`, `fix`, `test`, `docs`, `refactor`, `ci`, `chore`, `perf`
**Scopes**: project-specific — e.g. `cli`, `core`, `export`, `docs`. See recent `git log` for established scopes.

If the change closes a ticket: include the ticket ID — e.g. `feat(vai-001): scaffold Cargo workspace`.

Examples:
- `feat(cli): add --voice flag for zero-shot cloning`
- `fix(export): handle empty reference wav in parity check`
- `test(core): add coverage for EP fallback ordering`

---

## 8. Approval-gated operations

The following operations **always** require explicit human approval, beyond the standard plan + commit approval:

- **Anything that costs money to run** — e.g. spinning up cloud GPU instances for export/parity work, paid API calls.
- **Bundling or redistributing model weights** — packaging Chatterbox weights (or any derived ONNX export) into a release artifact.

If you are uncertain whether an operation requires approval, ask. The cost of asking is low.

---

## 9. Running individual test suites

```bash
# Full pre-commit gate
make check

# Tests only
cargo test --workspace && (cd export && pytest)

# Type check only (Rust: cargo check; no separate Python type checker configured)
cargo check --workspace
```

See `docs/agents/OVERVIEW.md` for per-subpackage commands if the project has multiple build targets.

---

## 10. Commit tracking policy (convention-only)

This repo uses **convention-only** commit tracking: no `Co-Authored-By` trailer, no post-commit hook, no `docs/commit-log.md`. Agents commit with plain `git commit`, exactly like a human would. There is no automated distinction between agent and manual commits, and no gate on push/merge based on author kind.

The convention that still applies: when an agent commit changes runtime behavior, its commit body should briefly note that it was agent-authored and reviewed, so a human skimming `git log` can tell at a glance. This is a courtesy, not an enforced rule.

If this project later grows a team or wants a stricter audit trail, revisit this choice (trailer + log, or pre-commit-block modes are both available via the `ai-sdlc-bootstrap` skill).

---

## 11. Commit co-authorship

This project's policy: **no** — commit messages do not carry a `Co-Authored-By` trailer for agent-authored work. Author kind is not automatically tracked (see §10).

---

## 12. Repository hygiene files — keep them updated

The following files are part of the project's contract with both humans and agents. **Treat them as code**: when a change makes them stale, update them in the same commit.

| File | When to update |
|------|----------------|
| `README.md` | Project description / install / quick-start changes. Always link to `docs/dev-setup.md` for full setup. |
| `LICENSE` | Only if changing license. Document the change in an ADR. |
| `CODEOWNERS` | When ownership of a directory or module changes. |
| `.gitignore` | When adding a new build artefact / cache dir / IDE config that should not be committed. Append, never wholesale-rewrite. |
| `.vscode/`, IDE configs | When adding a recommended extension, snippet, or workspace setting that benefits all contributors. |
| `docs/dev-setup.md` | When adding a new dependency, MCP server, skill, language toolchain, or required tool. The reproducibility of onboarding depends on this. |
| `docs/decisions/` | When making an architectural decision another agent might wonder about. See ADR template. |

Stale hygiene files are a documented failure mode of multi-agent SDLCs. Don't let them rot.

---

## 13. Dev environment setup

New contributors (human or agent) bootstrap their environment by following **[`docs/dev-setup.md`](docs/dev-setup.md)**. That document is the single source of truth for:

- Required language toolchains and versions (Rust, Python)
- Project dependencies (and how to install them)
- Required external tools / CLIs
- MCP servers / agent skills the project depends on
- The `make setup-hooks` step (installs the pre-commit hook)
- The first-time `make check` baseline run

If you add a new dependency or tool while working in this repo, update `docs/dev-setup.md` in the same commit. Onboarding the next agent on a fresh clone is the regression test for this file.
