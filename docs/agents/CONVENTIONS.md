# vocal-ai — Engineering Conventions

> Hard constraints and contribution rules for all agents working in this repository.
> These apply to every session, every agent, every change — no exceptions.

---

## 1. Contribution workflow

### What counts as non-trivial (requires a written plan + explicit human approval before coding)

- Any new file added to the repo
- More than one file modified in a single change
- Any change touching architecture, the public API/CLI surface, IPC, or ONNX export/inference
  boundaries

### Exempt from the planning step (implement directly)

- Single-file bug fixes of 1–3 lines
- Pure documentation updates
- Adding tests for an interface that is already fully designed and approved
- Dependency version bumps

### Exempt from manual testing (may commit after `make check` + diff approval)

- Pure documentation updates with zero code changes
- Test-only additions with no logic change
- Dependency version bumps with no behavioural change
- Pure internal refactors where the public API and observable output are provably unchanged

---

## 2. Implementation rules

1. **TDD is mandatory** — write the test before (or alongside) any logic change, in the same commit.
2. **`make check` must pass** before every commit.
3. **No `--no-verify`** except for docs/housekeeping commits with zero code changes.
4. **Documentation is part of done** — see `CONTRIBUTING.md §5` for the full list of docs to update per session.

---

## 3. Hard constraints

**Universal rules**:

- **No credentials in code**: API keys go in `.env` only (git-ignored). Secrets in code are a hard failure.
- **All decisions get an ADR**: If you're about to change something another agent might wonder about, write an ADR. Template in `docs/decisions/README.md`.
- **Linear history**: No `git merge --no-ff`. Use rebase or `git merge --ff-only`. Agents never run `git push`.

**Project-specific domain rules** (from `docs/phase1-onnx-rust-cli-plan.md`):

- **Execution-provider fallback order must stay explicit.** When registering `ort::SessionBuilder` execution providers, hardware EPs (CoreML on macOS, CUDA on Windows/Linux) must be listed *before* the CPU fallback, and any silent fallback to CPU must be logged — `ort` falls back silently down the EP list otherwise (plan §5).
- **No model weights or large binary artifacts in git.** `*.onnx`, `*.safetensors`, `*.pt`, `conds.pt`, and any bundled release artifact belong in `models/` (git-ignored) or the release pipeline — never checked into the repository (plan §2.3, §6).
- **ONNX exports require a passing parity check before use.** No exported component (T3, S3Gen, HiFiGAN, voice encoder, S3 tokenizer, PerthNet) may be wired into `vocalai-core` until `export/parity_check.py` confirms its output matches the PyTorch reference within tolerance (plan §8).

---

## 4. Approval-gated operations

The following operations **always** require explicit human approval before execution, beyond the standard plan + commit approval:

- **Anything that costs money to run** — e.g. spinning up cloud GPU instances for export/parity work, paid API calls. Ask before incurring any spend.
- **Bundling or redistributing model weights** — packaging Chatterbox weights (or any derived ONNX export) into a release artifact. Model licensing is an open item (plan §9) — confirm redistribution rights before shipping a bundle.

If you are uncertain whether an operation falls under one of these categories, ask. The cost of asking is low; the cost of an unwanted change in any of these areas is high.

---

## 5. Code-review and merge policy

This project's policy: **direct merge after local review**.

Every change is committed directly onto `main` after the agent and human have reviewed it locally — no feature branches or PRs required for this solo-developer repo. Agents still never run `git push`; the human decides when (and whether) to push to the remote.

Full detail in `CONTRIBUTING.md §6`.

---

## 6. Commit attribution and tracking

This repo uses **convention-only** commit tracking: no `Co-Authored-By` trailer, no post-commit hook, no `docs/commit-log.md`. Agents commit with plain `git commit`, the same as a human would.

The convention that still applies: when an agent-authored commit changes runtime behavior, its commit body should briefly note that it was agent-authored and reviewed, so a human skimming `git log` can tell at a glance. This is informational only, not tooling-enforced — see `CONTRIBUTING.md §10`.

---

## 7. Repository hygiene files

These files are project artefacts, not metadata. **Keep them current.** Updating them is part of "done" for any change that affects them.

| File | What changes it |
|------|-----------------|
| `README.md` | New install/quick-start step, public-facing description change |
| `LICENSE` | Change in license (requires ADR) |
| `CODEOWNERS` | Module / directory ownership change |
| `.gitignore` | New build artefact, cache dir, IDE config, or secret pattern |
| `.vscode/`, IDE configs | New recommended extension or workspace setting |
| `docs/dev-setup.md` | **New dependency, MCP, skill, language toolchain, or required tool** — onboarding will break otherwise |
| `docs/decisions/` | Architectural choice another agent might wonder about |

See `CONTRIBUTING.md §12` for the full policy.
