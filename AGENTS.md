# vocal-ai — Agent Instructions

> Entry point for AI coding agents: Codex (OpenAI), Antigravity (Google), and any AGENTS.md-compatible tool.
> Claude Code users: read `CLAUDE.md` instead — it adds Claude-specific workflow rules on top of these shared conventions.
>
> **Sync note**: The "Key rules" section below summarises `docs/agents/CONVENTIONS.md`. If you update that file, update the summary here too. Agent config files in this repo: `CLAUDE.md`, `AGENTS.md`, `GEMINI.md`.

---

## Before starting any work, read these three files

1. **[`docs/agents/OVERVIEW.md`](docs/agents/OVERVIEW.md)** — project context, architecture, tech stack, build commands.
2. **[`docs/agents/CONVENTIONS.md`](docs/agents/CONVENTIONS.md)** — all hard constraints and contribution rules. Non-negotiable.
3. **[`docs/agents/STATUS.md`](docs/agents/STATUS.md)** — current status, what's next, backlog priority order.

---

## Key rules (full detail in `docs/agents/CONVENTIONS.md`)

- **Plan before you code**: For any non-trivial change, write out every file you will create/modify and why, state what you will NOT do, and wait for explicit human approval before writing any implementation code. Silence is not approval.
- **`make check` must pass** before every commit.
- **TDD is mandatory**: write tests before or alongside logic changes, in the same commit.
- **Docs are part of done**: update `docs/agents/STATUS.md`, `docs/CHANGELOG.md`, `docs/issues.md`, `docs/requirements.md`, and `docs/dev-setup.md` (if dependencies/tools changed) in the same commit.
- **Commit with plain `git commit`**: this repo uses convention-only commit tracking — no trailer, no wrapper script, no commit log. Note in the commit body when a runtime-affecting change was agent-authored and manually verified.
- **Merge policy**: direct merge after local review. See `CONTRIBUTING.md §6`.
- **`git push` is always manual** — agents never run it.

---

## Contribution process

Full rules are in [`CONTRIBUTING.md`](CONTRIBUTING.md). All agents must follow it.

Key checklist before committing:
- [ ] Plan written and approved before implementation
- [ ] Tests written (TDD)
- [ ] `make check` passes
- [ ] Manual test completed (if runtime behaviour changed — see `CONTRIBUTING.md §3`)
- [ ] Docs updated (`docs/agents/STATUS.md`, `CHANGELOG.md`, `issues.md`, `requirements.md`, `dev-setup.md` if deps changed)
- [ ] Repo hygiene files updated where applicable (README, `.gitignore`, CODEOWNERS — see `CONTRIBUTING.md §12`)
- [ ] Committed via plain `git commit` (no `git push`)

---

## Architecture boundaries (never cross these)

- **Execution-provider fallback order must stay explicit.** Hardware EPs (CoreML/CUDA) before CPU in every `ort::SessionBuilder`; log any silent CPU fallback.
- **No model weights or large binary artifacts in git.** `models/`, `*.onnx`, `*.safetensors`, `*.pt` are build/release artifacts only.
- **ONNX exports require a passing parity check before use.** No exported component ships into `vocalai-core` until `export/parity_check.py` confirms numerical parity with the PyTorch reference.
