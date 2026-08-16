# vocal-ai — Changelog

> Chronological log of what changed in this repo and *why*. The "why" matters more than the "what" — the diff already shows the what.
>
> Update at the end of every session. Newest entries at the top.

---

## 2026-08-16 — ai-sdlc-bootstrap scaffold

**What changed**: Bootstrapped the AI-driven SDLC workflow on this repo via the `ai-sdlc-bootstrap` skill. Added agent-config layer (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`), `docs/agents/` triad, `CONTRIBUTING.md`, `docs/issues.md`, ADR template, pre-commit gate (`make check`), CI workflow, repo hygiene files (README, LICENSE, CODEOWNERS, `.editorconfig`, VS Code settings), and a throwaway placeholder Cargo workspace + Python `export/` stub carrying one intentionally-failing test each as a TDD seed.

**Why**: This project will be developed by humans + multiple AI agents across many sessions. Without the agent-config layer and a strict plan/test/commit workflow, every session would start from zero. The scaffold installs the contract.

**What was rejected**: Commit co-authorship trailer and the trailer-log/pre-commit-block commit-tracking modes — this is a solo-developer repo, convention-only tracking (no hook, no `docs/commit-log.md`) was chosen instead. PR-required merge policy was also rejected in favor of direct-merge-to-main.

**What's next**: Begin Phase 1 Milestone 1 (Cargo workspace scaffold, real `ort` wiring) as tracked in `docs/issues.md` (`VAI-001`).

---

*Add new entries above this line. Format: `## YYYY-MM-DD — Short title`, followed by `What / Why / Rejected / Next` sub-headings.*
