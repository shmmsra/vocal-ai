# vocal-ai

> A fully self-contained, cross-platform command-line TTS tool built around the Chatterbox TTS
> model — ONNX + native Rust inference, no bundled Python/PyTorch, no multi-GB runtime download.

---

## Status

See [`docs/agents/STATUS.md`](docs/agents/STATUS.md) for current phase, in-progress work, and what's next.

## Getting started

Full setup — Rust toolchain, `ort`/onnxruntime, Python `export/` environment, VS Code extensions,
git hooks — is documented in **[`docs/dev-setup.md`](docs/dev-setup.md)**. Read that first on a
fresh clone.

Quick start (assumes prerequisites already installed — see dev-setup.md for those):

```bash
git clone <repo-url>
cd vocal-ai
make setup-hooks    # one-time: installs the pre-commit gate
make check          # verify baseline
```

## Project layout

```
docs/
├── agents/                        # AI-agent rules — read FIRST on every session
│   ├── OVERVIEW.md                # Context, architecture, tech stack
│   ├── CONVENTIONS.md             # Hard constraints — non-negotiable
│   └── STATUS.md                  # Current state + what's next
├── dev-setup.md                   # Onboarding
├── decisions/                     # ADRs — read before changing anything covered
├── CHANGELOG.md
├── requirements.md
├── manual-testing.md
├── issues.md                      # Ticket tracker (in-repo, prefix VAI)
└── phase1-onnx-rust-cli-plan.md   # Approved Phase 1 implementation plan
```

## AI-Driven SDLC

This project is built by a human collaborating with multiple AI coding agents (Claude, Codex,
Gemini) across many sessions. The workflow is documented in
**[`CONTRIBUTING.md`](CONTRIBUTING.md)** and is non-optional. Highlights:

- **Plan first** — agents write a plan and wait for explicit human approval before coding.
- **TDD enforced** — `make check` runs before every commit via the pre-commit hook.
- **Docs as part of done** — every feature updates `STATUS.md`, `CHANGELOG.md`, `requirements.md`,
  `issues.md`, and (if dependencies changed) `dev-setup.md`.
- **Commit tracking**: convention-only — no trailer or hook enforcement (see
  `CONTRIBUTING.md §10`).
- **Merge policy**: direct merge to `main` after local review (see
  [`CONTRIBUTING.md §6`](CONTRIBUTING.md)).

If you're a new contributor (human or agent), read in this order: this file →
`docs/dev-setup.md` → `docs/agents/OVERVIEW.md` → `CONTRIBUTING.md`.

## License

MIT — see [`LICENSE`](LICENSE).

## Owner

Shivam Mishra
