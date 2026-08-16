# vocal-ai — Dev Environment Setup

> Canonical onboarding guide for vocal-ai. Follow this top-to-bottom on a fresh clone.
>
> **If you add a new dependency, tool, MCP server, or agent skill while working in this repo, update this file in the same commit.** Onboarding the next agent on a fresh clone is the regression test.

---

## 1. Prerequisites — language toolchains

| Language | Required version | How to install |
|----------|------------------|-----------------|
| Rust | stable (edition 2021) | `rustup default stable` — https://rustup.rs |
| Python | 3.10+ (dev-time only, for `export/`) | `pyenv install 3.12` or your system Python |

*Keep version pins here in sync with `rust-toolchain.toml` / `export/requirements.txt` once those exist.*

---

## 2. Clone and install dependencies

```bash
git clone <repo-url>
cd vocal-ai

# Rust workspace:
cargo fetch
```

### Python export/ tooling (dev-time only, not shipped)

`chatterbox-tts` requires **Python 3.10+**. Use an isolated venv scoped to `export/` so
this doesn't collide with any other Python on your machine — installing it pulls in
torch/torchaudio/transformers (~2 GB) and, on first run of the `export_*.py` scripts,
downloads the Chatterbox checkpoint from the HuggingFace Hub (~1-2 GB more).

**macOS / Linux**:
```bash
cd export
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell or cmd)**:
```bat
cd export
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Re-activate the venv (`source export/.venv/bin/activate` / `export\.venv\Scripts\activate`)
in any new shell before running `export_*.py`, `parity_check.py`, or `pytest` in `export/`
directly. `make check` finds this venv automatically (see §6) — no activation needed just
to commit.

---

## 3. Required external tools / CLIs

| Tool | Purpose | Install |
|------|---------|---------|
| `git` | Version control | Pre-installed on macOS/Linux |
| `make` | Build entry point | `brew install make` / `apt install make` |
| `cargo` / `rustup` | Rust toolchain + package manager | https://rustup.rs |
| `python3` / `pip` (3.10+) | Runs `export/` ONNX export + parity-check scripts, in `export/.venv` (§2) | pyenv or system Python |

*Add rows for platform-specific accelerator tooling (e.g. CUDA toolkit for local GPU testing) as Milestone 1+ work lands.*

---

## 4. Agent skills and MCP servers

This project integrates with the following agent skills / MCP servers. Install / authorise them in your local Claude Code / Codex / Gemini setup before working in this repo.

| Skill / MCP | Purpose | How to install |
|-------------|---------|-----------------|
| `ai-sdlc-bootstrap` | The skill that scaffolded this workflow (already applied) | n/a |

*Add rows when you integrate new skills or MCP servers. Include the install command + any auth steps. If a skill is required for an end-to-end test path, note that here.*

---

## 5. Install the git hooks (one-time)

```bash
make setup-hooks
```

This installs:

- **pre-commit** — runs `make check` before every commit. Aborts if anything fails.

(No post-commit hook is installed — this repo uses convention-only commit tracking, see `CONTRIBUTING.md §10`.)

You only need to run `make setup-hooks` once per clone. Re-run it if you delete `.git/hooks/`.

---

## 6. Run the baseline check

```bash
make check
```

Expected outcome on a fresh clone: `make check` passes cleanly (real tests only — no
placeholder/ignored tests remain as of Milestone 2). `make check`'s `test-py` step runs
`export/.venv/bin/python -m pytest` automatically if that venv exists (§2), otherwise falls
back to plain `pytest` — so committing doesn't require activating the venv yourself, but the
venv **does** need to exist and have `requirements.txt` installed (§2) for the export/parity
tests to pass rather than error on missing imports. If `make check` fails for any other reason
on a clean clone, fix `docs/dev-setup.md` first — the install instructions above are wrong.

---

## 7. Editor / IDE setup

The repo carries workspace settings for VS Code. Open the repo in VS Code and accept the recommended extensions when prompted.

- **VS Code**: `.vscode/extensions.json` lists recommended extensions (rust-analyzer, Python); settings in `.vscode/settings.json`.

*Add entries if you add IDE configs for other editors later.*

---

## 8. Environment variables / secrets

This project has no secrets today. If one is introduced later:

1. Copy `.env.example` to `.env`.
2. Fill in the required values (ask the project owner for any internal ones).
3. **Never commit `.env`.** It's in `.gitignore`.

---

## 9. Verify the agent workflow

To prove your environment can drive the agent-SDLC contract end-to-end:

1. Read `CLAUDE.md` (or `AGENTS.md` for non-Claude agents) and `docs/agents/CONVENTIONS.md`.
2. Make a trivial change (e.g. add a comment).
3. Try to commit via plain `git commit -m "test: verify hook installation"`.
4. Verify the pre-commit hook ran `make check` before the commit landed.
5. Revert the commit (`git reset --hard HEAD~1`).

If any of those steps fails, the local hooks are not installed correctly — re-run `make setup-hooks`.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `make check` fails on a fresh clone (beyond the known placeholder tests) | This file is stale | Update the install steps above and re-run |
| Pre-commit hook doesn't run | `make setup-hooks` was never run, or `.git/hooks/pre-commit` isn't executable | `make setup-hooks` |
| `cargo`/`pytest` not found | Toolchain not installed or not on `PATH` | Re-check §1/§2 |

*Add new rows as the team discovers recurring setup gotchas.*
