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

**Windows**: neither `rustup` nor `pyenv` ships with Windows, and no build tool (`make`) is
preinstalled either. See §3 for the exact install commands and §2 for `pyenv-win` setup
(it's a separate reimplementation, not a drop-in for `pyenv`). See §10 for known Windows-specific
gotchas — a `PATH` that doesn't refresh in an already-open shell, and a stale `pyenv-win`
version cache.

---

## 2. Clone and install dependencies

```bash
git clone <repo-url>
cd vocal-ai

# Rust workspace:
cargo fetch
```

**Windows**: if `cargo` isn't recognized, install the toolchain first (§3), then open a
*new* shell before running `cargo fetch` — a shell opened before install won't pick up the
updated `PATH`.

### Python toolchain on Windows (pyenv-win)

`pyenv` (used above) is macOS/Linux-only. On Windows, install
[pyenv-win](https://github.com/pyenv-win/pyenv-win) instead — it is a separate
reimplementation with its own CLI quirks, not a drop-in:

```powershell
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
# open a new shell so PATH picks up pyenv-win, then:
pyenv install 3.12.4
pyenv local 3.12.4   # run from the repo root — honors .python-version
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
| `git` | Version control | Pre-installed on macOS/Linux; Windows: https://git-scm.com or `winget install Git.Git` |
| `make` | Build entry point | `brew install make` / `apt install make` / Windows: `winget install ezwinports.make` (GnuWin32's `make` package is a stale 3.81 build — prefer `ezwinports.make`, a maintained GNU Make 4.x port) |
| `cargo` / `rustup` | Rust toolchain + package manager | https://rustup.rs / Windows: `winget install --id Rustlang.Rustup -e`, then open a new shell |
| `python3` / `pip` (3.10+) | Runs `export/` ONNX export + parity-check scripts, in `export/.venv` (§2) | pyenv or system Python / Windows: [pyenv-win](https://github.com/pyenv-win/pyenv-win) — see §2 |

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

> To actually run the app (generate speech to a WAV) rather than just pass the test gate, see
> **§11 — Generate model artifacts + run the app** below. `make check` does *not* produce the
> `models/` files the CLI needs.

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
| `cargo`/`pytest` not found | Toolchain not installed or not on `PATH` | Re-check §1/§2/§3 (Windows: §3 has `winget` commands) |
| `cargo`/`make` still "not recognized" immediately after installing it (Windows) | The installer updated `PATH`, but the current shell's environment was cached before install | Open a new PowerShell/terminal window |
| `pyenv install <version>` on Windows fails with `definition not found` for a version that exists on python.org | pyenv-win's local version cache is stale and `pyenv update` is broken on modern Windows | See the workaround in §2 (Python toolchain on Windows) |
| `cargo test` / `make check` fails on Windows with `LNK2038: RuntimeLibrary mismatch (MD_DynamicRelease vs MT_StaticRelease)` | A transitive C++ dep compiled against a different CRT than ONNX Runtime's prebuilt | Already fixed in-repo (ADR-0010): `tokenizers` is pinned with `default-features = false` to drop the static-CRT `esaxx_fast` C++. If it recurs after a dep bump, re-check the new dep's default features for a `/MT` C++ build |
| `make check` on Windows errors with `-x was unexpected at this time` or `'.venv' is not recognized` | An old Makefile recipe used POSIX-shell syntax / the POSIX venv path; `make` runs recipes through `cmd.exe` on Windows | Already fixed in-repo (ADR-0010): the `test-py*` recipes now OS-detect `.venv\Scripts\python.exe` and avoid shell conditionals. Pull latest `Makefile` |
| `error: failed to load models from models: ...` when running `vocalai` | The `models/` directory hasn't been populated (its contents are git-ignored build artifacts, never committed) | Run the export scripts in §11 to generate them |

*Add new rows as the team discovers recurring setup gotchas.*

---

## 11. Generate model artifacts + run the app (end-to-end)

`make check` verifies the code but does **not** produce the model files the CLI loads. The
`models/` directory holds ONNX graphs + `.npy` tensors that are dev-time build artifacts —
git-ignored, never committed (see `CLAUDE.md` §1 hard constraints). You generate them once by
running the `export/` scripts, which download the Chatterbox checkpoint from HuggingFace on first
run (~1–2 GB, cached afterward) and require the `export/.venv` from §2.

All scripts write to `<repo>/models/` regardless of the directory you run them from (the path is
anchored to the repo root, not the current working directory). The commands below call the venv
interpreter directly, so you don't need to activate the venv first.

### 11.1 Generate the model files (one-time, for the default-voice path)

These eight steps produce every file `vocalai` loads for default-voice synthesis:

| Script | Produces |
|--------|----------|
| `fetch_tokenizer.py` | `tokenizer.json` (downloaded, not exported — no ONNX graph) |
| `export_t3.py` | `t3_cond_prefill.onnx`, `t3_decoder.onnx`, `t3_speech_emb.npy`, `t3_speech_pos_emb.npy` |
| `export_s3gen.py` | `s3gen_estimator.onnx` |
| `export_s3gen_flow_encoder.py` | `s3gen_flow_encoder_{200,400,600,800,1000,1200}.onnx` (6 buckets) |
| `export_hifigan.py` | `hifigan.onnx` |
| `export_perthnet.py` | `perthnet_encoder.onnx` |
| `export_campplus.py` | `s3gen_spk_embed_affine_weight.npy`, `s3gen_spk_embed_affine_bias.npy` (+ `campplus.onnx`) |
| `export_default_voice.py` | `default_voice/*.npy` (6 conditioning tensors) |

**macOS / Linux**:
```bash
PY=export/.venv/bin/python
$PY export/fetch_tokenizer.py
$PY export/export_t3.py
$PY export/export_s3gen.py
$PY export/export_s3gen_flow_encoder.py
$PY export/export_hifigan.py
$PY export/export_perthnet.py
$PY export/export_campplus.py
$PY export/export_default_voice.py
```

**Windows (PowerShell)**:
```powershell
$py = "export\.venv\Scripts\python.exe"
& $py export\fetch_tokenizer.py
& $py export\export_t3.py
& $py export\export_s3gen.py
& $py export\export_s3gen_flow_encoder.py
& $py export\export_hifigan.py
& $py export\export_perthnet.py
& $py export\export_campplus.py
& $py export\export_default_voice.py
```

> `export_ve.py` (`ve.onnx`) and `export_s3tokenizer.py` (`s3tokenizer.onnx`) are **not** needed
> for default-voice synthesis — they're only used by the future `--voice` zero-shot cloning path
> (Milestone 6 part B.2, not yet implemented). Skip them unless you're working on that path.

### 11.2 Build the CLI

```bash
cargo build --release -p vocalai-cli
```

### 11.3 Synthesize speech

**macOS / Linux**:
```bash
./target/release/vocalai --text "hello world" --out out.wav --models-dir models
```

**Windows (PowerShell)**:
```powershell
.\target\release\vocalai.exe --text "hello world" --out out.wav --models-dir models
```

Each run prints `Wrote out.wav` and exits 0. The output is a mono 24 kHz 16-bit PCM WAV
(~0.9 s for "hello world") of audible, non-silent speech. Quick non-audio sanity check
(reuse the venv interpreter from §11.1):

```bash
export/.venv/bin/python -c "import wave; w=wave.open('out.wav'); print(w.getnchannels(), w.getframerate(), w.getnframes())"
```

**Tuning flags** (all optional, defaults in parentheses): `--exaggeration` (0.5),
`--cfg-weight` (0.5), `--temperature` (0.8), `--repetition-penalty` (1.2), `--min-p` (0.05),
`--top-p` (1.0), `--max-new-tokens` (1000). `--voice <ref.wav>` is reserved for zero-shot cloning
and currently errors out clearly (part B.2, not yet implemented).

See `docs/manual-testing.md` → "CLI: default-voice end-to-end synthesis" for the full pass/fail
criteria and known failure modes.
