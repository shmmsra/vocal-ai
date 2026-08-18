# ADR-0010: Windows build + `make check` portability (disable `esaxx_fast`, OS-detect the pytest interpreter)

**Date**: 2026-08-18
**Status**: Accepted
**Decider**: Shivam Mishra + AI agent (session 2026-08-18)

## Context

`make check` failed on a fresh Windows (MSVC) setup for two independent reasons,
neither of which reproduces on macOS/Linux (so CI stayed green and both were
invisible until someone actually ran the gate on Windows):

1. **`cargo test` link failure — CRT mismatch.** The MSVC linker aborted with
   `LNK2038: mismatch detected for 'RuntimeLibrary': value 'MD_DynamicRelease'
   doesn't match value 'MT_StaticRelease'`. Root cause: `tokenizers`' default
   feature set includes `esaxx_fast`, which enables `esaxx-rs/cpp`. That crate's
   `build.rs` hardcodes `.static_crt(true)`, compiling its bundled C++
   (`esaxx.cpp`) against the **static** CRT (`/MT`). ONNX Runtime (`ort` /
   `ort-sys`) ships a prebuilt binary linked against the **dynamic** CRT (`/MD`),
   and Rust std defaults to `/MD` on `*-pc-windows-msvc`. MSVC refuses to mix the
   two in one link. `esaxx_fast` is a C++ suffix-automaton used **only to train
   Unigram tokenizer models** — it is never exercised by tokenizer loading or
   encoding at inference time, which is all `vocalai-core` does.

2. **`test-py` recipe not portable to Windows.** The recipe used a POSIX-shell
   conditional (`if [ -x .venv/bin/python ]; then ...; else pytest; fi`) probing
   the POSIX venv layout (`.venv/bin/python`). On Windows there is no `sh` on
   PATH, so GNU Make runs recipes through `cmd.exe`, which can't parse `[ -x ]`
   (`-x was unexpected at this time`). Even once that parsed, the interpreter
   lives at `.venv/Scripts/python.exe` on Windows, and `cmd.exe` needs
   backslashes for a leading executable path (`.venv/Scripts/...` → `'.venv' is
   not recognized`).

## Decision

**1. Drop `esaxx_fast` from the `tokenizers` dependency.** In
`crates/vocalai-core/Cargo.toml`:

```toml
tokenizers = { version = "0.23.1", default-features = false, features = ["onig", "progressbar"] }
```

This keeps `onig` (regex pre-tokenization) and `progressbar`, and drops only
`esaxx_fast` → `esaxx-rs` is compiled without its `cpp` feature → no static-CRT
C++ object is produced → the link succeeds. Tokenizer *inference* output is
byte-identical; only Unigram *training* speed would change, and this project
does no tokenizer training. Applied unconditionally (not `[target.'cfg(windows)']`)
because the C++ path is dead weight on every platform.

**2. Select the pytest interpreter by OS at make-parse time, not via a shell
conditional.** In `Makefile`:

```makefile
ifeq ($(OS),Windows_NT)
  VENV_PY := .venv\Scripts\python.exe          # backslashes: cmd.exe leading-exe path
  VENV_PY_GLOB := export/.venv/Scripts/python.exe  # forward slashes: $(wildcard)
else
  VENV_PY := .venv/bin/python
  VENV_PY_GLOB := export/.venv/bin/python
endif

PYTEST := $(if $(wildcard $(VENV_PY_GLOB)),$(VENV_PY) -m pytest,pytest)
```

The four `test-py*` recipes then reduce to `cd export && $(PYTEST) [markers]`,
which is valid under both `cmd.exe` and POSIX `sh`. Existence is checked with
make's own `$(wildcard)` (no shell needed), preserving the original
"prefer-venv-else-PATH-pytest" behavior.

## Rationale

- **Disabling an unused feature beats fighting the CRT.** The alternatives
  (forcing `+crt-static` globally, patching/forking `esaxx-rs`) are heavier and
  each breaks something else — notably `ort`'s prebuilt is dynamic-CRT, so
  forcing static just moves the mismatch. Removing the C++ that shouldn't be
  compiled at all is the smallest correct change.
- **`$(wildcard)` + `$(OS)` keep the Makefile shell-agnostic.** GNU Make on
  Windows falls back to `cmd.exe` when no `sh` is on PATH; encoding the
  branch in make variables rather than shell syntax means the same recipe runs
  identically under `cmd.exe`, Git Bash, and Linux/macOS `sh`.

## Alternatives rejected

- **Force `-C target-feature=+crt-static` via `.cargo/config.toml`**: rejected —
  the `ort` prebuilt ONNX Runtime is dynamic-CRT, so this just flips the
  mismatch to `ort` instead of `esaxx`.
- **`[patch]` `esaxx-rs` to a fork with `.static_crt(false)`**: rejected —
  maintenance burden and a vendored fork, to keep a feature we don't use.
- **Set `SHELL := bash` in the Makefile**: rejected — the only `bash` on the
  dev's Windows box is WSL's, which runs in the Linux filesystem view and can't
  execute the Windows-native `.venv/Scripts/python.exe`.
- **Gate the tokenizers change behind `cfg(windows)`**: rejected — the C++
  suffix-automaton is unused on every platform; a global drop is simpler and
  keeps all platforms building the same feature set.

## Consequences

**Easier**: `make check` passes on Windows/MSVC with no per-developer env
hacks; the Makefile's Python targets now work under `cmd.exe`.

**Harder**: nothing for this project. If a *future* feature ever needs to
*train* a Unigram tokenizer in-process, `esaxx_fast` would have to be
re-enabled — and would reintroduce the CRT conflict on Windows, to be solved
then (likely by building `ort` from source with a static CRT, or vendoring an
`esaxx-rs` that respects the target CRT feature).

**New commitments**:
- If `tokenizers` is bumped, re-check its default feature list — a future
  version could re-introduce a C++/CRT-sensitive default.
