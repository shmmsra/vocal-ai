"""Structural-only validation of exported model artifacts / packaged release bundles.

Deliberately does NOT load an ONNX Runtime session or run any inference. Per the
repo owner's explicit instruction: automated checks (GitHub Actions) must never
execute model inference, CPU or GPU -- only verify the files are well-formed.
Real end-to-end audio validation is a manual step, see docs/manual-testing.md.

Used by both .github/workflows/models-export.yml (validates a freshly exported
models/ dir before publishing) and .github/workflows/release.yml (validates a
staged per-platform bundle: models/ + the compiled binary + license files).

Usage:
    python smoke_test_artifact.py --models-dir models [--binary path/to/vocalai]
        [--extra-file path/to/LICENSE ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


class SmokeTestError(Exception):
    """Raised when a staged artifact fails structural validation."""


def check_onnx_file(path: Path) -> None:
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def check_npy_file(path: Path) -> None:
    import numpy as np

    np.load(str(path), allow_pickle=False)


def check_tokenizer_json(path: Path) -> None:
    json.loads(path.read_text(encoding="utf-8"))


def check_models_dir(models_dir: Path) -> list[str]:
    """Validate every .onnx/.npy/tokenizer.json found under models_dir.

    Sweeps whatever is present rather than requiring an exhaustive fixed
    manifest -- the export/ scripts' output set changes over time (e.g.
    --with-voice-cloning adds extra files), so the invariant worth gating on
    is "every file that exists is well-formed", not "exactly these N files
    exist".
    """
    if not models_dir.is_dir():
        raise SmokeTestError(f"models dir not found: {models_dir}")

    checked: list[str] = []
    onnx_files = sorted(models_dir.rglob("*.onnx"))
    if not onnx_files:
        raise SmokeTestError(f"no .onnx files found under {models_dir} -- export produced nothing")

    for f in onnx_files:
        check_onnx_file(f)
        checked.append(str(f))

    for f in sorted(models_dir.rglob("*.npy")):
        check_npy_file(f)
        checked.append(str(f))

    for f in sorted(models_dir.rglob("tokenizer.json")):
        check_tokenizer_json(f)
        checked.append(str(f))

    return checked


def check_binary(binary: Path) -> None:
    """Confirm the compiled binary starts and links -- `--version` only, never a real run."""
    if not binary.is_file():
        raise SmokeTestError(f"binary not found: {binary}")
    result = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise SmokeTestError(
            f"{binary} --version exited {result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        )


def check_extra_files(paths: list[Path]) -> None:
    for p in paths:
        if not p.is_file():
            raise SmokeTestError(f"expected file missing: {p}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--extra-file", action="append", type=Path, default=[], dest="extra_files")
    args = parser.parse_args(argv)

    try:
        checked = check_models_dir(args.models_dir)
        if args.binary is not None:
            check_binary(args.binary)
            checked.append(str(args.binary))
        if args.extra_files:
            check_extra_files(args.extra_files)
            checked.extend(str(p) for p in args.extra_files)
    except SmokeTestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"smoke test passed: {len(checked)} files validated")
    for f in checked:
        print(f"  ok  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
