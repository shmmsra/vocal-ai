"""Publish exported model artifacts (models/) to the public HuggingFace Hub repo.

Run by .github/workflows/models-export.yml after `make export` + `make test-py-parity`
have both succeeded, or locally via `make publish-models`. Requires HF_TOKEN in the
environment (a HuggingFace write token) -- never commit it, never pass it on the
command line where it could land in shell history; export it in your shell session
(locally) or set it as a repo secret (CI) instead.

Usage:
    HF_TOKEN=hf_... python publish_models.py [--repo-id shmmsra/vocal-ai-models]
        [--models-dir models] [--commit-message "..."]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_REPO_ID = "shmmsra/vocal-ai-models"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_CARD = """---
license: mit
tags:
  - text-to-speech
  - onnx
  - chatterbox
---

# vocal-ai model artifacts

ONNX graphs and auxiliary `.npy` tensors exported from
[ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) (MIT) for use by
[vocal-ai](https://github.com/shmmsra/vocal-ai), a standalone Rust + ONNX Runtime TTS CLI.

These files are dev-time build artifacts, not source -- see vocal-ai's `export/` scripts
for how they're produced, and this repo's own `THIRD_PARTY_LICENSES` file (in the
vocal-ai release bundles) for the upstream license notices.
"""


class PublishError(Exception):
    """Raised for a misconfigured or failed publish attempt."""


def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise PublishError(
            "HF_TOKEN is not set. Export it in your shell (never on the command line or in "
            "a committed file) -- a HuggingFace write token from "
            "https://huggingface.co/settings/tokens -- or, in CI, set it as a repo secret."
        )
    return token


def publish(models_dir: Path, repo_id: str, token: str, commit_message: str) -> str:
    """Upload models_dir to repo_id, creating it (public) if it doesn't exist yet.

    Returns the resulting commit SHA.
    """
    if not models_dir.is_dir():
        raise PublishError(f"models dir not found: {models_dir}")
    if not any(models_dir.rglob("*.onnx")):
        raise PublishError(f"no .onnx files found under {models_dir} -- run `make export` first")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

    readme = models_dir / "README.md"
    wrote_readme = not readme.exists()
    if wrote_readme:
        readme.write_text(MODEL_CARD, encoding="utf-8")

    try:
        commit_info = api.upload_folder(
            folder_path=str(models_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
    finally:
        if wrote_readme:
            readme.unlink(missing_ok=True)

    return commit_info.oid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--commit-message", default="Publish exported model artifacts")
    args = parser.parse_args(argv)

    try:
        token = require_hf_token()
        sha = publish(args.models_dir, args.repo_id, token, args.commit_message)
    except PublishError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"published to https://huggingface.co/{args.repo_id} @ {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
