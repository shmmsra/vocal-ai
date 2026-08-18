"""Stage the text tokenizer (`tokenizer.json`) into `models/`.

Unlike every other file in `export/`, this isn't a model export -- `tokenizer.json`
is already the exact file format the Rust `tokenizers` crate reads directly
(`Tokenizer::from_file`), so there's no ONNX graph and no parity check (plan
§4: "Text tokenizer ... no ONNX export needed"). But it still needs to land in
`models/` alongside the real exports, the same as `ve.safetensors`/
`t3_cfg.safetensors`/`s3gen.safetensors`/`conds.pt` get pulled in by the other
scripts via `hf_hub_download` (see `_common.py`'s loaders and
`export_default_voice.py`) -- nothing did that for `tokenizer.json` until now,
so a from-scratch `models/` build was missing it (found while manually testing
VAI-006 part B.1).

Usage:
    python fetch_tokenizer.py [--out-dir models]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from _common import REPO_ID, models_dir


def fetch(out_dir: Path) -> Path:
    src = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json")
    out_path = out_dir / "tokenizer.json"
    shutil.copyfile(src, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: models/)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or models_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {fetch(out_dir)}")


if __name__ == "__main__":
    main()
