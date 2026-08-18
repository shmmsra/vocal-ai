"""Dump the built-in default voice's conditioning tensors (`conds.pt`) to `.npy`.

`ChatterboxTTS.from_local`/`from_pretrained` load a bundled `conds.pt`
(`chatterbox.tts.Conditionals.load`, a `torch.load`) when no reference voice is
given, so `generate()` has *some* T3/S3Gen conditioning to work with (see
`tts.py`'s `assert self.conds is not None, "Please prepare_conditionals first or
specify audio_prompt_path"`). `torch.load` isn't reachable from Rust; this script
is a one-off dump of each tensor field to `.npy` so `vocalai-core` can load them
directly, enabling `vocalai --text "..." --out out.wav` with no `--voice` flag
(Milestone 6 acceptance criterion, docs/issues.md VAI-006).

Not a model export -- no ONNX graph, no parity check (there's nothing to check
numerical parity *against*; this only reads tensors out of a checkpoint file
unchanged).

Usage:
    python export_default_voice.py [--out-dir models/default_voice]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from _common import REPO_ID, models_dir

FIELDS = (
    # (attribute path, output filename stem)
    ("t3.speaker_emb", "t3_speaker_emb"),
    ("t3.cond_prompt_speech_tokens", "t3_cond_prompt_speech_tokens"),
    ("t3.emotion_adv", "t3_emotion_adv"),
    ("gen.prompt_token", "s3gen_prompt_token"),
    ("gen.prompt_token_len", "s3gen_prompt_token_len"),
    ("gen.prompt_feat", "s3gen_prompt_feat"),
    ("gen.embedding", "s3gen_embedding"),
)


def _resolve(conds, attr_path: str):
    obj = conds
    for part in attr_path.split("."):
        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
    return obj


def dump(out_dir: Path, device: str = "cpu") -> list[Path]:
    from chatterbox.tts import Conditionals

    conds_path = hf_hub_download(repo_id=REPO_ID, filename="conds.pt")
    conds = Conditionals.load(conds_path, map_location=device)

    written = []
    for attr_path, stem in FIELDS:
        value = _resolve(conds, attr_path)
        if value is None:
            continue
        out_path = out_dir / f"{stem}.npy"
        np.save(out_path, value.detach().cpu().numpy())
        written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: models/default_voice)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or (models_dir() / "default_voice")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in dump(out_dir):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
