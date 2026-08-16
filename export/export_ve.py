"""Export the voice encoder (speaker embedding network) from chatterbox-tts to ONNX.

Usage:
    python export_ve.py [--out models/ve.onnx]

Input: unscaled mel spectrogram of shape (B, ve_partial_frames=160, num_mels=40).
Output: L2-normalized speaker embedding of shape (B, 256).

See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 2).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _common import export_onnx, load_voice_encoder, models_dir


def build_module(device: str = "cpu"):
    return load_voice_encoder(device=device)


def export(out_path: Path, device: str = "cpu") -> Path:
    ve = build_module(device=device)
    example = torch.rand(1, ve.hp.ve_partial_frames, ve.hp.num_mels, device=device)
    return export_onnx(
        ve,
        (example,),
        out_path,
        input_names=["mels"],
        output_names=["speaker_embedding"],
        dynamic_axes={"mels": {0: "batch"}, "speaker_embedding": {0: "batch"}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .onnx path (default: models/ve.onnx)",
    )
    args = parser.parse_args()
    out_path = args.out or (models_dir() / "ve.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export(out_path)
    print(f"Exported voice encoder to {out_path}")


if __name__ == "__main__":
    main()
