"""Export CAMPPlus (S3Gen's x-vector speaker encoder) to ONNX, plus the
speaker-embedding affine-layer weights.

`S3Gen.speaker_encoder` is a separate x-vector network from the voice encoder
(`ve.onnx`, Milestone 2) -- VE's speaker embedding conditions T3; CAMPPlus's
conditions S3Gen's flow-matching decoder as `spks`. See
docs/decisions/0009-s3gen-flow-encoder-and-campplus-export.md for why this
network was missing from the original export set.

Usage:
    python export_campplus.py [--out models/campplus.onnx]

**Why one fixed-length graph, not a dynamic or bucketed one**: `CAMPPlus.forward`
has no length/mask input at all -- unlike the flow encoder (`export_s3gen_flow_encoder.py`),
it can't distinguish real frames from padding. Empirically, its exported graph is
also *not* dynamic-length-safe in general, and the failure isn't simply "only the
traced length works" (contrast with the flow encoder) -- it's a genuine ONNX-export
bug in `CAMLayer.seg_pooling` (`xvector.py`): that method average-pools in
100-frame segments, expands each pooled value back across its segment, then trims
the result to the input's original length (`seg[..., :x.shape[-1]]`). Bisecting by
frame count shows the trim is computed correctly by the exported graph only when
it's a no-op -- i.e. only when the *pre-pooling* time length (`frames // 2`, after
`xvector.tdnn`'s stride-2 first layer) is itself an exact multiple of 100, which
happens iff `frames` is a multiple of 200. Every tested non-multiple (100, 150,
250, 300, 350, 500) is off by 0.4-1.8 absolute; every tested multiple of 200 (200,
400, 600, 800) matches the eager reference to ~1e-6, *regardless of content* --
confirming this is a structural export bug in that one op, not a generalization
gap. `CAMPPLUS_FRAMES` **must** stay a multiple of 200. Because there's no masking
to make a zero-padded-to-bucket input behave like the unpadded original
(`StatsPool`'s statistics would be computed over the padding too, corrupting the
embedding), bucketing the *input* the way the flow encoder does is unsound here
regardless of the graph. Instead, this exports a single fixed window
(`CAMPPLUS_FRAMES` frames, ~4s at Kaldi's typical 10ms frame shift -- a
common x-vector enrollment-window length) and Rust must always feed exactly that
many frames of *real* fbank content (trimming a longer reference clip, or
repeating a shorter one -- never zero-padding) -- a Milestone 6 (Part B) wiring
concern, not this export step's.

Input: precomputed Kaldi-style fbank features, shape (1, `CAMPPLUS_FRAMES`, 80).
Feature extraction itself (`torchaudio.compliance.kaldi.fbank`, called
per-utterance in `CAMPPlus.inference()`/`xvector.py::extract_feature`) is
host-side preprocessing, not part of the traced graph -- the same ONNX-graph-
boundary convention already used for `ve.onnx` (precomputed mels) and
`s3tokenizer.onnx` (precomputed log-mel). Correctness of the Rust-side Kaldi-fbank
port is a residual risk tracked in the ADR above, not silently assumed.

Output: L2-*un*normalized x-vector embedding, shape (1, 192). `flow.py`'s
`inference()` does `embedding = F.normalize(embedding, dim=1)` then
`spk_embed_affine_layer(embedding)` *after* this network runs -- both are cheap
enough (a norm + one Linear(192, 80)) to hand-roll in Rust rather than wrap in
their own ONNX session (same treatment as T3's embedding-table lookup, ADR-0005).
This script dumps `spk_embed_affine_layer.weight`/`.bias` to `.npy` for that Rust
matmul.

See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 6) and
docs/decisions/0009-s3gen-flow-encoder-and-campplus-export.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from _common import export_onnx, load_s3gen, models_dir

FEAT_DIM = 80
# Must be a multiple of 200 -- see module docstring (CAMLayer.seg_pooling export bug).
CAMPPLUS_FRAMES = 400


def build_module(device: str = "cpu") -> torch.nn.Module:
    s3gen = load_s3gen(device=device)
    campplus = s3gen.speaker_encoder
    campplus.eval()
    return campplus


def export(out_path: Path, device: str = "cpu") -> Path:
    assert CAMPPLUS_FRAMES % 200 == 0, "CAMPPLUS_FRAMES must be a multiple of 200 (see module docstring)"
    campplus = build_module(device=device)
    example = torch.randn(1, CAMPPLUS_FRAMES, FEAT_DIM, device=device)
    return export_onnx(
        campplus,
        (example,),
        out_path,
        input_names=["fbank"],
        output_names=["embedding"],
    )


def export_affine_layer_weights(out_dir: Path, device: str = "cpu") -> tuple[Path, Path]:
    s3gen = load_s3gen(device=device)
    affine = s3gen.flow.spk_embed_affine_layer
    weight_path = out_dir / "s3gen_spk_embed_affine_weight.npy"
    bias_path = out_dir / "s3gen_spk_embed_affine_bias.npy"
    np.save(weight_path, affine.weight.detach().cpu().numpy())
    np.save(bias_path, affine.bias.detach().cpu().numpy())
    return weight_path, bias_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .onnx path (default: models/campplus.onnx)",
    )
    args = parser.parse_args()
    out_path = args.out or (models_dir() / "campplus.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export(out_path)
    print(f"Exported CAMPPlus to {out_path}")
    weight_path, bias_path = export_affine_layer_weights(out_path.parent)
    print(f"Exported speaker-embedding affine-layer weights to {weight_path}, {bias_path}")


if __name__ == "__main__":
    main()
