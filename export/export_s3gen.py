"""Export the S3Gen flow-matching estimator (ConditionalDecoder) to ONNX.

Usage:
    python export_s3gen.py [--out models/s3gen_estimator.onnx]

Input: a single CFG-doubled step of `ConditionalCFM.solve_euler`'s per-step call
into `self.estimator.forward(x, mask, mu, t, spks, cond)` (see
chatterbox/models/s3gen/flow_matching.py):
  - x:    (2*B, 80, T) — current noisy mel state
  - mask: (2*B, 1, T)  — float 0/1 mask (already `.to(dtype)`-cast upstream)
  - mu:   (2*B, 80, T) — encoder condition (only first B rows real; rest zero for CFG)
  - t:    (2*B,)       — current timestep, broadcast per-batch
  - spks: (2*B, 80)    — speaker embedding (only first B rows real for CFG)
  - cond: (2*B, 80, T) — prompt condition (only first B rows real for CFG)
Output: dxdt (2*B, 80, T).

`r` (meanflow end-time) is intentionally not exposed: the base Chatterbox model
uses `meanflow=False`, so `ConditionalDecoder.forward` never reads `r` on this
path (see docs/phase1-onnx-rust-cli-plan.md §2.4 — meanflow/turbo is out of
scope for Phase 1). The CFG-doubling and Euler update itself is *not* traced
here — it is reimplemented as a host-side loop in `vocalai-core/src/s3gen.rs`
(and replicated in `parity_check.py` for validation), matching the pattern of
T3's decode loop (see plan §4).

The estimator's own control flow is static for this config (`static_chunk_size
= 0`, `use_dynamic_chunk=False` in ConditionalDecoder.__init__/decoder.py), so
`add_optional_chunk_mask` always takes its `else: chunk_masks = masks` branch
and traces cleanly with no complex-tensor or STFT tricks needed (contrast with
export_hifigan.py).

See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 3).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _common import export_onnx, load_s3gen, models_dir

MEL_CHANNELS = 80
SPK_EMB_DIM = 80
EXAMPLE_BATCH = 1  # solve_euler CFG-doubles this to 2 * EXAMPLE_BATCH
EXAMPLE_FRAMES = 50  # matches export_hifigan.py's fixed-frame-count example (see its module docstring)


def build_estimator(device: str = "cpu") -> torch.nn.Module:
    s3gen = load_s3gen(device=device)
    estimator = s3gen.flow.decoder.estimator
    estimator.eval()
    return estimator


def example_inputs(device: str = "cpu") -> tuple[torch.Tensor, ...]:
    batch = 2 * EXAMPLE_BATCH
    x = torch.randn(batch, MEL_CHANNELS, EXAMPLE_FRAMES, device=device)
    mask = torch.ones(batch, 1, EXAMPLE_FRAMES, device=device)
    mu = torch.randn(batch, MEL_CHANNELS, EXAMPLE_FRAMES, device=device)
    t = torch.rand(batch, device=device)
    spks = torch.randn(batch, SPK_EMB_DIM, device=device)
    cond = torch.randn(batch, MEL_CHANNELS, EXAMPLE_FRAMES, device=device)
    return x, mask, mu, t, spks, cond


def export(out_path: Path, device: str = "cpu") -> Path:
    estimator = build_estimator(device=device)
    example = example_inputs(device=device)
    return export_onnx(
        estimator,
        example,
        out_path,
        input_names=["x", "mask", "mu", "t", "spks", "cond"],
        output_names=["dxdt"],
        dynamic_axes={
            "x": {0: "batch", 2: "frames"},
            "mask": {0: "batch", 2: "frames"},
            "mu": {0: "batch", 2: "frames"},
            "t": {0: "batch"},
            "spks": {0: "batch"},
            "cond": {0: "batch", 2: "frames"},
            "dxdt": {0: "batch", 2: "frames"},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .onnx path (default: models/s3gen_estimator.onnx)",
    )
    args = parser.parse_args()
    out_path = args.out or (models_dir() / "s3gen_estimator.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export(out_path)
    print(f"Exported S3Gen flow estimator to {out_path}")


if __name__ == "__main__":
    main()
