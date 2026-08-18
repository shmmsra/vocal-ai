"""Export PerthNet's watermark encoder (from the external `resemble-perth`
package) to ONNX.

Only `PerthNet.encoder` — the Conv1d residual-encoder submodule, the sole
learned/exportable piece — is exported. `PerthImplicitWatermarker.apply_watermark`
does everything else (STFT, log-magnitude dB normalization, ISTFT, 24kHz<->32kHz
resampling) as classical DSP outside the network; that DSP is reimplemented
directly in `crates/vocalai-core/src/watermark.rs`, matching this plan's stated
architecture of doing preprocessing in Rust rather than baking it into ONNX
graphs (docs/phase1-onnx-rust-cli-plan.md §5).

Usage:
    python export_perthnet.py [--out models/perthnet_encoder.onnx]

Input: log-magnitude spectrogram `magspec` of shape (B, nfreq=1025, T), T dynamic
(nfreq = n_fft // 2 + 1 = 1025, per PerthNet's default hparams: n_fft=2048).
Output: watermarked log-magnitude spectrogram `wmarked_magspec`, same shape —
only the first `subband=128` frequency bins (below `max_wmark_freq=2000Hz` at
`sample_rate=32000Hz`) are modified; the encoder's `mask` output is dropped
(chatterbox's `apply_watermark` never uses it — see `perth_watermarker.py`) and
recomputed directly in Rust instead (plain arithmetic, no need to round-trip
through ONNX — same treatment as T3's hand-rolled sampling math).

See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 5) and
docs/decisions/0008-third-party-license-attribution.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from _common import export_onnx, load_perthnet, models_dir

NFREQ = 1025  # n_fft // 2 + 1, n_fft=2048


class PerthEncoderWrapper(nn.Module):
    """Wraps `PerthNet.encoder` to return only `wmarked_magspec` (drops `mask`)."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, magspec: torch.Tensor) -> torch.Tensor:
        wmarked, _mask = self.encoder(magspec)
        return wmarked


def build_wrapper(device: str = "cpu") -> PerthEncoderWrapper:
    perth_net = load_perthnet(device=device)
    return PerthEncoderWrapper(perth_net.encoder)


def export(out_path: Path, device: str = "cpu") -> Path:
    wrapper = build_wrapper(device=device)
    example = torch.randn(1, NFREQ, 50, device=device)
    return export_onnx(
        wrapper,
        (example,),
        out_path,
        input_names=["magspec"],
        output_names=["wmarked_magspec"],
        dynamic_axes={"magspec": {2: "time"}, "wmarked_magspec": {2: "time"}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .onnx path (default: models/perthnet_encoder.onnx)",
    )
    args = parser.parse_args()
    out_path = args.out or (models_dir() / "perthnet_encoder.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export(out_path)
    print(f"Exported PerthNet encoder to {out_path}")


if __name__ == "__main__":
    main()
