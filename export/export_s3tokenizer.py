"""Export the S3 (speech) tokenizer encoder+quantizer from chatterbox-tts to ONNX.

Usage:
    python export_s3tokenizer.py [--out models/s3tokenizer.onnx]

Input: log-mel spectrogram ``mel`` of shape (1, n_mels=128, T) plus ``mel_len`` (1,).
  T must stay under 3000 frames (short-audio path; see S3TokenizerV2.quantize).
Output: quantized speech tokens ``code`` (1, T') and ``code_len`` (1,).

See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 2).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from _common import export_onnx, load_s3gen, models_dir

N_MELS = 128
EXAMPLE_FRAMES = 400  # well under the 3000-frame short-audio threshold

_original_view_as_real = torch.view_as_real


def _view_as_real_or_passthrough(x: torch.Tensor) -> torch.Tensor:
    """torch.view_as_real, except a passthrough for already-real input.

    AudioEncoderV2 (s3tokenizer) precomputes rotary-embedding angles as a complex
    buffer (`self.freqs_cis`, via `torch.polar`) and calls `torch.view_as_real` on
    it at every forward. ONNX has no complex dtype, so tracing fails the moment that
    buffer would be embedded as a graph constant. build_wrapper() replaces the
    buffer with the already-"view_as_real"-shaped real equivalent (see
    `_real_freqs_cis`) computed with identical math; this patched function then
    just passes it through instead of erroring on a non-complex input. Genuinely
    complex inputs (anywhere else in the process) still go through the real
    `torch.view_as_real`, so this is behavior-preserving outside this one path.
    """
    if torch.is_complex(x):
        return _original_view_as_real(x)
    return x


torch.view_as_real = _view_as_real_or_passthrough


def _real_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Real-valued equivalent of `torch.view_as_real(precompute_freqs_cis(dim, end, theta))`
    (s3tokenizer.model_v2), computed without ever constructing a complex tensor."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end).float()
    freqs = torch.outer(t, freqs)  # (end, dim // 2)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (end, dim)
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)  # (end, dim, 2)


class S3TokenizerExportWrapper(nn.Module):
    """Mirrors S3TokenizerV2.quantize()'s short-audio path without @torch.inference_mode()."""

    def __init__(self, tokenizer: nn.Module):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, mel: torch.Tensor, mel_len: torch.Tensor):
        hidden, code_len = self.tokenizer.encoder(mel, mel_len)
        code = self.tokenizer.quantizer.encode(hidden)
        return code, code_len


def build_wrapper(device: str = "cpu") -> S3TokenizerExportWrapper:
    """`load_s3gen()` is `@lru_cache`d, so `encoder` below is a *shared* module
    instance across every call in the process (e.g. `check_s3tokenizer()` calls
    this once directly, then again via `export()`). Guard the `freqs_cis`
    replacement with `is_complex` so a second call is a no-op instead of reading
    the already-real buffer's shape (whose last dim is 2, not the head dim) and
    computing a corrupted replacement from it.
    """
    s3gen = load_s3gen(device=device)
    tokenizer = s3gen.tokenizer
    tokenizer.eval()
    encoder = tokenizer.encoder
    if torch.is_complex(encoder.freqs_cis):
        dim, end = encoder.freqs_cis.shape[-1], encoder.freqs_cis.shape[0]
        encoder.freqs_cis = _real_freqs_cis(dim, end).to(device)
    return S3TokenizerExportWrapper(tokenizer)


def export(out_path: Path, device: str = "cpu") -> Path:
    wrapper = build_wrapper(device=device)
    mel = torch.randn(1, N_MELS, EXAMPLE_FRAMES, device=device)
    mel_len = torch.tensor([EXAMPLE_FRAMES], dtype=torch.long, device=device)
    return export_onnx(
        wrapper,
        (mel, mel_len),
        out_path,
        input_names=["mel", "mel_len"],
        output_names=["code", "code_len"],
        dynamic_axes={
            "mel": {0: "batch", 2: "mel_frames"},
            "mel_len": {0: "batch"},
            "code": {0: "batch", 1: "tokens"},
            "code_len": {0: "batch"},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .onnx path (default: models/s3tokenizer.onnx)",
    )
    args = parser.parse_args()
    out_path = args.out or (models_dir() / "s3tokenizer.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export(out_path)
    print(f"Exported S3 tokenizer to {out_path}")


if __name__ == "__main__":
    main()
