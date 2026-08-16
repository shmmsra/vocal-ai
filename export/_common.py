"""Shared helpers for export/ scripts: model loading, ONNX export, output paths.

Dev-time only — not shipped. See docs/phase1-onnx-rust-cli-plan.md §6/§7.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


def models_dir() -> Path:
    """Dev-time export output dir. Git-ignored (see .gitignore); never commit its contents."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


REPO_ID = "ResembleAI/chatterbox"


@lru_cache(maxsize=None)
def load_voice_encoder(device: str = "cpu"):
    """Load just the voice-encoder checkpoint (downloads ve.safetensors on first run).

    Milestone 2 only needs VE/S3Gen/S3-tokenizer, not T3 or PerthNet, so we skip
    ChatterboxTTS.from_pretrained() entirely — it unconditionally constructs a
    PerthImplicitWatermarker, which errors in chatterbox-tts==0.1.7 + resemble-perth==1.0.1
    (perth's PerthNet import silently no-ops on a missing `pkg_resources`, and
    ChatterboxTTS.__init__ calls PerthImplicitWatermarker() unguarded). Not needed for
    HiFiGAN/VE/S3-tokenizer export, so avoid downloading T3's weights too.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from chatterbox.models.voice_encoder import VoiceEncoder

    ckpt = hf_hub_download(repo_id=REPO_ID, filename="ve.safetensors")
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(ckpt))
    ve.to(device).eval()
    return ve


@lru_cache(maxsize=None)
def load_s3gen(device: str = "cpu"):
    """Load just the S3Gen checkpoint (downloads s3gen.safetensors on first run). See
    load_voice_encoder() for why this bypasses ChatterboxTTS.from_pretrained()."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from chatterbox.models.s3gen import S3Gen

    ckpt = hf_hub_download(repo_id=REPO_ID, filename="s3gen.safetensors")
    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(ckpt), strict=False)
    s3gen.to(device).eval()
    return s3gen


def export_onnx(
    module: torch.nn.Module,
    example_inputs: tuple,
    out_path: Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
    opset: int = 18,
) -> Path:
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            example_inputs,
            str(out_path),
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dict(dynamic_axes) if dynamic_axes else None,
            opset_version=opset,
            do_constant_folding=True,
        )
    return out_path


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def allclose_report(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> tuple[bool, float]:
    """Returns (passed, max_abs_diff) comparing two same-shaped arrays."""
    if a.shape != b.shape:
        return False, float("inf")
    diff = max_abs_diff(a, b)
    ok = bool(np.allclose(a, b, atol=atol, rtol=rtol))
    return ok, diff
