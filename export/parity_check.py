"""Check numerical parity between exported ONNX graphs and the PyTorch reference.

Usage:
    python parity_check.py [--component hifigan|ve|s3tokenizer] [--atol 1e-4] [--rtol 1e-3]

Exits non-zero if any checked component fails. Per docs/agents/CONVENTIONS.md §3: no
exported component may be wired into vocalai-core until this passes for it.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import torch

import export_hifigan
import export_s3tokenizer
import export_ve
from _common import allclose_report, models_dir

DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 1e-3


@dataclass
class ParityResult:
    component: str
    passed: bool
    max_abs_diff: float


def _run_onnx(onnx_path, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)


def check_hifigan(atol: float, rtol: float) -> ParityResult:
    torch.manual_seed(0)
    wrapper = export_hifigan.build_wrapper()
    onnx_path = models_dir() / "hifigan.onnx"
    if not onnx_path.exists():
        export_hifigan.export(onnx_path)

    speech_feat = torch.randn(1, export_hifigan.MEL_CHANNELS, 50)
    with torch.no_grad():
        torch_out = wrapper(speech_feat).numpy()

    (onnx_out,) = _run_onnx(onnx_path, {"speech_feat": speech_feat.numpy()})
    passed, diff = allclose_report(torch_out, onnx_out, atol, rtol)
    return ParityResult("hifigan", passed, diff)


def check_ve(atol: float, rtol: float) -> ParityResult:
    torch.manual_seed(0)
    ve = export_ve.build_module()
    onnx_path = models_dir() / "ve.onnx"
    if not onnx_path.exists():
        export_ve.export(onnx_path)

    mels = torch.rand(1, ve.hp.ve_partial_frames, ve.hp.num_mels)
    with torch.no_grad():
        torch_out = ve(mels).numpy()

    (onnx_out,) = _run_onnx(onnx_path, {"mels": mels.numpy()})
    passed, diff = allclose_report(torch_out, onnx_out, atol, rtol)
    return ParityResult("ve", passed, diff)


def check_s3tokenizer(atol: float, rtol: float) -> ParityResult:
    torch.manual_seed(0)
    wrapper = export_s3tokenizer.build_wrapper()
    onnx_path = models_dir() / "s3tokenizer.onnx"
    if not onnx_path.exists():
        export_s3tokenizer.export(onnx_path)

    mel = torch.randn(1, export_s3tokenizer.N_MELS, export_s3tokenizer.EXAMPLE_FRAMES)
    mel_len = torch.tensor([export_s3tokenizer.EXAMPLE_FRAMES], dtype=torch.long)
    with torch.no_grad():
        torch_code, _ = wrapper(mel, mel_len)
    torch_code = torch_code.numpy()

    onnx_code, _ = _run_onnx(onnx_path, {"mel": mel.numpy(), "mel_len": mel_len.numpy()})
    # Tokens are discrete integer codes: parity means an exact match, not a tolerance band.
    passed = bool(np.array_equal(torch_code, onnx_code))
    diff = allclose_report(torch_code.astype(np.float64), onnx_code.astype(np.float64), 0, 0)[1]
    return ParityResult("s3tokenizer", passed, diff)


CHECKS = {
    "hifigan": check_hifigan,
    "ve": check_ve,
    "s3tokenizer": check_s3tokenizer,
}


def run_checks(components: list[str], atol: float, rtol: float) -> list[ParityResult]:
    return [CHECKS[name](atol, rtol) for name in components]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=sorted(CHECKS), default=None)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    args = parser.parse_args()

    components = [args.component] if args.component else sorted(CHECKS)
    results = run_checks(components, args.atol, args.rtol)

    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.component}: max_abs_diff={result.max_abs_diff:.3e}")
        all_passed = all_passed and result.passed

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
