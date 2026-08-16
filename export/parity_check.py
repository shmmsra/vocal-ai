"""Check numerical parity between exported ONNX graphs and the PyTorch reference.

Usage:
    python parity_check.py [--component hifigan|ve|s3tokenizer|s3gen] [--atol 1e-4] [--rtol 1e-3]

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
import export_s3gen
import export_s3tokenizer
import export_ve
from _common import allclose_report, load_s3gen, models_dir

DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 1e-3

S3GEN_N_TIMESTEPS = 10  # fixed by chatterbox/tts.py's s3gen.inference() call (no override, non-meanflow)


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


def _cosine_t_span(n_timesteps: int) -> np.ndarray:
    """Matches ConditionalCFM.solve_euler's cosine `t_scheduler` (the only scheduler
    the base Chatterbox config uses; see chatterbox/models/s3gen/configs.py)."""
    t_span = np.linspace(0.0, 1.0, n_timesteps + 1, dtype=np.float32)
    return (1.0 - np.cos(t_span * 0.5 * np.pi)).astype(np.float32)


def _solve_euler_onnx(
    session: ort.InferenceSession,
    x0: np.ndarray,
    t_span: np.ndarray,
    mu: np.ndarray,
    mask: np.ndarray,
    spks: np.ndarray,
    cond: np.ndarray,
    cfg_rate: float,
) -> np.ndarray:
    """Python-side replica of `ConditionalCFM.solve_euler`'s CFG-doubled Euler loop
    (chatterbox/models/s3gen/flow_matching.py), driving the exported estimator ONNX
    graph instead of the PyTorch module. This is the exact loop `vocalai-core::s3gen`
    reimplements in Rust — see `crates/vocalai-core/src/s3gen.rs`.
    """
    b = mu.shape[0]
    x = x0.copy()
    zeros_mu, zeros_spks, zeros_cond = np.zeros_like(mu), np.zeros_like(spks), np.zeros_like(cond)
    for i in range(len(t_span) - 1):
        t, r = t_span[i : i + 1], t_span[i + 1 : i + 2]
        feeds = {
            "x": np.concatenate([x, x], axis=0),
            "mask": np.concatenate([mask, mask], axis=0),
            "mu": np.concatenate([mu, zeros_mu], axis=0),
            "t": np.concatenate([t, t]),
            "spks": np.concatenate([spks, zeros_spks], axis=0),
            "cond": np.concatenate([cond, zeros_cond], axis=0),
        }
        (dxdt,) = session.run(None, feeds)
        dxdt_cond, dxdt_uncond = dxdt[:b], dxdt[b:]
        dxdt_combined = (1.0 + cfg_rate) * dxdt_cond - cfg_rate * dxdt_uncond
        dt = float((r - t)[0])
        x = x + dt * dxdt_combined
    return x


def check_s3gen(atol: float, rtol: float) -> ParityResult:
    torch.manual_seed(0)
    s3gen = load_s3gen()
    decoder = s3gen.flow.decoder
    cfg_rate = float(decoder.inference_cfg_rate)

    onnx_path = models_dir() / "s3gen_estimator.onnx"
    if not onnx_path.exists():
        export_s3gen.export(onnx_path)
    hifigan_path = models_dir() / "hifigan.onnx"
    if not hifigan_path.exists():
        export_hifigan.export(hifigan_path)

    batch, frames = export_s3gen.EXAMPLE_BATCH, export_s3gen.EXAMPLE_FRAMES
    mu = torch.randn(batch, export_s3gen.MEL_CHANNELS, frames)
    mask = torch.ones(batch, 1, frames)
    spks = torch.randn(batch, export_s3gen.SPK_EMB_DIM)
    cond = torch.randn(batch, export_s3gen.MEL_CHANNELS, frames)
    x0 = torch.randn(batch, export_s3gen.MEL_CHANNELS, frames)

    t_span = torch.from_numpy(_cosine_t_span(S3GEN_N_TIMESTEPS))
    with torch.no_grad():
        mel_ref = decoder.solve_euler(x0.clone(), t_span, mu, mask, spks, cond, meanflow=False)
        wav_ref = export_hifigan.build_wrapper()(mel_ref).numpy()
    mel_ref = mel_ref.numpy()

    estimator_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    mel_onnx = _solve_euler_onnx(
        estimator_session,
        x0.numpy(),
        _cosine_t_span(S3GEN_N_TIMESTEPS),
        mu.numpy(),
        mask.numpy(),
        spks.numpy(),
        cond.numpy(),
        cfg_rate,
    )
    (wav_onnx,) = _run_onnx(hifigan_path, {"speech_feat": mel_onnx.astype(np.float32)})

    mel_passed, mel_diff = allclose_report(mel_ref, mel_onnx, atol, rtol)
    wav_passed, wav_diff = allclose_report(wav_ref, wav_onnx, atol, rtol)
    return ParityResult("s3gen", mel_passed and wav_passed, max(mel_diff, wav_diff))


CHECKS = {
    "hifigan": check_hifigan,
    "ve": check_ve,
    "s3tokenizer": check_s3tokenizer,
    "s3gen": check_s3gen,
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
