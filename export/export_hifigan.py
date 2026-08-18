"""Export the HiFiGAN vocoder (HiFTGenerator) from chatterbox-tts to ONNX.

Usage:
    python export_hifigan.py [--out models/hifigan.onnx]

Input: mel-like feature ``speech_feat`` of shape (1, 80, T).
Output: waveform of shape (1, hop_len * (T' - 1)) where T' is the internal STFT
frame count derived from T (see HiFiGANExportWrapper for the exact relationship).

HiFTGenerator.decode() calls `torch.stft`/`torch.istft` internally (its Neural Source
Filter conditioning path). Neither has full ONNX support in this torch version's
legacy exporter:
  - `torch.stft(..., return_complex=True)` has no ONNX symbolic (complex dtype
    unsupported); `return_complex=False` DOES export (native ONNX STFT op), provided
    centering is done by hand first (the symbolic assumes `center=False`).
  - `torch.istft` has no ONNX symbolic at all in this exporter.
This wrapper reimplements both directions as ONNX-exportable primitives: `_stft`
via manual reflect-pad + `torch.stft(..., return_complex=False)`, `_istft` via a
precomputed inverse-DFT matrix + `torch.nn.functional.fold` (overlap-add), with
window-envelope (COLA) normalization matching `torch.istft`'s default behavior.

DYNAMIC LENGTH (VAI-009): `speech_feat`'s time axis is a genuine ONNX dynamic axis.
Two trace-baking bugs had to be fixed to get there (see ADR-0009 for the same
category of issue in the flow-encoder/CAMPPlus exports): the overlap-add envelope
used to build its window via `.repeat(1, 1, num_frames)` with `num_frames` read as a
plain Python int off `.shape`, baking the traced frame count; and the deterministic
source noise (`_sine_gen_deterministic`) used to draw `torch.tensor(rng.randn(*shape))`
sized off the traced sample count, which the exporter registers as a literal ONNX
constant (see its own `TracerWarning`). Both are now built from a fixed-size buffer
sliced dynamically, so the exported graph generalizes to any input length up to
`_NOISE_BUFFER_SAMPLES`.

See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 2).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from _common import export_onnx, load_s3gen, models_dir

MEL_CHANNELS = 80
_SOURCE_NOISE_SEED = 0


def _fuse_weight_norm(module: nn.Module) -> None:
    """Fuse every weight_norm parametrization under `module` into plain weights.

    HiFTGenerator.remove_weight_norm() assumes the old function-based `weight_norm`
    API, but chatterbox-tts applies the new parametrize-based one (torch.nn.utils
    .parametrizations.weight_norm) — calling `torch.nn.utils.remove_weight_norm` on
    those raises ValueError. Walk the tree instead and remove via the matching API.
    """
    for submodule in module.modules():
        if is_parametrized(submodule, "weight"):
            remove_parametrizations(submodule, "weight", leave_parametrized=True)


def _stft_onnx(x: torch.Tensor, n_fft: int, hop_len: int, window: torch.Tensor):
    """ONNX-exportable equivalent of `torch.stft(x, n_fft, hop_len, n_fft, window,
    center=True, pad_mode="reflect", onesided=True, return_complex=True)`, split
    into (real, imag) each shaped (B, n_fft // 2 + 1, T')."""
    padded = F.pad(x.unsqueeze(1), (n_fft // 2, n_fft // 2), mode="reflect").squeeze(1)
    spec = torch.stft(
        padded,
        n_fft,
        hop_length=hop_len,
        win_length=n_fft,
        window=window,
        center=False,
        onesided=True,
        return_complex=False,
    )  # (B, F, T', 2)
    return spec[..., 0], spec[..., 1]


def _inverse_dft_matrices(n_fft: int, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) matrices of shape (F, n_fft) s.t. for a single onesided
    frame spectrum (real, imag) of length F = n_fft // 2 + 1:
        windowed_time_frame = real @ cos + imag @ sin
    reproduces one windowed frame of the real inverse-DFT (matching what
    `torch.istft` computes per-frame before overlap-add), with the synthesis
    window folded in.
    """
    freqs = n_fft // 2 + 1
    k = torch.arange(freqs, dtype=torch.float64).unsqueeze(1)  # (F, 1)
    n = torch.arange(n_fft, dtype=torch.float64).unsqueeze(0)  # (1, N)
    theta = 2 * math.pi * k * n / n_fft
    scale = torch.full((freqs, 1), 2.0, dtype=torch.float64)
    scale[0, 0] = 1.0
    if n_fft % 2 == 0:
        scale[-1, 0] = 1.0
    window64 = window.double().unsqueeze(0)
    cos_mat = (scale * torch.cos(theta) / n_fft) * window64
    sin_mat = (-scale * torch.sin(theta) / n_fft) * window64
    return cos_mat.float(), sin_mat.float()


_NOISE_BUFFER_SAMPLES = 1_500_000  # comfortably above the largest realistic S3Gen bucket's sample count


def _sine_gen_deterministic(
    f0: torch.Tensor,
    harmonic_num: int,
    sine_amp: float,
    noise_std: float,
    voiced_threshold: float,
    sampling_rate: int,
    noise_buffer: torch.Tensor,
) -> torch.Tensor:
    """Reimplements SineGen.forward()'s math, replacing its two live-random draws
    (Uniform phase, randn noise) with numpy-RandomState-seeded constants.

    `torch.manual_seed` + `torch.randn`/`torch.distributions.Uniform.sample` do NOT
    reproduce identically between an eager call and the same call replayed through
    `torch.jit.trace` (confirmed empirically — `check_trace` itself flags a mismatch
    even with an identical seed reset immediately beforehand). `numpy.random.RandomState`
    isn't touched by JIT tracing's tensor-op interception, so it reproduces exactly:
    same seed in, bit-identical draw out, in both eager execution and inside the traced
    graph — the property we actually need to compare (this wrapper) against itself
    (the ONNX export of this wrapper) in parity_check.py. Only the sine/noise mixing
    (`sine_waves`) is returned — the caller only needs that, not `uv`/`noise` separately.

    `noise_buffer` (built once in `HiFiGANExportWrapper.__init__`, same seed) is
    sliced to the real dynamic length rather than drawn fresh here: `torch.tensor(
    rng.randn(*sine_waves.shape))` reads `.shape` as a plain Python int at trace
    time and the exporter registers the result as an ONNX constant (see its own
    TracerWarning), baking the traced sample count into the graph. Slicing a
    pre-built buffer keeps the same bit-exact-reproducibility property while
    staying dynamic-length-safe.
    """
    b, _, length = f0.shape
    freq_mat = torch.zeros((b, harmonic_num + 1, length), dtype=f0.dtype, device=f0.device)
    for i in range(harmonic_num + 1):
        freq_mat[:, i : i + 1, :] = f0 * (i + 1) / sampling_rate
    theta_mat = 2 * math.pi * (torch.cumsum(freq_mat, dim=-1) % 1)

    # torch.tensor(...), not torch.from_numpy(...): the latter traces as aten::lift_fresh,
    # which this exporter can't translate to ONNX. Same reason we build phase_vec's
    # zeroed first harmonic via torch.cat rather than an in-place `phase_vec[:, 0, :] = 0`
    # (in-place writes into a freshly-lifted constant also trigger aten::lift_fresh).
    rng = np.random.RandomState(_SOURCE_NOISE_SEED)
    phase_rest = torch.tensor(
        rng.uniform(-math.pi, math.pi, size=(b, harmonic_num, 1)).astype(np.float32)
    )
    phase_vec = torch.cat([torch.zeros(b, 1, 1, dtype=f0.dtype), phase_rest], dim=1)
    sine_waves = sine_amp * torch.sin(theta_mat + phase_vec)

    uv = (f0 > voiced_threshold).to(f0.dtype)
    noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
    noise_sample = noise_buffer[:, : harmonic_num + 1, :length].expand(b, -1, -1)
    noise = noise_amp * noise_sample

    return sine_waves * uv + noise


def _source_module_deterministic(
    x: torch.Tensor, m_source: nn.Module, noise_buffer: torch.Tensor
) -> torch.Tensor:
    """Reimplements SourceModuleHnNSF.forward()'s harmonic branch (the only one
    HiFTGenerator.decode() keeps — `s, _, _ = self.m_source(s)` discards the rest)
    using `_sine_gen_deterministic` in place of live SineGen randomness."""
    sine_gen = m_source.l_sin_gen
    sine_waves = _sine_gen_deterministic(
        x.transpose(1, 2),
        harmonic_num=sine_gen.harmonic_num,
        sine_amp=sine_gen.sine_amp,
        noise_std=sine_gen.noise_std,
        voiced_threshold=sine_gen.voiced_threshold,
        sampling_rate=sine_gen.sampling_rate,
        noise_buffer=noise_buffer,
    )
    sine_waves = sine_waves.transpose(1, 2)
    return m_source.l_tanh(m_source.l_linear(sine_waves))


class HiFiGANExportWrapper(nn.Module):
    """Traces HiFTGenerator's decode() path with cache_source fixed to empty (the
    default) and ONNX-exportable stand-ins for `_stft`/`_istft` (see module docstring)."""

    def __init__(self, generator: nn.Module):
        super().__init__()
        self.generator = generator
        n_fft = generator.istft_params["n_fft"]
        hop_len = generator.istft_params["hop_len"]
        self.n_fft = n_fft
        self.hop_len = hop_len
        cos_mat, sin_mat = _inverse_dft_matrices(n_fft, generator.stft_window)
        self.register_buffer("_idft_cos", cos_mat)
        self.register_buffer("_idft_sin", sin_mat)
        sine_gen = generator.m_source.l_sin_gen
        rng = np.random.RandomState(_SOURCE_NOISE_SEED)
        noise_buffer = rng.randn(1, sine_gen.harmonic_num + 1, _NOISE_BUFFER_SAMPLES).astype(np.float32)
        self.register_buffer("_noise_buffer", torch.from_numpy(noise_buffer))
        # Overlap-add via conv_transpose1d: an n_fft-channel "identity" kernel scatters
        # each windowed frame back to its hop-offset position, summing overlaps for free.
        # (Standard iSTFTNet-style trick; avoids F.fold/col2im, whose ONNX symbolic
        # requires a static output_size and errors on the traced dynamic-shape value.)
        self.register_buffer("_ola_kernel", torch.eye(n_fft).unsqueeze(1))  # (n_fft, 1, n_fft)

    def _overlap_add(self, frames_ct: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose1d(frames_ct, self._ola_kernel, stride=self.hop_len).squeeze(1)

    def _istft_onnx(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        magnitude = torch.clamp(magnitude, max=1e2)
        real = (magnitude * torch.cos(phase)).transpose(1, 2)  # (B, T', F)
        imag = (magnitude * torch.sin(phase)).transpose(1, 2)  # (B, T', F)
        frames = real @ self._idft_cos + imag @ self._idft_sin  # (B, T', n_fft)
        num_frames = frames.shape[1]

        frames_ct = frames.transpose(1, 2)  # (B, n_fft, T')
        ola = self._overlap_add(frames_ct)

        # Broadcasting against `frames_ct`'s own dynamic time dim, not `.repeat(1, 1,
        # num_frames)`: `repeat()` takes plain Python ints, so it bakes the traced
        # frame count into the graph (see module docstring / ADR-0009 precedent).
        window_sq_col = (self.generator.stft_window**2).view(1, self.n_fft, 1)
        window_sq = window_sq_col * torch.ones_like(frames_ct[:, :1, :])
        envelope = self._overlap_add(window_sq)
        ola = ola / torch.clamp(envelope, min=1e-11)

        start = self.n_fft // 2
        length = self.hop_len * (num_frames - 1)
        return ola[:, start : start + length]

    def forward(self, speech_feat: torch.Tensor) -> torch.Tensor:
        g = self.generator
        f0 = g.f0_predictor(speech_feat)
        s = g.f0_upsamp(f0[:, None]).transpose(1, 2)
        s = _source_module_deterministic(s, g.m_source, self._noise_buffer)
        s = s.transpose(1, 2)

        s_stft_real, s_stft_imag = _stft_onnx(s.squeeze(1), self.n_fft, self.hop_len, g.stft_window)
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        x = g.conv_pre(speech_feat)
        for i in range(g.num_upsamples):
            x = F.leaky_relu(x, g.lrelu_slope)
            x = g.ups[i](x)
            if i == g.num_upsamples - 1:
                x = g.reflection_pad(x)
            si = g.source_downs[i](s_stft)
            si = g.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(g.num_kernels):
                block_out = g.resblocks[i * g.num_kernels + j](x)
                xs = block_out if xs is None else xs + block_out
            x = xs / g.num_kernels

        x = F.leaky_relu(x)
        x = g.conv_post(x)
        n_freq = self.n_fft // 2 + 1
        magnitude = torch.exp(x[:, :n_freq, :])
        phase = torch.sin(x[:, n_freq:, :])
        x = self._istft_onnx(magnitude, phase)
        return torch.clamp(x, -g.audio_limit, g.audio_limit)


def build_wrapper(device: str = "cpu") -> HiFiGANExportWrapper:
    s3gen = load_s3gen(device=device)
    generator = s3gen.mel2wav
    _fuse_weight_norm(generator)  # idempotent: no-op once already fused
    generator.eval()
    return HiFiGANExportWrapper(generator)


def export(out_path: Path, device: str = "cpu") -> Path:
    wrapper = build_wrapper(device=device)
    example = torch.randn(1, MEL_CHANNELS, 50, device=device)
    return export_onnx(
        wrapper,
        (example,),
        out_path,
        input_names=["speech_feat"],
        output_names=["waveform"],
        dynamic_axes={"speech_feat": {2: "frames"}, "waveform": {1: "samples"}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .onnx path (default: models/hifigan.onnx)",
    )
    args = parser.parse_args()
    out_path = args.out or (models_dir() / "hifigan.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export(out_path)
    print(f"Exported HiFiGAN to {out_path}")


if __name__ == "__main__":
    main()
