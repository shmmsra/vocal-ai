"""Export S3Gen's flow *encoder* (token -> mu) to ONNX, as a set of fixed-length
bucket graphs.

Milestone 3 (`export_s3gen.py`) only exported `s3gen.flow.decoder.estimator` -- the
CFM diffusion network that maps `(x, mu, spks, cond) -> dxdt`. It never exported the
piece that *produces* `mu` from speech tokens: `flow.input_embedding` (token ->
512-dim embedding) -> `flow.encoder` (`UpsampleConformerEncoder`, upsamples the
25Hz token rate to the 50Hz mel rate) -> `flow.encoder_proj` (512 -> 80). Without
this, `mu` has no real source; `parity_check.py::check_s3gen`'s `mu`/`spks`/`cond`
were always random synthetic tensors (see its docstring). See
docs/decisions/0009-s3gen-flow-encoder-and-campplus-export.md for the full gap
writeup.

Usage:
    python export_s3gen_flow_encoder.py [--out-dir models/]

**Why bucketed, fixed-length graphs, not one dynamic-length graph**: `flow.encoder`
uses `EspnetRelPositionalEncoding`/relative-position attention
(chatterbox/models/s3gen/transformer/{embedding,attention}.py). Its
`position_encoding(size=x.size(1), ...)` takes `size` as a plain Python `int`, so
`torch.onnx.export`'s tracer bakes the *tracing example's* sequence length into the
graph as a constant. A single dynamic-`tokens`-axis export was tried and confirmed
broken: at a token count other than the tracing example's, the relative-position
term (`matrix_bd`) keeps the traced length while the rest of the graph scales
dynamically, and ONNX Runtime raises a broadcast-shape error. See
docs/decisions/0009-s3gen-flow-encoder-and-campplus-export.md for the full
diagnosis and why bucketing (not a hand-rolled reimplementation) was chosen.

Each bucket is a fully static graph (fixed batch=1, fixed token count = the
bucket size) -- the physical tensor length always matches what was traced, so the
relative-position math is correct by construction. `token_len` (the *true*,
unpadded length, <= the bucket size) stays a genuine runtime input: it only drives
`make_pad_mask`-based masking (a value computation, not a shape computation), which
traces correctly regardless of tracing length. Rust picks the smallest bucket >=
the real token count and right-pads `token` up to it (any padding token id works --
masked positions are zeroed by `make_pad_mask` before the encoder ever sees them).

Bucket schedule: speech-token count = `speech_cond_prompt_len` (150, fixed) +
T3-generated tokens (0..`max_new_tokens`=1000), so real inputs range from ~150 to
~1150 tokens. `TOKEN_BUCKETS` covers that with headroom; texts that generate more
than the largest bucket's worth of tokens are Milestone 6 scope to handle (e.g. by
truncating or adding a larger bucket), not this export step's problem.

Input: token ids `(1, bucket)` int64 -- the *concatenated* `prompt_token` +
T3-generated tokens (that concatenation happens host-side in Rust, matching
`CausalMaskedDiffWithXvec.inference()`'s own `torch.concat` one level above
`input_embedding`), right-padded to `bucket`; `token_len (1,)` int64 -- the true
length.
Output: `mu (1, 80, 2*bucket)`, `mask (1, 1, 2*bucket)` (`token_mel_ratio=2`,
confirmed empirically -- no pre-lookahead padding leaks in on the `finalize=True`,
i.e. non-streaming, path this wrapper always takes).

See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 6) and
docs/decisions/0009-s3gen-flow-encoder-and-campplus-export.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from _common import export_onnx, load_s3gen, models_dir

TOKEN_BUCKETS: tuple[int, ...] = (200, 400, 600, 800, 1000, 1200)


class S3GenFlowEncoderExport(nn.Module):
    """Reproduces `CausalMaskedDiffWithXvec.inference()`'s token -> mu path
    (chatterbox/models/s3gen/flow.py), against `flow`'s real
    `input_embedding`/`encoder`/`encoder_proj` submodules. The out-of-range-token
    warning log (`if (token >= vocab_size).any(): logger.error(...)`) is dropped --
    it never affects the output tensor, only a log line -- and isn't something a
    static ONNX graph can express as data-dependent control flow anyway.
    """

    def __init__(self, flow: nn.Module):
        super().__init__()
        self.input_embedding = flow.input_embedding
        self.encoder = flow.encoder
        self.encoder_proj = flow.encoder_proj

    def forward(
        self, token: torch.Tensor, token_len: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from chatterbox.models.s3gen.utils.mask import make_pad_mask

        # `max_len` must be passed explicitly (the bucket's fixed physical size),
        # not left to make_pad_mask's default `lengths.max()` -- production usage
        # never pads (the physical tensor length always equals the true length
        # there), so this distinction only matters for this bucketed export.
        mask = (~make_pad_mask(token_len, max_len=token.size(1))).unsqueeze(-1).float()
        embedded = self.input_embedding(token.long()) * mask

        h, h_masks = self.encoder(embedded, token_len)
        h_lengths = h_masks.sum(dim=-1).squeeze(dim=-1)
        h = self.encoder_proj(h)

        out_mask = (~make_pad_mask(h_lengths, max_len=h.size(1))).unsqueeze(1).float()
        return h.transpose(1, 2), out_mask


def build_module(device: str = "cpu") -> S3GenFlowEncoderExport:
    s3gen = load_s3gen(device=device)
    return S3GenFlowEncoderExport(s3gen.flow).eval()


def example_inputs(
    bucket: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    token = torch.randint(0, 6561, (1, bucket), device=device)
    token_len = torch.tensor([bucket], device=device)
    return token, token_len


def bucket_path(out_dir: Path, bucket: int) -> Path:
    return out_dir / f"s3gen_flow_encoder_{bucket}.onnx"


def export_bucket(bucket: int, out_path: Path, device: str = "cpu") -> Path:
    module = build_module(device=device)
    token, token_len = example_inputs(bucket, device=device)
    return export_onnx(
        module,
        (token, token_len),
        out_path,
        input_names=["token", "token_len"],
        output_names=["mu", "mask"],
    )


def export(out_dir: Path, device: str = "cpu") -> list[Path]:
    return [
        export_bucket(bucket, bucket_path(out_dir, bucket), device=device)
        for bucket in TOKEN_BUCKETS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: models/)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or models_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in export(out_dir):
        print(f"Exported S3Gen flow encoder bucket to {path}")


if __name__ == "__main__":
    main()
