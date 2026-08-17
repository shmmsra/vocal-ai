"""Export T3 (the Llama-style text->speech-token backbone) as a decoder-with-past ONNX
pair, plus raw embedding-table weights for the per-step token embedding.

Usage:
    python export_t3.py [--out-dir models/]

Produces four files in `--out-dir`:
  - t3_cond_prefill.onnx  — cond/text embedding builder (see `T3CondPrefillExport`).
      Inputs:  speaker_emb (1, 256), cond_prompt_speech_tokens (1, speech_cond_prompt_len),
               emotion_adv (1, 1, 1), text_tokens (2, len_text) int64,
               cfg_uncond_mask (2, 1, 1)
      Output:  inputs_embeds (2, len_cond + len_text + 2, dim) — the exact tensor
               `T3.inference()` feeds to its first decoder forward, including the
               reference's double-BOS-embedding quirk (see module docstring below).
  - t3_decoder.onnx       — hand-rolled Llama decoder-with-past + speech_head (see
      `T3DecoderExport`). Inputs: inputs_embeds (batch, seq, dim), past_kv
      (num_layers, 2, batch, num_heads, past_seq, head_dim). Outputs: logits
      (batch, seq, speech_vocab), present_kv (num_layers, 2, batch, num_heads,
      past_seq + seq, head_dim). Driven once per prefill (empty past_kv) and once per
      decode step (seq=1) by the KV-cache loop in `crates/vocalai-core/src/t3.rs`.
  - t3_speech_emb.npy      — raw `T3.speech_emb.weight` (speech_vocab, dim), for the
      Rust-side per-step new-token embedding lookup (no ONNX call needed per token —
      see docs/decisions/0005-t3-hand-rolled-decoder-export.md).
  - t3_speech_pos_emb.npy  — raw `T3.speech_pos_emb.emb.weight` (max_mel_seq_len, dim),
      same rationale.

Why two ONNX graphs, not a trace of `T3.inference()`/`T3HuggingfaceBackend` directly:
`transformers`' `LlamaModel.forward` (see `export/requirements.txt` pin) is built entirely
around a `Cache` object (`cache.update(...)`) and `masking_utils.create_causal_mask`, neither
of which traces through `torch.onnx.export`'s legacy tracer as a static tensor-in/tensor-out
graph. `T3DecoderExport` below re-implements the same math (RMSNorm, RoPE with the model's
own precomputed llama3-scaled `inv_freq`, SwiGLU MLP, no GQA since
`num_key_value_heads == num_attention_heads` for this config) directly against `T3.tfmr`'s
real submodules — see docs/decisions/0005-t3-hand-rolled-decoder-export.md for the full
rationale.

Why `t3_cond_prefill` reproduces a double-BOS-embedding quirk: `T3.inference()`
(chatterbox/models/t3/t3.py) builds `embeds` via `prepare_input_embeds()` — which already
appends one speech-BOS embedding — and *then* concatenates a second, independently-computed
BOS embedding at the same position index before the first decoder forward. The two
computations are mathematically identical (same token id, same position-0 positional
embedding), so this wrapper computes it once and appends it twice rather than literally
duplicating the (redundant) second computation.

See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 4).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from _common import export_onnx, load_t3, models_dir

DIM = 1024  # T3Config.n_channels for llama_config_name="Llama_520M"
SPEAKER_EMBED_SIZE = 256
SPEECH_COND_PROMPT_LEN = 150  # T3Config.speech_cond_prompt_len
NUM_LAYERS = 30
NUM_HEADS = 16
HEAD_DIM = 64
EXAMPLE_LEN_TEXT = 12


def _rms_norm(x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + norm.variance_epsilon)
    return norm.weight * x.to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # (B, 1, S, head_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class _ExportDecoderLayer(nn.Module):
    """Re-implements one `LlamaDecoderLayer` forward against plain tensors (no `Cache`
    object, no `masking_utils`), reusing the real layer's `nn.Linear`/norm submodules
    directly (no weight copying). See module docstring / ADR-0005."""

    def __init__(self, ref_layer: nn.Module, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.input_layernorm = ref_layer.input_layernorm
        self.post_attention_layernorm = ref_layer.post_attention_layernorm
        self.q_proj = ref_layer.self_attn.q_proj
        self.k_proj = ref_layer.self_attn.k_proj
        self.v_proj = ref_layer.self_attn.v_proj
        self.o_proj = ref_layer.self_attn.o_proj
        self.gate_proj = ref_layer.mlp.gate_proj
        self.up_proj = ref_layer.mlp.up_proj
        self.down_proj = ref_layer.mlp.down_proj

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = hidden.shape
        residual = hidden
        x = _rms_norm(hidden, self.input_layernorm)

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rotary(q, k, cos, sin)
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).reshape(batch, seq_len, -1)
        attn_out = self.o_proj(attn_out)
        hidden = residual + attn_out

        residual = hidden
        x = _rms_norm(hidden, self.post_attention_layernorm)
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        hidden = residual + x
        return hidden, k, v


class T3DecoderExport(nn.Module):
    """Hand-rolled Llama decoder-with-past + `speech_head`, reusing `t3`'s real
    submodules and its precomputed llama3-scaled RoPE `inv_freq`/`attention_scaling`
    buffers. See module docstring / ADR-0005."""

    def __init__(self, t3: nn.Module):
        super().__init__()
        cfg = t3.cfg
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.layers = nn.ModuleList(
            _ExportDecoderLayer(layer, self.num_heads, self.head_dim) for layer in t3.tfmr.layers
        )
        self.final_norm = t3.tfmr.norm
        self.speech_head = t3.speech_head
        self.register_buffer("inv_freq", t3.tfmr.rotary_emb.inv_freq.clone(), persistent=False)
        self.attention_scaling = float(t3.tfmr.rotary_emb.attention_scaling)

    def forward(
        self, inputs_embeds: torch.Tensor, past_kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = inputs_embeds.shape
        past_len = past_kv.shape[4]
        total_len = past_len + seq_len

        position_ids = torch.arange(past_len, total_len, device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(batch, -1).float()
        freqs = position_ids[:, :, None] * self.inv_freq[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        q_pos = torch.arange(past_len, total_len, device=inputs_embeds.device)
        k_pos = torch.arange(total_len, device=inputs_embeds.device)
        allowed = k_pos[None, :] <= q_pos[:, None]  # (seq_len, total_len)
        causal_mask = torch.zeros(seq_len, total_len, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        causal_mask = causal_mask.masked_fill(~allowed, float("-inf"))[None, None, :, :]

        hidden = inputs_embeds
        present = []
        for i, layer in enumerate(self.layers):
            past_k, past_v = past_kv[i, 0], past_kv[i, 1]
            hidden, present_k, present_v = layer(hidden, cos, sin, causal_mask, past_k, past_v)
            present.append(torch.stack([present_k, present_v], dim=0))
        present_kv = torch.stack(present, dim=0)

        hidden = _rms_norm(hidden, self.final_norm)
        logits = self.speech_head(hidden)
        return logits, present_kv


class T3CondPrefillExport(nn.Module):
    """Reproduces `T3.prepare_input_embeds()` + `T3.inference()`'s initial
    `inputs_embeds` construction (including the double-BOS quirk), against `t3`'s real
    `cond_enc`/`text_emb`/`speech_emb`/`*_pos_emb` submodules. See module docstring."""

    def __init__(self, t3: nn.Module):
        super().__init__()
        self.spkr_enc = t3.cond_enc.spkr_enc
        self.emotion_adv_fc = t3.cond_enc.emotion_adv_fc
        self.perceiver = t3.cond_enc.perceiver
        self.text_emb = t3.text_emb
        self.speech_emb = t3.speech_emb
        self.text_pos_emb = t3.text_pos_emb
        self.speech_pos_emb = t3.speech_pos_emb
        self.start_speech_token = t3.hp.start_speech_token

    def forward(
        self,
        speaker_emb: torch.Tensor,
        cond_prompt_speech_tokens: torch.Tensor,
        emotion_adv: torch.Tensor,
        text_tokens: torch.Tensor,
        cfg_uncond_mask: torch.Tensor,
    ) -> torch.Tensor:
        # prepare_conditioning() + T3CondEnc.forward()
        cond_prompt_speech_emb = self.speech_emb(cond_prompt_speech_tokens)
        cond_prompt_speech_emb = cond_prompt_speech_emb + self.speech_pos_emb(cond_prompt_speech_tokens)
        cond_prompt_speech_emb = self.perceiver(cond_prompt_speech_emb)

        cond_spkr = self.spkr_enc(speaker_emb.view(-1, SPEAKER_EMBED_SIZE))[:, None]  # (1, 1, dim)
        cond_emotion_adv = self.emotion_adv_fc(emotion_adv.view(-1, 1, 1))  # (1, 1, dim)
        cond_embeds = torch.cat((cond_spkr, cond_prompt_speech_emb, cond_emotion_adv), dim=1)

        # prepare_input_embeds()
        text_emb = self.text_emb(text_tokens) * cfg_uncond_mask
        text_emb = text_emb + self.text_pos_emb(text_tokens)

        init_speech_tokens = torch.full_like(text_tokens[:, :1], self.start_speech_token)
        speech_emb0 = self.speech_emb(init_speech_tokens) + self.speech_pos_emb(init_speech_tokens)

        if cond_embeds.size(0) != text_emb.size(0):
            cond_embeds = cond_embeds.expand(text_emb.size(0), -1, -1)
        embeds = torch.cat((cond_embeds, text_emb, speech_emb0), dim=1)

        # T3.inference()'s separate (numerically identical) second BOS embed.
        inputs_embeds = torch.cat((embeds, speech_emb0), dim=1)
        return inputs_embeds


def build_decoder(device: str = "cpu") -> tuple[nn.Module, nn.Module]:
    t3 = load_t3(device=device)
    return T3CondPrefillExport(t3).eval(), T3DecoderExport(t3).eval()


def _cond_prefill_example(device: str = "cpu") -> tuple[torch.Tensor, ...]:
    speaker_emb = torch.randn(1, SPEAKER_EMBED_SIZE, device=device)
    cond_prompt_speech_tokens = torch.randint(0, 6561, (1, SPEECH_COND_PROMPT_LEN), device=device)
    emotion_adv = torch.full((1, 1, 1), 0.5, device=device)
    text_tokens = torch.randint(1, 254, (2, EXAMPLE_LEN_TEXT), device=device)
    cfg_uncond_mask = torch.tensor([[[1.0]], [[0.0]]], device=device)
    return speaker_emb, cond_prompt_speech_tokens, emotion_adv, text_tokens, cfg_uncond_mask


def _decoder_example(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = 4
    inputs_embeds = torch.randn(2, seq_len, DIM, device=device)
    past_kv = torch.zeros(NUM_LAYERS, 2, 2, NUM_HEADS, 0, HEAD_DIM, device=device)
    return inputs_embeds, past_kv


def export(out_dir: Path, device: str = "cpu") -> tuple[Path, Path, Path, Path]:
    cond_prefill, decoder = build_decoder(device=device)

    cond_prefill_path = export_onnx(
        cond_prefill,
        _cond_prefill_example(device=device),
        out_dir / "t3_cond_prefill.onnx",
        input_names=[
            "speaker_emb",
            "cond_prompt_speech_tokens",
            "emotion_adv",
            "text_tokens",
            "cfg_uncond_mask",
        ],
        output_names=["inputs_embeds"],
        dynamic_axes={
            "cond_prompt_speech_tokens": {1: "cond_prompt_len"},
            "text_tokens": {1: "len_text"},
            "inputs_embeds": {1: "len_prefill"},
        },
    )

    decoder_path = export_onnx(
        decoder,
        _decoder_example(device=device),
        out_dir / "t3_decoder.onnx",
        input_names=["inputs_embeds", "past_kv"],
        output_names=["logits", "present_kv"],
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq_len"},
            "past_kv": {2: "batch", 4: "past_len"},
            "logits": {0: "batch", 1: "seq_len"},
            "present_kv": {2: "batch", 4: "total_len"},
        },
    )

    t3 = load_t3(device=device)
    speech_emb_path = out_dir / "t3_speech_emb.npy"
    speech_pos_emb_path = out_dir / "t3_speech_pos_emb.npy"
    np.save(speech_emb_path, t3.speech_emb.weight.detach().cpu().numpy())
    np.save(speech_pos_emb_path, t3.speech_pos_emb.emb.weight.detach().cpu().numpy())

    return cond_prefill_path, decoder_path, speech_emb_path, speech_pos_emb_path


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
    paths = export(out_dir)
    for path in paths:
        print(f"Exported {path}")


if __name__ == "__main__":
    main()
