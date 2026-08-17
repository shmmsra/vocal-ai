# ADR-0005: T3's decoder-with-past is exported from a hand-rolled Llama re-implementation, not by tracing `transformers.LlamaModel`

**Date**: 2026-08-17
**Status**: Accepted
**Decider**: Shivam Mishra + AI agent (session 2026-08-17)

## Context

Milestone 4 (`docs/issues.md` VAI-004) exports T3's autoregressive backbone — a `transformers`
`LlamaModel` wrapped by `chatterbox`'s `T3HuggingfaceBackend` — as a decoder-with-past ONNX graph
(`docs/phase1-onnx-rust-cli-plan.md` §4/§7 milestone 4), so the KV-cache token loop can be
re-driven from `crates/vocalai-core/src/t3.rs`.

The pinned `transformers==5.2.0` (`export/requirements.txt`) implements `LlamaModel.forward`
entirely around a `Cache`/`DynamicCache` object: `past_key_values` is read and written through
`cache.update(...)`, the causal mask comes from `masking_utils.create_causal_mask(...)` (branches
on cache/mask type), and the forward is wrapped in generic output-capturing decorators
(`@capture_outputs`, `@merge_with_config_defaults`). None of this is a static tensor-in/tensor-out
graph — `torch.onnx.export`'s tracer (the legacy exporter this repo already standardized on in
`export/_common.py::export_onnx`, opset 18; see ADR-0002) cannot turn a `Cache` object into
explicit `past_key_values.N.key/value` / `present.N.key/value` ONNX inputs/outputs without either a
newer dynamo-based exporter (a bigger toolchain change, unproven for this repo) or fighting the
`Cache` abstraction with monkeypatches on every release.

Every other exported component so far (HiFiGAN, voice encoder, S3 tokenizer, S3Gen estimator) is a
plain `nn.Module` with no such framework-owned control flow, so this problem hasn't come up before.

## Decision

`export/export_t3.py` defines a **from-scratch Llama decoder-with-past module**
(`T3DecoderExport` + a per-layer `_ExportDecoderLayer`) that:
- Reuses the *actual loaded* `nn.Linear`/`nn.Parameter` submodules from `T3.tfmr`'s real
  `LlamaDecoderLayer`s (`q_proj`/`k_proj`/`v_proj`/`o_proj`, `gate_proj`/`up_proj`/`down_proj`,
  `input_layernorm`/`post_attention_layernorm`, final `norm`) and `T3.speech_head` directly as
  attributes — no weight copying, no state-dict re-mapping, so weight values can't drift from the
  loaded checkpoint.
- Reuses the *actual* precomputed `T3.tfmr.rotary_emb.inv_freq` / `.attention_scaling` buffers
  (already correctly llama3-scaled by `transformers` at `T3` load time) rather than
  re-deriving the llama3 RoPE scaling formula by hand — this is the one piece of "T3-internal
  logic" the wrapper borrows instead of reimplementing, chosen specifically to avoid a second,
  error-prone implementation of that formula.
- Takes `past_key_values` as one plain stacked tensor, not a `Cache` object or 60 separate names
  (see "KV-cache tensor layout" below).
- Computes the causal mask and RoPE `position_ids` from tensor shapes inside `forward`, with no
  `Cache`/`masking_utils` dependency — ordinary ops that `torch.onnx.export` traces as dynamic
  `Shape`/`Range`/`Where` ops given the declared `dynamic_axes`.

Everything about the underlying math (RMSNorm formula, SwiGLU MLP, `rotate_half`/RoPE application,
`num_key_value_heads == num_attention_heads` so no GQA `repeat_kv`) is copied verbatim from
`transformers/models/llama/modeling_llama.py` (5.2.0) — only the `Cache`/mask/output-capturing
*plumbing* is replaced.

### KV-cache tensor layout

`past_kv`/`present_kv` are single stacked tensors of shape
`(num_layers=30, k_or_v=2, batch, num_heads=16, seq, head_dim=64)`, not 60 individually-named
`past_key_values.N.key`/`.value` tensors (the naming convention `optimum`/HF's own ONNX exporters
use). One tensor name in, one out, keeps the Rust session-calling code (`t3.rs`) and the ONNX
graph signature simple; there is no external consumer (HF's `generate()`, `onnxruntime-genai`)
that requires the per-layer-named convention, since the decode loop is hand-rolled in Rust anyway.

## Rationale

- Tracing is only reliable when the traced function is a plain sequence of tensor ops with no
  object-oriented cache/mask abstractions in the middle — a hand-rolled forward guarantees that by
  construction, and is a well-established pattern for exporting HF decoder models through the
  legacy `torch.onnx.export` tracer (the alternative — monkeypatching `Cache`/`create_causal_mask`
  internals to make the real `LlamaModel.forward` traceable — is *more* fragile: it depends on
  private `transformers` internals that change across minor versions, whereas re-implementing the
  public, stable Llama math does not).
- Reusing the live submodules/buffers instead of copying weights means there is exactly one copy
  of every learned parameter in memory during export, and no state-dict key-name mapping to get
  wrong or to silently go stale if `chatterbox` renames a submodule.
- A single stacked KV-cache tensor is one name to get right in both the ONNX I/O signature and the
  `ort` Rust call site, instead of 60 (30 layers × key/value) — less surface area for an
  off-by-one/layer-ordering bug, at the cost of one `ndarray` stack/slice per session call in Rust.

## Alternatives rejected

- **Trace `T3HuggingfaceBackend.forward` / `LlamaModel.forward` directly, monkeypatching
  `Cache.update`/`create_causal_mask` to emit plain tensor ops during tracing**: technically
  possible but ties the export script to `transformers`' private internals (`cache_utils`,
  `masking_utils`) at a specific minor version, which is a worse maintenance bet than
  re-implementing the small, stable, public Llama math ourselves.
- **Per-layer-named `past_key_values.N.key`/`.N.value` I/O (the `optimum`/HF ONNX convention)**:
  matches what external tools expect, but there is no external tool in this pipeline's path (the
  decode loop is hand-written in `t3.rs`, not `onnxruntime-genai`) — 60 names bought us nothing and
  cost more Rust-side bookkeeping.
- **Wait for a dynamo-based (`torch.export`) exporter path instead of the legacy tracer**: would
  fix the `Cache`-tracing problem at the tool level, but is an unproven toolchain change for this
  repo (every export so far uses the legacy `torch.onnx.export`, ADR-0002) and a bigger, riskier
  lift than hand-rolling one decoder module.

## Consequences

**Easier**:
- The exported `t3_decoder.onnx` graph has a small, stable, hand-controlled I/O surface
  (`inputs_embeds`, `past_kv` → `logits`, `present_kv`) independent of whatever internal
  abstraction `transformers` uses release-to-release.
- Rust-side KV-cache bookkeeping (`t3.rs`) is one tensor slice/concat per layer, not 60 named
  tensors to keep straight.

**Harder**:
- The hand-rolled decoder is a second implementation of Llama's forward pass that must be kept in
  sync by hand if `chatterbox` ever changes `T3`'s backbone config (e.g. a different
  `llama_config_name`, or GQA with `num_key_value_heads < num_attention_heads`, which would require
  adding the `repeat_kv` step this version omits).
- `export/parity_check.py::check_t3` is the only mechanical guard against this implementation
  drifting from the real `LlamaModel` — there is no type-level guarantee the two match.

**New commitments**:
- If `chatterbox` upgrades `T3`'s backbone (turbo/multilingual variants use different configs —
  out of scope per plan §2.4, but if that ever changes), `_ExportDecoderLayer` needs revisiting for
  GQA support and any RoPE-variant differences.
