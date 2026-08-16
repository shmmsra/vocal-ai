# ADR-002: Export to ONNX and re-drive the loops from native Rust (no Python-wrapper interim)

**Date**: 2026-08-16
**Status**: Accepted
**Decider**: Shivam Mishra + AI agent (session 2026-08-16)

---

## Context

`vocal-ai` wraps the Chatterbox TTS model (`resemble-ai/chatterbox`, Python/PyTorch). Two product goals force a decision on how the model is shipped:

1. **A true standalone binary.** The reference implementation needs a bundled Python interpreter, PyTorch, and per-platform accelerator builds (multi-GB), plus multi-GB weight downloads from the HuggingFace Hub at runtime.
2. **Predictable memory behavior on Apple Silicon.** PyTorch's MPS caching allocator overcommits into system RAM and doesn't release it; a single generation on an M3 Max (48 GB unified memory) drove macOS swap from ~5 GB to ~30 GB (confirmed live via `sysctl vm.swapusage`).

Both goals require running natively (CoreML on macOS, CUDA on Windows/Linux, CPU fallback everywhere) from one codebase, with no bundled Python runtime.

A source read of `resemble-ai/chatterbox/src/chatterbox/` showed the model is unusually favorable for this: every iterative part (T3's autoregressive token loop, S3Gen's flow-matching ODE solver) is **already** orchestrated in plain Python around *static* per-step network forwards (`inference()` does an explicit prefill + per-token loop passing `past_key_values`; `solve_euler` is a fixed-count `for` loop calling one static estimator forward per step). So exporting to ONNX means exporting a handful of static graphs and re-driving the loops from host code — not fighting exotic in-graph control flow.

## Decision

Export the model's static sub-graphs (T3 backbone, S3Gen flow estimator, HiFiGAN vocoder, voice encoder, S3 tokenizer, PerthNet watermarker) to ONNX, and re-implement the autoregressive/ODE driver loops in native Rust against `ort` (ONNX Runtime bindings), rather than shipping a Python-wrapper interim or an in-graph-control-flow export.

## Rationale

- **The loops export cleanly because they're already host-orchestrated in the reference code** — this is a rare case where a full ONNX+native-runtime rewrite is *lower* risk than it usually would be, since there's no exotic control flow to trace or reimplement from scratch. See `docs/phase1-onnx-rust-cli-plan.md` §4 for the per-component export-difficulty assessment (HiFiGAN easiest → T3 hardest).
- **Native execution is a hard requirement, not an optimization**: CoreML/CUDA support from one codebase, plus a standalone binary, both require an ONNX+native runtime — a Python-wrapper interim would still need a bundled interpreter and wouldn't be a step toward the actual goal.
- **Solves the memory problem directly**: ONNX Runtime's CoreML EP does not carry PyTorch/MPS's caching-allocator overcommit behavior; native execution gives control over allocation that a wrapped PyTorch process doesn't.

## Alternatives rejected

- **Ship a bundled Python + PyTorch interpreter (Python-wrapper interim)**: solves neither the standalone-binary goal nor the MPS memory problem; only defers the real rewrite while adding a multi-GB runtime dependency.
- **Keep PyTorch, try to fix the MPS allocator behavior directly**: the caching allocator's overcommit is PyTorch/MPS-internal behavior, not something this project can patch from the outside; would still require bundling PyTorch either way.
- **Export the whole model as one graph including the autoregressive/ODE loops (in-graph control flow)**: would fight ONNX's limited native support for dynamic per-step KV-cache growth and variable-length loops for no benefit, given the loops are already cleanly host-orchestrated in the reference implementation.

## Consequences

**Easier**:
- A single Rust codebase drives CoreML, CUDA, and CPU execution from the same decode/ODE loop code, gated by EP selection only (see ADR-003 and `crates/vocalai-core/src/session.rs`).
- The end-user artifact is a real standalone binary: no Python, no PyTorch, no first-run model download (see ADR-003 §"Consequences" and plan §2.3).
- Per-component export difficulty ascends cleanly (HiFiGAN → voice encoder → S3 tokenizer → S3Gen → T3 → PerthNet), so the export + parity toolchain gets proven on cheap components before the highest-risk one (Milestone sequencing in plan §7).

**Harder**:
- Each exported component needs a numerical parity check against the PyTorch reference before it can ship (hard constraint, `CLAUDE.md` §1) — this is real engineering work per component, not a one-time cost.
- The T3 KV-cache decode loop (naming/layout of `past_key_values.*`/`present.*`, dynamic sequence-length axes) must be hand-rolled in Rust with no `onnxruntime-genai`-equivalent to lean on (see ADR-003).
- Any change to the reference model's sampling/CFG/loop logic must be manually re-ported to the Rust implementation; there's no shared code path with upstream `chatterbox`.

**New commitments**:
- `export/parity_check.py` becomes a required gate before any exported component ships into `vocalai-core` (hard constraint, `CLAUDE.md` §1).
- Model licensing must permit redistributing weights inside a bundled release artifact (open item, plan §9) — this is a distribution decision, not an export-mechanics one, but it's a prerequisite for this ADR's approach to ship.
