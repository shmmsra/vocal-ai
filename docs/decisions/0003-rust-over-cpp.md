# ADR-003: Rust for the native runtime (evaluated against C++)

**Date**: 2026-08-16
**Status**: Accepted
**Decider**: Shivam Mishra + AI agent (session 2026-08-16)

---

## Context

Given ADR-002 (export to ONNX, re-drive the loops from native code), the runtime language needs to support:

- Hardware-accelerated execution on macOS (CoreML) *and* Windows/Linux (CUDA) from one codebase.
- A hand-written autoregressive KV-cache decode loop (T3) and flow-matching Euler ODE loop (S3Gen) driving static ONNX graphs.
- Cross-platform, self-contained packaging (no bundled Python/PyTorch — see ADR-002).
- Alignment with the Phase 2 roadmap item: a Tauri desktop frontend wrapping this CLI as an `externalBin` sidecar.

C++'s main apparent advantage is `onnxruntime-genai`, which hands callers a ready-made autoregressive KV-cache decode loop.

## Decision

Implement the native runtime (`vocalai-core`, `vocalai-cli`) in **Rust**, using the `ort` crate (pyke's ONNX Runtime bindings) with a hand-written KV-cache decode loop, rather than C++ with `onnxruntime-genai`.

## Rationale

- **`onnxruntime-genai`'s advantage collapses for this project on inspection**:
  - It has **no CoreML execution provider** (only CPU/CUDA/DirectML/TensorRT/OpenVINO/QNN/WebGPU) — on Apple Silicon it would run CPU-only, defeating the CoreML requirement from ADR-002. `ort`'s `coreml` Cargo feature gives genuine hardware-accelerated CoreML.
  - Its `Generate()` API is coupled to its model builder, which only accepts a fixed roster of standard HF architectures (Llama, Mistral, Phi, Gemma, Qwen, …). T3 is only Llama-*style*, not a recognized architecture, and triggers shape-inference errors in the builder — so the KV-cache loop would have to be hand-rolled regardless of language choice.
- With the KV-cache loop hand-rolled either way, C++ buys nothing this project needs, while Rust wins on:
  - **Cross-platform packaging**: `cargo` + `ort`'s `download-binaries` feature vs. per-platform CMake configuration and manual CUDA/cuDNN redistribution.
  - **Memory safety** for tensor and KV-cache bookkeeping — the decode loop mutates cache tensors every step; Rust's ownership model catches a class of bugs C++ would only catch at runtime (or not at all).
  - **Phase-2 alignment**: the planned Tauri frontend is a Rust/TypeScript stack; a Rust CLI sidecar shares tooling and (eventually) code with the frontend shell.
- **Precedent**: sbv2-api (Style-BERT-VITS2, a comparable TTS engine) already ships production inference on `ort` in Rust.

## Alternatives rejected

- **C++ with `onnxruntime-genai`**: rejected because genai's CoreML gap and fixed-architecture model builder mean the one thing it would have bought us (a free KV-cache loop) doesn't apply to T3 anyway — see Rationale above.
- **C++ with hand-rolled ONNX Runtime C++ API (no genai)**: technically viable, but then Rust wins on every axis in Rationale (packaging, memory safety, Phase-2 alignment) with no offsetting advantage.
- **Python + PyTorch native execution (no ONNX export at all)**: rejected in ADR-002 — doesn't solve the standalone-binary or MPS-memory goals.

## Consequences

**Easier**:
- One codebase drives CoreML, CUDA, and CPU execution via Cargo feature gating (`coreml`, `cuda` features on `vocalai-core`/`vocalai-cli`, wired to `ort`'s matching features — see `crates/vocalai-core/Cargo.toml`, `crates/vocalai-core/src/session.rs`).
- Per-platform release artifacts (plan §2.3) build from the same source tree with a `--features` flag selecting the hardware EP; no separate C++ build system or per-platform CMake config to maintain.
- Future Tauri frontend (Phase 2) can share Rust tooling/CI patterns with this crate.

**Harder**:
- No `onnxruntime-genai`-style batteries-included decode loop — the KV-cache prefill/decode loop, sampling (repetition penalty, top-p, min-p, temperature, CFG duplication), and mel/STFT preprocessing must all be hand-written and parity-checked against the PyTorch reference (Milestones 3-4, `docs/issues.md` VAI-003/VAI-004).
- `ort` is at `2.0.0-rc.13` (API not frozen) — pinned exactly (`=2.0.0-rc.13` in `crates/vocalai-core/Cargo.toml`) to avoid unannounced breakage; upgrading requires deliberate re-verification, not just a version bump.

**New commitments**:
- Hardware EPs must always be listed before CPU in every `ort::SessionBuilder` call, with silent-fallback behavior made explicit in code (hard constraint, `CLAUDE.md` §1; implemented in `crates/vocalai-core/src/session.rs`).
- The CUDA-enabled release artifact must bundle a matching CUDA runtime + cuDNN version (cuDNN 8 vs. 9 mismatches break it) — a packaging commitment for Milestone 7, not a Rust-vs-C++ concern, but one this ADR's EP choice creates.
