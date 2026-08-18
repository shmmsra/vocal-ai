# vocal-ai — Phase 1 Plan: Standalone ONNX + Rust TTS CLI

> **Status:** Approved plan, not yet implemented. This document is self-contained so
> a fresh session can begin implementation from it with no prior context.

---

## 1. Context & Motivation

`vocal-ai` is a fully self-contained, cross-platform command-line TTS tool built
around the **Chatterbox TTS** model. The reference implementation
([`resemble-ai/chatterbox`](https://github.com/resemble-ai/chatterbox),
Python/PyTorch) works but blocks two product goals:

1. **No true standalone binary.** It needs a bundled Python interpreter + PyTorch +
   per-platform accelerator builds (multi-GB), and downloads multi-GB weights from
   the HuggingFace Hub at runtime.
2. **Memory/allocator behavior on Apple Silicon.** PyTorch's MPS caching allocator
   overcommits into system RAM and doesn't release it back. On an M3 Max (48 GB
   unified memory), a single generation drove macOS swap from ~5 GB to ~30 GB
   (confirmed live via `sysctl vm.swapusage`). Because unified memory means GPU
   memory *is* system RAM, this manifests as a large transient disk-space drop
   (swapfiles) that frees after the run. This is not a download; it is swap pressure.

**Goal of Phase 1:**
```
vocalai --text "hello world" [--voice ref.wav] --out out.wav
```
→ produces a 24 kHz mono WAV, running natively on macOS (CoreML), Windows/Linux
(CUDA), and any platform (CPU fallback), with a smaller footprint and controlled
memory.

### Broader roadmap (for context; Phase 1 is only the CLI)
- **Phase 1 — CLI core** (this doc): ONNX + native Rust inference. Independently
  useful; the Claude skill and the frontend both wrap it.
- **Phase 2 — Frontend**: Tauri app using the CLI as an `externalBin` sidecar.
- **Phase 3 — Claude skill**: thin wrapper shelling the CLI (near-free once the CLI
  contract is stable).
- **Backlog**: turbo/multilingual model variants; further ONNX submodule optimization
  only if profiling shows a real bottleneck.

---

## 2. Key Decisions (with rationale)

### 2.1 Approach: go straight to ONNX + native Rust (no Python-wrapper interim)
Driven by a **hard requirement to support CUDA (Windows+Linux) and MPS/CoreML (macOS)
from one codebase**, plus the standalone-binary goal. A source read of the model
confirmed this is unusually favorable: every iterative part (T3's autoregressive token
loop, S3Gen's flow-matching ODE solver) is **already** orchestrated in plain Python
around *static* per-step network forwards. So ONNX export means exporting a handful of
static graphs and re-driving the loops from host code — not fighting exotic in-graph
control flow.

### 2.2 Language: Rust (evaluated against C++)
C++'s only real edge would be **onnxruntime-genai** handing us the autoregressive
KV-cache loop for free. That collapses for this project:
- **genai has no CoreML execution provider** (supports CPU/CUDA/DirectML/TensorRT/
  OpenVINO/QNN/WebGPU only); on Apple Silicon it runs CPU-only. The `ort` Rust crate's
  `coreml` feature *does* give hardware-accelerated CoreML.
- **genai's `Generate()` is coupled to its model builder**, which only accepts a fixed
  roster of standard HF architectures (Llama, Mistral, Phi, Gemma, Qwen…). T3 is only
  Llama-*style*, not a recognized arch → shape-inference errors. We'd hand-roll the
  KV-cache loop regardless.

So C++ buys nothing we need. **Rust wins** on: cross-platform packaging (cargo +
`ort`'s `download-binaries` vs per-platform cmake + manual CUDA/cuDNN redistribution),
memory safety for tensor/KV-cache bookkeeping, and Phase-2 Tauri alignment. Precedent:
**sbv2-api** (Style-BERT-VITS2 TTS engine) already ships on `ort`.

### 2.3 Distribution: fully self-contained bundles
- **Model weights: bundled into the release artifact.** Fully offline, zero first-run
  friction. Each platform artifact is multi-GB; weight updates ship as a new release.
- **CUDA/cuDNN: bundled alongside the CUDA binary.** The Windows/Linux GPU artifact
  ships the required CUDA runtime + cuDNN libs so it works out-of-box.
- **macOS/CPU artifacts stay lean** (CoreML needs no extra runtime).

Resulting artifacts:
| Artifact | EP | Notes |
|---|---|---|
| `vocalai-macos` | CoreML → CPU | lean; universal2 or arch-specific |
| `vocalai-windows-cuda` | CUDA → CPU | bundles CUDA/cuDNN libs (heavy) |
| `vocalai-linux-cuda` | CUDA → CPU | bundles CUDA/cuDNN libs (heavy) |
| `vocalai-{win,linux}-cpu` | CPU | lean fallback |

### 2.4 Scope: base model only in Phase 1
Target `ResembleAI/chatterbox` (base). Turbo (`chatterbox-turbo`/`-nano`, meanflow
`inference_turbo`) and multilingual (`mtl`, requires `language_id`) are follow-ups
with different weights and code paths.

---

## 3. Reference Model Facts

From a source read of `resemble-ai/chatterbox/src/chatterbox/`.

- **Output**: mono waveform, **24000 Hz** (`S3GEN_SR`), Perth-watermarked. Reference
  voice wavs are resampled to **16000 Hz** (`S3_SR`) internally.
- **Model artifacts** (base, loaded from a directory via `ChatterboxTTS.from_local`):
  - `ve.safetensors` — voice encoder
  - `t3_cfg.safetensors` — T3 text→speech-token backbone
  - `s3gen.safetensors` — S3Gen (flow-matching + HiFiGAN vocoder)
  - `tokenizer.json` — text tokenizer
  - `conds.pt` — optional built-in default voice conditioning
- **`generate()` surface → CLI flags** (defaults from reference):

  | Python param | Default | CLI flag |
  |---|---|---|
  | `text` (required) | — | positional or `--text` |
  | `audio_prompt_path` | `None` | `--voice <ref.wav>` (zero-shot cloning) |
  | `exaggeration` | `0.5` | `--exaggeration` |
  | `cfg_weight` | `0.5` | `--cfg-weight` |
  | `temperature` | `0.8` | `--temperature` |
  | `repetition_penalty` | `1.2` | `--repetition-penalty` |
  | `min_p` | `0.05` | `--min-p` |
  | `top_p` | `1.0` | `--top-p` |
  | `max_new_tokens` | `1000` (hardcoded) | `--max-new-tokens` (default 1000) |

  Plus: `--out <path.wav>` (output), `--device`/EP selection is automatic with fallback.

- **User I/O**: input is a text string + optional reference `.wav` for zero-shot voice
  cloning; output is a mono 24 kHz WAV (Perth-watermarked).

---

## 4. Components to Export to ONNX

Per-component export assessment (from source read):

| Component | Shape | Export difficulty | Notes |
|---|---|---|---|
| **T3 backbone** (Llama-style) | static per-step forward + KV-cache | Medium | decoder-with-past pattern; token loop + sampling driven from Rust |
| **S3Gen flow estimator** (CFM decoder) | static forward, called ~10× (Euler ODE) | Easy | Euler update loop in Rust; `x,mu,spks,cond -> dxdt` only — does not itself produce `mu`/`spks` (see next two rows, added after a gap found starting Milestone 6, ADR-0009) |
| **S3Gen flow *encoder*** (token → `mu`) | 6-block conformer, relative-position attention | Medium — dynamic-length export found broken, exported as 6 fixed-length buckets instead (ADR-0009) | `input_embedding`/`encoder`/`encoder_proj`; token-count buckets 200/400/600/800/1000/1200 |
| **CAMPPlus** (x-vector, `spks`) | CNN/TDNN + pooling | Easy in isolation, but dynamic-length export found broken — a single fixed 400-frame graph instead (ADR-0009) | Kaldi-fbank input computed host-side, like VE/S3-tokenizer's mel |
| **HiFiGAN vocoder** | Conv1d/ConvTranspose1d, no branches | Easiest | `remove_weight_norm()` before tracing |
| **Voice encoder** | mel → speaker embedding | Easy | needs mel preprocessing |
| **S3 tokenizer** (speech) | encoder | Easy | needs mel/feature preprocessing |
| **PerthNet watermarker** | small CNN | Easy, separate | lives in external `resemble-perth` package, not in-repo |

Text tokenizer is `tokenizer.json` → loaded at runtime by the Rust `tokenizers` crate
(no ONNX export needed).

**Why the loops export cleanly:** T3's `inference()` already does a manual prefill +
per-token loop passing `past_key_values` explicitly, with sampling (repetition penalty,
top-p, min-p, multinomial, CFG) as plain tensor math *outside* the model.
S3Gen's `solve_euler` is a fixed-count `for` loop calling one static estimator forward
per step then `x = x + dt*dxdt`. Both loops move to Rust; the networks export as static
graphs.

---

## 5. Rust Runtime Stack

- **`ort` 2.0.0-rc.13** (pyke) — wraps ONNX Runtime ~1.17+. EPs are opt-in Cargo
  features (`coreml`, `cuda`; CPU always on). Register per-session via
  `SessionBuilder::with_execution_providers([...])`; ORT **silently falls back** down
  the list to CPU if an EP won't initialize — so list `CoreML`/`CUDA` first, `CPU`
  last. Native lib via `ORT_STRATEGY=download` (default; auto-detects CUDA 11/12) or
  the `load-dynamic` feature to `dlopen` at runtime (preferred for robust
  distribution).
- **KV-cache**: no official Rust bindings for `onnxruntime-genai` → hand-write the
  decode loop against base `ort`, feeding `past_key_values.N.key/value` in and reading
  `present.N.key/value` out each step. Matches T3's already-explicit cache loop.
- **`tokenizers`** (~0.21) — `Tokenizer::from_file("tokenizer.json")`, pure Rust.
- **`hound`** — WAV read/write.
- **`realfft`/`rustfft`** + **`mel_spec`** — STFT/mel preprocessing for the voice
  encoder + S3 tokenizer (Whisper-style; match reference n_fft/hop/n_mels).
- **`clap`** — CLI arg parsing.

### Cross-platform / CUDA caveat (informs §2.3)
- macOS (CoreML) and CPU builds are genuinely self-contained: ship binary +
  `libonnxruntime` (dynamic or static). CoreML needs no extra runtime.
- CUDA-enabled `libonnxruntime` dynamically depends on a matching CUDA runtime +
  cuDNN (cuDNN 8 vs 9 mismatch breaks it) + NVIDIA driver → bundle these with the
  GPU artifact (decision §2.3). Prebuilt ORT CUDA binaries target CUDA ≥12.8 + cuDNN ≥9.

---

## 6. Proposed Repo Layout

```
vocal-ai/
  Cargo.toml                 # workspace
  crates/
    vocalai-cli/             # clap-based CLI entry (the binary)
      src/main.rs
    vocalai-core/            # inference runtime library
      src/lib.rs
      src/session.rs         # ort session setup + EP selection/fallback
      src/t3.rs              # T3 KV-cache decode loop + sampling
      src/s3gen.rs           # flow-matching Euler ODE loop + HiFiGAN
      src/voice_encoder.rs   # speaker embedding + mel preprocessing
      src/tokenizer.rs       # text tokenizer wrapper
      src/watermark.rs       # PerthNet
      src/audio.rs           # WAV I/O, resample, mel
  export/                    # Python export scripts (DEV-TIME ONLY, not shipped)
    export_t3.py             # decoder-with-past: past_key_values in / present out
    export_s3gen.py          # flow estimator (single static forward)
    export_hifigan.py
    export_ve.py             # voice encoder
    export_s3tokenizer.py
    export_perthnet.py
    parity_check.py          # numerical parity vs PyTorch reference
    requirements.txt         # torch, transformers, onnx, onnxruntime, chatterbox-tts
  models/                    # exported .onnx + tokenizer.json (bundled into release)
  docs/
    phase1-onnx-rust-cli-plan.md   # this file
  README.md
```

---

## 7. Implementation Milestones

1. **Scaffold** the Cargo workspace (`vocalai-cli`, `vocalai-core`), `export/` dir,
   `.gitignore`, README. Pin `ort` with `coreml`/`cuda` features gated per build
   profile. Set up `export/requirements.txt` with a working chatterbox + onnx env.
2. **Export the easy static graphs first** (HiFiGAN → voice encoder → S3 tokenizer)
   and stand up `export/parity_check.py`. This proves the export + parity toolchain
   end-to-end on low-risk pieces before the hard one. Validate ONNX outputs match
   PyTorch within tolerance on fixed input.
3. **Export the S3Gen flow estimator**; implement the Euler ODE loop in
   `vocalai-core/src/s3gen.rs`; chain into HiFiGAN. Parity-check mel→waveform.
4. **Export T3 as decoder-with-past** (explicit `past_key_values`/`present` I/O);
   implement the KV-cache decode loop + sampling (repetition penalty, top-p, min-p,
   temperature, CFG duplication) in `vocalai-core/src/t3.rs`. Parity-check token
   sequences against the reference on a fixed seed.
5. **Export PerthNet** (from external `resemble-perth`); wire watermarking into output.
6. **Wire the full pipeline** in `vocalai-core` and the `clap` CLI in `vocalai-cli`;
   implement `--voice` preprocessing (16 kHz resample + mel + speaker embedding) for
   zero-shot cloning. Support the built-in default voice when `--voice` is omitted.
7. **Per-platform packaging**: build the artifact matrix (§2.3), bundle weights +
   (for GPU) CUDA/cuDNN libs; smoke-test each artifact.

> **Sequencing rationale:** export difficulty ascends (HiFiGAN easiest → T3 hardest),
> so the toolchain and parity harness are proven on cheap components before the
> autoregressive backbone, where a hand-rolled KV-cache loop is the main risk.

---

## 8. Verification / Exit Criteria

- **Per-component parity**: exported ONNX outputs match the PyTorch reference within
  tolerance on fixed seed/input (`export/parity_check.py`), for each of T3, S3Gen
  estimator, HiFiGAN, voice encoder, S3 tokenizer, PerthNet.
- **End-to-end**: `vocalai --text "hello world" --out out.wav` produces audible,
  correct 24 kHz speech on macOS (CoreML EP) and on a CUDA box (Windows/Linux).
- **Voice cloning**: `--voice ref.wav` audibly matches the reference speaker.
- **CPU fallback**: forcing CPU EP produces equivalent output (slower).
- **Memory**: peak swap/RSS during a run is materially lower than the PyTorch/MPS
  baseline — measure on the same M3 Max with a `while true; do sysctl vm.swapusage;
  sleep 1; done` loop (macOS has no `watch` by default) and compare against the
  ~5 GB→~30 GB baseline.

---

## 9. Open Items / Risks

- **KV-cache export details** for T3 (naming/layout of `past_key_values.*` /
  `present.*`, dynamic axes for sequence length) — main technical risk; well-documented
  pattern but needs care.
- **Mel/feature preprocessing parity** — Rust `mel_spec`/`realfft` params must exactly
  match the reference (n_fft, hop, n_mels, windowing, normalization) or embeddings drift.
- **CFG path** — when `cfg_weight > 0` the reference duplicates the text sequence;
  replicate faithfully in the Rust decode loop.
- **`ort` is at 2.0.0-rc** (API not frozen) — pin exactly; expect possible churn.
- **PerthNet** is an external git-pinned package — confirm it exports cleanly and its
  license permits redistribution of exported weights.
- **Model licensing** — confirm Chatterbox weights may be redistributed inside a
  bundled artifact (decision §2.3 assumes yes).

---

## 10. References

- Reference model: `resemble-ai/chatterbox` (source read: `src/chatterbox/tts.py`,
  `models/t3/`, `models/s3gen/`, `models/voice_encoder/`, `models/s3tokenizer/`).
- `ort` crate: https://github.com/pykeio/ort · https://ort.pyke.io · https://docs.rs/ort
- onnxruntime-genai (evaluated, not used): https://github.com/microsoft/onnxruntime-genai
- CoreML EP: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- CUDA EP: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- `tokenizers`: https://crates.io/crates/tokenizers
- `hound`: https://crates.io/crates/hound · `mel_spec`: https://crates.io/crates/mel_spec
- Rust-on-ort TTS precedent: sbv2-api (Style-BERT-VITS2)
