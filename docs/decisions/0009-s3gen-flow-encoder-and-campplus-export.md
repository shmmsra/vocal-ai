# ADR-0009: Export S3Gen's flow encoder + CAMPPlus as fixed-length/bucketed graphs; the Milestone 3 export was incomplete

**Date**: 2026-08-18
**Status**: Accepted
**Decider**: Shivam Mishra + AI agent (session 2026-08-18)

## Context

Starting Milestone 6 (`docs/issues.md` VAI-006 — wire the full pipeline in `vocalai-core` +
`vocalai-cli`), a source read of the reference implementation
(`chatterbox/models/s3gen/{s3gen,flow}.py`) turned up a gap: Milestone 3
(`docs/phase1-onnx-rust-cli-plan.md` §7, `export_s3gen.py`) exported `s3gen.flow.decoder.estimator`
— the CFM diffusion network that maps `(x, mu, spks, cond) -> dxdt` — and called that "the S3Gen
flow estimator." That's accurate as far as it goes, but it's only the *downstream* half of S3Gen's
token-to-mel path. Nothing was ever exported for the *upstream* half: the network that turns S3
speech tokens (T3's output) plus a reference-audio prompt into the `mu`/`spks`/`cond` tensors the
estimator actually consumes.

Concretely, `chatterbox/models/s3gen/flow.py::CausalMaskedDiffWithXvec.inference()` does:
- `mu`: `input_embedding` (nn.Embedding, vocab 6561 → 512) on the concatenated
  `[prompt_token, generated_token]` sequence → `encoder` (`UpsampleConformerEncoder`, 6-block,
  upsamples the 25Hz token rate to the 50Hz mel rate) → `encoder_proj` (Linear 512 → 80).
- `spks`: `S3Gen.speaker_encoder` (`CAMPPlus`, an x-vector CNN over Kaldi-style fbank features of
  the 16kHz reference wav) → L2-normalize → `flow.spk_embed_affine_layer` (Linear 192 → 80).
- `cond`: the reference audio's own 24kHz/80-mel spectrogram (`mel_extractor` in
  `s3gen.py::embed_ref`), zero-padded out to the full token+prompt length.

None of `input_embedding`/`encoder`/`encoder_proj`/`CAMPPlus` were exported, and
`export/parity_check.py::check_s3gen` only ever fed **random synthetic** `mu`/`spks`/`cond` tensors
into the estimator (see that function and its comment "real audio preprocessing is Milestone 6
scope") — so nothing has ever mechanically proven the real token→mel chain produces correct output.
Per `CLAUDE.md` §1's hard constraint, no exported component ships into `vocalai-core` without a
passing parity check, so this had to be closed before Milestone 6's Rust wiring could touch real
audio.

**A first attempt at a single dynamic-length export for each network was tried and found broken.**
Both networks were initially exported with a dynamic `tokens`/`frames` axis (`dynamic_axes`, the
same convention every other export in this repo uses) and their `check_*` parity functions passed
— but only because those checks, like every existing `check_*` function (e.g. `check_ve`), compare
PyTorch-vs-ONNX at a *single fixed shape*, so a graph that only works at its own tracing-example
shape trivially "passes." A manual sanity check at a *different* shape than the tracing example
(not part of the original test) surfaced real bugs in both:
- **Flow encoder**: `EspnetRelPositionalEncoding.position_encoding(size=x.size(1), ...)`
  (`chatterbox/models/s3gen/transformer/embedding.py`) takes `size` as a plain Python `int`, so
  `torch.onnx.export`'s tracer bakes the tracing example's sequence length into the graph as a
  constant. At any other token count, the relative-position term (`matrix_bd` in `attention.py`)
  keeps the traced length while the rest of the graph scales dynamically, and ONNX Runtime raises a
  broadcast-shape error.
- **CAMPPlus**: bisecting by frame count (each traced *and* tested at the same length, so this
  wasn't even a generalization question) showed the exported graph is wrong at *most* lengths and
  correct only when a specific internal trim is a no-op. `CAMLayer.seg_pooling` (`xvector.py`)
  average-pools in 100-frame segments, expands each pooled value back across its segment, then trims
  the result to the input's original length (`seg[..., :x.shape[-1]]`). That trim is only a no-op
  — and only then does the exported graph match the eager reference to ~1e-6 regardless of content —
  when the pre-pooling time length (`frames // 2`, after `xvector.tdnn`'s stride-2 first layer) is
  itself an exact multiple of 100, i.e. iff `frames` is a multiple of 200. Every tested non-multiple
  (100, 150, 250, 300, 350, 500) was off by 0.4-1.8 absolute.

## Decision

Two new ONNX exports, gated by parity checks like every other component — both now fixed-length by
necessity, not dynamic:

- **`export/export_s3gen_flow_encoder.py`** → `models/s3gen_flow_encoder_{bucket}.onnx`, one static
  graph per bucket in `TOKEN_BUCKETS = (200, 400, 600, 800, 1000, 1200)` (covers the real range:
  `speech_cond_prompt_len`=150 fixed + up to `max_new_tokens`=1000 generated). Each graph is fully
  static (batch=1, token count = that bucket's exact size — no `dynamic_axes` at all) so the
  relative-position math is correct by construction: the physical tensor length always equals what
  was traced. Wraps `flow`'s real `input_embedding`/`encoder`/`encoder_proj` submodules directly (no
  weight copying — same pattern as every export script except T3's, ADR-0005). Input: `token (1,
  bucket)` int64 (the *already concatenated* prompt+generated sequence, right-padded to the bucket —
  concatenation and padding happen host-side in Rust, matching `flow.py`'s own `torch.concat` one
  level above `input_embedding`) + `token_len (1,)` int64, the *true* unpadded length. `token_len`
  stays a genuine runtime input even though the shape is fixed: it only drives
  `make_pad_mask`-based masking, a *value* computation (not a *shape* computation), which traces
  correctly regardless of tracing length. Output: `mu (1,80,2*bucket)`, `mask (1,1,2*bucket)`
  (`token_mel_ratio=2`, confirmed empirically). Rust picks the smallest bucket ≥ the real token
  count and right-pads. Only the `finalize=True` (non-streaming) path is reproduced — the
  `pre_lookahead` streaming-trim branch is out of scope (plan §2.4: no streaming synthesis in
  Phase 1).
- **`export/export_campplus.py`** → `models/campplus.onnx`, a single static graph at
  `CAMPPLUS_FRAMES = 400` (a multiple of 200, ~4s at Kaldi's typical 10ms frame shift — a common
  x-vector enrollment-window length; enforced by an `assert` in `export()`). Unlike the flow
  encoder, CAMPPlus has no length/mask input at all, so a padded-to-bucket input would corrupt
  `StatsPool`'s statistics with zeroed padding frames — bucketing the *input* the way the flow
  encoder does is unsound here regardless of the graph fix. Rust must always feed exactly
  `CAMPPLUS_FRAMES` frames of *real* fbank content (trimming a longer reference clip, or repeating a
  shorter one — never zero-padding) — a Milestone 6 (Part B) wiring concern, not this export step's.
  Wraps `CAMPPlus.forward` (not `.inference()`, which does Python-side Kaldi-fbank extraction over a
  Python list — that stays host-side, like every other network's preprocessing). Input: precomputed
  fbank features `(1, 400, 80)`. Output: raw 192-dim x-vector (pre-normalize, pre-affine). Also
  dumps `flow.spk_embed_affine_layer.weight`/`.bias` to `.npy` — a bare `Linear(192,80)` plus an L2
  normalize is cheap enough to hand-roll as a Rust matmul rather than wrap in its own ONNX session,
  the same call ADR-0005 made for T3's embedding-table lookup.
- **`export/export_default_voice.py`** → `models/default_voice/*.npy`. A one-off dump of the
  bundled `conds.pt`'s tensor fields (not a model export, no parity check — `torch.load` isn't
  reachable from Rust, so this just moves tensors into a Rust-readable format unchanged). Needed so
  `vocalai --text "..." --out out.wav` works with no `--voice` flag.
- `export/parity_check.py` gains `check_s3gen_flow_encoder` and `check_campplus`. Both go beyond the
  existing same-shape-only convention (`check_ve` etc.), specifically because that convention is
  what let the original dynamic-length bugs slip through invisibly:
  - `check_s3gen_flow_encoder` checks *every* bucket, each at a `token_len` deliberately less than
    the bucket size (the real runtime padding pattern), verifying both (a) ONNX matches eager
    PyTorch at that bucket, and (b) *padding invariance* — changing the padding tokens beyond
    `token_len` must not change `mu`'s valid-length prefix. (b) is the property Rust's
    one-bucket-serves-many-lengths scheme depends on; without it, (a) alone would only prove the
    graph works when there's zero padding, which is never the real usage pattern.
  - `check_campplus` checks only at exactly `CAMPPLUS_FRAMES` (there is no padding scheme to check
    invariance against, per the no-masking limitation above).

## Rationale

- Keeping the ONNX-graph boundary at "precomputed features in, network output out" (no raw-audio
  or Python-list preprocessing inside the traced graph) matches every existing export
  (`ve.onnx` takes precomputed mels, `s3tokenizer.onnx` takes precomputed log-mels) — one
  established convention, not a new one invented for this component.
- Bucketing/fixed-length export keeps wrapping the *real* submodules directly (no
  reimplementation-drift risk, unlike T3/ADR-0005) while sidestepping both discovered bugs
  entirely — neither bug is in the math this repo owns; both are legacy-tracer limitations
  (shape-dependent Python-int slicing) in third-party code. Re-deriving relative-position attention
  or `seg_pooling` by hand to be trace-safe would be substantially more implementation and
  parity-check surface for no accuracy benefit over "run the real module at a length it was
  actually traced at."
- CAMPPlus is a plain CNN/TDNN stack otherwise (Conv1d + BatchNorm + Linear) with no cache
  abstraction — the underlying bug is narrowly in one pooling op's shape handling, not pervasive.

## Alternatives rejected

- **Single dynamic-length export for each graph** (the first attempt): rejected — confirmed broken
  for the flow encoder (broadcast error at any non-traced token count) and for CAMPPlus (silently
  wrong output, ~0.4-1.8 absolute error, at any non-multiple-of-200 frame count). Not merely a
  theoretical risk; empirically demonstrated.
- **Hand-rolled reimplementation of the relative-position attention / `seg_pooling`** (T3/ADR-0005's
  pattern): rejected for now — both bugs are narrow, isolable (bucket around them) legacy-tracer
  limitations in specific ops, not a fundamental incompatibility with tracing the way T3's
  `Cache`-object-based `forward` was. Hand-rolling would trade a well-understood, cheap workaround
  (padding to a fixed size) for a second implementation of nontrivial math with its own drift risk.
  Revisit if the bucket/fixed-window scheme proves too limiting in practice (e.g. real text
  regularly exceeding the largest token bucket).
- **Export `CAMPPlus.inference()` (including the Kaldi-fbank extraction) as one graph**: rejected —
  `torchaudio.compliance.kaldi.fbank` is called per-utterance over a Python list inside
  `xvector.py::extract_feature`, not a static batched tensor op; every other network in this
  pipeline keeps preprocessing host-side for the same reason.
- **Skip CAMPPlus/flow-encoder exports and hand-roll both networks directly in Rust from the
  checkpoint weights**: rejected — re-deriving a 6-block conformer encoder's math and a CNN
  x-vector stack by hand (rather than tracing the real modules) would be strictly more
  implementation and parity-check surface than exporting them.

## Consequences

**Easier**: Milestone 6's Rust wiring has a real, parity-checked source for `mu` and `spks`
instead of an open question; the pipeline's data flow now matches
`docs/phase1-onnx-rust-cli-plan.md` §4's original per-component table once these two rows are
added to it.

**Harder**: two more ONNX sessions to load and drive from Rust; the flow encoder is now *six*
sessions (one per bucket) with runtime bucket-selection/padding logic, not one; a fourth (fifth,
counting VE's and S3-tokenizer's) DSP front-end to port faithfully — Kaldi-style fbank extraction
(`torchaudio.compliance.kaldi.fbank`) for CAMPPlus's input; and CAMPPlus's Rust-side caller must
always assemble exactly `CAMPPLUS_FRAMES` frames of real content (trim/repeat logic), never rely on
padding.

**New commitments / residual risk**:
- Rust-side Kaldi-fbank parity is **not** guaranteed bit-exact against
  `torchaudio.compliance.kaldi.fbank` and is tracked as an accepted residual risk, in the same
  category as `docs/issues.md` VAI-005's resampler gap (`watermark.rs`'s `rubato` vs `soxr_hq`) —
  not silently assumed, to be revisited if real generated audio sounds wrong once Milestone 6's Rust
  wiring lands.
- Text that generates more speech tokens than `TOKEN_BUCKETS`'s largest bucket (1200, i.e. >1050
  generated tokens after the 150-token prompt) has no bucket to fall into. Milestone 6 must either
  truncate, add a larger bucket, or otherwise handle this — not addressed here.
- If a future session needs true unbounded-length support (e.g. streaming synthesis, out of scope
  per plan §2.4), the bucket/fixed-window scheme won't extend cleanly and the hand-rolled
  reimplementation alternative above should be revisited.
