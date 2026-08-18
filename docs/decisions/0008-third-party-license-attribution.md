# ADR-0008: Third-party license attribution for bundled ML assets

**Date**: 2026-08-18
**Status**: Accepted
**Decider**: Shivam Mishra + AI agent (session 2026-08-18)

## Context

Plan §9 (Open Items) flagged two unresolved licensing questions blocking VAI-005
(export PerthNet, wire watermarking into the output pipeline):

- **PerthNet**: an external, git-pinned `resemble-perth` package — does its
  license permit redistributing the exported ONNX weights?
- **Chatterbox weights**: downloaded from HuggingFace at export time
  (`ResembleAI/chatterbox`, `export/_common.py:24`) — does its license permit
  redistributing the weights inside a bundled release artifact (plan §2.3
  assumes yes, but this was never verified)?

Both had to be checked before VAI-005 (and, later, Milestone 7 bundling) could
proceed without risk.

## Decision

Both resolve to **MIT**, verified directly rather than assumed:

- `resemble-perth==1.0.1` (installed in `export/.venv`): its dist-info
  `LICENSE` file is plain MIT (Copyright 2025 Resemble AI). The pretrained
  watermarker weights (`perth/perth_net/pretrained/implicit/perth_net_250000.pth.tar`)
  ship *inside* that same package/repo, not under a separate model license, so
  they're covered by the same MIT terms.
- `ResembleAI/chatterbox` on HuggingFace: the model card metadata lists
  `license: mit`, explicitly covering the weights for use, modification, and
  redistribution (including commercial use), provided the copyright/license
  notice is retained.

Neither license blocks exporting or redistributing these weights. This
unblocks VAI-005 to proceed without a licensing gate.

MIT's one condition — the original copyright and license notice must be
included in redistributed copies — becomes a new commitment: Milestone 7's
bundled release artifacts (plan §7 item 7) must ship a `THIRD_PARTY_LICENSES`
(or `NOTICE`) file carrying both MIT notices verbatim, alongside this repo's
own `LICENSE`. That file is not created now — there's nothing to bundle until
Milestone 7 actually produces release artifacts — but the commitment is
recorded here so it isn't missed later.

## Rationale

- **A package's code license doesn't automatically cover model weights** —
  weights are often distributed separately (e.g., via a model hub) under
  different terms than the surrounding code. Each source was checked
  independently rather than inferring one from the other: `resemble-perth`'s
  weights ship inside its own MIT-licensed package (same repo, same LICENSE
  file), while Chatterbox's weights live on HuggingFace and were checked
  against that model card directly.
- **Verifying now, once, is cheap; verifying late is expensive.** Confirming
  before implementation start (rather than after building the export
  pipeline) avoids sinking work into VAI-005 only to discover a licensing
  blocker afterward.

## Alternatives rejected

- **Assume MIT because the surrounding package/repo is MIT, without checking
  the weights specifically**: rejected — weights and code are often licensed
  separately in ML packages; this repo's `CLAUDE.md` hard constraints already
  require exported weights to be treated as build/release artifacts with their
  own provenance, so their license needed independent confirmation.
- **Defer the licensing check to Milestone 7 (bundling time)**: rejected —
  VAI-005 (export + wire watermarking) would already be built on top of an
  unresolved legal question; cheaper to confirm before any implementation work
  starts.

## Consequences

**Easier**: VAI-005 can proceed immediately with no licensing blocker.

**Harder**: nothing — no new constraint on the export or runtime code itself.

**New commitments**:
- Milestone 7 (VAI-007, per-platform packaging) must add a
  `THIRD_PARTY_LICENSES`/`NOTICE` file to every bundled release artifact,
  containing the verbatim MIT notices for `resemble-perth` and
  `ResembleAI/chatterbox`. Add this as an explicit acceptance-criteria line on
  VAI-007 when that ticket is picked up.
- If either upstream project changes its license in a future version bump,
  re-verify before updating the pin (`export/requirements.txt` for
  `chatterbox-tts`, and whatever pins `resemble-perth`'s version).
