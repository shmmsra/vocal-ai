# Architecture Decision Records (ADRs)

This directory holds the architectural decisions for vocal-ai. Every significant design choice that another agent might wonder about — *"why is this done this way?"* — should be recorded here.

---

## When to write an ADR

Write one when:
- You picked one option over plausible alternatives (e.g. one library vs another, one architecture vs another)
- The choice is hard to reverse without rewriting code
- A future contributor (human or agent) might want to change it without understanding the reason
- You explicitly rejected an approach that someone else might propose later

You do **not** need an ADR for:
- Style preferences (those go in `docs/agents/CONVENTIONS.md`)
- Bug fixes (the commit message tells the story)
- Minor refactors with no design implications

---

## ADR Template

Copy this template to `docs/decisions/NNNN-kebab-case-title.md` and fill in all sections. Number sequentially after the highest existing ADR (start with `0001`).

```markdown
# ADR-NNN: Title

**Date**: YYYY-MM-DD
**Status**: Accepted | Superseded by ADR-NNN | Deprecated
**Decider**: <name> + <agent> (session date)

## Context

What situation forced this decision? What constraints, requirements, or external factors are in play?

## Decision

What was chosen? State it crisply in one paragraph.

## Rationale

Why this option over alternatives? What does it buy us? What does it cost?

## Alternatives rejected

- **Option A**: why not
- **Option B**: why not
- **Option C**: why not

## Consequences

What becomes easier? What becomes harder? What new commitments does this create?
```

---

## ADR index

| # | Title | Status |
|---|-------|--------|
| [0001](0001-adopt-ai-sdlc.md) | Adopt the ai-sdlc-bootstrap workflow | Accepted |
| [0002](0002-onnx-plus-rust-runtime.md) | Export to ONNX and re-drive the loops from native Rust | Accepted |
| [0003](0003-rust-over-cpp.md) | Rust for the native runtime (evaluated against C++) | Accepted |
| [0004](0004-s3gen-euler-loop-generic-over-estimator-call.md) | S3Gen's Euler ODE loop is generic over the estimator call, not `ort`-coupled | Accepted |
| [0005](0005-t3-hand-rolled-decoder-export.md) | T3 exported as a hand-rolled decoder-with-past | Accepted |
| [0006](0006-split-ci-into-fast-and-parity-workflows.md) | Split CI into fast + parity workflows | Accepted |
| [0007](0007-exclude-t3-parity-from-ci.md) | Exclude T3's parity check from CI; run it locally instead | Accepted |
| [0008](0008-third-party-license-attribution.md) | Third-party license attribution for bundled ML assets (PerthNet + Chatterbox, both MIT) | Accepted |

*Add new rows as ADRs accumulate.*

---

## Conventions

- **Numbering**: `0001`, `0002`, ... — zero-padded, four digits, gapless.
- **Filename**: `NNNN-short-kebab-title.md`.
- **Status**: never delete an ADR. If superseded, change `Status` to `Superseded by ADR-NNN` and update the new ADR's `Context` section to reference the old one.
- **Read before changing**: if you're about to modify code that's covered by an ADR, read it first. The Rationale section often contains constraints that aren't obvious from the code.
