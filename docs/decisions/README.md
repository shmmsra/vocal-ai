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

*Add new rows as ADRs accumulate — the ONNX-vs-Python-wrapper and Rust-vs-C++ decisions from `docs/phase1-onnx-rust-cli-plan.md` (§2.1, §2.2) are good ADR-002/003 candidates once Milestone 1 starts.*

---

## Conventions

- **Numbering**: `0001`, `0002`, ... — zero-padded, four digits, gapless.
- **Filename**: `NNNN-short-kebab-title.md`.
- **Status**: never delete an ADR. If superseded, change `Status` to `Superseded by ADR-NNN` and update the new ADR's `Context` section to reference the old one.
- **Read before changing**: if you're about to modify code that's covered by an ADR, read it first. The Rationale section often contains constraints that aren't obvious from the code.
