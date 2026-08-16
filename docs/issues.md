# vocal-ai — Feature & Issue Tracker

> **Single source of truth for all planned, in-progress, and recently completed work.**
>
> Use this instead of GitHub Issues or JIRA. It lives in the repo so AI agents can read it without
> any external system access, and every status change is committed alongside the code that caused it.

---

## How to use this file

**Human**: Add new issues at the bottom of the Open section. Set priority, write acceptance criteria. No need to assign — just set status to `IN PROGRESS` when a session starts on it.

**AI Agent**: Before starting a session, scan this file for the highest-priority `OPEN` issue that matches the session goal. Update the status to `IN PROGRESS` (with the session date) when you begin. Mark `DONE` and move to "Recently closed" when complete. Add any new issues you discover (bugs, missing tests, follow-up work) during the session.

---

## Status legend

| Status | Meaning |
|--------|---------|
| `OPEN` | Ready to work on, not yet started |
| `IN PROGRESS` | Actively being worked on — note session date |
| `BLOCKED` | Cannot proceed — reason and blocker recorded |
| `DONE` | Complete and committed — note commit hash |
| `REJECTED` | Will not implement — reason recorded |

## Priority legend

| Priority | Meaning |
|----------|---------|
| **P0** | Blocking — nothing else should be worked on until resolved |
| **P1** | High — next logical thing to do in the current phase |
| **P2** | Medium — important but not urgent; can wait one session |
| **P3** | Low — nice to have; do it when there's slack |

---

## Ticket ID convention

Tickets use the prefix `VAI-NNN`, numbered sequentially (e.g. `VAI-001`, `VAI-002`). When closing, reference the ticket ID in the commit message: `feat(scope): vai-042 add retry logic`.

---

## Open Issues

### VAI-001 · P1 · OPEN · Milestone 1
**Scaffold the real Cargo workspace + export toolchain**

**Acceptance criteria**:
- [ ] Cargo workspace (`vocalai-cli`, `vocalai-core`) scaffolded per `docs/phase1-onnx-rust-cli-plan.md` §6
- [ ] `ort` pinned with `coreml`/`cuda` features gated per build profile
- [ ] `export/requirements.txt` set up with a working chatterbox + onnx env
- [ ] Throwaway placeholder tests (`crates/vocalai-core`, `export/tests/test_scaffold.py`) replaced with real ones
- [ ] Docs updated (CHANGELOG, STATUS, manual-testing)

**Notes**: See `docs/phase1-onnx-rust-cli-plan.md` §7, Milestone 1, for the full scope.

---

*Add new tickets below this line. Use the same format: heading with ID · priority · status · brief category; then bold one-line title; then acceptance criteria as checkboxes; then notes.*

---

## Recently closed

| Date | Ticket | Title | Commit |
|------|--------|-------|--------|
| 2026-08-16 | — | ai-sdlc-bootstrap scaffold | pending |

*When a ticket is closed: move it to this table, set the commit hash, and remove it from the Open section. Keep the last ~20 closures here; archive older ones to `docs/CHANGELOG.md`.*
