---
name: docs-as-part-of-done
description: Every feature/fix must update STATUS, CHANGELOG, requirements, issues, and (if runtime-affecting) manual-testing in the same commit
metadata:
  type: feedback
---

A feature or fix is not done until the documentation updates land in the same commit.

**Why**: This project is built by a human + multiple AI agents across many sessions. Without doc updates at commit time, the next agent reads stale state and makes wrong decisions. This is the SDLC's load-bearing rule (see `docs/agents/CONVENTIONS.md` and `CONTRIBUTING.md §5`).

**How to apply**: Before showing the commit diff, verify these files have been updated for the work in this session:
- `docs/agents/STATUS.md`
- `docs/CHANGELOG.md`
- `docs/requirements.md`
- `docs/issues.md`
- `docs/manual-testing.md` (if runtime behaviour changed)
- `docs/decisions/NNNN-*.md` (if an architectural decision was made)

If any is missing, fix it before requesting commit approval.
