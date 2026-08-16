---
name: no-auto-push
description: AI agents must never run `git push` — pushing is a one-way externally visible action that's always the human's call
metadata:
  type: feedback
---

Never run `git push` autonomously. Pushing is one-way, externally visible, and undoing it is expensive.

**Why**: Local commits are reversible. Pushed commits ripple to anyone watching the remote (CI, deploys). The human always makes the call (see `CONTRIBUTING.md §6`).

**How to apply**:
- Even when the human says "merge it" or "land it" — that means commit locally, not push (this repo uses direct-merge-to-main, no PRs).
- Even when all checks pass and the diff is approved.
- If the human explicitly types `git push` or says "push to remote", run it. Otherwise, stop after the local commit and report the SHA.
- Never offer to push or ask "should I push?".
