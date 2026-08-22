# ADR-0014: Version-bump-driven triggers for model-publish + release pipelines (VAI-016)

**Date**: 2026-08-22
**Status**: Accepted
**Decider**: Shivam Mishra + AI agent (session 2026-08-22)

## Context

Since ADR-0013, publishing model artifacts (`models-export.yml`) and cutting a release
(`release.yml`) both require a manual step every time: `gh workflow run models-export.yml`
and `git tag vX.Y.Z && git push origin vX.Y.Z` respectively. VAI-016 asked to replace this
with "bump a version file/field, push" — no more remembering the right `gh`/`git` incantation.

The natural way to auto-create the triggering tag is a CI job that runs `git tag && git push`
after detecting a version bump. That runs into a real GitHub Actions constraint: **a git push
performed by a workflow run using the default `GITHUB_TOKEN` does not fire other workflows'
`on: push` triggers** (a deliberate anti-recursion safeguard — otherwise a workflow that pushes
could trigger itself, or another workflow, in an uncontrolled loop). Working around it the
"standard" ways all cost ongoing management overhead the repo owner explicitly wanted to avoid:

1. A **PAT** (personal access token, not `GITHUB_TOKEN`) stored as a new repo secret, used only
   for the tag-push step — sidesteps the anti-recursion rule since it isn't `GITHUB_TOKEN`, but
   needs a human to mint it and rotate it before it expires.
2. A **GitHub App installation token** — same effect as a PAT, no expiry to manage, but requires
   creating and installing a GitHub App, a meaningfully bigger one-time setup for what is a P3
   ticket.
3. Explicitly **dispatching** the downstream workflow via `gh workflow run` (the `workflow_dispatch`
   REST API, which `GITHUB_TOKEN` *can* trigger — the anti-recursion rule is specific to events
   caused by a `GITHUB_TOKEN`-authenticated push/PR, not explicit API dispatches). This avoids new
   secrets, but `release.yml`'s "cut a real release" step currently only fires on `github.ref_type
   == 'tag'`, which a `workflow_dispatch` run never satisfies — would need a new input just to
   fake that condition.

## Decision

**Don't route through a tag push at all for the automatic path.** Both `models-export.yml` and
`release.yml` gain a `paths`-filtered `push: branches: [main]` trigger directly — a version-file
edit landing on `main` is itself the trigger, with no intermediate tag-push-triggers-workflow step
required, and therefore no PAT, App token, or dispatch call needed:

- `models-export.yml`: `on: push: branches: [main], paths: ["MODELS_VERSION"]` (root file, new in
  this change), still plus `workflow_dispatch` for manual/debug runs.
- `release.yml`: `on: push: branches: [main], paths: ["Cargo.toml"]`, alongside the existing
  `push: tags: ["v*"]` and `workflow_dispatch` triggers, both left unchanged.

Each workflow gains a leading guard job/step that reads the relevant version (root
`[workspace.package] version` for `release.yml`, `MODELS_VERSION`'s contents for
`models-export.yml`) and checks via `git ls-remote --exit-code --tags origin refs/tags/<name>`
whether a tag for that exact version already exists. If it does, the rest of the pipeline is
skipped — this is what implements "only proceed if it actually changed" (an *exact-tag-existence*
check, not a lexicographic/semver comparison against "the last matching tag" as literally worded
in VAI-016's acceptance criteria — see Rationale). This also makes the `paths` filter's coarseness
harmless: a whitespace-only edit to `MODELS_VERSION`/`Cargo.toml` still matches the path filter,
but the guard finds the tag already exists and no-ops.

The corresponding tag is still created, but purely as a release-anchor/record — no longer relied
on to trigger anything, so creating it with the default `GITHUB_TOKEN` is fine:
- `models-export.yml` explicitly `git tag`s + pushes `models-vN` after a successful publish, and
  passes the same tag to a new `publish_models.py --hf-tag` flag, which tags the HF Hub revision
  to match via `HfApi.create_tag(...)`. `MODELS_VERSION` itself is also copied into the published
  HF repo folder (same temporary-write-then-cleanup pattern ADR-0013 already used for
  `THIRD_PARTY_LICENSES`), so the tag is externally visible on the Hub too, not just in git.
- `release.yml` passes `tag_name: v$VERSION` explicitly to `softprops/action-gh-release`, which
  creates the tag itself as part of publishing the release (targeting `github.sha` explicitly) —
  no separate tag-push step needed there at all.

One non-obvious interaction worth recording: adding `paths: ["Cargo.toml"]` to `release.yml`'s
`push` trigger does **not** affect the existing manual `push: tags: ["v*"]` path, because GitHub
skips path filtering entirely for a push that creates a brand-new ref — and every tag push is, by
definition, a new ref. A human still running `git tag vX.Y.Z && git push` gets the exact same
unconditional trigger as before this change.

Package versioning is also consolidated: `Cargo.toml` gains `[workspace.package] version =
"0.1.2"` (matching the already-released `v0.1.2` tag, so landing this change is not itself
mistaken for a bump), and both `vocalai-cli`/`vocalai-core`'s `Cargo.toml` switch their duplicated
`version = "0.1.0"` (stale — it was never wired to actual release tags) to `version.workspace =
true`.

## Rationale

- Zero new secrets, tokens, or App installations to create or ever rotate — directly addresses the
  repo owner's explicit "standard, simple, free of management overhead" requirement.
- Reuses a first-class, thoroughly-documented Actions trigger (`push` + `paths`) instead of
  chaining workflow-triggers-workflow, which is exactly the kind of thing the anti-recursion
  safeguard exists to make awkward.
- The exact-tag-existence check is simpler to reason about and implement than parsing/comparing
  the "last matching tag" by semver, and gives the same practical guarantee ("don't re-publish/
  re-release a version that's already out") that the literal acceptance-criteria wording wanted.

## Alternatives rejected

- **PAT secret**: rejected per repo owner's explicit preference for zero ongoing token/secret
  management, even though it would have matched VAI-016's literal wording most closely (a real
  tag push triggering the existing tag-triggered path, unmodified).
- **GitHub App token**: rejected as disproportionate one-time setup for a P3 ticket.
- **`gh workflow run` dispatch**: rejected — would still need a new `release.yml` input to fake
  the tag-triggered release path under `workflow_dispatch`, without actually removing any
  complexity relative to the chosen design.
- **Lexicographic "compare against the last matching tag"**: rejected in favor of the simpler,
  equally-effective exact-existence check described above.

## Consequences

**Easier**: publishing a new model version or cutting a release is "edit a file, push to `main`"
— no `gh`/`git` incantation to remember, and no new secret to create or rotate.

**Harder**: two workflows now each carry a small amount of extra trigger/guard logic; a reader
verifying "why does adding a `paths` filter not break the existing tag-push trigger" needs to know
the new-ref-skips-path-filtering behavior documented above (also called out inline in
`release.yml`).

**New commitments**: `MODELS_VERSION`'s value must be bumped by hand at the same time as any real
model-artifact change intended for publishing (no auto-detection of "did `export/` output actually
change" — a deliberate, human-authored version bump remains the trigger, matching ADR-0013's
existing stance that publishing to a public model repo should be a deliberate action, not an
automatic side effect of unrelated commits).

## Addendum: `MODELS_VERSION` has no prior tag on first landing

Unlike `Cargo.toml` (seeded at `0.1.2` to match the already-existing `v0.1.2` tag so this change
doesn't itself look like a bump), no `models-v*` tag has ever existed — `models-export.yml` has
only ever run via manual `workflow_dispatch`. Seeding `MODELS_VERSION` at `0.1.0` therefore *will*
trigger one real `models-export.yml` run when this change lands on `main` (re-exporting and
re-publishing the current models to establish the `models-v0.1.0` baseline tag). This is harmless
and idempotent, not a design flaw — flagged here for whoever reviews that first run's job summary.
