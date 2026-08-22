# ADR-0015: CLI progress events (`--show-progress`) and install-script version-skip logic

**Date**: 2026-08-22
**Status**: Accepted
**Decider**: repo owner + Claude (session 2026-08-22)

## Context

Two related but separable pieces of work landed in the same session under `VAI-012`
(`docs/issues.md`): (1) the ticket's original scope, a `--show-progress` flag so a long
`vocalai` run (T3's decode loop dominates wall-clock, up to `--max-new-tokens` `session.run()`
calls) isn't silent; and (2) an addition the repo owner asked for while approving the plan --
`scripts/install.sh`/`install.ps1` should check the currently-installed binary/model versions
against what's latest and skip re-downloading anything already up to date, tracking those
versions in the install directory itself.

## Decision

**Progress events**: `vocalai-core::pipeline` gained two plain data types, `PipelinePhase`
(`VoiceConditioning`/`Decoding`/`Vocoding`/`Watermarking`) and `ProgressEvent`
(`Phase(PipelinePhase)` / `DecodeStep { step, max }`). `synthesize()` takes a new
`on_progress: &mut dyn FnMut(ProgressEvent)` parameter -- a trait object rather than `impl
FnMut`, so `vocalai-cli` can pick between a real renderer and a no-op behind one reference type
without duplicating the call site. `vocalai-core` renders nothing itself and gained no new
dependency; `vocalai-cli` added `indicatif` and renders phase labels as plain lines plus a
progress bar for the `Decoding` phase, active only behind `--show-progress` (default off, so
output is byte-identical to before when the flag is absent). The decode-loop counter wraps the
existing `decoder_step` closure already passed to `t3::generate_speech_tokens` in
`pipeline.rs::synthesize` -- `t3.rs` itself is unchanged.

**Install-script version tracking**: both `install.sh` and `install.ps1` now write a plain-text
version marker into the install directory after a successful install/update, and check it
before doing any work on a subsequent run:
- `${INSTALL_DIR}/.vocalai_version` (new file, this script's own invention -- nothing inside the
  release archive itself carries a version marker) holds the release tag (e.g. `v0.1.3`),
  written from the tag resolved by the version-check lookup below (guaranteed valid by the time
  it's written -- see "fail fast" below).
- `${MODELS_DIR}/MODELS_VERSION` reuses the file `scripts/publish_models.py` already copies from
  this repo's root `MODELS_VERSION` into the HF Hub repo -- but it is **not** fetched as part of
  the generic per-file download loop (excluded, alongside `README.md`/`.gitattributes`/
  `THIRD_PARTY_LICENSES`). It is written explicitly, only after every other model file in that
  run has downloaded successfully. This was a real bug caught during manual testing of this
  change, not a hypothetical: `MODELS_VERSION` happened to sort early in HF's file listing, so an
  interrupted first draft of this script left a version marker on disk claiming a genuinely
  incomplete model set (missing `t3_decoder.onnx`, `hifigan.onnx`, etc.) was current.
- The "latest version" lookups deliberately avoid `api.github.com`: an early draft used
  `https://api.github.com/repos/${GH_REPO}/releases/latest` to read `tag_name`, but manual
  testing on this session's own sandboxed environment hit that endpoint's unauthenticated rate
  limit (60 requests/hour/IP, shared with unrelated traffic on the same IP) with zero requests
  remaining. Both scripts instead read the tag straight off the existing "latest release"
  download URL's redirect `Location` header (a plain HEAD request against
  `github.com/.../releases/latest/download/<asset>`, the same URL the real download already
  hits) -- a normal web redirect, not subject to the REST API's rate limit.
- **Fail fast, don't degrade to a download, on any version-check lookup failure** (explicit
  repo-owner decision, made after an initial draft of this ADR proposed the opposite -- see
  "Alternatives rejected"). Every version-check request (the GitHub tag lookup, the HF
  `MODELS_VERSION` fetch, and the HF file-listing call used on a full model download) is first
  probed with a HEAD request. A dedicated check (`fail_if_rate_limited` in `install.sh`,
  `Test-RateLimited` in `install.ps1`) inspects the status/headers for a confirmed rate-limit
  signature -- HTTP 429, or GitHub's HTTP 403 with `x-ratelimit-remaining: 0` (the exact pattern
  observed live against `api.github.com` during this feature's own testing) -- and if matched,
  prints a specific "rate-limited while ..." error and exits 1 immediately, without ever
  attempting the real (much larger) download. Any *other* lookup failure (DNS blip, timeout,
  unparseable response) is also fatal: the tag/version is required to be non-empty before either
  script proceeds past the check, or it errors out with a generic "could not determine ..."
  message. Neither script silently falls back to "just download anyway" for any reason anymore.
- No force-reinstall bypass flag. If a local install gets corrupted in some way this session's
  checks don't catch, the documented fix is to delete the install directory and rerun (repo
  owner's explicit call).

## Rationale

- Keeping `vocalai-core` UI-agnostic (plain enums + a callback, no `indicatif` dependency)
  matches this repo's existing layering: rendering only ever lives in `vocalai-cli`, matching how
  `--use-gpu`/`--use-cpu` logging already works.
- Failing fast on any version-check failure, rather than silently proceeding to a full download,
  means a rate-limited or otherwise broken lookup can never be masked by an install that "just
  happens to work anyway" -- the repo owner's explicit reasoning: if the small metadata request
  is already failing, blindly attempting the real 26-file/~4GB download would likely hit the same
  wall repeatedly, burning time and bandwidth instead of surfacing one clear, actionable error.
- Distinguishing a confirmed rate-limit response from other failures (rather than treating every
  failure identically) lets the error message tell the user something concrete and actionable
  ("try again later" vs. "check your network connection"), instead of one generic message for
  every cause.
- Writing `MODELS_VERSION` last, after the whole download loop, rather than treating it as just
  another file HF happens to serve, is the only way to make the tracked marker mean "this model
  set is actually complete" rather than "a file named MODELS_VERSION happened to download."
- Avoiding `api.github.com` in favor of the existing download URL's redirect header means this
  feature adds zero new external dependencies/rate limits beyond what the script already relies
  on to fetch the binary itself -- and the fail-fast behavior above means that even this lower-
  risk endpoint being rate-limited is still caught and reported cleanly, not silently absorbed.

## Alternatives rejected

- **`api.github.com/repos/.../releases/latest` for the CLI version lookup**: rejected after
  manual testing showed it hitting a real, already-exhausted unauthenticated rate limit on this
  session's own sandbox -- a shared-IP failure mode that would be indistinguishable from a broken
  install to an end user.
- **Falling back to a full download on any version-check failure, including a confirmed rate
  limit (this ADR's original design)**: rejected by explicit repo-owner decision after review.
  The repo owner specifically did not want rate-limiting silently absorbed into "just download
  anyway" -- that would mean the *real*, much larger download is attempted immediately after
  being told to slow down, likely failing the same way repeatedly with no clear signal why.
  Superseded by the fail-fast design above.
- **Reading the installed CLI version back from the binary's own `--version` output** (this
  ADR's original design, meant to tolerate an empty/failed tag lookup): no longer needed once the
  tag lookup is required to succeed before the script proceeds at all -- the resolved tag is
  always valid at the point it's written, so there's nothing left for the readback to guard
  against. Dropped in favor of writing the resolved tag directly.
- **Treating `MODELS_VERSION` as just another downloaded file (original first-draft design)**:
  rejected after manual testing surfaced the exact failure mode described above -- an
  interrupted download could self-report as up to date.
- **A `VOCALAI_FORCE_INSTALL` env var to bypass both checks**: considered, rejected by explicit
  repo-owner decision -- kept out of scope for this session; deleting the install directory is
  the documented recovery path.
- **New automated tests for either change**: rejected -- both touch live ONNX sessions or real
  network calls, the same category of code this repo already verifies manually
  (`docs/manual-testing.md`) rather than mocking.

## Consequences

- `vocalai-core::pipeline::synthesize`'s signature changed (new trailing `on_progress`
  parameter) -- any future direct caller (currently only `vocalai-cli`) must pass one, even if
  it's a no-op closure.
- The install scripts now require network access to GitHub *and* HF Hub to do anything at all,
  including a first-time install -- there is no more "version check couldn't run, so just
  download normally" path. A user with a genuinely flaky connection will see the install fail at
  the version-check step rather than at the download step, which is a slightly different failure
  mode than before this feature existed (when there was no version check to fail), though the
  end result (install doesn't complete) is the same.
- The models-version skip check's completeness signal (a matching `MODELS_VERSION` plus at least
  one `*.onnx` file present) is a heuristic, not a full manifest diff -- it cannot detect a
  *partial* re-download that still leaves a stray `.onnx` file behind alongside missing others
  from a different failure mode than the one manual testing found. A full fix would require a
  cheap remote file-list diff, which isn't attempted here.
- `install.ps1`'s new redirect-header lookup and rate-limit detection
  (`[System.Net.HttpWebRequest]` with `AllowAutoRedirect = $false`, `Test-RateLimited`) could not
  be verified on real Windows this session (no Windows machine available) -- see
  `docs/manual-testing.md`'s note; `install.sh`'s equivalent was verified live against the real
  `v0.1.3` release + HF repo, including the fail-fast behavior on a deliberately-interrupted
  model download.
