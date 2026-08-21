# ADR-0013: HuggingFace Hub model distribution + release packaging (VAI-007)

**Date**: 2026-08-21
**Status**: Accepted
**Decider**: Shivam Mishra + AI agent (session 2026-08-21)

## Context

Milestone 7 (VAI-007, per-platform packaging) needed a concrete distribution design.
Plan §2.3 originally assumed "model weights bundled into the release artifact,"
without saying *where the weights come from* or *how CI produces them* -- and the
repo owner raised two new constraints while starting this ticket:

- The GitHub repo is going public, partly to get more free CI resource, and the
  generated `.onnx`/`.npy` files should be published to a public HuggingFace Hub
  repo (`shmmsra/vocal-ai-models`) instead of committed to git or shipped only as
  opaque release-asset blobs.
- Automated checks (GitHub Actions) must never execute real model inference --
  CPU or GPU -- as part of any gate. Basic exported-file validation is fine; real
  audio-generation validation stays a manual, human-run step.

Several sub-decisions had to be resolved along the way:

1. **Model delivery model**: bundle weights into the release artifact at build
   time (offline end-user CLI, matching the original "fully self-contained"
   goal), or have the CLI download from HF Hub on first run?
2. **Where does the export + HF-publish step run?** Initially planned as a local,
   human-run step (to sidestep T3's ~9GB export-time memory peak, which excluded
   it from CI per ADR-0007). The repo owner asked for it to run in GitHub Actions
   instead.
3. **Runner sizing**: what are GitHub-hosted runners' actual CPU/RAM specs, for
   public vs private repos, and which one can fit T3's export?
4. **T3 parity gating**: if export runs in CI, does the ~9GB-peak T3 parity check
   (excluded from the always-on `parity.yml` per ADR-0007) need to run somewhere
   before publishing?
5. **How does the release-build workflow know which HF Hub revision to bundle?**
6. **Smoke-test scope**: what exactly does "validate the release artifact" mean,
   given constraint above (no real inference in CI)?
7. **GPU artifact scope**: build the full Windows/Linux CUDA+cuDNN-bundled
   artifacts now, or defer?

## Decision

**1. Bundle-at-build, not download-on-first-run.** `release.yml` downloads the
pinned HF Hub revision during the CI build and embeds it into each platform's
archive. The CLI itself gains no new download/cache code; end users still get a
fully offline binary, matching the original plan §2.3 goal. HF Hub replaces
*where the weights are stored/versioned*, not *how the CLI gets them*.

**2. Export + publish runs in CI, on a dedicated, manual-trigger-only workflow**
(`.github/workflows/models-export.yml`, `workflow_dispatch` only). Not triggered
automatically on `export/**` pushes: publishing to a public model repo is a
deliberate, externally-visible action, and export is the heaviest job in the
whole pipeline -- auto-firing on every touch to `export/` (including test-only
edits) risked surprise heavy runs and accidental republishes.

**3. Corrected runner-spec numbers** (verified against GitHub's current docs,
2026-08-21 -- both the repo owner's and this session's initial guesses were
wrong in different ways):

| Runner | Public repos | Private repos |
|---|---|---|
| `ubuntu-latest` / `windows-latest` | **4-core, 16 GB RAM** | 2-core, 8 GB RAM |
| `macos-latest` (ARM64/M1) | 3-core, **7 GB RAM** (same either way) | 7 GB RAM |
| `macos-*-intel` label | 4-core, 14 GB RAM (same either way) | 14 GB RAM |

Going public gives `ubuntu-latest` **16 GB RAM** (double the 8 GB private cap) --
comfortably above T3's ~9 GB export peak (ADR-0007). `macos-latest` (the actual
default macOS label) is only 7 GB; the 14 GB figure applies only to the legacy
`-intel` label, not the ARM64 default. `models-export.yml` therefore runs on
`ubuntu-latest`, not macOS: it fits once public, it's the same environment
`parity.yml` already runs on (no new macOS-specific risk), and the export
scripts are pure Python/PyTorch, not OS-specific.

**4. Full parity gate before publish, without touching `parity.yml`.**
`models-export.yml` runs `make test-py-parity` (the *full* suite, including the
`heavy_build`/T3 check that `test-py-parity-ci` excludes) as an explicit step
after export and before publish. This is additive and separate from the
always-on `ci.yml`/`parity.yml` gates, which keep their existing scope and
triggers (every push/PR) unchanged -- ADR-0007's exclusion still describes
*that* gate correctly. The new gate only runs on this workflow's own manual
trigger, where the ~9GB cost is affordable (public `ubuntu-latest`, occasional
run, not per-push).

**5. HF revision resolution: query at build time, no git-tracked pin file.**
`release.yml` resolves the HF Hub repo's revision (default `main`, overridable
via a `workflow_dispatch` input) at build time via `HfApi.model_info(...).sha`,
rather than committing a `MODEL_REVISION` file to `main` that CI would need to
auto-update. Avoids a CI-bot-commits-to-main pattern entirely. Traceability
comes from recording the resolved commit SHA in the job summary (and,
implicitly, in which release assets exist), not from git history.

**6. Smoke tests are structural-only, in both workflows.**
`scripts/smoke_test_artifact.py` runs `onnx.checker.check_model()` on every
`.onnx` file, loads every `.npy`, parses `tokenizer.json`, checks required extra
files exist, and runs the compiled binary with `--version` only. It never opens
an ONNX Runtime inference session and never synthesizes audio. Real end-to-end
audio quality, CPU-fallback-equivalence, and the memory/swap-vs-PyTorch
benchmark (plan §8) stay a manual, per-platform step documented in
`docs/manual-testing.md` -- the same pattern this repo already uses for T3's
local-only parity check.

**7. GPU artifact scope deferred.** This pass ships `vocalai-macos`
(CoreML→CPU fallback, built-in), `vocalai-windows-cpu`, `vocalai-linux-cpu` --
all buildable and structurally verifiable on standard public GitHub-hosted
runners with no new licensing research. Windows/Linux CUDA+cuDNN-bundled
artifacts (original plan §2.3's `vocalai-{windows,linux}-cuda`) are deferred to
a new ticket, **VAI-015**: they need real GPU hardware to smoke-test and a
redistribution-license check for NVIDIA's runtime libs that hasn't been done
(only the model weights were checked, in ADR-0008).

**8. `THIRD_PARTY_LICENSES`** added at the repo root and copied into every
bundle, fulfilling ADR-0008's standing commitment (verbatim MIT notices for
`resemble-perth` and `ResembleAI/chatterbox`). **Addendum (2026-08-21, caught by
the repo owner after the first real `models-export.yml` run):** the public HF
Hub repo is itself a redistribution of these weights, not just the CLI release
bundle -- MIT's notice-inclusion condition applies there too. The model card's
`license: mit` front matter is metadata, not the notice text itself, so
`scripts/publish_models.py` now also copies `THIRD_PARTY_LICENSES` into the
HF repo alongside the exported files (same temp-write-then-cleanup pattern
already used for the generated `README.md`).

## Rationale

- Decoupling export/publish from the release build means `release.yml` never
  pays T3's ~9GB export cost on every release -- it only downloads
  already-published, already-verified files. Cheaper, faster, and avoids
  re-verifying the same weights on every tag push.
- Structural-only smoke tests match the repo owner's explicit instruction and
  don't duplicate `parity.yml`'s job (numeric correctness) or manual testing's
  job (perceptual/output correctness) -- they catch a narrower, real failure
  mode: a corrupted/truncated/incomplete artifact reaching a release.
- Querying HF Hub for the current revision at build time is simpler than a
  git-tracked pin file and avoids introducing bot commits to `main`, which this
  repo's merge policy (`CONTRIBUTING.md §6`) doesn't otherwise use.

## Alternatives rejected

- **Download-on-first-run CLI**: rejected -- would add new download/cache/
  checksum code to `vocalai-core`/`vocalai-cli` and change the offline-first
  design goal, for no benefit once bundle-at-build is viable.
- **Run export/publish on `macos-latest` for its RAM**: rejected once the real
  spec was checked -- `macos-latest` (ARM64) is only 7 GB, *less* than public
  `ubuntu-latest`'s 16 GB, and macOS runners are pricier/slower for a job with
  no macOS-specific requirement.
- **Auto-trigger `models-export.yml` on `export/**` pushes**: rejected by the
  repo owner -- publishing to a public model repo should be deliberate, and
  path-based filters are too coarse (would fire on test-only edits).
- **Git-tracked `MODEL_REVISION` pin file, auto-committed by CI**: rejected --
  adds a bot-commit-to-main pattern and a re-trigger-loop risk for one line of
  traceability that a job-summary SHA already provides.
- **Full CUDA/cuDNN bundling now**: rejected for this pass -- real GPU hardware
  and an unresolved NVIDIA redistribution-license question make it a
  meaningfully different, riskier scope; tracked separately as VAI-015.
- **Retiring ADR-0007's `parity.yml` T3 exclusion**: rejected for this pass --
  the repo owner explicitly chose to leave `parity.yml`/`ci.yml` untouched and
  add the full parity gate only inside `models-export.yml`; revisit later if
  desired.

## Consequences

**Easier**: publishing a new model version is a single manual GitHub Actions
run, self-gated by the full parity suite; cutting a release never re-pays the
export cost; no new Rust download/cache code.

**Harder**: two separate workflows to reason about instead of one; a release
always bundles whatever is currently at HF Hub's `main` unless a specific
revision is passed to `workflow_dispatch` -- a model update and a code release
aren't atomically linked by git history (mitigated by the job-summary SHA
record, but worth remembering if bisecting a bad release).

**New commitments**:
- `HF_TOKEN` (a HuggingFace write token) must be added as a GitHub Actions repo
  secret before `models-export.yml` can publish -- a manual, one-time,
  human-only setup step (never done by an agent).
- VAI-015 tracks the deferred CUDA/cuDNN-bundled GPU artifacts.
- Revisit ADR-0007's `parity.yml` T3 exclusion as a candidate follow-up now that
  public `ubuntu-latest` has 16 GB RAM -- not required, but newly viable.
