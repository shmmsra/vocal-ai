#!/usr/bin/env bash
#
# Install the project's git hooks. Run once after cloning:  make setup-hooks
#
# This installs one hook:
#   - pre-commit: runs `make check` before every commit
#
# The pre-commit hook can be bypassed with `git commit --no-verify`, but per
# CONTRIBUTING.md §2 that's only allowed for docs-only / housekeeping commits
# with zero code changes. Bypassing on code changes is a violation.
#
# Manual-commit policy for this repo: convention-only (see CONTRIBUTING.md §10).
# No post-commit hook, no docs/commit-log.md, no Co-Authored-By trailer.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "${REPO_ROOT}" ]; then
  echo "error: not inside a git repository" >&2
  exit 1
fi

HOOKS_DIR="${REPO_ROOT}/.git/hooks"
PRE_COMMIT="${HOOKS_DIR}/pre-commit"

# ─── pre-commit ───────────────────────────────────────────────────────────────

cat > "${PRE_COMMIT}" <<'HOOK'
#!/usr/bin/env bash
#
# Pre-commit hook installed by `make setup-hooks`.
# Aborts the commit if `make check` fails.

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

echo "→ Running make check (pre-commit)..."
if ! make check; then
  echo
  echo "✗ make check failed — commit aborted."
  echo "  Fix the failures and run 'git commit' again."
  echo "  To bypass for docs-only commits: git commit --no-verify"
  exit 1
fi

echo "✓ make check passed"
HOOK

chmod +x "${PRE_COMMIT}"
echo "✓ Installed pre-commit hook at ${PRE_COMMIT}"

echo
echo "Hooks ready. Test the pre-commit gate with: make check"
