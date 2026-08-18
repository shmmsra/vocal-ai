#!/usr/bin/env bash
#
# Generate every ONNX/npy model artifact vocalai needs, in the required order.
# See docs/dev-setup.md §11.1 for what each script produces.
#
# Usage:
#   scripts/export-all.sh                        # default-voice path only
#   scripts/export-all.sh --with-voice-cloning    # + --voice zero-shot cloning models

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "${REPO_ROOT}" ]; then
  echo "error: not inside a git repository" >&2
  exit 1
fi

VENV_PY="${REPO_ROOT}/export/.venv/bin/python"
if [ ! -x "${VENV_PY}" ]; then
  echo "error: ${VENV_PY} not found — set up the export venv first (docs/dev-setup.md §2)" >&2
  exit 1
fi

WITH_VOICE_CLONING=0
for arg in "$@"; do
  case "$arg" in
    --with-voice-cloning) WITH_VOICE_CLONING=1 ;;
    *)
      echo "error: unknown argument '$arg' (expected --with-voice-cloning)" >&2
      exit 1
      ;;
  esac
done

SCRIPTS=(
  fetch_tokenizer.py
  export_t3.py
  export_s3gen.py
  export_s3gen_flow_encoder.py
  export_hifigan.py
  export_perthnet.py
  export_campplus.py
  export_default_voice.py
)

if [ "${WITH_VOICE_CLONING}" -eq 1 ]; then
  SCRIPTS+=(export_ve.py export_s3tokenizer.py)
fi

for script in "${SCRIPTS[@]}"; do
  echo "→ Running export/${script}..."
  "${VENV_PY}" "${REPO_ROOT}/export/${script}"
done

echo "✓ All model artifacts generated in ${REPO_ROOT}/models/"
