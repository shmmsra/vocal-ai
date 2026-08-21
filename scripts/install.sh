#!/usr/bin/env bash
#
# Install vocalai: downloads the latest binary release from GitHub + the model
# artifacts from the public HuggingFace Hub repo (no HF token needed -- it's a
# public, anonymously-readable repo), laid out together in ./vocalai/ ready to run.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/shmmsra/vocal-ai/main/scripts/install.sh | bash
#
# Override the install directory with VOCALAI_INSTALL_DIR (default: ./vocalai).

set -euo pipefail

GH_REPO="shmmsra/vocal-ai"
HF_REPO="shmmsra/vocal-ai-models"
INSTALL_DIR="${VOCALAI_INSTALL_DIR:-./vocalai}"

case "$(uname -s)" in
  Darwin) ASSET="vocalai-macos" ;;
  Linux) ASSET="vocalai-linux-cpu" ;;
  *)
    echo "error: unsupported OS $(uname -s) -- see scripts/install.ps1 for Windows" >&2
    exit 1
    ;;
esac

echo "==> Installing vocalai (${ASSET}) into ${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"

echo "==> Downloading latest release binary..."
TMP_ARCHIVE="$(mktemp -t vocalai-release.XXXXXX.tar.gz)"
trap 'rm -f "${TMP_ARCHIVE}"' EXIT
curl -fsSL "https://github.com/${GH_REPO}/releases/latest/download/${ASSET}.tar.gz" -o "${TMP_ARCHIVE}"
tar -xzf "${TMP_ARCHIVE}" -C "${INSTALL_DIR}"
chmod +x "${INSTALL_DIR}/vocalai"

echo "==> Downloading model artifacts from https://huggingface.co/${HF_REPO}..."
MODELS_DIR="${INSTALL_DIR}/models"
mkdir -p "${MODELS_DIR}"
FILE_LIST="$(curl -fsSL "https://huggingface.co/api/models/${HF_REPO}" \
  | grep -o '"rfilename":"[^"]*"' \
  | sed -E 's/"rfilename":"([^"]*)"/\1/')"

if [ -z "${FILE_LIST}" ]; then
  echo "error: could not list files for ${HF_REPO} -- check the repo exists and is public" >&2
  exit 1
fi

while IFS= read -r f; do
  case "${f}" in
    README.md|.gitattributes|THIRD_PARTY_LICENSES) continue ;;  # not needed to run vocalai
  esac
  mkdir -p "${MODELS_DIR}/$(dirname "${f}")"
  echo "    ${f}"
  curl -fsSL "https://huggingface.co/${HF_REPO}/resolve/main/${f}" -o "${MODELS_DIR}/${f}"
done <<< "${FILE_LIST}"

echo "==> Done. Run it with:"
echo "    ${INSTALL_DIR}/vocalai --text \"hello world\" --out out.wav --models-dir ${MODELS_DIR}"
