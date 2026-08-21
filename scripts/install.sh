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

# --retry covers curl's own transient-error set (connection failures, timeouts, and HTTP
# 408/429/500/502/503/504); --connect-timeout only bounds the initial connection, not the
# whole transfer, so large model files aren't killed mid-download on a slow link.
# Two variants: silent for small metadata calls, a visible progress bar for real
# downloads -- the model files are several hundred MB to ~2GB each, and HF Hub can be slow
# (rate limiting on the free/anonymous tier), so a silent multi-minute hang looks identical
# to a stall without one.
CURL_QUIET=(curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 10)
CURL_DL=(curl -fL --retry 3 --retry-delay 2 --connect-timeout 10 --progress-bar)

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
"${CURL_DL[@]}" "https://github.com/${GH_REPO}/releases/latest/download/${ASSET}.tar.gz" -o "${TMP_ARCHIVE}"
tar -xzf "${TMP_ARCHIVE}" -C "${INSTALL_DIR}"
chmod +x "${INSTALL_DIR}/vocalai"

echo "==> Listing model artifacts from https://huggingface.co/${HF_REPO}..."
MODELS_DIR="${INSTALL_DIR}/models"
mkdir -p "${MODELS_DIR}"
RAW_FILE_LIST="$("${CURL_QUIET[@]}" "https://huggingface.co/api/models/${HF_REPO}" \
  | grep -o '"rfilename":"[^"]*"' \
  | sed -E 's/"rfilename":"([^"]*)"/\1/')"

if [ -z "${RAW_FILE_LIST}" ]; then
  echo "error: could not list files for ${HF_REPO} -- check the repo exists and is public" >&2
  exit 1
fi

# Filter out the files vocalai doesn't need to run before counting, so the [i/N] counter
# below reflects what's actually about to be downloaded.
FILES_TO_DOWNLOAD=()
while IFS= read -r f; do
  case "${f}" in
    README.md|.gitattributes|THIRD_PARTY_LICENSES) continue ;;  # not needed to run vocalai
  esac
  FILES_TO_DOWNLOAD+=("${f}")
done <<< "${RAW_FILE_LIST}"

TOTAL="${#FILES_TO_DOWNLOAD[@]}"
echo "==> Downloading ${TOTAL} model files (this can take a while -- HF Hub's anonymous tier is rate-limited, and some files are close to 2GB)"
i=0
for f in "${FILES_TO_DOWNLOAD[@]}"; do
  i=$((i + 1))
  mkdir -p "${MODELS_DIR}/$(dirname "${f}")"
  echo "==> [${i}/${TOTAL}] ${f}"
  "${CURL_DL[@]}" "https://huggingface.co/${HF_REPO}/resolve/main/${f}" -o "${MODELS_DIR}/${f}"
done

echo "==> Done. Run it with:"
echo "    ${INSTALL_DIR}/vocalai --text \"hello world\" --out out.wav --models-dir ${MODELS_DIR}"
