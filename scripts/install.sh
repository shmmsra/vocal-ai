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
  Darwin) ASSET="vocalai-macos"; BINARY_NAME="vocalai" ;;
  Linux) ASSET="vocalai-linux-cpu"; BINARY_NAME="vocalai" ;;
  *)
    echo "error: unsupported OS $(uname -s) -- see scripts/install.ps1 for Windows" >&2
    exit 1
    ;;
esac

echo "==> Installing vocalai (${ASSET}) into ${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"

MODELS_DIR="${INSTALL_DIR}/models"
mkdir -p "${MODELS_DIR}"

# Version-tracking files (VAI-012): a plain-text tag/version dropped into the install
# dir after a successful install/update, checked before the next run so an
# already-up-to-date binary/model set isn't re-downloaded. CLI_VERSION_FILE is our own
# (nothing inside the release archive carries a version marker). MODELS_VERSION_FILE
# reuses the name of a file scripts/publish_models.py already copies from this repo's
# root MODELS_VERSION into the HF Hub repo itself -- but it's deliberately written by
# this script only once the whole model set has downloaded successfully (see below),
# not fetched as just another file in the generic per-file loop, so a mid-download
# failure can't leave behind a stamp claiming a genuinely incomplete model set is current.
CLI_VERSION_FILE="${INSTALL_DIR}/.vocalai_version"
MODELS_VERSION_FILE="${MODELS_DIR}/MODELS_VERSION"

# Exits 1 with a clear message if the given HTTP status / response headers indicate
# this request was rate-limited -- a structured, unambiguous signal that falling back
# to the real (much larger) download would not help either. GitHub returns 403 with
# `x-ratelimit-remaining: 0` for its rate-limited endpoints (observed live against
# api.github.com during this feature's own testing); both GitHub and HF Hub can also
# return a plain 429. Any OTHER lookup failure (network hiccup, unexpected response
# shape) is handled by the caller, which also exits rather than silently falling back
# to downloading -- a version check that can't be trusted isn't a check at all.
fail_if_rate_limited() {
  local what="$1" status="$2" headers="$3"
  if [ "${status}" = "429" ] || { [ "${status}" = "403" ] && printf '%s' "${headers}" | grep -qi '^x-ratelimit-remaining: *0'; }; then
    echo "error: rate-limited while ${what} (HTTP ${status}) -- try again later" >&2
    exit 1
  fi
}

echo "==> Checking latest vocalai release..."
# Reads the tag straight off the "latest release" redirect's Location header -- the
# same URL the actual download below hits, just as a HEAD request -- rather than
# api.github.com, which carries its own much stingier unauthenticated rate limit
# (60/hour/IP) shared with unrelated traffic on that IP.
LATEST_URL="https://github.com/${GH_REPO}/releases/latest/download/${ASSET}.tar.gz"
LATEST_HEAD="$(curl -sI --retry 3 --retry-delay 2 --connect-timeout 10 "${LATEST_URL}" || true)"
LATEST_STATUS="$(printf '%s' "${LATEST_HEAD}" | awk 'NR==1{print $2}')"
fail_if_rate_limited "checking the latest vocalai release" "${LATEST_STATUS}" "${LATEST_HEAD}"

LATEST_TAG="$(printf '%s' "${LATEST_HEAD}" | grep -i '^location:' | sed -E 's#.*/releases/download/([^/]+)/.*#\1#' | tr -d '\r\n')"
if [ -z "${LATEST_TAG}" ]; then
  echo "error: could not determine the latest vocalai release tag from ${LATEST_URL} (HTTP ${LATEST_STATUS:-unknown}) -- check your network connection and that ${GH_REPO} has a published release" >&2
  exit 1
fi

if [ -f "${CLI_VERSION_FILE}" ] && [ -f "${INSTALL_DIR}/${BINARY_NAME}" ] \
  && [ "$(cat "${CLI_VERSION_FILE}")" = "${LATEST_TAG}" ]; then
  echo "==> vocalai binary is up to date (${LATEST_TAG}), skipping download"
else
  echo "==> Downloading latest release binary (${LATEST_TAG})..."
  TMP_ARCHIVE="$(mktemp -t vocalai-release.XXXXXX.tar.gz)"
  trap 'rm -f "${TMP_ARCHIVE}"' EXIT
  "${CURL_DL[@]}" "${LATEST_URL}" -o "${TMP_ARCHIVE}"
  tar -xzf "${TMP_ARCHIVE}" -C "${INSTALL_DIR}"
  chmod +x "${INSTALL_DIR}/${BINARY_NAME}"
  echo "${LATEST_TAG}" > "${CLI_VERSION_FILE}"
fi

echo "==> Checking latest model artifacts version..."
MODELS_VERSION_URL="https://huggingface.co/${HF_REPO}/resolve/main/MODELS_VERSION"
MODELS_VERSION_HEAD="$(curl -sI --retry 3 --retry-delay 2 --connect-timeout 10 "${MODELS_VERSION_URL}" || true)"
MODELS_VERSION_STATUS="$(printf '%s' "${MODELS_VERSION_HEAD}" | awk 'NR==1{print $2}')"
fail_if_rate_limited "checking the latest model artifacts version" "${MODELS_VERSION_STATUS}" "${MODELS_VERSION_HEAD}"

REMOTE_MODELS_VERSION="$("${CURL_QUIET[@]}" "${MODELS_VERSION_URL}")"
if [ -z "${REMOTE_MODELS_VERSION}" ]; then
  echo "error: could not determine the latest model artifacts version from ${MODELS_VERSION_URL}" >&2
  exit 1
fi

if [ -f "${MODELS_VERSION_FILE}" ] \
  && [ -n "$(find "${MODELS_DIR}" -name '*.onnx' -print -quit 2>/dev/null)" ] \
  && [ "$(cat "${MODELS_VERSION_FILE}")" = "${REMOTE_MODELS_VERSION}" ]; then
  echo "==> models are up to date (${REMOTE_MODELS_VERSION}), skipping model download"
else
  echo "==> Listing model artifacts from https://huggingface.co/${HF_REPO}..."
  API_URL="https://huggingface.co/api/models/${HF_REPO}"
  API_HEAD="$(curl -sI --retry 3 --retry-delay 2 --connect-timeout 10 "${API_URL}" || true)"
  API_STATUS="$(printf '%s' "${API_HEAD}" | awk 'NR==1{print $2}')"
  fail_if_rate_limited "listing model artifacts" "${API_STATUS}" "${API_HEAD}"

  RAW_FILE_LIST="$("${CURL_QUIET[@]}" "${API_URL}" \
    | grep -o '"rfilename":"[^"]*"' \
    | sed -E 's/"rfilename":"([^"]*)"/\1/')"

  if [ -z "${RAW_FILE_LIST}" ]; then
    echo "error: could not list files for ${HF_REPO} -- check the repo exists and is public" >&2
    exit 1
  fi

  # Filter out the files vocalai doesn't need to run before counting, so the [i/N] counter
  # below reflects what's actually about to be downloaded. MODELS_VERSION is deliberately
  # excluded from this generic loop too (see below): it must only be written once every
  # other file has downloaded successfully, not incidentally partway through, so a
  # `set -e`-triggered abort mid-loop can't leave behind a version marker for an
  # actually-incomplete model set.
  FILES_TO_DOWNLOAD=()
  while IFS= read -r f; do
    case "${f}" in
      README.md|.gitattributes|THIRD_PARTY_LICENSES|MODELS_VERSION) continue ;;  # handled separately / not needed to run vocalai
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

  # Written last, only once every file above succeeded (`set -e` aborts the script before
  # reaching this line on any download failure) -- the definitive "this model set is
  # complete" marker, not just a byproduct of MODELS_VERSION happening to be one of the
  # files HF's listing returned.
  echo "${REMOTE_MODELS_VERSION}" > "${MODELS_VERSION_FILE}"
fi

echo "==> Done. Run it with:"
echo "    ${INSTALL_DIR}/${BINARY_NAME} --text \"hello world\" --out out.wav --models-dir ${MODELS_DIR}"
