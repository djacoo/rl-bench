#!/usr/bin/env bash
# Clone/update reference copy of the MACURA paper repo (gitignored).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/vendor/macura-upstream"
REF="${MACURA_UPSTREAM_REF:-master}"
URL="https://github.com/Data-Science-in-Mechanical-Engineering/macura.git"

if [[ -d "${DEST}/.git" ]]; then
  git -C "${DEST}" fetch --depth 1 origin "${REF}"
  git -C "${DEST}" checkout "${REF}"
else
  mkdir -p "${ROOT}/vendor"
  git clone --depth 1 --branch "${REF}" "${URL}" "${DEST}"
fi
echo "Upstream MACURA at ${DEST} ($(git -C "${DEST}" rev-parse --short HEAD))"
