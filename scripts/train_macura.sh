#!/usr/bin/env bash
# Usage: bash scripts/train_macura.sh [seed ...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_train_common.sh"
run_seeds macura configs/macura.yaml "$@"
