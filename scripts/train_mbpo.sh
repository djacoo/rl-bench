#!/usr/bin/env bash
# Usage: bash scripts/train_mbpo.sh [seed ...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_train_common.sh"
run_seeds mbpo configs/mbpo.yaml "$@"
