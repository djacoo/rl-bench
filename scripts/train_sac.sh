#!/usr/bin/env bash
# Usage: bash scripts/train_sac.sh [seed ...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_train_common.sh"
run_seeds sac configs/sac.yaml "$@"
