#!/usr/bin/env bash
# Archive prior MBPO runs, then train all seeds (default 0 1 2) with videos every 100k.
# Usage: bash scripts/run_mbpo_all_seeds.sh [seed ...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
cd "${ROOT}"

bash "${HERE}/archive_mbpo_runs.sh"
bash "${HERE}/train_mbpo.sh" "$@"
