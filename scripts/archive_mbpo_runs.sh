#!/usr/bin/env bash
# Move existing MBPO run dirs out of runs/ so a fresh multi-seed job starts clean.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

shopt -s nullglob
dirs=(runs/mbpo_seed*)
if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "no runs/mbpo_seed* directories to archive"
    exit 0
fi

stamp="$(date +%Y%m%d_%H%M%S)"
dest="runs/archive/${stamp}"
mkdir -p "${dest}"
for d in "${dirs[@]}"; do
    echo "archive ${d} -> ${dest}/"
    mv "${d}" "${dest}/"
done
echo "archived ${#dirs[@]} run(s) under ${dest}"
