#!/usr/bin/env bash
# Shared multi-seed launcher. Source this file and call run_seeds <algo> <config> [seeds...]
set -euo pipefail

run_seeds() {
    local algo="$1"
    local config="$2"
    shift 2
    local seeds=("$@")
    [[ ${#seeds[@]} -eq 0 ]] && seeds=(0 1 2)

    for seed in "${seeds[@]}"; do
        echo "[$(date +%H:%M:%S)] training ${algo} seed=${seed}"
        uv run python -m "rl_bench.train_${algo}" --config "${config}" --seed "${seed}"
    done
}
