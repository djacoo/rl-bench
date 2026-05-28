#!/usr/bin/env bash
# Shared multi-seed launcher. Source this file and call run_seeds <algo> <config> [seeds...]
set -euo pipefail

ask_once() {
    # Prompt user once for device / live_plot / render; export env vars consumed by trainers.
    if [[ -z "${RL_BENCH_DEVICE:-}" ]]; then
        read -p "Device [cpu/mps/cuda, default cpu]: " ans
        export RL_BENCH_DEVICE="${ans:-cpu}"
    fi
    if [[ -z "${RL_BENCH_LIVE_PLOT:-}" ]]; then
        read -p "Live reward graph? [Y/n]: " ans
        case "${ans:-y}" in n|N|no|NO) export RL_BENCH_LIVE_PLOT=0 ;; *) export RL_BENCH_LIVE_PLOT=1 ;; esac
    fi
    if [[ -z "${RL_BENCH_RENDER:-}" ]]; then
        read -p "Live game viewer? [Y/n]: " ans
        case "${ans:-y}" in n|N|no|NO) export RL_BENCH_RENDER=0 ;; *) export RL_BENCH_RENDER=1 ;; esac
    fi
    echo "[setup] device=${RL_BENCH_DEVICE} live_plot=${RL_BENCH_LIVE_PLOT} render=${RL_BENCH_RENDER}"
}

run_seeds() {
    local algo="$1"
    local config="$2"
    shift 2
    local seeds=("$@")
    [[ ${#seeds[@]} -eq 0 ]] && seeds=(0 1 2)

    ask_once

    for seed in "${seeds[@]}"; do
        echo "[$(date +%H:%M:%S)] training ${algo} seed=${seed}"
        uv run python -m "rl_bench.train_${algo}" --config "${config}" --seed "${seed}"
    done
}
