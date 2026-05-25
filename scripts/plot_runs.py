import argparse
import csv
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_eval(run_dir):
    p = Path(run_dir) / "eval.csv"
    if not p.exists():
        return None
    steps, means, stds = [], [], []
    with open(p) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            means.append(float(row["mean"]))
            stds.append(float(row["std"]))
    return np.array(steps), np.array(means), np.array(stds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--algos", nargs="+", required=True)
    ap.add_argument("--out", default="results/learning_curves.png")
    args = ap.parse_args()

    plt.figure(figsize=(8, 5))
    for algo in args.algos:
        seed_dirs = sorted(glob.glob(f"{args.runs_dir}/{algo}_seed*"))
        if not seed_dirs:
            print(f"no runs found for {algo}")
            continue
        per_seed = []
        steps_ref = None
        for sd in seed_dirs:
            res = load_eval(sd)
            if res is None:
                continue
            steps, means, _ = res
            if steps_ref is None:
                steps_ref = steps
            per_seed.append(means)
        if not per_seed:
            continue
        n_min = min(len(m) for m in per_seed)
        arr = np.stack([m[:n_min] for m in per_seed], axis=0)
        mean = arr.mean(0)
        std = arr.std(0)
        x = steps_ref[:n_min]
        plt.plot(x, mean, label=algo)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel("env steps")
    plt.ylabel("eval return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
