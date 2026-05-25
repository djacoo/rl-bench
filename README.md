# rl-bench

Academic implementations of three actor-critic algorithms applied to Gymnasium continuous-control tasks.
First environment: `LunarLanderContinuous-v3` (Box2D).

| Algorithm | Family                | Reference                                                       |
| --------- | --------------------- | --------------------------------------------------------------- |
| SAC       | Model-free            | Haarnoja et al., *Soft Actor-Critic Algorithms* (2018b)         |
| MBPO      | Model-based, Dyna     | Janner et al., *When to Trust Your Model* (NeurIPS 2019)        |
| MACURA    | Model-based, Dyna     | Frauenknecht et al., *Trust the Model Where It Trusts Itself* (ICML 2024) |

The MACURA paper PDF and the assignment brief are kept under `docs/`.

## Layout

```
configs/      # YAML hyperparameters, one per algorithm
src/rl_bench/ # algorithm + shared primitives (one file per algorithm)
scripts/      # bash launchers (multi-seed) and plotting
tests/        # pytest unit + smoke tests
docs/         # PDFs of paper + assignment (specs and plans are local-only, gitignored)
runs/         # tensorboard + jsonl per seed (gitignored)
results/      # aggregated plots (gitignored)
```

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

This creates `.venv/` and installs all dependencies (including dev).

## Running

Single seed:

```
uv run python -m rl_bench.train_sac    --config configs/sac.yaml    --seed 0
uv run python -m rl_bench.train_mbpo   --config configs/mbpo.yaml   --seed 0
uv run python -m rl_bench.train_macura --config configs/macura.yaml --seed 0
```

Multi-seed (bash launcher fans out three seeds):

```
bash scripts/train_sac.sh
bash scripts/train_mbpo.sh
bash scripts/train_macura.sh
```

## Plotting

```
uv run python scripts/plot_runs.py --algos sac mbpo macura --out results/learning_curves.png
```

## Tests

```
uv run pytest                 # unit tests
uv run pytest -m slow         # also runs the SAC-on-Pendulum smoke test
```

## Branching model

`main` (stable) — `develop` (integration) — `feature/*`, `hotfix/*`, `release/*` per gitflow.
