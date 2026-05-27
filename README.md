# rl-bench

Three actor-critic algorithms on Gymnasium MuJoCo continuous-control environments.

| Algorithm | Idea |
|-----------|------|
| **SAC**    | Maximum-entropy off-policy actor-critic with twin Q-functions and learned temperature. |
| **MBPO**   | Dyna with a probabilistic dynamics ensemble; short branched rollouts augment the replay buffer. |
| **MACURA** | MBPO with a per-step uncertainty gate (GJS divergence) and adaptive rollout length / update count. |

## Stack

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.4-EE4C2C?logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-MuJoCo-0081A5)
![uv](https://img.shields.io/badge/uv-package%20mgr-DE5FE9)
![TensorBoard](https://img.shields.io/badge/TensorBoard-logs-FF6F00?logo=tensorflow&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-tests-0A9EDC?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

## Layout

```
rl-bench/
├─ configs/         # one YAML per algorithm
├─ src/rl_bench/
│   ├─ envs.py      # gym wrapper + running obs normalizer
│   ├─ buffer.py    # replay buffer (env + model)
│   ├─ nets.py      # MLP, GaussianTanhPolicy, QNet, ProbEnsembleMember
│   ├─ ensemble.py  # probabilistic ensemble + GJS divergence
│   ├─ exploration.py  # stochastic / white / pink noise
│   ├─ sac.py       # SAC agent
│   ├─ train_sac.py
│   ├─ train_mbpo.py
│   ├─ train_macura.py
│   ├─ eval.py
│   ├─ live_plot.py # interactive learning-curve window
│   ├─ logger.py    # TB + jsonl + csv
│   └─ utils.py
├─ scripts/         # bash launchers + plot_runs.py
├─ tests/
├─ runs/            # per-seed artifacts (gitignored)
└─ results/         # aggregated plots (gitignored)
```

## Reproduce

```bash
git clone <repo-url> rl-bench && cd rl-bench
uv sync
```

Single seed:

```bash
uv run python -m rl_bench.train_sac    --config configs/sac.yaml    --seed 0
uv run python -m rl_bench.train_mbpo   --config configs/mbpo.yaml   --seed 0
uv run python -m rl_bench.train_macura --config configs/macura.yaml --seed 0
```

Multi-seed (`0 1 2` by default):

```bash
bash scripts/train_sac.sh
bash scripts/train_mbpo.sh
bash scripts/train_macura.sh
```

Default configs now target `Hopper-v4` (MuJoCo). Runs are headless by default (`env.render: false`, `train.live_plot: false`). Enable rendering/live curves in YAML only when needed.

Aggregate seeds into one figure:

```bash
uv run python scripts/plot_runs.py --algos sac mbpo macura --out results/learning_curves.png
```

TensorBoard:

```bash
uv run tensorboard --logdir runs/
```

## Tests

```bash
uv run pytest                 # unit tests
uv run pytest -m slow         # also the 2k-step SAC smoke on Pendulum-v1
```
