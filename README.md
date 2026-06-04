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
![License](https://img.shields.io/badge/license-MIT-green)

## Upstream MACURA code

MBPO, MACURA, and SAC training call into the reference [`mbrl`](https://github.com/Data-Science-in-Mechanical-Engineering/macura) package from the paper repo. It is **not** vendored in git; clone it locally:

```bash
bash scripts/sync_macura_upstream.sh   # → vendor/macura-upstream/ (gitignored)
```

[`cfg_bridge.py`](src/rl_bench/cfg_bridge.py) maps this repo’s flat YAML configs to upstream OmegaConf (your hyperparameters, not the paper’s default Hydra overrides). [`upstream_path.py`](src/rl_bench/upstream_path.py) adds the clone to `sys.path` at import time.

## Layout

```
rl-bench/
├─ configs/              # YAML per algorithm × env
│   ├─ *_hopper.yaml     # Hopper-v4, 500k steps (sac/mbpo/macura defaults)
│   └─ *_halfcheetah.yaml
├─ scripts/
│   ├─ sync_macura_upstream.sh
│   ├─ train_*.sh
│   └─ plot_runs.py
├─ src/rl_bench/
│   ├─ cfg_bridge.py, upstream_*.py
│   ├─ envs.py, exploration.py, eval.py, logger.py, video.py
│   └─ train_sac.py, train_mbpo.py, train_macura.py
├─ vendor/macura-upstream/  # after sync; gitignored
├─ runs/                 # per-seed checkpoints/logs (gitignored)
└─ results/              # aggregated summaries / plots
```

## Reproduce

```bash
git clone <repo-url> rl-bench && cd rl-bench
uv sync
bash scripts/sync_macura_upstream.sh
```

Set `device: auto` (or `cuda`) in the YAML you use. Check GPU:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Hopper (default)

```bash
uv run python -m rl_bench.train_sac    --config configs/sac.yaml    --seed 0
uv run python -m rl_bench.train_mbpo   --config configs/mbpo.yaml   --seed 0
uv run python -m rl_bench.train_macura --config configs/macura.yaml --seed 0
```

### HalfCheetah

```bash
uv run python -m rl_bench.train_sac    --config configs/sac_halfcheetah.yaml    --seed 0
uv run python -m rl_bench.train_mbpo   --config configs/mbpo_halfcheetah.yaml   --seed 0
uv run python -m rl_bench.train_macura --config configs/macura_halfcheetah.yaml --seed 0
```

Multi-seed (`0 1 2` by default): `bash scripts/train_sac.sh` (and mbpo/macura).

Runs are headless by default (`env.render: false`). Optional MP4 rollouts: `train.video_every` in YAML; saved under `runs/.../videos/` via [`video.py`](src/rl_bench/video.py) from `sac.pth`.

Change `paths.run_dir` in YAML if you need a new experiment folder and must not overwrite an existing `runs/*_seed0/`.

Aggregate learning curves:

```bash
uv run python scripts/plot_runs.py --algos sac mbpo macura --out results/learning_curves.png
```

TensorBoard:

```bash
uv run tensorboard --logdir runs/
```
