from pathlib import Path

import pytest

from rl_bench.cfg_bridge import build_mbrl_cfg
from rl_bench.utils import load_yaml


def test_macura_bridge_keeps_user_params():
    cfg = load_yaml(Path("configs/macura.yaml"))
    m = build_mbrl_cfg(cfg)
    assert m.overrides.num_steps == 500_000
    assert m.overrides.sac_batch_size == 128
    assert m.algorithm.initial_exploration_steps == 10_000
    assert m.overrides.xi == 2.0
    assert m.overrides.zeta == 95
    assert m.overrides.num_sac_updates_per_step == 8
    assert m.algorithm.name == "macura"


def test_mbpo_rollout_horizon_one():
    cfg = load_yaml(Path("configs/mbpo.yaml"))
    m = build_mbrl_cfg(cfg)
    assert m.algorithm.name == "mbpo"
    assert m.overrides.rollout_schedule == [1, 1_000_000, 1, 1]


def test_halfcheetah_uses_no_termination():
    cfg = load_yaml(Path("configs/mbpo_halfcheetah.yaml"))
    m = build_mbrl_cfg(cfg)
    assert m.overrides.term_fn == "no_termination"


def test_mbpo_rollout_batch_divisible_by_ensemble_size():
    cfg = load_yaml(Path("configs/mbpo.yaml"))
    m = build_mbrl_cfg(cfg)
    n = cfg["model"]["n_members"]
    batch = m.overrides.effective_model_rollouts_per_step * m.overrides.freq_train_model
    assert batch % n == 0
    assert m.overrides.freq_train_model == 252


def test_upstream_path_missing(tmp_path, monkeypatch):
    from rl_bench import upstream_path

    monkeypatch.setattr(upstream_path, "_VENDOR", tmp_path / "missing")
    with pytest.raises(RuntimeError, match="sync_macura_upstream"):
        upstream_path.ensure_upstream()
