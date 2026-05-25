import numpy as np
import torch

from rl_bench.buffer import ReplayBuffer
from rl_bench.ensemble import ProbabilisticEnsemble


def _populate_linear_buffer(buf, n, obs_dim, act_dim, rng):
    for _ in range(n):
        s = rng.standard_normal(obs_dim).astype(np.float32)
        a = rng.uniform(-1, 1, act_dim).astype(np.float32)
        pad = np.zeros(obs_dim - act_dim) if obs_dim > act_dim else np.empty(0)
        s2 = s + 0.1 * np.concatenate([a, pad])[:obs_dim]
        r = float(-(s ** 2).sum())
        buf.add(s, a, r, s2, False)


def test_predict_all_shape():
    ens = ProbabilisticEnsemble(
        obs_dim=3, act_dim=2, n_members=4, n_elites=2, hidden=(16, 16), device="cpu"
    )
    s = torch.randn(5, 3)
    a = torch.randn(5, 2)
    mu, var = ens.predict_all(s, a)
    assert mu.shape == (4, 5, 4)
    assert var.shape == (4, 5, 4)
    assert (var > 0).all()


def test_predict_elite_uses_elite_idx():
    ens = ProbabilisticEnsemble(
        obs_dim=3, act_dim=2, n_members=4, n_elites=2, hidden=(16, 16), device="cpu"
    )
    ens.elite_idx = [1, 3]
    mu, _ = ens.predict_elite(torch.randn(5, 3), torch.randn(5, 2))
    assert mu.shape == (2, 5, 4)


def test_fit_runs_sets_elites_and_returns_finite_loss():
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(2000, 3, 2, "cpu")
    _populate_linear_buffer(buf, 1000, 3, 2, rng)
    ens = ProbabilisticEnsemble(
        obs_dim=3, act_dim=2, n_members=4, n_elites=2, hidden=(16, 16), device="cpu"
    )
    out = ens.fit(buf, max_samples=500, max_epochs=10, batch=128)
    assert len(ens.elite_idx) == 2
    assert np.isfinite(out["holdout_loss_best"])


def test_sample_from_shapes():
    ens = ProbabilisticEnsemble(
        obs_dim=3, act_dim=2, n_members=4, n_elites=2, hidden=(8, 8), device="cpu"
    )
    s = torch.randn(6, 3)
    a = torch.randn(6, 2)
    idx_subset = [0, 1]
    chosen = torch.zeros(6, dtype=torch.long)
    s_next, r = ens.sample_from(s, a, chosen, idx_subset)
    assert s_next.shape == (6, 3)
    assert r.shape == (6,)
