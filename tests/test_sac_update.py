import numpy as np
import torch

from rl_bench.buffer import ReplayBuffer
from rl_bench.sac import SACAgent


def test_sac_update_finite_and_moves_actor_params():
    agent = SACAgent(obs_dim=4, act_dim=2, device="cpu", hidden=(32, 32))
    buf = ReplayBuffer(1000, 4, 2, "cpu")
    rng = np.random.default_rng(0)
    for _ in range(300):
        buf.add(
            rng.standard_normal(4).astype(np.float32),
            rng.uniform(-1, 1, 2).astype(np.float32),
            float(rng.standard_normal()),
            rng.standard_normal(4).astype(np.float32),
            False,
        )
    before = [p.detach().clone() for p in agent.actor.parameters()]
    losses = agent.update(buf.sample(64))
    for k, v in losses.items():
        assert np.isfinite(v), f"{k} not finite ({v})"
    moved = any(
        not torch.allclose(b, p) for b, p in zip(before, agent.actor.parameters())
    )
    assert moved


def test_sac_act_shape_and_range():
    agent = SACAgent(obs_dim=4, act_dim=2, device="cpu", hidden=(16, 16))
    obs = np.zeros(4, dtype=np.float32)
    a = agent.act(obs, deterministic=False)
    assert a.shape == (2,)
    assert (np.abs(a) <= 1.0 + 1e-6).all()
    a_det = agent.act(obs, deterministic=True)
    assert a_det.shape == (2,)


def test_sac_save_load_roundtrip(tmp_path):
    agent = SACAgent(obs_dim=4, act_dim=2, device="cpu", hidden=(16, 16))
    p = tmp_path / "ckpt.pt"
    agent.save(p)
    other = SACAgent(obs_dim=4, act_dim=2, device="cpu", hidden=(16, 16))
    other.load(p)
    for p1, p2 in zip(agent.actor.parameters(), other.actor.parameters()):
        assert torch.allclose(p1, p2)
