import numpy as np
import pytest

from rl_bench.buffer import ReplayBuffer
from rl_bench.envs import make_env
from rl_bench.sac import SACAgent


@pytest.mark.slow
def test_sac_smoke_pendulum_finishes_with_finite_losses():
    env = make_env("Pendulum-v1", seed=0, obs_norm=False)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_high = float(env.action_space.high[0])
    agent = SACAgent(obs_dim, act_dim, device="cpu", hidden=(64, 64))
    buf = ReplayBuffer(10_000, obs_dim, act_dim, "cpu")
    obs, _ = env.reset(seed=0)
    for t in range(2000):
        if t < 200:
            a = env.action_space.sample().astype(np.float32) / act_high
        else:
            a = agent.act(obs, deterministic=False)
        obs2, r, term, trunc, _ = env.step(a * act_high)
        buf.add(obs, a, r, obs2, bool(term))
        if term or trunc:
            obs, _ = env.reset()
        else:
            obs = obs2
        if t >= 200:
            losses = agent.update(buf.sample(64))
            for k, v in losses.items():
                assert np.isfinite(v), f"{k} not finite at step {t} ({v})"
