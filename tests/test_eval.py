import numpy as np

from rl_bench.envs import make_env
from rl_bench.eval import evaluate


class _RandomAgent:
    def __init__(self, act_dim, seed=0):
        self.rng = np.random.default_rng(seed)
        self.act_dim = act_dim

    def act(self, obs, deterministic=True):
        return self.rng.uniform(-1, 1, self.act_dim).astype(np.float32)


def test_evaluate_returns_expected_shape():
    env = make_env("Pendulum-v1", seed=0, obs_norm=False)
    agent = _RandomAgent(env.action_space.shape[0])
    m, s, rets = evaluate(agent, env, n_episodes=3, eval_seed=42)
    assert len(rets) == 3
    assert isinstance(m, float)
    assert isinstance(s, float)
