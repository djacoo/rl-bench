import gymnasium as gym
import numpy as np


class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None]
        batch_mean = x.mean(0)
        batch_var = x.var(0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)


class ObsNormalizer(gym.ObservationWrapper):
    def __init__(self, env, update_stats=True):
        super().__init__(env)
        self.rms = RunningMeanStd(env.observation_space.shape)
        self.update_stats = update_stats

    def observation(self, obs):
        arr = np.asarray(obs, dtype=np.float64)
        if self.update_stats:
            self.rms.update(arr)
        return self.rms.normalize(arr).astype(np.float32)


def make_sac_envs(yaml_cfg, seed):
    """Train + eval envs for the custom SAC loop (shared obs-norm stats)."""
    c = yaml_cfg["env"]
    train = make_env(
        c["env_id"], seed=seed,
        max_episode_steps=c.get("max_episode_steps"),
        obs_norm=c.get("obs_norm", True),
        render_mode="human" if c.get("render", False) else None,
    )
    shared_rms = getattr(train, "rms", None)
    eval_env = make_env(
        c["env_id"], seed=c["eval_seed"], eval=True,
        max_episode_steps=c.get("max_episode_steps"),
        obs_norm=c.get("obs_norm", True),
        shared_rms=shared_rms,
    )
    return train, eval_env, shared_rms


def make_env(env_id, seed, eval=False, max_episode_steps=None, obs_norm=True, shared_rms=None, render_mode=None):
    kwargs = {}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    env = gym.make(env_id, **kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if obs_norm:
        env = ObsNormalizer(env, update_stats=not eval)
        if shared_rms is not None:
            env.rms = shared_rms
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
