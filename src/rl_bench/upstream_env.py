"""Build MBPO/MACURA envs in the exact shape upstream's state ops expect.

Upstream saves/restores raw MuJoCo state via ``env.env.data`` — i.e. a single
``TimeLimit`` wrapper over the raw env. Modern gymnasium's ``gym.make`` adds extra
wrappers (``OrderEnforcing``/``PassiveEnvChecker``) that break that shallow access,
so we strip to the raw env and re-wrap with ``TimeLimit``, matching upstream's own
mujoco_envs path. Upstream owns observation normalization
(``cfg.algorithm.normalize``), so these envs stay un-normalized.
"""

import gymnasium as gym

from .upstream_path import ensure_upstream


def _make_one(env_id, max_steps, seed):
    env = gym.wrappers.TimeLimit(gym.make(env_id).unwrapped, max_episode_steps=max_steps)
    env.reset(seed=seed)
    env.observation_space.seed(seed + 1)
    env.action_space.seed(seed + 2)
    return env


def make_upstream_envs(mbrl_cfg, n_eval=1):
    ensure_upstream()
    import mbrl.env.termination_fns as termination_fns

    env_id = mbrl_cfg.overrides.env.split("___")[1]
    max_steps = int(mbrl_cfg.overrides.get("trial_length", 1000))
    seed = int(mbrl_cfg.seed)

    env = _make_one(env_id, max_steps, seed)
    eval_envs = [_make_one(env_id, max_steps, seed) for _ in range(n_eval)]
    term_fn = getattr(termination_fns, mbrl_cfg.overrides.term_fn)
    return env, term_fn, eval_envs
