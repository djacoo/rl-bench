"""Create Gymnasium envs and termination fns for upstream mbrl."""

from __future__ import annotations

from .upstream_path import ensure_upstream

ensure_upstream()
import mbrl.env.termination_fns as termination_fns

from .envs import make_env


def make_train_envs(yaml_cfg: dict, seed: int):
    env_cfg = yaml_cfg["env"]
    env = make_env(
        env_cfg["env_id"],
        seed=seed,
        max_episode_steps=env_cfg.get("max_episode_steps"),
        obs_norm=env_cfg.get("obs_norm", True),
        render_mode="human" if env_cfg.get("render", False) else None,
    )
    shared_rms = getattr(env, "rms", None)
    test_env = make_env(
        env_cfg["env_id"],
        seed=env_cfg["eval_seed"],
        eval=True,
        max_episode_steps=env_cfg.get("max_episode_steps"),
        obs_norm=env_cfg.get("obs_norm", True),
        shared_rms=shared_rms,
    )
    distance_env = make_env(
        env_cfg["env_id"],
        seed=env_cfg["eval_seed"] + 1,
        eval=True,
        max_episode_steps=env_cfg.get("max_episode_steps"),
        obs_norm=env_cfg.get("obs_norm", True),
        shared_rms=shared_rms,
    )
    return env, test_env, distance_env, shared_rms


def get_term_fn(mbrl_cfg):
    name = mbrl_cfg.overrides.term_fn
    return getattr(termination_fns, name)
