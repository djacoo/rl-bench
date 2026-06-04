from pathlib import Path

import imageio.v3 as iio
import numpy as np

from .envs import make_env


def should_record_video(step: int, total: int, video_every: int | None) -> bool:
    """True when a rollout video should be saved at this env-step count."""
    if not video_every or step <= 0:
        return False
    if step % video_every == 0:
        return True
    return step == total


def _rollout_frames(agent, env, *, eval_seed: int, ep: int, deterministic: bool) -> list[np.ndarray]:
    obs, _ = env.reset(seed=eval_seed + ep)
    frames = [np.asarray(env.render(), dtype=np.uint8)]
    done = False
    while not done:
        action = agent.act(obs, deterministic=deterministic)
        obs, _r, term, trunc, _ = env.step(action)
        frames.append(np.asarray(env.render(), dtype=np.uint8))
        done = bool(term or trunc)
    return frames


def record_policy_video(
    agent,
    env_cfg: dict,
    out_dir: Path,
    step: int,
    *,
    shared_rms=None,
    n_episodes: int = 1,
    eval_seed: int | None = None,
    deterministic: bool = True,
    fps: int | None = None,
) -> Path:
    """Record deterministic policy rollouts to mp4 under out_dir (imageio, no MoviePy)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_seed = env_cfg["eval_seed"] if eval_seed is None else eval_seed
    prefix = f"step_{step:07d}"
    last_path: Path | None = None

    for ep in range(n_episodes):
        env = make_env(
            env_cfg["env_id"],
            seed=eval_seed + ep,
            eval=True,
            max_episode_steps=env_cfg.get("max_episode_steps"),
            obs_norm=env_cfg.get("obs_norm", True),
            shared_rms=shared_rms,
            render_mode="rgb_array",
        )
        render_fps = fps if fps is not None else int(env.metadata.get("render_fps", 50))
        name = prefix if n_episodes == 1 else f"{prefix}_ep{ep:02d}"
        out_path = out_dir / f"{name}.mp4"
        frames = _rollout_frames(
            agent, env, eval_seed=eval_seed, ep=ep, deterministic=deterministic
        )
        env.close()
        iio.imwrite(out_path, frames, fps=render_fps, codec="libx264")
        last_path = out_path

    assert last_path is not None
    return last_path
