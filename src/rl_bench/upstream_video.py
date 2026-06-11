"""Record one final-policy video from upstream sac.pth after training."""

from pathlib import Path

from .upstream_agent import load_upstream_sac
from .video import record_policy_video


def maybe_record_upstream_videos(
    run_dir: Path,
    env,
    env_cfg: dict,
    mbrl_cfg,
    yaml_cfg: dict,
) -> None:
    train = yaml_cfg.get("train", {})
    if not train.get("video_every"):
        return
    ckpt = run_dir / "sac.pth"
    if not ckpt.exists():
        return
    agent = load_upstream_sac(env, mbrl_cfg, ckpt)
    path = record_policy_video(
        agent,
        {**env_cfg, "obs_norm": False},  # upstream policy is trained on raw obs
        run_dir / "videos",
        int(train["total_env_steps"]),
        n_episodes=int(train.get("video_episodes", 1)),
    )
    print(f"saved video: {path}")
