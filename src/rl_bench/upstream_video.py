"""Record videos from upstream sac.pth after training."""

from pathlib import Path

from .upstream_agent import load_upstream_sac
from .video import record_policy_video, should_record_video


def maybe_record_upstream_videos(
    run_dir: Path,
    env,
    env_cfg: dict,
    mbrl_cfg,
    yaml_cfg: dict,
    shared_rms=None,
) -> None:
    train = yaml_cfg.get("train", {})
    video_every = train.get("video_every")
    if not video_every:
        return
    ckpt = run_dir / "sac.pth"
    if not ckpt.exists():
        return
    total = int(train["total_env_steps"])
    steps = [s for s in range(1, total + 1) if should_record_video(s, total, video_every)]
    if not steps:
        return
    agent = load_upstream_sac(env, mbrl_cfg, ckpt)
    video_dir = run_dir / "videos"
    n_ep = int(train.get("video_episodes", 1))
    for step in steps:
        path = record_policy_video(
            agent,
            env_cfg,
            video_dir,
            step,
            shared_rms=shared_rms,
            n_episodes=n_ep,
        )
        print(f"saved video: {path}")
