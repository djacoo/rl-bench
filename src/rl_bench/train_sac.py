import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .cfg_bridge import build_mbrl_cfg
from .envs import make_sac_envs
from .eval import evaluate
from .exploration import make_noise
from .upstream_agent import UpstreamSacAdapter
from .upstream_path import ensure_upstream
from .video import record_policy_video, should_record_video
from .logger import Logger
from .utils import dump_config, load_yaml, set_seed, setup_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    yaml_cfg = load_yaml(args.config)
    if args.seed is not None:
        yaml_cfg["seed"] = args.seed
    seed = yaml_cfg["seed"]
    set_seed(seed)
    device = setup_device(yaml_cfg.get("device", "auto"))

    run_dir = Path(yaml_cfg["paths"]["run_dir"].format(seed=seed))
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(yaml_cfg, run_dir)

    ensure_upstream()
    import mbrl.planning
    import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac
    from mbrl.planning.sac_wrapper import SACAgent
    from mbrl.util.replay_buffer import ReplayBuffer as MbrlReplayBuffer

    mbrl_cfg = build_mbrl_cfg(yaml_cfg)
    env, eval_env, shared_rms = make_sac_envs(yaml_cfg, seed)
    mbrl.planning.complete_agent_cfg(env, mbrl_cfg.algorithm.agent)
    sac = pytorch_sac.SAC(
        mbrl_cfg.algorithm.agent.num_inputs,
        env.action_space,
        mbrl_cfg.algorithm.agent.args,
    )
    agent = SACAgent(sac)
    policy = UpstreamSacAdapter(agent)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    rng = np.random.default_rng(seed)
    memory = MbrlReplayBuffer(
        mbrl_cfg.overrides.sac_buffer_size,
        obs_shape,
        act_shape,
        rng=rng,
    )

    sac_cfg = yaml_cfg["sac"]
    train_cfg = yaml_cfg["train"]
    env_cfg = yaml_cfg["env"]
    ex_cfg = yaml_cfg.get("exploration", {"kind": "stochastic", "scale": 0.0})
    act_dim = env.action_space.shape[0]
    ep_len = env_cfg.get("max_episode_steps") or 1000
    noise = make_noise(ex_cfg["kind"], act_dim, ex_cfg["scale"], ep_len, rng)

    total = int(train_cfg["total_env_steps"])
    warmup = int(sac_cfg["warmup_steps"])
    batch = int(sac_cfg["batch_size"])
    upd_per_step = int(train_cfg.get("updates_per_step", 1))
    logger = Logger(run_dir)
    updates = 0

    video_every = train_cfg.get("video_every")
    video_episodes = int(train_cfg.get("video_episodes", 1))
    video_dir = run_dir / "videos"

    obs, _ = env.reset(seed=seed)
    ep_ret = 0.0
    t0 = time.time()
    last_ep_ret = float("nan")

    pbar = tqdm(range(total), desc=f"sac seed={seed}", dynamic_ncols=True, mininterval=0.5)
    for t in pbar:
        if t < warmup:
            a = env.action_space.sample().astype(np.float32)
        else:
            det = ex_cfg["kind"] == "deterministic"
            a = policy.act(obs, deterministic=det)
            if ex_cfg["kind"] in ("white", "pink"):
                a = np.clip(a + noise(t), -1.0, 1.0).astype(np.float32)

        next_obs, r, term, trunc, _ = env.step(a)
        memory.add(obs, a, next_obs, float(r), bool(term), bool(trunc))

        if term or trunc:
            last_ep_ret = ep_ret + float(r)
            ep_ret = 0.0
            obs, _ = env.reset()
            noise.reset()
        else:
            ep_ret += float(r)
            obs = next_obs

        if t >= warmup and len(memory) >= batch:
            for _ in range(upd_per_step):
                agent.sac_agent.update_parameters(
                    memory, batch, updates, logger=None, reverse_mask=True
                )
                updates += 1

        step1 = t + 1
        sps = step1 / (time.time() - t0)
        pbar.set_postfix(ep_ret=f"{last_ep_ret:.1f}", sps=f"{sps:.0f}", refresh=False)

        if step1 % train_cfg["eval_every"] == 0:
            _, _, rets = evaluate(
                policy,
                eval_env,
                n_episodes=train_cfg["eval_episodes"],
                eval_seed=env_cfg["eval_seed"],
                deterministic=True,
            )
            logger.log_eval(step1, rets)

        if step1 % train_cfg["ckpt_every"] == 0:
            agent.sac_agent.save_checkpoint(ckpt_path=str(run_dir / f"ckpt_{step1}.pt"))

        if should_record_video(step1, total, video_every):
            path = record_policy_video(
                policy,
                env_cfg,
                video_dir,
                step1,
                shared_rms=shared_rms,
                n_episodes=video_episodes,
            )
            tqdm.write(f"saved video: {path}")

    agent.sac_agent.save_checkpoint(ckpt_path=str(run_dir / "sac.pth"))
    pbar.close()
    logger.close()
    print(f"Using device: {device}")


if __name__ == "__main__":
    main()
