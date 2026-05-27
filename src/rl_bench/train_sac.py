import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .buffer import ReplayBuffer
from .envs import make_env
from .eval import evaluate
from .exploration import make_noise
from .live_plot import LivePlot
from .logger import Logger
from .sac import SACAgent
from .utils import dump_config, load_yaml, prompt_device, resolve_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = cfg["seed"]
    set_seed(seed)
    device_spec = prompt_device(cfg.get("device", "auto"))
    device = resolve_device(device_spec)
    cfg["device"] = str(device)
    run_dir = Path(cfg["paths"]["run_dir"].format(seed=seed))
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir)

    env_cfg = cfg["env"]
    env = make_env(
        env_cfg["env_id"],
        seed=seed,
        max_episode_steps=env_cfg.get("max_episode_steps"),
        obs_norm=env_cfg.get("obs_norm", True),
        render_mode="human" if env_cfg.get("render", False) else None,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    shared_rms = getattr(env, "rms", None)
    eval_env = make_env(
        env_cfg["env_id"],
        seed=env_cfg["eval_seed"],
        eval=True,
        max_episode_steps=env_cfg.get("max_episode_steps"),
        obs_norm=env_cfg.get("obs_norm", True),
        shared_rms=shared_rms,
    )

    sac_cfg = cfg["sac"]
    agent = SACAgent(
        obs_dim,
        act_dim,
        device,
        gamma=sac_cfg["gamma"],
        tau=sac_cfg["tau"],
        lr_actor=sac_cfg["lr_actor"],
        lr_critic=sac_cfg["lr_critic"],
        lr_alpha=sac_cfg["lr_alpha"],
        hidden=tuple(sac_cfg["hidden"]),
    )
    buf = ReplayBuffer(sac_cfg["replay_capacity"], obs_dim, act_dim, device)
    logger = Logger(run_dir)

    rng = np.random.default_rng(seed)
    ex_cfg = cfg.get("exploration", {"kind": "stochastic", "scale": 0.0})
    ep_len = env_cfg.get("max_episode_steps") or 1000
    noise = make_noise(ex_cfg["kind"], act_dim, ex_cfg["scale"], ep_len, rng)

    train_cfg = cfg["train"]
    total = train_cfg["total_env_steps"]
    warmup = sac_cfg["warmup_steps"]
    batch = sac_cfg["batch_size"]
    upd_per_step = train_cfg["updates_per_step"]
    log_every = train_cfg["log_every"]
    eval_every = train_cfg["eval_every"]
    ckpt_every = train_cfg["ckpt_every"]
    n_eval = train_cfg["eval_episodes"]
    live = bool(train_cfg.get("live_plot", True))
    lp = LivePlot(title=f"sac seed={seed}") if live else None

    obs, _ = env.reset(seed=seed)
    ep_ret = 0.0
    ep_len_acc = 0
    t0 = time.time()
    last_losses = {}
    last_ep_ret = float("nan")

    pbar = tqdm(range(total), desc=f"sac seed={seed}", dynamic_ncols=True, mininterval=0.5)
    for t in pbar:
        if t < warmup:
            a = env.action_space.sample().astype(np.float32)
        else:
            det = ex_cfg["kind"] == "deterministic"
            a = agent.act(obs, deterministic=det)
            if ex_cfg["kind"] in ("white", "pink"):
                a = np.clip(a + noise(t), -1.0, 1.0).astype(np.float32)

        next_obs, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, r, next_obs, bool(term))  # done = terminated only
        ep_ret += float(r)
        ep_len_acc += 1
        if term or trunc:
            logger.log_scalar("train/ep_return", ep_ret, t)
            logger.log_scalar("train/ep_len", ep_len_acc, t)
            if lp is not None:
                lp.add_train(t, ep_ret)
            last_ep_ret = ep_ret
            ep_ret = 0.0
            ep_len_acc = 0
            obs, _ = env.reset()
            noise.reset()
        else:
            obs = next_obs

        if t >= warmup:
            for _ in range(upd_per_step):
                last_losses = agent.update(buf.sample(batch))

        step1 = t + 1
        sps = step1 / (time.time() - t0)
        pbar.set_postfix(ep_ret=f"{last_ep_ret:.1f}", sps=f"{sps:.0f}", refresh=False)
        if step1 % log_every == 0 and last_losses:
            logger.log_dict({**last_losses, "train/sps": sps}, t)
        if step1 % eval_every == 0:
            _, _, rets = evaluate(
                agent,
                eval_env,
                n_episodes=n_eval,
                eval_seed=env_cfg["eval_seed"],
                deterministic=True,
            )
            logger.log_eval(step1, rets)
            if lp is not None:
                lp.add_eval(step1, float(np.mean(rets)))
        if step1 % ckpt_every == 0:
            agent.save(run_dir / f"ckpt_{step1}.pt")

    pbar.close()
    if lp is not None:
        lp.close()
    logger.close()


if __name__ == "__main__":
    main()
