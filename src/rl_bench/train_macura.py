import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .buffer import ReplayBuffer
from .ensemble import ProbabilisticEnsemble
from .envs import make_env
from .eval import evaluate
from .exploration import make_noise
from .logger import Logger
from .sac import SACAgent
from .utils import dump_config, load_yaml, resolve_device, set_seed


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
    device = resolve_device(cfg.get("device", "auto"))
    run_dir = Path(cfg["paths"]["run_dir"].format(seed=seed))
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir)

    env_cfg = cfg["env"]
    env = make_env(
        env_cfg["env_id"], seed=seed,
        max_episode_steps=env_cfg.get("max_episode_steps"),
        obs_norm=env_cfg.get("obs_norm", True),
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    shared_rms = getattr(env, "rms", None)
    eval_env = make_env(
        env_cfg["env_id"], seed=env_cfg["eval_seed"], eval=True,
        max_episode_steps=env_cfg.get("max_episode_steps"),
        obs_norm=env_cfg.get("obs_norm", True), shared_rms=shared_rms,
    )

    sac_cfg = cfg["sac"]
    mod_cfg = cfg["model"]
    agent = SACAgent(
        obs_dim, act_dim, device,
        gamma=sac_cfg["gamma"], tau=sac_cfg["tau"],
        lr_actor=sac_cfg["lr_actor"], lr_critic=sac_cfg["lr_critic"],
        lr_alpha=sac_cfg["lr_alpha"], hidden=tuple(sac_cfg["hidden"]),
    )
    d_env = ReplayBuffer(sac_cfg["replay_capacity"], obs_dim, act_dim, device)
    d_mod = ReplayBuffer(mod_cfg["model_buf_capacity"], obs_dim, act_dim, device)
    ensemble = ProbabilisticEnsemble(
        obs_dim, act_dim,
        n_members=mod_cfg["n_members"], n_elites=mod_cfg["n_elites"],
        hidden=tuple(mod_cfg["hidden"]), device=device,
    )
    logger = Logger(run_dir)

    rng = np.random.default_rng(seed)
    ex_cfg = cfg["exploration"]
    ep_len = env_cfg.get("max_episode_steps") or 1000
    noise = make_noise(ex_cfg["kind"], act_dim, ex_cfg["scale"], ep_len, rng)

    train_cfg = cfg["train"]
    total = train_cfg["total_env_steps"]
    warmup = sac_cfg["warmup_steps"]
    batch = sac_cfg["batch_size"]
    refit_every = mod_cfg["refit_every"]
    rollout_M = mod_cfg["rollout_M"]
    T_max = mod_cfg["T_max"]
    zeta = mod_cfg["zeta"]
    xi = mod_cfg["xi"]
    real_ratio = mod_cfg["real_ratio"]
    G_max = mod_cfg["G_max"]
    max_refit = mod_cfg["max_refit_samples"]
    holdout_frac = mod_cfg["holdout_frac"]
    n_members = mod_cfg["n_members"]
    d_mod_cap = mod_cfg["model_buf_capacity"]
    log_every = train_cfg["log_every"]
    eval_every = train_cfg["eval_every"]
    ckpt_every = train_cfg["ckpt_every"]
    n_eval = train_cfg["eval_episodes"]
    n_real = int(real_ratio * batch)
    n_mod = batch - n_real

    # Adaptive uncertainty threshold (Frauenknecht 2024 eq 21).
    kappa_K = 0
    kappa_sum = 0.0
    kappa = float("inf")

    obs, _ = env.reset(seed=seed)
    ep_ret = 0.0
    ep_len_acc = 0
    t0 = time.time()
    last_losses = {}

    for t in range(total):
        if t < warmup:
            a = env.action_space.sample().astype(np.float32)
        else:
            det = ex_cfg["kind"] == "deterministic"
            a = agent.act(obs, deterministic=det)
            if ex_cfg["kind"] in ("white", "pink"):
                a = np.clip(a + noise(t), -1.0, 1.0).astype(np.float32)

        next_obs, r, term, trunc, _ = env.step(a)
        d_env.add(obs, a, r, next_obs, bool(term))
        ep_ret += float(r)
        ep_len_acc += 1
        if term or trunc:
            logger.log_scalar("train/ep_return", ep_ret, t)
            logger.log_scalar("train/ep_len", ep_len_acc, t)
            ep_ret = 0.0
            ep_len_acc = 0
            obs, _ = env.reset()
            noise.reset()
        else:
            obs = next_obs

        if t >= warmup and ((t - warmup) % refit_every == 0):
            fit_stats = ensemble.fit(
                d_env, max_samples=max_refit, holdout_frac=holdout_frac
            )
            logger.log_dict({f"model/{k}": v for k, v in fit_stats.items()}, t)

            s = d_env.sample_states(rollout_M)
            alive = torch.ones(rollout_M, dtype=torch.bool, device=device)
            all_idx = list(range(n_members))
            kept_total = 0
            for h in range(T_max):
                idx_alive = torch.nonzero(alive, as_tuple=False).squeeze(-1)
                if idx_alive.numel() == 0:
                    break
                sa = s[idx_alive]
                with torch.no_grad():
                    a_pi, _ = agent.actor.sample(sa)
                    mu, var = ensemble.predict_all(sa, a_pi)
                    u = ensemble.gjs_uncertainty(mu, var)
                    if h == 0:
                        u_base = float(torch.quantile(u, zeta).item())
                        kappa_K += 1
                        kappa_sum += u_base
                        kappa = xi * kappa_sum / kappa_K
                        logger.log_scalar("macura/kappa", kappa, t)
                        logger.log_scalar("macura/u_base", u_base, t)
                    e_idx = torch.randint(
                        0, n_members, (sa.shape[0],), device=device
                    )
                    s_next, rew = ensemble.sample_from(sa, a_pi, e_idx, all_idx)
                    keep = u < kappa
                if bool(keep.any()):
                    n_keep = int(keep.sum().item())
                    d_mod.add_batch(
                        sa[keep].cpu().numpy(),
                        a_pi[keep].cpu().numpy(),
                        rew[keep].cpu().numpy(),
                        s_next[keep].cpu().numpy(),
                        np.zeros(n_keep, dtype=np.float32),
                    )
                    kept_total += n_keep
                # propagate only kept rollouts forward; drop the rest
                kept_global = idx_alive[keep]
                dropped_global = idx_alive[~keep]
                s = s.clone()
                s[kept_global] = s_next[keep]
                alive = alive.clone()
                alive[dropped_global] = False
            logger.log_scalar("macura/rollout_kept", float(kept_total), t)

        if t >= warmup:
            G = max(1, int(G_max * len(d_mod) / d_mod_cap))
            for _ in range(G):
                if len(d_mod) < n_mod:
                    last_losses = agent.update(d_env.sample(batch))
                else:
                    br = d_env.sample(n_real)
                    bm = d_mod.sample(n_mod)
                    mixed = {k: torch.cat([br[k], bm[k]], dim=0) for k in br}
                    last_losses = agent.update(mixed)

        step1 = t + 1
        if step1 % log_every == 0 and last_losses:
            sps = step1 / (time.time() - t0)
            logger.log_dict(
                {
                    **last_losses,
                    "train/sps": sps,
                    "buf/d_env": float(len(d_env)),
                    "buf/d_mod": float(len(d_mod)),
                    "macura/G": float(max(1, int(G_max * len(d_mod) / d_mod_cap))),
                },
                t,
            )
        if step1 % eval_every == 0:
            _, _, rets = evaluate(
                agent, eval_env, n_episodes=n_eval,
                eval_seed=env_cfg["eval_seed"], deterministic=True,
            )
            logger.log_eval(step1, rets)
        if step1 % ckpt_every == 0:
            agent.save(run_dir / f"ckpt_{step1}.pt")

    logger.close()


if __name__ == "__main__":
    main()
