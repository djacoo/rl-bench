"""Map rl-bench flat YAML configs to OmegaConf for upstream mbrl train()."""

from __future__ import annotations

import math

from omegaconf import DictConfig, OmegaConf

from .utils import load_yaml, resolve_device


def _zeta_percentile(z: float) -> int:
    return int(z * 100) if z <= 1.0 else int(z)


def _hidden_size(sac: dict) -> int:
    h = sac.get("hidden", 256)
    if isinstance(h, (list, tuple)):
        return int(h[0])
    return int(h)


def _exploration_env(kind: str) -> str:
    return {"pink": "pink", "white": "white", "deterministic": "det", "stochastic": "white"}[kind]


def _term_fn(env_id: str) -> str:
    name = env_id.lower().split("-")[0]
    mapping = {
        "hopper": "hopper",
        "halfcheetah": "no_termination",
        "walker2d": "walker2d",
        "walker": "walker2d",
        "ant": "ant",
        "humanoid": "humanoid",
    }
    return mapping.get(name, "no_termination")


def _model_hidden(model: dict) -> int:
    h = model.get("hidden", [200, 200, 200, 200])
    if isinstance(h, (list, tuple)):
        return int(h[0])
    return int(h)


def build_mbrl_cfg(yaml_cfg: dict) -> DictConfig:
    """Build OmegaConf for mbrl from rl-bench YAML (user values, not paper Hopper defaults)."""
    algo = yaml_cfg["algo"]
    env_cfg = yaml_cfg["env"]
    sac = yaml_cfg["sac"]
    train = yaml_cfg["train"]
    model = yaml_cfg.get("model", {})
    ex = yaml_cfg.get("exploration", {"kind": "pink"})

    device = resolve_device(yaml_cfg.get("device", "auto"))
    trial_length = int(env_cfg.get("max_episode_steps") or 1000)
    epoch_length = trial_length
    sac_hidden = _hidden_size(sac)
    updates = int(model.get("G_max", model.get("updates_G", 8)))

    n_models = int(model.get("n_members", 7))
    rollout_M = int(model.get("rollout_M", 400))
    refit_every = int(model.get("refit_every", 250))
    r = (rollout_M * refit_every) % n_models
    if r:
        g = math.gcd(rollout_M, n_models)
        refit_every += (n_models - r) // g

    overrides = {
        "env": f"gym___{env_cfg['env_id']}",
        "term_fn": _term_fn(env_cfg["env_id"]),
        "trial_length": trial_length,
        "num_steps": int(train["total_env_steps"]),
        "epoch_length": epoch_length,
        "improvement_threshold": 0.01,
        "num_epochs_train_model": None,
        "patience": 5,
        "model_lr": float(sac.get("lr_actor", 3e-4)),
        "model_wd": 1e-5,
        "model_batch_size": int(sac.get("batch_size", 256)),
        "model_hidden_size": _model_hidden(model),
        "validation_ratio": float(model.get("holdout_frac", 0.2)),
        "minimum_variance_exponent": -10,
        "freq_train_model": refit_every,
        "effective_model_rollouts_per_step": rollout_M,
        "num_sac_updates_per_step": updates,
        "sac_updates_every_steps": 1,
        "num_epochs_to_retain_sac_buffer": 1,
        "sac_buffer_size": int(sac.get("replay_capacity", 200_000)),
        "real_data_ratio": float(model.get("real_ratio", 0.1)),
        "sac_gamma": float(sac["gamma"]),
        "sac_tau": float(sac["tau"]),
        "sac_alpha": 0.2,
        "sac_policy": "Gaussian",
        "sac_target_update_interval": 1,
        "sac_automatic_entropy_tuning": True,
        "sac_target_entropy": None,
        "sac_hidden_size": sac_hidden,
        "sac_lr": float(sac.get("lr_actor", 3e-4)),
        "sac_batch_size": int(sac["batch_size"]),
        "exploration_type_env": _exploration_env(ex.get("kind", "pink")),
        "rollout_schedule": [1, 1_000_000, 1, 1],
    }

    if algo == "macura":
        overrides.update(
            {
                "max_rollout_length": int(model.get("T_max", 10)),
                "unc_tresh_run_avg_history": 2000,
                "pink_noise_exploration_mod": False,
                "xi": float(model.get("xi", 2.0)),
                "zeta": _zeta_percentile(float(model.get("zeta", 0.95))),
            }
        )

    algorithm = {
        "name": algo,
        "normalize": bool(env_cfg.get("obs_norm", False)),
        "normalize_double_precision": False,
        "target_is_delta": True,
        "learned_rewards": True,
        "freq_train_model": overrides["freq_train_model"],
        "real_data_ratio": overrides["real_data_ratio"],
        "critic_reset": False,
        "critic_reset_factor": 1.0,
        "critic_reset_every_step": 20000,
        "sac_samples_action": True,
        "initial_exploration_steps": int(sac.get("warmup_steps", 5000)),
        "random_initial_explore": False,
        "num_eval_episodes": int(train.get("eval_episodes", 30)),
        "dataset_size": int(sac.get("replay_capacity", 200_000)),
        "agent": {
            "num_inputs": "???",
            "action_space": {"low": "???", "high": "???", "shape": "???"},
            "args": {
                "layernorm": False,
                "gamma": overrides["sac_gamma"],
                "tau": overrides["sac_tau"],
                "alpha": overrides["sac_alpha"],
                "policy": overrides["sac_policy"],
                "target_update_interval": overrides["sac_target_update_interval"],
                "automatic_entropy_tuning": overrides["sac_automatic_entropy_tuning"],
                "target_entropy": overrides["sac_target_entropy"],
                "hidden_size": overrides["sac_hidden_size"],
                "device": str(device),
                "lr": overrides["sac_lr"],
            },
        },
    }
    if algo == "macura":
        algorithm["max_rollout_length"] = overrides["max_rollout_length"]

    dynamics_model = {
        "_target_": "mbrl.models.GaussianMLP",
        "device": str(device),
        "num_layers": len(model.get("hidden", [200, 200, 200, 200])),
        "hid_size": overrides["model_hidden_size"],
        "ensemble_size": int(model.get("n_members", 7)),
        "deterministic": False,
        "propagation_method": "random_model",
        "learn_logvar_bounds": False,
        "minimum_variance_exponent": overrides["minimum_variance_exponent"],
    }

    return OmegaConf.create(
        {
            "seed": int(yaml_cfg.get("seed", 0)),
            "device": str(device),
            "log_frequency_agent": int(train.get("log_every", 1000)),
            "save_video": False,
            "debug_mode": False,
            "algorithm": algorithm,
            "overrides": overrides,
            "dynamics_model": dynamics_model,
        }
    )


def build_mbrl_cfg_from_path(path) -> DictConfig:
    return build_mbrl_cfg(load_yaml(path))
