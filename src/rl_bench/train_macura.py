import argparse
from pathlib import Path

from .cfg_bridge import build_mbrl_cfg
from .upstream_env import make_upstream_envs
from .upstream_path import ensure_upstream
from .upstream_video import maybe_record_upstream_videos
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
    setup_device(yaml_cfg.get("device", "auto"))

    run_dir = Path(yaml_cfg["paths"]["run_dir"].format(seed=seed))
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(yaml_cfg, run_dir)

    ensure_upstream()
    import mbrl.algorithms.macura as macura

    mbrl_cfg = build_mbrl_cfg(yaml_cfg)
    env, term_fn, (test_env, distance_env) = make_upstream_envs(mbrl_cfg, n_eval=2)

    print(f"Using device: {mbrl_cfg.device}")
    macura.train(env, test_env, distance_env, term_fn, mbrl_cfg, work_dir=str(run_dir))

    maybe_record_upstream_videos(run_dir, test_env, yaml_cfg["env"], mbrl_cfg, yaml_cfg)


if __name__ == "__main__":
    main()
