import argparse
from pathlib import Path

from .cfg_bridge import build_mbrl_cfg
from .upstream_env import get_term_fn, make_train_envs
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
    import mbrl.algorithms.mbpo as mbpo

    mbrl_cfg = build_mbrl_cfg(yaml_cfg)
    env, test_env, _, shared_rms = make_train_envs(yaml_cfg, seed)
    term_fn = get_term_fn(mbrl_cfg)

    print(f"Using device: {mbrl_cfg.device}")
    mbpo.train(env, test_env, term_fn, mbrl_cfg, work_dir=str(run_dir))

    maybe_record_upstream_videos(
        run_dir, test_env, yaml_cfg["env"], mbrl_cfg, yaml_cfg, shared_rms
    )


if __name__ == "__main__":
    main()
