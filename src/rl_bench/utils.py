import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prompt_yes_no(question: str, default: bool = True, env_var: str | None = None) -> bool:
    """Y/N prompt. Honors env_var if set; skips if non-tty (returns default)."""
    import os
    import sys

    if env_var and env_var in os.environ:
        return os.environ[env_var].strip().lower() in ("1", "true", "yes", "y")
    if not sys.stdin.isatty():
        return default
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{question} {suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Pick y or n.")


def prompt_device(cfg_device: str = "auto", env_var: str = "RL_BENCH_DEVICE") -> str:
    """Interactive device picker. Honors env_var; skips if non-tty (CI, piped, bash subshell)."""
    import os
    import sys

    if env_var and env_var in os.environ:
        return os.environ[env_var]
    if not sys.stdin.isatty():
        return cfg_device
    options = ["cpu"]
    if torch.backends.mps.is_available():
        options.append("mps")
    if torch.cuda.is_available():
        options.append("cuda")
    if len(options) == 1:
        return options[0]
    default_idx = options.index(cfg_device) if cfg_device in options else 0
    print("\nDevice:")
    for i, opt in enumerate(options):
        marker = " (default)" if i == default_idx else ""
        print(f"  [{i}] {opt}{marker}")
    while True:
        choice = input(f"Pick [0-{len(options) - 1}], Enter for default: ").strip()
        if not choice:
            return options[default_idx]
        try:
            idx = int(choice)
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print(f"Invalid. Pick 0-{len(options) - 1}.")


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def dump_config(cfg: dict, run_dir) -> None:
    p = Path(run_dir) / "config.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
