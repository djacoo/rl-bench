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


def setup_device(spec: str = "auto") -> torch.device:
    device = resolve_device(spec)
    extra = ""
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        extra = f" ({torch.cuda.get_device_name(idx)})"
    print(f"Using device: {device}{extra} (config: {spec!r})")
    return device


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def dump_config(cfg: dict, run_dir) -> None:
    p = Path(run_dir) / "config.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
