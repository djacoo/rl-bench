import numpy as np
import torch

from rl_bench.utils import dump_config, load_yaml, resolve_device, set_seed


def test_set_seed_reproducible_numpy():
    set_seed(42)
    a = np.random.randn(3)
    set_seed(42)
    b = np.random.randn(3)
    assert np.array_equal(a, b)


def test_set_seed_reproducible_torch():
    set_seed(7)
    a = torch.randn(3)
    set_seed(7)
    b = torch.randn(3)
    assert torch.equal(a, b)


def test_resolve_device_explicit_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_returns_torch_device():
    d = resolve_device("auto")
    assert isinstance(d, torch.device)


def test_load_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\nb:\n  - 1\n  - 2\n")
    cfg = load_yaml(p)
    assert cfg == {"a": 1, "b": [1, 2]}


def test_dump_config_writes_file(tmp_path):
    dump_config({"x": 1, "y": "z"}, tmp_path)
    f = tmp_path / "config.yaml"
    assert f.exists()
    assert "x: 1" in f.read_text()
