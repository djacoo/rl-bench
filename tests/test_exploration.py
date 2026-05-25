import numpy as np
import pytest

from rl_bench.exploration import make_noise


def test_stochastic_returns_zero():
    n = make_noise("stochastic", 3, 0.5, 100, None)
    assert float(np.asarray(n(0)).sum()) == 0.0
    n.reset()


def test_deterministic_returns_zero():
    n = make_noise("deterministic", 3, 0.5, 100, None)
    assert float(np.asarray(n(0)).sum()) == 0.0


def test_white_noise_shape_and_scale():
    rng = np.random.default_rng(0)
    n = make_noise("white", 3, 0.1, 100, rng)
    samples = np.stack([n(t) for t in range(1000)])
    assert samples.shape == (1000, 3)
    assert abs(samples.std() - 0.1) < 0.05


def test_pink_noise_shape_per_step():
    rng = np.random.default_rng(0)
    n = make_noise("pink", 2, 0.1, 50, rng)
    for t in range(50):
        x = n(t)
        assert np.asarray(x).shape == (2,)


def test_pink_noise_resets_on_episode_end():
    rng = np.random.default_rng(0)
    n = make_noise("pink", 2, 0.1, 5, rng)
    first = [n(t).copy() for t in range(5)]
    n.reset()
    second = [n(t).copy() for t in range(5)]
    assert not np.array_equal(first[0], second[0])


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        make_noise("rainbow", 3, 0.1, 100, None)
