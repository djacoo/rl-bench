import colorednoise
import numpy as np


class NoOpNoise:
    def __call__(self, t):
        return 0.0

    def reset(self):
        pass


class WhiteNoise:
    def __init__(self, act_dim, scale, rng):
        self.act_dim = act_dim
        self.scale = scale
        self.rng = rng

    def __call__(self, t):
        return self.rng.normal(scale=self.scale, size=self.act_dim)

    def reset(self):
        pass


class PinkNoise:
    """Eberhard et al. 2023 — pink (1/f) noise regenerated per episode."""

    def __init__(self, act_dim, scale, episode_len, rng):
        self.act_dim = act_dim
        self.scale = scale
        self.episode_len = int(episode_len)
        self.rng = rng
        self._buf = None
        self._t = 0
        self.reset()

    def reset(self):
        self._buf = (
            colorednoise.powerlaw_psd_gaussian(
                1, size=(self.act_dim, self.episode_len), random_state=self.rng
            )
            * self.scale
        )
        self._t = 0

    def __call__(self, t):
        if self._t >= self.episode_len:
            self.reset()
        x = self._buf[:, self._t]
        self._t += 1
        return x


def make_noise(kind, act_dim, scale, episode_len, rng):
    if kind in ("stochastic", "deterministic"):
        return NoOpNoise()
    if kind == "white":
        return WhiteNoise(act_dim, scale, rng)
    if kind == "pink":
        return PinkNoise(act_dim, scale, episode_len, rng)
    raise ValueError(f"unknown noise kind: {kind}")
