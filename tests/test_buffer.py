import numpy as np
import torch

from rl_bench.buffer import ReplayBuffer


def test_buffer_len_after_adds():
    buf = ReplayBuffer(100, 3, 2, "cpu")
    assert len(buf) == 0
    for i in range(10):
        buf.add(np.ones(3) * i, np.ones(2), float(i), np.ones(3), False)
    assert len(buf) == 10


def test_buffer_sample_shapes_and_types():
    buf = ReplayBuffer(100, 3, 2, "cpu")
    for i in range(20):
        buf.add(np.ones(3) * i, np.ones(2), float(i), np.ones(3), False)
    b = buf.sample(8)
    assert isinstance(b, dict)
    assert b["obs"].shape == (8, 3)
    assert b["act"].shape == (8, 2)
    assert b["rew"].shape == (8,)
    assert b["next_obs"].shape == (8, 3)
    assert b["done"].shape == (8,)
    assert b["obs"].dtype == torch.float32


def test_buffer_ring_overwrite():
    buf = ReplayBuffer(5, 1, 1, "cpu")
    for i in range(7):
        buf.add(np.array([float(i)]), np.array([0.0]), 0.0, np.array([0.0]), False)
    assert len(buf) == 5
    assert buf.obs[0, 0] == 5.0
    assert buf.obs[1, 0] == 6.0


def test_buffer_sample_states():
    buf = ReplayBuffer(10, 2, 1, "cpu")
    for i in range(8):
        buf.add(np.ones(2) * i, np.zeros(1), 0.0, np.zeros(2), False)
    s = buf.sample_states(3)
    assert s.shape == (3, 2)
    assert s.dtype == torch.float32


def test_buffer_add_batch():
    buf = ReplayBuffer(10, 2, 1, "cpu")
    s = np.arange(8).reshape(4, 2).astype(np.float32)
    a = np.zeros((4, 1), dtype=np.float32)
    r = np.ones(4, dtype=np.float32)
    s2 = s + 1
    d = np.zeros(4, dtype=np.float32)
    buf.add_batch(s, a, r, s2, d)
    assert len(buf) == 4
    assert np.array_equal(buf.obs[:4], s)


def test_buffer_done_stored_as_float():
    buf = ReplayBuffer(5, 1, 1, "cpu")
    buf.add(np.array([0.0]), np.array([0.0]), 0.0, np.array([0.0]), True)
    assert buf.done[0] == 1.0
