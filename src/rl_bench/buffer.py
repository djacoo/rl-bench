import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((self.capacity,), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s_next, d):
        i = self.ptr
        self.obs[i] = s
        self.act[i] = a
        self.rew[i] = float(r)
        self.next_obs[i] = s_next
        self.done[i] = float(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, s, a, r, s_next, d):
        n = s.shape[0]
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.obs[idx] = s
        self.act[idx] = a
        self.rew[idx] = np.asarray(r, dtype=np.float32).reshape(-1)
        self.next_obs[idx] = s_next
        self.done[idx] = np.asarray(d, dtype=np.float32).reshape(-1)
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self._to_torch(idx)

    def sample_states(self, n):
        idx = np.random.randint(0, self.size, size=n)
        return torch.from_numpy(self.obs[idx]).to(self.device)

    def _to_torch(self, idx):
        d = self.device
        return {
            "obs": torch.from_numpy(self.obs[idx]).to(d),
            "act": torch.from_numpy(self.act[idx]).to(d),
            "rew": torch.from_numpy(self.rew[idx]).to(d),
            "next_obs": torch.from_numpy(self.next_obs[idx]).to(d),
            "done": torch.from_numpy(self.done[idx]).to(d),
        }

    def __len__(self):
        return self.size
