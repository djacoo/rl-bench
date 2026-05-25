import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nets import ProbEnsembleMember


class ProbabilisticEnsemble(nn.Module):
    """PETS-style probabilistic ensemble predicting (Δs, r) given (s, a)."""

    def __init__(
        self,
        obs_dim,
        act_dim,
        n_members=7,
        n_elites=5,
        hidden=(200, 200, 200, 200),
        device="cpu",
        lr=1e-3,
        weight_decay=1e-5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_members = n_members
        self.n_elites = n_elites
        self.device = torch.device(device)
        in_dim = obs_dim + act_dim
        out_dim = obs_dim + 1
        self.members = nn.ModuleList(
            [ProbEnsembleMember(in_dim, out_dim, hidden) for _ in range(n_members)]
        )
        self.to(self.device)
        self.elite_idx = list(range(n_elites))
        self.register_buffer("in_mean", torch.zeros(in_dim, device=self.device))
        self.register_buffer("in_std", torch.ones(in_dim, device=self.device))
        self.opt = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _norm_in(self, x):
        return (x - self.in_mean) / self.in_std

    @staticmethod
    def _nll(mean, logvar, target):
        inv_var = torch.exp(-logvar)
        return ((mean - target) ** 2 * inv_var + logvar).mean()

    def fit(
        self,
        buffer,
        max_samples=40_000,
        holdout_frac=0.2,
        batch=256,
        max_epochs=200,
        patience=5,
    ):
        n = min(max_samples, len(buffer))
        idx = np.random.choice(len(buffer), n, replace=False)
        obs = torch.from_numpy(buffer.obs[idx]).to(self.device)
        act = torch.from_numpy(buffer.act[idx]).to(self.device)
        rew = torch.from_numpy(buffer.rew[idx]).to(self.device).unsqueeze(-1)
        next_obs = torch.from_numpy(buffer.next_obs[idx]).to(self.device)
        x = torch.cat([obs, act], dim=-1)
        y = torch.cat([next_obs - obs, rew], dim=-1)

        self.in_mean = x.mean(0)
        self.in_std = x.std(0).clamp_min(1e-6)
        x = self._norm_in(x)

        n_h = max(1, int(n * holdout_frac))
        perm = torch.randperm(n, device=self.device)
        x_h, y_h = x[perm[:n_h]], y[perm[:n_h]]
        x_tr, y_tr = x[perm[n_h:]], y[perm[n_h:]]
        n_tr = x_tr.shape[0]

        boot = [
            torch.randint(0, n_tr, (n_tr,), device=self.device)
            for _ in range(self.n_members)
        ]

        best_loss = [float("inf")] * self.n_members
        bad_epochs = [0] * self.n_members
        best_state = [None] * self.n_members
        done = [False] * self.n_members

        for _ in range(max_epochs):
            perm_e = torch.randperm(n_tr, device=self.device)
            for start in range(0, n_tr, batch):
                bidx = perm_e[start : start + batch]
                total = None
                for m, mem in enumerate(self.members):
                    if done[m]:
                        continue
                    sel = boot[m][bidx]
                    mean, logvar = mem(x_tr[sel])
                    loss = self._nll(mean, logvar, y_tr[sel])
                    loss = loss + 0.01 * mem.max_logvar.sum() - 0.01 * mem.min_logvar.sum()
                    total = loss if total is None else total + loss
                if total is None:
                    continue
                self.opt.zero_grad(set_to_none=True)
                total.backward()
                self.opt.step()

            with torch.no_grad():
                for m, mem in enumerate(self.members):
                    mean, _ = mem(x_h)
                    h_loss = F.mse_loss(mean, y_h).item()
                    if h_loss < best_loss[m] - 1e-4:
                        best_loss[m] = h_loss
                        bad_epochs[m] = 0
                        best_state[m] = {
                            k: v.detach().clone() for k, v in mem.state_dict().items()
                        }
                    else:
                        bad_epochs[m] += 1
                        if bad_epochs[m] >= patience:
                            done[m] = True
            if all(done):
                break

        for m, mem in enumerate(self.members):
            if best_state[m] is not None:
                mem.load_state_dict(best_state[m])

        ranked = sorted(range(self.n_members), key=lambda i: best_loss[i])
        self.elite_idx = ranked[: self.n_elites]
        return {
            "holdout_loss_mean": float(np.mean(best_loss)),
            "holdout_loss_best": float(min(best_loss)),
            "n_samples": int(n),
        }

    @torch.no_grad()
    def _predict_subset(self, s, a, idx_list):
        x = torch.cat([s, a], dim=-1)
        x = self._norm_in(x)
        means, logvars = [], []
        for i in idx_list:
            mean, logvar = self.members[i](x)
            means.append(mean)
            logvars.append(logvar)
        means = torch.stack(means, dim=0)
        logvars = torch.stack(logvars, dim=0)
        return means, logvars.exp()

    def predict_all(self, s, a):
        return self._predict_subset(s, a, list(range(self.n_members)))

    def predict_elite(self, s, a):
        return self._predict_subset(s, a, self.elite_idx)

    @torch.no_grad()
    def sample_from(self, s, a, member_indices, idx_subset):
        """Sample (s_next, r) using member_indices into idx_subset per batch element."""
        mu, var = self._predict_subset(s, a, idx_subset)
        B = s.shape[0]
        arange = torch.arange(B, device=self.device)
        chosen_mu = mu[member_indices, arange]
        chosen_var = var[member_indices, arange]
        sample = chosen_mu + torch.randn_like(chosen_mu) * chosen_var.sqrt()
        s_next = s + sample[..., : self.obs_dim]
        r = sample[..., self.obs_dim]
        return s_next, r

    @torch.no_grad()
    def gjs_uncertainty(self, mu, var, eps=1e-8):
        """Geometric JS divergence across ensemble members (Frauenknecht 2024 eqs 15-19).

        mu, var: [E, B, D] (diagonal Gaussians).
        Returns [B]: mean of pairwise D_GJS over e<f pairs, averaged across D dims.
        """
        E, B, D = mu.shape
        var = var.clamp_min(eps)
        mu_e = mu.unsqueeze(1)  # [E, 1, B, D]
        mu_f = mu.unsqueeze(0)  # [1, E, B, D]
        var_e = var.unsqueeze(1)
        var_f = var.unsqueeze(0)
        inv_e = 1.0 / var_e
        inv_f = 1.0 / var_f
        var_ef = 1.0 / (0.5 * inv_e + 0.5 * inv_f)
        mu_ef = var_ef * (0.5 * inv_e * mu_e + 0.5 * inv_f * mu_f)

        def kl(mu_a, var_a, mu_b, var_b):
            return 0.5 * (
                torch.log(var_b / var_a) + (var_a + (mu_a - mu_b) ** 2) / var_b - 1.0
            )

        d_gjs = 0.5 * (
            kl(mu_e, var_e, mu_ef, var_ef) + kl(mu_f, var_f, mu_ef, var_ef)
        )
        d_gjs = d_gjs.sum(dim=-1) / D  # average over output dims
        mask = torch.triu(
            torch.ones(E, E, device=mu.device, dtype=torch.bool), diagonal=1
        )
        pairs = d_gjs[mask]  # [n_pairs, B]
        return pairs.mean(dim=0)
