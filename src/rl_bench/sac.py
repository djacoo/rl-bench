import copy

import torch
import torch.nn.functional as F

from .nets import GaussianTanhPolicy, QNet


class SACAgent:
    """Soft Actor-Critic (Haarnoja et al. 2018b) with learned alpha."""

    def __init__(
        self,
        obs_dim,
        act_dim,
        device,
        gamma=0.99,
        tau=0.005,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        hidden=(256, 256),
        target_entropy=None,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.actor = GaussianTanhPolicy(obs_dim, act_dim, hidden).to(self.device)
        self.q1 = QNet(obs_dim, act_dim, hidden).to(self.device)
        self.q2 = QNet(obs_dim, act_dim, hidden).to(self.device)
        self.q1_tgt = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_tgt = copy.deepcopy(self.q2).requires_grad_(False)
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.target_entropy = (
            -float(act_dim) if target_entropy is None else float(target_entropy)
        )
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr_critic
        )
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            a = self.actor.deterministic(o)
        else:
            a, _ = self.actor.sample(o)
        return a.squeeze(0).cpu().numpy()

    def update(self, batch):
        s = batch["obs"]
        a = batch["act"]
        r = batch["rew"]
        s2 = batch["next_obs"]
        d = batch["done"]

        with torch.no_grad():
            a2, lp2 = self.actor.sample(s2)
            q1_t = self.q1_tgt(s2, a2)
            q2_t = self.q2_tgt(s2, a2)
            v_t = torch.min(q1_t, q2_t) - self.alpha * lp2.squeeze(-1)
            y = r + self.gamma * (1.0 - d) * v_t

        q1_pred = self.q1(s, a)
        q2_pred = self.q2(s, a)
        critic_loss = F.mse_loss(q1_pred, y) + F.mse_loss(q2_pred, y)
        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.opt_critic.step()

        a_pi, lp = self.actor.sample(s)
        q1_pi = self.q1(s, a_pi)
        q2_pi = self.q2(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * lp.squeeze(-1) - q_pi).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt_actor.step()

        alpha_loss = -(self.log_alpha * (lp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, p_t in zip(self.q1.parameters(), self.q1_tgt.parameters()):
                p_t.data.mul_(1.0 - self.tau)
                p_t.data.add_(self.tau * p.data)
            for p, p_t in zip(self.q2.parameters(), self.q2_tgt.parameters()):
                p_t.data.mul_(1.0 - self.tau)
                p_t.data.add_(self.tau * p.data)

        return {
            "loss/critic": float(critic_loss.item()),
            "loss/actor": float(actor_loss.item()),
            "loss/alpha": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "entropy": float(-lp.mean().item()),
        }

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_tgt": self.q1_tgt.state_dict(),
                "q2_tgt": self.q2_tgt.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_tgt.load_state_dict(ckpt["q1_tgt"])
        self.q2_tgt.load_state_dict(ckpt["q2_tgt"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
