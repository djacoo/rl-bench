import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, activation=nn.ReLU):
        super().__init__()
        dims = [in_dim] + list(hidden) + [out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianTanhPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.act_dim = act_dim
        self.trunk = MLP(obs_dim, 2 * act_dim, list(hidden))

    def forward(self, obs):
        h = self.trunk(obs)
        mean, log_std = h.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs):
        mean, _ = self(obs)
        return torch.tanh(mean)


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = MLP(obs_dim + act_dim, 1, list(hidden))

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class ProbEnsembleMember(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(200, 200, 200, 200)):
        super().__init__()
        self.out_dim = out_dim
        self.net = MLP(in_dim, 2 * out_dim, list(hidden), activation=nn.SiLU)
        # soft logvar bounds (Chua 2018)
        self.max_logvar = nn.Parameter(torch.ones(out_dim) * 0.5)
        self.min_logvar = nn.Parameter(-torch.ones(out_dim) * 10.0)

    def forward(self, x):
        h = self.net(x)
        mean, logvar = h.chunk(2, dim=-1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar
