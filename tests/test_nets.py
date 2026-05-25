import torch

from rl_bench.nets import MLP, GaussianTanhPolicy, ProbEnsembleMember, QNet


def test_mlp_shape():
    m = MLP(4, 2, [8, 8])
    out = m(torch.randn(3, 4))
    assert out.shape == (3, 2)


def test_policy_sample_shape_and_range():
    p = GaussianTanhPolicy(4, 2, [16, 16])
    obs = torch.randn(5, 4)
    a, lp = p.sample(obs)
    assert a.shape == (5, 2)
    assert lp.shape == (5, 1)
    assert (a.abs() <= 1.0 + 1e-6).all()


def test_policy_deterministic_returns_tanh_mean():
    p = GaussianTanhPolicy(4, 2, [16, 16])
    a = p.deterministic(torch.randn(3, 4))
    assert a.shape == (3, 2)
    assert (a.abs() <= 1.0 + 1e-6).all()


def test_qnet_scalar_per_sample():
    q = QNet(4, 2, [16, 16])
    o = torch.randn(7, 4)
    a = torch.randn(7, 2)
    out = q(o, a)
    assert out.shape == (7,)


def test_ensemble_member_output_shapes():
    m = ProbEnsembleMember(5, 3, [16, 16])
    mean, logvar = m(torch.randn(4, 5))
    assert mean.shape == (4, 3)
    assert logvar.shape == (4, 3)


def test_ensemble_member_logvar_softbounded():
    m = ProbEnsembleMember(5, 3, [16, 16])
    _, logvar = m(torch.randn(4, 5))
    assert (logvar < m.max_logvar + 1e-3).all()
    assert (logvar > m.min_logvar - 1e-3).all()
