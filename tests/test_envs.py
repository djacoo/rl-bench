import numpy as np

from rl_bench.envs import RunningMeanStd, make_env


def test_running_mean_std_updates():
    rms = RunningMeanStd(shape=(2,))
    x = np.random.RandomState(0).randn(500, 2) * 3.0 + 1.0
    rms.update(x)
    assert np.allclose(rms.mean, x.mean(0), atol=0.1)
    assert np.allclose(rms.var, x.var(0), atol=0.5)


def test_running_mean_std_normalize_zero_mean_unit_var():
    rms = RunningMeanStd(shape=(2,))
    x = np.random.RandomState(0).randn(2000, 2) * 2.0 + 5.0
    rms.update(x)
    y = rms.normalize(x)
    assert abs(float(y.mean())) < 0.1
    assert abs(float(y.std()) - 1.0) < 0.1


def test_make_env_step_and_reset():
    env = make_env("Pendulum-v1", seed=0, obs_norm=False)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    a = env.action_space.sample()
    obs2, r, term, trunc, info = env.step(a)
    assert obs2.shape == env.observation_space.shape
    assert isinstance(float(r), float)


def test_make_env_obs_norm_updates_stats():
    env = make_env("Pendulum-v1", seed=0, obs_norm=True)
    env.reset()
    for _ in range(20):
        env.step(env.action_space.sample())
    assert not np.allclose(env.rms.mean, 0.0)


def test_make_env_eval_does_not_update_stats():
    train = make_env("Pendulum-v1", seed=0, obs_norm=True)
    train.reset()
    for _ in range(50):
        train.step(train.action_space.sample())
    eval_env = make_env(
        "Pendulum-v1", seed=1, eval=True, obs_norm=True, shared_rms=train.rms
    )
    mean_before = train.rms.mean.copy()
    count_before = train.rms.count
    eval_env.reset()
    for _ in range(20):
        eval_env.step(eval_env.action_space.sample())
    assert np.array_equal(mean_before, train.rms.mean)
    assert count_before == train.rms.count
