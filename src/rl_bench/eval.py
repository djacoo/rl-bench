import numpy as np


def evaluate(agent, env, n_episodes=30, eval_seed=10000, deterministic=True):
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=eval_seed + i)
        done = False
        total = 0.0
        while not done:
            a = agent.act(obs, deterministic=deterministic)
            obs, r, term, trunc, _ = env.step(a)
            total += float(r)
            done = bool(term or trunc)
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns)), returns
