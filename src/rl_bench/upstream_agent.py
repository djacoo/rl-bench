"""Load upstream SAC checkpoint for eval / video recording."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .upstream_path import ensure_upstream


class UpstreamSacAdapter:
    """Minimal .act() wrapper around mbrl SACAgent."""

    def __init__(self, sac_agent):
        self._agent = sac_agent

    def act(self, obs, deterministic=True):
        return self._agent.act(
            np.asarray(obs, dtype=np.float32),
            sample=not deterministic,
            batched=False,
        )


def load_upstream_sac(env, mbrl_cfg, ckpt_path: Path) -> UpstreamSacAdapter:
    ensure_upstream()
    import mbrl.planning
    import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac
    from mbrl.planning.sac_wrapper import SACAgent

    mbrl.planning.complete_agent_cfg(env, mbrl_cfg.algorithm.agent)
    sac = pytorch_sac.SAC(
        mbrl_cfg.algorithm.agent.num_inputs,
        env.action_space,
        mbrl_cfg.algorithm.agent.args,
    )
    agent = SACAgent(sac)
    agent.sac_agent.load_checkpoint(str(ckpt_path), evaluate=True)
    return UpstreamSacAdapter(agent)
