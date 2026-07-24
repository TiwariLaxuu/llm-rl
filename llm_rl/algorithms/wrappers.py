"""
Algorithm Wrappers for Stable-Baselines3 algorithms (PPO, DQN, A2C, SAC, TD3, RecurrentPPO).
"""

from typing import Any, Dict, Optional


class AlgorithmWrapper:
    """
    Uniform wrapper for RL model inspection across PPO, DQN, A2C, SAC, TD3, RecurrentPPO.
    """

    def __init__(self, model: Any):
        self.model = model
        self.name = model.__class__.__name__

    def get_info(self) -> Dict[str, Any]:
        """Extract algorithm metadata."""
        env_name = "UnknownEnv"
        if hasattr(self.model, "env") and self.model.env is not None:
            if hasattr(self.model.env, "envs") and self.model.env.envs:
                env_name = self.model.env.envs[0].spec.id if getattr(self.model.env.envs[0], "spec", None) else str(type(self.model.env.envs[0]))
            elif hasattr(self.model.env, "spec") and self.model.env.spec:
                env_name = self.model.env.spec.id

        return {
            "name": self.name,
            "policy": self.model.policy.__class__.__name__ if hasattr(self.model, "policy") else "Policy",
            "learning_rate": getattr(self.model, "learning_rate", 0.0003),
            "env_name": env_name,
            "has_replay_buffer": hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None
        }

    def get_replay_buffer(self) -> Optional[Any]:
        """Get model replay buffer if present (DQN, SAC, TD3)."""
        return getattr(self.model, "replay_buffer", None)
