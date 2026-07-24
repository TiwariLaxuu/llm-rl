"""
PPO and DQN specialized wrappers for LLM-RL.
"""

from typing import Any, Dict
from llm_rl.algorithms.wrappers import AlgorithmWrapper


class PPOWrapper(AlgorithmWrapper):
    """Specialized wrapper for PPO and RecurrentPPO algorithms."""

    def extract_ppo_metrics(self) -> Dict[str, Any]:
        return {
            "n_steps": getattr(self.model, "n_steps", 2048),
            "batch_size": getattr(self.model, "batch_size", 64),
            "n_epochs": getattr(self.model, "n_epochs", 10),
            "gae_lambda": getattr(self.model, "gae_lambda", 0.95),
            "clip_range": getattr(self.model, "clip_range", 0.2),
        }


class DQNWrapper(AlgorithmWrapper):
    """Specialized wrapper for DQN algorithm."""

    def extract_dqn_metrics(self) -> Dict[str, Any]:
        return {
            "buffer_size": getattr(self.model, "buffer_size", 1000000),
            "exploration_fraction": getattr(self.model, "exploration_fraction", 0.1),
            "exploration_initial_eps": getattr(self.model, "exploration_initial_eps", 1.0),
            "exploration_final_eps": getattr(self.model, "exploration_final_eps", 0.05),
            "target_update_interval": getattr(self.model, "target_update_interval", 10000),
        }
