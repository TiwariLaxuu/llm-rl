"""
Stable-Baselines3 Callback integration for LLM-RL.
"""

from typing import Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from llm_rl.monitor import TrainingMonitor
from llm_rl.logger import logger


class TrainingCallback(BaseCallback):
    """
    Stable-Baselines3 callback that collects training state, metrics, trajectories,
    and updates the LLM-RL TrainingMonitor.
    """

    def __init__(self, monitor: TrainingMonitor, verbose: int = 0):
        super().__init__(verbose)
        self.monitor = monitor
        self._current_episode_reward = 0.0
        self._current_episode_steps = 0
        self._current_trajectory = {"states": [], "actions": [], "rewards": []}

    def _on_step(self) -> bool:
        """Called at each environment step in SB3."""
        step = self.num_timesteps
        self.monitor.update_step(step)

        # Extract step observations, actions, rewards
        if self.locals and "actions" in self.locals:
            action = self.locals["actions"]
            obs = self.locals.get("new_obs", self.locals.get("obs_tensor"))
            rewards = self.locals.get("rewards", [0.0])

            r = float(rewards[0]) if hasattr(rewards, "__len__") else float(rewards)
            self._current_episode_reward += r
            self._current_episode_steps += 1

            if obs is not None and hasattr(obs, "tolist"):
                self._current_trajectory["states"].append(obs.tolist())
            if action is not None and hasattr(action, "tolist"):
                self._current_trajectory["actions"].append(action.tolist())
            self._current_trajectory["rewards"].append(r)

            # Check for episode termination/dones
            dones = self.locals.get("dones", [False])
            done = bool(dones[0]) if hasattr(dones, "__len__") else bool(dones)

            if done:
                self._current_trajectory["total_reward"] = self._current_episode_reward
                self._current_trajectory["steps"] = self._current_episode_steps
                self.monitor.record_episode(
                    reward=self._current_episode_reward,
                    length=self._current_episode_steps,
                    trajectory=self._current_trajectory
                )
                # Reset episode tracking
                self._current_episode_reward = 0.0
                self._current_episode_steps = 0
                self._current_trajectory = {"states": [], "actions": [], "rewards": []}

        return True

    def _on_rollout_end(self) -> None:
        """Extract loss and entropy values at rollout end."""
        if hasattr(self.logger, "name_to_value"):
            logs = self.logger.name_to_value
            policy_loss = logs.get("train/policy_gradient_loss", logs.get("train/loss"))
            value_loss = logs.get("train/value_loss")
            entropy = logs.get("train/entropy_loss")

            self.monitor.record_losses(
                policy_loss=float(policy_loss) if policy_loss is not None else None,
                value_loss=float(value_loss) if value_loss is not None else None,
                entropy=float(entropy) if entropy is not None else None
            )
