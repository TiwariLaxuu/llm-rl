"""
Training Monitor for tracking, storing, and analyzing live RL training state.
"""

from dataclasses import dataclass, field
import threading
from typing import Any, Dict, List, Optional
import numpy as np
from llm_rl.logger import logger


@dataclass
class SharedTrainingState:
    """Dataclass holding thread-safe shared training state metrics."""
    timestep: int = 0
    total_timesteps: int = 100000
    total_episodes: int = 0
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    policy_loss_history: List[float] = field(default_factory=list)
    value_loss_history: List[float] = field(default_factory=list)
    entropy_history: List[float] = field(default_factory=list)
    learning_rate: float = 0.0003
    algorithm_name: str = "PPO"
    env_name: str = "CartPole-v1"
    recent_observations: List[Any] = field(default_factory=list)
    recent_actions: List[Any] = field(default_factory=list)
    trajectories: List[Dict[str, Any]] = field(default_factory=list)
    llm_recommendations: List[Dict[str, str]] = field(default_factory=list)


class TrainingMonitor:
    """
    Monitors RL training execution, maintains state buffers, and prepares contexts for LLM prompts.
    """

    def __init__(self, total_timesteps: int = 100000, env_name: str = "GymEnvironment", algorithm_name: str = "RL-Algo"):
        self.state = SharedTrainingState(
            total_timesteps=total_timesteps,
            env_name=env_name,
            algorithm_name=algorithm_name
        )
        self._lock = threading.Lock()

    def update_step(self, step: int, obs: Optional[Any] = None, action: Optional[Any] = None):
        """Update current timestep and sample observations/actions."""
        with self._lock:
            self.state.timestep = step
            if obs is not None:
                self.state.recent_observations.append(obs)
                if len(self.state.recent_observations) > 50:
                    self.state.recent_observations.pop(0)
            if action is not None:
                self.state.recent_actions.append(action)
                if len(self.state.recent_actions) > 50:
                    self.state.recent_actions.pop(0)

    def record_episode(self, reward: float, length: int, trajectory: Optional[Dict[str, Any]] = None):
        """Record completed episode return, length, and trajectory data."""
        with self._lock:
            self.state.total_episodes += 1
            self.state.episode_rewards.append(reward)
            self.state.episode_lengths.append(length)

            if trajectory:
                self.state.trajectories.append(trajectory)
                if len(self.state.trajectories) > 100:
                    self.state.trajectories.pop(0)

    def record_losses(self, policy_loss: Optional[float] = None, value_loss: Optional[float] = None, entropy: Optional[float] = None):
        """Record policy/value loss and entropy metrics."""
        with self._lock:
            if policy_loss is not None:
                self.state.policy_loss_history.append(policy_loss)
            if value_loss is not None:
                self.state.value_loss_history.append(value_loss)
            if entropy is not None:
                self.state.entropy_history.append(entropy)

    def add_recommendation(self, topic: str, content: str):
        """Store an LLM recommendation."""
        with self._lock:
            self.state.llm_recommendations.append({"topic": topic, "content": content, "timestep": self.state.timestep})

    def get_reward_stats(self, window: int = 100) -> Dict[str, float]:
        """Compute reward statistics over the recent episode window."""
        with self._lock:
            rewards = self.state.episode_rewards[-window:]
            if not rewards:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "trend": "Insufficient Data"}

            mean_rew = float(np.mean(rewards))
            std_rew = float(np.std(rewards))
            min_rew = float(np.min(rewards))
            max_rew = float(np.max(rewards))

            # Compute trend
            if len(rewards) >= 20:
                first_half = np.mean(rewards[: len(rewards) // 2])
                second_half = np.mean(rewards[len(rewards) // 2 :])
                diff = second_half - first_half
                if diff > 0.05 * abs(first_half or 1.0):
                    trend = "Increasing (Improving)"
                elif diff < -0.05 * abs(first_half or 1.0):
                    trend = "Decreasing (Degrading)"
                else:
                    trend = "Stable / Plateaued"
            else:
                trend = "Initializing"

            return {
                "mean": mean_rew,
                "std": std_rew,
                "min": min_rew,
                "max": max_rew,
                "trend": trend
            }

    def get_context_summary(self) -> Dict[str, Any]:
        """Generate a context dictionary formatted for LLM prompts."""
        stats = self.get_reward_stats()
        with self._lock:
            pol_loss = self.state.policy_loss_history[-1] if self.state.policy_loss_history else "N/A"
            val_loss = self.state.value_loss_history[-1] if self.state.value_loss_history else "N/A"
            ent = self.state.entropy_history[-1] if self.state.entropy_history else "N/A"

            traj_summary = "Recent trajectories recorded: " + str(len(self.state.trajectories))
            if self.state.trajectories:
                last_t = self.state.trajectories[-1]
                traj_summary += f" | Last episode reward: {last_t.get('total_reward', 'N/A')}, steps: {last_t.get('steps', 'N/A')}"

            return {
                "timestep": self.state.timestep,
                "total_timesteps": self.state.total_timesteps,
                "total_episodes": self.state.total_episodes,
                "mean_reward": stats["mean"],
                "std_reward": stats["std"],
                "min_reward": stats["min"],
                "max_reward": stats["max"],
                "reward_trend": stats["trend"],
                "policy_loss": pol_loss,
                "value_loss": val_loss,
                "entropy": ent,
                "learning_rate": self.state.learning_rate,
                "algorithm_name": self.state.algorithm_name,
                "env_name": self.state.env_name,
                "recent_rewards": self.state.episode_rewards[-10:],
                "trajectory_summary": traj_summary
            }
