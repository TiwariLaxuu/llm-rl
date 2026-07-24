"""
Reward Analysis module for evaluating reward functions and shaping signals.
"""

from typing import Dict, List, Optional
from llm_rl.llm.base import BaseLLM
from llm_rl.monitor import TrainingMonitor
from llm_rl.prompts import REWARD_ANALYSIS_PROMPT_TEMPLATE, SYSTEM_PROMPT


class RewardAnalyzer:
    """
    Analyzes reward trajectories and suggests reward function improvements.
    """

    def __init__(self, llm: BaseLLM, monitor: TrainingMonitor):
        self.llm = llm
        self.monitor = monitor

    def analyze_rewards(self, notes: Optional[str] = None) -> str:
        """
        Analyze recent reward distribution and generate reward shaping recommendations.
        """
        stats = self.monitor.get_reward_stats(window=100)
        recent_rews = self.monitor.state.episode_rewards[-20:]
        trajectory_notes = notes or f"Total episodes logged: {self.monitor.state.total_episodes}. Recent trend: {stats['trend']}."

        prompt = REWARD_ANALYSIS_PROMPT_TEMPLATE.format(
            mean_reward=stats["mean"],
            min_reward=stats["min"],
            max_reward=stats["max"],
            std_reward=stats["std"],
            recent_rewards=recent_rews,
            trajectory_notes=trajectory_notes
        )

        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
