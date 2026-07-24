"""
Replay Buffer Analysis module for experience inspection.
"""

from typing import Any, List, Optional
from llm_rl.llm.base import BaseLLM
from llm_rl.monitor import TrainingMonitor
from llm_rl.prompts import REPLAY_ANALYSIS_PROMPT_TEMPLATE, SYSTEM_PROMPT


class ReplayAnalyzer:
    """
    Analyzes replay buffer contents and trajectory experiences to identify failure modes.
    """

    def __init__(self, llm: BaseLLM, monitor: TrainingMonitor):
        self.llm = llm
        self.monitor = monitor

    def analyze_replay_buffer(self, replay_buffer: Optional[Any] = None) -> str:
        """
        Scan transitions for failure cases, bottleneck states, and successful trajectories.
        """
        trajectories = self.monitor.state.trajectories
        total_transitions = sum(t.get("steps", 0) for t in trajectories) if trajectories else 0

        # Sort trajectories into failure and success buckets
        sorted_trajs = sorted(trajectories, key=lambda x: x.get("total_reward", 0)) if trajectories else []
        failure_trajs = sorted_trajs[:3] if sorted_trajs else []
        success_trajs = sorted_trajs[-3:] if sorted_trajs else []

        failure_samples = (
            "\n".join([f"Ep Reward: {t.get('total_reward')}, Steps: {t.get('steps')}" for t in failure_trajs])
            if failure_trajs else "No explicit failure samples recorded."
        )

        success_samples = (
            "\n".join([f"Ep Reward: {t.get('total_reward')}, Steps: {t.get('steps')}" for t in success_trajs])
            if success_trajs else "No explicit success samples recorded."
        )

        prompt = REPLAY_ANALYSIS_PROMPT_TEMPLATE.format(
            num_transitions=total_transitions,
            failure_count=len(failure_trajs),
            failure_samples=failure_samples,
            success_samples=success_samples
        )

        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
