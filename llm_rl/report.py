"""
Report Generation module for automatic post-training summaries.
"""

import os
from typing import Optional
from llm_rl.llm.base import BaseLLM
from llm_rl.monitor import TrainingMonitor
from llm_rl.prompts import REPORT_GENERATION_PROMPT_TEMPLATE, SYSTEM_PROMPT
from llm_rl.logger import logger


class ReportGenerator:
    """
    Generates detailed markdown training reports.
    """

    def __init__(self, llm: BaseLLM, monitor: TrainingMonitor):
        self.llm = llm
        self.monitor = monitor

    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate markdown report from current training monitor state.

        Args:
            save_path: Optional file path to save the generated markdown report.

        Returns:
            Markdown formatted report string.
        """
        context = self.monitor.get_context_summary()

        metrics_summary = (
            f"- Total Timesteps: {context['timestep']}/{context['total_timesteps']}\n"
            f"- Mean Reward: {context['mean_reward']:.2f} (std: {context['std_reward']:.2f})\n"
            f"- Reward Range: [{context['min_reward']:.2f}, {context['max_reward']:.2f}]\n"
            f"- Reward Trend: {context['reward_trend']}\n"
            f"- Final Policy Loss: {context['policy_loss']}\n"
            f"- Final Value Loss: {context['value_loss']}\n"
            f"- Entropy: {context['entropy']}"
        )

        failure_summary = (
            f"Logged {context['total_episodes']} total episodes.\n"
            f"Trajectory Summary: {context['trajectory_summary']}"
        )

        prompt = REPORT_GENERATION_PROMPT_TEMPLATE.format(
            env_name=context["env_name"],
            algorithm_name=context["algorithm_name"],
            timestep=context["timestep"],
            total_timesteps=context["total_timesteps"],
            total_episodes=context["total_episodes"],
            mean_reward=context["mean_reward"],
            max_reward=context["max_reward"],
            policy_loss=context["policy_loss"],
            value_loss=context["value_loss"],
            reward_trend=context["reward_trend"],
            metrics_history_summary=metrics_summary,
            failure_analysis_summary=failure_summary
        )

        report_content = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)

        if save_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(report_content)
                logger.info(f"Report saved successfully to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save report to {save_path}: {e}")

        return report_content
