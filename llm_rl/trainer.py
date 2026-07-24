"""
LLMTrainer main orchestrator module with automated background dashboard API.
"""

from typing import Any, List, Optional
from llm_rl.llm import get_llm_backend
from llm_rl.llm.base import BaseLLM
from llm_rl.monitor import TrainingMonitor
from llm_rl.callback import TrainingCallback
from llm_rl.explain import PolicyExplainer
from llm_rl.reward import RewardAnalyzer
from llm_rl.replay import ReplayAnalyzer
from llm_rl.report import ReportGenerator
from llm_rl.algorithms.wrappers import AlgorithmWrapper
from llm_rl.prompts import LIVE_CHAT_PROMPT_TEMPLATE, SYSTEM_PROMPT
from llm_rl.logger import logger


class LLMTrainer:
    """
    Main LLM-RL Trainer orchestrating RL training, live background monitoring,
    LLM reasoning backends, policy explanations, and real-time dashboard API.
    """

    def __init__(
        self,
        model: Any,
        llm_backend: str = "ollama",
        model_name: str = "llama3.1:8b",
        **kwargs
    ):
        """
        Initialize LLMTrainer.

        Args:
            model: Stable-Baselines3 model (PPO, DQN, A2C, SAC, TD3, etc.).
            llm_backend: Provider name ('ollama', 'openai', 'huggingface', 'anthropic', 'gemini', 'groq', 'llamacpp').
            model_name: LLM model name/identifier.
            **kwargs: Extra parameters passed to LLM backend.
        """
        self.model = model
        self.algo_wrapper = AlgorithmWrapper(model)
        info = self.algo_wrapper.get_info()

        self.env_name = info["env_name"]
        self.algorithm_name = info["name"]
        self._dashboard_url = None

        # Instantiate LLM Driver
        if isinstance(llm_backend, BaseLLM):
            self.llm = llm_backend
        else:
            self.llm = get_llm_backend(backend=llm_backend, model_name=model_name, **kwargs)

        # Training Monitor
        self.monitor = TrainingMonitor(
            total_timesteps=100000,
            env_name=self.env_name,
            algorithm_name=self.algorithm_name
        )

        # Helper Modules
        self.explainer = PolicyExplainer(self.llm, env_name=self.env_name)
        self.reward_analyzer = RewardAnalyzer(self.llm, self.monitor)
        self.replay_analyzer = ReplayAnalyzer(self.llm, self.monitor)
        self.report_generator = ReportGenerator(self.llm, self.monitor)

        logger.info(f"Initialized LLMTrainer for model {self.algorithm_name} on {self.env_name} with backend '{llm_backend}' ({model_name}).")

    def train(
        self,
        total_timesteps: int = 100000,
        auto_dashboard: bool = True,
        port: int = 7860,
        callback: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        Train the RL agent while attached to LLM-RL monitoring callback.
        Optionally launches a non-blocking background dashboard API.

        Args:
            total_timesteps: Total environment timesteps to train.
            auto_dashboard: Whether to automatically launch live background dashboard URL.
            port: Server port for live dashboard API.
            callback: Optional user SB3 callback to combine.

        Returns:
            Trained model.
        """
        self.monitor.state.total_timesteps = total_timesteps
        cb = TrainingCallback(self.monitor)

        callbacks = [cb]
        if callback:
            if isinstance(callback, list):
                callbacks.extend(callback)
            else:
                callbacks.append(callback)

        # Automatically launch non-blocking live dashboard in background thread if requested
        if auto_dashboard and not self._dashboard_url:
            try:
                self._dashboard_url = self.dashboard(port=port)
            except Exception as e:
                logger.warning(f"Could not launch live dashboard automatically: {e}")

        if self._dashboard_url:
            print("\n" + "=" * 80)
            print(f"📊 LLM-RL LIVE DASHBOARD URL: {self._dashboard_url}")
            print(f"🌐 Monitoring live reward curves, loss plots & trajectories in background without disturbing training.")
            print("=" * 80 + "\n")

        logger.info(f"Starting RL training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks, **kwargs)
        logger.info("Training complete.")
        return self.model

    def dashboard(self, port: int = 7860, share: bool = False) -> str:
        """
        Launch interactive non-blocking Gradio Dashboard in background thread.

        Returns:
            URL string of the running dashboard.
        """
        from llm_rl.dashboard import launch_dashboard
        url = launch_dashboard(self, port=port, share=share)
        self._dashboard_url = url
        return url

    def explain_action(
        self,
        observation: Any,
        action: Any,
        action_meaning: Optional[str] = None,
        env_info: Optional[str] = None
    ) -> str:
        """Explain an action taken given state observation."""
        return self.explainer.explain_action(observation, action, action_meaning, env_info)

    def analyze_rewards(self, notes: Optional[str] = None) -> str:
        """Analyze reward distributions and generate recommendations."""
        return self.reward_analyzer.analyze_rewards(notes)

    def analyze_replay_buffer(self, replay_buffer: Optional[Any] = None) -> str:
        """Analyze replay buffer / trajectory failure modes."""
        buf = replay_buffer or self.algo_wrapper.get_replay_buffer()
        return self.replay_analyzer.analyze_replay_buffer(buf)

    def generate_report(self, save_path: Optional[str] = "training_report.md") -> str:
        """Generate markdown report."""
        return self.report_generator.generate_report(save_path)

    def chat(self, user_query: str) -> str:
        """Interactive live chat with training copilot."""
        context = self.monitor.get_context_summary()
        prompt = LIVE_CHAT_PROMPT_TEMPLATE.format(
            user_query=user_query,
            **context
        )
        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
