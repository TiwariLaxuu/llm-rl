"""
Ollama LLM Backend implementation for LLM-RL.
"""

from typing import Optional
import requests
from llm_rl.llm.base import BaseLLM
from llm_rl.logger import logger


class OllamaLLM(BaseLLM):
    """
    Ollama local LLM backend driver.
    """

    def __init__(self, model_name: str = "llama3.1:8b", host: str = "http://localhost:11434", **kwargs):
        super().__init__(model_name, **kwargs)
        self.host = host.rstrip('/')

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                res_text = data.get("response", "").strip()
                if res_text:
                    return res_text
            logger.warning(f"Ollama API returned HTTP {response.status_code}: {response.text}")
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama service at {self.host}: {e}")

        # Fallback response if Ollama server is unreachable or model is not pulled yet
        return self._generate_fallback(prompt)

    def _generate_fallback(self, prompt: str) -> str:
        """Generate structured contextual fallback response when local model is unavailable."""
        lines = prompt.strip().splitlines()
        header = f"### [LLM-RL Ollama Driver Notice]: Local Ollama model '{self.model_name}' unavailable or not pulled.\n"

        if "Generate a comprehensive" in prompt or "REPORT" in prompt.upper():
            return header + (
                "## Reinforcement Learning Training Report\n\n"
                "### 1. Executive Summary\n"
                "The agent completed training steps under LLM-RL observation. Metrics demonstrate active rollout collection.\n\n"
                "### 2. Metric Analysis & Reward Curves\n"
                "- Timestep progression tracked successfully.\n"
                "- Policy and Value losses logged.\n\n"
                "### 3. Failure & Replay Analysis\n"
                "- Trajectory samples recorded in shared monitor state.\n\n"
                "### 4. Recommendations & Next Steps\n"
                "- Ensure local Ollama model (`ollama pull llama3.1:8b`) is loaded for detailed LLM insights.\n"
            )
        elif "Explain" in prompt or "EXPLAIN" in prompt.upper():
            return header + (
                "**Action Explanation**: The agent evaluated current state observations to select an action that maximizes projected return.\n"
            )
        elif "reward" in prompt.lower():
            return header + (
                "**Reward Analysis**: Training reward signals monitored. Suggestions: Ensure reward scaling is balanced and penalize out-of-bounds states.\n"
            )
        else:
            return header + f"Received query regarding training metrics:\n{lines[0] if lines else ''}"
