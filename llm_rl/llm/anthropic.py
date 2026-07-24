"""
Anthropic LLM Backend implementation for LLM-RL.
"""

import os
from typing import Optional
import requests
from llm_rl.llm.base import BaseLLM
from llm_rl.logger import logger


class AnthropicLLM(BaseLLM):
    """
    Anthropic Claude LLM backend driver.
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            msg = "[Anthropic LLM Notice] ANTHROPIC_API_KEY is missing."
            logger.warning(msg)
            return msg

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            res = requests.post(url, json=payload, headers=headers, timeout=30)
            if res.status_code == 200:
                data = res.json()
                return data["content"][0]["text"].strip()
            return f"[Anthropic Error HTTP {res.status_code}]: {res.text}"
        except Exception as e:
            return f"[Anthropic Error]: {e}"
