"""
OpenAI LLM Backend implementation for LLM-RL.
"""

import os
from typing import Optional
import requests
from llm_rl.llm.base import BaseLLM
from llm_rl.logger import logger


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM backend driver.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            msg = "[OpenAI LLM Notice] OPENAI_API_KEY is missing."
            logger.warning(msg)
            return msg

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.kwargs.get("temperature", 0.7)
        }

        try:
            res = requests.post(url, json=payload, headers=headers, timeout=30)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                err_msg = f"OpenAI API HTTP {res.status_code}: {res.text}"
                logger.warning(err_msg)
                return f"[OpenAI Error]: {err_msg}"
        except Exception as e:
            logger.warning(f"Error calling OpenAI API: {e}")
            return f"[OpenAI Error]: {e}"
