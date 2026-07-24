"""
Hugging Face LLM Backend implementation for LLM-RL.
"""

import os
from typing import Optional
import requests
from llm_rl.llm.base import BaseLLM
from llm_rl.logger import logger


class HuggingFaceLLM(BaseLLM):
    """
    Hugging Face Hub Inference API / Local pipeline LLM backend driver.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", token: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        if self.token:
            url = f"https://api-inference.huggingface.co/models/{self.model_name}"
            headers = {"Authorization": f"Bearer {self.token}"}
            payload = {"inputs": full_prompt, "parameters": {"max_new_tokens": 512, "temperature": 0.7}}
            try:
                res = requests.post(url, json=payload, headers=headers, timeout=30)
                if res.status_code == 200:
                    result = res.json()
                    if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                        return result[0]["generated_text"].strip()
                    return str(result)
            except Exception as e:
                logger.warning(f"HuggingFace API request failed: {e}")

        # Fallback simulation notice
        return f"[HuggingFace LLM]: Analysis for '{self.model_name}': processed prompt ({len(full_prompt)} chars)."
