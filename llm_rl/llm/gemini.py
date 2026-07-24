"""
Gemini, Groq, and LlamaCpp LLM Backends for LLM-RL.
"""

import os
from typing import Optional
import requests
from llm_rl.llm.base import BaseLLM
from llm_rl.logger import logger


class GeminiLLM(BaseLLM):
    """
    Google Gemini LLM backend driver.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            msg = "[Gemini LLM Notice] GEMINI_API_KEY is missing."
            logger.warning(msg)
            return msg

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        text_content = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        payload = {"contents": [{"parts": [{"text": text_content}]}]}

        try:
            res = requests.post(url, json=payload, timeout=30)
            if res.status_code == 200:
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return f"[Gemini Error HTTP {res.status_code}]: {res.text}"
        except Exception as e:
            return f"[Gemini Error]: {e}"


class GroqLLM(BaseLLM):
    """
    Groq API LLM backend driver (Fast Llama/Mixtral inference).
    """

    def __init__(self, model_name: str = "llama-3.1-70b-versatile", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            msg = "[Groq LLM Notice] GROQ_API_KEY is missing."
            logger.warning(msg)
            return msg

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {"model": self.model_name, "messages": messages}
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=30)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"].strip()
            return f"[Groq Error HTTP {res.status_code}]: {res.text}"
        except Exception as e:
            return f"[Groq Error]: {e}"


class LlamaCppLLM(BaseLLM):
    """
    Llama.cpp local server or Python bindings backend driver.
    """

    def __init__(self, model_name: str = "local-llama", host: str = "http://localhost:8080", **kwargs):
        super().__init__(model_name, **kwargs)
        self.host = host.rstrip('/')

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.host}/completion"
        full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:" if system_prompt else prompt
        payload = {"prompt": full_prompt, "n_predict": 512}

        try:
            res = requests.post(url, json=payload, timeout=30)
            if res.status_code == 200:
                return res.json().get("content", "").strip()
            return f"[LlamaCpp Error HTTP {res.status_code}]: {res.text}"
        except Exception as e:
            return f"[LlamaCpp Server Notice] Could not connect to llama.cpp server at {self.host}: {e}"
