"""
Base abstract class for LLM backends in LLM-RL.
"""

from abc import ABC, abstractmethod
import asyncio
from typing import Dict, List, Optional
from llm_rl.logger import logger


class BaseLLM(ABC):
    """Abstract Base Class for LLM providers."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Synchronously generate text response from the LLM.

        Args:
            prompt: User/Task prompt.
            system_prompt: Optional system prompt context.

        Returns:
            Generated text string response.
        """
        pass

    async def async_generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Asynchronously generate text response from the LLM. Default implementation offloads to executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate, prompt, system_prompt)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Chat completion interface given conversation message history.

        Args:
            messages: List of dicts with keys 'role' ('system', 'user', 'assistant') and 'content'.

        Returns:
            Assistant response string.
        """
        system_content = None
        user_prompt = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_content = content
            elif role == "user":
                user_prompt += f"\nUser: {content}"
            elif role == "assistant":
                user_prompt += f"\nAssistant: {content}"

        return self.generate(user_prompt.strip(), system_prompt=system_content)
