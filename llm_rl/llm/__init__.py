"""
LLM Backends Factory for LLM-RL.
"""

from typing import Dict, Type
from llm_rl.llm.base import BaseLLM
from llm_rl.llm.ollama import OllamaLLM
from llm_rl.llm.openai import OpenAILLM
from llm_rl.llm.huggingface import HuggingFaceLLM
from llm_rl.llm.anthropic import AnthropicLLM
from llm_rl.llm.gemini import GeminiLLM, GroqLLM, LlamaCppLLM
from llm_rl.logger import logger


LLM_BACKENDS: Dict[str, Type[BaseLLM]] = {
    "ollama": OllamaLLM,
    "openai": OpenAILLM,
    "huggingface": HuggingFaceLLM,
    "hf": HuggingFaceLLM,
    "anthropic": AnthropicLLM,
    "claude": AnthropicLLM,
    "gemini": GeminiLLM,
    "google": GeminiLLM,
    "groq": GroqLLM,
    "llamacpp": LlamaCppLLM,
    "llama.cpp": LlamaCppLLM,
}


def get_llm_backend(backend: str = "ollama", model_name: str = "llama3.1:8b", **kwargs) -> BaseLLM:
    """
    Factory function to instantiate appropriate LLM backend driver.

    Args:
        backend: Provider identifier ('ollama', 'openai', 'huggingface', 'anthropic', 'gemini', 'groq', 'llamacpp').
        model_name: Model identifier name.
        **kwargs: Additional parameters passed to LLM initializer.

    Returns:
        BaseLLM instance.
    """
    backend_key = backend.lower().strip()
    if backend_key not in LLM_BACKENDS:
        logger.warning(f"Unknown backend '{backend}', falling back to Ollama backend.")
        backend_key = "ollama"

    llm_cls = LLM_BACKENDS[backend_key]
    return llm_cls(model_name=model_name, **kwargs)
