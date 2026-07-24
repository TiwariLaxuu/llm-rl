"""
Helper functions and formatting utilities for LLM-RL.
"""

import socket
from typing import Any, List, Dict


def find_free_port(start_port: int = 7860, max_tries: int = 50) -> int:
    """
    Find an available TCP port starting from start_port.
    """
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    return start_port


def format_observation(obs: Any) -> str:
    """Format observation array or object into concise string representation."""
    if hasattr(obs, "tolist"):
        obs = obs.tolist()
    if isinstance(obs, list):
        if len(obs) > 10:
            return f"[{', '.join(f'{x:.3f}' for x in obs[:5])}, ... {len(obs)-5} more elements]"
        return f"[{', '.join(f'{x:.3f}' if isinstance(x, (int, float)) else str(x) for x in obs)}]"
    return str(obs)


def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """Format chat history into text string."""
    output = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        output.append(f"**{role}**: {content}")
    return "\n\n".join(output)
