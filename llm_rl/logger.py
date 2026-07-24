"""
Logger module for LLM-RL.
Provides structured logging for training, LLM interactions, and monitoring events.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str = "llm_rl", level: int = logging.INFO) -> logging.Logger:
    """
    Get or create a logger configured for LLM-RL.

    Args:
        name: Logger name.
        level: Logging level (default: logging.INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


logger = get_logger()
