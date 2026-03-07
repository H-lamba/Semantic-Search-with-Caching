"""
Centralized logging configuration.
Replaces print() with structured logging throughout the application.
"""

import logging
import sys


def setup_logger(name="semantic_search", level=logging.INFO):
    """Create a configured logger instance.

    Usage:
        from app.core.logger import setup_logger
        logger = setup_logger(__name__)
        logger.info("Message")
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler with structured format
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
