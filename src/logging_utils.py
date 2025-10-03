"""Utilities for consistent logging configuration across the project."""
from __future__ import annotations

import logging
from typing import Optional

DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(level: int = logging.INFO, *, format: Optional[str] = None) -> None:
    """Initialise global logging configuration if not already configured."""

    if format is None:
        format = DEFAULT_LOG_FORMAT

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format=format)
    else:
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured with the project defaults."""

    configure_logging()
    return logging.getLogger(name)
