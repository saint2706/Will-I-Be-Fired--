"""Utilities for consistent logging configuration across the project.

This module provides simple helper functions to ensure that all other modules
in the project use a consistent logging setup. It defines a default log
format and provides a `get_logger` function that should be used to obtain a
logger instance.

By centralizing the configuration, we can easily change the log level or
format for the entire application in one place.
"""
from __future__ import annotations

import logging
from typing import Optional

# The default format string for log messages.
# Example: 2023-10-27 10:30:00,123 INFO [my_module] My log message
DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(level: int = logging.INFO, *, log_format: Optional[str] = None) -> None:
    """Initialize or update the global logging configuration.

    This function sets up the root logger for the application. If no handlers
    are configured, it calls `logging.basicConfig`. If handlers already
    exist (e.g., in an environment like Streamlit that pre-configures
    logging), it updates the level on the root logger and its handlers instead.

    Parameters
    ----------
    level:
        The logging level to set (e.g., `logging.INFO`, `logging.DEBUG`).
    log_format:
        The format string for log messages. If None, `DEFAULT_LOG_FORMAT`
        is used.
    """
    if log_format is None:
        log_format = DEFAULT_LOG_FORMAT

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # If no handlers are configured, initialize with basicConfig.
        # This is the standard case for scripts.
        logging.basicConfig(level=level, format=log_format)
        logging.info("Initialized new logging configuration.")
    else:
        # If handlers already exist, just update their levels.
        # This is common in environments like Streamlit or when imported.
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
        logging.info("Updated existing logging configuration to level %s.", logging.getLevelName(level))


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured with project defaults.

    This is the main entry point for obtaining a logger in any other module.
    It ensures that `configure_logging` is called at least once before
    returning the logger instance.

    Parameters
    ----------
    name:
        The name for the logger, typically `__name__` of the calling module.

    Returns
    -------
    A `logging.Logger` instance.
    """
    # Ensure that logging is configured before returning the logger.
    configure_logging()
    return logging.getLogger(name)
