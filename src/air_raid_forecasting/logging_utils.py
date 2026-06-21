"""Project-wide logging configuration.

A single :func:`get_logger` entry point gives every module a consistently
formatted logger whose level is controlled by the ``ARF_LOG_LEVEL`` environment
variable (default ``INFO``). Calling it repeatedly is safe — handlers are only
attached once.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger once with a stream handler to stdout."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.environ.get("ARF_LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger()
    root.setLevel(lvl)
    # Avoid duplicate handlers when running under pytest / uvicorn reloaders.
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    # Tame noisy third-party loggers.
    for noisy in ("matplotlib", "PIL", "cmdstanpy", "prophet", "urllib3", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for *name* (typically ``__name__``)."""
    setup_logging()
    return logging.getLogger(name)
