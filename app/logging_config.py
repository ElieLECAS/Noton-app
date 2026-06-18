"""
Configuration logging compatible uvicorn.

Branche un handler stdout sur le logger root (avec flush immédiat) pour que
tous les modules ``app.*`` et les appels Mistral soient visibles dans Docker.
"""
from __future__ import annotations

import logging
import os
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


class _FlushingStreamHandler(logging.StreamHandler):
    """Écrit et flush immédiatement (évite les logs perdus avec uvicorn/tqdm)."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_app_logging() -> None:
    """Idempotent : configure root + namespace app."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    handler = _FlushingStreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, _FlushingStreamHandler) for h in root.handlers):
        root.addHandler(handler)

    app_logger = logging.getLogger("app")
    app_logger.setLevel(level)
    app_logger.propagate = True
    for old_handler in list(app_logger.handlers):
        app_logger.removeHandler(old_handler)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(level)

    _CONFIGURED = True
    logging.getLogger("app.main").info(
        "Logging configuré (niveau=%s, handler=stdout+flush)",
        level_name,
    )
