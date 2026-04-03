"""
Journal fichier dédié au traitement des documents bibliothèque / espaces.

Fichier par défaut : logs/library_document_processing.log (racine du projet).
Surcharge : variable d'environnement LIBRARY_DOCUMENT_LOG_PATH.
"""
from __future__ import annotations

import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, List

_LOCK = threading.Lock()
_CONFIGURED = False

LOGGER_NAME = "noton.library_document"


def _default_log_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "logs" / "library_document_processing.log"


def setup_library_document_file_logging() -> Path:
    """Idempotent : ajoute le handler fichier au logger dédié (une fois par processus)."""
    global _CONFIGURED
    with _LOCK:
        if _CONFIGURED:
            return Path(os.environ.get("LIBRARY_DOCUMENT_LOG_PATH", str(_default_log_path())))
        log_path = Path(os.environ.get("LIBRARY_DOCUMENT_LOG_PATH", str(_default_log_path())))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        lib_logger = logging.getLogger(LOGGER_NAME)
        lib_logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(
            str(log_path),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        lib_logger.addHandler(handler)
        lib_logger.propagate = False
        _CONFIGURED = True
        lib_logger.info(
            "========== Journal bibliothèque/espaces prêt (fichier=%s) ==========",
            log_path.resolve(),
        )
        return log_path.resolve()


def get_library_document_logger() -> logging.Logger:
    """Logger à utiliser pour tout le pipeline document bibliothèque."""
    setup_library_document_file_logging()
    return logging.getLogger(LOGGER_NAME)


def log_chunk_inventory(
    ld: logging.Logger,
    document_id: int,
    chunks: List[Any],
    context: str,
) -> None:
    """Résumé des chunks (feuilles, parents, présence de node_id) pour analyse RAG."""
    if not chunks:
        ld.info(
            "[%s] document_id=%s — aucun chunk après cette étape.",
            context,
            document_id,
        )
        return
    n = len(chunks)
    leaves = sum(1 for c in chunks if getattr(c, "is_leaf", True))
    parents = n - leaves
    with_node_id = sum(1 for c in chunks if getattr(c, "node_id", None))
    with_parent_ref = sum(1 for c in chunks if getattr(c, "parent_node_id", None))
    ld.info(
        "[%s] document_id=%s — inventaire: total=%d feuilles(is_leaf=True)=%d "
        "parents(is_leaf=False)=%d lignes_avec_node_id=%d lignes_avec_parent_node_id=%d",
        context,
        document_id,
        n,
        leaves,
        parents,
        with_node_id,
        with_parent_ref,
    )
