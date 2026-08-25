"""Embeddings via API Mistral (mistral-embed) — pas de PyTorch local."""
from __future__ import annotations

import logging
import time
from threading import Lock
from typing import List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 16
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5
# Limite pratique par requête API (éviter payloads trop gros)
MAX_INPUTS_PER_REQUEST = 64

_client = None
_client_lock = Lock()


def _get_mistral_client():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        if not settings.MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY n'est pas configurée pour les embeddings")
        from mistralai import Mistral

        _client = Mistral(api_key=settings.MISTRAL_API_KEY)
        logger.info(
            "Client Mistral Embeddings initialisé (model=%s, dim=%s)",
            settings.EMBEDDING_MODEL,
            settings.EMBEDDING_DIMENSION,
        )
        return _client


def _embed_texts_api(texts: List[str]) -> List[List[float]]:
    """Appelle POST /v1/embeddings pour une liste de textes non vides."""
    client = _get_mistral_client()
    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "[MISTRAL] Embeddings API — model=%s, batch=%d texte(s)",
                settings.EMBEDDING_MODEL,
                len(texts),
            )
            response = client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                inputs=texts,
            )
            vectors: List[List[float]] = []
            for item in response.data or []:
                emb = getattr(item, "embedding", None) or []
                vectors.append([float(x) for x in emb])
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"Mistral embeddings: {len(vectors)} vecteurs pour {len(texts)} entrées"
                )
            logger.info("[MISTRAL] Embeddings OK — %d vecteur(s), dim=%d", len(vectors), len(vectors[0]) if vectors else 0)
            return vectors
        except Exception as exc:
            last_err = exc
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "Mistral embeddings tentative %s/%s échouée (%s), retry dans %.1fs",
                    attempt,
                    MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)
    raise RuntimeError(f"Mistral embeddings échoué: {last_err}") from last_err


def generate_embedding(text: str) -> Optional[List[float]]:
    """Génère un embedding pour un texte (requête ou passage)."""
    if not text or not text.strip():
        logger.warning("Texte vide fourni pour génération d'embedding")
        return None
    try:
        logger.info(
            "[MISTRAL] Embedding requête RAG — model=%s, chars=%d",
            settings.EMBEDDING_MODEL,
            len(text.strip()),
        )
        vectors = _embed_texts_api([text.strip()])
        dim = len(vectors[0]) if vectors and vectors[0] else 0
        logger.info("[MISTRAL] Embedding requête OK — dim=%d", dim)
        return vectors[0] if vectors else None
    except Exception as e:
        logger.error("Erreur génération embedding Mistral: %s", e, exc_info=True)
        return None


def generate_embeddings_batch(
    texts: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Optional[List[float]]]:
    """
    Génère des embeddings en batch via l'API Mistral.
    Ne pas appeler chunk par chunk : regrouper via cette fonction.
    """
    if not texts:
        return []

    result: List[Optional[List[float]]] = [None] * len(texts)
    valid_texts: List[str] = []
    valid_indices: List[int] = []

    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts.append(text.strip())
            valid_indices.append(i)

    if not valid_texts:
        return result

    effective_batch = max(1, min(int(batch_size or DEFAULT_BATCH_SIZE), MAX_INPUTS_PER_REQUEST))

    try:
        for batch_start in range(0, len(valid_texts), effective_batch):
            batch_end = min(batch_start + effective_batch, len(valid_texts))
            batch_texts = valid_texts[batch_start:batch_end]
            batch_indices = valid_indices[batch_start:batch_end]

            try:
                vectors = _embed_texts_api(batch_texts)
            except Exception as exc:
                logger.error(
                    "Échec batch embeddings Mistral [%s:%s]: %s",
                    batch_start,
                    batch_end,
                    exc,
                    exc_info=True,
                )
                continue

            for idx, vector in zip(batch_indices, vectors):
                if vector and len(vector) == settings.EMBEDDING_DIMENSION:
                    result[idx] = vector
                elif vector:
                    logger.warning(
                        "Dimension embedding inattendue: %s (attendu %s)",
                        len(vector),
                        settings.EMBEDDING_DIMENSION,
                    )
                    result[idx] = vector

        return result
    except Exception as e:
        logger.error("Erreur batch embeddings Mistral: %s", e, exc_info=True)
        return [None] * len(texts)
