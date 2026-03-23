from typing import List, Optional
import logging
import os
import time
from pathlib import Path
from threading import Lock
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
DEFAULT_BATCH_SIZE = 16
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5

_embed_model: Optional[HuggingFaceEmbedding] = None
_embed_model_lock = Lock()


def _check_model_cache(model_name: str) -> bool:
    """Vérifie si le modèle existe déjà dans le cache HuggingFace."""
    # Vérifier plusieurs emplacements possibles du cache
    possible_cache_dirs = [
        os.getenv("HF_HOME"),
        os.getenv("HUGGINGFACE_HUB_CACHE"),
        os.path.expanduser("~/.cache/huggingface"),
        Path("/root/.cache/huggingface"),  # Docker par défaut
    ]
    
    model_slug = model_name.replace("/", "--")
    
    for cache_dir in possible_cache_dirs:
        if not cache_dir:
            continue
        cache_path = Path(cache_dir)
        # Vérifier dans hub/models--{org}--{model}
        model_path = cache_path / "hub" / f"models--{model_slug}"
        if model_path.exists() and any(model_path.iterdir()):
            return True
        # Vérifier aussi dans sentence_transformers (ancien format)
        st_path = cache_path / "sentence_transformers" / model_slug
        if st_path.exists() and any(st_path.iterdir()):
            return True
    
    return False


def _get_embed_model() -> HuggingFaceEmbedding:
    """Retourne un client HuggingFaceEmbedding singleton thread-safe."""
    global _embed_model
    if _embed_model is None:
        with _embed_model_lock:
            if _embed_model is None:
                # Vérifier si le modèle est déjà en cache
                from_cache = _check_model_cache(EMBEDDING_MODEL)
                cache_status = "depuis le cache" if from_cache else "téléchargement en cours"
                
                # Si le modèle est en cache, activer le mode offline pour éviter les vérifications réseau
                if from_cache and not os.getenv("HF_HUB_OFFLINE"):
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    logger.debug("Mode offline activé (modèle en cache)")
                
                logger.info(
                    "Initialisation du modèle d'embeddings HuggingFace (%s): %s",
                    cache_status,
                    EMBEDDING_MODEL,
                )
                
                _embed_model = HuggingFaceEmbedding(
                    model_name=EMBEDDING_MODEL,
                    device=EMBEDDING_DEVICE,
                    embed_batch_size=DEFAULT_BATCH_SIZE,
                )
                logger.info(
                    "✅ Client HuggingFaceEmbedding initialisé (model=%s, device=%s)",
                    EMBEDDING_MODEL,
                    EMBEDDING_DEVICE,
                )
    return _embed_model


def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Génère un embedding pour un texte donné en utilisant HuggingFace (modèle local).
    
    Args:
        text: Le texte à encoder
        
    Returns:
        Liste de floats représentant l'embedding ou None si erreur
    """
    if not text or not text.strip():
        logger.warning("Texte vide fourni pour génération d'embedding")
        return None
    
    try:
        embed_model = _get_embed_model()
        embedding = embed_model.get_text_embedding(text.strip())
        if not embedding:
            logger.error("Aucun embedding généré par HuggingFace/LlamaIndex")
            return None
        return [float(x) for x in embedding]
    except Exception as e:
        logger.error(f"Erreur lors de la génération d'embedding: {e}", exc_info=True)
        return None


def generate_embeddings_batch(texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[Optional[List[float]]]:
    """
    Génère des embeddings pour plusieurs textes en batch via HuggingFace (modèle local).
    IMPORTANT: Ne jamais envoyer les chunks un par un, toujours utiliser cette fonction en batch.
    
    Args:
        texts: Liste de textes à encoder
        batch_size: Nombre de textes à traiter par batch
        
    Returns:
        Liste d'embeddings (ou None si erreur pour un texte)
    """
    if not texts:
        return []
    
    # Filtrer les textes vides
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts.append(text.strip())
            valid_indices.append(i)
    
    if not valid_texts:
        return [None] * len(texts)
    
    try:
        result = [None] * len(texts)
        embed_model = _get_embed_model()
        effective_batch_size = max(1, int(batch_size or DEFAULT_BATCH_SIZE))

        for batch_start in range(0, len(valid_texts), effective_batch_size):
            batch_end = min(batch_start + effective_batch_size, len(valid_texts))
            batch_texts = valid_texts[batch_start:batch_end]
            batch_indices = valid_indices[batch_start:batch_end]

            embeddings_batch: Optional[List[List[float]]] = None
            last_error: Optional[Exception] = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    embeddings_batch = embed_model.get_text_embedding_batch(batch_texts)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < MAX_RETRIES:
                        sleep_time = RETRY_BACKOFF_SECONDS * attempt
                        logger.warning(
                            "Echec embedding batch %s-%s tentative %s/%s, retry dans %.1fs: %s",
                            batch_start,
                            batch_end,
                            attempt,
                            MAX_RETRIES,
                            sleep_time,
                            exc,
                        )
                        time.sleep(sleep_time)

            if embeddings_batch is None:
                logger.error(
                    "Echec définitif embedding batch %s-%s: %s",
                    batch_start,
                    batch_end,
                    last_error,
                )
                continue

            for idx, embedding in zip(batch_indices, embeddings_batch):
                result[idx] = [float(x) for x in embedding] if embedding else None

        return result
    except Exception as e:
        logger.error(f"Erreur lors de la génération d'embeddings en batch: {e}", exc_info=True)
        return [None] * len(texts)


def generate_note_embedding(title: str, content: Optional[str] = None) -> Optional[List[float]]:
    """
    Génère un embedding pour une note complète (titre + contenu).
    
    Args:
        title: Le titre de la note
        content: Le contenu optionnel de la note
        
    Returns:
        Liste de floats représentant l'embedding ou None si erreur
    """
    # Combiner titre et contenu pour créer un texte complet
    text_parts = [title]
    if content and content.strip():
        text_parts.append(content.strip())
    
    combined_text = " ".join(text_parts)
    
    if not combined_text.strip():
        logger.warning("Note vide (titre et contenu vides)")
        return None
    
    return generate_embedding(combined_text)

