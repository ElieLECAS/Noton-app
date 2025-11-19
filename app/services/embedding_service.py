from typing import List, Optional
import logging
import os
import numpy as np
from fastembed import TextEmbedding
from app.embedding_config import EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

# Modèle d'embedding FastEmbed par défaut (384 dimensions)
# Modèle léger et rapide : BAAI/bge-small-en-v1.5
# Alternatives disponibles : BAAI/bge-base-en-v1.5 (768 dim), BAAI/bge-large-en-v1.5 (1024 dim)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Instance singleton du modèle d'embedding
_embedder: Optional[TextEmbedding] = None


def get_embedder() -> TextEmbedding:
    """Charger le modèle FastEmbed (singleton)"""
    global _embedder
    if _embedder is None:
        logger.info(f"Chargement du modèle d'embedding FastEmbed: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSION} dimensions)")
        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        logger.info(f"Modèle d'embedding FastEmbed chargé avec succès ({EMBEDDING_DIMENSION} dimensions)")
    return _embedder


def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Génère un embedding pour un texte donné en utilisant FastEmbed (modèle local).
    
    Args:
        text: Le texte à encoder
        
    Returns:
        Liste de floats représentant l'embedding ou None si erreur
    """
    if not text or not text.strip():
        logger.warning("Texte vide fourni pour génération d'embedding")
        return None
    
    try:
        embedder = get_embedder()
        
        # FastEmbed retourne un générateur, on prend le premier élément
        # Le modèle génère déjà des embeddings normalisés pour la similarité cosinus
        embeddings = list(embedder.embed([text.strip()]))
        
        if embeddings and len(embeddings) > 0:
            embedding = embeddings[0]
            # Convertir en liste de floats
            if isinstance(embedding, np.ndarray):
                return embedding.tolist()
            elif isinstance(embedding, list):
                return [float(x) for x in embedding]
            else:
                return list(embedding)
        else:
            logger.error("Aucun embedding généré par FastEmbed")
            return None
    except Exception as e:
        logger.error(f"Erreur lors de la génération d'embedding: {e}", exc_info=True)
        return None


def generate_embeddings_batch(texts: List[str], batch_size: int = 8) -> List[Optional[List[float]]]:
    """
    Génère des embeddings pour plusieurs textes en batch via FastEmbed (modèle local).
    
    Args:
        texts: Liste de textes à encoder
        batch_size: Nombre de textes à traiter par batch (FastEmbed gère automatiquement)
        
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
        embedder = get_embedder()
        
        # FastEmbed peut traiter plusieurs textes en une seule passe
        # Il retourne un générateur d'embeddings
        embeddings_generator = embedder.embed(valid_texts, batch_size=batch_size)
        embeddings_list = list(embeddings_generator)
        
        # Reconstruire la liste complète avec None pour les textes vides ou en erreur
        result = [None] * len(texts)
        for idx, embedding in zip(valid_indices, embeddings_list):
            try:
                # Convertir en liste de floats
                if isinstance(embedding, np.ndarray):
                    result[idx] = embedding.tolist()
                elif isinstance(embedding, list):
                    result[idx] = [float(x) for x in embedding]
                else:
                    result[idx] = list(embedding)
            except Exception as e:
                logger.error(f"Erreur lors de la conversion de l'embedding pour l'index {idx}: {e}")
                result[idx] = None
        
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

