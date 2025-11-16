from typing import List, Optional
from sentence_transformers import SentenceTransformer
import logging
import os
from app.embedding_config import EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

# Modèle singleton chargé une seule fois
_model: Optional[SentenceTransformer] = None

# Modèle léger et rapide : all-MiniLM-L6-v2 (384 dimensions)
# Alternative plus lourde mais meilleure qualité : all-mpnet-base-v2 (768 dimensions)
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_model() -> SentenceTransformer:
    """Charger le modèle sentence-transformers (singleton) avec optimisations CPU"""
    global _model
    if _model is None:
        logger.info(f"Chargement du modèle d'embedding: {MODEL_NAME} ({EMBEDDING_DIMENSION} dimensions)")
        # Options pour optimiser sur CPU
        _model = SentenceTransformer(
            MODEL_NAME,
            device='cpu',  # Forcer CPU pour éviter les problèmes GPU
            model_kwargs={'low_cpu_mem_usage': True}  # Réduire l'utilisation mémoire
        )
        logger.info(f"Modèle d'embedding chargé avec succès ({EMBEDDING_DIMENSION} dimensions)")
    return _model


def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Génère un embedding pour un texte donné.
    
    Args:
        text: Le texte à encoder
        
    Returns:
        Liste de floats représentant l'embedding ou None si erreur
    """
    if not text or not text.strip():
        logger.warning("Texte vide fourni pour génération d'embedding")
        return None
    
    try:
        model = get_model()
        # Générer l'embedding avec options optimisées
        # batch_size=1 pour réduire la mémoire, show_progress_bar=False pour la performance
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True  # Normaliser pour similarité cosinus
        )
        # Convertir en liste de floats
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Erreur lors de la génération d'embedding: {e}")
        return None


def generate_embeddings_batch(texts: List[str], batch_size: int = 8) -> List[Optional[List[float]]]:
    """
    Génère des embeddings pour plusieurs textes en batch (plus efficace).
    
    Args:
        texts: Liste de textes à encoder
        batch_size: Nombre de textes à traiter en parallèle
        
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
        model = get_model()
        # Générer les embeddings en batch (beaucoup plus rapide)
        embeddings = model.encode(
            valid_texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        # Reconstruire la liste complète avec None pour les textes vides
        result = [None] * len(texts)
        for idx, embedding in zip(valid_indices, embeddings):
            result[idx] = embedding.tolist()
        
        return result
    except Exception as e:
        logger.error(f"Erreur lors de la génération d'embeddings en batch: {e}")
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

