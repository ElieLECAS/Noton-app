from typing import List, Optional
import logging
import os
import httpx
import asyncio
from app.embedding_config import EMBEDDING_DIMENSION
from app.config import settings

logger = logging.getLogger(__name__)

# Modèle d'embedding Ollama Nomic Embed v1 (768 dimensions)
# Très rapide et performant pour l'embedding local
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Génère un embedding pour un texte donné en utilisant Ollama Nomic Embed (modèle local).
    
    Args:
        text: Le texte à encoder
        
    Returns:
        Liste de floats représentant l'embedding ou None si erreur
    """
    if not text or not text.strip():
        logger.warning("Texte vide fourni pour génération d'embedding")
        return None
    
    try:
        # Utiliser l'API HTTP d'Ollama pour générer l'embedding
        payload = {
            "model": EMBEDDING_MODEL,
            "prompt": text.strip()
        }
        
        # Utiliser httpx de manière synchrone (ou asyncio.run pour async)
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{settings.OLLAMA_BASE_URL}/api/embeddings",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
        
        if result and "embedding" in result:
            embedding = result["embedding"]
            # Convertir en liste de floats
            return [float(x) for x in embedding]
        else:
            logger.error("Aucun embedding généré par Ollama")
            return None
    except Exception as e:
        logger.error(f"Erreur lors de la génération d'embedding: {e}", exc_info=True)
        return None


def generate_embeddings_batch(texts: List[str], batch_size: int = 32) -> List[Optional[List[float]]]:
    """
    Génère des embeddings pour plusieurs textes en batch via Ollama Nomic Embed (modèle local).
    IMPORTANT: Ne jamais envoyer les chunks un par un, toujours utiliser cette fonction en batch.
    
    Args:
        texts: Liste de textes à encoder
        batch_size: Nombre de textes à traiter par batch (Ollama gère efficacement les batches)
        
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
        # Traiter par batches pour optimiser les performances
        result = [None] * len(texts)
        
        # Utiliser httpx pour traiter les batches
        with httpx.Client(timeout=60.0) as client:
            # Traiter tous les textes valides en batches
            for batch_start in range(0, len(valid_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(valid_texts))
                batch_texts = valid_texts[batch_start:batch_end]
                batch_indices = valid_indices[batch_start:batch_end]
                
                try:
                    # Traiter chaque texte du batch (Ollama API nécessite un appel par texte)
                    # Mais on utilise une connexion HTTP réutilisable pour optimiser
                    for text, idx in zip(batch_texts, batch_indices):
                        try:
                            payload = {
                                "model": EMBEDDING_MODEL,
                                "prompt": text
                            }
                            
                            response = client.post(
                                f"{settings.OLLAMA_BASE_URL}/api/embeddings",
                                json=payload
                            )
                            response.raise_for_status()
                            embedding_result = response.json()
                            
                            if embedding_result and "embedding" in embedding_result:
                                result[idx] = [float(x) for x in embedding_result["embedding"]]
                            else:
                                logger.warning(f"Aucun embedding généré pour le texte à l'index {idx}")
                                result[idx] = None
                        except Exception as e:
                            logger.error(f"Erreur lors de la génération d'embedding pour l'index {idx}: {e}")
                            result[idx] = None
                            
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du batch {batch_start}-{batch_end}: {e}")
                    # Marquer tous les textes du batch comme None
                    for idx in batch_indices:
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

