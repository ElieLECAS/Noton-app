"""
Service MMR (Maximal Marginal Relevance) pour diversification du contexte RAG.

Formule MMR : λ · sim(q, d_i) - (1-λ) · max_j∈selected sim(d_i, d_j)

Objectif : éviter la redondance dans les passages envoyés au LLM en pénalisant
les documents trop similaires aux documents déjà sélectionnés.

Contrainte additionnelle : MMR_MAX_PER_PARENT pour éviter qu'une seule section
(parent_node_id) domine le contexte.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from llama_index.core.schema import NodeWithScore
from sqlalchemy import text
from sqlmodel import Session

from app.config import settings

logger = logging.getLogger(__name__)


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Similarité cosinus entre deux vecteurs."""
    a = np.array(vec1, dtype=float)
    b = np.array(vec2, dtype=float)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


def fetch_embeddings_for_chunks(
    session: Session,
    chunk_ids: List[int],
) -> Dict[int, List[float]]:
    """
    Récupère les embeddings de chunks depuis la base de données.
    
    Args:
        session: Session SQLModel
        chunk_ids: Liste d'IDs de chunks
        
    Returns:
        Dictionnaire {chunk_id: embedding} (fallback [] si embedding NULL)
    """
    if not chunk_ids:
        return {}
    
    # Requête SQL directe pour récupérer les embeddings
    query = text("""
        SELECT id, embedding
        FROM documentchunk
        WHERE id = ANY(:chunk_ids)
    """)
    
    result = session.execute(query, {"chunk_ids": chunk_ids})
    
    embeddings = {}
    for row in result:
        chunk_id = row.id
        embedding = row.embedding
        
        # Convertir pgvector en liste Python
        if embedding is not None:
            # embedding est un objet pgvector, on peut le cast en liste
            embeddings[chunk_id] = list(embedding)
        else:
            embeddings[chunk_id] = []
    
    logger.info(
        "Embeddings récupérés : %d/%d chunks ont un embedding",
        len([e for e in embeddings.values() if e]),
        len(chunk_ids),
    )
    
    return embeddings


def compute_mmr(
    query_embedding: List[float],
    candidates: List[Tuple[NodeWithScore, List[float]]],
    *,
    lambda_: float,
    k: int,
    max_per_parent: int,
) -> List[Tuple[NodeWithScore, float]]:
    """
    Sélection MMR avec contrainte max_per_parent.
    
    Args:
        query_embedding: Embedding de la requête utilisateur
        candidates: Liste (nœud, embedding) des candidats déjà rerankés
        lambda_: Trade-off pertinence (λ haut) vs diversité (λ bas), ex: 0.7
        k: Nombre de documents à sélectionner
        max_per_parent: Max de documents par parent_node_id (évite domination d'une section)
        
    Returns:
        Liste (nœud, score_mmr) triée par ordre de sélection MMR
    """
    if not candidates:
        return []
    
    if not query_embedding:
        logger.warning("MMR : query_embedding vide, fallback ordre rerank")
        return [(nws, 1.0 - i * 0.01) for i, (nws, _) in enumerate(candidates[:k])]
    
    # Filtrer les candidats qui ont un embedding non vide
    valid_candidates = [
        (nws, emb) for nws, emb in candidates if emb
    ]
    
    if not valid_candidates:
        logger.warning("MMR : aucun candidat avec embedding, fallback ordre rerank")
        return [(nws, 1.0 - i * 0.01) for i, (nws, _) in enumerate(candidates[:k])]
    
    selected: List[Tuple[NodeWithScore, float]] = []
    remaining = list(valid_candidates)
    
    # Compteur par parent_node_id
    parent_count: Dict[str, int] = {}
    
    for _ in range(min(k, len(valid_candidates))):
        if not remaining:
            break
        
        best_idx = None
        best_score = -float('inf')
        
        for idx, (nws, doc_emb) in enumerate(remaining):
            # Contrainte max_per_parent
            parent_id = getattr(nws.node, "metadata", {}).get("parent_node_id")
            if parent_id and parent_count.get(parent_id, 0) >= max_per_parent:
                continue
            
            # Terme 1 : similarité avec la requête
            sim_query = _cosine_similarity(query_embedding, doc_emb)
            
            # Terme 2 : max similarité avec les documents déjà sélectionnés
            max_sim_selected = 0.0
            if selected:
                for _, sel_emb in selected:
                    sim = _cosine_similarity(doc_emb, sel_emb)
                    max_sim_selected = max(max_sim_selected, sim)
            
            # Score MMR
            mmr_score = lambda_ * sim_query - (1.0 - lambda_) * max_sim_selected
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        # Si aucun candidat valide (tous bloqués par max_per_parent), arrêter
        if best_idx is None:
            logger.warning(
                "MMR : arrêt anticipé (max_per_parent=%d atteint pour tous les parents restants)",
                max_per_parent,
            )
            break
        
        # Sélectionner le meilleur
        nws, doc_emb = remaining.pop(best_idx)
        selected.append((nws, doc_emb))
        
        # Incrémenter le compteur parent
        parent_id = getattr(nws.node, "metadata", {}).get("parent_node_id")
        if parent_id:
            parent_count[parent_id] = parent_count.get(parent_id, 0) + 1
    
    # Convertir en (nœud, score_position) pour compatibilité
    # Le score est décroissant : premier sélectionné = score le plus haut
    result = [
        (nws, 1.0 - i * 0.01)
        for i, (nws, _) in enumerate(selected)
    ]
    
    logger.info(
        "MMR : %d documents sélectionnés (λ=%.2f, max_per_parent=%d), distribution parents=%s",
        len(result),
        lambda_,
        max_per_parent,
        dict(parent_count),
    )
    
    return result
