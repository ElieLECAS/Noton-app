"""
Service de reranking cross-encoder (MiniLM CPU-only) avec early stopping et guardrails statistiques.

Pipeline :
1. Early stopping : court-circuite le rerank si les scores RRF top-N sont déjà très élevés
2. Rerank cross-encoder : re-scorer paires (query, passage) avec ms-marco-MiniLM-L-6-v2
3. Guardrails : détection "bégaiement" (scores plats) → low_confidence_clarification
4. K dynamique : sélection softmax cumulée bornée entre MIN_DYNAMIC_K et MAX_DYNAMIC_K
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
from llama_index.core.schema import NodeWithScore

from app.config import settings

logger = logging.getLogger(__name__)

_cross_encoder = None
_cross_encoder_lock = threading.Lock()


@dataclass
class RerankResult:
    """Résultat du rerank avec métadonnées statistiques pour guardrails."""
    nodes: List[NodeWithScore]
    status: Literal["ok", "low_confidence_clarification", "early_stopped", "disabled"]
    reason: Optional[str]
    raw_scores: List[float]
    softmax_scores: List[float]
    gap_top1_top2: Optional[float]
    zscore_flatness: Optional[float]


def _get_cross_encoder():
    """Charge le modèle cross-encoder une seule fois (singleton lazy thread-safe)."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    
    with _cross_encoder_lock:
        if _cross_encoder is not None:
            return _cross_encoder
        
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder(
                settings.RERANKER_MODEL,
                device="cpu",
                max_length=512,
            )
            logger.info(
                "Cross-encoder chargé : %s (device=cpu, max_length=512)",
                settings.RERANKER_MODEL,
            )
        except Exception as e:
            logger.exception("Échec chargement cross-encoder : %s", e)
            raise
        
        return _cross_encoder


def should_early_stop(rrf_scores: List[float], threshold: float) -> bool:
    """
    Early stopping : court-circuite le rerank si la moyenne des top-N scores RRF
    dépasse un seuil de confiance très élevé (questions faciles).
    
    Args:
        rrf_scores: Scores RRF normalisés des candidats top-N
        threshold: Seuil moyen au-delà duquel on skip le rerank
        
    Returns:
        True si early stop (scores déjà excellents), False sinon
    """
    if not rrf_scores:
        return False
    
    # Prendre les EARLY_STOP_TOP_N premiers scores
    top_n = rrf_scores[: settings.EARLY_STOP_TOP_N]
    if not top_n:
        return False
    
    mean_score = float(np.mean(top_n))
    should_stop = mean_score >= threshold
    
    if should_stop:
        logger.info(
            "Early stop activé : mean(top-%d RRF)=%.3f >= %.3f",
            len(top_n),
            mean_score,
            threshold,
        )
    
    return should_stop


def rerank_nodes(
    query_text: str,
    nodes_with_score: List[NodeWithScore],
    *,
    char_cap: int,
    batch_size: int,
    detected_refs: Optional[List[str]] = None,
) -> List[Tuple[NodeWithScore, float]]:
    """
    Rerank un pool de nœuds avec le cross-encoder.

    Args:
        query_text: Question de l'utilisateur
        nodes_with_score: Pool de candidats à reranker
        char_cap: Limite de caractères par passage (évite over-feeding)
        batch_size: Taille de batch pour CrossEncoder.predict
        detected_refs: Références produit détectées (P1-D). Si un chunk contient
            une référence exacte, le préfixe [MATCH_REF] est ajouté au texte de
            la paire pour sensibiliser le cross-encoder.

    Returns:
        Liste (nœud, score_cross_encoder) triée par score décroissant
    """
    if not nodes_with_score:
        return []

    model = _get_cross_encoder()

    # Pré-compiler les patterns de référence exacte (P1-D)
    ref_patterns = []
    if detected_refs:
        import re
        ref_patterns = [
            re.compile(r'\b' + re.escape(r) + r'\b', re.IGNORECASE)
            for r in detected_refs
        ]

    # Tronquer le texte de chaque nœud à char_cap
    pairs = []
    for nws in nodes_with_score:
        text = getattr(nws.node, "text", "") or ""
        truncated = text[:char_cap] if len(text) > char_cap else text
        # P1-D : préfixe [MATCH_REF] si le chunk contient la référence exacte demandée
        if ref_patterns and any(pat.search(truncated) for pat in ref_patterns):
            truncated = f"[MATCH_REF] {truncated}"
        pairs.append([query_text, truncated])

    # Batch predict
    try:
        raw_scores = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )
    except Exception as e:
        logger.exception("Échec rerank cross-encoder : %s", e)
        # Fallback : garder les scores RRF originaux
        return [(nws, float(nws.score or 0.0)) for nws in nodes_with_score]

    # Associer chaque nœud à son score cross-encoder
    scored = list(zip(nodes_with_score, raw_scores))
    # Trier par score décroissant
    scored.sort(key=lambda x: x[1], reverse=True)

    logger.info(
        "Rerank cross-encoder : %d candidats → top-1 score=%.3f, top-3 scores=%s%s",
        len(scored),
        scored[0][1] if scored else 0.0,
        [round(s, 3) for _, s in scored[:3]],
        f" (MATCH_REF actif: {detected_refs})" if detected_refs else "",
    )

    return scored


def apply_dynamic_filtering(
    scored: List[Tuple[NodeWithScore, float]],
    *,
    min_k: int,
    max_k: int,
    softmax_cum_threshold: float,
    stutter_gap: float,
    zscore_flat_threshold: float,
) -> RerankResult:
    """
    Applique les guardrails statistiques et la sélection K dynamique.
    
    Guardrails :
    - Gap P@1-P@2 trop faible ET z-score plat → low_confidence_clarification
    
    K dynamique :
    - Sélectionne les nœuds dont la somme cumulée softmax atteint softmax_cum_threshold
    - Borné entre min_k et max_k
    
    Args:
        scored: Liste (nœud, score_cross_encoder) triée décroissant
        min_k: Minimum de documents à garder
        max_k: Maximum de documents à garder
        softmax_cum_threshold: Seuil de masse softmax cumulée (ex: 0.8)
        stutter_gap: Delta minimum entre P@1 et P@2 (ex: 0.05)
        zscore_flat_threshold: Seuil stdev softmax pour détecter planéité
        
    Returns:
        RerankResult avec status, nodes sélectionnés, et métadonnées stats
    """
    if not scored:
        return RerankResult(
            nodes=[],
            status="ok",
            reason="no_candidates",
            raw_scores=[],
            softmax_scores=[],
            gap_top1_top2=None,
            zscore_flatness=None,
        )
    
    raw_scores = [s for _, s in scored]
    
    # Softmax pour normaliser les scores
    raw_array = np.array(raw_scores, dtype=float)
    # Stabilité numérique : soustraire max
    exp_scores = np.exp(raw_array - np.max(raw_array))
    softmax_scores = (exp_scores / exp_scores.sum()).tolist()
    
    # Guardrail 1 : Gap P@1-P@2
    gap_top1_top2 = None
    if len(softmax_scores) >= 2:
        gap_top1_top2 = softmax_scores[0] - softmax_scores[1]
    
    # Guardrail 2 : Z-score / planéité (stdev des scores softmax)
    zscore_flatness = float(np.std(softmax_scores)) if len(softmax_scores) > 1 else 0.0
    
    # Détection "bégaiement" : gap faible ET distribution plate
    is_stuttering = False
    if gap_top1_top2 is not None:
        is_stuttering = (gap_top1_top2 < stutter_gap) and (zscore_flatness < zscore_flat_threshold)
    
    if is_stuttering:
        # Low confidence : garder seulement top 1-2 pour forcer clarification
        selected_nodes = [nws for nws, _ in scored[:2]]
        logger.warning(
            "Guardrail bégaiement déclenché : gap_P1-P2=%.4f < %.4f, stdev=%.4f < %.4f → top 1-2 seulement",
            gap_top1_top2,
            stutter_gap,
            zscore_flatness,
            zscore_flat_threshold,
        )
        return RerankResult(
            nodes=selected_nodes,
            status="low_confidence_clarification",
            reason=f"gap_top1_top2={gap_top1_top2:.4f} < {stutter_gap}, zscore={zscore_flatness:.4f} < {zscore_flat_threshold}",
            raw_scores=raw_scores,
            softmax_scores=softmax_scores,
            gap_top1_top2=gap_top1_top2,
            zscore_flatness=zscore_flatness,
        )
    
    # K dynamique : sélection par seuil de masse cumulée softmax
    cumsum = np.cumsum(softmax_scores)
    # Trouver l'index où cumsum >= threshold
    indices = np.where(cumsum >= softmax_cum_threshold)[0]
    if len(indices) > 0:
        # Premier indice atteignant le seuil
        k_dynamic = int(indices[0]) + 1
    else:
        # Si même tous les candidats n'atteignent pas le seuil, prendre tous
        k_dynamic = len(scored)
    
    # Borner entre min_k et max_k
    k_dynamic = max(min_k, min(k_dynamic, max_k))
    
    selected_nodes = [nws for nws, _ in scored[:k_dynamic]]
    
    logger.info(
        "K dynamique : %d candidats sélectionnés (cumsum softmax >= %.2f, borné [%d, %d])",
        k_dynamic,
        softmax_cum_threshold,
        min_k,
        max_k,
    )
    logger.debug(
        "Guardrails OK : gap_P1-P2=%.4f, stdev=%.4f",
        gap_top1_top2 or 0.0,
        zscore_flatness,
    )
    
    return RerankResult(
        nodes=selected_nodes,
        status="ok",
        reason=f"dynamic_k={k_dynamic}",
        raw_scores=raw_scores,
        softmax_scores=softmax_scores,
        gap_top1_top2=gap_top1_top2,
        zscore_flatness=zscore_flatness,
    )
