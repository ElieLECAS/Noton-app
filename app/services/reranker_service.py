"""
Service de reranking cross-encoder (MiniLM CPU-only) avec early stopping et guardrails statistiques.

Pipeline :
1. Early stopping : court-circuite le rerank si les scores RRF top-N sont déjà très élevés
2. Rerank cross-encoder : re-scorer paires (query, passage) avec ms-marco-MiniLM-L-6-v2
3. Guardrails : détection "bégaiement" (scores plats) → low_confidence_clarification
4. K dynamique : sélection softmax cumulée bornée entre MIN_DYNAMIC_K et MAX_DYNAMIC_K

Métriques de troncature :
- Longueur chars/tokens avant rerank
- Ratio de chunks tronqués
- Latence P50/P95 du rerank
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

# Métriques de troncature (thread-safe accumulators)
_truncation_stats = {
    "total_chunks": 0,
    "truncated_chunks": 0,
    "total_chars_before": 0,
    "total_chars_after": 0,
    "rerank_latencies": [],
}
_stats_lock = threading.Lock()


def get_truncation_stats() -> dict:
    """Retourne les statistiques de troncature accumulées."""
    with _stats_lock:
        total = _truncation_stats["total_chunks"]
        truncated = _truncation_stats["truncated_chunks"]
        chars_before = _truncation_stats["total_chars_before"]
        chars_after = _truncation_stats["total_chars_after"]
        latencies = _truncation_stats["rerank_latencies"].copy()
        
        return {
            "total_chunks": total,
            "truncated_chunks": truncated,
            "truncation_ratio": truncated / total if total > 0 else 0.0,
            "avg_chars_before": chars_before / total if total > 0 else 0,
            "avg_chars_after": chars_after / total if total > 0 else 0,
            "avg_reduction_pct": (
                (1.0 - chars_after / chars_before) * 100 if chars_before > 0 else 0.0
            ),
            "rerank_latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "rerank_latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "rerank_latency_p99_ms": float(np.percentile(latencies, 99)) if latencies else 0.0,
            "rerank_count": len(latencies),
        }


def reset_truncation_stats() -> None:
    """Réinitialise les statistiques de troncature."""
    with _stats_lock:
        _truncation_stats["total_chunks"] = 0
        _truncation_stats["truncated_chunks"] = 0
        _truncation_stats["total_chars_before"] = 0
        _truncation_stats["total_chars_after"] = 0
        _truncation_stats["rerank_latencies"] = []


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
    
    NOTE: Désactivé par défaut (EARLY_STOP_ENABLED=False) car :
    - Latence MiniLM CPU acceptable pour RERANK_POOL=30 (~50-100ms)
    - Risque de skip sur des faux-positifs RRF (fusion ne garantit pas qualité absolue)
    - Gain latence marginal vs risque de dégradation qualité
    
    Pour l'activer : EARLY_STOP_ENABLED=True dans config + appel dans search pipeline.
    
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


async def rerank_nodes(
    query_text: str,
    nodes_with_score: List[NodeWithScore],
    *,
    char_cap: int,
    batch_size: int,
) -> List[Tuple[NodeWithScore, float]]:
    """
    Rerank un pool de nœuds avec le cross-encoder local.
    Collecte des métriques de troncature pour observabilité.
    
    Args:
        query_text: Question de l'utilisateur
        nodes_with_score: Pool de candidats à reranker
        char_cap: Limite de caractères par passage (évite over-feeding)
        batch_size: Taille de batch pour CrossEncoder.predict
        
    Returns:
        Liste (nœud, score_cross_encoder) triée par score décroissant
    """
    if not nodes_with_score:
        return []
    
    import time
    t_start = time.perf_counter()
    
    # Tronquer le texte de chaque nœud à char_cap + collecter métriques
    pairs = []
    truncation_count = 0
    total_chars_before = 0
    total_chars_after = 0
    
    for nws in nodes_with_score:
        text = getattr(nws.node, "text", "") or ""
        text_len = len(text)
        total_chars_before += text_len
        
        if text_len > char_cap:
            truncated = text[:char_cap]
            truncation_count += 1
        else:
            truncated = text
        
        total_chars_after += len(truncated)
        pairs.append([query_text, truncated])
    
    model = _get_cross_encoder()
    # Batch predict
    try:
        raw_scores = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )
    except Exception as e:
        logger.exception("Échec rerank cross-encoder local : %s", e)
        # Fallback : garder les scores RRF originaux
        return [(nws, float(nws.score or 0.0)) for nws in nodes_with_score]
    
    # Collecter latence
    t_elapsed = (time.perf_counter() - t_start) * 1000  # ms
    
    # Mettre à jour statistiques thread-safe
    with _stats_lock:
        _truncation_stats["total_chunks"] += len(nodes_with_score)
        _truncation_stats["truncated_chunks"] += truncation_count
        _truncation_stats["total_chars_before"] += total_chars_before
        _truncation_stats["total_chars_after"] += total_chars_after
        _truncation_stats["rerank_latencies"].append(t_elapsed)
        # Limiter historique latences à 1000 derniers appels
        if len(_truncation_stats["rerank_latencies"]) > 1000:
            _truncation_stats["rerank_latencies"] = _truncation_stats["rerank_latencies"][-1000:]
    
    # Associer chaque nœud à son score
    scored = list(zip(nodes_with_score, raw_scores))
    # Trier par score décroissant
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Log avec métriques de troncature
    truncation_pct = (truncation_count / len(nodes_with_score)) * 100 if nodes_with_score else 0
    avg_chars_before = total_chars_before / len(nodes_with_score) if nodes_with_score else 0
    avg_chars_after = total_chars_after / len(nodes_with_score) if nodes_with_score else 0
    
    logger.info(
        "Rerank local : %d candidats, %.1fms, top-1=%.3f, top-3=%s | "
        "Troncature: %d/%d chunks (%.1f%%), avg %d→%d chars (char_cap=%d)",
        len(scored),
        t_elapsed,
        scored[0][1] if scored else 0.0,
        [round(s, 3) for _, s in scored[:3]],
        truncation_count,
        len(nodes_with_score),
        truncation_pct,
        int(avg_chars_before),
        int(avg_chars_after),
        char_cap,
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
    
    # Filtrer par score cross-encoder minimum
    min_score = getattr(settings, "RERANKER_MIN_SCORE", -3.0)
    scored_filtered = [item for item in scored if item[1] >= min_score]
    
    if not scored_filtered:
        logger.info(
            "Rerank filtrage: aucun candidat au-dessus du seuil %s (meilleur score: %s)",
            min_score,
            raw_scores[0] if raw_scores else None,
        )
        return RerankResult(
            nodes=[],
            status="ok",
            reason="no_candidates_above_threshold",
            raw_scores=raw_scores,
            softmax_scores=[],
            gap_top1_top2=None,
            zscore_flatness=None,
        )

    # Softmax pour normaliser les scores des candidats filtrés
    filtered_scores = [s for _, s in scored_filtered]
    raw_array = np.array(filtered_scores, dtype=float)
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
        selected_nodes = [nws for nws, _ in scored_filtered[:2]]
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
        k_dynamic = len(scored_filtered)
    
    # Borner entre min_k et max_k
    k_dynamic = max(min_k, min(k_dynamic, max_k))
    
    selected_nodes = [nws for nws, _ in scored_filtered[:k_dynamic]]
    
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
