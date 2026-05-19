"""
Utilitaires d'évaluation retrieval (Precision@K, MRR) sans dépendance DB.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union


def precision_at_k(
    retrieved: List[Dict[str, Any]],
    relevant_ids: Set[Union[int, str]],
    k: int,
    id_key: str = "chunk_id",
) -> float:
    """Fraction des k premiers résultats qui sont pertinents."""
    if k <= 0 or not retrieved:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for r in top if r.get(id_key) in relevant_ids)
    return hits / min(k, len(top))


def mean_reciprocal_rank(
    retrieved: List[Dict[str, Any]],
    relevant_ids: Set[Union[int, str]],
    id_key: str = "chunk_id",
) -> float:
    """RR du premier document pertinent (0 si aucun)."""
    for rank, item in enumerate(retrieved, start=1):
        if item.get(id_key) in relevant_ids:
            return 1.0 / rank
    return 0.0


def noise_ratio(
    retrieved: List[Dict[str, Any]],
    irrelevant_keywords: Iterable[str],
    text_key: str = "passage_raw",
) -> float:
    """Part des passages contenant au moins un mot-clé hors-sujet (heuristique)."""
    if not retrieved:
        return 0.0
    keywords = [k.lower() for k in irrelevant_keywords if k]
    if not keywords:
        return 0.0
    noisy = 0
    for item in retrieved:
        text = (item.get(text_key) or item.get("passage") or "").lower()
        if any(kw in text for kw in keywords):
            noisy += 1
    return noisy / len(retrieved)


def load_eval_dataset(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("cases") or [])


def score_retrieval_case(
    retrieved: List[Dict[str, Any]],
    case: Dict[str, Any],
    k: int = 6,
) -> Dict[str, float]:
    """
    Score un cas du jeu d'éval : relevant_ids déduits des substrings dans le texte
    si chunk_id annotés absents.
    """
    substrings = [s.lower() for s in case.get("relevant_chunk_id_substrings") or []]
    relevant_ids: Set[Union[int, str]] = set()
    for item in retrieved:
        cid = item.get("chunk_id")
        text = (item.get("passage_raw") or item.get("passage") or "").lower()
        if substrings and all(s in text for s in substrings[:1]):
            if cid is not None:
                relevant_ids.add(cid)
        elif substrings and any(s in text for s in substrings):
            if cid is not None:
                relevant_ids.add(cid)

    explicit = case.get("relevant_chunk_ids")
    if explicit:
        relevant_ids = set(explicit)

    return {
        "precision_at_k": precision_at_k(retrieved, relevant_ids, k=k),
        "mrr": mean_reciprocal_rank(retrieved, relevant_ids),
        "noise_ratio": noise_ratio(
            retrieved,
            case.get("irrelevant_keywords") or [],
        ),
    }
