"""
MMR (Maximal Marginal Relevance) pour la diversification des candidats leaves.

Utilisé par les deux pipelines chat (bibliothèque / espace) avant la résolution
des parents. Pas de ré-embedding : les embeddings sont lus en base (pgvector)
par les services appelants puis fournis à ``mmr_select``.
"""

from typing import Callable, Dict, List, Optional, Sequence

from llama_index.core.schema import NodeWithScore


MMR_ENABLED = True
MMR_LAMBDA = 0.7
MMR_POOL = 50


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosinus pur Python entre deux vecteurs de même dimension."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def pair_similarity(
    a: NodeWithScore,
    b: NodeWithScore,
    emb_by_cid: Dict[int, List[float]],
    parse_chunk_id: Callable[[object], Optional[int]],
) -> float:
    """
    Similarité entre deux candidats.

    1. Cosinus entre embeddings stockés si disponibles pour les deux côtés.
    2. Sinon proxy métadonnées (même parent > même doc/note > autre).
    """
    cid_a = parse_chunk_id(a.node)
    cid_b = parse_chunk_id(b.node)
    va = emb_by_cid.get(cid_a) if cid_a is not None else None
    vb = emb_by_cid.get(cid_b) if cid_b is not None else None
    if va is not None and vb is not None:
        return max(0.0, _cosine(va, vb))

    ma = dict(getattr(a.node, "metadata", {}) or {})
    mb = dict(getattr(b.node, "metadata", {}) or {})
    pa = ma.get("parent_node_id")
    pb = mb.get("parent_node_id")
    if pa and pb and pa == pb:
        return 0.9
    da = ma.get("document_id")
    db = mb.get("document_id")
    if da is not None and db is not None and da == db:
        return 0.4
    na_id = ma.get("note_id")
    nb_id = mb.get("note_id")
    if na_id is not None and nb_id is not None and na_id == nb_id:
        return 0.4
    return 0.0


def mmr_select(
    pool: List[NodeWithScore],
    emb_by_cid: Dict[int, List[float]],
    k: int,
    parse_chunk_id: Callable[[object], Optional[int]],
) -> List[NodeWithScore]:
    """
    Sélection gloutonne MMR :
        score(c) = MMR_LAMBDA * rel(c) - (1 - MMR_LAMBDA) * max_sim_to_selected(c)

    ``rel`` est la pertinence brute du candidat normalisée min-max sur le pool.
    """
    if not pool or k <= 0:
        return list(pool[:k])
    if len(pool) <= k:
        return list(pool)

    scores = [float(nws.score or 0.0) for nws in pool]
    smin = min(scores)
    smax = max(scores)
    spread = smax - smin

    def rel(i: int) -> float:
        if spread <= 0.0:
            return 1.0
        return (scores[i] - smin) / spread

    selected_idx: List[int] = []
    remaining = list(range(len(pool)))

    first = max(remaining, key=rel)
    selected_idx.append(first)
    remaining.remove(first)

    while remaining and len(selected_idx) < k:
        best_i: Optional[int] = None
        best_val = -1e18
        for i in remaining:
            sim_max = 0.0
            for j in selected_idx:
                sim = pair_similarity(pool[i], pool[j], emb_by_cid, parse_chunk_id)
                if sim > sim_max:
                    sim_max = sim
            val = MMR_LAMBDA * rel(i) - (1.0 - MMR_LAMBDA) * sim_max
            if val > best_val:
                best_val = val
                best_i = i
        if best_i is None:
            break
        selected_idx.append(best_i)
        remaining.remove(best_i)

    return [pool[i] for i in selected_idx]
