"""
Retrieval KAG — 4e canal page-centric via entity linking et traversal graphe.

Pipeline :
  1. Matching entités query (trigram + embedding sémantique + alias)
  2. Traversal chunkentityrelation → pages seed (hop 0)
  3. Traversal entityentityrelation → entités voisines → pages (hop 1)
  4. Scoring agrégé → UnifiedPageHit compatibles RRF
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import text
from sqlmodel import Session

from app.config import settings
from app.services.kag_extraction_service import normalize_entity_name
from app.services.page_retrieval_service import UnifiedPageHit, _page_no_sql_expr

logger = logging.getLogger(__name__)


def _query_entity_candidates(
    session: Session,
    space_id: int,
    query_text: str,
    query_embedding: Optional[List[float]],
    limit: int = 15,
) -> List[Tuple[int, float, str]]:
    """
    Retourne (entity_id, match_score, match_source) triés par score décroissant.
    """
    normalized_query = normalize_entity_name(query_text)
    tokens = [t for t in re.split(r"\W+", normalized_query) if len(t) >= 3][:8]
    candidates: Dict[int, Tuple[float, str]] = {}

    if tokens:
        for token in tokens:
            rows = session.execute(
                text(
                    """
                    SELECT ke.id,
                           GREATEST(
                               similarity(ke.name_normalized, :token),
                               COALESCE(
                                   (SELECT MAX(similarity(ea.alias_normalized, :token))
                                    FROM entityalias ea
                                    WHERE ea.entity_id = ke.id AND ea.space_id = :space_id),
                                   0
                               )
                           ) AS sim
                    FROM knowledgeentity ke
                    WHERE ke.space_id = :space_id
                      AND (
                          ke.name_normalized % :token
                          OR EXISTS (
                              SELECT 1 FROM entityalias ea
                              WHERE ea.entity_id = ke.id
                                AND ea.space_id = :space_id
                                AND ea.alias_normalized % :token
                          )
                      )
                    ORDER BY sim DESC
                    LIMIT :lim
                    """
                ),
                {"space_id": space_id, "token": token, "lim": limit},
            ).all()
            for entity_id, sim in rows:
                score = float(sim or 0.0)
                prev = candidates.get(entity_id)
                if prev is None or score > prev[0]:
                    candidates[entity_id] = (score, "trigram")

    if query_embedding:
        embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
        rows = session.execute(
            text(
                """
                SELECT id, 1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
                FROM knowledgeentity
                WHERE space_id = :space_id
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:query_vec AS vector)
                LIMIT :lim
                """
            ),
            {"space_id": space_id, "query_vec": embedding_str, "lim": limit},
        ).all()
        for entity_id, sim in rows:
            score = float(sim or 0.0)
            prev = candidates.get(entity_id)
            if prev is None or score > prev[0]:
                candidates[entity_id] = (score, "embedding")
            elif score > prev[0] * 0.9:
                candidates[entity_id] = (max(score, prev[0]), "hybrid")

    ranked = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
    min_score = settings.KAG_ENTITY_MATCH_MIN_SCORE
    return [(eid, score, src) for eid, (score, src) in ranked if score >= min_score][:limit]


def _pages_for_entities(
    session: Session,
    doc_ids: List[int],
    entity_ids: Set[int],
) -> Dict[str, Tuple[float, int, str]]:
    """
    Retourne page_key -> (score, chunk_id, document_title) pour entités données.
    """
    if not entity_ids or not doc_ids:
        return {}

    page_no_expr = _page_no_sql_expr("dc")
    rows = session.execute(
        text(
            f"""
            SELECT
                dc.document_id,
                {page_no_expr} AS page_no,
                d.title AS document_title,
                dc.id AS chunk_id,
                MAX(cer.relevance_score * COALESCE(ke.confidence_score, 1.0)) AS score
            FROM chunkentityrelation cer
            INNER JOIN documentchunk dc ON dc.id = cer.chunk_id
            INNER JOIN document d ON d.id = dc.document_id
            INNER JOIN knowledgeentity ke ON ke.id = cer.entity_id
            WHERE cer.entity_id IN :entity_ids
              AND dc.document_id IN :doc_ids
              AND dc.is_leaf = true
              AND {page_no_expr} IS NOT NULL
            GROUP BY dc.document_id, {page_no_expr}, d.title, dc.id
            """
        ),
        {"entity_ids": tuple(entity_ids), "doc_ids": tuple(doc_ids)},
    ).all()

    page_scores: Dict[str, Tuple[float, int, str]] = {}
    for doc_id, page_no, title, chunk_id, score in rows:
        if page_no is None:
            continue
        key = f"{doc_id}:{int(page_no)}"
        weighted = float(score or 0.0)
        prev = page_scores.get(key)
        if prev is None or weighted > prev[0]:
            page_scores[key] = (weighted, int(chunk_id), title or "Document sans titre")

    return page_scores


def _neighbor_entities(
    session: Session,
    space_id: int,
    seed_entity_ids: Set[int],
    hop_limit: int,
) -> Set[int]:
    if hop_limit <= 0 or not seed_entity_ids:
        return set()

    rows = session.execute(
        text(
            """
            SELECT entity_a_id, entity_b_id
            FROM entityentityrelation
            WHERE space_id = :space_id
              AND (entity_a_id IN :entity_ids OR entity_b_id IN :entity_ids)
            ORDER BY weight DESC
            LIMIT 100
            """
        ),
        {"space_id": space_id, "entity_ids": tuple(seed_entity_ids)},
    ).all()

    neighbors: Set[int] = set()
    for a_id, b_id in rows:
        if a_id in seed_entity_ids and b_id not in seed_entity_ids:
            neighbors.add(b_id)
        if b_id in seed_entity_ids and a_id not in seed_entity_ids:
            neighbors.add(a_id)
    return neighbors


def retrieve_kag_pages(
    session: Session,
    space_id: int,
    doc_ids: List[int],
    query_text: str,
    query_embedding: Optional[List[float]],
    limit: int,
) -> List[UnifiedPageHit]:
    """
    Retrieval page-centric via graphe KAG (entity linking + 1-hop expansion).
    """
    if not settings.KAG_ENABLED or not doc_ids or not query_text.strip():
        return []

    try:
        matched = _query_entity_candidates(session, space_id, query_text, query_embedding)
    except Exception as exc:
        logger.warning("[KAG retrieval] Matching entités échoué : %s", exc)
        return []

    if not matched:
        return []

    seed_ids = {eid for eid, _, _ in matched}
    hop_limit = settings.KAG_RETRIEVAL_HOP_LIMIT

    page_index: Dict[str, UnifiedPageHit] = {}

    def _merge_pages(pages: Dict[str, Tuple[float, int, str]], hop: int) -> None:
        hop_factor = 1.0 if hop == 0 else 0.65
        for key, (score, chunk_id, title) in pages.items():
            doc_id_str, page_str = key.split(":", 1)
            doc_id = int(doc_id_str)
            page_no = int(page_str)
            weighted = score * hop_factor
            if key not in page_index:
                page_index[key] = UnifiedPageHit(
                    document_id=doc_id,
                    page_no=page_no,
                    kag_score=weighted,
                    chunk_id=chunk_id,
                    document_title=title,
                    retrieval_sources=["kag"],
                )
            else:
                hit = page_index[key]
                hit.kag_score = max(hit.kag_score or 0.0, weighted)
                if "kag" not in hit.retrieval_sources:
                    hit.retrieval_sources.append("kag")
                if chunk_id and hit.chunk_id is None:
                    hit.chunk_id = chunk_id
                if title:
                    hit.document_title = title

    try:
        seed_pages = _pages_for_entities(session, doc_ids, seed_ids)
        _merge_pages(seed_pages, hop=0)

        if hop_limit >= 1:
            neighbor_ids = _neighbor_entities(session, space_id, seed_ids, hop_limit)
            if neighbor_ids:
                neighbor_pages = _pages_for_entities(session, doc_ids, neighbor_ids)
                _merge_pages(neighbor_pages, hop=1)
    except Exception as exc:
        logger.warning("[KAG retrieval] Traversal graphe échoué : %s", exc)
        return []

    hits = sorted(page_index.values(), key=lambda h: h.kag_score or 0.0, reverse=True)
    return hits[:limit]
