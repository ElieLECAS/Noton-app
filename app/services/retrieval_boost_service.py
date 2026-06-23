"""Boosts souples retrieval basés sur les signaux query understanding.

- Catégories : appliquées APRÈS la fusion RRF, en multiplicatif sur rrf_score.
  Échelle cohérente quel que soit le canal d'origine (ColPali/pgvector/BM25/KAG),
  et amplifie le signal existant au lieu de fabriquer un rang (cf. anciens boosts
  additifs sur scores bruts d'échelles incompatibles).
- Source, matériau, entités KAG : appliqués post-retrieval, en delta proportionnel
  à l'étendue des scores (évite d'écraser l'ordre du reranker).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from sqlalchemy import text
from sqlmodel import Session, select

from app.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.query_signals_schemas import LightweightQuerySignals

if TYPE_CHECKING:
    from app.services.page_retrieval_service import PageRetrievalHit, UnifiedPageHit

logger = logging.getLogger(__name__)


def _categories_from_metadata(metadata_json: Optional[dict]) -> List[str]:
    if not metadata_json:
        return []
    categories = metadata_json.get("categories", [])
    return categories if isinstance(categories, list) else []


def _get_chunk_categories(session: Session, chunk_id: Optional[int]) -> List[str]:
    if not chunk_id:
        return []
    chunk = session.get(DocumentChunk, chunk_id)
    if not chunk:
        return []
    return _categories_from_metadata(chunk.metadata_json)


def _bulk_get_chunk_categories(session: Session, chunk_ids: Sequence[int]) -> Dict[int, List[str]]:
    unique_ids = list(dict.fromkeys(cid for cid in chunk_ids if cid is not None))
    if not unique_ids:
        return {}
    rows = session.exec(select(DocumentChunk).where(DocumentChunk.id.in_(unique_ids))).all()
    return {int(chunk.id): _categories_from_metadata(chunk.metadata_json) for chunk in rows}


def _category_boost_for_chunk(
    chunk_categories: List[str],
    inferred_categories: List[str],
) -> float:
    matched = set(inferred_categories) & {c.lower() for c in chunk_categories}
    if not matched:
        return 0.0
    return len(matched) * settings.RETRIEVAL_CATEGORY_BOOST


def _bulk_get_page_categories(
    session: Session,
    pages: Sequence[Tuple[int, int]],
) -> Dict[Tuple[int, int], List[str]]:
    """
    Catégories (slugs) par page via la table chunkcategoryrelation (source de vérité).

    Plus robuste que la lecture du metadata d'un chunk représentatif : un hit ColPali
    porte le chunk_id de l'ancre L0 (sans catégories), alors que les catégories vivent
    sur les chunks L1 de la même page.
    """
    wanted = {(int(d), int(p)) for d, p in pages}
    if not wanted:
        return {}

    doc_ids = tuple({d for d, _ in wanted})
    page_nos = tuple({p for _, p in wanted})

    rows = session.execute(
        text(
            """
            SELECT ccr.document_id, ccr.page_no, dc.slug
            FROM chunkcategoryrelation ccr
            INNER JOIN documentcategory dc ON dc.id = ccr.category_id
            WHERE ccr.document_id IN :doc_ids
              AND ccr.page_no IN :page_nos
            """
        ),
        {"doc_ids": doc_ids, "page_nos": page_nos},
    ).all()

    result: Dict[Tuple[int, int], List[str]] = {}
    for document_id, page_no, slug in rows:
        key = (int(document_id), int(page_no))
        if key not in wanted or not slug:
            continue
        bucket = result.setdefault(key, [])
        if slug not in bucket:
            bucket.append(slug)
    return result


def apply_category_boost_to_fused_hits(
    session: Session,
    fused_hits: List[Any],
    signals: Optional[LightweightQuerySignals],
) -> List[Any]:
    """
    Boost catégorie multiplicatif sur rrf_score, APRÈS la fusion RRF (avant rerank).

    Pour chaque page dont une catégorie de contenu matche une catégorie inférée par la
    requête : rrf_score *= (1 + n_match * RETRIEVAL_CATEGORY_BOOST).

    Échelle cohérente (rrf) quel que soit le canal d'origine, borné, et amplifie le
    signal de retrieval existant plutôt que de fabriquer un rang. Mute en place + re-trie.
    Compatible UnifiedPageHit et PageRetrievalHit (duck typing sur rrf_score).
    """
    if not fused_hits or not signals or not signals.inferred_categories:
        return fused_hits

    inferred = {c.strip().lower() for c in signals.inferred_categories if c}
    if not inferred:
        return fused_hits

    page_keys = [(int(h.document_id), int(h.page_no)) for h in fused_hits]
    cats_by_page = _bulk_get_page_categories(session, page_keys)

    boosted = 0
    for hit in fused_hits:
        page_cats = cats_by_page.get((int(hit.document_id), int(hit.page_no)), [])
        matched = inferred & {c.lower() for c in page_cats}
        if not matched:
            continue
        factor = 1.0 + len(matched) * settings.RETRIEVAL_CATEGORY_BOOST
        hit.rrf_score = (hit.rrf_score or 0.0) * factor
        boosted += 1

    if boosted:
        fused_hits.sort(key=lambda h: h.rrf_score or 0.0, reverse=True)
        for rank, hit in enumerate(fused_hits, start=1):
            if hasattr(hit, "final_rank"):
                hit.final_rank = rank
        logger.info(
            "[retrieval_boost] post-fusion catégorie: %d page(s) ×(1+%.2f·n), catégories=%s",
            boosted,
            settings.RETRIEVAL_CATEGORY_BOOST,
            sorted(inferred),
        )
    return fused_hits


def _get_document_source(session: Session, document_id: int) -> Optional[str]:
    doc = session.get(Document, document_id)
    return doc.source if doc else None


def _get_document_materials(session: Session, document_id: int) -> List[str]:
    doc = session.get(Document, document_id)
    return list(doc.materials or []) if doc else []


def _chunk_id_from_passage(passage: Dict[str, Any]) -> Optional[int]:
    chunk_id = passage.get("chunk_id")
    if chunk_id is not None:
        try:
            return int(chunk_id)
        except (TypeError, ValueError):
            pass
    return None


def _document_id_from_passage(passage: Dict[str, Any]) -> Optional[int]:
    doc_id = passage.get("document_id")
    if doc_id is not None:
        try:
            return int(doc_id)
        except (TypeError, ValueError):
            pass
    return None


def _passage_score_span(passages: List[Dict[str, Any]]) -> float:
    """
    Étendue des scores des passages, pour dimensionner les boosts dans la bonne échelle.

    Le `score` post-retrieval peut être un logit de reranker (parfois négatif) ou un
    rrf_score (~0,01-0,05) : un boost additif fixe écraserait l'un et serait négligeable
    pour l'autre. On exprime donc le boost en fraction de l'étendue observée.
    """
    scores = [float(p.get("score", 0.0)) for p in passages]
    if not scores:
        return 1.0
    span = max(scores) - min(scores)
    if span > 0:
        return span
    # Pool homogène (1 passage ou scores égaux) : repli sur la magnitude moyenne.
    magnitude = sum(abs(s) for s in scores) / len(scores)
    return magnitude if magnitude > 0 else 1.0


def apply_soft_boosts_to_passages(
    session: Session,
    passages: List[Dict[str, Any]],
    signals: LightweightQuerySignals,
) -> List[Dict[str, Any]]:
    """
    Applique des boosts souples (source / matériau / entités KAG) sur les passages finaux.

    Le boost est exprimé comme une FRACTION de l'étendue des scores du pool puis ajouté,
    afin d'amplifier les préférences sans écraser l'ordre du reranker (cf. ancien additif
    fixe sur une échelle de score ambiguë). Les catégories ne sont PLUS boostées ici :
    elles le sont en amont, sur rrf_score (apply_category_boost_to_fused_hits).
    """
    if not passages or not signals:
        return passages

    source_boost_max = settings.RETRIEVAL_SOURCE_BOOST_MAX
    material_boost = settings.RETRIEVAL_MATERIAL_BOOST
    entity_boost = settings.RETRIEVAL_ENTITY_BOOST
    score_span = _passage_score_span(passages)

    refined: List[Dict[str, Any]] = []
    for passage in passages:
        p_copy = dict(passage)
        score = float(p_copy.get("score", 0.0))
        boost_frac = 0.0

        doc_id = _document_id_from_passage(p_copy)

        if doc_id:
            doc_source = _get_document_source(session, doc_id)
            if doc_source:
                p_copy["source"] = doc_source

        if signals.primary_source and doc_id:
            doc_source = _get_document_source(session, doc_id)
            if doc_source and doc_source.lower() == signals.primary_source.lower():
                boost_frac += source_boost_max * signals.confidence

        if signals.material_hint and doc_id:
            doc_materials = _get_document_materials(session, doc_id)
            if signals.material_hint.lower() in [m.lower() for m in doc_materials]:
                boost_frac += material_boost

        retrieval_sources = p_copy.get("retrieval_sources") or []
        if "kag" in retrieval_sources and signals.entity_texts:
            boost_frac += entity_boost * min(len(signals.entity_texts), 3)

        if boost_frac > 0:
            delta = boost_frac * score_span
            p_copy["score"] = score + delta
            p_copy["retrieval_boost"] = round(delta, 4)

        refined.append(p_copy)

    refined.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    if any(p.get("retrieval_boost") for p in refined):
        logger.info(
            "[retrieval_boost] %d passage(s) boosté(s) (span=%.4f), top score=%.4f",
            sum(1 for p in refined if p.get("retrieval_boost")),
            score_span,
            float(refined[0].get("score", 0)) if refined else 0,
        )
    return refined
