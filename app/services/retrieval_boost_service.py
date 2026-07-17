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
from sqlmodel import Session

from app.config import settings
from app.models.document import Document
from app.services.query_signals_schemas import LightweightQuerySignals

if TYPE_CHECKING:
    from app.services.page_retrieval_service import PageRetrievalHit, UnifiedPageHit

logger = logging.getLogger(__name__)


def _bulk_get_page_categories(
    session: Session,
    pages: Sequence[Tuple[int, int]],
) -> Dict[Tuple[int, int], Dict[str, Tuple[str, float]]]:
    """
    Catégories par page (slug → (axis, confidence)) via chunkcategoryrelation.

    Plus robuste que la lecture du metadata d'un chunk représentatif : un hit ColPali
    porte le chunk_id de l'ancre L0 (sans catégories), alors que les catégories vivent
    sur les chunks L1 de la même page. L'axe pondère le boost par facette ; la confiance
    (MAX sur la page) évite qu'un tag faible/secondaire fasse remonter une page à tort.
    """
    wanted = {(int(d), int(p)) for d, p in pages}
    if not wanted:
        return {}

    doc_ids = tuple({d for d, _ in wanted})
    page_nos = tuple({p for _, p in wanted})

    rows = session.execute(
        text(
            """
            SELECT ccr.document_id, ccr.page_no, dc.slug, dc.axis,
                   MAX(ccr.confidence) AS confidence
            FROM chunkcategoryrelation ccr
            INNER JOIN documentcategory dc ON dc.id = ccr.category_id
            WHERE ccr.document_id IN :doc_ids
              AND ccr.page_no IN :page_nos
            GROUP BY ccr.document_id, ccr.page_no, dc.slug, dc.axis
            """
        ),
        {"doc_ids": doc_ids, "page_nos": page_nos},
    ).all()

    result: Dict[Tuple[int, int], Dict[str, Tuple[str, float]]] = {}
    for document_id, page_no, slug, axis, confidence in rows:
        key = (int(document_id), int(page_no))
        if key not in wanted or not slug:
            continue
        result.setdefault(key, {})[slug] = (axis or "task", float(confidence if confidence is not None else 1.0))
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
    axis_weights: Dict[str, float] = settings.RETRIEVAL_AXIS_BOOST_WEIGHTS or {}

    boosted = 0
    for hit in fused_hits:
        slug_meta = cats_by_page.get((int(hit.document_id), int(hit.page_no)), {})
        matched = inferred & {s.lower() for s in slug_meta}
        if not matched:
            continue
        # Contribution pondérée par axe (symptôme > task > doc_type) ET par la confiance
        # de la catégorie sur la page, plafonnée aux 3 plus fortes. Une page faiblement
        # taggée (confiance basse) ne remonte donc plus à tort.
        contributions = sorted(
            (
                float(axis_weights.get(slug_meta.get(slug, ("task", 1.0))[0], 1.0))
                * float(slug_meta.get(slug, ("task", 1.0))[1])
                for slug in matched
            ),
            reverse=True,
        )[:3]
        # Plafonné : empêche qu'un cumul de matches (ex. 3 catégories symptôme) fabrique
        # un facteur ~1.9 capable d'inverser un vrai signal de pertinence — d'autant plus
        # critique que le reranker cross-encoder est désactivé (boost = principal signal).
        factor = min(
            1.0 + settings.RETRIEVAL_CATEGORY_BOOST * sum(contributions),
            settings.RETRIEVAL_CATEGORY_BOOST_MAX,
        )
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


def apply_anchor_boost_to_fused_hits(
    fused_hits: List[Any],
    anchor_document_ids: Optional[Sequence[int]],
) -> List[Any]:
    """Boost multiplicatif des pages des documents ANCRÉS (continuité de conversation).

    Appliqué après la fusion RRF (et le boost catégorie), AVANT la coupe top_k : une page
    d'un document du sujet courant remonte et survit à la coupe, même quand le tour est
    formulé en suivi elliptique (« et les autres ? ») qui matche faiblement en propre.
    Empêche la conversation de sauter d'un produit à l'autre. Duck typing sur rrf_score.
    """
    if not fused_hits or not anchor_document_ids:
        return fused_hits

    anchor = {int(d) for d in anchor_document_ids}
    factor = 1.0 + settings.CONVERSATION_ANCHOR_BOOST
    boosted = 0
    for hit in fused_hits:
        if int(getattr(hit, "document_id", -1)) in anchor:
            hit.rrf_score = (hit.rrf_score or 0.0) * factor
            boosted += 1

    if boosted:
        fused_hits.sort(key=lambda h: h.rrf_score or 0.0, reverse=True)
        for rank, hit in enumerate(fused_hits, start=1):
            if hasattr(hit, "final_rank"):
                hit.final_rank = rank
        logger.info(
            "[retrieval_boost] ancre conversation: %d page(s) ×%.2f, docs=%s",
            boosted,
            factor,
            sorted(anchor),
        )
    return fused_hits


def compute_anchor_documents(passages: List[Dict[str, Any]], *, max_docs: int) -> List[int]:
    """Documents dominants d'un lot de passages (somme des scores par document, top-N).

    Sert à mémoriser le « sujet documentaire » d'un tour pour ancrer les tours suivants.
    """
    if not passages or max_docs <= 0:
        return []
    score_by_doc: Dict[int, float] = {}
    for p in passages:
        did = p.get("document_id")
        if did is None:
            continue
        score_by_doc[int(did)] = score_by_doc.get(int(did), 0.0) + float(p.get("score") or 0.0)
    ranked = sorted(score_by_doc.items(), key=lambda kv: kv[1], reverse=True)
    return [did for did, _ in ranked[:max_docs]]


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
