"""Boosts souples retrieval basés sur les signaux query understanding.

- Ancre conversationnelle : multiplicatif sur rrf_score, APRÈS la fusion RRF et avant
  la coupe top_k, sur les meilleures pages des documents du sujet courant.
- Source, matériau : appliqués post-retrieval, en delta proportionnel à l'étendue des
  scores (évite d'écraser l'ordre du reranker).

Le boost CATÉGORIE a été retiré le 2026-07-28 avec le KAG : son entrée mêlait des tags
posés en masse à confiance 1.0 (doc_type/lifecycle_phase sur tous les chunks d'une page)
et de vraies classifications au seuil 0.55, pour un levier allant jusqu'à x1.5 sur le
classement — reranker éteint, c'était devenu le principal signal de reclassement.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlmodel import Session

from app.config import settings
from app.models.document import Document
from app.services.query_signals_schemas import LightweightQuerySignals

logger = logging.getLogger(__name__)


def apply_anchor_boost_to_fused_hits(
    fused_hits: List[Any],
    anchor_document_ids: Optional[Sequence[int]],
) -> Optional[Dict[str, Any]]:
    """Boost multiplicatif des MEILLEURES pages des documents ANCRÉS (continuité de conversation).

    Appliqué après la fusion RRF (et le boost catégorie), AVANT la coupe top_k : une page
    d'un document du sujet courant remonte et survit à la coupe, même quand le tour est
    formulé en suivi elliptique (« et les autres ? ») qui matche faiblement en propre.
    Empêche la conversation de sauter d'un produit à l'autre. Duck typing sur rrf_score.

    DEUX GARDE-FOUS (2026-07-27), après un cas où un tour « comment l'installer ? » packait
    le catalogue *conception* du tour précédent au lieu du catalogue *fabrication* qui
    contenait la procédure :

    * **Facteur réduit** — à ×1.5, l'ancre accordait gratuitement à chacune de ses pages le
      boost catégorie MAXIMAL (plafonné à 1.5, et qui exige lui des correspondances fortes) :
      une page sans le moindre rapport avec la question recevait autant qu'une page
      parfaitement catégorisée. L'ancre annulait le signal de pertinence au lieu de l'aider.
    * **Pages plafonnées** — seules les N meilleures pages de chaque document ancré sont
      boostées. Le but est qu'une page pertinente survive à la coupe, pas qu'un document
      entier s'installe en tête.

    Mute ``fused_hits`` en place. Retourne le détail du boost appliqué (pour le cheminement)
    ou ``None`` si aucune page n'a été boostée.
    """
    if not fused_hits or not anchor_document_ids:
        return None

    anchor = {int(d) for d in anchor_document_ids}
    factor = 1.0 + settings.CONVERSATION_ANCHOR_BOOST
    max_pages = max(0, settings.CONVERSATION_ANCHOR_BOOST_MAX_PAGES)

    # Les mieux classées d'abord, pour que le plafond retienne les pages les plus fortes.
    candidates = sorted(
        (h for h in fused_hits if int(getattr(h, "document_id", -1)) in anchor),
        key=lambda h: h.rrf_score or 0.0,
        reverse=True,
    )
    per_doc: Dict[int, int] = {}
    boosted_pages: List[Tuple[int, int]] = []
    for hit in candidates:
        doc_id = int(hit.document_id)
        if max_pages and per_doc.get(doc_id, 0) >= max_pages:
            continue
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        hit.rrf_score = (hit.rrf_score or 0.0) * factor
        boosted_pages.append((doc_id, int(getattr(hit, "page_no", 0) or 0)))

    boosted = len(boosted_pages)
    if not boosted:
        return None

    fused_hits.sort(key=lambda h: h.rrf_score or 0.0, reverse=True)
    for rank, hit in enumerate(fused_hits, start=1):
        if hasattr(hit, "final_rank"):
            hit.final_rank = rank
    logger.info(
        "[retrieval_boost] ancre conversation: %d page(s) ×%.2f (max %s/doc), docs=%s",
        boosted,
        factor,
        max_pages or "∞",
        sorted(anchor),
    )
    # Renvoyé au cheminement (A5) : un tour biaisé par l'ancre doit être identifiable.
    return {
        "factor": round(factor, 3),
        "max_pages_per_document": max_pages,
        "documents": sorted(anchor),
        "boosted_pages": [{"document_id": d, "page_no": p} for d, p in boosted_pages],
    }


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
    Applique des boosts souples (source / matériau) sur les passages finaux.

    Le boost est exprimé comme une FRACTION de l'étendue des scores du pool puis ajouté,
    afin d'amplifier les préférences sans écraser l'ordre du reranker (cf. ancien additif
    fixe sur une échelle de score ambiguë).

    Boosts restants : source (fournisseur) et matériau. Le boost catégorie a été retiré
    avec le KAG (2026-07-28) : son entrée était en partie des tags posés en masse à
    confiance 1.0, pour un levier allant jusqu'à x1.5 sur le classement.
    """
    if not passages or not signals:
        return passages

    source_boost_max = settings.RETRIEVAL_SOURCE_BOOST_MAX
    material_boost = settings.RETRIEVAL_MATERIAL_BOOST
    score_span = _passage_score_span(passages)

    refined: List[Dict[str, Any]] = []
    for passage in passages:
        p_copy = dict(passage)
        score = float(p_copy.get("score", 0.0))
        boost_frac = 0.0

        doc_id = _document_id_from_passage(p_copy)

        # Une seule lecture de la source par passage (évite le double session.get).
        doc_source = _get_document_source(session, doc_id) if doc_id else None
        if doc_source:
            p_copy["source"] = doc_source

        if (
            signals.primary_source
            and doc_source
            and doc_source.lower() == signals.primary_source.lower()
        ):
            boost_frac += source_boost_max * signals.confidence

        if signals.material_hint and doc_id:
            doc_materials = _get_document_materials(session, doc_id)
            if signals.material_hint.lower() in [m.lower() for m in doc_materials]:
                boost_frac += material_boost

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
