"""Santé d'indexation par document — texte, embeddings, ColPali (sync LanceDB), chunks contextuels.

Répond à la question « ce document a-t-il TOUT (texte, ColPali, entités, catégories),
et faut-il le retraiter — en quel mode ? ». Croise l'état Postgres (chunks, embeddings,
entités, catégories) avec l'état LanceDB (patches ColPali) pour détecter les
désynchronisations que seuls les logs de retrieval révélaient jusqu'ici
(patches orphelins → pages invisibles du canal visuel).

Statuts ColPali :
  ok         — chaque page_anchor a ses patches, aucun patch orphelin
  partial    — des pages sans patches (index visuel incomplet)
  desync     — patches pointant vers des chunks supprimés ou non-anchor (re-indexer)
  missing    — aucun patch alors que le document en attend
  unknown    — scan LanceDB en échec
  not_applicable / disabled — pas concerné (non-PDF sans anchors / ColPali off)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set

from sqlalchemy import text
from sqlmodel import Session

from app.config import settings
from app.models.document import Document

logger = logging.getLogger(__name__)

# Littéral partagé avec document_indexing_service / page_retrieval_service
# (redéfini localement comme dans page_retrieval_service pour éviter un import lourd).
CONTENT_TYPE_PAGE_ANCHOR = "page_anchor"

# Statuts de traitement pendant lesquels l'audit serait trompeur (chunks en cours d'écriture).
_IN_PROGRESS_STATUSES = {"pending", "processing", "reindex_queued"}


def _rows_by_document(rows: Sequence[Any]) -> Dict[int, list]:
    grouped: Dict[int, list] = {}
    for row in rows:
        grouped.setdefault(int(row[0]), []).append(row)
    return grouped


def _fetch_chunk_summaries(session: Session, doc_ids: List[int]) -> Dict[int, dict]:
    rows = session.execute(
        text(
            """
            SELECT document_id,
                   COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE is_leaf) AS leaves,
                   COUNT(*) FILTER (WHERE is_leaf AND embedding IS NOT NULL) AS leaves_with_embedding
            FROM documentchunk
            WHERE document_id = ANY(:ids)
            GROUP BY document_id
            """
        ),
        {"ids": doc_ids},
    ).all()
    return {
        int(r.document_id): {
            "chunk_count": int(r.total or 0),
            "leaf_count": int(r.leaves or 0),
            "leaves_with_embedding": int(r.leaves_with_embedding or 0),
        }
        for r in rows
    }


def _fetch_chunk_id_sets(
    session: Session, doc_ids: List[int]
) -> tuple[Dict[int, Set[int]], Dict[int, Set[int]], Dict[int, Set[int]]]:
    """(all_ids, anchor_ids, leaf_ids) par document — pour l'audit de sync ColPali."""
    rows = session.execute(
        text(
            """
            SELECT document_id, id, is_leaf,
                   (metadata_json->>'content_type') = :anchor AS is_anchor
            FROM documentchunk
            WHERE document_id = ANY(:ids)
            """
        ),
        {"ids": doc_ids, "anchor": CONTENT_TYPE_PAGE_ANCHOR},
    ).all()
    all_ids: Dict[int, Set[int]] = {d: set() for d in doc_ids}
    anchor_ids: Dict[int, Set[int]] = {d: set() for d in doc_ids}
    leaf_ids: Dict[int, Set[int]] = {d: set() for d in doc_ids}
    for r in rows:
        did = int(r.document_id)
        cid = int(r.id)
        all_ids[did].add(cid)
        if r.is_anchor:
            anchor_ids[did].add(cid)
        if r.is_leaf:
            leaf_ids[did].add(cid)
    return all_ids, anchor_ids, leaf_ids


def _fetch_enrichment_counts(session: Session, doc_ids: List[int]) -> Dict[int, dict]:
    """Nombre de chunks contextuels (L2) par document.

    Remplace l'ancien volet KAG (entités/relations/catégories) retiré le 2026-07-28 :
    la question utile est désormais « ce document a-t-il ses synthèses ? », puisque
    text_only ne les produit plus et qu'un passage enrichment_only est requis.
    """
    counts: Dict[int, dict] = {d: {"enrichment_count": 0} for d in doc_ids}
    if not doc_ids:
        return counts

    rows = session.execute(
        text(
            """
            SELECT document_id, COUNT(*)
            FROM documentchunk
            WHERE document_id = ANY(:ids)
              AND COALESCE(
                  metadata_json->>'content_type',
                  metadata_->>'content_type',
                  ''
              ) = 'contextual_enrichment'
            GROUP BY document_id
            """
        ),
        {"ids": doc_ids},
    ).all()
    for did, n in rows:
        counts[int(did)]["enrichment_count"] = int(n or 0)
    return counts


def _text_health(summary: dict) -> dict:
    chunk_count = summary["chunk_count"]
    leaf_count = summary["leaf_count"]
    leaves_with_embedding = summary["leaves_with_embedding"]
    if chunk_count == 0:
        status = "missing"
    elif leaves_with_embedding == 0:
        status = "missing"
    elif leaves_with_embedding < leaf_count:
        status = "partial"
    else:
        status = "ok"
    return {
        "status": status,
        "chunk_count": chunk_count,
        "leaf_count": leaf_count,
        "leaves_with_embedding": leaves_with_embedding,
        "missing_embeddings": max(0, leaf_count - leaves_with_embedding),
    }


def _colpali_health(
    *,
    is_pdf: bool,
    lancedb_ids: Optional[Set[int]],
    all_ids: Set[int],
    anchor_ids: Set[int],
    leaf_ids: Set[int],
) -> dict:
    base = {
        "expected_pages": 0,
        "indexed_pages": 0,
        "orphan_count": 0,
        "missing_count": 0,
        "legacy_count": 0,
    }
    if not settings.COLPALI_ENABLED:
        return {"status": "disabled", **base}
    if lancedb_ids is None:
        return {"status": "unknown", **base}

    # Pipeline actuel : patches liés aux page_anchor. Ancien pipeline : liés aux feuilles.
    expected = anchor_ids if anchor_ids else leaf_ids
    legacy_mode = not anchor_ids

    if not expected and not lancedb_ids:
        return {"status": "not_applicable" if not is_pdf else "missing", **base}

    orphans = lancedb_ids - all_ids
    synced = lancedb_ids & expected
    stale = (lancedb_ids & all_ids) - expected  # patches vers des chunks existants mais hors cible
    missing = expected - lancedb_ids

    detail = {
        "expected_pages": len(expected),
        "indexed_pages": len(synced),
        "orphan_count": len(orphans),
        "missing_count": len(missing),
        "legacy_count": len(stale),
        "legacy_mode": legacy_mode,
    }
    if orphans or (stale and not legacy_mode):
        return {"status": "desync", **detail}
    if not lancedb_ids:
        return {"status": "missing", **detail}
    if missing:
        return {"status": "partial", **detail}
    return {"status": "ok", **detail}


def _enrichment_health(enrichment_counts: dict, chunk_count: int) -> dict:
    if not settings.CONTEXTUAL_ENRICHMENT_ENABLED:
        return {"status": "disabled", **enrichment_counts}
    if chunk_count == 0:
        return {"status": "missing", **enrichment_counts}
    status = "ok" if enrichment_counts["enrichment_count"] > 0 else "missing"
    return {"status": status, **enrichment_counts}


def _overall_and_mode(
    text_h: dict, colpali_h: dict, enrichment_h: dict
) -> tuple[str, Optional[str]]:
    """Verdict global + mode de retraitement suggéré (aligné sur ReindexRequest.mode)."""
    text_broken = text_h["status"] == "missing"
    colpali_broken = colpali_h["status"] in ("desync", "missing", "partial")
    enrichment_broken = enrichment_h["status"] == "missing"

    if text_broken:
        return "error", "full"
    if colpali_broken and enrichment_broken:
        return "warning", "full"
    if colpali_broken:
        return "warning", "colpali_only"
    if enrichment_broken:
        return "warning", "enrichment_only"
    if text_h["status"] == "partial":
        return "warning", "text_only"
    if colpali_h["status"] == "unknown":
        return "warning", None
    return "ok", None


def build_indexing_health_bulk(
    session: Session, documents: List[Document]
) -> Dict[int, dict]:
    """Santé d'indexation pour un lot de documents (requêtes groupées + 1 scan LanceDB)."""
    docs = [d for d in documents if d.id is not None and d.document_type == "document"]
    if not docs:
        return {}
    doc_ids = [int(d.id) for d in docs]

    summaries = _fetch_chunk_summaries(session, doc_ids)
    all_ids, anchor_ids, leaf_ids = _fetch_chunk_id_sets(session, doc_ids)
    enrichment_counts = _fetch_enrichment_counts(session, doc_ids)

    if settings.COLPALI_ENABLED:
        from app.services.lancedb_service import get_colpali_chunk_ids_by_document

        lancedb_by_doc = get_colpali_chunk_ids_by_document(doc_ids)
    else:
        lancedb_by_doc = {d: set() for d in doc_ids}

    result: Dict[int, dict] = {}
    for doc in docs:
        did = int(doc.id)
        summary = summaries.get(
            did, {"chunk_count": 0, "leaf_count": 0, "leaves_with_embedding": 0}
        )
        text_h = _text_health(summary)
        colpali_h = _colpali_health(
            is_pdf=(doc.title or "").lower().endswith(".pdf"),
            lancedb_ids=lancedb_by_doc.get(did, set()),
            all_ids=all_ids.get(did, set()),
            anchor_ids=anchor_ids.get(did, set()),
            leaf_ids=leaf_ids.get(did, set()),
        )
        enrichment_h = _enrichment_health(
            enrichment_counts.get(did, {"enrichment_count": 0}),
            summary["chunk_count"],
        )

        if doc.processing_status in _IN_PROGRESS_STATUSES:
            overall, suggested = "in_progress", None
        else:
            overall, suggested = _overall_and_mode(text_h, colpali_h, enrichment_h)

        result[did] = {
            "document_id": did,
            "overall": overall,
            "suggested_reindex_mode": suggested,
            "text": text_h,
            "colpali": colpali_h,
            "enrichment": enrichment_h,
            "classification_status": doc.classification_status,
            "processing_status": doc.processing_status,
        }
    return result


def build_indexing_health_issues(health: dict) -> List[str]:
    """Issues lisibles pour le diagnostic d'un document (bouton Diagnostiquer / logs)."""
    issues: List[str] = []
    text_h = health.get("text") or {}
    colpali_h = health.get("colpali") or {}
    enrichment_h = health.get("enrichment") or {}

    if text_h.get("status") == "missing":
        issues.append("Texte : aucun chunk ou aucun embedding — retraitement complet requis.")
    elif text_h.get("status") == "partial":
        issues.append(
            f"Texte : {text_h.get('missing_embeddings', 0)} feuille(s) sans embedding."
        )

    status = colpali_h.get("status")
    if status == "desync":
        issues.append(
            f"ColPali désynchronisé : {colpali_h.get('orphan_count', 0)} patch(es) orphelin(s), "
            f"{colpali_h.get('legacy_count', 0)} obsolète(s) — retraiter en mode colpali_only."
        )
    elif status == "partial":
        issues.append(
            f"ColPali incomplet : {colpali_h.get('missing_count', 0)} page(s) sans index visuel."
        )
    elif status == "missing":
        issues.append("ColPali : aucun patch indexé pour ce document.")
    elif status == "unknown":
        issues.append("ColPali : scan LanceDB en échec — état de sync inconnu.")

    if enrichment_h.get("status") == "missing":
        issues.append(
            "Chunks contextuels : aucune synthèse pour ce document — "
            "retraiter en mode enrichment_only."
        )

    return issues
