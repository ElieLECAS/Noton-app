"""Santé d'indexation par document — texte, ColPali (sync LanceDB), chunks contextuels.

Répond à la question « ce document a-t-il TOUT (texte, ColPali, entités, catégories),
et faut-il le retraiter — en quel mode ? ». Croise l'état Postgres (chunks,
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
                   COUNT(*) FILTER (WHERE is_leaf) AS leaves
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
        }
        for r in rows
    }


def _fetch_chunk_id_sets(
    session: Session, doc_ids: List[int]
) -> tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """(all_ids, anchor_ids) par document — pour l'audit de sync ColPali."""
    rows = session.execute(
        text(
            """
            SELECT document_id, id,
                   (metadata_json->>'content_type') = :anchor AS is_anchor
            FROM documentchunk
            WHERE document_id = ANY(:ids)
            """
        ),
        {"ids": doc_ids, "anchor": CONTENT_TYPE_PAGE_ANCHOR},
    ).all()
    all_ids: Dict[int, Set[int]] = {d: set() for d in doc_ids}
    anchor_ids: Dict[int, Set[int]] = {d: set() for d in doc_ids}
    for r in rows:
        did = int(r.document_id)
        cid = int(r.id)
        all_ids[did].add(cid)
        if r.is_anchor:
            anchor_ids[did].add(cid)
    return all_ids, anchor_ids


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
    """BM25 (tsv_content) est une colonne générée à l'insertion des chunks : dès que le
    texte existe, il est indexé — pas d'état "partial" séparé comme du temps des
    embeddings (échec API possible entre la création du chunk et son embedding)."""
    chunk_count = summary["chunk_count"]
    leaf_count = summary["leaf_count"]
    status = "missing" if chunk_count == 0 else "ok"
    return {
        "status": status,
        "chunk_count": chunk_count,
        "leaf_count": leaf_count,
    }


def _colpali_health(
    *,
    is_pdf: bool,
    lancedb_ids: Optional[Set[int]],
    all_ids: Set[int],
    anchor_ids: Set[int],
) -> dict:
    """Topologie UNIQUE : les patches doivent viser les anchors de page, un par page.

    Tout patch hors anchor (héritage de l'ancien pipeline feuille, qui dupliquait
    chaque page sur tous ses chunks texte) est un ``desync`` réparable in-place
    (endpoint colpali-repair), sans ré-embedding.
    """
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

    if not anchor_ids and not lancedb_ids:
        return {"status": "not_applicable" if not is_pdf else "missing", **base}

    orphans = lancedb_ids - all_ids
    synced = lancedb_ids & anchor_ids
    stale = (lancedb_ids & all_ids) - anchor_ids  # patches vers des chunks existants mais hors cible
    missing = anchor_ids - lancedb_ids

    detail = {
        "expected_pages": len(anchor_ids),
        "indexed_pages": len(synced),
        "orphan_count": len(orphans),
        "missing_count": len(missing),
        "legacy_count": len(stale),
    }
    if orphans or stale:
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
    all_ids, anchor_ids = _fetch_chunk_id_sets(session, doc_ids)
    enrichment_counts = _fetch_enrichment_counts(session, doc_ids)

    if settings.COLPALI_ENABLED:
        from app.services.lancedb_service import get_colpali_chunk_ids_by_document

        lancedb_by_doc = get_colpali_chunk_ids_by_document(doc_ids)
    else:
        lancedb_by_doc = {d: set() for d in doc_ids}

    result: Dict[int, dict] = {}
    for doc in docs:
        did = int(doc.id)
        summary = summaries.get(did, {"chunk_count": 0, "leaf_count": 0})
        text_h = _text_health(summary)
        colpali_h = _colpali_health(
            is_pdf=(doc.title or "").lower().endswith(".pdf"),
            lancedb_ids=lancedb_by_doc.get(did, set()),
            all_ids=all_ids.get(did, set()),
            anchor_ids=anchor_ids.get(did, set()),
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


# Remédiations, de la MOINS coûteuse à la plus coûteuse. `colpali_repair` ne passe
# jamais le modèle (pur I/O LanceDB) : quand il suffit, il doit être proposé en premier.
ACTION_NONE = "none"
ACTION_COLPALI_REPAIR = "colpali_repair"
ACTION_COLPALI_ONLY = "colpali_only"
ACTION_ENRICHMENT_ONLY = "enrichment_only"
ACTION_FULL = "full"

_ACTION_LABELS = {
    ACTION_COLPALI_REPAIR: "Réparer ColPali",
    ACTION_COLPALI_ONLY: "Retraiter ColPali",
    ACTION_ENRICHMENT_ONLY: "Régénérer les synthèses",
    ACTION_FULL: "Retraitement complet",
}
# "free" = aucun passage du modèle (secondes) ; "heavy" = ré-embedding / appels LLM.
_ACTION_COSTS = {
    ACTION_COLPALI_REPAIR: "free",
    ACTION_COLPALI_ONLY: "heavy",
    ACTION_ENRICHMENT_ONLY: "heavy",
    ACTION_FULL: "heavy",
}


def recommended_action(health: dict) -> dict:
    """Remédiation la MOINS coûteuse qui règle réellement le document.

    Distinction clé sur un ColPali ``desync`` — les deux cas n'ont pas le même remède :

    * des patches valides mais rattachés au mauvais chunk (``legacy_count``, héritage du
      pipeline feuilles) se ré-attachent aux ancres SANS ré-embedding → ``colpali_repair`` ;
    * des patches orphelins seuls (chunks détruits) ne sont pas récupérables : la
      réparation ne ferait que purger, il faut ré-embedder → ``colpali_only``.

    Retourne ``{action, label, cost, reason}``. ``cost="free"`` signale une remédiation
    sans passage du modèle, à lancer en premier.
    """
    text_h = health.get("text") or {}
    colpali_h = health.get("colpali") or {}
    enrichment_h = health.get("enrichment") or {}

    if health.get("overall") == "in_progress":
        return {
            "action": ACTION_NONE,
            "label": "Traitement en cours",
            "cost": "none",
            "reason": "Le document est en cours de traitement — audit disponible ensuite.",
        }

    if text_h.get("status") == "missing":
        return {
            "action": ACTION_FULL,
            "label": _ACTION_LABELS[ACTION_FULL],
            "cost": "heavy",
            "reason": "Aucun chunk texte : ni BM25 ni ColPali n'ont de cible.",
        }

    colpali_status = colpali_h.get("status")
    enrichment_broken = enrichment_h.get("status") == "missing"

    if colpali_status == "desync" and (colpali_h.get("legacy_count") or 0) > 0:
        orphans = colpali_h.get("orphan_count") or 0
        reason = (
            f"{colpali_h.get('legacy_count')} patch(es) hors ancre de page : "
            "ré-attachables aux ancres sans ré-embedding."
        )
        if orphans:
            reason += f" {orphans} orphelin(s) seront purgés (un ColPali seul restera peut-être requis)."
        return {
            "action": ACTION_COLPALI_REPAIR,
            "label": _ACTION_LABELS[ACTION_COLPALI_REPAIR],
            "cost": "free",
            "reason": reason,
        }

    if colpali_status in ("desync", "missing", "partial"):
        if colpali_status == "desync":
            reason = (
                f"{colpali_h.get('orphan_count', 0)} patch(es) orphelin(s) : les vecteurs "
                "pointent vers des chunks détruits, le document est invisible pour ColPali."
            )
        elif colpali_status == "missing":
            reason = "Aucun index visuel pour ce document."
        else:
            reason = (
                f"{colpali_h.get('missing_count', 0)} page(s) sans index visuel "
                f"({colpali_h.get('indexed_pages', 0)}/{colpali_h.get('expected_pages', 0)})."
            )
        if enrichment_broken:
            return {
                "action": ACTION_FULL,
                "label": _ACTION_LABELS[ACTION_FULL],
                "cost": "heavy",
                "reason": reason + " Les synthèses manquent également.",
            }
        return {
            "action": ACTION_COLPALI_ONLY,
            "label": _ACTION_LABELS[ACTION_COLPALI_ONLY],
            "cost": "heavy",
            "reason": reason,
        }

    if enrichment_broken:
        return {
            "action": ACTION_ENRICHMENT_ONLY,
            "label": _ACTION_LABELS[ACTION_ENRICHMENT_ONLY],
            "cost": "heavy",
            "reason": "Aucun chunk contextuel : le retrieval texte perd les synthèses de fenêtre.",
        }

    if colpali_status == "unknown":
        return {
            "action": ACTION_NONE,
            "label": "Scan LanceDB en échec",
            "cost": "none",
            "reason": "État de synchronisation ColPali inconnu — vérifier LanceDB.",
        }

    return {
        "action": ACTION_NONE,
        "label": "Rien à faire",
        "cost": "none",
        "reason": "Indexation complète.",
    }


def build_indexing_issues_report(
    session: Session, documents: List[Document]
) -> dict:
    """Rapport d'indexation pour le tableau de bord admin.

    Ne remonte QUE les documents à problème, chacun avec sa remédiation recommandée,
    et des totaux permettant d'annoncer d'un coup d'œil « N réparables gratuitement,
    M à retraiter ».
    """
    health_by_doc = build_indexing_health_bulk(session, documents)
    doc_by_id = {int(d.id): d for d in documents if d.id is not None}

    totals = {
        "documents": len(health_by_doc),
        "ok": 0,
        "in_progress": 0,
        "issues": 0,
        "free_fix": 0,
        "heavy_fix": 0,
        "by_action": {},
    }
    issues: List[dict] = []

    for did, health in health_by_doc.items():
        action = recommended_action(health)
        if health.get("overall") == "in_progress":
            totals["in_progress"] += 1
            continue
        if health.get("overall") == "ok" and action["action"] == ACTION_NONE:
            totals["ok"] += 1
            continue

        totals["issues"] += 1
        if action["cost"] == "free":
            totals["free_fix"] += 1
        elif action["cost"] == "heavy":
            totals["heavy_fix"] += 1
        totals["by_action"][action["action"]] = (
            totals["by_action"].get(action["action"], 0) + 1
        )

        doc = doc_by_id.get(did)
        issues.append(
            {
                **health,
                "title": (doc.title if doc else None) or "Sans titre",
                "folder_id": getattr(doc, "folder_id", None),
                "recommended_action": action,
            }
        )

    # Gratuit d'abord (à lancer en premier), puis erreurs avant avertissements.
    severity = {"error": 0, "warning": 1}
    issues.sort(
        key=lambda i: (
            0 if i["recommended_action"]["cost"] == "free" else 1,
            severity.get(i.get("overall"), 2),
            (i.get("title") or "").lower(),
        )
    )
    return {"totals": totals, "documents": issues}


def build_indexing_health_issues(health: dict) -> List[str]:
    """Issues lisibles pour le diagnostic d'un document (bouton Diagnostiquer / logs)."""
    issues: List[str] = []
    text_h = health.get("text") or {}
    colpali_h = health.get("colpali") or {}
    enrichment_h = health.get("enrichment") or {}

    if text_h.get("status") == "missing":
        issues.append("Texte : aucun chunk indexé — retraitement complet requis.")

    status = colpali_h.get("status")
    if status == "desync":
        issues.append(
            f"ColPali désynchronisé : {colpali_h.get('orphan_count', 0)} patch(es) orphelin(s), "
            f"{colpali_h.get('legacy_count', 0)} hors anchor — lancer la réparation de topologie "
            "(colpali-repair, sans ré-embedding) ou retraiter en mode colpali_only."
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
