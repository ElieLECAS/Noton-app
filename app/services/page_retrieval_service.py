"""
Retrieval hybride au niveau page : ColPali + BM25, fusion RRF,
expansion small-to-big (L1 consolidés) et voisinage conditionnel N±1.
"""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import text
from sqlmodel import Session, select

from app.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.document_service_new import feedback_corrective_sql_filter
logger = logging.getLogger(__name__)

CONTENT_TYPE_SEMANTIC_LEAF = "semantic_leaf"
CONTENT_TYPE_PAGE_ANCHOR = "page_anchor"
CONTENT_TYPE_CONTEXTUAL_ENRICHMENT = "contextual_enrichment"

_BM25_STOPWORDS = {
    "pour", "dans", "avec", "une", "des", "les", "est", "sur", "pas", "plus", "que",
    "qui", "quoi", "comment", "quel", "quelle", "quels", "quelles", "par", "sans",
    "mais", "donc", "car", "son", "ses", "notre", "nos", "votre", "vos", "leur",
    "leurs", "tout", "tous", "toute", "toutes", "autre", "autres", "aussi", "très",
    "bien", "encore", "ici", "entre", "après", "avant", "sous", "chez", "vers",
    "depuis", "pendant", "comme", "utiliser", "quelle", "quelle", "régler", "faire",
    "the", "and", "for", "with", "this", "that", "what", "how", "from", "you", "your",
}


@dataclass
class UnifiedPageHit:
    """Hit page unifié pour la fusion multimodale (ColPali + BM25)."""

    document_id: int
    page_no: int

    colpali_score: Optional[float] = None
    bm25_score: Optional[float] = None

    rrf_score: float = 0.0
    final_rank: int = 0
    rerank_score: Optional[float] = None
    rerank_raw_score: Optional[float] = None

    document_title: str = "Document sans titre"
    retrieval_sources: List[str] = field(default_factory=list)
    chunk_id: Optional[int] = None

    text_chunks: List[DocumentChunk] = field(default_factory=list)
    neighbor_pages: List[int] = field(default_factory=list)
    expansion_reason: Optional[str] = None

    # Pages sources d'un chunk d'enrichissement contextuel ayant matché cette page
    # (déplié à l'expansion pour retourner tout le batch 1/2/3 pages).
    enrichment_source_pages: List[int] = field(default_factory=list)

    # Groupe de requêtes source (multi-query) — index et label du groupe ayant produit ce hit.
    query_group_index: int = 0
    query_group_label: str = ""

    @property
    def page_key(self) -> str:
        return f"{self.document_id}:{self.page_no}"


def _page_no_sql_expr(prefix: str = "dc") -> str:
    """Expression SQL pour extraire le numéro de page depuis metadata_json / metadata_."""
    return f"""COALESCE(
        ({prefix}.metadata_json->>'page_no')::int,
        ({prefix}.metadata_json->>'page_start')::int,
        ({prefix}.metadata_->>'page_no')::int,
        ({prefix}.metadata_->>'page_start')::int
    )"""


def _semantic_leaf_filter(prefix: str = "dc") -> str:
    return (
        f"COALESCE({prefix}.metadata_json->>'content_type', "
        f"{prefix}.metadata_->>'content_type', '') = 'semantic_leaf'"
    )


def _retrievable_text_leaf_filter(prefix: str = "dc") -> str:
    """Chunks texte indexés pour retrieval vectoriel / BM25 (source + enrichissement)."""
    return (
        f"COALESCE({prefix}.metadata_json->>'content_type', "
        f"{prefix}.metadata_->>'content_type', '') IN "
        f"('semantic_leaf', 'contextual_enrichment')"
    )


def _enrichment_source_pages_agg(prefix: str = "dc") -> str:
    """Agrège (MAX) les source_pages des chunks d'enrichissement ayant matché une page.

    Utilisé dans les requêtes agrégées par page (BM25) pour savoir, quand un
    chunk `contextual_enrichment` figure parmi les chunks retrouvés d'une page, sur quelles
    pages sources (batch) il s'étend — afin de les déplier à l'expansion.
    """
    return (
        f"MAX(CASE WHEN COALESCE({prefix}.metadata_json->>'content_type', "
        f"{prefix}.metadata_->>'content_type', '') = '{CONTENT_TYPE_CONTEXTUAL_ENRICHMENT}' "
        f"THEN COALESCE({prefix}.metadata_json->>'source_pages', "
        f"{prefix}.metadata_->>'source_pages') END)"
    )


def _parse_source_pages(raw: Any) -> List[int]:
    """Normalise une valeur source_pages (liste, JSON texte ou None) en List[int]."""
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    if not isinstance(raw, (list, tuple)):
        return []
    pages: List[int] = []
    for value in raw:
        try:
            pages.append(int(value))
        except (TypeError, ValueError):
            continue
    return pages


def _enrichment_pages_from_meta(meta: dict) -> List[int]:
    """source_pages d'un chunk si c'est un enrichissement contextuel, sinon []."""
    if (meta or {}).get("content_type") != CONTENT_TYPE_CONTEXTUAL_ENRICHMENT:
        return []
    return _parse_source_pages(meta.get("source_pages"))


def _bulk_resolve_chunk_to_page(
    session: Session,
    chunk_ids: List[int],
) -> Dict[int, Dict[str, Any]]:
    """Résout en une requête les chunk_id ColPali vers document_id, page_no, titre."""
    if not chunk_ids:
        return {}

    sql = text(f"""
        SELECT
            dc.id,
            dc.document_id,
            {_page_no_sql_expr("dc")} AS page_no,
            d.title AS document_title
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        WHERE dc.id IN :chunk_ids
    """)
    rows = session.execute(sql, {"chunk_ids": tuple(chunk_ids)}).all()
    mapping: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if row.page_no is None:
            logger.warning(
                "[bulk_resolve_chunk_to_page] chunk_id=%s sans page_no — ignoré",
                row.id,
            )
            continue
        mapping[int(row.id)] = {
            "document_id": int(row.document_id),
            "page_no": int(row.page_no),
            "document_title": row.document_title or "Document sans titre",
        }
    return mapping


def _bm25_tsquery_fn() -> str:
    return "websearch_to_tsquery" if settings.BM25_USE_WEBSEARCH_QUERY else "plainto_tsquery"


def _top_unified_scores(hits: List[UnifiedPageHit], score_attr: str, n: int = 5) -> List[float]:
    values: List[float] = []
    for hit in hits:
        val = getattr(hit, score_attr, None)
        if val is not None:
            values.append(round(float(val), 4))
    values.sort(reverse=True)
    return values[:n]


def _normalize_bm25_query(query: str) -> str:
    """Normalise guillemets/apostrophes typographiques pour le FTS PostgreSQL."""
    normalized = unicodedata.normalize("NFKC", query or "")
    for src, dst in (
        ("\u2019", "'"),
        ("\u2018", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u00a0", " "),
    ):
        normalized = normalized.replace(src, dst)
    return " ".join(normalized.split())


def _extract_bm25_fallback_query(query: str, max_terms: int = 6) -> str:
    """
    Extrait les termes discriminants (marques, références, mots techniques).
    Retourne une chaîne « term1 OR term2 OR … » pour websearch_to_tsquery.
    """
    normalized = _normalize_bm25_query(query)
    ref_tokens: List[str] = []
    word_tokens: List[str] = []
    seen: Set[str] = set()

    # Priorité : références produit / marques (ROTO, NX, Designo II, codes chiffrés…)
    # On ne retient ici que les tokens « discriminants » (présence d'une majuscule
    # ou d'un chiffre), afin que les articles/prépositions ne les évincent pas.
    for match in re.finditer(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9.\-/]{0,}", normalized):
        token = match.group(0).strip(".-/")
        if len(token) < 2:
            continue
        key = token.lower()
        if key in seen or key in _BM25_STOPWORDS:
            continue
        if any(c.isupper() for c in token) or any(c.isdigit() for c in token):
            seen.add(key)
            ref_tokens.append(token)

    # Mots techniques français (ressort, tension, clé…)
    for word in re.findall(r"[\w'-]{4,}", normalized, flags=re.UNICODE):
        key = word.lower().strip("'")
        if key in _BM25_STOPWORDS or key in seen or len(key) < 4:
            continue
        seen.add(key)
        word_tokens.append(word)

    tokens = ref_tokens + word_tokens
    if not tokens:
        return ""

    top = tokens[:max_terms]
    return " OR ".join(top)


def _build_bm25_or_tsquery(terms: List[str]) -> str:
    """Construit un tsquery OR sûr pour to_tsquery."""
    safe: List[str] = []
    for term in terms:
        cleaned = re.sub(r"[^\w\-]", "", term, flags=re.UNICODE)
        if len(cleaned) >= 2 and cleaned.lower() not in _BM25_STOPWORDS:
            safe.append(cleaned)
    if not safe:
        return ""
    return " | ".join(dict.fromkeys(safe))


def _format_unified_hit_line(hit: UnifiedPageHit, *, show_rrf: bool = False) -> str:
    sources = "+".join(hit.retrieval_sources or ["?"])
    scores: List[str] = []
    if hit.colpali_score is not None:
        scores.append(f"colpali={hit.colpali_score:.3f}")
    if hit.bm25_score is not None:
        scores.append(f"bm25={hit.bm25_score:.3f}")
    if show_rrf:
        scores.append(f"rrf={hit.rrf_score:.4f}")
    if hit.rerank_score is not None:
        scores.append(f"rerank={hit.rerank_score:.3f}")
    score_str = " ".join(scores) if scores else "score=?"
    title = (hit.document_title or "Doc")[:40]
    return f"doc={hit.document_id} p.{hit.page_no} [{sources}] {score_str} — {title}"


def _log_bm25_zero_diagnostic(
    session: Session,
    doc_ids: List[int],
    query_text: str,
) -> None:
    """Diagnostic explicite quand BM25 retourne 0 page."""
    tsquery_fn = _bm25_tsquery_fn()
    normalized = _normalize_bm25_query(query_text)
    fallback = _extract_bm25_fallback_query(query_text)

    try:
        parsed = session.execute(
            text(f"SELECT {tsquery_fn}('french', :query) AS q"),
            {"query": normalized},
        ).scalar()
        chunk_matches = session.execute(
            text(f"""
                SELECT COUNT(*) AS cnt
                FROM documentchunk dc
                WHERE dc.document_id IN :doc_ids
                  AND dc.is_leaf = true
                  AND dc.tsv_content IS NOT NULL
                  AND dc.tsv_content @@ {tsquery_fn}('french', :query)
                  AND {_semantic_leaf_filter("dc")}
            """),
            {"doc_ids": tuple(doc_ids), "query": normalized},
        ).scalar()
        indexed = session.execute(
            text("""
                SELECT COUNT(*) AS cnt
                FROM documentchunk dc
                WHERE dc.document_id IN :doc_ids
                  AND dc.is_leaf = true
                  AND dc.tsv_content IS NOT NULL
                  AND COALESCE(dc.metadata_json->>'content_type', dc.metadata_->>'content_type', '') = 'semantic_leaf'
            """),
            {"doc_ids": tuple(doc_ids)},
        ).scalar()
        fallback_matches = 0
        or_matches = 0
        or_tsquery = ""
        if fallback and " OR " in fallback:
            or_matches = session.execute(
                text(f"""
                    SELECT COUNT(*) AS cnt
                    FROM documentchunk dc
                    WHERE dc.document_id IN :doc_ids
                      AND dc.is_leaf = true
                      AND dc.tsv_content @@ websearch_to_tsquery('french', :query)
                      AND {_semantic_leaf_filter("dc")}
                """),
                {"doc_ids": tuple(doc_ids), "query": fallback},
            ).scalar()
        terms = [t.strip() for t in fallback.replace(" OR ", "|").split("|") if t.strip()] if fallback else []
        or_tsquery = _build_bm25_or_tsquery(terms)
        if or_tsquery:
            or_matches = max(or_matches, session.execute(
                text(f"""
                    SELECT COUNT(*) AS cnt
                    FROM documentchunk dc
                    WHERE dc.document_id IN :doc_ids
                      AND dc.is_leaf = true
                      AND dc.tsv_content @@ to_tsquery('french', :tsq)
                      AND {_semantic_leaf_filter("dc")}
                """),
                {"doc_ids": tuple(doc_ids), "tsq": or_tsquery},
            ).scalar() or 0)

        logger.warning(
            "[BM25 diagnostic] 0 page — tsquery=%s | parsé=%r | chunks indexés=%s | "
            "matches AND=%s | fallback OR=%r → %s matches | to_tsquery OR=%r → %s matches",
            tsquery_fn,
            str(parsed),
            indexed,
            chunk_matches,
            fallback or "(vide)",
            fallback_matches,
            or_tsquery or "(vide)",
            or_matches,
        )
    except Exception as exc:
        logger.warning("[BM25 diagnostic] impossible : %s", exc)


def log_multimodal_retrieval_summary(
    *,
    query_text: str,
    doc_ids: List[int],
    colpali_hits: List[UnifiedPageHit],
    bm25_hits: List[UnifiedPageHit],
    fused_hits: List[UnifiedPageHit],
    final_hits: List[UnifiedPageHit],
    passages: List[Dict[str, Any]],
    images: List[str],
    top_k: int,
    pool_size: int,
    rerank_enabled: bool = False,
    rerank_status: Optional[str] = None,
    dynamic_k: Optional[int] = None,
    protected_hits: Optional[List[UnifiedPageHit]] = None,
) -> None:
    """Résumé lisible en une seule entrée de log (visible dans docker logs)."""
    lines = [
        "══════════════════ RAG MULTIMODAL — RÉSUMÉ ══════════════════",
        f"Requête : {query_text[:100]}{'…' if len(query_text) > 100 else ''}",
        f"Documents : {len(doc_ids)} ids={doc_ids[:8]}{'…' if len(doc_ids) > 8 else ''} | pool={pool_size} top_k={top_k}",
        "",
        "── Étape 1 : Double retriever (ColPali + BM25) ──",
        f"  ColPali  : {len(colpali_hits)} page(s)",
    ]
    for hit in colpali_hits[:5]:
        lines.append(f"    • {_format_unified_hit_line(hit)}")
    if len(colpali_hits) > 5:
        lines.append(f"    … +{len(colpali_hits) - 5} autres")

    lines.append(f"  BM25     : {len(bm25_hits)} page(s)")
    if bm25_hits:
        for hit in bm25_hits[:5]:
            lines.append(f"    • {_format_unified_hit_line(hit)}")
    else:
        lines.append("    • (aucun — voir [BM25 diagnostic] ci-dessus si 0)")

    lines.extend(["", "── Étape 2 : Fusion RRF ──"])
    if fused_hits:
        for hit in fused_hits[: min(len(fused_hits), pool_size)]:
            lines.append(f"  • {_format_unified_hit_line(hit, show_rrf=True)}")
        if len(fused_hits) > pool_size:
            lines.append(f"  … +{len(fused_hits) - pool_size} autres dans le pool")
    else:
        lines.append("  (aucune page dans le pool RRF)")

    if rerank_enabled:
        lines.extend(["", "── Étape 2b : Rerank MiniLM ──"])
        lines.append(f"  Status     : {rerank_status or 'ok'}")
        lines.append(f"  K dynamique: {dynamic_k if dynamic_k is not None else len(final_hits)} (max {top_k})")
        if final_hits:
            for hit in final_hits:
                lines.append(f"  #{hit.final_rank} {_format_unified_hit_line(hit, show_rrf=True)}")
        else:
            lines.append("  (aucune page retenue après rerank)")
        if protected_hits:
            lines.append(f"  Slots ColPali protégés : {len(protected_hits)}")
            for hit in protected_hits:
                lines.append(f"    ↳ {_format_unified_hit_line(hit)}")
    elif final_hits:
        lines.extend(["", "── Étape 2b : Sélection (sans rerank, quota + slots ColPali) ──"])
        for hit in final_hits:
            lines.append(f"  #{hit.final_rank} {_format_unified_hit_line(hit, show_rrf=True)}")
        if protected_hits:
            lines.append(f"  Slots ColPali protégés : {len(protected_hits)}")
            for hit in protected_hits:
                lines.append(f"    ↳ {_format_unified_hit_line(hit)}")

    page_nums = sorted(
        {
            pno
            for p in passages
            for pno in range(
                int(p.get("page_start") or p.get("page_no") or 0),
                int(p.get("page_end") or p.get("page_no") or 0) + 1,
            )
            if pno > 0
        }
    )
    lines.extend([
        "",
        "── Étape 3 : Envoi LLM ──",
        f"  Passages texte : {len(passages)}",
        f"  Images PNG     : {len(images)} (max {settings.RAG_MAX_IMAGES})",
        f"  Pages couvertes: {page_nums[:15]}{'…' if len(page_nums) > 15 else ''}",
        "══════════════════════════════════════════════════════════════",
    ])
    logger.info("\n".join(lines))


def _merged_chunk_metadata(primary: Optional[dict], legacy: Optional[dict]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if isinstance(legacy, dict):
        merged.update(legacy)
    if isinstance(primary, dict):
        merged.update(primary)
    return merged


def _page_no_from_metadata(meta: dict) -> Optional[int]:
    for key in ("page_no", "page_start", "page"):
        val = meta.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return None


def get_space_document_ids(
    session: Session,
    space_id: int,
    document_filter: str = "all",
) -> List[int]:
    filter_clause = feedback_corrective_sql_filter(document_filter, "d")
    extra_clauses = ["AND d.classification_status = 'complete'"]
    params: Dict[str, Any] = {"space_id": space_id}

    extra_sql = "\n          ".join(extra_clauses)
    sql_docs = text(f"""
        SELECT DISTINCT d.id
        FROM document d
        INNER JOIN document_space ds ON ds.document_id = d.id
        WHERE ds.space_id = :space_id
          {filter_clause}
          {extra_sql}
    """)
    return [row[0] for row in session.execute(sql_docs, params)]


def retrieve_colpali_pages(
    session: Session,
    doc_ids: List[int],
    query_text: str,
    limit: int,
) -> List[UnifiedPageHit]:
    """Retrieval ColPali — retourne des pages directement."""
    if not doc_ids or not settings.COLPALI_ENABLED:
        return []

    from app.services.colpali_service import embed_query_colpali
    from app.services.lancedb_service import search_colpali_lancedb

    query_token_embeddings = embed_query_colpali(query_text)
    if not query_token_embeddings:
        return []

    search_results = search_colpali_lancedb(query_token_embeddings, doc_ids, limit=limit)
    if not search_results:
        return []

    chunk_ids = [int(row["id"]) for row in search_results]
    page_mapping = _bulk_resolve_chunk_to_page(session, chunk_ids)
    unmapped_ids = [cid for cid in chunk_ids if cid not in page_mapping]
    if unmapped_ids:
        try:
            existing = session.execute(
                text("SELECT id FROM documentchunk WHERE id IN :ids"),
                {"ids": tuple(unmapped_ids[:50])},
            ).all()
            existing_set = {int(r.id) for r in existing}
            orphans = [cid for cid in unmapped_ids if cid not in existing_set]
            no_page = [cid for cid in unmapped_ids if cid in existing_set]
            logger.warning(
                "[retrieve_colpali_pages] %d/%d chunks LanceDB non mappés — "
                "orphelins DB=%d (patches LanceDB obsolètes → re-indexer ColPali) | "
                "sans page_no=%d | échantillon orphelins=%s",
                len(unmapped_ids),
                len(chunk_ids),
                len(orphans),
                len(no_page),
                orphans[:5],
            )
        except Exception:
            logger.warning(
                "[retrieve_colpali_pages] %d/%d chunks LanceDB sans page_no mappable (re-indexation ColPali ?)",
                len(unmapped_ids),
                len(chunk_ids),
            )

    best_by_page: Dict[str, UnifiedPageHit] = {}
    for result in search_results:
        chunk_id = int(result["id"])
        page_info = page_mapping.get(chunk_id)
        if not page_info:
            continue
        similarity = 1.0 - float(result.get("_distance", 1.0))
        key = f"{page_info['document_id']}:{page_info['page_no']}"
        existing = best_by_page.get(key)
        if existing is None or similarity > (existing.colpali_score or 0):
            best_by_page[key] = UnifiedPageHit(
                document_id=page_info["document_id"],
                page_no=page_info["page_no"],
                colpali_score=similarity,
                document_title=page_info["document_title"],
                retrieval_sources=["colpali"],
                chunk_id=chunk_id,
            )

    hits = sorted(best_by_page.values(), key=lambda h: h.colpali_score or 0, reverse=True)
    if hits:
        logger.info(
            "[retrieve_colpali_pages] %d pages — meilleure: %s",
            len(hits),
            _format_unified_hit_line(hits[0]),
        )
    else:
        logger.info("[retrieve_colpali_pages] 0 pages (LanceDB=%d patches, mappables=%d)", len(search_results), len(page_mapping))
    return hits[:limit]


def filter_colpali_pages_dynamic(
    hits: List[UnifiedPageHit],
    *,
    min_threshold: Optional[float] = None,
    relative_margin: Optional[float] = None,
) -> List[UnifiedPageHit]:
    """Seuil absolu + marge relative sur le score ColPali (parité pipeline legacy)."""
    min_threshold = min_threshold if min_threshold is not None else settings.COLPALI_MIN_THRESHOLD
    relative_margin = relative_margin if relative_margin is not None else settings.COLPALI_RELATIVE_MARGIN
    if not hits:
        return []

    above_abs = [h for h in hits if (h.colpali_score or 0.0) >= min_threshold]
    if not above_abs:
        logger.info(
            "[filter_colpali_dynamic] 0 page retenue — max=%.3f < seuil %.3f",
            max((h.colpali_score or 0.0) for h in hits),
            min_threshold,
        )
        return []

    max_score = max(h.colpali_score or 0.0 for h in above_abs)
    cutoff = max_score - relative_margin
    filtered = [h for h in above_abs if (h.colpali_score or 0.0) >= cutoff]
    logger.info(
        "[filter_colpali_dynamic] %d → %d pages (seuil=%.2f cutoff=%.2f max=%.3f)",
        len(hits),
        len(filtered),
        min_threshold,
        cutoff,
        max_score,
    )
    return filtered


def unified_hits_to_eval_passages(hits: List[UnifiedPageHit], top_k: int) -> List[Dict[str, Any]]:
    """Passages légers pour métriques d'éval (sans expansion L1 / PNG)."""
    passages: List[Dict[str, Any]] = []
    for rank, hit in enumerate(hits[:top_k], start=1):
        if hit.rerank_score is not None:
            score = hit.rerank_score
        elif hit.rrf_score:
            score = hit.rrf_score
        else:
            score = hit.colpali_score or hit.bm25_score or 0.0
        passages.append(
            {
                "rank": rank,
                "document_title": hit.document_title,
                "document_id": hit.document_id,
                "page_no": hit.page_no,
                "page_start": hit.page_no,
                "page_end": hit.page_no,
                "score": float(score or 0.0),
                "retrieval_sources": list(hit.retrieval_sources),
                "colpali_score": hit.colpali_score,
                "bm25_score": hit.bm25_score,
                "rerank_score": hit.rerank_score,
            }
        )
    return passages


def _run_bm25_pages_query(
    session: Session,
    doc_ids: List[int],
    query: str,
    limit: int,
) -> List[Any]:
    """Exécute la requête BM25 agrégée par page."""
    tsquery_fn = _bm25_tsquery_fn()
    page_no_expr = _page_no_sql_expr("dc")
    semantic_filter = f"AND {_retrievable_text_leaf_filter('dc')}" if settings.BM25_FILTER_SEMANTIC_LEAF else ""

    sql = text(f"""
        SELECT
            dc.document_id,
            {page_no_expr} AS page_no,
            d.title AS document_title,
            MAX(ts_rank_cd(dc.tsv_content, {tsquery_fn}('french', :query))) AS rank,
            {_enrichment_source_pages_agg("dc")} AS enrichment_source_pages_text
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        WHERE dc.document_id IN :doc_ids
          AND dc.is_leaf = true
          AND dc.tsv_content IS NOT NULL
          AND dc.tsv_content @@ {tsquery_fn}('french', :query)
          AND {page_no_expr} IS NOT NULL
          {semantic_filter}
        GROUP BY dc.document_id, {page_no_expr}, d.title
        ORDER BY rank DESC
        LIMIT :limit
    """)
    return session.execute(
        sql,
        {"doc_ids": tuple(doc_ids), "query": query, "limit": limit},
    ).all()


def _run_bm25_or_pages_query(
    session: Session,
    doc_ids: List[int],
    or_tsquery: str,
    limit: int,
) -> List[Any]:
    """Recherche BM25 avec to_tsquery OR (fallback quand AND échoue)."""
    page_no_expr = _page_no_sql_expr("dc")
    semantic_filter = f"AND {_retrievable_text_leaf_filter('dc')}" if settings.BM25_FILTER_SEMANTIC_LEAF else ""

    sql = text(f"""
        SELECT
            dc.document_id,
            {page_no_expr} AS page_no,
            d.title AS document_title,
            MAX(ts_rank_cd(dc.tsv_content, to_tsquery('french', :tsq))) AS rank,
            {_enrichment_source_pages_agg("dc")} AS enrichment_source_pages_text
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        WHERE dc.document_id IN :doc_ids
          AND dc.is_leaf = true
          AND dc.tsv_content IS NOT NULL
          AND dc.tsv_content @@ to_tsquery('french', :tsq)
          AND {page_no_expr} IS NOT NULL
          {semantic_filter}
        GROUP BY dc.document_id, {page_no_expr}, d.title
        ORDER BY rank DESC
        LIMIT :limit
    """)
    return session.execute(
        sql,
        {"doc_ids": tuple(doc_ids), "tsq": or_tsquery, "limit": limit},
    ).all()


def _run_bm25_websearch_or_pages_query(
    session: Session,
    doc_ids: List[int],
    or_query: str,
    limit: int,
) -> List[Any]:
    """Recherche BM25 avec websearch_to_tsquery sur une requête « term1 OR term2 »."""
    page_no_expr = _page_no_sql_expr("dc")
    semantic_filter = f"AND {_retrievable_text_leaf_filter('dc')}" if settings.BM25_FILTER_SEMANTIC_LEAF else ""

    sql = text(f"""
        SELECT
            dc.document_id,
            {page_no_expr} AS page_no,
            d.title AS document_title,
            MAX(ts_rank_cd(dc.tsv_content, websearch_to_tsquery('french', :query))) AS rank,
            {_enrichment_source_pages_agg("dc")} AS enrichment_source_pages_text
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        WHERE dc.document_id IN :doc_ids
          AND dc.is_leaf = true
          AND dc.tsv_content IS NOT NULL
          AND dc.tsv_content @@ websearch_to_tsquery('french', :query)
          AND {page_no_expr} IS NOT NULL
          {semantic_filter}
        GROUP BY dc.document_id, {page_no_expr}, d.title
        ORDER BY rank DESC
        LIMIT :limit
    """)
    return session.execute(
        sql,
        {"doc_ids": tuple(doc_ids), "query": or_query, "limit": limit},
    ).all()


def retrieve_bm25_pages(
    session: Session,
    doc_ids: List[int],
    query_text: str,
    limit: int,
) -> List[UnifiedPageHit]:
    """Retrieval lexical — agrégation SQL par page avec websearch_to_tsquery."""
    if not doc_ids or not query_text.strip():
        return []

    tsquery_fn = _bm25_tsquery_fn()
    normalized_query = _normalize_bm25_query(query_text.strip())
    used_query = normalized_query
    query_mode = "AND"

    try:
        rows = _run_bm25_pages_query(
            session, doc_ids, normalized_query, limit
        )

        if not rows:
            or_websearch = _extract_bm25_fallback_query(query_text)
            if or_websearch:
                rows = _run_bm25_websearch_or_pages_query(session, doc_ids, or_websearch, limit)
                if rows:
                    used_query = or_websearch
                    query_mode = "OR-websearch"
                    logger.info(
                        "[retrieve_bm25_pages] fallback OR-websearch — %r → %d pages",
                        or_websearch[:80],
                        len(rows),
                    )

        if not rows:
            terms = [t.strip() for t in _extract_bm25_fallback_query(query_text).replace(" OR ", "|").split("|") if t.strip()]
            or_tsquery = _build_bm25_or_tsquery(terms)
            if or_tsquery:
                rows = _run_bm25_or_pages_query(session, doc_ids, or_tsquery, limit)
                if rows:
                    used_query = or_tsquery
                    query_mode = "OR-tsquery"
                    logger.info(
                        "[retrieve_bm25_pages] fallback OR-tsquery — %r → %d pages",
                        or_tsquery,
                        len(rows),
                    )
    except Exception as exc:
        logger.warning("[retrieve_bm25_pages] recherche échouée (%s) : %s", tsquery_fn, exc)
        return []

    if not rows:
        _log_bm25_zero_diagnostic(session, doc_ids, normalized_query)
        logger.info(
            "[retrieve_bm25_pages] 0 pages — %s n'a matché aucun chunk (requête trop conversationnelle ?)",
            tsquery_fn,
        )
        return []

    hits = [
        UnifiedPageHit(
            document_id=int(row.document_id),
            page_no=int(row.page_no),
            bm25_score=float(row.rank or 0.0),
            document_title=row.document_title or "Document sans titre",
            retrieval_sources=["bm25"],
            enrichment_source_pages=_parse_source_pages(
                getattr(row, "enrichment_source_pages_text", None)
            ),
        )
        for row in rows
    ]
    logger.info(
        "[retrieve_bm25_pages] %d pages — meilleure: %s (mode=%s query=%r)",
        len(hits),
        _format_unified_hit_line(hits[0]),
        query_mode,
        used_query[:60],
    )
    return hits


def fuse_multimodal_hits(
    colpali_hits: List[UnifiedPageHit],
    bm25_hits: List[UnifiedPageHit],
    *,
    rrf_k: Optional[int] = None,
    top_k: int = 10,
    min_colpali_score: Optional[float] = None,
) -> List[UnifiedPageHit]:
    """Fusion RRF page-centric sans pré-filtrage ColPali."""
    rrf_k = rrf_k if rrf_k is not None else settings.RRF_K
    min_colpali_score = (
        min_colpali_score
        if min_colpali_score is not None
        else settings.COLPALI_POST_FUSION_MIN_SCORE
    )

    page_index: Dict[str, UnifiedPageHit] = {}

    def add_channel(hits: List[UnifiedPageHit], channel: str) -> None:
        score_attr = f"{channel}_score"
        sorted_hits = sorted(
            hits,
            key=lambda h: getattr(h, score_attr) or 0.0,
            reverse=True,
        )
        for rank_idx, hit in enumerate(sorted_hits, start=1):
            key = hit.page_key
            if key not in page_index:
                page_index[key] = UnifiedPageHit(
                    document_id=hit.document_id,
                    page_no=hit.page_no,
                    document_title=hit.document_title,
                    chunk_id=hit.chunk_id,
                )
            target = page_index[key]
            target.rrf_score += 1.0 / (rrf_k + rank_idx)
            current_score = getattr(hit, score_attr)
            if current_score is not None:
                setattr(target, score_attr, current_score)
            if channel not in target.retrieval_sources:
                target.retrieval_sources.append(channel)
            if hit.document_title:
                target.document_title = hit.document_title
            if hit.chunk_id is not None:
                target.chunk_id = hit.chunk_id
            if hit.enrichment_source_pages:
                merged_pages = set(target.enrichment_source_pages)
                merged_pages.update(hit.enrichment_source_pages)
                target.enrichment_source_pages = sorted(merged_pages)

    add_channel(colpali_hits, "colpali")
    add_channel(bm25_hits, "bm25")

    if not page_index:
        return []

    all_hits = sorted(page_index.values(), key=lambda h: h.rrf_score, reverse=True)
    quality_filtered: List[UnifiedPageHit] = []
    for hit in all_hits:
        if hit.retrieval_sources == ["colpali"] and (hit.colpali_score or 0) < min_colpali_score:
            logger.debug(
                "[fuse_multimodal] page %s ColPali-only score %.3f < seuil %.3f — filtrée",
                hit.page_key,
                hit.colpali_score or 0,
                min_colpali_score,
            )
            continue
        quality_filtered.append(hit)

    final = quality_filtered[:top_k]
    for rank, hit in enumerate(final, start=1):
        hit.final_rank = rank
    return final


def select_final_hits(
    fused_hits: List[UnifiedPageHit],
    top_k: int,
    *,
    per_doc_quota_ratio: Optional[float] = None,
    colpali_slots: Optional[int] = None,
) -> Tuple[List[UnifiedPageHit], List[UnifiedPageHit]]:
    """Coupe du pool RRF vers le top-K final, avec deux garde-fous d'équité.

    Remplace la coupe brute ``fused_hits[:top_k]`` du chemin SANS reranker, qui laissait
    deux biais structurels décider seuls :

    * **Volume** — aucun plafond par document : un catalogue de 200 pages pouvait occuper
      les K slots, puis remporter l'élection CAG grâce à ce volume qu'il venait de
      fabriquer. Le quota est SOUPLE : les slots restés vides après le premier passage
      sont rendus aux hits écartés, donc il ne mord qu'en situation de compétition.
    * **Pages visuelles muettes en texte** — une page sans aucune évidence lexicale
      (dessin coté, schéma) que seul ColPali sait voir peut être éjectée par des pages
      texte moyennes mais plus nombreuses. La protection existante
      (``protect_colpali_visual_hits``) ne vit que dans le chemin du reranker :
      reranker désactivé = aucune protection. On réserve donc ici les mêmes slots.

    Retourne ``(final_hits, protected_hits)``. ``final_rank`` est réaffecté sur le résultat.
    """
    if not fused_hits:
        return [], []

    ratio = (
        per_doc_quota_ratio
        if per_doc_quota_ratio is not None
        else settings.RETRIEVAL_PER_DOC_QUOTA_RATIO
    )
    slots = colpali_slots if colpali_slots is not None else settings.COLPALI_PROTECTED_SLOTS

    if ratio and ratio > 0:
        quota = max(1, math.ceil(top_k * ratio))
        selected: List[UnifiedPageHit] = []
        deferred: List[UnifiedPageHit] = []
        per_doc: Dict[int, int] = {}
        for hit in fused_hits:
            if len(selected) >= top_k:
                break
            doc_id = int(hit.document_id)
            if per_doc.get(doc_id, 0) >= quota:
                deferred.append(hit)
                continue
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            selected.append(hit)
        quota_kept = sum(per_doc.values())
        if len(selected) < top_k and deferred:
            # Quota souple : personne d'autre ne réclame ces slots → on les rend.
            selected.extend(deferred[: top_k - len(selected)])
        if deferred:
            logger.info(
                "[select_final_hits] quota %d page(s)/document — %d hit(s) écarté(s), "
                "%d réintégré(s) faute de concurrence",
                quota,
                len(deferred),
                len(selected) - quota_kept,
            )
    else:
        selected = list(fused_hits[:top_k])

    protected: List[UnifiedPageHit] = []
    if slots and slots > 0:
        # Import local : page_reranker_service importe ce module (cycle au niveau module).
        from app.services.page_reranker_service import is_colpali_visual_priority

        selected_keys = {h.page_key for h in selected}
        already_visual = sum(1 for h in selected if is_colpali_visual_priority(h))
        missing = slots - already_visual
        if missing > 0:
            candidates = [
                h
                for h in fused_hits
                if h.page_key not in selected_keys and is_colpali_visual_priority(h)
            ]
            candidates.sort(key=lambda h: h.colpali_score or 0.0, reverse=True)
            protected = candidates[:missing]
            for hit in protected:
                logger.info(
                    "[select_final_hits] slot ColPali réservé (hors rerank) doc=%s p.%s "
                    "colpali=%.3f bm25=%s",
                    hit.document_id,
                    hit.page_no,
                    hit.colpali_score or 0.0,
                    f"{hit.bm25_score:.3f}" if hit.bm25_score is not None else "—",
                )

    final = selected + protected
    for rank, hit in enumerate(final, start=1):
        hit.final_rank = rank
    return final, protected


def fuse_multi_query_groups(
    per_group_hits: List[List[UnifiedPageHit]],
    *,
    pool_size: int = 30,
    min_quota_per_group: int = 3,
) -> List[UnifiedPageHit]:
    """Fusion cross-groupes avec quota garanti par groupe.

    Garantit que chaque groupe contribue au minimum `min_quota_per_group` hits
    indépendamment de leurs scores absolus — les pages de la requête 2 ne sont
    jamais écrasées par les 20 meilleurs hits de la requête 1.
    Les pages présentes dans plusieurs groupes reçoivent un bonus de multi-pertinence.
    """
    if not per_group_hits:
        return []
    if len(per_group_hits) == 1:
        return per_group_hits[0][:pool_size]

    n_groups = len(per_group_hits)
    quota = max(min_quota_per_group, pool_size // n_groups)

    # Normalise rrf_score dans [0,1] par groupe
    group_norm: List[List[Tuple[UnifiedPageHit, float]]] = []
    for hits in per_group_hits:
        if not hits:
            group_norm.append([])
            continue
        max_score = max((h.rrf_score for h in hits), default=1.0) or 1.0
        group_norm.append([(h, h.rrf_score / max_score) for h in hits])

    # pool[page_key] = (hit, best_norm_score, count_groups)
    pool: Dict[str, Tuple[UnifiedPageHit, float, int]] = {}

    # Phase 1 : quota garanti par groupe
    for norm_hits in group_norm:
        for hit, ns in norm_hits[:quota]:
            key = hit.page_key
            if key not in pool:
                pool[key] = (hit, ns, 1)
            else:
                old_hit, old_ns, cnt = pool[key]
                best_hit = hit if ns > old_ns else old_hit
                pool[key] = (best_hit, max(old_ns, ns), cnt + 1)

    # Phase 2 : remplissage jusqu'à pool_size avec les meilleurs restants
    if len(pool) < pool_size:
        remaining: List[Tuple[UnifiedPageHit, float]] = []
        for norm_hits in group_norm:
            for hit, ns in norm_hits[quota:]:
                if hit.page_key not in pool:
                    remaining.append((hit, ns))
        remaining.sort(key=lambda x: x[1], reverse=True)
        for hit, ns in remaining:
            if len(pool) >= pool_size:
                break
            key = hit.page_key
            if key not in pool:
                pool[key] = (hit, ns, 1)

    # Phase 3 : tri final — bonus de 0.1 par groupe supplémentaire
    def _combined(entry: Tuple[UnifiedPageHit, float, int]) -> float:
        _, ns, cnt = entry
        return ns + (cnt - 1) * 0.1

    sorted_entries = sorted(pool.values(), key=_combined, reverse=True)
    result = [hit for hit, _, _ in sorted_entries[:pool_size]]
    for rank, hit in enumerate(result, start=1):
        hit.final_rank = rank
    return result


def _apply_enrichment_span_expansion(session: Session, hit: UnifiedPageHit) -> None:
    """Déplie les pages sources d'un chunk d'enrichissement contextuel retrouvé.

    Quand un chunk `contextual_enrichment` couvrant plusieurs pages (1/2/3) a matché la
    page du hit, on charge les L1 de toutes ses `source_pages` afin que le retrieval
    retourne l'intégralité du batch et non uniquement la page principale.
    """
    extra_pages = sorted(
        {p for p in hit.enrichment_source_pages if p and p != hit.page_no}
    )
    if not extra_pages:
        return

    seen_ids = {c.id for c in hit.text_chunks if c.id is not None}
    for pno in extra_pages:
        for chunk in load_l1_chunks_for_page(session, hit.document_id, pno):
            if chunk.id is not None and chunk.id in seen_ids:
                continue
            hit.text_chunks.append(chunk)
            if chunk.id is not None:
                seen_ids.add(chunk.id)
        if pno not in hit.neighbor_pages:
            hit.neighbor_pages.append(pno)

    hit.text_chunks.sort(key=lambda c: (c.chunk_index or 0, c.id or 0))
    hit.neighbor_pages.sort()
    hit.expansion_reason = hit.expansion_reason or "enrichment_span"


def expand_page_context(
    session: Session,
    hits: List[UnifiedPageHit],
    *,
    neighbor_strategy: Optional[str] = None,
) -> List[UnifiedPageHit]:
    """Expansion voisinage N±1 et chargement des chunks L1."""
    strategy = neighbor_strategy or settings.RAG_NEIGHBOR_STRATEGY

    expanded: List[UnifiedPageHit] = []
    for hit in hits:
        hit.text_chunks = load_l1_chunks_for_page(session, hit.document_id, hit.page_no)
        hit.neighbor_pages = []
        hit.expansion_reason = None

        # Dépliage des pages sources d'un chunk d'enrichissement, indépendant de la
        # stratégie de voisinage : un chunk contextuel retrouvé ramène tout son batch.
        _apply_enrichment_span_expansion(session, hit)

        if strategy == "none":
            expanded.append(hit)
            continue

        should_expand = strategy == "always"
        if strategy == "conditional" and hit.text_chunks:
            last_meta = _merged_chunk_metadata(
                hit.text_chunks[-1].metadata_json,
                hit.text_chunks[-1].metadata_,
            )
            if last_meta.get("continues_on_next_page"):
                should_expand = True
                hit.expansion_reason = "continues_on_next_page"

        if should_expand and not _has_cross_page_coverage(session, hit.document_id, hit.page_no):
            next_chunks = load_l1_chunks_for_page(session, hit.document_id, hit.page_no + 1)
            if next_chunks:
                seen_ids = {c.id for c in hit.text_chunks if c.id is not None}
                for chunk in next_chunks:
                    if chunk.id not in seen_ids:
                        hit.text_chunks.append(chunk)
                        if chunk.id is not None:
                            seen_ids.add(chunk.id)
                hit.text_chunks.sort(key=lambda c: (c.chunk_index or 0, c.id or 0))
                if hit.page_no + 1 not in hit.neighbor_pages:
                    hit.neighbor_pages.append(hit.page_no + 1)
                    hit.neighbor_pages.sort()

        expanded.append(hit)
    return expanded


def format_multimodal_passages(
    session: Session,
    hits: List[UnifiedPageHit],
    *,
    max_passage_chars: Optional[int] = None,
    render_all_images: bool = False,
    image_dpi: int = 150,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Formatage passages texte + rendu PNG selon politique ColPali-dominant."""
    import base64
    import os

    from app.config import settings
    from app.services.multimodal_page_service import render_page_png_cached
    from app.services.page_reranker_service import compute_page_image_policy

    if max_passage_chars is None:
        max_passage_chars = settings.SPACE_CONTEXT_MAX_PASSAGE_CHARS

    passages: List[Dict[str, Any]] = []
    images_b64: List[str] = []
    seen_image_keys: Set[str] = set()

    for hit in hits:
        if not hit.text_chunks:
            hit.text_chunks = load_l1_chunks_for_page(session, hit.document_id, hit.page_no)

        consolidated = build_consolidated_page_text(hit.text_chunks)
        if not consolidated:
            consolidated = f"Page {hit.page_no} — contenu visuel uniquement"

        if len(consolidated) > max_passage_chars:
            consolidated = consolidated[:max_passage_chars] + "..."

        pages_included = sorted({hit.page_no, *hit.neighbor_pages})
        page_start = min(pages_included)
        page_end = max(pages_included)
        doc_title = hit.document_title or "Document sans titre"
        passage_text = f"**{doc_title}**\n{consolidated}"

        needs_image = render_all_images or compute_page_image_policy(hit, consolidated)
        image_pages: List[Tuple[int, int]] = []
        if needs_image:
            for pno in pages_included:
                image_pages.append((hit.document_id, pno))

        display_score = hit.rerank_score if hit.rerank_score is not None else hit.rrf_score
        enrichment_chunks = load_enrichment_chunks_for_pages(
            session, hit.document_id, pages_included
        )
        enrichment_passages = _format_enrichment_passages(enrichment_chunks)
        passage_dict: Dict[str, Any] = {
            "rank": hit.final_rank,
            "passage": passage_text,
            "passage_raw": consolidated,
            "document_title": doc_title,
            "document_id": hit.document_id,
            "chunk_id": hit.chunk_id,
            "score": float(display_score or 0.0),
            # page_no = page ancre réellement matchée par le retriever (pas min du span).
            # Écraser page_no par page_start faussait l'éval et les citations quand un
            # chunk d'enrichissement dépliait un batch [N-2..N] sous le n° de sa 1ère page.
            "page_no": hit.page_no,
            "page_start": page_start,
            "page_end": page_end,
            "retrieval_sources": list(hit.retrieval_sources),
            "colpali_score": hit.colpali_score,
            "bm25_score": hit.bm25_score,
            "raw_rrf_score": hit.rrf_score,
            "rerank_score": hit.rerank_score,
            "expanded_neighbor_pages": list(hit.neighbor_pages),
            "expansion_reason": hit.expansion_reason,
            "needs_page_image": needs_image and len(image_pages) > 0,
            "image_pages": image_pages,
            "content_type": "multimodal_page_passage",
            "is_enrichment": False,
            "enrichment_passages": enrichment_passages,
        }
        passages.append(passage_dict)

        if not needs_image:
            continue

        doc = session.get(Document, hit.document_id)
        if not doc or not doc.source_file_path or not os.path.exists(doc.source_file_path):
            continue

        for doc_id, pno in image_pages:
            img_key = f"{doc_id}:{pno}"
            if img_key in seen_image_keys:
                continue
            try:
                png_bytes = render_page_png_cached(doc.source_file_path, pno, dpi=image_dpi)
                images_b64.append(base64.b64encode(png_bytes).decode("utf-8"))
                seen_image_keys.add(img_key)
            except Exception as exc:
                logger.warning(
                    "Erreur render PNG page %s (document %s): %s",
                    pno,
                    doc_id,
                    exc,
                )

    return passages, images_b64


def load_l1_chunks_for_page(
    session: Session,
    document_id: int,
    page_no: int,
) -> List[DocumentChunk]:
    """Charge tous les semantic_leaf (texte source) couvrant une page donnée."""
    stmt = select(DocumentChunk).where(
        DocumentChunk.document_id == document_id,
        DocumentChunk.is_leaf == True,  # noqa: E712
    )
    all_leaves = list(session.exec(stmt).all())
    result: List[DocumentChunk] = []
    for chunk in all_leaves:
        meta = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        content_type = meta.get("content_type")
        if content_type in (CONTENT_TYPE_PAGE_ANCHOR, CONTENT_TYPE_CONTEXTUAL_ENRICHMENT):
            continue
        if content_type not in (None, CONTENT_TYPE_SEMANTIC_LEAF):
            continue
        p_start = meta.get("page_start") or meta.get("page_no")
        p_end = meta.get("page_end") or p_start
        try:
            p_start_i = int(p_start) if p_start is not None else None
            p_end_i = int(p_end) if p_end is not None else p_start_i
        except (TypeError, ValueError):
            continue
        if p_start_i is not None and p_end_i is not None and p_start_i <= page_no <= p_end_i:
            result.append(chunk)
    result.sort(key=lambda c: (c.chunk_index or 0, c.id or 0))
    return result


def load_enrichment_chunks_for_pages(
    session: Session,
    document_id: int,
    page_numbers: List[int],
) -> List[DocumentChunk]:
    """Charge les chunks contextual_enrichment liés aux pages données."""
    if not page_numbers:
        return []

    page_set = set(page_numbers)
    stmt = select(DocumentChunk).where(
        DocumentChunk.document_id == document_id,
        DocumentChunk.is_leaf == True,  # noqa: E712
    )
    result: List[DocumentChunk] = []
    for chunk in session.exec(stmt).all():
        meta = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        if meta.get("content_type") != CONTENT_TYPE_CONTEXTUAL_ENRICHMENT:
            continue
        source_pages = meta.get("source_pages") or []
        source_page = meta.get("source_page") or meta.get("page_no")
        covers = False
        if isinstance(source_pages, list):
            covers = any(int(p) in page_set for p in source_pages if p is not None)
        if not covers and source_page is not None:
            try:
                covers = int(source_page) in page_set
            except (TypeError, ValueError):
                pass
        if not covers:
            p_start = meta.get("page_start") or meta.get("page_no")
            p_end = meta.get("page_end") or p_start
            try:
                p_start_i = int(p_start) if p_start is not None else None
                p_end_i = int(p_end) if p_end is not None else p_start_i
                if p_start_i is not None and p_end_i is not None:
                    covers = any(p_start_i <= p <= p_end_i for p in page_set)
            except (TypeError, ValueError):
                pass
        if covers:
            result.append(chunk)
    result.sort(key=lambda c: (c.chunk_index or 0, c.id or 0))
    return result


def _format_enrichment_passages(
    enrichment_chunks: List[DocumentChunk],
) -> List[Dict[str, Any]]:
    passages: List[Dict[str, Any]] = []
    for chunk in enrichment_chunks:
        meta = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        content = (chunk.content or chunk.text or "").strip()
        if not content:
            continue
        source_page = meta.get("source_page") or meta.get("page_no")
        passages.append(
            {
                "content": content,
                "theme": meta.get("theme"),
                "category_slug": meta.get("category_slug"),
                "source_page": source_page,
                "source_pages": meta.get("source_pages") or [],
                "is_enrichment": True,
                "chunk_id": chunk.id,
            }
        )
    return passages


def build_consolidated_page_text(chunks: List[DocumentChunk]) -> str:
    """Concatène les L1 d'une ou plusieurs pages en texte structuré."""
    parts: List[str] = []
    for chunk in chunks:
        meta = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        content = (chunk.content or chunk.text or "").strip()
        if not content:
            continue
        heading = meta.get("heading") or meta.get("parent_heading")
        step_no = meta.get("step_number")
        if heading:
            parts.append(f"### {heading}")
        elif step_no is not None:
            parts.append(f"### Étape {step_no}")
        parts.append(content)
    return "\n\n".join(parts).strip()


def _chunk_covers_page(meta: dict, page_no: int) -> bool:
    p_start = meta.get("page_start") or meta.get("page_no")
    p_end = meta.get("page_end") or p_start
    try:
        p_start_i = int(p_start) if p_start is not None else None
        p_end_i = int(p_end) if p_end is not None else p_start_i
    except (TypeError, ValueError):
        return False
    return p_start_i is not None and p_end_i is not None and p_start_i <= page_no <= p_end_i


def _has_cross_page_coverage(session: Session, document_id: int, page_no: int) -> bool:
    """True si un chunk L1 couvre déjà page_no..page_no+1 (merge indexation)."""
    for chunk in load_l1_chunks_for_page(session, document_id, page_no):
        meta = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        if meta.get("cross_page_merge"):
            p_end = meta.get("page_end") or meta.get("page_no")
            try:
                if int(p_end) > page_no:
                    return True
            except (TypeError, ValueError):
                pass
    return False


