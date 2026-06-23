"""
Retrieval hybride au niveau page : ColPali + pgvector L1 + BM25, fusion RRF,
expansion small-to-big (L1 consolidés) et voisinage conditionnel N±1.
"""
from __future__ import annotations

import json
import logging
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
class PageRetrievalHit:
    document_id: int
    page_no: int
    score: float = 0.0
    rrf_score: float = 0.0
    retrieval_sources: List[str] = field(default_factory=list)
    colpali_score: Optional[float] = None
    pgvector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    document_title: str = "Document sans titre"
    chunk_id: Optional[int] = None
    enrichment_source_pages: List[int] = field(default_factory=list)

    @property
    def page_key(self) -> str:
        return f"{self.document_id}:{self.page_no}"


@dataclass
class UnifiedPageHit:
    """Hit page unifié pour la fusion multimodale (ColPali + pgvector + BM25 + KAG)."""

    document_id: int
    page_no: int

    colpali_score: Optional[float] = None
    pgvector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    kag_score: Optional[float] = None

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

    Utilisé dans les requêtes agrégées par page (pgvector / BM25) pour savoir, quand un
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


def _category_metadata_filter_clause(
    content_categories: Optional[List[str]],
    prefix: str = "dc",
) -> Tuple[str, Dict[str, Any]]:
    """Clause SQL optionnelle pour filtrer par catégories de contenu (JSONB array)."""
    if not content_categories:
        return "", {}
    slugs = [c.strip().lower() for c in content_categories if c and str(c).strip()]
    if not slugs:
        return "", {}
    clause = (
        f"AND COALESCE({prefix}.metadata_json->'categories', '[]'::jsonb) ?| :category_slugs"
    )
    return clause, {"category_slugs": slugs}


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
    if hit.pgvector_score is not None:
        scores.append(f"vec={hit.pgvector_score:.3f}")
    if hit.bm25_score is not None:
        scores.append(f"bm25={hit.bm25_score:.3f}")
    if hit.kag_score is not None:
        scores.append(f"kag={hit.kag_score:.3f}")
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


def _kag_exclusive_hits(
    kag_hits: List[UnifiedPageHit],
    colpali_hits: List[UnifiedPageHit],
    pgvector_hits: List[UnifiedPageHit],
    bm25_hits: List[UnifiedPageHit],
) -> List[UnifiedPageHit]:
    """Pages trouvées uniquement par le graphe KAG (absentes des 3 autres canaux)."""
    other_keys = set()
    for hits in (colpali_hits, pgvector_hits, bm25_hits):
        other_keys.update(h.page_key for h in hits)
    return [h for h in kag_hits if h.page_key not in other_keys]


def _kag_fusion_gains(
    fused_hits: List[UnifiedPageHit],
    pre_kag_fused_hits: List[UnifiedPageHit],
    *,
    pool_size: int,
) -> List[UnifiedPageHit]:
    """Pages entrées dans le pool RRF grâce au canal KAG (absentes sans graphe)."""
    pre_keys = {h.page_key for h in pre_kag_fused_hits[:pool_size]}
    return [
        h
        for h in fused_hits[:pool_size]
        if h.page_key not in pre_keys and "kag" in (h.retrieval_sources or [])
    ]


def log_multimodal_retrieval_summary(
    *,
    query_text: str,
    doc_ids: List[int],
    colpali_hits: List[UnifiedPageHit],
    pgvector_hits: List[UnifiedPageHit],
    bm25_hits: List[UnifiedPageHit],
    kag_hits: Optional[List[UnifiedPageHit]] = None,
    pre_kag_fused_hits: Optional[List[UnifiedPageHit]] = None,
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
    kag_hits = kag_hits or []
    lines = [
        "══════════════════ RAG MULTIMODAL — RÉSUMÉ ══════════════════",
        f"Requête : {query_text[:100]}{'…' if len(query_text) > 100 else ''}",
        f"Documents : {len(doc_ids)} ids={doc_ids[:8]}{'…' if len(doc_ids) > 8 else ''} | pool={pool_size} top_k={top_k}",
        "",
        "── Étape 1 : Triple retriever + graphe (KAG) ──",
        f"  ColPali  : {len(colpali_hits)} page(s)",
    ]
    for hit in colpali_hits[:5]:
        lines.append(f"    • {_format_unified_hit_line(hit)}")
    if len(colpali_hits) > 5:
        lines.append(f"    … +{len(colpali_hits) - 5} autres")

    lines.append(f"  pgvector : {len(pgvector_hits)} page(s)")
    for hit in pgvector_hits[:5]:
        lines.append(f"    • {_format_unified_hit_line(hit)}")
    if len(pgvector_hits) > 5:
        lines.append(f"    … +{len(pgvector_hits) - 5} autres")

    lines.append(f"  BM25     : {len(bm25_hits)} page(s)")
    if bm25_hits:
        for hit in bm25_hits[:5]:
            lines.append(f"    • {_format_unified_hit_line(hit)}")
    else:
        lines.append("    • (aucun — voir [BM25 diagnostic] ci-dessus si 0)")

    lines.append(f"  KAG      : {len(kag_hits)} page(s) (graphe entités)")
    if kag_hits:
        for hit in kag_hits[:5]:
            lines.append(f"    • {_format_unified_hit_line(hit)}")
        if len(kag_hits) > 5:
            lines.append(f"    … +{len(kag_hits) - 5} autres")
        kag_exclusive = _kag_exclusive_hits(kag_hits, colpali_hits, pgvector_hits, bm25_hits)
        if kag_exclusive:
            lines.append(f"  ↳ Exclusives KAG (hors triple retriever) : {len(kag_exclusive)} page(s)")
            for hit in kag_exclusive[:5]:
                lines.append(f"      ★ {_format_unified_hit_line(hit)}")
            if len(kag_exclusive) > 5:
                lines.append(f"      … +{len(kag_exclusive) - 5} autres")
        if pre_kag_fused_hits is not None:
            kag_gains = _kag_fusion_gains(fused_hits, pre_kag_fused_hits, pool_size=pool_size)
            if kag_gains:
                lines.append(f"  ↳ Gains fusion RRF grâce au KAG : {len(kag_gains)} page(s)")
                for hit in kag_gains[:5]:
                    lines.append(f"      ↑ {_format_unified_hit_line(hit, show_rrf=True)}")
    else:
        lines.append("    • (aucune — entités non matchées ou KAG désactivé)")

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
        lines.extend(["", "── Étape 2b : Sélection (sans rerank) ──"])
        for hit in final_hits[:top_k]:
            lines.append(f"  #{hit.final_rank} {_format_unified_hit_line(hit, show_rrf=True)}")

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


def _merge_enrichment_source_pages(
    target: PageRetrievalHit,
    source: PageRetrievalHit,
) -> None:
    if not source.enrichment_source_pages:
        return
    merged = set(target.enrichment_source_pages)
    merged.update(source.enrichment_source_pages)
    target.enrichment_source_pages = sorted(merged)


def _aggregate_hits_by_page(hits: List[PageRetrievalHit]) -> List[PageRetrievalHit]:
    """Garde le meilleur score par (document_id, page_no)."""
    best: Dict[str, PageRetrievalHit] = {}
    for hit in hits:
        existing = best.get(hit.page_key)
        if existing is None or hit.score > existing.score:
            if existing is not None:
                _merge_enrichment_source_pages(hit, existing)
                for src in existing.retrieval_sources:
                    if src not in hit.retrieval_sources:
                        hit.retrieval_sources.append(src)
            best[hit.page_key] = hit
        else:
            _merge_enrichment_source_pages(existing, hit)
            for src in hit.retrieval_sources:
                if src not in existing.retrieval_sources:
                    existing.retrieval_sources.append(src)
    return list(best.values())


def top_hit_scores(hits: List[PageRetrievalHit], n: int = 5) -> List[float]:
    return [round(h.score, 4) for h in sorted(hits, key=lambda h: h.score, reverse=True)[:n]]


def retrieve_colpali_page_hits(
    session: Session,
    doc_ids: List[int],
    query_text: str,
    limit: int,
) -> List[PageRetrievalHit]:
    if not doc_ids or not settings.COLPALI_ENABLED:
        logger.info("[retrieve_colpali] ignoré — doc_ids=%d colpali_enabled=%s", len(doc_ids), settings.COLPALI_ENABLED)
        return []

    logger.info("[retrieve_colpali] démarrage — %d docs, limit=%d, query=%r", len(doc_ids), limit, query_text[:80])

    from app.services.colpali_service import embed_query_colpali
    from app.services.lancedb_service import search_colpali_lancedb

    query_token_embeddings = embed_query_colpali(query_text)
    if not query_token_embeddings:
        logger.info("[retrieve_colpali] aucun embedding requête")
        return []

    search_results = search_colpali_lancedb(query_token_embeddings, doc_ids, limit=limit)
    if not search_results:
        logger.info("[retrieve_colpali] aucun résultat LanceDB")
        return []

    logger.info("[retrieve_colpali] LanceDB — %d patches candidats", len(search_results))

    chunk_ids = [row["id"] for row in search_results]
    sql_chunks = text("""
        SELECT
            dc.id,
            dc.document_id,
            dc.metadata_json,
            dc.metadata_,
            d.title AS document_title
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        WHERE dc.id IN :chunk_ids
    """)
    chunk_rows = session.execute(sql_chunks, {"chunk_ids": tuple(chunk_ids)}).all()
    rows_map = {row.id: row for row in chunk_rows}

    hits: List[PageRetrievalHit] = []
    for row_lancedb in search_results:
        chunk_id = row_lancedb["id"]
        row = rows_map.get(chunk_id)
        if not row:
            continue
        distance = float(row_lancedb.get("_distance", 1.0))
        similarity = 1.0 - distance
        meta = _merged_chunk_metadata(row.metadata_json, row.metadata_)
        page_no = _page_no_from_metadata(meta)
        if page_no is None:
            continue
        hits.append(
            PageRetrievalHit(
                document_id=int(row.document_id),
                page_no=page_no,
                score=similarity,
                colpali_score=similarity,
                retrieval_sources=["colpali"],
                document_title=row.document_title or "Document sans titre",
                chunk_id=int(chunk_id),
            )
        )
    hits = _aggregate_hits_by_page(hits)
    if search_results and not hits:
        sample_ids = [row.get("id") for row in search_results[:5]]
        logger.warning(
            "[retrieve_colpali] %d résultats LanceDB mais 0 page mappée — chunk_ids échantillon=%s",
            len(search_results),
            sample_ids,
        )
    logger.info(
        "[retrieve_colpali] %d pages — top scores: %s",
        len(hits),
        top_hit_scores(hits),
    )
    return hits


def retrieve_pgvector_page_hits(
    session: Session,
    doc_ids: List[int],
    query_embedding: List[float],
    limit: int,
) -> List[PageRetrievalHit]:
    if not doc_ids or not query_embedding:
        logger.info(
            "[retrieve_pgvector] ignoré — doc_ids=%d embedding=%s",
            len(doc_ids),
            "ok" if query_embedding else "absent",
        )
        return []

    logger.info("[retrieve_pgvector] démarrage — %d docs, limit=%d", len(doc_ids), limit)

    embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
    sql = text(f"""
        SELECT
            dc.id,
            dc.document_id,
            dc.metadata_json,
            dc.metadata_,
            d.title AS document_title,
            1 - (dc.embedding <=> CAST(:query_vec AS vector)) AS similarity
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        WHERE dc.document_id IN :doc_ids
          AND dc.is_leaf = true
          AND dc.embedding IS NOT NULL
          AND {_retrievable_text_leaf_filter("dc")}
        ORDER BY dc.embedding <=> CAST(:query_vec AS vector)
        LIMIT :limit
    """)
    rows = session.execute(
        sql,
        {"doc_ids": tuple(doc_ids), "query_vec": embedding_str, "limit": limit},
    ).all()

    hits: List[PageRetrievalHit] = []
    for row in rows:
        meta = _merged_chunk_metadata(row.metadata_json, row.metadata_)
        page_no = _page_no_from_metadata(meta)
        if page_no is None:
            continue
        sim = float(row.similarity or 0.0)
        hits.append(
            PageRetrievalHit(
                document_id=int(row.document_id),
                page_no=page_no,
                score=sim,
                pgvector_score=sim,
                retrieval_sources=["pgvector"],
                document_title=row.document_title or "Document sans titre",
                chunk_id=int(row.id),
                enrichment_source_pages=_enrichment_pages_from_meta(meta),
            )
        )
    hits = _aggregate_hits_by_page(hits)
    logger.info(
        "[retrieve_pgvector] %d pages — top scores: %s",
        len(hits),
        top_hit_scores(hits),
    )
    return hits


def retrieve_bm25_page_hits(
    session: Session,
    doc_ids: List[int],
    query_text: str,
    limit: int,
) -> List[PageRetrievalHit]:
    if not doc_ids or not query_text.strip():
        logger.info("[retrieve_bm25] ignoré — doc_ids=%d query vide=%s", len(doc_ids), not query_text.strip())
        return []

    logger.info("[retrieve_bm25] démarrage — %d docs, limit=%d, query=%r", len(doc_ids), limit, query_text[:80])

    tsquery_fn = _bm25_tsquery_fn()
    semantic_filter = f"AND {_retrievable_text_leaf_filter('dc')}" if settings.BM25_FILTER_SEMANTIC_LEAF else ""
    sql = text(f"""
        SELECT
            dc.id,
            dc.document_id,
            dc.metadata_json,
            dc.metadata_,
            d.title AS document_title,
            ts_rank_cd(dc.tsv_content, {tsquery_fn}('french', :query)) AS rank
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        WHERE dc.document_id IN :doc_ids
          AND dc.is_leaf = true
          AND dc.tsv_content IS NOT NULL
          AND dc.tsv_content @@ {tsquery_fn}('french', :query)
          {semantic_filter}
        ORDER BY rank DESC
        LIMIT :limit
    """)
    try:
        rows = session.execute(
            sql,
            {"doc_ids": tuple(doc_ids), "query": query_text.strip(), "limit": limit},
        ).all()
    except Exception as exc:
        logger.warning("[BM25] recherche échouée (%s) : %s", tsquery_fn, exc)
        return []

    hits: List[PageRetrievalHit] = []
    dropped_no_page = 0
    for row in rows:
        meta = _merged_chunk_metadata(row.metadata_json, row.metadata_)
        page_no = _page_no_from_metadata(meta)
        if page_no is None:
            dropped_no_page += 1
            continue
        rank = float(row.rank or 0.0)
        hits.append(
            PageRetrievalHit(
                document_id=int(row.document_id),
                page_no=page_no,
                score=rank,
                bm25_score=rank,
                retrieval_sources=["bm25"],
                document_title=row.document_title or "Document sans titre",
                chunk_id=int(row.id),
                enrichment_source_pages=_enrichment_pages_from_meta(meta),
            )
        )
    hits = _aggregate_hits_by_page(hits)
    if dropped_no_page:
        logger.warning("[retrieve_bm25] %d chunks SQL sans page_no filtrés", dropped_no_page)
    logger.info(
        "[retrieve_bm25] %d pages — top scores: %s",
        len(hits),
        top_hit_scores(hits),
    )
    return hits


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


def filter_colpali_page_hits_dynamic(
    hits: List[PageRetrievalHit],
    *,
    min_threshold: Optional[float] = None,
    relative_margin: Optional[float] = None,
) -> List[PageRetrievalHit]:
    """Variante PageRetrievalHit pour le chemin hybrid legacy."""
    min_threshold = min_threshold if min_threshold is not None else settings.COLPALI_MIN_THRESHOLD
    relative_margin = relative_margin if relative_margin is not None else settings.COLPALI_RELATIVE_MARGIN
    if not hits:
        return []

    def _score(h: PageRetrievalHit) -> float:
        return h.colpali_score if h.colpali_score is not None else h.score

    above_abs = [h for h in hits if _score(h) >= min_threshold]
    if not above_abs:
        return []

    max_score = max(_score(h) for h in above_abs)
    cutoff = max_score - relative_margin
    return [h for h in above_abs if _score(h) >= cutoff]


def unified_hits_to_eval_passages(hits: List[UnifiedPageHit], top_k: int) -> List[Dict[str, Any]]:
    """Passages légers pour métriques d'éval (sans expansion L1 / PNG)."""
    passages: List[Dict[str, Any]] = []
    for rank, hit in enumerate(hits[:top_k], start=1):
        if hit.rerank_score is not None:
            score = hit.rerank_score
        elif hit.rrf_score:
            score = hit.rrf_score
        else:
            score = (
                hit.colpali_score
                or hit.pgvector_score
                or hit.bm25_score
                or hit.kag_score
                or 0.0
            )
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
                "pgvector_score": hit.pgvector_score,
                "bm25_score": hit.bm25_score,
                "kag_score": hit.kag_score,
                "rerank_score": hit.rerank_score,
            }
        )
    return passages


def page_hits_to_eval_passages(hits: List[PageRetrievalHit], top_k: int) -> List[Dict[str, Any]]:
    """Passages légers pour éval — chemin hybrid legacy."""
    passages: List[Dict[str, Any]] = []
    for rank, hit in enumerate(hits[:top_k], start=1):
        passages.append(
            {
                "rank": rank,
                "document_title": hit.document_title,
                "document_id": hit.document_id,
                "page_no": hit.page_no,
                "page_start": hit.page_no,
                "page_end": hit.page_no,
                "score": float(hit.rrf_score or hit.score or 0.0),
                "retrieval_sources": list(hit.retrieval_sources),
                "colpali_score": hit.colpali_score,
                "pgvector_score": hit.pgvector_score,
                "bm25_score": hit.bm25_score,
            }
        )
    return passages


def retrieve_pgvector_pages(
    session: Session,
    doc_ids: List[int],
    query_embedding: List[float],
    limit: int,
) -> List[UnifiedPageHit]:
    """Retrieval pgvector — agrégation SQL par page."""
    if not doc_ids or not query_embedding:
        return []

    embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
    page_no_expr = _page_no_sql_expr("dc")
    sql = text(f"""
        SELECT
            dc.document_id,
            {page_no_expr} AS page_no,
            d.title AS document_title,
            MAX(1 - (dc.embedding <=> CAST(:query_vec AS vector))) AS similarity,
            {_enrichment_source_pages_agg("dc")} AS enrichment_source_pages_text
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        WHERE dc.document_id IN :doc_ids
          AND dc.is_leaf = true
          AND dc.embedding IS NOT NULL
          AND {_retrievable_text_leaf_filter("dc")}
          AND {page_no_expr} IS NOT NULL
        GROUP BY dc.document_id, {page_no_expr}, d.title
        ORDER BY similarity DESC
        LIMIT :limit
    """)
    rows = session.execute(
        sql,
        {"doc_ids": tuple(doc_ids), "query_vec": embedding_str, "limit": limit},
    ).all()

    hits = [
        UnifiedPageHit(
            document_id=int(row.document_id),
            page_no=int(row.page_no),
            pgvector_score=float(row.similarity or 0.0),
            document_title=row.document_title or "Document sans titre",
            retrieval_sources=["pgvector"],
            enrichment_source_pages=_parse_source_pages(
                getattr(row, "enrichment_source_pages_text", None)
            ),
        )
        for row in rows
    ]
    if hits:
        logger.info(
            "[retrieve_pgvector_pages] %d pages — meilleure: %s",
            len(hits),
            _format_unified_hit_line(hits[0]),
        )
    else:
        logger.info("[retrieve_pgvector_pages] 0 pages")
    return hits


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
    pgvector_hits: List[UnifiedPageHit],
    bm25_hits: List[UnifiedPageHit],
    *,
    kag_hits: Optional[List[UnifiedPageHit]] = None,
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
    add_channel(pgvector_hits, "pgvector")
    add_channel(bm25_hits, "bm25")
    if kag_hits:
        add_channel(kag_hits, "kag")

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
            "pgvector_score": hit.pgvector_score,
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


def fuse_page_hits_rrf(
    colpali_hits: List[PageRetrievalHit],
    pgvector_hits: List[PageRetrievalHit],
    bm25_hits: List[PageRetrievalHit],
    *,
    rrf_k: int = 60,
    top_n: int = 15,
) -> List[PageRetrievalHit]:
    merged: Dict[str, PageRetrievalHit] = {}

    def _add_channel(hits: List[PageRetrievalHit], channel: str) -> None:
        sorted_hits = sorted(hits, key=lambda h: h.score, reverse=True)
        for rank_idx, hit in enumerate(sorted_hits, start=1):
            key = hit.page_key
            if key not in merged:
                merged[key] = PageRetrievalHit(
                    document_id=hit.document_id,
                    page_no=hit.page_no,
                    document_title=hit.document_title,
                    chunk_id=hit.chunk_id,
                    retrieval_sources=[],
                )
            target = merged[key]
            target.rrf_score += 1.0 / (rrf_k + rank_idx)
            if channel not in target.retrieval_sources:
                target.retrieval_sources.append(channel)
            if channel == "colpali":
                target.colpali_score = hit.colpali_score
                target.score = max(target.score, hit.score)
            elif channel == "pgvector":
                target.pgvector_score = hit.pgvector_score
                target.score = max(target.score, hit.score)
            elif channel == "bm25":
                target.bm25_score = hit.bm25_score
                target.score = max(target.score, hit.score)
            if hit.document_title:
                target.document_title = hit.document_title
            if hit.chunk_id is not None:
                target.chunk_id = hit.chunk_id
            _merge_enrichment_source_pages(target, hit)

    _add_channel(colpali_hits, "colpali")
    _add_channel(pgvector_hits, "pgvector")
    _add_channel(bm25_hits, "bm25")

    if not merged:
        return []

    result = sorted(merged.values(), key=lambda h: h.rrf_score, reverse=True)
    return result[:top_n]


def build_weak_hit_pool(
    colpali_hits: List[PageRetrievalHit],
    pgvector_hits: List[PageRetrievalHit],
    bm25_hits: List[PageRetrievalHit],
) -> Dict[str, Dict[str, Any]]:
    """Pool élargi pour détecter les weak hits voisins."""
    pool: Dict[str, Dict[str, Any]] = {}
    for hit in colpali_hits + pgvector_hits + bm25_hits:
        key = hit.page_key
        entry = pool.setdefault(
            key,
            {
                "document_id": hit.document_id,
                "page_no": hit.page_no,
                "max_score": 0.0,
                "sources": set(),
            },
        )
        entry["max_score"] = max(entry["max_score"], hit.score)
        entry["sources"].update(hit.retrieval_sources)
    return pool


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


def expand_neighbor_pages(
    session: Session,
    hit: PageRetrievalHit,
    weak_pool: Dict[str, Dict[str, Any]],
) -> Tuple[List[int], List[int], Optional[str]]:
    """
    Détermine quelles pages inclure (page principale + voisines).
    Retourne (pages_incluses, expanded_neighbor_pages, expansion_reason).
    """
    doc_id = hit.document_id
    page_no = hit.page_no
    pages: Set[int] = {page_no}
    expanded_neighbors: Set[int] = set()
    reason: Optional[str] = None

    # Signal 0 : dépliage des pages sources d'un chunk d'enrichissement contextuel
    # retrouvé sur cette page (retourne tout le batch 1/2/3 pages).
    for pno in hit.enrichment_source_pages:
        if pno and pno != page_no:
            pages.add(pno)
            expanded_neighbors.add(pno)
            reason = reason or "enrichment_span"

    if _has_cross_page_coverage(session, doc_id, page_no):
        expanded_neighbors.discard(page_no)
        return sorted(pages), sorted(expanded_neighbors), reason

    l1_chunks = load_l1_chunks_for_page(session, doc_id, page_no)

    # Signal 1 : continues_on_next_page sur le dernier chunk
    if l1_chunks:
        last_meta = _merged_chunk_metadata(l1_chunks[-1].metadata_json, l1_chunks[-1].metadata_)
        if last_meta.get("continues_on_next_page"):
            pages.add(page_no + 1)
            expanded_neighbors.add(page_no + 1)
            reason = reason or "continues_on_next_page"

    # Signal 2 : continues_from_previous_page sur N+1
    next_chunks = load_l1_chunks_for_page(session, doc_id, page_no + 1)
    if next_chunks:
        first_meta = _merged_chunk_metadata(next_chunks[0].metadata_json, next_chunks[0].metadata_)
        if first_meta.get("continues_from_previous_page"):
            pages.add(page_no + 1)
            expanded_neighbors.add(page_no + 1)
            reason = reason or "continues_from_previous_page"

    # Signal 3 : weak retrieval hit dans le pool
    for neighbor in (page_no - 1, page_no + 1):
        if neighbor < 1:
            continue
        nkey = f"{doc_id}:{neighbor}"
        if nkey in weak_pool and neighbor not in pages:
            pages.add(neighbor)
            expanded_neighbors.add(neighbor)
            reason = reason or "weak_hit"

    # Signal 4 : radius config (filet de sécurité)
    if settings.RETRIEVAL_EXPAND_ENABLED and settings.RETRIEVAL_PAGE_RADIUS >= 1:
        ratio = settings.RETRIEVAL_NEIGHBOR_MIN_SCORE_RATIO
        for neighbor in (page_no - 1, page_no + 1):
            if neighbor < 1:
                continue
            nkey = f"{doc_id}:{neighbor}"
            pool_entry = weak_pool.get(nkey)
            if pool_entry and neighbor not in pages:
                if pool_entry["max_score"] >= hit.score * ratio:
                    pages.add(neighbor)
                    expanded_neighbors.add(neighbor)
                    reason = reason or "radius"

    expanded_neighbors.discard(page_no)
    return sorted(pages), sorted(expanded_neighbors), reason


def compute_needs_page_image(
    retrieval_sources: List[str],
    expanded_pages_sources: Optional[Dict[int, Set[str]]] = None,
) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    PNG requis si ColPali seul (sans pgvector ni bm25) sur une page.
    Retourne (needs_any, list of (doc_id, page_no) needing PNG) — doc_id filled by caller.
    """
    sources_set = set(retrieval_sources or [])
    text_sources = sources_set & {"pgvector", "bm25"}
    if "colpali" in sources_set and not text_sources:
        return True, []
    return False, []


def compute_image_pages_for_passage(
    document_id: int,
    primary_sources: List[str],
    pages_included: List[int],
    weak_pool: Dict[str, Dict[str, Any]],
) -> List[Tuple[int, int]]:
    """Liste les (doc_id, page_no) nécessitant un PNG pour la génération."""
    image_pages: List[Tuple[int, int]] = []
    for pno in pages_included:
        pkey = f"{document_id}:{pno}"
        pool_entry = weak_pool.get(pkey, {})
        sources = set(pool_entry.get("sources") or [])
        if pno == pages_included[0]:
            sources.update(primary_sources)
        text_hit = sources & {"pgvector", "bm25"}
        if "colpali" in sources and not text_hit:
            image_pages.append((document_id, pno))
    return image_pages


def format_hybrid_passages(
    session: Session,
    fused_hits: List[PageRetrievalHit],
    weak_pool: Dict[str, Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    """Construit les passages finaux avec L1 consolidés, expansion voisine et flags image."""
    passages: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()

    for hit in fused_hits:
        if len(passages) >= k:
            break

        pages_included, expanded_neighbors, expansion_reason = expand_neighbor_pages(
            session, hit, weak_pool
        )
        passage_key = f"{hit.document_id}:{pages_included[0]}-{pages_included[-1]}"
        if passage_key in seen_keys:
            continue
        seen_keys.add(passage_key)

        all_chunks: List[DocumentChunk] = []
        for pno in pages_included:
            all_chunks.extend(load_l1_chunks_for_page(session, hit.document_id, pno))
        # Dédupliquer par chunk id
        seen_chunk_ids: Set[int] = set()
        unique_chunks: List[DocumentChunk] = []
        for c in all_chunks:
            if c.id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(c.id)
            unique_chunks.append(c)
        unique_chunks.sort(key=lambda c: (c.chunk_index or 0, c.id or 0))

        consolidated = build_consolidated_page_text(unique_chunks)
        if not consolidated:
            consolidated = f"Page {hit.page_no} — contenu visuel uniquement"

        page_start = min(pages_included)
        page_end = max(pages_included)
        doc_title = hit.document_title or "Document sans titre"
        passage_text = f"**{doc_title}**\n{consolidated}"

        image_pages = compute_image_pages_for_passage(
            hit.document_id,
            hit.retrieval_sources,
            pages_included,
            weak_pool,
        )
        needs_page_image = len(image_pages) > 0
        enrichment_chunks = load_enrichment_chunks_for_pages(
            session, hit.document_id, pages_included
        )
        enrichment_passages = _format_enrichment_passages(enrichment_chunks)

        out: Dict[str, Any] = {
            "passage": passage_text,
            "passage_raw": consolidated,
            "document_title": doc_title,
            "document_id": hit.document_id,
            "chunk_id": hit.chunk_id,
            "score": float(hit.rrf_score or hit.score),
            # page_no = page ancre matchée (cf. format_multimodal_passages) — pas min du span.
            "page_no": hit.page_no,
            "page_start": page_start,
            "page_end": page_end,
            "retrieval_sources": list(hit.retrieval_sources),
            "colpali_score": hit.colpali_score,
            "pgvector_score": hit.pgvector_score,
            "bm25_score": hit.bm25_score,
            "raw_rrf_score": hit.rrf_score,
            "expanded_neighbor_pages": expanded_neighbors,
            "expansion_reason": expansion_reason,
            "needs_page_image": needs_page_image,
            "image_pages": image_pages,
            "content_type": "hybrid_page_passage",
            "is_enrichment": False,
            "enrichment_passages": enrichment_passages,
        }
        passages.append(out)

    return passages


def _retrieve_leaves_bm25_sql(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    candidate_k: int,
    query_embedding: Optional[List[float]] = None,
    document_filter: str = "all",
) -> List[Any]:
    """Stub de compatibilité tests — délègue à retrieve_bm25_page_hits."""
    doc_ids = get_space_document_ids(session, space_id, document_filter)
    hits = retrieve_bm25_page_hits(session, doc_ids, query_text, candidate_k)
    from llama_index.core.schema import NodeWithScore, TextNode

    nodes: List[NodeWithScore] = []
    for hit in hits:
        meta = {
            "document_id": hit.document_id,
            "page_no": hit.page_no,
            "document_title": hit.document_title,
            "content_type": CONTENT_TYPE_SEMANTIC_LEAF,
            "retrieval_sources": hit.retrieval_sources,
        }
        node = TextNode(
            id_=f"chunk-{hit.chunk_id or hit.page_key}",
            text="",
            metadata=meta,
        )
        nodes.append(NodeWithScore(node=node, score=hit.score))
    return nodes
