"""
Recherche dans les espaces : chunks feuilles via pgvector + tsvector + RRF + rerank + MMR.

Pipeline RAG :
1. Retrieval hybride (pgvector dense + tsvector lexical)
2. Fusion RRF (Reciprocal Rank Fusion)
3. Résolution parents (contexte hiérarchique)
4. Early stopping (court-circuite rerank si scores déjà excellents)
5. Reranker cross-encoder CPU (ms-marco-MiniLM-L-6-v2)
6. Guardrails statistiques (détection bégaiement → clarification)
7. MMR (diversification pour éviter redondance)
"""

from __future__ import annotations

import logging
import math
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from llama_index.core.schema import NodeWithScore, TextNode
from sqlalchemy import or_, text
from sqlmodel import Session, select

from app.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.services.embedding_service import generate_embedding
from app.services.space_service import get_space_by_id
from app.tracing import trace_run
from app.services import reranker_service

logger = logging.getLogger(__name__)

MIN_VECTOR_SIMILARITY_THRESHOLD = float(os.getenv("MIN_VECTOR_SIMILARITY", "0.25"))
TITLE_QUERY_BOOST_PER_MATCH = float(os.getenv("TITLE_QUERY_BOOST_PER_MATCH", "0.5"))
TITLE_QUERY_BOOST_CAP = float(os.getenv("TITLE_QUERY_BOOST_CAP", "2.0"))

_FALLBACK_STOPWORDS = {
    # English
    "the", "and", "for", "with", "this", "that", "what", "how",
    "from", "you", "your", "not", "are", "was", "were", "have", "has",
    "will", "can", "could", "would", "should", "been", "being", "about",
    # French
    "dans", "avec", "pour", "une", "des", "les", "est", "sur", "pas",
    "plus", "que", "qui", "quoi", "comment", "quel", "quelle", "quels",
    "quelles", "par", "sans", "mais", "donc", "car", "son", "ses",
    "notre", "nos", "votre", "vos", "leur", "leurs", "tout", "tous",
    "toute", "toutes", "autre", "autres", "même", "aussi", "très",
    "bien", "encore", "ici", "entre", "après", "avant", "sous",
    "chez", "vers", "depuis", "pendant", "comme",
}

def _merged_chunk_metadata(primary: Optional[dict], legacy: Optional[dict]) -> Dict:
    merged: Dict = {}
    if isinstance(legacy, dict):
        merged.update(legacy)
    if isinstance(primary, dict):
        merged.update(primary)
    return merged


def _retrieve_leaves_sql(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    candidate_k: int,
    query_embedding: Optional[List[float]] = None,
    document_filter: str = "all",
) -> List[NodeWithScore]:
    """
    Recherche vectorielle pgvector sur les feuilles.
    
    Args:
        query_embedding: Embedding pré-calculé (évite un appel API si déjà disponible)
        document_filter: "all" (tous), "technical" (exclut FAQ), "faq_corrective" (FAQ uniquement)
    """
    if query_embedding is None:
        query_embedding = generate_embedding(query_text)
    if not query_embedding:
        logger.warning("Embedding requête vide pour space_id=%s", space_id)
        return []
    query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    
    from app.services.document_service_new import feedback_corrective_sql_filter

    filter_clause = feedback_corrective_sql_filter(document_filter, "d")

    sql_query = text(f"""
        SELECT
            dc.id,
            dc.content,
            dc.text,
            dc.chunk_index,
            dc.document_id,
            dc.metadata_json,
            dc.metadata_,
            dc.source AS chunk_source,
            d.title AS document_title,
            d.source AS document_source,
            d.id AS document_id,
            1 - (dc.embedding <=> '{query_embedding_str}'::vector) AS similarity_score
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        INNER JOIN document_space ds ON ds.document_id = d.id
        WHERE ds.space_id = :space_id
          AND dc.embedding IS NOT NULL
          AND dc.is_leaf = true
          {filter_clause}
        ORDER BY dc.embedding <=> '{query_embedding_str}'::vector
        LIMIT :limit_k
    """)

    result = session.execute(sql_query, {"space_id": space_id, "limit_k": candidate_k})
    nodes: List[NodeWithScore] = []
    for row in result:
        metadata = _merged_chunk_metadata(row.metadata_json, row.metadata_)
        metadata.setdefault("document_id", row.document_id)
        metadata.setdefault("document_title", row.document_title or "Document sans titre")
        metadata.setdefault("chunk_index", row.chunk_index)
        if getattr(row, "document_source", None):
            metadata.setdefault("source", row.document_source)
        if getattr(row, "chunk_source", None):
            metadata.setdefault("source", row.chunk_source)
        node = TextNode(
            id_=f"chunk-{row.id}",
            text=row.content or row.text or "",
            metadata=metadata,
        )
        nodes.append(NodeWithScore(node=node, score=float(row.similarity_score)))
    logger.info("Vector pgvector (space): %d feuilles (limit=%d)", len(nodes), candidate_k)
    return nodes


def _retrieve_leaves_bm25_sql(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    candidate_k: int,
    document_filter: str = "all",
) -> List[NodeWithScore]:
    """Recherche lexicale BM25 native via tsvector sur les feuilles.

    Args:
        document_filter: "all" (tous), "technical" (exclut FAQ), "faq_corrective" (FAQ uniquement)
    """
    terms = _extract_query_terms(query_text)
    if not terms:
        logger.info("BM25 lexical (space): Aucun terme significatif extrait de la requête.")
        return []

    # Construction d'une requête OR pour websearch_to_tsquery (ex: "vitrage OR soleal")
    or_query = " OR ".join(terms)
    
    from app.services.document_service_new import feedback_corrective_sql_filter

    filter_clause = feedback_corrective_sql_filter(document_filter, "d")

    # Retrieval avec ts_rank_cd + flag 33 (1|32 = normalisation par log de la longueur + division par longueur doc + 1)
    sql_query = text(f"""
        SELECT
            dc.id,
            dc.content,
            dc.text,
            dc.chunk_index,
            dc.document_id,
            dc.metadata_json,
            dc.metadata_,
            dc.source AS chunk_source,
            d.title AS document_title,
            d.source AS document_source,
            d.id AS document_id,
            ts_rank_cd(dc.tsv_content, websearch_to_tsquery('french', :query), 33) AS similarity_score
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        INNER JOIN document_space ds ON ds.document_id = d.id
        WHERE ds.space_id = :space_id
          AND dc.is_leaf = true
          AND dc.tsv_content @@ websearch_to_tsquery('french', :query)
          {filter_clause}
        ORDER BY similarity_score DESC
        LIMIT :limit_k
    """)

    result = session.execute(
        sql_query,
        {"space_id": space_id, "query": or_query, "limit_k": candidate_k}
    )
    nodes: List[NodeWithScore] = []
    for row in result:
        metadata = _merged_chunk_metadata(row.metadata_json, row.metadata_)
        metadata.setdefault("document_id", row.document_id)
        metadata.setdefault("document_title", row.document_title or "Document sans titre")
        metadata.setdefault("chunk_index", row.chunk_index)
        if getattr(row, "document_source", None):
            metadata.setdefault("source", row.document_source)
        if getattr(row, "chunk_source", None):
            metadata.setdefault("source", row.chunk_source)
        node = TextNode(
            id_=f"chunk-{row.id}",
            text=row.content or row.text or "",
            metadata=metadata,
        )
        nodes.append(NodeWithScore(node=node, score=float(row.similarity_score)))

    logger.info("BM25 lexical (space): %d feuilles (limit=%d, terms=%s)", len(nodes), candidate_k, terms)
    return nodes


def _extract_alphanumeric_codes(query_text: str) -> List[str]:
    """
    Extrait les codes, normes, modèles et références de la requête pour recherche exacte/substring.
    """
    if not query_text:
        return []
    
    # 1. Mots contenant des chiffres (ex: 36.5, perform-70, 1991, v4)
    with_digits = re.findall(r"\b[a-zA-Z0-9\.\-]*\d+[a-zA-Z0-9\.\-]*\b", query_text)
    
    # 2. Acronymes tout en majuscules (ex: DTU, NF, EN, ISO, PVC, RAG)
    acronyms = re.findall(r"\b[A-Z]{2,}\b", query_text)
    
    # 3. Noms propres / modèles capitalisés (ex: Soleal, Perform, Lumeal)
    capitalized = re.findall(r"\b[A-Z][a-z]{2,}\b", query_text)
    
    results = []
    seen = set()
    for t in with_digits + acronyms + capitalized:
        cleaned = t.strip(".-").lower()
        if len(cleaned) >= 2 and cleaned not in seen:
            seen.add(cleaned)
            results.append(cleaned)
            
    return results


def _retrieve_leaves_alphanumeric_sql(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    candidate_k: int,
    document_filter: str = "all",
) -> List[NodeWithScore]:
    """
    Recherche directe par correspondance de sous-chaîne pour les codes et références alphanumériques
    afin de fiabiliser les réponses sur des termes techniques complexes (normes, gammes, etc.).
    """
    all_terms = _extract_alphanumeric_codes(query_text)
    if not all_terms:
        return []

    from app.services.document_service_new import feedback_corrective_sql_filter
    filter_clause = feedback_corrective_sql_filter(document_filter, "d")

    or_clauses = []
    params = {"space_id": space_id, "limit_k": candidate_k}
    for idx, term in enumerate(all_terms):
        param_name = f"term_{idx}"
        or_clauses.append(f"dc.content ILIKE :{param_name}")
        params[param_name] = f"%{term}%"

    or_clause_str = " OR ".join(or_clauses)

    sql_query = text(f"""
        SELECT
            dc.id,
            dc.content,
            dc.text,
            dc.chunk_index,
            dc.document_id,
            dc.metadata_json,
            dc.metadata_,
            dc.source AS chunk_source,
            d.title AS document_title,
            d.source AS document_source,
            d.id AS document_id
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        INNER JOIN document_space ds ON ds.document_id = d.id
        WHERE ds.space_id = :space_id
          AND dc.is_leaf = true
          AND ({or_clause_str})
          {filter_clause}
        LIMIT :limit_k
    """)

    result = session.execute(sql_query, params)
    nodes: List[NodeWithScore] = []
    
    for row in result:
        content_lower = (row.content or row.text or "").lower()
        
        matches = sum(1 for t in all_terms if t in content_lower)
        if matches == 0:
            continue
            
        base_score = matches / len(all_terms)
        
        metadata = _merged_chunk_metadata(row.metadata_json, row.metadata_)
        metadata.setdefault("document_id", row.document_id)
        metadata.setdefault("document_title", row.document_title or "Document sans titre")
        metadata.setdefault("chunk_index", row.chunk_index)
        if getattr(row, "document_source", None):
            metadata.setdefault("source", row.document_source)
        if getattr(row, "chunk_source", None):
            metadata.setdefault("source", row.chunk_source)
            
        node = TextNode(
            id_=f"chunk-{row.id}",
            text=row.content or row.text or "",
            metadata=metadata,
        )
        nodes.append(NodeWithScore(node=node, score=base_score))
        
    nodes.sort(key=lambda n: n.score, reverse=True)
    logger.info("Alphanumeric substring (space): %d feuilles (terms=%s)", len(nodes), all_terms)
    return nodes


def reciprocal_rank_fusion(
    vector_results: List[NodeWithScore],
    lexical_results: List[NodeWithScore],
    alphanumeric_results: Optional[List[NodeWithScore]] = None,
    k: int = 60,
    top_n: int = 15,
    normalize: bool = False,
) -> List[NodeWithScore]:
    """
    Fusionne les résultats de recherche vectorielle, lexicale et alphanumérique avec l'algorithme RRF.
    """
    rrf_scores: Dict[str, float] = {}
    nodes_map: Dict[str, NodeWithScore] = {}

    for rank, nws in enumerate(vector_results, start=1):
        node_id = nws.node.id_
        nodes_map[node_id] = nws
        rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + (1.0 / (k + rank))

    for rank, nws in enumerate(lexical_results, start=1):
        node_id = nws.node.id_
        if node_id not in nodes_map:
            nodes_map[node_id] = nws
        rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + (1.0 / (k + rank))

    for rank, nws in enumerate(alphanumeric_results or [], start=1):
        node_id = nws.node.id_
        if node_id not in nodes_map:
            nodes_map[node_id] = nws
        rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + (1.0 / (k + rank))

    # Trier par score RRF décroissant
    sorted_node_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results: List[NodeWithScore] = []
    for node_id, rrf_score in sorted_node_ids:
        node = nodes_map[node_id].node
        node.metadata["raw_rrf_score"] = rrf_score
        results.append(NodeWithScore(node=node, score=rrf_score))

    # Normalisation linéaire dans [0.1, 0.9] si demandée
    if normalize and results:
        scores = [nws.score for nws in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        for nws in results:
            if score_range > 0:
                nws.score = 0.1 + 0.8 * ((nws.score - min_score) / score_range)
            else:
                nws.score = 0.5

    return results


def _parse_chunk_id_from_node(node: TextNode) -> Optional[int]:
    nid = getattr(node, "id_", None) or ""
    if isinstance(nid, str) and nid.startswith("chunk-"):
        try:
            return int(nid.split("-", 1)[1])
        except ValueError:
            return None
    return None


def _build_parent_node_dict(
    session: Session, space_id: int, user_id: int
) -> Dict[str, TextNode]:
    statement = (
        select(DocumentChunk, Document.title)
        .join(Document, Document.id == DocumentChunk.document_id)
        .join(DocumentSpace, DocumentSpace.document_id == Document.id)
        .where(
            DocumentSpace.space_id == space_id,
            DocumentChunk.is_leaf.is_(False),
        )
    )
    rows = session.exec(statement).all()
    node_dict: Dict[str, TextNode] = {}
    for chunk, document_title in rows:
        metadata = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        metadata.setdefault("document_id", chunk.document_id)
        metadata.setdefault("document_title", document_title or "Document sans titre")
        metadata.setdefault("chunk_index", chunk.chunk_index)
        llama_id = f"chunk-{chunk.id}"
        node = TextNode(
            id_=llama_id,
            text=chunk.content or chunk.text or "",
            metadata=metadata,
        )
        if chunk.node_id:
            node_dict[str(chunk.node_id)] = node
        node_dict[llama_id] = node
    logger.info("Parents chargés (space): %d", len(rows))
    return node_dict


_PARENT_MULTIHOP_MAX = 4


def _resolve_space_parent_with_multihop(
    session: Session,
    space_id: int,
    user_id: int,
    document_id: Optional[int],
    parent_node_id: Optional[str],
    parent_node_dict: Dict[str, TextNode],
) -> Optional[TextNode]:
    if not parent_node_id or document_id is None:
        return None
    if parent_node_id in parent_node_dict:
        return None

    intermediates: List[str] = []
    current_pid: Optional[str] = parent_node_id
    hops = 0

    while current_pid and hops < _PARENT_MULTIHOP_MAX:
        hops += 1
        stmt = (
            select(DocumentChunk, Document.title)
            .join(Document, Document.id == DocumentChunk.document_id)
            .join(DocumentSpace, DocumentSpace.document_id == Document.id)
            .where(
                DocumentSpace.space_id == space_id,
                DocumentChunk.document_id == document_id,
                Document.user_id == user_id,
                DocumentChunk.node_id == current_pid,
            )
        )
        row = session.exec(stmt).first()
        if not row:
            break
        chunk, document_title = row
        metadata = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        metadata.setdefault("document_id", chunk.document_id)
        metadata.setdefault("document_title", document_title or "Document sans titre")
        metadata.setdefault("chunk_index", chunk.chunk_index)
        llama_id = f"chunk-{chunk.id}"
        text_body = (chunk.content or chunk.text or "").strip()

        if not chunk.is_leaf:
            if intermediates:
                prefix = "\n\n---\n\n".join(reversed(intermediates))
                text_body = f"{prefix}\n\n---\n\n{text_body}"
            return TextNode(id_=llama_id, text=text_body, metadata=metadata)

        if text_body:
            intermediates.append(text_body)
        current_pid = chunk.parent_node_id

    return None


def _extract_query_terms(query_text: str) -> List[str]:
    """Extrait les termes significatifs de la requête pour la recherche lexicale.

    Améliorations v2 :
    - Conserve les termes courts (≥ 2 chars) pour les codes produits (76, PF, PVC)
    - Limite relevée à BM25_MAX_QUERY_TERMS (15 par défaut)
    - Détecte les termes composés avec tirets/points (DTU-36.5, Perform-70)
    """
    if not query_text or not query_text.strip():
        return []
    # Extraire tokens alphanumériques + composés avec tirets/points
    raw_tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]+(?:[.\-][A-Za-zÀ-ÿ0-9]+)*", query_text)
    terms = []
    for t in raw_tokens:
        low = t.lower()
        if low in _FALLBACK_STOPWORDS:
            continue
        if len(low) < 2:
            continue
        terms.append(low)
    max_terms = settings.BM25_MAX_QUERY_TERMS
    return terms[:max_terms]


async def _keyword_fallback_passages(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    k: int,
) -> List[Dict]:
    terms = _extract_query_terms(query_text)
    if not terms:
        # Pas de termes significatifs
        return []
    
    or_query = " OR ".join(terms)
    
    # Essayer le tsvector pour une recherche rapide et pertinente
    try:
        sql_query = text("""
            SELECT
                dc.id,
                dc.content,
                dc.text,
                dc.chunk_index,
                dc.document_id,
                dc.metadata_json,
                dc.metadata_,
                d.title AS document_title,
                d.source AS document_source,
                d.id AS document_id,
                ts_rank_cd(dc.tsv_content, websearch_to_tsquery('french', :query), 1) AS score
            FROM documentchunk dc
            INNER JOIN document d ON dc.document_id = d.id
            INNER JOIN document_space ds ON ds.document_id = d.id
            WHERE ds.space_id = :space_id
              AND dc.is_leaf = true
              AND dc.tsv_content @@ websearch_to_tsquery('french', :query)
            ORDER BY score DESC
            LIMIT :limit_k
        """)
        result = session.execute(sql_query, {"space_id": space_id, "query": or_query, "limit_k": k})
        passages: List[Dict] = []
        for row in result:
            meta = _merged_chunk_metadata(row.metadata_json, row.metadata_)
            meta.setdefault("document_id", row.document_id)
            meta.setdefault("document_title", row.document_title or "Document sans titre")
            meta.setdefault("chunk_index", row.chunk_index)
            if getattr(row, "document_source", None):
                meta.setdefault("source", row.document_source)
            node = TextNode(id_=f"chunk-{row.id}", text=row.content or row.text or "", metadata=meta)
            passages.append(_node_to_passage(node, fallback_score=float(row.score or 0.05)))
        if passages:
            logger.info("Fallback tsvector (space): %d passages", len(passages))
            return passages
    except Exception as e:
        logger.warning("Le fallback tsvector a échoué (migration non appliquée ?), retour au mode ILIKE : %s", e)

    # Mode dégradé d'origine (ILIKE)
    base_stmt = (
        select(DocumentChunk, Document.title)
        .join(Document, Document.id == DocumentChunk.document_id)
        .join(DocumentSpace, DocumentSpace.document_id == Document.id)
        .where(DocumentSpace.space_id == space_id)
        .order_by(DocumentChunk.is_leaf.desc(), Document.updated_at.desc(), DocumentChunk.chunk_index)
    )
    rows = []
    if terms:
        stmt = base_stmt.where(
            or_(*[DocumentChunk.content.ilike(f"%{term}%") for term in terms])
        ).limit(max(k * 4, 12))
        rows = session.exec(stmt).all()
    if not rows:
        rows = session.exec(base_stmt.limit(max(k * 2, 8))).all()

    passages = []
    seen: set = set()
    for chunk, document_title in rows:
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        content = (chunk.content or chunk.text or "").strip()
        if not content:
            continue
        lowered = content.lower()
        match_count = sum(1 for term in terms if term in lowered) if terms else 0
        score = (match_count / max(len(terms), 1)) if terms else 0.05
        meta = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        meta.setdefault("document_id", chunk.document_id)
        meta.setdefault("document_title", document_title or "Document sans titre")
        meta.setdefault("chunk_index", chunk.chunk_index)
        node = TextNode(id_=f"chunk-{chunk.id}", text=content, metadata=meta)
        passages.append(_node_to_passage(node, fallback_score=score))
        if len(passages) >= k:
            break
    return passages


def _enrich_content_with_heading_and_figure(content: str, metadata: dict) -> str:
    section_label = (
        metadata.get("heading_path")
        or metadata.get("section_parent_heading")
        or metadata.get("scope_label")
        or metadata.get("parent_heading")
        or metadata.get("heading")
    )
    figure_title = metadata.get("figure_title") or metadata.get("image_anchor")
    parts = []
    
    # Injection du résumé de page en contexte additionnel (Pass 2)
    page_summary = metadata.get("page_summary")
    if page_summary and str(page_summary).strip():
        parts.append(f"[Contexte de la page: {page_summary.strip()}]")
        
    if section_label and str(section_label).strip():
        parts.append(f"[Section: {section_label.strip()}]")
    if figure_title and str(figure_title).strip():
        parts.append(str(figure_title).strip())
    if not parts:
        return content
        
    if page_summary:
        prefix = parts[0] + "\n" + " ".join(parts[1:])
    else:
        prefix = " ".join(parts)
        
    return prefix.strip() + "\n\n" + content if content else prefix.strip()


def _merge_leaf_page_into_node_metadata(leaf_node, target_node) -> None:
    leaf_meta = dict(getattr(leaf_node, "metadata", {}) or {})
    m = dict(getattr(target_node, "metadata", {}) or {})
    pn = leaf_meta.get("page_no")
    if pn is not None:
        try:
            m["page_no"] = int(pn)
        except (TypeError, ValueError):
            pass
    elif m.get("page_start") is not None:
        try:
            m["page_no"] = int(m["page_start"])
        except (TypeError, ValueError):
            pass
    ps = leaf_meta.get("page_start")
    pe = leaf_meta.get("page_end")
    if ps is not None:
        try:
            m.setdefault("page_start", int(ps))
        except (TypeError, ValueError):
            pass
    if pe is not None:
        try:
            m.setdefault("page_end", int(pe))
        except (TypeError, ValueError):
            pass
    leaf_chunk_id = _parse_chunk_id_from_node(leaf_node)
    if leaf_chunk_id is not None:
        m["source_leaf_chunk_id"] = leaf_chunk_id
    # Transférer le score RRF brut de la feuille vers le parent
    raw_rrf = leaf_meta.get("raw_rrf_score")
    if raw_rrf is not None:
        m["raw_rrf_score"] = raw_rrf
    setattr(target_node, "metadata", m)


def _chunk_row_to_text_node(row: DocumentChunk, score: float) -> NodeWithScore:
    meta = _merged_chunk_metadata(row.metadata_json, row.metadata_)
    meta.setdefault("document_id", row.document_id)
    meta.setdefault("chunk_index", row.chunk_index)
    node = TextNode(
        text=row.content or "",
        id_=str(row.id),
        metadata=meta,
    )
    return NodeWithScore(node=node, score=score)


def _node_to_passage(node, fallback_score: float = 0.0) -> Dict:
    metadata = dict(getattr(node, "metadata", {}) or {})
    document_title = metadata.get("document_title", "Document sans titre")
    document_id = metadata.get("document_id")
    chunk_index = metadata.get("chunk_index", 0)
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    raw_page = metadata.get("page_no")
    resolved_page = None
    if raw_page is not None:
        try:
            resolved_page = int(raw_page)
        except (TypeError, ValueError):
            pass
    if resolved_page is None:
        for key in ["page_start", "page_label", "page_idx"]:
            val = metadata.get(key)
            if val is not None:
                try:
                    resolved_page = int(val)
                    break
                except (TypeError, ValueError):
                    continue
    page_no = resolved_page
    parent_heading = metadata.get("parent_heading")
    content = node.get_content() if hasattr(node, "get_content") else str(node)
    content_enriched = _enrich_content_with_heading_and_figure(content, metadata)
    passage_text = f"**{document_title}**\n{content_enriched}"
    chunk_id = _parse_chunk_id_from_node(node)
    out = {
        "passage": passage_text,
        "passage_raw": content,
        "document_title": document_title,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "source_leaf_chunk_id": metadata.get("source_leaf_chunk_id"),
        "chunk_index": int(chunk_index) if isinstance(chunk_index, (int, str)) else 0,
        "score": float(fallback_score or 0.0),
        "page_no": page_no,
        "section": parent_heading,
        "source": metadata.get("source"),
    }
    if page_start is not None:
        try:
            out["page_start"] = int(page_start)
        except (TypeError, ValueError):
            pass
    if page_end is not None:
        try:
            out["page_end"] = int(page_end)
        except (TypeError, ValueError):
            pass
    content_type = metadata.get("content_type")
    if content_type:
        out["content_type"] = content_type
    if metadata.get("row_index") is not None:
        out["row_index"] = metadata.get("row_index")
    if metadata.get("table_id"):
        out["table_id"] = metadata.get("table_id")
    raw_rrf = metadata.get("raw_rrf_score")
    if raw_rrf is not None:
        out["raw_rrf_score"] = float(raw_rrf)
    return out


async def search_relevant_passages(
    session: Session,
    space_id: int,
    query_text: str,
    user_id: int,
    k: int = 15,
    document_filter: str = "all",
) -> Dict:
    """
    RAG espace : recherche hybride (pgvector + tsvector) + RRF + rerank.
    
    Args:
        document_filter: "all" (tous), "technical" (exclut FAQ), "faq_corrective" (FAQ uniquement)
    
    Returns:
        Dict avec clés : passages (List[Dict]), status (str), reason (Optional[str])
        Status possibles : "ok", "early_stopped", "low_confidence_clarification", "disabled"
    """
    space = get_space_by_id(session, space_id, user_id)
    if not space:
        logger.warning("Espace %d inaccessible (user %d)", space_id, user_id)
        return {"passages": [], "status": "disabled", "reason": "space_not_found"}
    if not query_text or not query_text.strip():
        return {"passages": [], "status": "disabled", "reason": "empty_query"}

    try:
        candidate_k = max(k, min(k * 4, 80))

        # Calculer l'embedding une seule fois (réutilisé par retrieval)
        query_embedding = generate_embedding(query_text)
        if not query_embedding:
            logger.warning("Embedding requête vide pour space_id=%s", space_id)
            return {"passages": [], "status": "disabled", "reason": "embedding_failed"}
        
        # 1. Recherche vectorielle dense (pgvector)
        with trace_run(
            "vector_retrieval",
            run_type="retriever",
            inputs={"query": query_text, "space_id": space_id, "candidate_k": candidate_k},
            tags=["retrieval", "vector", "space"],
        ) as vr:
            raw_vector = _retrieve_leaves_sql(
                session, space_id, user_id, query_text, candidate_k, query_embedding=query_embedding
            )
            vr.end(outputs={"nb": len(raw_vector)})

        # 2. Recherche lexicale BM25 native (tsvector + ts_rank_cd 33)
        with trace_run(
            "bm25_lexical_retrieval",
            run_type="retriever",
            inputs={"query": query_text, "space_id": space_id, "candidate_k": candidate_k, "document_filter": document_filter},
            tags=["retrieval", "bm25", "lexical", "space"],
        ) as lr:
            try:
                raw_lexical = _retrieve_leaves_bm25_sql(
                    session, space_id, user_id, query_text, candidate_k, document_filter=document_filter
                )
            except Exception as e:
                logger.warning("Recherche BM25 lexicale échouée (migration probablement non appliquée) : %s", e)
                session.rollback()
                raw_lexical = []
            lr.end(outputs={"nb": len(raw_lexical)})

        # 3. Recherche directe par correspondances alphanumériques exactes/substrings (regex local)
        with trace_run(
            "alphanumeric_retrieval",
            run_type="retriever",
            inputs={"query": query_text, "space_id": space_id, "candidate_k": candidate_k},
            tags=["retrieval", "alphanumeric", "space"],
        ) as ar:
            raw_alphanumeric = _retrieve_leaves_alphanumeric_sql(
                session,
                space_id,
                user_id,
                query_text,
                candidate_k,
                document_filter=document_filter
            )
            ar.end(outputs={"nb": len(raw_alphanumeric)})

        if not raw_vector and not raw_lexical and not raw_alphanumeric:
            return {"passages": [], "status": "ok", "reason": "no_results"}

        # Fusion RRF (Reciprocal Rank Fusion) - pas de normalisation ici, on garde les scores bruts
        fused_results = reciprocal_rank_fusion(
            raw_vector,
            raw_lexical,
            alphanumeric_results=raw_alphanumeric,
            top_n=candidate_k,
            normalize=False
        )
        
        # Sélection des meilleurs résultats hybrides (on garde candidate_k candidats pour la résolution
        # des parents et la déduplication, garantissant qu'on dispose de k résultats uniques à la fin)
        top_leaves = fused_results[:candidate_k]

        with trace_run(
            "parent_resolution",
            run_type="chain",
            inputs={"space_id": space_id, "nb": len(top_leaves)},
            tags=["parent", "space"],
        ) as pr:
            parent_node_dict = _build_parent_node_dict(session, space_id, user_id)
            final_nodes: List[NodeWithScore] = []
            seen_node_ids: set = set()

            for nws in top_leaves:
                score = float(getattr(nws, "score", 0.0) or 0.0)
                leaf_meta = dict(getattr(nws.node, "metadata", {}) or {})
                content_type = leaf_meta.get("content_type")
                parent_node_id = leaf_meta.get("parent_node_id")
                target_node = None
                
                is_multimodal = content_type in (
                    "page_raw_enriched",
                    "page_section_report",
                    "page_window_report",
                )

                if content_type in ("table_row", "table_summary"):
                    target_node = nws.node
                elif is_multimodal and parent_node_id:
                    # Résolution du résumé parent Pass 2 comme contexte additionnel
                    parent_node = parent_node_dict.get(parent_node_id)
                    if parent_node is None:
                        doc_id = leaf_meta.get("document_id")
                        try:
                            doc_id_int = int(doc_id) if doc_id is not None else None
                        except (TypeError, ValueError):
                            doc_id_int = None
                        parent_node = _resolve_space_parent_with_multihop(
                            session,
                            space_id,
                            user_id,
                            doc_id_int,
                            parent_node_id,
                            parent_node_dict,
                        )
                    if parent_node:
                        # Assigner le contenu du parent à la métadonnée page_summary
                        leaf_meta["page_summary"] = parent_node.text
                        nws.node.metadata = leaf_meta
                    target_node = nws.node
                elif parent_node_id:
                    target_node = parent_node_dict.get(parent_node_id)
                    if target_node is None:
                        doc_id = leaf_meta.get("document_id")
                        try:
                            doc_id_int = int(doc_id) if doc_id is not None else None
                        except (TypeError, ValueError):
                            doc_id_int = None
                        target_node = _resolve_space_parent_with_multihop(
                            session,
                            space_id,
                            user_id,
                            doc_id_int,
                            parent_node_id,
                            parent_node_dict,
                        )
                if target_node is None:
                    target_node = nws.node
                elif not is_multimodal:
                    _merge_leaf_page_into_node_metadata(nws.node, target_node)
                
                node_id = getattr(target_node, "id_", None)
                if node_id and node_id in seen_node_ids:
                    continue
                if node_id:
                    seen_node_ids.add(node_id)
                final_nodes.append(NodeWithScore(node=target_node, score=score))
            pr.end(outputs={"nb_final": len(final_nodes)})

        if not final_nodes:
            return {"passages": [], "status": "ok", "reason": "no_results_after_resolution"}

        # === RERANKER ===
        
        # Reranker cross-encoder (sur le pool, pas sur tout)
        if settings.RERANKER_ENABLED:
            pool_size = min(settings.RERANK_POOL, len(final_nodes))
            pool_nodes = final_nodes[:pool_size]
            
            with trace_run(
                "cross_encoder_rerank",
                run_type="reranker",
                inputs={
                    "pool_size": pool_size,
                    "char_cap": settings.RERANK_CHAR_CAP,
                    "batch_size": settings.RERANK_BATCH_SIZE,
                },
                tags=["rerank", "cross_encoder"],
            ) as cer:
                scored = await reranker_service.rerank_nodes(
                    query_text,
                    pool_nodes,
                    char_cap=settings.RERANK_CHAR_CAP,
                    batch_size=settings.RERANK_BATCH_SIZE,
                )
                cer.end(outputs={
                    "nb_scored": len(scored),
                    "top3_scores": [round(s, 3) for _, s in scored[:3]] if scored else [],
                })
            
            with trace_run(
                "rerank_guardrails",
                run_type="chain",
                inputs={
                    "min_k": settings.MIN_DYNAMIC_K,
                    "max_k": settings.MAX_DYNAMIC_K,
                    "softmax_cum_threshold": settings.SOFTMAX_CUM_THRESHOLD,
                },
                tags=["rerank", "guardrails"],
            ) as rg:
                rerank_result = reranker_service.apply_dynamic_filtering(
                    scored,
                    min_k=settings.MIN_DYNAMIC_K,
                    max_k=settings.MAX_DYNAMIC_K,
                    softmax_cum_threshold=settings.SOFTMAX_CUM_THRESHOLD,
                    stutter_gap=settings.STUTTER_GAP,
                    zscore_flat_threshold=settings.ZSCORE_FLAT_THRESHOLD,
                )
                rg.end(outputs={
                    "status": rerank_result.status,
                    "nb_nodes": len(rerank_result.nodes),
                    "gap_top1_top2": rerank_result.gap_top1_top2,
                    "zscore_flatness": rerank_result.zscore_flatness,
                })
            
            # Si bégaiement détecté : top 1-2 seulement pour forcer clarification
            if rerank_result.status == "low_confidence_clarification":
                passages_low_conf = [
                    _node_to_passage(nws.node, fallback_score=1.0 - i * 0.01)
                    for i, nws in enumerate(rerank_result.nodes)
                ]
                logger.warning(
                    "Low confidence détectée : %d passages seulement (gap=%.4f, zscore=%.4f)",
                    len(passages_low_conf),
                    rerank_result.gap_top1_top2 or 0.0,
                    rerank_result.zscore_flatness or 0.0,
                )
                return {
                    "passages": passages_low_conf,
                    "status": "low_confidence_clarification",
                    "reason": rerank_result.reason,
                }
            
            # Utiliser le résultat du rerank directement (sans MMR et sans feedback boost)
            passages = [
                _node_to_passage(nws.node, fallback_score=1.0 - i * 0.01)
                for i, nws in enumerate(rerank_result.nodes)
            ]
            return {
                "passages": passages,
                "status": "ok",
                "reason": rerank_result.reason,
            }
        
        # Reranker désactivé : retour simple des top-K RRF
        passages = [
            _node_to_passage(nws.node, fallback_score=float(nws.score or 0.0))
            for nws in final_nodes[:k]
        ]
        return {"passages": passages, "status": "disabled", "reason": "reranker_disabled"}
        
    except Exception as e:
        logger.error("search_relevant_passages (space): %s", e, exc_info=True)
        return {"passages": [], "status": "disabled", "reason": f"error: {str(e)}"}


async def search_technical_passages(
    session: Session,
    space_id: int,
    query_text: str,
    user_id: int,
    k: int = 15,
) -> Dict:
    """
    Recherche RAG limitée aux documents techniques (exclut les FAQ correctives).
    
    Wrapper autour de search_relevant_passages avec document_filter="technical".
    
    Returns:
        Dict avec clés : passages (List[Dict]), status (str), reason (Optional[str])
    """
    return await search_relevant_passages(
        session=session,
        space_id=space_id,
        query_text=query_text,
        user_id=user_id,
        k=k,
        document_filter="technical",
    )


async def search_corrective_faq_passages(
    session: Session,
    space_id: int,
    query_text: str,
    user_id: int,
    draft_response: str = "",
    k: Optional[int] = None,
) -> Dict:
    """
    Recherche post-brouillon dédiée aux FAQ correctives issues des feedbacks négatifs.
    
    Pipeline léger : vector + BM25 → RRF (normalisé) → filtre score.
    
    Args:
        draft_response: Réponse brouillon générée, ajoutée à la requête pour améliorer le rappel
        k: Nombre de FAQ à retourner (défaut: FAQ_TOP_K depuis config)
    
    Returns:
        Dict avec clés : passages (List[Dict]), status (str), reason (Optional[str])
        Status "below_threshold" si aucune FAQ ne dépasse FAQ_MIN_SIMILARITY
    """
    if k is None:
        k = settings.FAQ_TOP_K
    
    if not settings.FAQ_POST_DRAFT_ENABLED:
        return {"passages": [], "status": "disabled", "reason": "faq_post_draft_disabled"}
    
    space = get_space_by_id(session, space_id, user_id)
    if not space:
        logger.warning("Espace %d inaccessible (user %d) pour recherche FAQ", space_id, user_id)
        return {"passages": [], "status": "disabled", "reason": "space_not_found"}
    
    # Enrichir la requête avec le brouillon (tronqué) pour capturer les erreurs
    enriched_query = query_text
    if draft_response:
        draft_preview = draft_response[:800]
        enriched_query = f"{query_text}\n\nRéponse générée: {draft_preview}"
    
    try:
        candidate_k = min(k * 4, 40)  # Pool plus petit que la recherche technique
        
        # Embedding de la requête enrichie
        query_embedding = generate_embedding(enriched_query)
        if not query_embedding:
            logger.warning("Embedding requête FAQ vide pour space_id=%s", space_id)
            return {"passages": [], "status": "disabled", "reason": "embedding_failed"}
        
        # Recherche vectorielle FAQ uniquement
        with trace_run(
            "faq_vector_retrieval",
            run_type="retriever",
            inputs={"query": query_text, "space_id": space_id, "candidate_k": candidate_k},
            tags=["retrieval", "vector", "faq_corrective"],
        ) as vr:
            raw_vector = _retrieve_leaves_sql(
                session, space_id, user_id, enriched_query, candidate_k,
                query_embedding=query_embedding, document_filter="faq_corrective"
            )
            vr.end(outputs={"nb": len(raw_vector)})
        
        # Recherche lexicale FAQ uniquement
        with trace_run(
            "faq_bm25_retrieval",
            run_type="retriever",
            inputs={"query": query_text, "space_id": space_id, "candidate_k": candidate_k},
            tags=["retrieval", "bm25", "faq_corrective"],
        ) as lr:
            try:
                raw_lexical = _retrieve_leaves_bm25_sql(
                    session, space_id, user_id, enriched_query, candidate_k,
                    document_filter="faq_corrective"
                )
            except Exception as e:
                logger.warning("Recherche BM25 FAQ échouée: %s", e)
                session.rollback()
                raw_lexical = []
            lr.end(outputs={"nb": len(raw_lexical)})
        
        if not raw_vector and not raw_lexical:
            return {"passages": [], "status": "no_faq", "reason": "no_faq_documents_in_space"}
        
        # Fusion RRF (normalisée pour comparaison avec seuil)
        fused_results = reciprocal_rank_fusion(raw_vector, raw_lexical, top_n=candidate_k, normalize=True)
        
        # Filtrer par seuil de similarité FAQ dédié
        filtered_faq = []
        for nws in fused_results[:k]:
            score = float(getattr(nws, "score", 0.0) or 0.0)
            if score >= settings.FAQ_MIN_SIMILARITY:
                filtered_faq.append(nws)
        
        if not filtered_faq:
            logger.info(
                "FAQ search: aucune FAQ au-dessus du seuil %.2f (meilleur score: %.3f)",
                settings.FAQ_MIN_SIMILARITY,
                fused_results[0].score if fused_results else 0.0,
            )
            return {"passages": [], "status": "below_threshold", "reason": "no_faq_above_threshold"}
        
        # Conversion en passages (pas de résolution parent ni MMR pour les FAQ)
        passages = [
            _node_to_passage(nws.node, fallback_score=float(nws.score or 0.0))
            for nws in filtered_faq
        ]
        
        logger.info(
            "FAQ corrective search: %d FAQ trouvées (seuil=%.2f, top_score=%.3f)",
            len(passages), settings.FAQ_MIN_SIMILARITY, passages[0]["score"] if passages else 0.0,
        )
        
        return {
            "passages": passages,
            "status": "ok",
            "reason": "faq_corrective_found",
        }
        
    except Exception as e:
        logger.error("search_corrective_faq_passages: %s", e, exc_info=True)
        return {"passages": [], "status": "disabled", "reason": f"error: {str(e)}"}


def refine_with_source_authority(
    passages: List[Dict],
    query: str,
    reasoning_result: Any,
) -> List[Dict]:
    """
    Optimise l'autorité des sources par rapport à l'intention détectée.
    Si primary_source correspond à la source du passage, on applique un boost au score.
    """
    if not passages or not reasoning_result:
        return passages
    
    primary = getattr(reasoning_result, "primary_source", None)
    confidence = getattr(reasoning_result, "confidence", 0.0)
    
    boost_val = 0.8 * confidence if primary else 0.0
    
    refined = []
    for p in passages:
        p_copy = dict(p)
        score = p_copy.get("score", 0.0)
        source = p_copy.get("source")
        
        if primary and source and source.lower() == primary.lower():
            p_copy["score"] = score + boost_val
            
        refined.append(p_copy)
        
    refined.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return refined
