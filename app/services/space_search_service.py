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
from typing import Dict, List, Optional, Set, Tuple

from llama_index.core.schema import NodeWithScore, TextNode
from sqlalchemy import or_, text
from sqlmodel import Session, select

from app.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.services.embedding_service import generate_embedding
from app.services.query_reasoning_service import QueryIntent, reason_query_intent
from app.services.space_service import get_space_by_id
from app.tracing import trace_run
from app.services import reranker_service, mmr_service

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
    
    # Clause de filtre selon le type de document
    filter_clause = ""
    if document_filter == "technical":
        filter_clause = f"AND d.title NOT LIKE '{settings.FAQ_CORRECTIVE_TITLE_PREFIX}%'"
    elif document_filter == "faq_corrective":
        filter_clause = f"AND d.title LIKE '{settings.FAQ_CORRECTIVE_TITLE_PREFIX}%'"

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


def _compute_term_idfs(
    session: Session,
    space_id: int,
    terms: List[str],
) -> Tuple[Dict[str, float], int]:
    """Calcule l'IDF BM25 de chaque terme dans un espace donné.

    Exécute une seule requête SQL groupée pour obtenir le document frequency
    de chaque terme, puis calcule l'IDF standard BM25 :
        idf = ln((N - df + 0.5) / (df + 0.5) + 1)

    Returns:
        (dict[term -> idf], total_docs_in_space)
    """
    if not terms:
        return {}, 0

    # Compter le total de documents-feuilles dans l'espace
    total_sql = text("""
        SELECT COUNT(DISTINCT dc.id)
        FROM documentchunk dc
        INNER JOIN document_space ds ON ds.document_id = dc.document_id
        WHERE ds.space_id = :space_id AND dc.is_leaf = true
    """)
    total_docs = session.execute(total_sql, {"space_id": space_id}).scalar() or 1

    # Compter le doc frequency de chaque terme en une seule requête
    # unnest + LATERAL pour éviter N+1
    idf_sql = text("""
        SELECT
            t.term,
            COUNT(*) AS df
        FROM unnest(CAST(:terms_array AS text[])) AS t(term)
        INNER JOIN documentchunk dc ON dc.is_leaf = true
        INNER JOIN document_space ds ON ds.document_id = dc.document_id
        WHERE ds.space_id = :space_id
          AND dc.tsv_content @@ websearch_to_tsquery('french', t.term)
        GROUP BY t.term
    """)
    result = session.execute(idf_sql, {"space_id": space_id, "terms_array": terms})

    idfs: Dict[str, float] = {}
    for row in result:
        df = row.df
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
        idfs[row.term] = idf

    # Termes absents du corpus → IDF maximal (très rares)
    for term in terms:
        if term not in idfs:
            idfs[term] = math.log((total_docs + 0.5) / 0.5 + 1)

    logger.debug(
        "IDF BM25 (space %d, N=%d): %s",
        space_id, total_docs,
        {t: round(v, 3) for t, v in idfs.items()},
    )
    return idfs, total_docs


def _bm25_rescore(
    nodes: List[NodeWithScore],
    terms: List[str],
    idfs: Dict[str, float],
) -> List[NodeWithScore]:
    """Re-score les résultats lexicaux avec une pondération IDF BM25.

    Le score ts_rank_cd (avec flag normalisation longueur) sert de proxy pour
    le TF normalisé. On le multiplie par la somme pondérée des IDF des termes
    trouvés dans le contenu du chunk.

    Score final = ts_rank_cd_norm * sum(idf[t] pour t dans termes ∩ contenu)
    """
    if not nodes or not idfs:
        return nodes

    for nws in nodes:
        content_lower = (nws.node.text or "").lower()
        # Somme des IDF des termes qui apparaissent dans le contenu
        idf_sum = sum(
            idfs.get(t, 0.0)
            for t in terms
            if t in content_lower
        )
        # Multiplier le score ts_rank_cd par le poids IDF
        # Plancher à 1.0 pour ne pas réduire les scores si aucun IDF
        idf_weight = max(idf_sum, 1.0)
        nws.score = float(nws.score) * idf_weight

    # Re-trier par score décroissant
    nodes.sort(key=lambda n: n.score, reverse=True)
    return nodes


def _retrieve_leaves_bm25_sql(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    candidate_k: int,
    document_filter: str = "all",
) -> List[NodeWithScore]:
    """Recherche lexicale BM25 approximative sur les feuilles.

    Étapes :
    1. Extraction de termes améliorée (v2)
    2. Retrieval via ts_rank_cd avec flag 1 (normalisation log-longueur)
    3. Calcul IDF par terme (requête SQL unique)
    4. Re-scoring BM25 approximatif : ts_rank_cd_norm × Σ IDF(terme)
    
    Args:
        document_filter: "all" (tous), "technical" (exclut FAQ), "faq_corrective" (FAQ uniquement)
    """
    terms = _extract_query_terms(query_text)
    if not terms:
        logger.info("BM25 lexical (space): Aucun terme significatif extrait de la requête.")
        return []

    # Construction d'une requête OR pour websearch_to_tsquery (ex: "vitrage OR soleal")
    or_query = " OR ".join(terms)
    
    # Clause de filtre selon le type de document
    filter_clause = ""
    if document_filter == "technical":
        filter_clause = f"AND d.title NOT LIKE '{settings.FAQ_CORRECTIVE_TITLE_PREFIX}%'"
    elif document_filter == "faq_corrective":
        filter_clause = f"AND d.title LIKE '{settings.FAQ_CORRECTIVE_TITLE_PREFIX}%'"

    # Retrieval avec ts_rank_cd + flag 1 (normalisation par log de la longueur)
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
            ts_rank_cd(dc.tsv_content, websearch_to_tsquery('french', :query), 1) AS similarity_score
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

    # Phase IDF : calculer les poids et re-scorer
    if nodes:
        idfs, _ = _compute_term_idfs(session, space_id, terms)
        nodes = _bm25_rescore(nodes, terms, idfs)

    logger.info("BM25 lexical (space): %d feuilles (limit=%d, terms=%s)", len(nodes), candidate_k, terms)
    return nodes


def reciprocal_rank_fusion(
    vector_results: List[NodeWithScore],
    lexical_results: List[NodeWithScore],
    k: int = 60,
    top_n: int = 15,
) -> List[NodeWithScore]:
    """
    Fusionne les résultats de recherche vectorielle et lexicale avec l'algorithme RRF,
    puis normalise les scores dans l'intervalle [0.1, 0.9] pour rester compatibles avec les boosts.
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

    # Trier par score RRF décroissant
    sorted_node_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results: List[NodeWithScore] = []
    for node_id, rrf_score in sorted_node_ids:
        node = nodes_map[node_id].node
        node.metadata["raw_rrf_score"] = rrf_score
        results.append(NodeWithScore(node=node, score=rrf_score))

    # Normalisation linéaire dans [0.1, 0.9]
    if results:
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


def _filter_by_vector_score(
    candidates: List[NodeWithScore],
    min_threshold: float = MIN_VECTOR_SIMILARITY_THRESHOLD,
) -> List[NodeWithScore]:
    filtered = [c for c in candidates if float(c.score or 0.0) >= min_threshold]
    if filtered:
        return filtered
    return sorted(candidates, key=lambda c: float(c.score or 0.0), reverse=True)[: max(5, len(candidates) // 2 or 1)]


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
    if section_label and str(section_label).strip():
        parts.append(f"[Section: {section_label.strip()}]")
    if figure_title and str(figure_title).strip():
        parts.append(str(figure_title).strip())
    if not parts:
        return content
    prefix = " ".join(parts) + "\n\n"
    return prefix + content if content else prefix.strip()


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


def filter_passages_by_rrf_score(
    passages: List[Dict],
    *,
    enabled: bool = True,
    min_k: int = 1,
    max_k: int = 10,
    factor: float = 0.70,
) -> List[Dict]:
    """
    Filtre dynamiquement une liste de passages basée sur leurs scores RRF bruts (Option A).
    Conserve uniquement les passages ayant un score RRF brut >= max_score * factor.
    S'il n'y a pas de score brut, utilise la clé 'score' (normalisée).
    """
    if not enabled or not passages:
        return passages

    # Récupérer tous les scores RRF (bruts ou normalisés) pour trouver le maximum absolu
    all_scores = [p.get("raw_rrf_score", p["score"]) for p in passages]
    if not all_scores:
        return passages

    max_score = max(all_scores)
    threshold = max_score * factor

    # Conserver obligatoirement les min_k premiers passages (les meilleurs après boosts)
    best_passages = passages[:min_k]
    
    # Filtrer le reste des passages
    other_passages = [
        p for p in passages[min_k:]
        if p.get("raw_rrf_score", p["score"]) >= threshold
    ]
    
    filtered = best_passages + other_passages
    
    # Limiter à la borne supérieure max_k
    result = filtered[:max_k]
    
    logger.info(
        "RRF Dynamique (Option A) : %d/%d passages conservés (seuil RRF >= %.4f * %.2f = %.4f, min_k=%d, max_k=%d)",
        len(result),
        len(passages),
        max_score,
        factor,
        threshold,
        min_k,
        max_k,
    )
    return result


def _normalize_for_gamme(s: str) -> str:
    if not s:
        return ""
    n = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in n if unicodedata.category(c) != "Mn")


def _get_meaningful_words(text: str) -> Set[str]:
    if not text or not text.strip():
        return set()
    normalized = _normalize_for_gamme(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return {w for w in tokens if len(w) > 3 and w not in _FALLBACK_STOPWORDS}


def refine_with_source_authority(
    passages: List[Dict],
    query_text: str,
    reasoning_result: Optional[QueryIntent] = None,
) -> List[Dict]:
    if not passages:
        return passages
    if reasoning_result and reasoning_result.primary_source:
        source_to_boost = reasoning_result.primary_source.lower()
        for p in passages:
            doc_source = (p.get("source") or "").lower()
            if doc_source == source_to_boost:
                p["score"] = float(p.get("score") or 0.0) + 0.8
    if query_text and query_text.strip():
        query_words = _get_meaningful_words(query_text)
        if query_words:
            for p in passages:
                document_title = (p.get("document_title") or "").strip()
                if not document_title:
                    continue
                title_words = _get_meaningful_words(document_title)
                common = query_words & title_words
                if common:
                    boost = min(
                        TITLE_QUERY_BOOST_PER_MATCH * len(common),
                        TITLE_QUERY_BOOST_CAP,
                    )
                    p["score"] = float(p.get("score") or 0.0) + boost
    passages.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return passages


def apply_feedback_boost(
    passages: List[Dict],
    space_id: int,
    session: Session,
) -> List[Dict]:
    """
    Ajuste les scores des passages basés sur les retours utilisateurs 👍/👎.
    Boost modéré pour les passages utiles, pénalité pour les passages incorrects.

    Pondération temporelle (P1-C) : les feedbacks anciens sont atténués par une
    décroissance exponentielle de demi-vie FEEDBACK_HALFLIFE_DAYS (défaut: 30 jours).
    FEEDBACK_HALFLIFE_DAYS=0 désactive la pondération temporelle.
    """
    if not passages:
        return passages

    from datetime import datetime, timezone
    from app.models.message_feedback import MessageFeedback

    # 1. Identifier tous les chunk_ids présents dans les passages candidats
    candidate_chunk_ids = set()
    for p in passages:
        cid = p.get("source_leaf_chunk_id") or p.get("chunk_id")
        if cid:
            candidate_chunk_ids.add(cid)

    if not candidate_chunk_ids:
        return passages

    # 2. Récupérer tous les feedbacks pour cet espace (avec created_at pour pondération temporelle)
    feedbacks = session.exec(
        select(MessageFeedback.is_positive, MessageFeedback.chunk_ids, MessageFeedback.created_at)
        .where(MessageFeedback.space_id == space_id)
    ).all()

    halflife = settings.FEEDBACK_HALFLIFE_DAYS
    now_utc = datetime.now(timezone.utc)

    # 3. Compter les 👍/👎 par chunk avec pondération temporelle
    chunk_stats: Dict[int, Dict[str, float]] = {}
    for is_positive, chunk_ids_list, created_at in feedbacks:
        if not chunk_ids_list:
            continue
        # Calcul du poids temporel (1.0 si halflife désactivé)
        if halflife > 0 and created_at is not None:
            # S'assurer que created_at est timezone-aware
            if created_at.tzinfo is None:
                created_at_utc = created_at.replace(tzinfo=timezone.utc)
            else:
                created_at_utc = created_at
            age_days = max(0.0, (now_utc - created_at_utc).total_seconds() / 86400)
            time_weight = 0.5 ** (age_days / halflife)
        else:
            time_weight = 1.0

        for cid in chunk_ids_list:
            if cid in candidate_chunk_ids:
                if cid not in chunk_stats:
                    chunk_stats[cid] = {"positive": 0.0, "negative": 0.0}
                if is_positive:
                    chunk_stats[cid]["positive"] += time_weight
                else:
                    chunk_stats[cid]["negative"] += time_weight

    # 4. Appliquer le boost/pénalité sur le score normalisé
    # Boost : +0.15 * ratio si ratio > 0, Pénalité : -0.10 * |ratio| si ratio < 0
    for p in passages:
        cid = p.get("source_leaf_chunk_id") or p.get("chunk_id")
        if cid and cid in chunk_stats:
            stats = chunk_stats[cid]
            pos = stats["positive"]
            neg = stats["negative"]
            total = pos + neg
            if total > 0:
                ratio = (pos - neg) / total
                if ratio > 0:
                    boost = 0.15 * ratio
                else:
                    boost = 0.10 * ratio  # ratio négatif → pénalité
                p["score"] = float(p.get("score") or 0.0) + boost
                logger.info(
                    "Feedback boost (temporel) chunk %d : +%.2f/-%.2f (ratio=%.2f, boost=%.3f)",
                    cid, pos, neg, ratio, boost
                )

    # 5. Re-trier
    passages.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return passages


def _build_competitor_patterns(
    detected_refs: List[str],
) -> List[re.Pattern]:
    """
    Construit les patterns regex des gammes "concurrentes" proches des références demandées.

    Exemple : si on cherche "Perform-76", les concurrents sont les autres variantes
    de Perform (Perform-70, Perform 80, etc.) mais PAS Perform-76 lui-même.

    Stratégie : pour chaque référence, extraire la racine alphanumérique et construire
    un pattern qui matche la même famille SANS la référence exacte.
    """
    if not detected_refs:
        return []

    competitor_patterns: List[re.Pattern] = []
    for ref in detected_refs:
        # Extraire la racine : partie alphabétique avant les chiffres (ex: "Perform" depuis "Perform-76")
        root_match = re.match(r'^([A-Za-z]+)', ref.strip())
        if not root_match:
            continue
        root = root_match.group(1)
        if len(root) < 3:  # Trop court = trop de faux positifs
            continue
        # Pattern : même racine + séparateur optionnel + chiffres différents du suffix de ref
        # On extrait le suffix numérique de ref pour l'exclure
        suffix_match = re.search(r'(\d+)', ref)
        if not suffix_match:
            continue
        exact_suffix = suffix_match.group(1)
        # Pattern "même famille mais pas le même numéro"
        # ex: Perform(?:[\s\-]?)(?!76\b)\d{2,3}
        pattern_str = (
            rf'\b{re.escape(root)}'
            rf'[\s\-\.]*'
            rf'(?!{re.escape(exact_suffix)}\b)'
            rf'\d{{2,3}}\b'
        )
        try:
            competitor_patterns.append(re.compile(pattern_str, re.IGNORECASE))
        except re.error:
            logger.debug("Pattern concurrent invalide pour ref '%s', ignoré", ref)
    return competitor_patterns


def apply_exact_ref_scoring(
    passages: List[Dict],
    detected_refs: List[str],
    *,
    bonus: float = 1.5,
    competitor_penalty: float = 0.4,
) -> List[Dict]:
    """
    Pour les requêtes intent=exact_reference :
    - Bonus massif (+bonus) si le passage contient la référence exacte demandée
    - Pénalité multiplicative (*competitor_penalty) si le passage contient une
      référence concurrente de la même famille mais avec un numéro différent.

    Les deux modifications sont cumulatives si un chunk contient les deux (rare).
    """
    if not passages or not detected_refs:
        return passages

    ref_patterns = [
        re.compile(r'\b' + re.escape(r) + r'\b', re.IGNORECASE)
        for r in detected_refs
    ]
    competitor_patterns = _build_competitor_patterns(detected_refs)

    for p in passages:
        content = (p.get("passage_raw") or p.get("passage") or "").lower()
        has_exact = any(pat.search(content) for pat in ref_patterns)
        has_competitor = bool(competitor_patterns) and any(
            pat.search(content) for pat in competitor_patterns
        )

        score = float(p.get("score") or 0.0)
        if has_exact:
            score += bonus
            logger.debug(
                "exact_ref_scoring: +%.2f bonus (chunk_id=%s, refs=%s)",
                bonus, p.get("chunk_id"), detected_refs
            )
        if has_competitor and not has_exact:
            # Pénalité uniquement si la ref exacte est absente
            score *= competitor_penalty
            logger.debug(
                "exact_ref_scoring: *%.2f pénalité concurrent (chunk_id=%s)",
                competitor_penalty, p.get("chunk_id")
            )
        p["score"] = score

    passages.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    logger.info(
        "apply_exact_ref_scoring: %d passages triés (refs=%s, bonus=%.2f, penalty=%.2f)",
        len(passages), detected_refs, bonus, competitor_penalty
    )
    return passages


async def search_relevant_passages(
    session: Session,
    space_id: int,
    query_text: str,
    user_id: int,
    k: int = 15,
    document_filter: str = "all",
) -> Dict:
    """
    RAG espace : recherche hybride (pgvector + tsvector) + RRF + rerank + MMR.
    
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

    reasoning_result = await reason_query_intent(query_text)
    if reasoning_result.intent != "generic":
        logger.info(
            "CQR [space]: intent=%s primary_source=%s",
            reasoning_result.intent,
            reasoning_result.primary_source,
        )

    try:
        candidate_k = max(k, min(k * 4, 80))

        # Query expansion : enrichir le texte d'embedding avec les termes du LLM
        expanded_query = query_text
        if reasoning_result.search_terms:
            extra = " ".join(reasoning_result.search_terms)
            expanded_query = f"{query_text} {extra}"
            logger.info(
                "Query expansion [space]: +%d termes → '%s'",
                len(reasoning_result.search_terms), extra,
            )

        # Calculer l'embedding une seule fois (réutilisé par retrieval + MMR)
        # Utilise la query enrichie pour un meilleur rappel vectoriel
        query_embedding = generate_embedding(expanded_query)
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

        # 2. Recherche lexicale BM25 approximative (tsvector + IDF)
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

        if not raw_vector and not raw_lexical:
            fallback_passages = await _keyword_fallback_passages(
                session, space_id, user_id, query_text, k
            )
            return {"passages": fallback_passages, "status": "ok", "reason": "fallback_keyword"}

        # Filtrage par score minimum sur la partie vectorielle uniquement pour éviter le bruit sémantique
        filtered_vector = _filter_by_vector_score(raw_vector)

        # Fusion RRF (Reciprocal Rank Fusion)
        fused_results = reciprocal_rank_fusion(filtered_vector, raw_lexical, top_n=candidate_k)
        
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
                if content_type in ("table_row", "table_summary"):
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
                else:
                    _merge_leaf_page_into_node_metadata(nws.node, target_node)
                node_id = getattr(target_node, "id_", None)
                if node_id and node_id in seen_node_ids:
                    continue
                if node_id:
                    seen_node_ids.add(node_id)
                final_nodes.append(NodeWithScore(node=target_node, score=score))
            pr.end(outputs={"nb_final": len(final_nodes)})

        passages = [
            _node_to_passage(nws.node, fallback_score=float(nws.score or 0.0))
            for nws in final_nodes
        ]
        passages = refine_with_source_authority(
            passages, query_text, reasoning_result=reasoning_result
        )
        passages = apply_feedback_boost(passages, space_id, session)

        # P0-C : Bonus exact-match + pénalité gamme concurrente si intent=exact_reference
        is_exact_ref = (
            reasoning_result.intent == "exact_reference"
            and bool(getattr(reasoning_result, "detected_refs", None))
        )
        if is_exact_ref:
            passages = apply_exact_ref_scoring(
                passages,
                reasoning_result.detected_refs,
                bonus=settings.EXACT_REF_EXACT_MATCH_BONUS,
                competitor_penalty=settings.EXACT_REF_PENALTY_COMPETITOR,
            )

        if not passages:
            fallback_passages = await _keyword_fallback_passages(
                session, space_id, user_id, query_text, k
            )
            return {"passages": fallback_passages, "status": "ok", "reason": "fallback_keyword"}

        # === RERANKER + MMR ===

        # P0-B : Pour les requêtes exact_reference, toujours forcer le reranker
        # (même si les scores RRF sont déjà élevés, le reranker est le seul à détecter
        # la preuve exacte dans le texte)
        is_exact_ref_rerank_forced = (
            is_exact_ref and settings.EXACT_REF_FORCE_RERANK
        )

        # Early stopping : court-circuite le reranker si scores RRF déjà excellents
        # P1-B : opère sur raw_rrf_score (brut) et non sur le score normalisé+boosté
        if settings.RERANKER_ENABLED and not is_exact_ref_rerank_forced:
            with trace_run(
                "early_stopping",
                run_type="chain",
                inputs={
                    "top_n": settings.EARLY_STOP_TOP_N,
                    "threshold": settings.EARLY_STOP_MEAN_THRESHOLD,
                    "exact_ref_forced": is_exact_ref_rerank_forced,
                },
                tags=["rerank", "early_stop"],
            ) as es:
                # P1-B : utiliser raw_rrf_score si disponible (non affecté par boosts amont)
                rrf_scores = [
                    p.get("raw_rrf_score", p["score"])
                    for p in passages[:settings.EARLY_STOP_TOP_N]
                ]
                should_stop = reranker_service.should_early_stop(rrf_scores, settings.EARLY_STOP_MEAN_THRESHOLD)
                es.end(outputs={"triggered": should_stop})

                if should_stop:
                    logger.info("Early stop activé : scores RRF déjà excellents, skip rerank + MMR")
                    filtered_passages = filter_passages_by_rrf_score(
                        passages,
                        enabled=settings.RRF_DYNAMIC_K_ENABLED,
                        min_k=settings.RRF_MIN_K,
                        max_k=settings.RRF_MAX_K,
                        factor=settings.RRF_RELATIVE_THRESHOLD_FACTOR,
                    )
                    return {
                        "passages": filtered_passages,
                        "status": "early_stopped",
                        "reason": "mean_score_above_threshold",
                    }
        elif is_exact_ref_rerank_forced:
            logger.info(
                "Early stop DÉSACTIVÉ : intent=exact_reference (refs=%s), reranker forcé",
                getattr(reasoning_result, 'detected_refs', []),
            )
        
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
                scored = reranker_service.rerank_nodes(
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
            
            # MMR sur les survivants du rerank
            if settings.MMR_ENABLED and rerank_result.nodes:
                # Extraire chunk_ids pour fetch embeddings
                chunk_ids = []
                for nws in rerank_result.nodes:
                    chunk_id = _parse_chunk_id_from_node(nws.node)
                    if chunk_id:
                        chunk_ids.append(chunk_id)
                
                with trace_run(
                    "mmr_selection",
                    run_type="chain",
                    inputs={
                        "lambda": settings.MMR_LAMBDA,
                        "k": settings.MMR_K,
                        "max_per_parent": settings.MMR_MAX_PER_PARENT,
                    },
                    tags=["mmr", "diversification"],
                ) as mmr_trace:
                    embeddings_map = mmr_service.fetch_embeddings_for_chunks(session, chunk_ids)
                    
                    # Associer chaque nœud à son embedding
                    candidates = []
                    for nws in rerank_result.nodes:
                        chunk_id = _parse_chunk_id_from_node(nws.node)
                        embedding = embeddings_map.get(chunk_id, []) if chunk_id else []
                        candidates.append((nws, embedding))
                    
                    mmr_nodes = mmr_service.compute_mmr(
                        query_embedding,
                        candidates,
                        lambda_=settings.MMR_LAMBDA,
                        k=settings.MMR_K,
                        max_per_parent=settings.MMR_MAX_PER_PARENT,
                    )
                    mmr_trace.end(outputs={"selected_count": len(mmr_nodes)})
                
                passages = [
                    _node_to_passage(nws.node, fallback_score=score)
                    for nws, score in mmr_nodes
                ]
            else:
                # MMR désactivé : utiliser le résultat du rerank directement
                passages = [
                    _node_to_passage(nws.node, fallback_score=1.0 - i * 0.01)
                    for i, nws in enumerate(rerank_result.nodes)
                ]
            
            passages = apply_feedback_boost(passages, space_id, session)
            return {
                "passages": passages,
                "status": "ok",
                "reason": rerank_result.reason,
            }
        
        # Reranker désactivé : retour simple
        filtered_passages = filter_passages_by_rrf_score(
            passages,
            enabled=settings.RRF_DYNAMIC_K_ENABLED,
            min_k=settings.RRF_MIN_K,
            max_k=settings.RRF_MAX_K,
            factor=settings.RRF_RELATIVE_THRESHOLD_FACTOR,
        )
        return {"passages": filtered_passages, "status": "disabled", "reason": "reranker_disabled"}
        
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
    Utilise le pipeline complet : RRF → rerank → MMR → feedback boost.
    
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
    
    Pipeline léger : vector + BM25 → RRF → filtre score (pas de rerank/MMR coûteux).
    La requête est enrichie avec le brouillon pour capturer les erreurs concrètes.
    
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
        
        # Fusion RRF (léger, pas de filtrage vectoriel minimal ici car FAQ rares)
        fused_results = reciprocal_rank_fusion(raw_vector, raw_lexical, top_n=candidate_k)
        
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

