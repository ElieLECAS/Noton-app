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
    Recherche vectorielle LanceDB/ColPali sur les feuilles.
    
    Args:
        query_embedding: Ignoré (conservé pour compatibilité de signature)
        document_filter: "all" (tous), "technical" (exclut FAQ), "faq_corrective" (FAQ uniquement)
    """
    logger.info(
        "[_retrieve_leaves_sql] Starting retrieval for space_id=%s, user_id=%s, query_text='%s', candidate_k=%d, document_filter='%s'",
        space_id,
        user_id,
        query_text,
        candidate_k,
        document_filter,
    )
    from app.services.document_service_new import feedback_corrective_sql_filter

    filter_clause = feedback_corrective_sql_filter(document_filter, "d")

    # 1. Récupérer la liste des document_ids appartenant à cette space_id et satisfaisant le filter_clause
    sql_docs = text(f"""
        SELECT DISTINCT d.id
        FROM document d
        INNER JOIN document_space ds ON ds.document_id = d.id
        WHERE ds.space_id = :space_id
          {filter_clause}
    """)
    doc_ids = [row[0] for row in session.execute(sql_docs, {"space_id": space_id})]
    logger.info(
        "[_retrieve_leaves_sql] Resolved document IDs in space %s: %s",
        space_id,
        doc_ids,
    )
    if not doc_ids:
        logger.info("LanceDB (space): aucun document correspondant au filtre dans l'espace %s", space_id)
        return []

    # 2. Rechercher dans LanceDB avec ColPali
    from app.services.colpali_service import embed_query_colpali
    from app.services.lancedb_service import search_colpali_lancedb
    
    logger.info("[_retrieve_leaves_sql] Generating ColPali query token embeddings...")
    query_token_embeddings = embed_query_colpali(query_text)
    logger.info(
        "[_retrieve_leaves_sql] Generated %d query token embeddings for ColPali. Querying LanceDB...",
        len(query_token_embeddings) if query_token_embeddings else 0,
    )
    search_results = search_colpali_lancedb(query_token_embeddings, doc_ids, limit=candidate_k)
    logger.info(
        "[_retrieve_leaves_sql] LanceDB search returned %d raw colpali patch matches.",
        len(search_results),
    )
        
    if not search_results:
        logger.info("[_retrieve_leaves_sql] No search results returned from LanceDB.")
        return []

    # 3. Récupérer les données textuelles complètes et métadonnées depuis PostgreSQL pour les chunks trouvés
    chunk_ids = [row["id"] for row in search_results]
    logger.info(
        "[_retrieve_leaves_sql] Fetching details from PostgreSQL for chunk IDs: %s",
        chunk_ids,
    )
    sql_chunks = text("""
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
        WHERE dc.id IN :chunk_ids
    """)
    chunk_rows = session.execute(sql_chunks, {"chunk_ids": tuple(chunk_ids)}).all()
    logger.info(
        "[_retrieve_leaves_sql] PostgreSQL returned %d rows for chunk details.",
        len(chunk_rows),
    )

    # Conserver l'ordre trié retourné par LanceDB
    rows_map = {row.id: row for row in chunk_rows}
    nodes: List[NodeWithScore] = []
    
    for row_lancedb in search_results:
        chunk_id = row_lancedb["id"]
        row = rows_map.get(chunk_id)
        if not row:
            logger.warning(
                "[_retrieve_leaves_sql] Chunk ID %d found in LanceDB but not found in PostgreSQL!",
                chunk_id,
            )
            continue
        
        # LanceDB retourne '_distance' en tant que distance cosinus (1 - cosine_similarity)
        # Donc similarity_score = 1.0 - _distance
        distance = float(row_lancedb.get("_distance", 1.0))
        similarity_score = 1.0 - distance
        
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
        nodes.append(NodeWithScore(node=node, score=similarity_score))
        
    logger.info(
        "[_retrieve_leaves_sql] Vector LanceDB/ColPali (space): %d feuilles (limit=%d)",
        len(nodes),
        candidate_k,
    )
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


def _augment_and_format_passages(
    session: Session,
    nodes: List[NodeWithScore],
    k: int,
) -> List[Dict]:
    """
    Augmente et formate les passages après reranking / sélection finale.
    Regroupe et déduplique par window_id (pour la v4) ou par (document_id, chunk_index) pour le reste.
    Charge tout le contexte de la page/fenêtre ou les chunks adjacents pour ne pas tronquer l'information.
    """
    passages: List[Dict] = []
    seen_windows = set()
    seen_chunk_ids = set()

    for nws in nodes:
        node = nws.node
        score = nws.score
        meta = dict(node.metadata or {})
        
        doc_id = meta.get("document_id")
        wid = meta.get("window_id")
        chunk_id = _parse_chunk_id_from_node(node)
        
        # 1. Cas Multimodal v4 (avec window_id)
        if wid and doc_id is not None:
            if wid in seen_windows:
                continue
            seen_windows.add(wid)
            
            # Récupérer tous les chunks de la même fenêtre
            stmt = select(DocumentChunk).where(
                DocumentChunk.document_id == doc_id,
                text("(coalesce(metadata_json->>'window_id', metadata_->>'window_id')) = :window_id")
            )
            window_chunks = session.execute(stmt, {"window_id": wid}).scalars().all()
            
            # Séparer reports et raw text
            report_contents = []
            raw_contents = []
            
            # Trier pour conserver l'ordre de lecture
            sorted_chunks = sorted(window_chunks, key=lambda c: (c.chunk_index or 0, c.id or 0))
            
            doc_title = meta.get("document_title") or "Document sans titre"
            page_start = meta.get("page_start") or meta.get("page_no") or 0
            page_end = meta.get("page_end") or page_start
            
            for chunk in sorted_chunks:
                chunk_meta = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
                ctype = chunk_meta.get("content_type")
                content_text = (chunk.content or chunk.text or "").strip()
                if not content_text:
                    continue
                
                if ctype == "page_window_report":
                    report_contents.append(content_text)
                else:
                    raw_contents.append(content_text)
            
            # Si pas de raw trouvé, utiliser le contenu du nœud actuel comme fallback
            if not raw_contents:
                raw_contents.append((node.get_content() if hasattr(node, "get_content") else str(node)).strip())
                
            joined_reports = "\n\n".join(report_contents).strip()
            joined_raw = "\n\n".join(raw_contents).strip()
            
            # Formatage propre de liaison
            parts = []
            if joined_reports:
                parts.append("--- RAPPORT DE SYNTHÈSE DE LA FENÊTRE ---")
                parts.append(joined_reports)
            parts.append("--- TEXTE BRUT DU DOCUMENT ---")
            parts.append(joined_raw)
            
            augmented_content = "\n\n".join(parts)
            passage_text = f"**{doc_title}**\n{augmented_content}"
            
            out = {
                "passage": passage_text,
                "passage_raw": joined_raw,
                "document_title": doc_title,
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": int(meta.get("chunk_index", 0)),
                "score": float(score),
                "page_no": page_start,
                "page_start": page_start,
                "page_end": page_end,
                "section": meta.get("parent_heading") or meta.get("heading"),
                "source": meta.get("source"),
                "content_type": "augmented_multimodal_window",
            }
            if meta.get("row_index") is not None:
                out["row_index"] = meta.get("row_index")
            if meta.get("table_id"):
                out["table_id"] = meta.get("table_id")
            raw_rrf = meta.get("raw_rrf_score")
            if raw_rrf is not None:
                out["raw_rrf_score"] = float(raw_rrf)
                
            passages.append(out)
            
        # 2. Cas classique / Legacy / FAQ (sans window_id)
        else:
            if chunk_id is not None:
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
            
            # Essayer d'augmenter avec les chunks adjacents (+/- 1) du même document
            chunk_index = meta.get("chunk_index")
            doc_title = meta.get("document_title") or "Document sans titre"
            node_text = (node.get_content() if hasattr(node, "get_content") else str(node)).strip()
            
            if doc_id is not None and chunk_index is not None:
                # Récupérer chunk_index - 1, chunk_index, chunk_index + 1
                stmt = select(DocumentChunk).where(
                    DocumentChunk.document_id == doc_id,
                    DocumentChunk.chunk_index.in_([chunk_index - 1, chunk_index, chunk_index + 1])
                )
                adj_chunks = session.execute(stmt).scalars().all()
                sorted_adj = sorted(adj_chunks, key=lambda c: c.chunk_index)
                
                joined_text = "\n\n".join([(c.content or c.text or "").strip() for c in sorted_adj if (c.content or c.text or "").strip()])
                if not joined_text:
                    joined_text = node_text
            else:
                joined_text = node_text
                
            passage_text = f"**{doc_title}**\n{joined_text}"
            
            out = {
                "passage": passage_text,
                "passage_raw": node_text,
                "document_title": doc_title,
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": int(chunk_index or 0),
                "score": float(score),
                "page_no": meta.get("page_no") or meta.get("page_start"),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "section": meta.get("parent_heading") or meta.get("heading"),
                "source": meta.get("source"),
                "content_type": meta.get("content_type", "augmented_legacy_sliding_window"),
            }
            if meta.get("row_index") is not None:
                out["row_index"] = meta.get("row_index")
            if meta.get("table_id"):
                out["table_id"] = meta.get("table_id")
            raw_rrf = meta.get("raw_rrf_score")
            if raw_rrf is not None:
                out["raw_rrf_score"] = float(raw_rrf)
                
            passages.append(out)
            
        # Limiter à k passages finaux
        if len(passages) >= k:
            break
            
    return passages


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
    RAG espace : recherche ColPali-only via LanceDB.
    
    Args:
        document_filter: "all" (tous), "technical" (exclut FAQ), "faq_corrective" (FAQ uniquement)
    
    Returns:
        Dict avec clés : passages (List[Dict]), status (str), reason (Optional[str])
    """
    space = get_space_by_id(session, space_id, user_id)
    if not space:
        logger.warning("Espace %d inaccessible (user %d)", space_id, user_id)
        return {"passages": [], "status": "disabled", "reason": "space_not_found"}
    if not query_text or not query_text.strip():
        return {"passages": [], "status": "disabled", "reason": "empty_query"}

    try:
        # Augmentation du nombre de candidats pour le reranker
        candidate_k = max(100, k * 6)

        # 1. Recherche vectorielle ColPali (LanceDB)
        with trace_run(
            "vector_retrieval",
            run_type="retriever",
            inputs={"query": query_text, "space_id": space_id, "candidate_k": candidate_k},
            tags=["retrieval", "colpali", "space"],
        ) as vr:
            final_nodes = _retrieve_leaves_sql(
                session, space_id, user_id, query_text, candidate_k, document_filter=document_filter
            )
            vr.end(outputs={"nb": len(final_nodes)})

        if not final_nodes:
            return {"passages": [], "status": "ok", "reason": "no_results"}

        # === RERANKER ===
        if settings.RERANKER_ENABLED:
            pool_size = min(max(50, settings.RERANK_POOL), len(final_nodes))
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
                passages_low_conf = _augment_and_format_passages(
                    session,
                    rerank_result.nodes,
                    len(rerank_result.nodes)
                )
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
            
            # Utiliser le résultat du rerank avec augmentation de contexte
            passages = _augment_and_format_passages(
                session,
                rerank_result.nodes,
                k
            )
            return {
                "passages": passages,
                "status": "ok",
                "reason": rerank_result.reason,
            }
        
        # Reranker désactivé : retour simple des top-K ColPali augmentés
        passages = _augment_and_format_passages(session, final_nodes, k)
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
    Désactivée en dur.
    """
    return {"passages": [], "status": "disabled", "reason": "faq_post_draft_disabled"}


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
