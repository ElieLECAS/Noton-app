"""
Recherche dans les espaces : retrieval hybride ColPali + pgvector L1 + BM25,
fusion RRF page-centric, expansion L1 + PNG multimodal pour le LLM.

Pipeline RAG (unique, page-centric) :
1. Retrieval ColPali + pgvector + BM25 (+ KAG) au niveau page
2. Fusion RRF + boost catégorie / ancrage conversationnel
3. Rerank MiniLM (si activé) + expansion L1 + voisinage conditionnel
4. Génération : texte consolidé + PNG pour toutes les pages top K
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
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
from app.services.query_understanding_graph import QueryGroup, RetrievalQueries
from app.services.query_signals_schemas import LightweightQuerySignals
from app.services.retrieval_boost_service import apply_category_boost_to_fused_hits
from app.tracing import trace_run
from app.services import reranker_service


# Marqueurs textuels qui signalent un besoin de recherche VISUELLE (schéma, plan, coupe,
# localisation) → ColPali garde toute sa valeur. Volontairement larges : mieux vaut lancer
# ColPali à tort (le fallback ne coûte rien) que le manquer sur une vraie question visuelle.
_COLPALI_VISUAL_MARKERS = (
    "schéma", "schema", "plan ", "plans", "dessin", "figure", "illustration",
    "image", "photo", "croquis", "vue", "coupe", "éclaté", "eclate", "diagramme",
    "où se trouve", "ou se trouve", "où est", "ou est", "où sont", "ou sont",
    "emplacement", "positionn", "situé", "situe", "localis", "visuel",
    "montre", "montrer", "à quoi ressemble", "a quoi ressemble", "repère", "repere",
)


def should_use_colpali(
    query_text: Optional[str],
    signals: Optional[LightweightQuerySignals],
) -> Tuple[bool, str]:
    """Décide si ColPali (retriever visuel coûteux) doit tourner pour cette requête.

    Retourne (use_colpali, raison) pour la télémétrie. ColPali tourne si :
      - un marqueur visuel figure dans le texte (schéma, plan, coupe, « où se trouve »…) ;
      - l'intent extrait fait partie de COLPALI_GATING_INTENTS (par défaut « installation »,
        où les schémas de pose sont déterminants).
    Sinon on l'écarte : BM25 + pgvector suffisent pour du texte, et un filet de sécurité en
    aval le relance si les retrievers texte reviennent faibles (aucun rappel perdu en silence).
    """
    if not settings.COLPALI_ENABLED:
        return False, "colpali_disabled"
    if not settings.COLPALI_GATING_ENABLED:
        return True, "gating_off"
    lowered = (query_text or "").lower()
    if any(marker in lowered for marker in _COLPALI_VISUAL_MARKERS):
        return True, "visual_marker"
    intent = (getattr(signals, "intent", None) or "").lower() if signals else ""
    if intent and intent in settings.colpali_gating_intents:
        return True, f"intent:{intent}"
    return False, "text_query_skipped"


async def _run_retrievers(
    session: Session,
    space_id: int,
    doc_ids: List[int],
    colpali_q: str,
    semantic_q: str,
    lexical_q: str,
    query_embedding: Optional[List[float]],
    pool_size: int,
    use_colpali: bool = True,
) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    """Exécute les 4 retrievers (ColPali/pgvector/BM25/KAG).

    En mode parallèle (``RETRIEVAL_PARALLEL_ENABLED``), chaque retriever tourne dans un
    thread avec sa PROPRE session DB (une ``Session`` SQLAlchemy n'est pas concurrente),
    ce qui recouvre l'encodage ColPali CPU avec les requêtes SQL des trois autres. Les
    hits retournés ne portent que des identifiants (document_id/page_no/chunk_id) et des
    scores : le texte est chargé plus tard avec la session principale, donc fermer les
    sessions des threads est sans risque. En repli séquentiel, tout passe par ``session``.
    """
    from app.services.page_retrieval_service import (
        filter_colpali_pages_dynamic,
        retrieve_bm25_pages,
        retrieve_colpali_pages,
        retrieve_pgvector_pages,
    )

    kag_enabled = settings.KAG_ENABLED

    def _colpali(s: Session) -> List[Any]:
        # Gate : requête texte → on n'exécute NI l'encode ColQwen2 NI le MaxSim (le poste
        # le plus lourd du pipeline sur CPU). Le fallback en aval relancera ColPali si les
        # retrievers texte sont faibles.
        if not use_colpali:
            return []
        return filter_colpali_pages_dynamic(retrieve_colpali_pages(s, doc_ids, colpali_q, pool_size))

    def _pgvector(s: Session) -> List[Any]:
        return retrieve_pgvector_pages(s, doc_ids, query_embedding or [], pool_size)

    def _bm25(s: Session) -> List[Any]:
        return retrieve_bm25_pages(s, doc_ids, lexical_q, pool_size)

    def _kag(s: Session) -> List[Any]:
        if not kag_enabled:
            return []
        from app.services.kag_retrieval_service import retrieve_kag_pages

        return retrieve_kag_pages(s, space_id, doc_ids, semantic_q, query_embedding, pool_size)

    if settings.RETRIEVAL_PARALLEL_ENABLED:
        from app.database import engine

        def _threaded(fn) -> List[Any]:
            with Session(engine) as own_session:
                return fn(own_session)

        colpali_hits, pgvector_hits, bm25_hits, kag_hits = await asyncio.gather(
            asyncio.to_thread(_threaded, _colpali),
            asyncio.to_thread(_threaded, _pgvector),
            asyncio.to_thread(_threaded, _bm25),
            asyncio.to_thread(_threaded, _kag),
        )
        return colpali_hits, pgvector_hits, bm25_hits, kag_hits

    return _colpali(session), _pgvector(session), _bm25(session), _kag(session)

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
        metadata["document_id"] = row.document_id
        metadata["document_title"] = row.document_title or "Document sans titre"
        metadata["chunk_index"] = row.chunk_index
        if getattr(row, "document_source", None):
            metadata["source"] = row.document_source
        if getattr(row, "chunk_source", None):
            metadata["source"] = row.chunk_source
            
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
            if meta.get("rerank_score") is not None:
                out["rerank_score"] = float(meta.get("rerank_score"))
                
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
            if meta.get("rerank_score") is not None:
                out["rerank_score"] = float(meta.get("rerank_score"))
                
            passages.append(out)
            
        # Limiter à k passages finaux
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
    meta["document_id"] = row.document_id
    meta["chunk_index"] = row.chunk_index
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


def _multimodal_eval_stages(
    *,
    colpali_hits: List[Any],
    pgvector_hits: List[Any],
    bm25_hits: List[Any],
    kag_hits: List[Any],
    pre_kag_fused_hits: List[Any],
    fused_hits: List[Any],
    top_k: int,
    reason: str,
    post_rerank_passages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Étapes d'éval RAG exposées à l'admin (avant/après KAG, ColPali, rerank)."""
    from app.services.page_retrieval_service import unified_hits_to_eval_passages

    stages: Dict[str, Any] = {
        "colpali_only": unified_hits_to_eval_passages(colpali_hits, top_k),
        "colpali": unified_hits_to_eval_passages(colpali_hits, top_k),
        "pgvector_only": unified_hits_to_eval_passages(pgvector_hits, top_k),
        "lexical_only": unified_hits_to_eval_passages(bm25_hits, top_k),
        "pre_kag_rrf": unified_hits_to_eval_passages(pre_kag_fused_hits, top_k),
        "kag_only": unified_hits_to_eval_passages(kag_hits, top_k) if kag_hits else [],
        "post_rrf": unified_hits_to_eval_passages(fused_hits, top_k),
        "minilm_rerank_enabled": settings.RERANKER_ENABLED,
        "vision_rerank_enabled": False,
        "kag_enabled": settings.KAG_ENABLED,
        "reason": reason,
    }
    if post_rerank_passages is not None:
        stages["post_rerank"] = post_rerank_passages
    return stages


def _retrieve_one_group_hits(
    session: Session,
    space_id: int,
    doc_ids: List[int],
    colpali_q: str,
    semantic_q: str,
    lexical_q: str,
    pool_size: int,
    *,
    group_index: int = 0,
    group_label: str = "",
) -> List[Any]:
    """Retrieval (colpali + pgvec + bm25 + kag) + fusion RRF pour un groupe de requêtes."""
    from app.services.page_retrieval_service import (
        filter_colpali_pages_dynamic,
        fuse_multimodal_hits,
        retrieve_bm25_pages,
        retrieve_colpali_pages,
        retrieve_pgvector_pages,
    )

    query_embedding: Optional[List[float]] = None
    try:
        query_embedding = generate_embedding(semantic_q)
    except Exception as exc:
        logger.warning("[RAG multi-group] Embedding indisponible (group=%r): %s", group_label, exc)

    colpali_hits = retrieve_colpali_pages(session, doc_ids, colpali_q, pool_size)
    colpali_hits = filter_colpali_pages_dynamic(colpali_hits)
    pgvector_hits = retrieve_pgvector_pages(session, doc_ids, query_embedding or [], pool_size)
    bm25_hits = retrieve_bm25_pages(session, doc_ids, lexical_q, pool_size)

    kag_hits: List[Any] = []
    if settings.KAG_ENABLED:
        from app.services.kag_retrieval_service import retrieve_kag_pages
        kag_hits = retrieve_kag_pages(
            session, space_id, doc_ids, semantic_q, query_embedding, pool_size
        )

    fused = fuse_multimodal_hits(
        colpali_hits,
        pgvector_hits,
        bm25_hits,
        kag_hits=kag_hits if settings.KAG_ENABLED else None,
        rrf_k=settings.RRF_K,
        top_k=pool_size,
    )

    for hit in fused:
        hit.query_group_index = group_index
        hit.query_group_label = group_label

    logger.info(
        "[RAG multi-group] group=%r — colpali=%d pgvec=%d bm25=%d kag=%d → fused=%d",
        group_label,
        len(colpali_hits),
        len(pgvector_hits),
        len(bm25_hits),
        len(kag_hits),
        len(fused),
    )
    return fused


async def search_multimodal_passages(
    session: Session,
    space_id: int,
    query_text: str,
    user_id: int,
    *,
    k: Optional[int] = None,
    document_filter: str = "all",
    include_retrieval_stages: bool = False,
    queries: Optional[RetrievalQueries] = None,
    signals: Optional[LightweightQuerySignals] = None,
    query_groups: Optional[List[QueryGroup]] = None,
    anchor_document_ids: Optional[List[int]] = None,
    allowed_document_ids: Optional[List[int]] = None,
) -> Dict:
    """
    Pipeline retrieval multimodal page-centric unifié.
    ColPali + pgvector + BM25 → RRF → expansion L1 → texte + PNG.

    ``allowed_document_ids`` (périmètre confirmé) : si fourni, restreint les documents
    candidats à cette liste (déjà calculée avec la politique wildcard côté résolveur).
    Un périmètre qui ne laisse aucun document → status ``no_results`` motif
    ``scope_empty`` : l'appelant peut alors élargir (retry sans périmètre).
    """
    from app.services.page_retrieval_service import (
        expand_page_context,
        filter_colpali_pages_dynamic,
        format_multimodal_passages,
        fuse_multimodal_hits,
        get_space_document_ids,
        log_multimodal_retrieval_summary,
        retrieve_bm25_pages,
        retrieve_colpali_pages,
        retrieve_pgvector_pages,
    )

    space = get_space_by_id(session, space_id, user_id)
    if not space:
        logger.warning("Espace %d inaccessible (user %d)", space_id, user_id)
        return {"passages": [], "images": [], "status": "disabled", "reason": "space_not_found"}
    if not query_text or not query_text.strip():
        return {"passages": [], "images": [], "status": "disabled", "reason": "empty_query"}

    colpali_q = queries.colpali if queries else query_text
    semantic_q = queries.semantic if queries else query_text
    lexical_q = queries.lexical if queries else query_text
    rerank_q = semantic_q

    top_k = k if k is not None else settings.RAG_TOP_K
    pool_size = max(settings.RERANK_POOL, settings.RAG_POOL_SIZE, top_k)

    logger.info(
        "[RAG multimodal] Démarrage — space_id=%s top_k=%s pool=%s rerank=%s filter=%s "
        "colpali=%r semantic=%r lexical=%r",
        space_id,
        top_k,
        pool_size,
        settings.RERANKER_ENABLED,
        document_filter,
        colpali_q[:80],
        semantic_q[:80],
        lexical_q[:80],
    )

    try:
        doc_ids = get_space_document_ids(
            session, space_id, document_filter
        )
        if not doc_ids:
            result: Dict = {"passages": [], "images": [], "status": "ok", "reason": "no_results"}
            if include_retrieval_stages:
                result["retrieval_stages"] = {"colpali": [], "post_rerank": [], "reason": "no_results"}
            return result

        # Périmètre de recherche confirmé : intersection avec les documents autorisés.
        # Point d'application UNIQUE — les 4 retrievers (mono/multi-groupe) reçoivent ensuite
        # ce doc_ids déjà scopé, donc ColPali/pgvector/BM25/KAG et le packing restent dans
        # le périmètre sans autre modification.
        if allowed_document_ids is not None:
            allowed_set = {int(d) for d in allowed_document_ids}
            scoped = [d for d in doc_ids if int(d) in allowed_set]
            logger.info(
                "[RAG multimodal] Périmètre confirmé — %d/%d documents retenus",
                len(scoped),
                len(doc_ids),
            )
            if not scoped:
                # Périmètre trop étroit : aucun document. L'appelant élargit (retry sans
                # périmètre) — jamais d'échec silencieux.
                return {"passages": [], "images": [], "status": "ok", "reason": "scope_empty"}
            doc_ids = scoped

        # --- Retrieval : mode multi-groupe ou mode unique ---
        active_groups = query_groups if (query_groups and len(query_groups) > 1) else None

        if active_groups:
            from app.services.page_retrieval_service import fuse_multi_query_groups

            logger.info(
                "[RAG multimodal] Mode multi-groupe — %d groupes : %s",
                len(active_groups),
                [g.label for g in active_groups],
            )
            per_group_hits = [
                _retrieve_one_group_hits(
                    session,
                    space_id,
                    doc_ids,
                    g.queries.colpali,
                    g.queries.semantic,
                    g.queries.lexical,
                    pool_size,
                    group_index=i,
                    group_label=g.label,
                )
                for i, g in enumerate(active_groups)
            ]
            fused_hits = fuse_multi_query_groups(per_group_hits, pool_size=pool_size)
            pre_kag_fused_hits = fused_hits
            # Pour le rerank, on utilise la requête sémantique du premier groupe
            rerank_q = active_groups[0].queries.semantic
            colpali_hits = per_group_hits[0] if per_group_hits else []
            pgvector_hits: List[Any] = []
            bm25_hits: List[Any] = []
            kag_hits: List[Any] = []
        else:
            query_embedding: Optional[List[float]] = None
            try:
                query_embedding = generate_embedding(semantic_q)
            except Exception as exc:
                logger.warning("[RAG multimodal] Embedding requête indisponible : %s", exc)

            # Gate ColPali : ne lancer le retriever visuel (coûteux sur CPU) que si la
            # requête en a besoin. Décidé AVANT le lancement pour économiser encode + MaxSim.
            use_colpali, gate_reason = should_use_colpali(query_text, signals)
            logger.info(
                "[RAG multimodal] ColPali gating — use_colpali=%s (%s) query=%r",
                use_colpali,
                gate_reason,
                (query_text or "")[:80],
            )

            with trace_run(
                "multimodal_retrieval",
                run_type="retriever",
                inputs={
                    "query": query_text,
                    "colpali_query": colpali_q,
                    "semantic_query": semantic_q,
                    "lexical_query": lexical_q,
                    "space_id": space_id,
                    "pool_size": pool_size,
                    "use_colpali": use_colpali,
                    "colpali_gate_reason": gate_reason,
                },
                tags=["retrieval", "multimodal", "space"],
            ) as hr:
                # 4 retrievers en parallèle (threads + sessions dédiées) ou en séquence,
                # selon RETRIEVAL_PARALLEL_ENABLED. Voir _run_retrievers.
                colpali_hits, pgvector_hits, bm25_hits, kag_hits = await _run_retrievers(
                    session,
                    space_id,
                    doc_ids,
                    colpali_q,
                    semantic_q,
                    lexical_q,
                    query_embedding,
                    pool_size,
                    use_colpali=use_colpali,
                )

                # Filet de sécurité : ColPali écarté mais retrievers texte trop faibles →
                # on le relance en rattrapage (dans un thread + session dédiée, comme le
                # mode parallèle) pour ne perdre aucun rappel en silence.
                if not use_colpali:
                    text_pages = {
                        (h.document_id, h.page_no) for h in pgvector_hits
                    } | {(h.document_id, h.page_no) for h in bm25_hits}
                    if len(text_pages) < settings.COLPALI_GATING_FALLBACK_MIN_HITS:
                        from app.database import engine
                        from app.services.page_retrieval_service import (
                            filter_colpali_pages_dynamic,
                            retrieve_colpali_pages,
                        )

                        def _fallback_colpali() -> List[Any]:
                            with Session(engine) as own_session:
                                return filter_colpali_pages_dynamic(
                                    retrieve_colpali_pages(own_session, doc_ids, colpali_q, pool_size)
                                )

                        # Un échec du rattrapage ne doit jamais casser la requête : on
                        # dégrade proprement vers les seuls hits texte déjà obtenus.
                        try:
                            colpali_hits = await asyncio.to_thread(_fallback_colpali)
                            use_colpali = True
                            gate_reason = "text_weak_fallback"
                            logger.info(
                                "[RAG multimodal] ColPali fallback — %d page(s) texte < seuil %d → "
                                "relance ColPali (%d page(s) récupérée(s))",
                                len(text_pages),
                                settings.COLPALI_GATING_FALLBACK_MIN_HITS,
                                len(colpali_hits),
                            )
                        except Exception as fb_exc:
                            logger.warning(
                                "[RAG multimodal] ColPali fallback échoué (%s) — on garde les hits texte",
                                fb_exc,
                            )

                hr.end(
                    outputs={
                        "colpali": len(colpali_hits),
                        "pgvector": len(pgvector_hits),
                        "bm25": len(bm25_hits),
                        "kag": len(kag_hits),
                        "colpali_gate_reason": gate_reason,
                    }
                )

            # ColPali / pgvector / BM25 loguent déjà leur résultat depuis
            # page_retrieval_service ; KAG n'avait aucune ligne → on l'ajoute pour que
            # TOUS les retrievers apparaissent dans les logs.
            logger.info(
                "[retrieve_kag_pages] %d pages — %s",
                len(kag_hits),
                ", ".join(f"doc={h.document_id} p.{h.page_no}" for h in kag_hits[:3])
                or "aucune",
            )

            logger.info(
                "[RAG multimodal] Retrievers — colpali=%d | pgvector=%d | bm25=%d | kag=%d",
                len(colpali_hits),
                len(pgvector_hits),
                len(bm25_hits),
                len(kag_hits),
            )

            pre_kag_fused_hits = fuse_multimodal_hits(
                colpali_hits,
                pgvector_hits,
                bm25_hits,
                rrf_k=settings.RRF_K,
                top_k=pool_size,
            )
            fused_hits = fuse_multimodal_hits(
                colpali_hits,
                pgvector_hits,
                bm25_hits,
                kag_hits=kag_hits if settings.KAG_ENABLED else None,
                rrf_k=settings.RRF_K,
                top_k=pool_size,
            )

        # Boost catégorie multiplicatif sur rrf_score (post-fusion, avant rerank)
        apply_category_boost_to_fused_hits(session, fused_hits, signals)

        # Ancrage conversation : booste les pages des documents du sujet courant AVANT la
        # coupe top_k, pour que la conversation reste sur le même produit d'un tour à l'autre.
        anchor_boost_debug = None
        if settings.CONVERSATION_ANCHOR_ENABLED and anchor_document_ids:
            from app.services.retrieval_boost_service import apply_anchor_boost_to_fused_hits

            anchor_boost_debug = apply_anchor_boost_to_fused_hits(fused_hits, anchor_document_ids)

        if not fused_hits:
            result = {"passages": [], "images": [], "status": "ok", "reason": "no_results", "dynamic_k": 0}
            if include_retrieval_stages:
                result["retrieval_stages"] = _multimodal_eval_stages(
                    colpali_hits=colpali_hits,
                    pgvector_hits=pgvector_hits,
                    bm25_hits=bm25_hits,
                    kag_hits=kag_hits,
                    pre_kag_fused_hits=pre_kag_fused_hits,
                    fused_hits=[],
                    top_k=top_k,
                    reason="no_results",
                    post_rerank_passages=[],
                )
            return result

        rerank_status = "disabled"
        dynamic_k = len(fused_hits)
        protected_hits: List[Any] = []
        final_hits = fused_hits

        if settings.RERANKER_ENABLED:
            from app.services.page_reranker_service import rerank_unified_page_hits

            _t_minilm = time.perf_counter()
            with trace_run(
                "minilm_rerank",
                run_type="reranker",
                inputs={"query": rerank_q, "pool_size": len(fused_hits), "max_k": top_k},
                tags=["rerank", "minilm", "page"],
            ) as rr:
                final_hits, rerank_result, protected_hits = await rerank_unified_page_hits(
                    session,
                    rerank_q,
                    fused_hits,
                    max_k=top_k,
                )
                rerank_status = rerank_result.status
                dynamic_k = len(final_hits)
                logger.info(
                    "[PERF][retrieval] rerank MiniLM %.2fs — pool=%d → %d",
                    time.perf_counter() - _t_minilm,
                    len(fused_hits),
                    dynamic_k,
                )
                rr.end(
                    outputs={
                        "status": rerank_status,
                        "dynamic_k": dynamic_k,
                        "protected": len(protected_hits),
                    }
                )
        else:
            # Coupe équitable (M1) : quota souple par document + slots ColPali réservés.
            # Sans reranker, la coupe brute [:top_k] laissait un gros document occuper
            # tous les slots (puis remporter l'élection CAG grâce à ce volume) et n'offrait
            # aucune protection aux pages visuelles — protect_colpali_visual_hits ne vit
            # que dans le chemin du reranker.
            from app.services.page_retrieval_service import select_final_hits

            final_hits, protected_hits = select_final_hits(fused_hits, top_k)
            dynamic_k = len(final_hits)

        if rerank_status == "low_confidence_clarification" and not protected_hits:
            passages: List[Dict[str, Any]] = []
            images: List[str] = []
            log_multimodal_retrieval_summary(
                query_text=query_text,
                doc_ids=doc_ids,
                colpali_hits=colpali_hits,
                pgvector_hits=pgvector_hits,
                bm25_hits=bm25_hits,
                kag_hits=kag_hits,
                pre_kag_fused_hits=pre_kag_fused_hits,
                fused_hits=fused_hits,
                final_hits=final_hits,
                passages=passages,
                images=images,
                top_k=top_k,
                pool_size=pool_size,
                rerank_enabled=settings.RERANKER_ENABLED,
                rerank_status=rerank_status,
                dynamic_k=0,
                protected_hits=protected_hits,
            )
            low_conf_result = {
                "passages": [],
                "images": [],
                "status": "low_confidence_clarification",
                "reason": rerank_status,
                "dynamic_k": len(final_hits),
                "rerank_status": rerank_status,
            }
            if include_retrieval_stages:
                low_conf_result["retrieval_stages"] = _multimodal_eval_stages(
                    colpali_hits=colpali_hits,
                    pgvector_hits=pgvector_hits,
                    bm25_hits=bm25_hits,
                    kag_hits=kag_hits,
                    pre_kag_fused_hits=pre_kag_fused_hits,
                    fused_hits=fused_hits,
                    top_k=top_k,
                    reason=rerank_status,
                    post_rerank_passages=[],
                )
            return low_conf_result

        if not final_hits:
            log_multimodal_retrieval_summary(
                query_text=query_text,
                doc_ids=doc_ids,
                colpali_hits=colpali_hits,
                pgvector_hits=pgvector_hits,
                bm25_hits=bm25_hits,
                kag_hits=kag_hits,
                pre_kag_fused_hits=pre_kag_fused_hits,
                fused_hits=fused_hits,
                final_hits=[],
                passages=[],
                images=[],
                top_k=top_k,
                pool_size=pool_size,
                rerank_enabled=settings.RERANKER_ENABLED,
                rerank_status=rerank_status,
                dynamic_k=0,
                protected_hits=protected_hits,
            )
            return {
                "passages": [],
                "images": [],
                "status": "ok",
                "reason": "no_results_after_rerank",
                "dynamic_k": 0,
                "rerank_status": rerank_status,
                **(
                    {
                        "retrieval_stages": _multimodal_eval_stages(
                            colpali_hits=colpali_hits,
                            pgvector_hits=pgvector_hits,
                            bm25_hits=bm25_hits,
                            kag_hits=kag_hits,
                            pre_kag_fused_hits=pre_kag_fused_hits,
                            fused_hits=fused_hits,
                            top_k=top_k,
                            reason="no_results_after_rerank",
                            post_rerank_passages=[],
                        )
                    }
                    if include_retrieval_stages
                    else {}
                ),
            }

        expanded_hits = expand_page_context(
            session,
            final_hits,
            neighbor_strategy=settings.RAG_NEIGHBOR_STRATEGY,
        )
        passages, images = format_multimodal_passages(
            session,
            expanded_hits,
            max_passage_chars=settings.SPACE_CONTEXT_MAX_PASSAGE_CHARS,
            render_all_images=settings.RAG_RENDER_ALL_IMAGES,
        )

        max_images = settings.RAG_MAX_IMAGES
        if len(images) > max_images:
            logger.warning(
                "[RAG multimodal] Troncature images: %d → %d",
                len(images),
                max_images,
            )
            images = images[:max_images]

        log_multimodal_retrieval_summary(
            query_text=query_text,
            doc_ids=doc_ids,
            colpali_hits=colpali_hits,
            pgvector_hits=pgvector_hits,
            bm25_hits=bm25_hits,
            kag_hits=kag_hits,
            pre_kag_fused_hits=pre_kag_fused_hits,
            fused_hits=fused_hits,
            final_hits=final_hits,
            passages=passages,
            images=images,
            top_k=top_k,
            pool_size=pool_size,
            rerank_enabled=settings.RERANKER_ENABLED,
            rerank_status=rerank_status,
            dynamic_k=dynamic_k,
            protected_hits=protected_hits,
        )

        reason = "multimodal_rrf_minilm" if settings.RERANKER_ENABLED else "multimodal_rrf"
        result = {
            "passages": passages,
            "images": images,
            "status": "ok",
            "reason": reason,
            "total_hits": len(final_hits),
            "dynamic_k": dynamic_k,
            "rerank_status": rerank_status,
            # Détail du boost d'ancre (None si aucun) : rend visible dans le cheminement
            # qu'un tour a été biaisé par le sujet courant.
            "anchor_boost": anchor_boost_debug,
            # Top-5 du classement final, pour comparer les deux retrievals quand un retry
            # a lieu (le retry REMPLACE les passages : sans ça, la substitution est invisible).
            "top_hits": [
                {
                    "document_id": int(h.document_id),
                    "page_no": int(h.page_no),
                    "score": round(float(h.rrf_score or 0.0), 4),
                    "sources": list(h.retrieval_sources or []),
                }
                for h in final_hits[:5]
            ],
        }
        if include_retrieval_stages:
            result["retrieval_stages"] = _multimodal_eval_stages(
                colpali_hits=colpali_hits,
                pgvector_hits=pgvector_hits,
                bm25_hits=bm25_hits,
                kag_hits=kag_hits,
                pre_kag_fused_hits=pre_kag_fused_hits,
                fused_hits=fused_hits,
                top_k=top_k,
                reason=reason,
                post_rerank_passages=passages,
            )
        return result

    except Exception as exc:
        logger.error("search_multimodal_passages (space): %s", exc, exc_info=True)
        return {
            "passages": [],
            "images": [],
            "status": "disabled",
            "reason": f"error: {str(exc)}",
        }


async def search_relevant_passages(
    session: Session,
    space_id: int,
    query_text: str,
    user_id: int,
    k: int = 15,
    document_filter: str = "all",
    include_retrieval_stages: bool = False,
    queries: Optional[RetrievalQueries] = None,
    signals: Optional[LightweightQuerySignals] = None,
    query_groups: Optional[List[QueryGroup]] = None,
    anchor_document_ids: Optional[List[int]] = None,
    allowed_document_ids: Optional[List[int]] = None,
) -> Dict:
    """
    RAG espace : délègue au pipeline multimodal page-centric unifié
    (ColPali + pgvector + BM25 + KAG, fusion RRF, boost catégorie, rerank MiniLM,
    expansion page entière + voisinage conditionnel, PNG conditionnel).
    """
    return await search_multimodal_passages(
        session=session,
        space_id=space_id,
        query_text=query_text,
        user_id=user_id,
        k=k,
        document_filter=document_filter,
        include_retrieval_stages=include_retrieval_stages,
        queries=queries,
        anchor_document_ids=anchor_document_ids,
        signals=signals,
        query_groups=query_groups,
        allowed_document_ids=allowed_document_ids,
    )


async def search_technical_passages(
    session: Session,
    space_id: int,
    query_text: str,
    user_id: int,
    k: int = 15,
    queries: Optional[RetrievalQueries] = None,
    signals: Optional[LightweightQuerySignals] = None,
    query_groups: Optional[List[QueryGroup]] = None,
    anchor_document_ids: Optional[List[int]] = None,
    allowed_document_ids: Optional[List[int]] = None,
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
        queries=queries,
        signals=signals,
        query_groups=query_groups,
        anchor_document_ids=anchor_document_ids,
        allowed_document_ids=allowed_document_ids,
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

    DÉPRÉCIÉ (2026-07-20) : plus appelé sur le chemin de génération. L'autorité de
    source est portée par apply_soft_boosts_to_passages (retrieval_boost_service),
    proportionnelle à l'étendue des scores. Appeler les deux = double-comptage.
    Conservé pour compatibilité ; ne pas rebrancher sans retirer le boost amont.
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


def reciprocal_rank_fusion(
    vector_results: List[NodeWithScore],
    lexical_results: List[NodeWithScore],
    alphanumeric_results: Optional[List[NodeWithScore]] = None,
    k: int = 60,
    top_n: int = 15,
    normalize: bool = False,
) -> List[NodeWithScore]:
    """
    RRF (Reciprocal Rank Fusion) unifié pour fusionner les canaux vectoriel, lexical et alphanumérique.
    """
    id_to_node = {}
    ranks: Dict[str, Dict[str, int]] = {}

    def _add_results(results: List[NodeWithScore], channel_name: str):
        if not results:
            return
        for rank_idx, nws in enumerate(results, start=1):
            node_id = nws.node.id_
            id_to_node[node_id] = nws.node
            ranks.setdefault(node_id, {})[channel_name] = rank_idx

    _add_results(vector_results, "vector")
    _add_results(lexical_results, "lexical")
    if alphanumeric_results:
        _add_results(alphanumeric_results, "alphanumeric")

    if not id_to_node:
        return []

    fused_results = []
    for node_id, node in id_to_node.items():
        rrf_score = 0.0
        for channel_name, rank_idx in ranks[node_id].items():
            rrf_score += 1.0 / (k + rank_idx)

        meta = dict(node.metadata or {})
        meta["raw_rrf_score"] = rrf_score
        node.metadata = meta

        fused_results.append(NodeWithScore(node=node, score=rrf_score))

    fused_results.sort(key=lambda x: x.score, reverse=True)
    fused_results = fused_results[:top_n]

    if normalize and fused_results:
        scores = [x.score for x in fused_results]
        min_score = min(scores)
        max_score = max(scores)

        for nws in fused_results:
            if max_score > min_score:
                norm = 0.1 + 0.8 * ((nws.score - min_score) / (max_score - min_score))
            else:
                norm = 0.9
            nws.score = norm

    return fused_results


def _extract_alphanumeric_codes(query: str) -> List[str]:
    """Extrait les codes alphanumériques d'une requête."""
    words = re.findall(r'[a-zA-Z0-9.\-]+', query.lower())
    codes = []
    for w in words:
        w_clean = w.strip(".,;:!?()")
        if not w_clean or w_clean in _FALLBACK_STOPWORDS:
            continue
        if any(c.isdigit() for c in w_clean) or len(w_clean) >= 3:
            if w_clean not in codes:
                codes.append(w_clean)
    return codes


def _retrieve_leaves_bm25_sql(*args, **kwargs):
    """BM25 tsvector — implémenté dans page_retrieval_service."""
    from app.services.page_retrieval_service import _retrieve_leaves_bm25_sql as _bm25_impl
    return _bm25_impl(*args, **kwargs)


def _retrieve_leaves_alphanumeric_sql(*args, **kwargs):
    """Stub pour compatibilité de tests."""
    return []
