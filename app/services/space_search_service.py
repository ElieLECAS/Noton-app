"""
Service de recherche sémantique pour les espaces (architecture Document/Space).

Aligné sur semantic_search_service.py (projets) :
  1. Embedding de la requête via BGE-m3
  2. Recherche vectorielle SQL (pgvector) sur les DocumentChunk LEAF
  3. Enrichissement KAG : graphe d'entités uniquement, fusion vectoriel + graphe
  4. Filtrage pré-reranking (similarité minimale)
  5. Early stopping si similarité déjà élevée
  6. Reranking cross-encoder (BGE-reranker-v2-m3)
  7. Résolution des parents pour contexte enrichi au LLM
  8. Source authority (boost titre-requête)
  9. Fallback lexical si besoin
"""

from typing import Dict, List, Optional, Set
import os
import re
import unicodedata
from sqlmodel import Session, select
from sqlalchemy import or_, text
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.models.knowledge_entity import KnowledgeEntity
from app.models.chunk_entity_relation import ChunkEntityRelation
from app.models.space import Space
from app.services.space_service import get_space_by_id
from app.config import settings
import logging
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

try:
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("FlagEmbeddingReranker non disponible, reranking désactivé")

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_CANDIDATE_MULTIPLIER = 3
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
MIN_VECTOR_SIMILARITY_THRESHOLD = float(os.getenv("MIN_VECTOR_SIMILARITY", "0.25"))
MAX_RERANK_CANDIDATES = int(os.getenv("MAX_RERANK_CANDIDATES", "50"))
SKIP_RERANK_THRESHOLD = float(os.getenv("SKIP_RERANK_THRESHOLD", "0.85"))

# Singletons
_reranker_instance = None
_embed_model_instance = None

_FALLBACK_STOPWORDS = {
    "the", "and", "for", "with", "dans", "avec", "pour", "une", "des", "les",
    "est", "sur", "pas", "plus", "que", "qui", "this", "that", "what", "how",
    "quoi", "comment", "quel", "quelle", "quels", "quelles", "from", "par",
    "sans", "mais", "donc", "car", "you", "your", "not", "are", "was", "were",
}

TITLE_QUERY_BOOST_PER_MATCH = float(os.getenv("TITLE_QUERY_BOOST_PER_MATCH", "0.5"))
TITLE_QUERY_BOOST_CAP = float(os.getenv("TITLE_QUERY_BOOST_CAP", "2.0"))


def _get_reranker():
    """Retourne le reranker (singleton)."""
    global _reranker_instance
    if not RERANKER_AVAILABLE:
        return None
    if _reranker_instance is None:
        device = os.getenv("RERANKER_DEVICE", "cpu")
        logger.info("Initialisation reranker %s sur %s...", RERANKER_MODEL, device)
        _reranker_instance = FlagEmbeddingReranker(
            model=RERANKER_MODEL,
            top_n=None,
            device=device,
        )
        logger.info("✅ Reranker initialisé (device=%s)", device)
    return _reranker_instance


def _get_embed_model() -> HuggingFaceEmbedding:
    """Retourne le modèle d'embedding BGE-m3 (singleton)."""
    global _embed_model_instance
    if _embed_model_instance is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        device = os.getenv("EMBEDDING_DEVICE", "cpu")
        logger.info("Initialisation embedding %s sur %s...", model_name, device)
        _embed_model_instance = HuggingFaceEmbedding(
            model_name=model_name,
            device=device,
            embed_batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
        logger.info("✅ Modèle d'embedding initialisé")
    return _embed_model_instance


def _retrieve_leaves_sql(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    candidate_k: int,
) -> List[NodeWithScore]:
    """
    Recherche vectorielle SQL directe sur les DocumentChunk LEAF via pgvector.
    """
    embed_model = _get_embed_model()
    query_embedding = embed_model.get_query_embedding(query_text)
    query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    sql_query = text(f"""
        SELECT
            dc.id,
            dc.content,
            dc.text,
            dc.chunk_index,
            dc.document_id,
            dc.metadata_json,
            dc.metadata_,
            d.title AS document_title,
            d.id AS document_id,
            1 - (dc.embedding <=> '{query_embedding_str}'::vector) AS similarity_score
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        INNER JOIN document_space ds ON ds.document_id = d.id
        WHERE ds.space_id = :space_id
          AND dc.embedding IS NOT NULL
          AND dc.is_leaf = true
        ORDER BY dc.embedding <=> '{query_embedding_str}'::vector
        LIMIT :limit_k
    """)

    result = session.execute(
        sql_query,
        {"space_id": space_id, "limit_k": candidate_k},
    )

    nodes_with_scores: List[NodeWithScore] = []
    for row in result:
        metadata = dict(row.metadata_json or row.metadata_ or {})
        metadata.setdefault("document_id", row.document_id)
        metadata.setdefault("document_title", row.document_title or "Document sans titre")
        metadata.setdefault("chunk_index", row.chunk_index)

        node = TextNode(
            id_=f"chunk-{row.id}",
            text=row.content or row.text or "",
            metadata=metadata,
        )
        nodes_with_scores.append(
            NodeWithScore(node=node, score=float(row.similarity_score))
        )

    logger.info(
        "Recherche SQL pgvector (space): %d nœuds leaf récupérés (candidate_k=%d)",
        len(nodes_with_scores),
        candidate_k,
    )
    return nodes_with_scores


def _build_parent_node_dict(
    session: Session, space_id: int, user_id: int
) -> Dict[str, TextNode]:
    """
    Charge les nœuds parents (DocumentChunk is_leaf=False) pour enrichir le contexte.
    """
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
        metadata = dict(chunk.metadata_json or {})
        metadata.setdefault("document_id", chunk.document_id)
        metadata.setdefault("document_title", document_title or "Document sans titre")
        metadata.setdefault("chunk_index", chunk.chunk_index)
        node_id = f"chunk-{chunk.id}"
        node_dict[node_id] = TextNode(
            id_=node_id,
            text=chunk.content or chunk.text or "",
            metadata=metadata,
        )

    logger.info(
        "Chargé %d nœuds parents pour space_id=%d",
        len(node_dict),
        space_id,
    )
    return node_dict


def _extract_query_terms(query_text: str) -> List[str]:
    terms = [t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ0-9]+", query_text or "")]
    return [t for t in terms if len(t) >= 3 and t not in _FALLBACK_STOPWORDS][:8]


def _normalize_pivot_entities(pivot_entity_names: Optional[List[str]]) -> List[str]:
    """Normalise/déduplique les entités pivot extraites côté requête."""
    if not pivot_entity_names:
        return []
    seen: Set[str] = set()
    normalized: List[str] = []
    for name in pivot_entity_names:
        value = (name or "").strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
        if len(normalized) >= 12:
            break
    return normalized


def _keyword_fallback_passages(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    k: int,
) -> List[Dict]:
    """Fallback lexical si aucun embedding disponible."""
    terms = _extract_query_terms(query_text)
    base_stmt = (
        select(DocumentChunk, Document.title)
        .join(Document, Document.id == DocumentChunk.document_id)
        .join(DocumentSpace, DocumentSpace.document_id == Document.id)
        .where(
            DocumentSpace.space_id == space_id,
        )
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

    passages: List[Dict] = []
    seen_chunk_ids: set = set()
    for chunk, document_title in rows:
        if chunk.id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.id)

        content = (chunk.content or chunk.text or "").strip()
        if not content:
            continue

        lowered = content.lower()
        match_count = sum(1 for term in terms if term in lowered) if terms else 0
        score = (match_count / max(len(terms), 1)) if terms else 0.05

        node_metadata = dict(chunk.metadata_json or {})
        node_metadata.setdefault("document_id", chunk.document_id)
        node_metadata.setdefault("document_title", document_title or "Document sans titre")
        node_metadata.setdefault("chunk_index", chunk.chunk_index)
        node = TextNode(
            id_=f"chunk-{chunk.id}",
            text=content,
            metadata=node_metadata,
        )
        passages.append(_node_to_passage(node, fallback_score=score))
        if len(passages) >= k:
            break

    logger.info(
        "Fallback lexical (space): %d passages construits (terms=%s)",
        len(passages),
        terms,
    )
    return passages


def _enrich_content_with_heading_and_figure(content: str, metadata: dict) -> str:
    """Préfixe le contenu avec parent_heading et figure_title."""
    parent_heading = metadata.get("parent_heading") or metadata.get("heading")
    figure_title = metadata.get("figure_title") or metadata.get("image_anchor")
    parts = []
    if parent_heading and str(parent_heading).strip():
        parts.append(f"[Section: {parent_heading.strip()}]")
    if figure_title and str(figure_title).strip():
        parts.append(str(figure_title).strip())
    if not parts:
        return content
    prefix = " ".join(parts) + "\n\n"
    return prefix + content if content else prefix.strip()


def _node_to_passage(node, fallback_score: float = 0.0) -> Dict:
    metadata = dict(getattr(node, "metadata", {}) or {})
    document_title = metadata.get("document_title", "Document sans titre")
    document_id = metadata.get("document_id")
    chunk_index = metadata.get("chunk_index", 0)
    page_no = metadata.get("page_no")
    parent_heading = metadata.get("parent_heading")

    content = node.get_content() if hasattr(node, "get_content") else str(node)
    content_enriched = _enrich_content_with_heading_and_figure(content, metadata)
    passage_text = f"**{document_title}**\n{content_enriched}"
    return {
        "passage": passage_text,
        "passage_raw": content,
        "document_title": document_title,
        "document_id": document_id,
        "chunk_index": int(chunk_index) if isinstance(chunk_index, (int, str)) else 0,
        "score": float(fallback_score or 0.0),
        "page_no": int(page_no) if page_no is not None else None,
        "section": parent_heading,
    }


def _filter_low_similarity_candidates(
    candidates: List[NodeWithScore],
    min_threshold: float = MIN_VECTOR_SIMILARITY_THRESHOLD,
) -> List[NodeWithScore]:
    """Filtre les candidats avec faible similarité vectorielle."""
    filtered = [c for c in candidates if float(c.score or 0.0) >= min_threshold]
    if len(filtered) < len(candidates):
        logger.debug(
            "Filtrage similarité (space): %d → %d candidats (seuil=%.2f)",
            len(candidates),
            len(filtered),
            min_threshold,
        )
    return filtered


def _retrieve_via_knowledge_graph(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    limit: int = 10,
    pivot_entity_names: Optional[List[str]] = None,
) -> List[NodeWithScore]:
    """
    Récupère des chunks via le graphe de connaissances KAG de l'espace.
    
    Args:
        pivot_entity_names: Entités normalisées extraites de la requête par LLM (prioritaires)
    """
    try:
        from app.services.kag_extraction_service import normalize_entity_name

        # Stratégie 1: utiliser les entités pivot LLM si disponibles
        normalized_pivots = _normalize_pivot_entities(pivot_entity_names)
        if normalized_pivots:
            query_terms = normalized_pivots
            logger.debug(
                "KAG retrieval (space): utilisation de %d entités pivot LLM", 
                len(query_terms)
            )
        else:
            # Fallback: split naïf de la requête
            query_terms = [t.strip().lower() for t in re.findall(r"[A-Za-zÀ-ÿ0-9]+", query_text)]
            query_terms = [t for t in query_terms if len(t) >= 3 and t not in _FALLBACK_STOPWORDS]

        if not query_terms:
            return []

        # Matching exact sur name_normalized
        stmt = (
            select(
                DocumentChunk,
                Document.title,
                KnowledgeEntity.name,
                ChunkEntityRelation.relevance_score,
            )
            .join(ChunkEntityRelation, ChunkEntityRelation.chunk_id == DocumentChunk.id)
            .join(KnowledgeEntity, KnowledgeEntity.id == ChunkEntityRelation.entity_id)
            .join(Document, Document.id == DocumentChunk.document_id)
            .join(DocumentSpace, DocumentSpace.document_id == Document.id)
            .where(
                DocumentSpace.space_id == space_id,
                DocumentChunk.is_leaf == True,
                KnowledgeEntity.space_id == space_id,
                KnowledgeEntity.name_normalized.in_(query_terms),
            )
            .order_by(ChunkEntityRelation.relevance_score.desc())
            .limit(limit)
        )
        results = list(session.exec(stmt).all())

        # Fallback ILIKE partiel si peu de résultats exacts
        if len(results) < limit // 2 and query_terms:
            logger.debug(
                "KAG retrieval (space): fallback ILIKE (résultats exacts=%d)", 
                len(results)
            )
            # Limiter aux 5 premiers termes pour éviter des requêtes trop larges
            ilike_terms = query_terms[:5]
            ilike_conditions = [
                KnowledgeEntity.name_normalized.ilike(f"%{term}%") 
                for term in ilike_terms
            ]
            stmt_ilike = (
                select(
                    DocumentChunk,
                    Document.title,
                    KnowledgeEntity.name,
                    ChunkEntityRelation.relevance_score,
                )
                .join(ChunkEntityRelation, ChunkEntityRelation.chunk_id == DocumentChunk.id)
                .join(KnowledgeEntity, KnowledgeEntity.id == ChunkEntityRelation.entity_id)
                .join(Document, Document.id == DocumentChunk.document_id)
                .join(DocumentSpace, DocumentSpace.document_id == Document.id)
                .where(
                    DocumentSpace.space_id == space_id,
                    DocumentChunk.is_leaf == True,
                    KnowledgeEntity.space_id == space_id,
                    or_(*ilike_conditions),
                )
                .order_by(ChunkEntityRelation.relevance_score.desc())
                .limit(limit - len(results))
            )
            ilike_results = list(session.exec(stmt_ilike).all())
            # Déduplication par chunk.id
            existing_chunk_ids = {row[0].id for row in results}
            for row in ilike_results:
                if row[0].id not in existing_chunk_ids:
                    results.append(row)
                    existing_chunk_ids.add(row[0].id)

        nodes_with_scores: List[NodeWithScore] = []
        for chunk, document_title, entity_name, relevance in results:
            metadata = dict(chunk.metadata_json or {})
            metadata.setdefault("document_id", chunk.document_id)
            metadata.setdefault("document_title", document_title or "Document sans titre")
            metadata.setdefault("chunk_index", chunk.chunk_index)
            metadata["kag_matched_entity"] = entity_name

            node = TextNode(
                id_=f"chunk-{chunk.id}",
                text=chunk.content or chunk.text or "",
                metadata=metadata,
            )
            nodes_with_scores.append(NodeWithScore(node=node, score=float(relevance)))

        logger.debug(
            "KAG retrieval (space): %d chunks via graphe (query_terms=%s)",
            len(nodes_with_scores),
            query_terms[:5],
        )
        return nodes_with_scores

    except Exception as e:
        logger.warning("Erreur KAG retrieval (space): %s", e)
        return []


def _merge_with_graph_candidates(
    vector_candidates: List[NodeWithScore],
    graph_candidates: List[NodeWithScore],
    graph_boost: float = 0.2,
    pivot_entity_names: Optional[List[str]] = None,
) -> List[NodeWithScore]:
    """Fusionne les candidats vectoriels et KAG."""
    seen_node_ids = set()
    merged: List[NodeWithScore] = []
    pivot_set = set(pivot_entity_names or [])
    normalize_entity_name = None
    if pivot_set:
        from app.services.kag_extraction_service import normalize_entity_name as _norm
        normalize_entity_name = _norm

    for nws in vector_candidates:
        node_id = getattr(nws.node, "id_", None)
        if node_id:
            seen_node_ids.add(node_id)
        merged.append(nws)

    for nws in graph_candidates:
        node_id = getattr(nws.node, "id_", None)
        matched_entity = (nws.node.metadata or {}).get("kag_matched_entity", "")
        is_pivot = bool(
            pivot_set and matched_entity and normalize_entity_name
            and normalize_entity_name(matched_entity) in pivot_set
        )
        boost = 2.0 * graph_boost if is_pivot else graph_boost
        if node_id and node_id in seen_node_ids:
            for existing in merged:
                existing_id = getattr(existing.node, "id_", None)
                if existing_id == node_id:
                    existing.score = max(existing.score, nws.score + boost)
                    break
            continue

        boosted_score = min(1.0, float(nws.score or 0.0) + boost)
        nws.score = boosted_score
        merged.append(nws)
        if node_id:
            seen_node_ids.add(node_id)

    merged.sort(key=lambda x: float(x.score or 0.0), reverse=True)

    logger.debug(
        "Fusion KAG (space): %d vectoriels + %d graphe → %d total",
        len(vector_candidates),
        len(graph_candidates),
        len(merged),
    )
    return merged


def _normalize_for_gamme(s: str) -> str:
    """Lowercase + suppression des accents."""
    if not s:
        return ""
    n = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in n if unicodedata.category(c) != "Mn")


def _get_meaningful_words(text: str) -> Set[str]:
    """Extrait les mots significatifs."""
    if not text or not text.strip():
        return set()
    normalized = _normalize_for_gamme(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return {
        w for w in tokens
        if len(w) > 3 and w not in _FALLBACK_STOPWORDS
    }


def refine_with_source_authority(
    passages: List[Dict],
    query_text: str,
) -> List[Dict]:
    """
    Source authority : boost les passages dont le titre correspond à la requête.
    """
    if not passages or not query_text or not query_text.strip():
        return passages
    query_words = _get_meaningful_words(query_text)
    if not query_words:
        return passages
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


def search_relevant_passages(
    session: Session,
    space_id: int,
    query_text: str,
    user_id: int,
    k: int = 15,
) -> List[Dict]:
    """
    Recherche sémantique RAG + KAG sur les documents d'un espace.

    Pipeline (aligné projets) :
      1. SQL pgvector sur les leaves → k*3 candidats
      2. KAG : graphe d'entités uniquement ; fusion vectoriel + graphe
      3. Filtrage pré-reranking (similarité minimale)
      4. Early stopping si similarité élevée
      5. Reranking cross-encoder (BGE-reranker-v2-m3)
      6. Résolution des parents pour contexte enrichi
      7. Source authority (boost titre-requête)
      8. Fallback lexical si besoin

    Args:
        session    : Session SQLModel
        space_id   : ID de l'espace
        query_text : Texte de la requête
        user_id    : ID de l'utilisateur
        k          : Nombre de passages à retourner

    Returns:
        Liste de dicts { passage, document_title, document_id, chunk_index, score }
    """
    space = get_space_by_id(session, space_id, user_id)
    if not space:
        logger.warning("Espace %d non trouvé ou non accessible pour l'utilisateur %d", space_id, user_id)
        return []

    if not query_text or not query_text.strip():
        logger.warning("Requête vide fournie")
        return []

    try:
        candidate_k = (
            k * RERANKER_CANDIDATE_MULTIPLIER
            if (RERANKER_AVAILABLE and RERANKER_ENABLED)
            else k
        )

        # --- Étape 1 : retrieval vectoriel SQL (leaves) ---
        leaf_candidates = _retrieve_leaves_sql(
            session=session,
            space_id=space_id,
            user_id=user_id,
            query_text=query_text,
            candidate_k=candidate_k,
        )

        if not leaf_candidates:
            logger.info("Aucun résultat vectoriel (space), activation fallback lexical")
            return _keyword_fallback_passages(
                session=session,
                space_id=space_id,
                user_id=user_id,
                query_text=query_text,
                k=k,
            )

        # --- Étape 1b : enrichissement KAG (graphe d'entités uniquement) ---
        pivot_entity_names: List[str] = []

        if settings.KAG_ENABLED:
            try:
                from app.services.kag_extraction_service import extract_entities_from_query_sync
                pivot_entity_names = extract_entities_from_query_sync(query_text)
                if pivot_entity_names:
                    logger.debug("Entités pivot requête (space): %s", pivot_entity_names[:5])
            except Exception as ext_err:
                logger.debug("Extraction entités requête ignorée (space): %s", ext_err)

            try:
                graph_candidates = _retrieve_via_knowledge_graph(
                    session=session,
                    space_id=space_id,
                    user_id=user_id,
                    query_text=query_text,
                    limit=k,
                    pivot_entity_names=pivot_entity_names or None,
                )
                if graph_candidates:
                    leaf_candidates = _merge_with_graph_candidates(
                        vector_candidates=leaf_candidates,
                        graph_candidates=graph_candidates,
                        graph_boost=0.15,
                        pivot_entity_names=pivot_entity_names or None,
                    )
                    logger.info(
                        "KAG enrichissement (space): +%d candidats graphe fusionnés",
                        len(graph_candidates),
                    )
            except Exception as kag_err:
                logger.warning("KAG enrichissement échoué (space): %s", kag_err)

        # --- Étape 2 : filtrage pré-reranking ---
        filtered_candidates = _filter_low_similarity_candidates(
            leaf_candidates, MIN_VECTOR_SIMILARITY_THRESHOLD
        )

        # Early stopping
        skip_reranking = False
        if len(filtered_candidates) >= k:
            top_k_scores = [float(c.score or 0.0) for c in filtered_candidates[:k]]
            avg_top_k = sum(top_k_scores) / len(top_k_scores)

            if avg_top_k >= SKIP_RERANK_THRESHOLD:
                logger.info(
                    "Similarité élevée (space) (%.3f >= %.2f), skip reranking",
                    avg_top_k,
                    SKIP_RERANK_THRESHOLD,
                )
                skip_reranking = True
                top_leaves = filtered_candidates[:k]

        # --- Étape 3 : reranking sur les LEAVES ---
        if not skip_reranking and RERANKER_AVAILABLE and RERANKER_ENABLED:
            try:
                max_rerank = min(
                    len(filtered_candidates),
                    max(k * 2, MAX_RERANK_CANDIDATES),
                )
                candidates_to_rerank = filtered_candidates[:max_rerank]

                for nws in candidates_to_rerank:
                    node = nws.node
                    meta = dict(getattr(node, "metadata", {}) or {})
                    content = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "") or ""
                    enriched = _enrich_content_with_heading_and_figure(content, meta)
                    if hasattr(node, "set_content"):
                        node.set_content(enriched)
                    else:
                        setattr(node, "text", enriched)

                logger.info(
                    "Démarrage reranking (space) sur %d candidats leaves...",
                    len(candidates_to_rerank),
                )

                reranker = _get_reranker()
                if reranker:
                    reranked = reranker.postprocess_nodes(
                        candidates_to_rerank,
                        query_bundle=QueryBundle(query_str=query_text),
                    )
                    logger.info(
                        "✅ Reranking (space) terminé: %d → %d candidats",
                        len(candidates_to_rerank),
                        len(reranked),
                    )
                    top_leaves = reranked[:k]
                else:
                    logger.warning("Reranker non disponible, fallback ordre vectoriel")
                    top_leaves = filtered_candidates[:k]
            except Exception as rerank_err:
                logger.warning(
                    "Reranking échoué (space), fallback ordre vectoriel: %s", rerank_err
                )
                top_leaves = filtered_candidates[:k]
        elif not skip_reranking:
            if not RERANKER_ENABLED:
                logger.debug("Reranker désactivé (RERANKER_ENABLED=false)")
            top_leaves = filtered_candidates[:k]

        # --- Étape 4 : résolution des parents ---
        parent_node_dict = _build_parent_node_dict(session, space_id, user_id)

        final_nodes: List[NodeWithScore] = []
        seen_node_ids: set = set()
        parents_resolved = 0
        parents_not_found = 0

        for nws in top_leaves:
            score = float(getattr(nws, "score", 0.0) or 0.0)
            leaf_meta = dict(getattr(nws.node, "metadata", {}) or {})
            parent_node_id = leaf_meta.get("parent_node_id")

            target_node = (
                parent_node_dict.get(parent_node_id) if parent_node_id else None
            )
            if target_node is None:
                target_node = nws.node
                if parent_node_id:
                    parents_not_found += 1
            else:
                parents_resolved += 1

            node_id = getattr(target_node, "id_", None)
            if node_id and node_id in seen_node_ids:
                continue
            if node_id:
                seen_node_ids.add(node_id)

            final_nodes.append(NodeWithScore(node=target_node, score=score))

        logger.info(
            "Résolution parents (space): %d passages finaux (%d parents résolus, %d non trouvés)",
            len(final_nodes),
            parents_resolved,
            parents_not_found,
        )

        passages = [
            _node_to_passage(nws.node, fallback_score=float(nws.score or 0.0))
            for nws in final_nodes
        ]

        score_strs = [f"{p['score']:.3f}" for p in passages[:3]]
        logger.info(
            "Trouvé %d passages (space)%s (scores: %s...)",
            len(passages),
            " [reranked]" if (RERANKER_AVAILABLE and RERANKER_ENABLED) else "",
            score_strs,
        )

        passages = refine_with_source_authority(passages, query_text)

        if not passages:
            return _keyword_fallback_passages(
                session=session,
                space_id=space_id,
                user_id=user_id,
                query_text=query_text,
                k=k,
            )

        return passages

    except Exception as e:
        logger.error("Erreur recherche passages (space): %s", e, exc_info=True)
        return []
