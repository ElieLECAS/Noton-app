"""
Service de recherche sémantique pour les espaces (architecture Document/Space).

Pipeline principal :
  1. Embedding de la requête via BGE-m3
  2. Détection multi-hop (heuristique : mots-clés + entités pivot)
  3a. [Standard] Candidats vectoriels (pgvector) + lexicaux (ts_rank_cd) + KAG (graphe)
  3b. [Multi-hop]  Orchestrateur max 3 sauts : hop0 hybride → expansion graphe → sauts suivants
  4. Fusion hybride pondérée + scoring par profondeur (pénalité par hop)
  5. Filtrage pré-reranking
  6. Early stopping si similarité vectorielle déjà élevée
  7. Reranking cross-encoder (BGE-reranker-v2-m3)
  8. Résolution des parents (clé ``node_id`` Docling + ``chunk-{id}``)
  9. Source authority (boost titre-requête)
  10. Fallback lexical ILIKE si aucun candidat hybride
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
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
from app.tracing import trace_run
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
# Deux étapes : large pool tronqué puis raffinement sur texte complet
RERANK_STAGE1_MAX = int(os.getenv("RERANK_STAGE1_MAX", "100"))
RERANK_STAGE2_POOL = int(os.getenv("RERANK_STAGE2_POOL", "25"))
RERANK_STAGE1_CHAR_CAP = int(os.getenv("RERANK_STAGE1_CHAR_CAP", "800"))
SKIP_RERANK_THRESHOLD = float(os.getenv("SKIP_RERANK_THRESHOLD", "0.85"))
_FLAG_RERANK_TOP_N = int(os.getenv("RERANKER_TOP_N", "4096"))
TRACE_VERBOSE_TEXT = os.getenv("TRACE_VERBOSE_TEXT", "false").lower() == "true"
TRACE_TEXT_MAX_CHARS = int(os.getenv("TRACE_TEXT_MAX_CHARS", "12000"))

# Singletons
_reranker_instance = None
_embed_model_instance = None


def _text_for_trace(node: TextNode) -> str:
    raw = (
        node.get_content()
        if hasattr(node, "get_content")
        else getattr(node, "text", "") or ""
    )
    if not isinstance(raw, str):
        raw = str(raw)
    if TRACE_TEXT_MAX_CHARS > 0 and len(raw) > TRACE_TEXT_MAX_CHARS:
        return raw[:TRACE_TEXT_MAX_CHARS]
    return raw


def _nodes_for_trace(candidates: List[NodeWithScore], limit: int = 80) -> List[Dict]:
    rows: List[Dict] = []
    for nws in candidates[:limit]:
        meta = dict(getattr(nws.node, "metadata", {}) or {})
        rows.append(
            {
                "score": round(float(nws.score or 0.0), 4),
                "document_title": meta.get("document_title"),
                "section": meta.get("parent_heading") or meta.get("heading"),
                "kag_entity": meta.get("kag_matched_entity"),
                "text": _text_for_trace(nws.node),
            }
        )
    return rows

_FALLBACK_STOPWORDS = {
    "the", "and", "for", "with", "dans", "avec", "pour", "une", "des", "les",
    "est", "sur", "pas", "plus", "que", "qui", "this", "that", "what", "how",
    "quoi", "comment", "quel", "quelle", "quels", "quelles", "from", "par",
    "sans", "mais", "donc", "car", "you", "your", "not", "are", "was", "were",
}

TITLE_QUERY_BOOST_PER_MATCH = float(os.getenv("TITLE_QUERY_BOOST_PER_MATCH", "0.5"))
TITLE_QUERY_BOOST_CAP = float(os.getenv("TITLE_QUERY_BOOST_CAP", "2.0"))

# Fusion hybride permanente (vectoriel + BM25-like + KAG) — constantes applicatives
# (réglage direct dans le code, pas de variables d'environnement)
RAG_HYBRID_ALPHA = 0.60
RAG_HYBRID_BETA = 0.25
RAG_HYBRID_GAMMA = 0.15
HYBRID_MIN_SCORE = 0.06

# ---------------------------------------------------------------------------
# Multi-hop — constantes en dur (pas de variables d'environnement)
# ---------------------------------------------------------------------------
MULTI_HOP_ENABLED = True
MULTI_HOP_MAX_HOPS = 3
MULTI_HOP_CANDIDATE_BUDGET = 80   # plafond global de candidats (tous hops confondus)
MULTI_HOP_PER_HOP_LIMIT = 20      # candidats KAG max par hop d'expansion
MULTI_HOP_PATIENCE = 1            # sauts consécutifs sans nouveaux chunks avant arrêt
MULTI_HOP_MIN_DELTA_NEW_CHUNKS = 1  # nb minimum de nouveaux chunks pour continuer

# Scoring multi-hop unifié : score = 0.55*v + 0.20*l + 0.20*k + 0.05*evidence - penalty
MH_WEIGHT_VECTOR = 0.55
MH_WEIGHT_LEXICAL = 0.20
MH_WEIGHT_KAG = 0.20
MH_WEIGHT_EVIDENCE = 0.05
# Pénalité par profondeur : hop0→0.00, hop1→0.05, hop2→0.10, hop3→0.15
MH_HOP_PENALTIES = {0: 0.00, 1: 0.05, 2: 0.10, 3: 0.15}

# Mots-clés heuristiques indiquant une requête multi-hop
_MH_TRIGGER_PATTERNS = re.compile(
    r"\b(et\b|comparaison|impact|cause|depend|dépend|influence|relation|lien"
    r"|si\b|alors\b|pourquoi|comment|implique|nécessite|necessite|versus|vs\b"
    r"|différence|difference|avantage|inconvénient|inconvenient)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Structure d'état multi-hop
# ---------------------------------------------------------------------------

@dataclass
class _MultiHopState:
    """État de recherche conservé entre les sauts d'un pipeline multi-hop."""
    seen_chunk_ids: Set[int] = field(default_factory=set)
    seen_entity_ids: Set[int] = field(default_factory=set)
    seen_entity_names: Set[str] = field(default_factory=set)
    # chunk_id → {"vector": float, "lexical": float, "kag": float,
    #              "evidence": float, "hop": int, "path": str}
    chunk_signals: Dict[int, Dict] = field(default_factory=dict)
    # hop → nombre de nouveaux chunks apportés
    new_chunks_count_by_hop: Dict[int, int] = field(default_factory=dict)
    # hop_traces : chunk_id → liste des entités qui ont amené ce chunk
    hop_traces: Dict[int, List[str]] = field(default_factory=dict)


def _merged_chunk_metadata(primary: Optional[dict], legacy: Optional[dict]) -> Dict:
    """
    Fusionne les métadonnées modernes + legacy.
    Les clés de ``primary`` (metadata_json) priment si présentes.
    """
    merged: Dict = {}
    if isinstance(legacy, dict):
        merged.update(legacy)
    if isinstance(primary, dict):
        merged.update(primary)
    return merged


def _get_reranker():
    """Retourne le reranker (singleton)."""
    global _reranker_instance
    if not RERANKER_AVAILABLE:
        return None
    if _reranker_instance is None:
        use_fp16 = os.getenv("RERANKER_USE_FP16", "false").lower() == "true"
        logger.info(
            "Initialisation reranker %s (top_n=%s, use_fp16=%s)...",
            RERANKER_MODEL,
            _FLAG_RERANK_TOP_N,
            use_fp16,
        )
        _reranker_instance = FlagEmbeddingReranker(
            model=RERANKER_MODEL,
            top_n=_FLAG_RERANK_TOP_N,
            use_fp16=use_fp16,
        )
        logger.info("✅ Reranker initialisé")
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


def _set_node_text_content(node, text: str) -> None:
    if hasattr(node, "set_content"):
        node.set_content(text)
    else:
        setattr(node, "text", text)


def _two_stage_rerank_leaves(
    filtered_candidates: List[NodeWithScore],
    query_text: str,
    k: int,
) -> List[NodeWithScore]:
    """
    Étape 1 : rerank rapide sur un large pool avec texte tronqué.
    Étape 2 : rerank sur les meilleurs avec texte enrichi complet.
    """
    reranker = _get_reranker()
    if not reranker:
        return filtered_candidates[:k]
    stage1_max = min(
        len(filtered_candidates),
        max(RERANK_STAGE1_MAX, k * 2),
    )
    pool = filtered_candidates[:stage1_max]
    backup: Dict[str, str] = {}
    for nws in pool:
        node = nws.node
        nid = str(getattr(node, "id_", None) or "")
        raw = (
            node.get_content()
            if hasattr(node, "get_content")
            else getattr(node, "text", "") or ""
        )
        backup[nid] = raw
        meta = dict(getattr(node, "metadata", {}) or {})
        enriched = _enrich_content_with_heading_and_figure(raw, meta)
        short = (
            enriched[:RERANK_STAGE1_CHAR_CAP]
            if len(enriched) > RERANK_STAGE1_CHAR_CAP
            else enriched
        )
        _set_node_text_content(node, short)
    try:
        r1 = reranker.postprocess_nodes(
            pool,
            query_bundle=QueryBundle(query_str=query_text),
        )
    except Exception as e:
        logger.warning("Rerank étape 1 (space) échoué: %s", e)
        for nws in pool:
            nid = str(getattr(nws.node, "id_", None) or "")
            if nid in backup:
                _set_node_text_content(nws.node, backup[nid])
        return filtered_candidates[:k]

    n_stage2 = min(RERANK_STAGE2_POOL, len(r1))
    for nws in pool:
        nid = str(getattr(nws.node, "id_", None) or "")
        if nid in backup:
            _set_node_text_content(nws.node, backup[nid])

    stage2: List[NodeWithScore] = []
    for nws in r1[:n_stage2]:
        node = nws.node
        nid = str(getattr(node, "id_", None) or "")
        raw = backup.get(nid, "")
        meta = dict(getattr(node, "metadata", {}) or {})
        enriched = _enrich_content_with_heading_and_figure(raw, meta)
        _set_node_text_content(node, enriched)
        stage2.append(NodeWithScore(node=node, score=float(nws.score or 0.0)))

    try:
        r2 = reranker.postprocess_nodes(
            stage2,
            query_bundle=QueryBundle(query_str=query_text),
        )
        logger.info(
            "Reranking 2 étapes (space): pool=%d → stage2=%d → final=%d",
            len(pool),
            len(stage2),
            len(r2),
        )
        return r2[:k]
    except Exception as e:
        logger.warning("Rerank étape 2 (space) échoué: %s", e)
        return r1[:k]


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
        metadata = _merged_chunk_metadata(row.metadata_json, row.metadata_)
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


def _parse_chunk_id_from_node(node: TextNode) -> Optional[int]:
    """Extrait l'ID base du chunk depuis ``chunk-{id}``."""
    nid = getattr(node, "id_", None) or ""
    if isinstance(nid, str) and nid.startswith("chunk-"):
        try:
            return int(nid.split("-", 1)[1])
        except ValueError:
            return None
    return None


def _normalize_min_max(scores: Dict[int, float]) -> Dict[int, float]:
    """Min-max sur un dict non vide ; si une seule valeur, tout à 1.0."""
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _retrieve_leaves_lexical_sql(
    session: Session,
    space_id: int,
    query_text: str,
    candidate_k: int,
) -> List[NodeWithScore]:
    """
    Recherche lexicale (ts_rank_cd / type BM25) sur les feuilles de l'espace.
    """
    if not query_text or not query_text.strip():
        return []

    sql_query = text(
        """
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
            ts_rank_cd(
                to_tsvector('simple', coalesce(dc.content, dc.text, '')),
                plainto_tsquery('simple', :q)
            ) AS lex_score
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        INNER JOIN document_space ds ON ds.document_id = d.id
        WHERE ds.space_id = :space_id
          AND dc.is_leaf = true
          AND coalesce(dc.content, dc.text, '') <> ''
          AND to_tsvector('simple', coalesce(dc.content, dc.text, ''))
              @@ plainto_tsquery('simple', :q)
        ORDER BY lex_score DESC
        LIMIT :limit_k
        """
    )

    result = session.execute(
        sql_query,
        {"space_id": space_id, "q": query_text.strip(), "limit_k": candidate_k},
    )

    nodes_with_scores: List[NodeWithScore] = []
    for row in result:
        metadata = _merged_chunk_metadata(row.metadata_json, row.metadata_)
        metadata.setdefault("document_id", row.document_id)
        metadata.setdefault("document_title", row.document_title or "Document sans titre")
        metadata.setdefault("chunk_index", row.chunk_index)

        node = TextNode(
            id_=f"chunk-{row.id}",
            text=row.content or row.text or "",
            metadata=metadata,
        )
        nodes_with_scores.append(
            NodeWithScore(node=node, score=float(row.lex_score or 0.0))
        )

    logger.info(
        "Recherche lexicale tsvector (space): %d feuilles (candidate_k=%d)",
        len(nodes_with_scores),
        candidate_k,
    )
    return nodes_with_scores


def _hybrid_fuse_candidates(
    vector_candidates: List[NodeWithScore],
    lexical_candidates: List[NodeWithScore],
    graph_candidates: List[NodeWithScore],
) -> List[NodeWithScore]:
    """
    Fusionne vectoriel, lexical (BM25-like) et KAG avec normalisation min-max
    par signal puis score = alpha*v + beta*l + gamma*k.
    """
    v_by_id: Dict[int, Tuple[float, TextNode]] = {}
    for nws in vector_candidates:
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is None:
            continue
        v_by_id[cid] = (float(nws.score or 0.0), nws.node)

    l_by_id: Dict[int, float] = {}
    l_nodes: Dict[int, TextNode] = {}
    for nws in lexical_candidates:
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is None:
            continue
        l_by_id[cid] = float(nws.score or 0.0)
        l_nodes[cid] = nws.node

    k_by_id: Dict[int, Tuple[float, TextNode]] = {}
    for nws in graph_candidates:
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is None:
            continue
        sc = float(nws.score or 0.0)
        if cid not in k_by_id or sc > k_by_id[cid][0]:
            k_by_id[cid] = (sc, nws.node)

    all_ids = set(v_by_id) | set(l_by_id) | set(k_by_id)
    if not all_ids:
        return []

    v_raw = {i: v_by_id[i][0] for i in all_ids if i in v_by_id}
    l_raw = {i: l_by_id[i] for i in all_ids if i in l_by_id}
    k_raw = {i: k_by_id[i][0] for i in all_ids if i in k_by_id}

    v_nmap = _normalize_min_max(v_raw)
    l_nmap = _normalize_min_max(l_raw)
    k_nmap = _normalize_min_max(k_raw)

    fused: List[NodeWithScore] = []
    for cid in all_ids:
        v_raw_s = v_by_id[cid][0] if cid in v_by_id else None
        v_norm = v_nmap.get(cid, 0.0) if cid in v_raw else 0.0
        l_norm = l_nmap.get(cid, 0.0) if cid in l_raw else 0.0
        k_norm = k_nmap.get(cid, 0.0) if cid in k_raw else 0.0

        hybrid = (
            RAG_HYBRID_ALPHA * v_norm
            + RAG_HYBRID_BETA * l_norm
            + RAG_HYBRID_GAMMA * k_norm
        )

        if cid in v_by_id:
            node = v_by_id[cid][1]
        elif cid in l_nodes:
            node = l_nodes[cid]
        else:
            node = k_by_id[cid][1]

        meta = dict(getattr(node, "metadata", {}) or {})
        if v_raw_s is not None:
            meta["vector_similarity"] = v_raw_s
        meta["lexical_norm"] = l_norm
        meta["kag_norm"] = k_norm
        meta["hybrid_score"] = hybrid
        meta["retrieval_signal"] = "hybrid"
        node.metadata = meta

        fused.append(NodeWithScore(node=node, score=hybrid))

    fused.sort(key=lambda x: float(x.score or 0.0), reverse=True)
    logger.debug(
        "Fusion hybride (space): %d candidats (vector=%d lexical=%d kag=%d)",
        len(fused),
        len(v_raw),
        len(l_raw),
        len(k_raw),
    )
    return fused


def _filter_hybrid_candidates(
    candidates: List[NodeWithScore],
) -> List[NodeWithScore]:
    """
    Filtre les candidats hybrides : garde si similarité vectorielle suffisante,
    ou score hybride / lexical / KAG suffisant.
    """
    out: List[NodeWithScore] = []
    for nws in candidates:
        meta = dict(getattr(nws.node, "metadata", {}) or {})
        h = float(nws.score or 0.0)
        vx = meta.get("vector_similarity")
        v_ok = vx is not None and float(vx) >= MIN_VECTOR_SIMILARITY_THRESHOLD
        ln = float(meta.get("lexical_norm", 0) or 0)
        kn = float(meta.get("kag_norm", 0) or 0)
        if v_ok or h >= HYBRID_MIN_SCORE or ln >= 0.12 or kn >= 0.12:
            out.append(nws)
    return out


def _build_parent_node_dict(
    session: Session, space_id: int, user_id: int
) -> Dict[str, TextNode]:
    """
    Charge les nœuds parents (DocumentChunk is_leaf=False) pour enrichir le contexte.

    Indexation par ``node_id`` Docling (UUID) lorsque présent : les feuilles référencent
    ``parent_node_id`` avec cet identifiant, pas ``chunk-{id}``.
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
        metadata = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        metadata.setdefault("document_id", chunk.document_id)
        metadata.setdefault("document_title", document_title or "Document sans titre")
        metadata.setdefault("chunk_index", chunk.chunk_index)
        llama_id = f"chunk-{chunk.id}"
        text = chunk.content or chunk.text or ""
        node = TextNode(
            id_=llama_id,
            text=text,
            metadata=metadata,
        )
        # Clé principale : UUID Docling / hiérarchie (aligné avec leaf.parent_node_id)
        if chunk.node_id:
            node_dict[str(chunk.node_id)] = node
        # Compatibilité anciens chemins
        node_dict[llama_id] = node

    logger.info(
        "Chargé %d nœuds parents pour space_id=%d (index node_id + chunk-id)",
        len(rows),
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

        node_metadata = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
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


def _merge_leaf_page_into_node_metadata(leaf_node, target_node) -> None:
    """
    Recopie page_no / plage depuis la feuille matchée vers le nœud cible (ex. parent résolu).
    Priorité à la page de la feuille pour l'ouverture PDF au bon endroit.
    """
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
    # Traçabilité: conserver l'ID du chunk feuille à l'origine de la citation.
    leaf_chunk_id = _parse_chunk_id_from_node(leaf_node)
    if leaf_chunk_id is not None:
        m["source_leaf_chunk_id"] = leaf_chunk_id
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
    if resolved_page is None and page_start is not None:
        try:
            resolved_page = int(page_start)
        except (TypeError, ValueError):
            pass
    page_no = resolved_page
    parent_heading = metadata.get("parent_heading")

    content = node.get_content() if hasattr(node, "get_content") else str(node)
    content_enriched = _enrich_content_with_heading_and_figure(content, metadata)
    passage_text = f"**{document_title}**\n{content_enriched}"
    chunk_id = _parse_chunk_id_from_node(node)
    source_leaf_chunk_id = metadata.get("source_leaf_chunk_id")
    out = {
        "passage": passage_text,
        "passage_raw": content,
        "document_title": document_title,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "source_leaf_chunk_id": source_leaf_chunk_id,
        "chunk_index": int(chunk_index) if isinstance(chunk_index, (int, str)) else 0,
        "score": float(fallback_score or 0.0),
        "page_no": page_no,
        "section": parent_heading,
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
    return out


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
        from app.services.kag_graph_service import (
            expand_kag_query_terms_for_space,
            _neighbor_entity_ids_for_entities,
        )

        # Stratégie 1: utiliser les entités pivot LLM si disponibles
        normalized_pivots = _normalize_pivot_entities(pivot_entity_names)
        if normalized_pivots:
            query_terms = normalized_pivots
            logger.debug(
                "KAG retrieval (space): utilisation de %d entités pivot LLM",
                len(query_terms),
            )
        else:
            # Fallback: split naïf de la requête
            query_terms = [t.strip().lower() for t in re.findall(r"[A-Za-zÀ-ÿ0-9]+", query_text)]
            query_terms = [t for t in query_terms if len(t) >= 3 and t not in _FALLBACK_STOPWORDS]

        if not query_terms:
            return []

        query_terms = expand_kag_query_terms_for_space(session, space_id, query_terms)
        query_terms = query_terms[:120]

        # Matching exact sur name_normalized
        stmt = (
            select(
                DocumentChunk,
                Document.title,
                KnowledgeEntity.name,
                KnowledgeEntity.id,
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
                len(results),
            )
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
                    KnowledgeEntity.id,
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
            existing_chunk_ids = {row[0].id for row in results}
            for row in ilike_results:
                if row[0].id not in existing_chunk_ids:
                    results.append(row)
                    existing_chunk_ids.add(row[0].id)

        neighbor_chunk_ids: Set[int] = set()
        seed_entity_ids = {row[3] for row in results if row[3] is not None}
        neighbor_entity_ids = _neighbor_entity_ids_for_entities(
            session, space_id, seed_entity_ids
        )
        if neighbor_entity_ids and len(results) < limit:
            stmt_neigh = (
                select(
                    DocumentChunk,
                    Document.title,
                    KnowledgeEntity.name,
                    KnowledgeEntity.id,
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
                    KnowledgeEntity.id.in_(neighbor_entity_ids),
                )
                .order_by(ChunkEntityRelation.relevance_score.desc())
                .limit(max(0, limit - len(results)))
            )
            existing_chunk_ids = {row[0].id for row in results}
            for row in session.exec(stmt_neigh).all():
                if row[0].id not in existing_chunk_ids:
                    results.append(row)
                    existing_chunk_ids.add(row[0].id)
                    neighbor_chunk_ids.add(row[0].id)

        nodes_with_scores: List[NodeWithScore] = []
        for chunk, document_title, entity_name, _entity_id, relevance in results:
            metadata = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
            metadata.setdefault("document_id", chunk.document_id)
            metadata.setdefault("document_title", document_title or "Document sans titre")
            metadata.setdefault("chunk_index", chunk.chunk_index)
            metadata["kag_matched_entity"] = entity_name
            if chunk.id in neighbor_chunk_ids:
                metadata["kag_neighbor_match"] = True
            rel = float(relevance)
            if chunk.id in neighbor_chunk_ids:
                rel *= 0.88

            node = TextNode(
                id_=f"chunk-{chunk.id}",
                text=chunk.content or chunk.text or "",
                metadata=metadata,
            )
            nodes_with_scores.append(NodeWithScore(node=node, score=rel))

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


def _needs_multi_hop(query_text: str, pivot_entity_names: List[str]) -> bool:
    """
    Détecte si la requête nécessite un retrieval multi-hop.

    Critères (OR) :
    - La requête contient au moins un mot-clé indicateur multi-hop.
    - Au moins 2 entités pivot distinctes ont été extraites de la requête.
    """
    if not query_text:
        return False
    if _MH_TRIGGER_PATTERNS.search(query_text):
        return True
    if len(pivot_entity_names) >= 2:
        return True
    return False


def _extract_top_entity_names_from_candidates(
    candidates: List[NodeWithScore],
    top_n: int = 10,
) -> List[str]:
    """
    Extrait les noms d'entités KAG dominantes des meilleurs candidats courants.
    Utilisé entre deux hops pour définir les seeds du saut suivant.
    """
    entity_counter: Dict[str, int] = {}
    for nws in candidates:
        meta = dict(getattr(nws.node, "metadata", {}) or {})
        entity = meta.get("kag_matched_entity")
        if entity and isinstance(entity, str):
            key = entity.strip().lower()
            if key:
                entity_counter[key] = entity_counter.get(key, 0) + 1
    sorted_entities = sorted(entity_counter, key=lambda e: entity_counter[e], reverse=True)
    return sorted_entities[:top_n]


def _retrieve_kag_for_entity_seeds(
    session: Session,
    space_id: int,
    entity_names: List[str],
    limit: int,
    seen_chunk_ids: Set[int],
    seen_entity_ids: Set[int],
) -> List[NodeWithScore]:
    """
    Retrieval KAG ciblé à partir d'une liste de noms d'entités seeds.
    Exclut les chunks et entités déjà vus dans les hops précédents.
    Retourne uniquement des chunks nouveaux.
    """
    try:
        from app.services.kag_graph_service import (
            expand_kag_query_terms_for_space,
            _neighbor_entity_ids_for_entities,
        )
        from app.services.kag_extraction_service import normalize_entity_name

        normalized = [normalize_entity_name(n) for n in entity_names if n]
        normalized = [n for n in normalized if n]
        if not normalized:
            return []

        expanded = expand_kag_query_terms_for_space(session, space_id, normalized)
        expanded = expanded[:80]

        stmt = (
            select(
                DocumentChunk,
                Document.title,
                KnowledgeEntity.name,
                KnowledgeEntity.id,
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
                KnowledgeEntity.name_normalized.in_(expanded),
            )
            .order_by(ChunkEntityRelation.relevance_score.desc())
            .limit(limit * 2)
        )
        rows = list(session.exec(stmt).all())

        # Expansion voisins graphe depuis les entités seeds
        seed_entity_ids = {row[3] for row in rows if row[3] is not None} - seen_entity_ids
        if seed_entity_ids:
            neighbor_ids = _neighbor_entity_ids_for_entities(
                session, space_id, seed_entity_ids
            ) - seen_entity_ids
            if neighbor_ids:
                stmt_n = (
                    select(
                        DocumentChunk,
                        Document.title,
                        KnowledgeEntity.name,
                        KnowledgeEntity.id,
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
                        KnowledgeEntity.id.in_(neighbor_ids),
                    )
                    .order_by(ChunkEntityRelation.relevance_score.desc())
                    .limit(limit)
                )
                existing_ids = {row[0].id for row in rows}
                for row in session.exec(stmt_n).all():
                    if row[0].id not in existing_ids:
                        rows.append(row)
                        existing_ids.add(row[0].id)

        nodes: List[NodeWithScore] = []
        seen_in_batch: Set[int] = set()
        for chunk, doc_title, entity_name, _eid, relevance in rows:
            if chunk.id in seen_chunk_ids or chunk.id in seen_in_batch:
                continue
            seen_in_batch.add(chunk.id)
            metadata = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
            metadata.setdefault("document_id", chunk.document_id)
            metadata.setdefault("document_title", doc_title or "Document sans titre")
            metadata.setdefault("chunk_index", chunk.chunk_index)
            metadata["kag_matched_entity"] = entity_name
            node = TextNode(
                id_=f"chunk-{chunk.id}",
                text=chunk.content or chunk.text or "",
                metadata=metadata,
            )
            nodes.append(NodeWithScore(node=node, score=float(relevance)))
            if len(nodes) >= limit:
                break

        return nodes

    except Exception as exc:
        logger.warning("KAG ciblé (multi-hop) échoué: %s", exc)
        return []


def _apply_multihop_depth_scoring(
    state: _MultiHopState,
    all_nodes: Dict[int, NodeWithScore],
) -> List[NodeWithScore]:
    """
    Calcule le score unifié multi-hop pour chaque chunk en tenant compte
    des signaux par source (vector, lexical, kag, evidence) et de la pénalité
    de profondeur (hop d'origine du chunk).

    Formule : score = 0.55*v + 0.20*l + 0.20*k + 0.05*evidence - hop_penalty
    """
    if not state.chunk_signals:
        return list(all_nodes.values())

    v_raw = {cid: sig.get("vector", 0.0) for cid, sig in state.chunk_signals.items()}
    l_raw = {cid: sig.get("lexical", 0.0) for cid, sig in state.chunk_signals.items()}
    k_raw = {cid: sig.get("kag", 0.0) for cid, sig in state.chunk_signals.items()}
    e_raw = {cid: sig.get("evidence", 0.0) for cid, sig in state.chunk_signals.items()}

    v_norm = _normalize_min_max(v_raw)
    l_norm = _normalize_min_max(l_raw)
    k_norm = _normalize_min_max(k_raw)
    e_norm = _normalize_min_max(e_raw)

    scored: List[NodeWithScore] = []
    for cid, sig in state.chunk_signals.items():
        nws = all_nodes.get(cid)
        if nws is None:
            continue
        hop = sig.get("hop", 0)
        penalty = MH_HOP_PENALTIES.get(hop, MH_HOP_PENALTIES[MULTI_HOP_MAX_HOPS])
        mh_score = (
            MH_WEIGHT_VECTOR * v_norm.get(cid, 0.0)
            + MH_WEIGHT_LEXICAL * l_norm.get(cid, 0.0)
            + MH_WEIGHT_KAG * k_norm.get(cid, 0.0)
            + MH_WEIGHT_EVIDENCE * e_norm.get(cid, 0.0)
            - penalty
        )
        mh_score = max(0.0, mh_score)

        meta = dict(getattr(nws.node, "metadata", {}) or {})
        meta["retrieval_hop"] = hop
        meta["hop_penalty"] = penalty
        meta["evidence_score"] = sig.get("evidence", 0.0)
        meta["retrieval_path"] = sig.get("path", f"hop{hop}")
        meta["mh_score"] = mh_score
        nws.node.metadata = meta
        scored.append(NodeWithScore(node=nws.node, score=mh_score))

    scored.sort(key=lambda x: float(x.score or 0.0), reverse=True)
    return scored


def multi_hop_retrieve_space(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    pivot_entity_names: List[str],
    candidate_k: int,
) -> List[NodeWithScore]:
    """
    Orchestrateur multi-hop pour la recherche dans un espace.

    Enchaîne jusqu'à MULTI_HOP_MAX_HOPS sauts :
    - Hop 0 : retrieval hybride standard (vector + lexical + KAG pivot).
    - Hop N : expansion graphe à partir des entités dominantes des meilleurs
              chunks du saut précédent + retrieval KAG ciblé.
    Applique le scoring par profondeur et retourne les candidats fusionnés
    triés, prêts pour le filtre pré-rerank et le reranker existants.
    """
    state = _MultiHopState()
    all_nodes: Dict[int, NodeWithScore] = {}

    # --- Hop 0 : retrieval hybride complet ---
    vector_candidates = _retrieve_leaves_sql(
        session=session,
        space_id=space_id,
        user_id=user_id,
        query_text=query_text,
        candidate_k=candidate_k,
    )
    lexical_candidates = _retrieve_leaves_lexical_sql(
        session=session,
        space_id=space_id,
        query_text=query_text,
        candidate_k=candidate_k,
    )
    graph_candidates_hop0 = _retrieve_via_knowledge_graph(
        session=session,
        space_id=space_id,
        user_id=user_id,
        query_text=query_text,
        limit=candidate_k,
        pivot_entity_names=pivot_entity_names or None,
    )

    # Indexer les signaux bruts du hop 0
    v_by_id: Dict[int, float] = {}
    for nws in vector_candidates:
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is not None:
            v_by_id[cid] = float(nws.score or 0.0)

    l_by_id: Dict[int, float] = {}
    for nws in lexical_candidates:
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is not None:
            l_by_id[cid] = float(nws.score or 0.0)

    k_by_id: Dict[int, float] = {}
    for nws in graph_candidates_hop0:
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is not None:
            sc = float(nws.score or 0.0)
            if cid not in k_by_id or sc > k_by_id[cid]:
                k_by_id[cid] = sc

    hop0_ids = set(v_by_id) | set(l_by_id) | set(k_by_id)
    new_at_hop0 = 0

    # Fusionner dans l'état global
    for cid in hop0_ids:
        state.seen_chunk_ids.add(cid)
        state.chunk_signals[cid] = {
            "vector": v_by_id.get(cid, 0.0),
            "lexical": l_by_id.get(cid, 0.0),
            "kag": k_by_id.get(cid, 0.0),
            "evidence": 0.0,
            "hop": 0,
            "path": "hop0:hybrid",
        }
        new_at_hop0 += 1

    # Conserver les nœuds en utilisant la fusion hybride pour le nœud de référence
    fused_hop0 = _hybrid_fuse_candidates(
        vector_candidates=vector_candidates,
        lexical_candidates=lexical_candidates,
        graph_candidates=graph_candidates_hop0,
    )
    for nws in fused_hop0:
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is not None:
            all_nodes[cid] = nws

    state.new_chunks_count_by_hop[0] = new_at_hop0
    logger.info(
        "Multi-hop (space) hop0: %d candidats hybrides (space_id=%d)",
        len(all_nodes),
        space_id,
    )

    # Enregistrer les entités vues en hop 0
    for nws in graph_candidates_hop0:
        meta = dict(getattr(nws.node, "metadata", {}) or {})
        entity = meta.get("kag_matched_entity")
        if entity:
            state.seen_entity_names.add(entity.strip().lower())

    # --- Hops 1..N ---
    stagnation_count = 0

    for hop in range(1, MULTI_HOP_MAX_HOPS + 1):
        if len(state.seen_chunk_ids) >= MULTI_HOP_CANDIDATE_BUDGET:
            logger.info(
                "Multi-hop (space) arrêt budget (%d >= %d) au hop %d",
                len(state.seen_chunk_ids),
                MULTI_HOP_CANDIDATE_BUDGET,
                hop,
            )
            break

        # Extraire les entités seeds des meilleurs candidats actuels
        top_current = sorted(
            all_nodes.values(),
            key=lambda x: float(x.score or 0.0),
            reverse=True,
        )[:MULTI_HOP_PER_HOP_LIMIT]
        seed_entity_names = _extract_top_entity_names_from_candidates(top_current, top_n=12)

        # Exclure les entités déjà explorées
        new_seeds = [n for n in seed_entity_names if n not in state.seen_entity_names]
        if not new_seeds:
            logger.info(
                "Multi-hop (space) arrêt saturation entités au hop %d (toutes vues)",
                hop,
            )
            break

        state.seen_entity_names.update(new_seeds)

        hop_candidates = _retrieve_kag_for_entity_seeds(
            session=session,
            space_id=space_id,
            entity_names=new_seeds,
            limit=MULTI_HOP_PER_HOP_LIMIT,
            seen_chunk_ids=state.seen_chunk_ids,
            seen_entity_ids=state.seen_entity_ids,
        )

        new_count = 0
        for nws in hop_candidates:
            cid = _parse_chunk_id_from_node(nws.node)
            if cid is None or cid in state.seen_chunk_ids:
                continue
            state.seen_chunk_ids.add(cid)
            kag_score = float(nws.score or 0.0)
            meta = dict(getattr(nws.node, "metadata", {}) or {})
            entity_name = meta.get("kag_matched_entity", "")
            # L'evidence = nombre d'entités pivot déjà connues dans ce chunk
            evidence = sum(
                1 for e in pivot_entity_names
                if e and e.lower() in (meta.get("kag_matched_entity") or "").lower()
            )
            state.chunk_signals[cid] = {
                "vector": 0.0,
                "lexical": 0.0,
                "kag": kag_score,
                "evidence": float(evidence),
                "hop": hop,
                "path": f"hop{hop}:{','.join(new_seeds[:3])}",
            }
            if entity_name:
                state.hop_traces.setdefault(cid, []).append(entity_name)
            all_nodes[cid] = nws
            new_count += 1

        state.new_chunks_count_by_hop[hop] = new_count

        logger.info(
            "Multi-hop (space) hop%d: +%d nouveaux chunks (seeds=%s)",
            hop,
            new_count,
            new_seeds[:3],
        )

        if new_count < MULTI_HOP_MIN_DELTA_NEW_CHUNKS:
            stagnation_count += 1
            if stagnation_count >= MULTI_HOP_PATIENCE:
                logger.info(
                    "Multi-hop (space) arrêt stagnation après hop %d (%d saut(s) vide(s))",
                    hop,
                    stagnation_count,
                )
                break
        else:
            stagnation_count = 0

    # --- Scoring final par profondeur ---
    scored_candidates = _apply_multihop_depth_scoring(state, all_nodes)

    logger.info(
        "Multi-hop (space) terminé: %d candidats finaux (hops=%s)",
        len(scored_candidates),
        dict(state.new_chunks_count_by_hop),
    )
    return scored_candidates


def search_relevant_passages(
    session: Session,
    space_id: int,
    query_text: str,
    user_id: int,
    k: int = 15,
) -> List[Dict]:
    """
    Recherche sémantique RAG + KAG sur les documents d'un espace.

    Pipeline :
      1. Extraction entités pivot (KAG) + détection multi-hop heuristique
      2a. [Multi-hop] Orchestrateur max 3 sauts avec scoring par profondeur
      2b. [Standard]  Fusion hybride vectoriel + lexical + KAG
      3. Filtrage pré-reranking
      4. Early stopping si similarité vectorielle élevée
      5. Reranking cross-encoder (BGE-reranker-v2-m3)
      6. Résolution des parents pour contexte enrichi
      7. Source authority (boost titre-requête)
      8. Fallback lexical ILIKE si aucun candidat hybride

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

        # --- Étape 1 : extraction entités pivot (KAG) ---
        pivot_entity_names: List[str] = []
        if settings.KAG_ENABLED:
            try:
                from app.services.kag_extraction_service import extract_entities_from_query_sync

                pivot_entity_names = extract_entities_from_query_sync(query_text)
                if pivot_entity_names:
                    logger.debug("Entités pivot requête (space): %s", pivot_entity_names[:5])
            except Exception as ext_err:
                logger.debug("Extraction entités requête ignorée (space): %s", ext_err)

        # --- Étape 2 : sélection du mode retrieval ---
        use_multi_hop = (
            MULTI_HOP_ENABLED
            and settings.KAG_ENABLED
            and _needs_multi_hop(query_text, pivot_entity_names)
        )

        if use_multi_hop:
            logger.info(
                "Multi-hop activé (space_id=%d) — pivots=%s",
                space_id,
                pivot_entity_names[:4],
            )
            with trace_run(
                "multi_hop_retrieval",
                run_type="retriever",
                inputs={
                    "query": query_text,
                    "space_id": space_id,
                    "pivot_entities": pivot_entity_names[:10],
                    "candidate_k": candidate_k,
                    "max_hops": MULTI_HOP_MAX_HOPS,
                },
                tags=["retrieval", "multi-hop", "kag", "space"],
            ) as mh_run:
                leaf_candidates = multi_hop_retrieve_space(
                    session=session,
                    space_id=space_id,
                    user_id=user_id,
                    query_text=query_text,
                    pivot_entity_names=pivot_entity_names,
                    candidate_k=candidate_k,
                )
                top3_scores = [round(float(c.score or 0), 4) for c in leaf_candidates[:3]]
                mh_outputs = {
                    "nb_candidates": len(leaf_candidates),
                    "top3_scores": top3_scores,
                }
                if TRACE_VERBOSE_TEXT:
                    mh_outputs["candidates_text"] = _nodes_for_trace(leaf_candidates)
                mh_run.end(outputs=mh_outputs)
        else:
            # Pipeline hybride standard
            with trace_run(
                "vector_retrieval",
                run_type="retriever",
                inputs={"query": query_text, "space_id": space_id, "candidate_k": candidate_k},
                tags=["retrieval", "vector", "pgvector", "space"],
            ) as vr_run:
                vector_candidates = _retrieve_leaves_sql(
                    session=session,
                    space_id=space_id,
                    user_id=user_id,
                    query_text=query_text,
                    candidate_k=candidate_k,
                )
                vr_outputs = {
                    "nb_candidates": len(vector_candidates),
                    "top3_scores": [round(float(c.score or 0), 4) for c in vector_candidates[:3]],
                }
                if TRACE_VERBOSE_TEXT:
                    vr_outputs["candidates_text"] = _nodes_for_trace(vector_candidates)
                vr_run.end(outputs=vr_outputs)

            with trace_run(
                "lexical_retrieval",
                run_type="retriever",
                inputs={"query": query_text, "space_id": space_id, "candidate_k": candidate_k},
                tags=["retrieval", "lexical", "tsvector", "space"],
            ) as lr_run:
                lexical_candidates = _retrieve_leaves_lexical_sql(
                    session=session,
                    space_id=space_id,
                    query_text=query_text,
                    candidate_k=candidate_k,
                )
                lr_outputs = {"nb_candidates": len(lexical_candidates)}
                if TRACE_VERBOSE_TEXT:
                    lr_outputs["candidates_text"] = _nodes_for_trace(lexical_candidates)
                lr_run.end(outputs=lr_outputs)

            graph_candidates: List[NodeWithScore] = []
            if settings.KAG_ENABLED:
                try:
                    with trace_run(
                        "kag_graph_retrieval",
                        run_type="retriever",
                        inputs={"query": query_text, "pivot_entities": pivot_entity_names[:10], "space_id": space_id},
                        tags=["retrieval", "kag", "graph", "space"],
                    ) as kag_run:
                        graph_candidates = _retrieve_via_knowledge_graph(
                            session=session,
                            space_id=space_id,
                            user_id=user_id,
                            query_text=query_text,
                            limit=candidate_k,
                            pivot_entity_names=pivot_entity_names or None,
                        )
                        matched = list({
                            (c.node.metadata or {}).get("kag_matched_entity", "")
                            for c in graph_candidates
                            if (c.node.metadata or {}).get("kag_matched_entity")
                        })
                        kag_outputs = {"nb_chunks": len(graph_candidates), "matched_entities": matched[:10]}
                        if TRACE_VERBOSE_TEXT:
                            kag_outputs["candidates_text"] = _nodes_for_trace(graph_candidates)
                        kag_run.end(outputs=kag_outputs)
                except Exception as kag_err:
                    logger.warning("KAG retrieval échoué (space): %s", kag_err)

            with trace_run(
                "hybrid_fusion",
                run_type="chain",
                inputs={
                    "nb_vector": len(vector_candidates),
                    "nb_lexical": len(lexical_candidates),
                    "nb_kag": len(graph_candidates),
                    "alpha": RAG_HYBRID_ALPHA,
                    "beta": RAG_HYBRID_BETA,
                    "gamma": RAG_HYBRID_GAMMA,
                },
                tags=["fusion", "hybrid", "space"],
            ) as fusion_run:
                leaf_candidates = _hybrid_fuse_candidates(
                    vector_candidates=vector_candidates,
                    lexical_candidates=lexical_candidates,
                    graph_candidates=graph_candidates,
                )
                fusion_outputs = {"nb_fused": len(leaf_candidates)}
                if TRACE_VERBOSE_TEXT:
                    fusion_outputs["fused_text"] = _nodes_for_trace(leaf_candidates)
                fusion_run.end(outputs=fusion_outputs)

        if not leaf_candidates:
            logger.info(
                "Aucun candidat (space, multi_hop=%s), activation fallback lexical ILIKE",
                use_multi_hop,
            )
            return _keyword_fallback_passages(
                session=session,
                space_id=space_id,
                user_id=user_id,
                query_text=query_text,
                k=k,
            )

        # --- Étape 3 : filtrage pré-reranking ---
        filtered_candidates = _filter_hybrid_candidates(leaf_candidates)
        if not filtered_candidates:
            logger.info(
                "Filtre hybride vide (space) — repli sur candidats fusionnés bruts"
            )
            filtered_candidates = leaf_candidates

        # Early stopping
        skip_reranking = False
        if len(filtered_candidates) >= k:
            top_k_scores = []
            for c in filtered_candidates[:k]:
                meta = dict(getattr(c.node, "metadata", {}) or {})
                vs = meta.get("vector_similarity")
                if vs is not None:
                    top_k_scores.append(float(vs))
                else:
                    top_k_scores.append(float(c.score or 0.0))
            avg_top_k = sum(top_k_scores) / len(top_k_scores)

            if avg_top_k >= SKIP_RERANK_THRESHOLD:
                logger.info(
                    "Similarité élevée (space) (%.3f >= %.2f), skip reranking",
                    avg_top_k,
                    SKIP_RERANK_THRESHOLD,
                )
                skip_reranking = True
                top_leaves = filtered_candidates[:k]

        # --- Étape 3 : reranking deux étapes sur les LEAVES ---
        if not skip_reranking and RERANKER_AVAILABLE and RERANKER_ENABLED:
            try:
                if _get_reranker():
                    with trace_run(
                        "reranking",
                        run_type="chain",
                        inputs={
                            "nb_candidates": len(filtered_candidates),
                            "stage1_max": RERANK_STAGE1_MAX,
                            "stage2_pool": RERANK_STAGE2_POOL,
                            "k": k,
                            "multi_hop": use_multi_hop,
                        },
                        tags=["reranking", "bge-reranker", "space"],
                    ) as rerank_run:
                        top_leaves = _two_stage_rerank_leaves(
                            filtered_candidates,
                            query_text,
                            k,
                        )
                        top_scores = [round(float(n.score or 0), 4) for n in top_leaves[:5]]
                        rerank_outputs = {"nb_final": len(top_leaves), "top5_scores": top_scores}
                        if TRACE_VERBOSE_TEXT:
                            rerank_outputs["top_leaves_text"] = _nodes_for_trace(top_leaves)
                            rerank_outputs["filtered_candidates_text"] = _nodes_for_trace(filtered_candidates)
                        rerank_run.end(outputs=rerank_outputs)
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
        with trace_run(
            "parent_resolution",
            run_type="chain",
            inputs={"space_id": space_id, "nb_top_leaves": len(top_leaves)},
            tags=["parent", "context", "space"],
        ) as parent_run:
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
                    _merge_leaf_page_into_node_metadata(nws.node, target_node)

                node_id = getattr(target_node, "id_", None)
                if node_id and node_id in seen_node_ids:
                    continue
                if node_id:
                    seen_node_ids.add(node_id)

                final_nodes.append(NodeWithScore(node=target_node, score=score))

            parent_outputs = {
                "nb_final_passages": len(final_nodes),
                "parents_resolved": parents_resolved,
                "parents_not_found": parents_not_found,
            }
            if TRACE_VERBOSE_TEXT:
                parent_outputs["top_leaves_text"] = _nodes_for_trace(top_leaves)
                parent_outputs["final_nodes_text"] = _nodes_for_trace(final_nodes)
            parent_run.end(outputs=parent_outputs)

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
