"""
Service de recherche sémantique RAG simplifié et optimisé.

Pipeline :
  1. Embedding de la requête via BGE-m3 (singleton)
  2. Recherche vectorielle SQL directe (pgvector <=>) sur les chunks LEAF uniquement
  3. Optimisations pré-reranking :
     - Filtrage des candidats avec faible similarité vectorielle (< MIN_VECTOR_SIMILARITY_THRESHOLD)
     - Early stopping si similarité déjà élevée (>= SKIP_RERANK_THRESHOLD)
  4. Reranking des candidats avec BGE-reranker-v2-m3 (cross-encoder, sur les leaves,
     courts → rapide) — remplace le reranking sur les parents qui prenait ~56s
  5. Résolution des parents pour fournir un contexte enrichi au LLM
  6. Fallback lexical si aucun embedding disponible

Optimisations :
  - Filtrage pré-reranking : évite de reranker des candidats peu pertinents (-20 à -40% de temps)
  - Early stopping : skip le reranking si similarité vectorielle déjà élevée (~10-20% des cas)
  - Limite dynamique : ajuste le nombre de candidats selon les besoins (max MAX_RERANK_CANDIDATES)
  - Reranker : GPU si PyTorch voit CUDA ; ``RERANKER_USE_FP16`` / ``RERANKER_TOP_N`` en option

Les anciens composants LlamaIndex (PGVectorStore, VectorStoreIndex, RecursiveRetriever,
MetadataFilter) ont été supprimés : la recherche SQL directe est plus fiable et cohérente
avec l'architecture de la table notechunk (insertions via SQLModel ORM).
"""

from typing import Any, Dict, List, Optional, Set
import os
import re
import threading
import unicodedata
from sqlmodel import Session, select
from sqlalchemy import or_
from app.models.note import Note
from app.models.note_chunk import NoteChunk
from app.services.project_service import get_project_by_id
from app.services.retrieval_pipeline import (
    VECTOR_RETRIEVE_MAX,
    RetrievalStats,
    adaptive_gate,
    adaptive_top_n,
    analyze_query,
    annotate_kag_matches_note,
    log_retrieval_stats,
    merge_vector_and_exact,
    retrieve_exact_refs_note,
    smart_parent_or_leaf,
)
import logging
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.config import settings
from app.tracing import trace_run
from app.services.chunk_metadata_utils import (
    apply_row_metadata_defaults,
    enrich_docling_page_metadata,
    enrich_passage_content_for_llm,
    merged_chunk_metadata,
    resolve_page_range_from_metadata,
    table_citation_hint,
)

logger = logging.getLogger(__name__)

try:
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("FlagEmbeddingReranker non disponible, reranking désactivé")

RERANKER_MODEL = settings.RERANKER_MODEL
RERANKER_CANDIDATE_MULTIPLIER = 3
RERANKER_ENABLED = settings.RERANKER_ENABLED
# Seuils retrieval notes (contenu libre, légèrement plus permissif que espaces)
MIN_VECTOR_SIMILARITY_THRESHOLD = 0.45
MAX_RERANK_CANDIDATES = int(os.getenv("MAX_RERANK_CANDIDATES", "50"))
RERANK_STAGE1_MAX = int(os.getenv("RERANK_STAGE1_MAX", "100"))
RERANK_STAGE2_POOL = int(os.getenv("RERANK_STAGE2_POOL", "25"))
RERANK_STAGE1_CHAR_CAP = int(os.getenv("RERANK_STAGE1_CHAR_CAP", "800"))
SKIP_RERANK_THRESHOLD = float(os.getenv("SKIP_RERANK_THRESHOLD", "0.85"))
# FlagEmbeddingReranker tronque à top_n (pas de device= dans llama-index 0.4.x).
_FLAG_RERANK_TOP_N = int(os.getenv("RERANKER_TOP_N", "4096"))
TRACE_VERBOSE_TEXT = os.getenv("TRACE_VERBOSE_TEXT", "false").lower() == "true"
TRACE_TEXT_MAX_CHARS = int(os.getenv("TRACE_TEXT_MAX_CHARS", "12000"))

# ---------------------------------------------------------------------------
# Singletons — chargement unique des modèles lourds
# ---------------------------------------------------------------------------

_reranker_instance = None
_reranker_lock = threading.Lock() if RERANKER_AVAILABLE else None

_embed_model_instance = None
_embed_model_lock = threading.Lock()


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
                "note_title": meta.get("note_title"),
                "section": meta.get("parent_heading") or meta.get("heading"),
                "kag_entity": meta.get("kag_matched_entity"),
                "text": _text_for_trace(nws.node),
            }
        )
    return rows


def _get_reranker():
    """
    Retourne le reranker (singleton).

    FlagEmbeddingReranker n'accepte pas ``device=`` ; le device suit PyTorch
    (CUDA si disponible). ``RERANKER_TOP_N`` borne la sortie du postprocessor ;
    le pipeline tronque ensuite à ``k`` via ``_two_stage_rerank_leaves``.
    """
    global _reranker_instance
    if not RERANKER_AVAILABLE:
        return None
    with _reranker_lock:
        if _reranker_instance is None:
            use_fp16 = os.getenv("RERANKER_USE_FP16", "false").lower() == "true"
            logger.info(
                "Initialisation du reranker %s (top_n=%s, use_fp16=%s)...",
                RERANKER_MODEL,
                _FLAG_RERANK_TOP_N,
                use_fp16,
            )
            _reranker_instance = FlagEmbeddingReranker(
                model=RERANKER_MODEL,
                top_n=_FLAG_RERANK_TOP_N,
                use_fp16=use_fp16,
            )
            logger.info("✅ Reranker initialisé et prêt")
        return _reranker_instance


def _get_embed_model() -> HuggingFaceEmbedding:
    """Retourne le modèle d'embedding BGE-m3 (singleton)."""
    global _embed_model_instance
    with _embed_model_lock:
        if _embed_model_instance is None:
            model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
            device = os.getenv("EMBEDDING_DEVICE", "cpu")
            logger.info(
                "Initialisation du modèle d'embedding %s sur %s (une seule fois)...",
                model_name,
                device,
            )
            _embed_model_instance = HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
                embed_batch_size=settings.EMBEDDING_BATCH_SIZE,
            )
            logger.info("✅ Modèle d'embedding initialisé et prêt")
        return _embed_model_instance


# ---------------------------------------------------------------------------
# Recherche vectorielle SQL directe (pgvector)
# ---------------------------------------------------------------------------

def _retrieve_leaves_sql(
    session: Session,
    project_id: int,
    user_id: int,
    query_text: str,
    candidate_k: int,
) -> List[NodeWithScore]:
    """
    Recherche vectorielle SQL directe sur les chunks LEAF via l'opérateur pgvector <=>.

    Avantages par rapport à LlamaIndex PGVectorStore :
    - Filtre directement sur les colonnes SQL (is_leaf, project_id, user_id) au lieu des
      métadonnées JSON → résultats toujours cohérents, sans friction de typage
    - Pas de dépendance à la structure de table LlamaIndex
    - Une seule requête SQL, pas de boucles de fallback
    """
    from sqlalchemy import text

    embed_model = _get_embed_model()
    query_embedding = embed_model.get_query_embedding(query_text)
    query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    sql_query = text(f"""
        SELECT
            nc.id,
            nc.content,
            nc.text,
            nc.node_id,
            nc.parent_node_id,
            nc.metadata_json,
            nc.metadata_,
            n.title  AS note_title,
            n.id     AS note_id,
            1 - (nc.embedding <=> '{query_embedding_str}'::vector) AS similarity_score
        FROM notechunk nc
        INNER JOIN note n ON nc.note_id = n.id
        WHERE n.project_id = :project_id
          AND n.user_id    = :user_id
          AND nc.embedding IS NOT NULL
          AND nc.is_leaf   = true
        ORDER BY nc.embedding <=> '{query_embedding_str}'::vector
        LIMIT :limit_k
    """)

    result = session.execute(
        sql_query,
        {"project_id": project_id, "user_id": user_id, "limit_k": candidate_k},
    )

    nodes_with_scores: List[NodeWithScore] = []
    for row in result:
        metadata = merged_chunk_metadata(row.metadata_json, row.metadata_)
        apply_row_metadata_defaults(
            metadata,
            note_id=row.note_id,
            note_title=row.note_title or "Note sans titre",
            node_id=row.node_id,
            parent_node_id=row.parent_node_id,
        )
        metadata["chunk_db_id"] = row.id

        node = TextNode(
            id_=row.node_id or f"chunk-{row.id}",
            text=row.content or row.text or "",
            metadata=metadata,
        )
        nodes_with_scores.append(
            NodeWithScore(node=node, score=float(row.similarity_score))
        )

    logger.info(
        "Recherche SQL pgvector : %d nœuds leaf récupérés (candidate_k=%d)",
        len(nodes_with_scores),
        candidate_k,
    )
    return nodes_with_scores


# ---------------------------------------------------------------------------
# Résolution des parents (contexte enrichi pour le LLM)
# ---------------------------------------------------------------------------

def _build_parent_node_dict(
    session: Session, project_id: int, user_id: int
) -> Dict[str, TextNode]:
    """
    Charge les nœuds parents depuis la base pour enrichir le contexte envoyé au LLM.
    Appelé UNE SEULE FOIS par requête, après le reranking sur les leaves.
    """
    statement = (
        select(NoteChunk, Note.title)
        .join(Note, Note.id == NoteChunk.note_id)
        .where(
            Note.project_id == project_id,
            Note.user_id == user_id,
            NoteChunk.is_leaf.is_(False),
            NoteChunk.node_id.is_not(None),
        )
    )
    rows = session.exec(statement).all()

    node_dict: Dict[str, TextNode] = {}
    for chunk, note_title in rows:
        metadata = merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        metadata.setdefault("note_id", chunk.note_id)
        metadata.setdefault("note_title", note_title or "Note sans titre")
        metadata.setdefault("node_id", chunk.node_id)
        metadata.setdefault("parent_node_id", chunk.parent_node_id)
        node_dict[chunk.node_id] = TextNode(
            id_=chunk.node_id,
            text=chunk.content or chunk.text or "",
            metadata=metadata,
        )

    logger.info(
        "Chargé %d nœuds parents pour project_id=%d, user_id=%d",
        len(node_dict),
        project_id,
        user_id,
    )
    return node_dict


_PARENT_MULTIHOP_MAX = 4


def _resolve_note_parent_with_multihop(
    session: Session,
    project_id: int,
    user_id: int,
    note_id: Optional[int],
    parent_node_id: Optional[str],
    parent_node_dict: Dict[str, TextNode],
) -> Optional[TextNode]:
    """
    Remonte la chaîne parent_node_id (text_full, table_full, …) jusqu'au chunk section
    (is_leaf=False) et fusionne les textes intermédiaires.
    """
    if not parent_node_id or note_id is None:
        return None
    if parent_node_id in parent_node_dict:
        return None

    intermediates: List[str] = []
    current_pid: Optional[str] = parent_node_id
    hops = 0

    while current_pid and hops < _PARENT_MULTIHOP_MAX:
        hops += 1
        stmt = (
            select(NoteChunk, Note.title)
            .join(Note, Note.id == NoteChunk.note_id)
            .where(
                Note.project_id == project_id,
                Note.user_id == user_id,
                NoteChunk.note_id == note_id,
                NoteChunk.node_id == current_pid,
            )
        )
        row = session.exec(stmt).first()
        if not row:
            break
        chunk, note_title = row
        metadata = merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        metadata.setdefault("note_id", chunk.note_id)
        metadata.setdefault("note_title", note_title or "Note sans titre")
        metadata.setdefault("node_id", chunk.node_id)
        metadata.setdefault("parent_node_id", chunk.parent_node_id)
        text = (chunk.content or chunk.text or "").strip()

        if not chunk.is_leaf:
            if intermediates:
                prefix = "\n\n---\n\n".join(reversed(intermediates))
                text = f"{prefix}\n\n---\n\n{text}"
            return TextNode(
                id_=chunk.node_id,
                text=text,
                metadata=metadata,
            )

        if text:
            intermediates.append(text)
        current_pid = chunk.parent_node_id

    return None


# ---------------------------------------------------------------------------
# Fallback lexical
# ---------------------------------------------------------------------------

_FALLBACK_STOPWORDS = {
    "the", "and", "for", "with", "dans", "avec", "pour", "une", "des", "les",
    "est", "sur", "pas", "plus", "que", "qui", "this", "that", "what", "how",
    "quoi", "comment", "quel", "quelle", "quels", "quelles", "from", "par",
    "sans", "mais", "donc", "car", "you", "your", "not", "are", "was", "were",
}


def _extract_query_terms(query_text: str) -> List[str]:
    terms = [t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ0-9]+", query_text or "")]
    return [t for t in terms if len(t) >= 3 and t not in _FALLBACK_STOPWORDS][:8]


async def _keyword_fallback_passages(
    session: Session,
    project_id: int,
    user_id: int,
    query_text: str,
    k: int,
) -> List[Dict]:
    """
    Fallback lexical si aucun embedding n'est disponible ou si la recherche vectorielle
    ne retourne rien. Évite d'envoyer un contexte vide au LLM.
    """
    terms = _extract_query_terms(query_text)
    base_stmt = (
        select(NoteChunk, Note.title)
        .join(Note, Note.id == NoteChunk.note_id)
        .where(Note.project_id == project_id, Note.user_id == user_id)
        .order_by(NoteChunk.is_leaf.desc(), Note.updated_at.desc(), NoteChunk.chunk_index)
    )

    rows = []
    if terms:
        stmt = base_stmt.where(
            or_(*[NoteChunk.content.ilike(f"%{term}%") for term in terms])
        ).limit(max(k * 4, 12))
        rows = session.exec(stmt).all()

    if not rows:
        rows = session.exec(base_stmt.limit(max(k * 2, 8))).all()

    passages: List[Dict] = []
    seen_chunk_ids: set = set()
    for chunk, note_title in rows:
        if chunk.id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.id)

        content = (chunk.content or chunk.text or "").strip()
        if not content:
            continue

        lowered = content.lower()
        match_count = sum(1 for term in terms if term in lowered) if terms else 0
        score = (match_count / max(len(terms), 1)) if terms else 0.05

        node_metadata = merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        node_metadata.setdefault("note_id", chunk.note_id)
        node_metadata.setdefault("note_title", note_title or "Note sans titre")
        node_metadata.setdefault("node_id", chunk.node_id or f"chunk-{chunk.id}")
        node_metadata.setdefault("chunk_index", chunk.chunk_index)
        node = TextNode(
            id_=chunk.node_id or f"chunk-{chunk.id}",
            text=content,
            metadata=node_metadata,
        )
        passages.append(_node_to_passage(node, fallback_score=score))
        if len(passages) >= k:
            break

    logger.info(
        "Fallback lexical activé: %d passages construits (terms=%s)",
        len(passages),
        terms,
    )
    return passages


# ---------------------------------------------------------------------------
# Conversion nœud → passage
# ---------------------------------------------------------------------------

def _enrich_content_with_heading_and_figure(content: str, metadata: dict) -> str:
    """Préfixe le contenu avec section, figure et résumé parent KAG."""
    return enrich_passage_content_for_llm(content, metadata)


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
    """Rerank large pool (texte tronqué) puis raffinement sur texte enrichi complet."""
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
        logger.warning("Rerank étape 1 échoué: %s", e)
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
            "Reranking 2 étapes: pool=%d → stage2=%d → final=%d",
            len(pool),
            len(stage2),
            len(r2),
        )
        return r2[:k]
    except Exception as e:
        logger.warning("Rerank étape 2 échoué: %s", e)
        return r1[:k]


def _merge_leaf_page_into_node_metadata(leaf_node, target_node) -> None:
    """Recopie page_no / plage depuis la feuille vers le parent résolu."""
    leaf_meta = enrich_docling_page_metadata(
        dict(getattr(leaf_node, "metadata", {}) or {})
    )
    m = dict(getattr(target_node, "metadata", {}) or {})
    pn, ps, pe = resolve_page_range_from_metadata(leaf_meta)
    if pn is not None:
        m["page_no"] = pn
    elif m.get("page_start") is not None:
        try:
            m["page_no"] = int(m["page_start"])
        except (TypeError, ValueError):
            pass
    if ps is not None:
        m.setdefault("page_start", ps)
    if pe is not None:
        m.setdefault("page_end", pe)
    setattr(target_node, "metadata", m)


def _node_to_passage(node, fallback_score: float = 0.0) -> Dict:
    metadata = dict(getattr(node, "metadata", {}) or {})
    note_title = metadata.get("note_title", "Note sans titre")
    note_id = metadata.get("note_id")
    node_id = metadata.get("node_id")
    chunk_index = metadata.get("chunk_index", 0)
    page_no, page_start, page_end = resolve_page_range_from_metadata(metadata)
    parent_heading = metadata.get("parent_heading") or metadata.get("heading")
    image_path = metadata.get("image_path")
    image_filename = metadata.get("image_filename")
    is_image_chunk = bool(metadata.get("is_image_chunk"))
    if not is_image_chunk and image_path and not image_filename:
        is_image_chunk = True
        image_filename = str(image_path).split("/")[-1].split("\\")[-1]
    caption = metadata.get("caption") or metadata.get("figure_title") or ""

    content = node.get_content() if hasattr(node, "get_content") else str(node)
    content_enriched = _enrich_content_with_heading_and_figure(content, metadata)
    passage_text = f"**{note_title}**\n{content_enriched}"
    table_hint = table_citation_hint(metadata)
    out = {
        "passage": passage_text,
        "passage_raw": content,
        "note_title": note_title,
        "note_id": note_id,
        "chunk_id": node_id,
        "chunk_index": int(chunk_index) if isinstance(chunk_index, (int, str)) else 0,
        "score": float(fallback_score or 0.0),
        "page_no": page_no,
        "section": parent_heading,
        "image_path": image_path,
        "image_filename": image_filename,
        "is_image_chunk": is_image_chunk,
        "caption": caption,
        "content_type": metadata.get("content_type"),
    }
    if table_hint:
        out["table_citation"] = table_hint
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


RERANK_MIN_SCORE = 0.30


def _single_stage_rerank_leaves(
    filtered_candidates: List[NodeWithScore],
    query_text: str,
    k: int,
    char_cap: int = 2000,
) -> List[NodeWithScore]:
    reranker = _get_reranker()
    if not reranker:
        return filtered_candidates[:k]
    pool = filtered_candidates
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
        enriched = enrich_passage_content_for_llm(raw, meta)
        short = enriched[:char_cap] if len(enriched) > char_cap else enriched
        _set_node_text_content(node, short)
    try:
        r = reranker.postprocess_nodes(
            pool,
            query_bundle=QueryBundle(query_str=query_text),
        )
        for nws in pool:
            nid = str(getattr(nws.node, "id_", None) or "")
            if nid in backup:
                _set_node_text_content(nws.node, backup[nid])
        for nws in r:
            nid = str(getattr(nws.node, "id_", None) or "")
            raw = backup.get(nid, "")
            meta = dict(getattr(nws.node, "metadata", {}) or {})
            enriched = enrich_passage_content_for_llm(raw, meta)
            _set_node_text_content(nws.node, enriched)
        return r[:k]
    except Exception as e:
        logger.warning("Rerank (note) échoué: %s", e)
        for nws in pool:
            nid = str(getattr(nws.node, "id_", None) or "")
            if nid in backup:
                _set_node_text_content(nws.node, backup[nid])
        return filtered_candidates[:k]


def _apply_rerank_min_score(top_leaves: List[NodeWithScore], k: int) -> List[NodeWithScore]:
    if not top_leaves:
        return []
    ranked = sorted(top_leaves, key=lambda n: float(n.score or 0), reverse=True)
    max_score = float(ranked[0].score or 0)
    if max_score < 0:
        return ranked[:k]
    filtered = [nws for nws in ranked if float(nws.score or 0) >= RERANK_MIN_SCORE]
    return (filtered or ranked)[:k]


def _normalize_for_gamme(s: str) -> str:
    """Lowercase + suppression des accents pour comparaison titre / termes requête."""
    if not s:
        return ""
    n = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in n if unicodedata.category(c) != "Mn")


def _get_meaningful_words(text: str) -> Set[str]:
    """
    Extrait les mots significatifs d'un texte (pour Title-Query Alignment).
    Normalisation NFD sans accents, mots de plus de 3 caractères, hors stopwords.
    """
    if not text or not text.strip():
        return set()
    normalized = _normalize_for_gamme(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return {
        w for w in tokens
        if len(w) > 3 and w not in _FALLBACK_STOPWORDS
    }


# Coefficient et plafond du boost Title-Query (configurables par env)
# Valeurs par défaut plus agressives pour privilégier fortement les documents dédiés.
TITLE_QUERY_BOOST_PER_MATCH = float(os.getenv("TITLE_QUERY_BOOST_PER_MATCH", "0.5"))
TITLE_QUERY_BOOST_CAP = float(os.getenv("TITLE_QUERY_BOOST_CAP", "2.0"))


def refine_with_source_authority(
    passages: List[Dict],
    query_text: str,
    reasoning_result: Optional[Any] = None,
) -> List[Dict]:
    """
    Source authority : boost les passages dont le titre correspond à la requête,
    OU qui correspondent à la source privilégiée déterminée par le raisonnement (CQR).
    """
    if not passages:
        return passages

    # 1. Boost basé sur le raisonnement (CQR)
    if reasoning_result and hasattr(reasoning_result, 'primary_source') and reasoning_result.primary_source:
        source_to_boost = reasoning_result.primary_source.lower()
        boost_value = 0.8
        for p in passages:
            # Pour les notes, la 'source' peut ne pas être présente de la même façon, 
            # mais on peut vérifier si le titre contient la marque ou si un champ source existe.
            doc_source = (p.get("source") or "").lower()
            note_title = (p.get("note_title") or "").lower()
            if doc_source == source_to_boost or source_to_boost in note_title:
                p["score"] = float(p.get("score") or 0.0) + boost_value

    # 2. Boost basé sur les mots du titre (Existant)
    if query_text and query_text.strip():
        query_words = _get_meaningful_words(query_text)
        if query_words:
            for p in passages:
                note_title = (p.get("note_title") or "").strip()
                if not note_title:
                    continue
                title_words = _get_meaningful_words(note_title)
                common = query_words & title_words
                if common:
                    boost = min(
                        TITLE_QUERY_BOOST_PER_MATCH * len(common),
                        TITLE_QUERY_BOOST_CAP,
                    )
                    p["score"] = float(p.get("score") or 0.0) + boost

    passages.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return passages


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

async def search_relevant_passages(
    session: Session,
    project_id: int,
    query_text: str,
    user_id: int,
    k: int = 15,
    passage_size: int = 500,  # ignoré, conservé pour compatibilité API
) -> List[Dict]:
    """Recherche lean sur les notes : vector + refs + KAG boost + rerank + K dynamique."""
    from app.services.query_reasoning_service import reason_query_intent

    reasoning_result = await reason_query_intent(query_text)
    project = get_project_by_id(session, project_id, user_id)
    if not project:
        logger.warning("Projet %d inaccessible (user %d)", project_id, user_id)
        return []
    if not query_text or not query_text.strip():
        return []

    stats = RetrievalStats()
    try:
        qa = analyze_query(query_text)
        with trace_run(
            "vector_retrieval",
            run_type="retriever",
            inputs={"query": query_text, "project_id": project_id},
            tags=["retrieval", "vector", "pgvector"],
        ) as vr_run:
            vector_nodes = _retrieve_leaves_sql(
                session, project_id, user_id, query_text, VECTOR_RETRIEVE_MAX
            )
            vr_run.end(
                outputs={
                    "nb": len(vector_nodes),
                    "top3": [round(float(c.score or 0), 4) for c in vector_nodes[:3]],
                }
            )

        exact_nodes: List[NodeWithScore] = []
        if qa.product_refs:
            exact_nodes = retrieve_exact_refs_note(
                session, project_id, user_id, qa.product_refs
            )
            stats.ref_match_used = bool(exact_nodes)

        candidates = merge_vector_and_exact(vector_nodes, exact_nodes, qa)
        stats.pool_after_merge = len(candidates)
        stats.top1_vector_similarity = max(
            (c.vector_similarity for c in candidates), default=0.0
        )

        if settings.KAG_ENABLED and qa.pivot_entities:
            stats.kag_boost_used = annotate_kag_matches_note(
                session, project_id, user_id, candidates, qa.pivot_entities
            )

        gated = adaptive_gate(candidates)
        stats.pool_after_gate = len(gated)

        if not gated:
            return await _keyword_fallback_passages(
                session, project_id, user_id, query_text, k
            )

        rerank_input = [c.to_node_with_score() for c in gated]
        if RERANKER_AVAILABLE and RERANKER_ENABLED and _get_reranker():
            with trace_run(
                "reranking",
                run_type="chain",
                inputs={"nb": len(rerank_input), "k_max": k},
                tags=["reranking"],
            ) as rr:
                reranked = _single_stage_rerank_leaves(
                    rerank_input, query_text, k=len(rerank_input), char_cap=2000
                )
                reranked = _apply_rerank_min_score(reranked, k=len(rerank_input))
                rr.end(outputs={"nb": len(reranked)})
        else:
            reranked = rerank_input

        top_n = adaptive_top_n(reranked, k_max=k)
        stats.final_k = len(top_n)

        parent_node_dict = _build_parent_node_dict(session, project_id, user_id)

        def _resolve_parent(parent_node_id: str, leaf_node: TextNode) -> Optional[TextNode]:
            leaf_meta = dict(getattr(leaf_node, "metadata", {}) or {})
            nid = leaf_meta.get("note_id")
            try:
                note_id_int = int(nid) if nid is not None else None
            except (TypeError, ValueError):
                note_id_int = None
            return _resolve_note_parent_with_multihop(
                session,
                project_id,
                user_id,
                note_id_int,
                parent_node_id,
                parent_node_dict,
            )

        final_nodes: List[NodeWithScore] = []
        seen_node_ids: set = set()
        for nws in top_n:
            leaf = nws.node
            target = smart_parent_or_leaf(
                leaf, parent_node_dict, resolve_parent_fn=_resolve_parent
            )
            if target is not leaf:
                _merge_leaf_page_into_node_metadata(leaf, target)
            node_id = getattr(target, "id_", None) or (
                getattr(target, "metadata", {}) or {}
            ).get("node_id")
            if node_id and node_id in seen_node_ids:
                continue
            if node_id:
                seen_node_ids.add(node_id)
            final_nodes.append(
                NodeWithScore(node=target, score=float(nws.score or 0.0))
            )

        passages = [
            _node_to_passage(nws.node, fallback_score=float(nws.score or 0.0))
            for nws in final_nodes
        ]
        passages = refine_with_source_authority(
            passages, query_text, reasoning_result=reasoning_result
        )
        log_retrieval_stats(stats, "note")

        if not passages:
            return await _keyword_fallback_passages(
                session, project_id, user_id, query_text, k
            )
        return passages

    except Exception as e:
        logger.error("Erreur recherche passages (note): %s", e, exc_info=True)
        return []


async def search_relevant_notes(
    session: Session,
    project_id: int,
    query_text: str,
    user_id: int,
    k: int = 10,
) -> List[Dict]:
    """
    Recherche sémantique sur les notes (agrégation depuis les passages).

    Args:
        session    : Session SQLModel
        project_id : ID du projet
        query_text : Texte de la requête
        user_id    : ID de l'utilisateur
        k          : Nombre de notes à retourner

    Returns:
        Liste de dicts { note, score }
    """
    project = get_project_by_id(session, project_id, user_id)
    if not project:
        logger.warning(
            "Projet %d non trouvé ou n'appartient pas à l'utilisateur %d",
            project_id,
            user_id,
        )
        return []

    if not query_text or not query_text.strip():
        logger.warning("Requête vide fournie")
        return []

    try:
        passages = await search_relevant_passages(
            session=session,
            project_id=project_id,
            query_text=query_text,
            user_id=user_id,
            k=k,
        )
        note_ids = [p["note_id"] for p in passages if p.get("note_id")]
        if not note_ids:
            return []

        note_stmt = select(Note).where(Note.id.in_(note_ids))
        notes = {note.id: note for note in session.exec(note_stmt).all()}

        results = []
        for passage in passages:
            note_id = passage.get("note_id")
            note = notes.get(note_id)
            if note:
                results.append(
                    {"note": note, "score": float(passage.get("score", 0.0))}
                )
        return results

    except Exception as e:
        logger.error("Erreur lors de la recherche sémantique: %s", e, exc_info=True)
        return []
