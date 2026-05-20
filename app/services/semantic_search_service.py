"""
Recherche RAG sur les notes (projet) : pgvector sur feuilles uniquement.

Pas de KAG, pas de reranker cross-encoder.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set

from llama_index.core.schema import NodeWithScore, TextNode
from sqlalchemy import or_, text
from sqlmodel import Session, select

from app.config import settings
from app.models.note import Note
from app.models.note_chunk import NoteChunk
from app.services.embedding_service import generate_embedding
from app.services.project_service import get_project_by_id
from app.tracing import trace_run

logger = logging.getLogger(__name__)

MIN_VECTOR_SIMILARITY_THRESHOLD = float(os.getenv("MIN_VECTOR_SIMILARITY", "0.25"))

TITLE_QUERY_BOOST_PER_MATCH = float(os.getenv("TITLE_QUERY_BOOST_PER_MATCH", "0.5"))
TITLE_QUERY_BOOST_CAP = float(os.getenv("TITLE_QUERY_BOOST_CAP", "2.0"))

_FALLBACK_STOPWORDS = {
    "the", "and", "for", "with", "dans", "avec", "pour", "une", "des", "les",
    "est", "sur", "pas", "plus", "que", "qui", "this", "that", "what", "how",
    "quoi", "comment", "quel", "quelle", "quels", "quelles", "from", "par",
    "sans", "mais", "donc", "car", "you", "your", "not", "are", "was", "were",
}

def _merged_note_metadata(primary: Optional[dict], legacy: Optional[dict]) -> Dict:
    merged: Dict = {}
    if isinstance(legacy, dict):
        merged.update(legacy)
    if isinstance(primary, dict):
        merged.update(primary)
    return merged


def _retrieve_leaves_sql(
    session: Session,
    project_id: int,
    user_id: int,
    query_text: str,
    candidate_k: int,
) -> List[NodeWithScore]:
    query_embedding = generate_embedding(query_text)
    if not query_embedding:
        logger.warning("Embedding requête vide pour project_id=%s", project_id)
        return []
    query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    sql_query = text(f"""
        SELECT
            nc.id,
            nc.content,
            nc.text,
            nc.node_id,
            nc.parent_node_id,
            nc.chunk_index,
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
    nodes: List[NodeWithScore] = []
    for row in result:
        metadata = _merged_note_metadata(row.metadata_json, row.metadata_)
        metadata.setdefault("note_id", row.note_id)
        metadata.setdefault("note_title", row.note_title or "Note sans titre")
        metadata.setdefault("node_id", row.node_id)
        metadata.setdefault("parent_node_id", row.parent_node_id)
        metadata.setdefault("chunk_index", row.chunk_index)
        node = TextNode(
            id_=row.node_id or f"chunk-{row.id}",
            text=row.content or row.text or "",
            metadata=metadata,
        )
        nodes.append(NodeWithScore(node=node, score=float(row.similarity_score)))
    logger.info("Vector pgvector (notes): %d feuilles (limit=%d)", len(nodes), candidate_k)
    return nodes


def _filter_by_vector_score(
    candidates: List[NodeWithScore],
    min_threshold: float = MIN_VECTOR_SIMILARITY_THRESHOLD,
) -> List[NodeWithScore]:
    filtered = [c for c in candidates if float(c.score or 0.0) >= min_threshold]
    if filtered:
        return filtered
    return sorted(candidates, key=lambda c: float(c.score or 0.0), reverse=True)[: max(5, len(candidates) // 2 or 1)]


def _build_parent_node_dict(
    session: Session, project_id: int, user_id: int
) -> Dict[str, TextNode]:
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
        metadata = _merged_note_metadata(chunk.metadata_json, chunk.metadata_)
        metadata.setdefault("note_id", chunk.note_id)
        metadata.setdefault("note_title", note_title or "Note sans titre")
        metadata.setdefault("node_id", chunk.node_id)
        metadata.setdefault("parent_node_id", chunk.parent_node_id)
        node_dict[chunk.node_id] = TextNode(
            id_=chunk.node_id,
            text=chunk.content or chunk.text or "",
            metadata=metadata,
        )
    logger.info("Parents chargés (notes): %d", len(node_dict))
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
        metadata = _merged_note_metadata(chunk.metadata_json, chunk.metadata_)
        metadata.setdefault("note_id", chunk.note_id)
        metadata.setdefault("note_title", note_title or "Note sans titre")
        metadata.setdefault("node_id", chunk.node_id)
        metadata.setdefault("parent_node_id", chunk.parent_node_id)
        text = (chunk.content or chunk.text or "").strip()

        if not chunk.is_leaf:
            if intermediates:
                prefix = "\n\n---\n\n".join(reversed(intermediates))
                text = f"{prefix}\n\n---\n\n{text}"
            return TextNode(id_=chunk.node_id, text=text, metadata=metadata)

        if text:
            intermediates.append(text)
        current_pid = chunk.parent_node_id

    return None


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
    seen: set = set()
    for chunk, note_title in rows:
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        content = (chunk.content or chunk.text or "").strip()
        if not content:
            continue
        lowered = content.lower()
        match_count = sum(1 for term in terms if term in lowered) if terms else 0
        score = (match_count / max(len(terms), 1)) if terms else 0.05
        meta = _merged_note_metadata(chunk.metadata_json, chunk.metadata_)
        meta.setdefault("note_id", chunk.note_id)
        meta.setdefault("note_title", note_title or "Note sans titre")
        meta.setdefault("node_id", chunk.node_id or f"chunk-{chunk.id}")
        meta.setdefault("chunk_index", chunk.chunk_index)
        node = TextNode(id_=chunk.node_id or f"chunk-{chunk.id}", text=content, metadata=meta)
        passages.append(_node_to_passage(node, fallback_score=score))
        if len(passages) >= k:
            break
    return passages


def _enrich_content_with_heading_and_figure(content: str, metadata: dict) -> str:
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
    setattr(target_node, "metadata", m)


def _node_to_passage(node, fallback_score: float = 0.0) -> Dict:
    metadata = dict(getattr(node, "metadata", {}) or {})
    note_title = metadata.get("note_title", "Note sans titre")
    note_id = metadata.get("note_id")
    node_id = metadata.get("node_id")
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
    image_path = metadata.get("image_path")
    image_filename = metadata.get("image_filename")
    is_image_chunk = metadata.get("is_image_chunk", False)
    caption = metadata.get("caption", "")
    content = node.get_content() if hasattr(node, "get_content") else str(node)
    content_enriched = _enrich_content_with_heading_and_figure(content, metadata)
    passage_text = f"**{note_title}**\n{content_enriched}"
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
    return out


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
    reasoning_result: Optional[Any] = None,
) -> List[Dict]:
    if not passages:
        return passages
    if reasoning_result and getattr(reasoning_result, "primary_source", None):
        source_to_boost = reasoning_result.primary_source.lower()
        for p in passages:
            doc_source = (p.get("source") or "").lower()
            note_title = (p.get("note_title") or "").lower()
            if doc_source == source_to_boost or source_to_boost in note_title:
                p["score"] = float(p.get("score") or 0.0) + 0.8
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


async def search_relevant_passages(
    session: Session,
    project_id: int,
    query_text: str,
    user_id: int,
    k: int = 15,
    passage_size: int = 500,
) -> List[Dict]:
    from app.services.query_reasoning_service import reason_query_intent

    _ = passage_size
    reasoning_result = await reason_query_intent(query_text)
    project = get_project_by_id(session, project_id, user_id)
    if not project:
        logger.warning("Projet %d inaccessible (user %d)", project_id, user_id)
        return []
    if not query_text or not query_text.strip():
        return []

    try:
        candidate_k = max(k, min(k * 4, 80))
        with trace_run(
            "vector_retrieval",
            run_type="retriever",
            inputs={"query": query_text, "project_id": project_id, "candidate_k": candidate_k},
            tags=["retrieval", "vector", "notes"],
        ) as vr:
            raw = _retrieve_leaves_sql(session, project_id, user_id, query_text, candidate_k)
            vr.end(outputs={"nb": len(raw)})

        if not raw:
            return await _keyword_fallback_passages(
                session, project_id, user_id, query_text, k
            )

        filtered = _filter_by_vector_score(raw)
        filtered.sort(key=lambda n: float(n.score or 0.0), reverse=True)
        top_leaves = filtered[:k]

        with trace_run(
            "parent_resolution",
            run_type="chain",
            inputs={"project_id": project_id, "nb": len(top_leaves)},
            tags=["parent", "notes"],
        ) as pr:
            parent_node_dict = _build_parent_node_dict(session, project_id, user_id)
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
                        nid = leaf_meta.get("note_id")
                        try:
                            note_id_int = int(nid) if nid is not None else None
                        except (TypeError, ValueError):
                            note_id_int = None
                        target_node = _resolve_note_parent_with_multihop(
                            session,
                            project_id,
                            user_id,
                            note_id_int,
                            parent_node_id,
                            parent_node_dict,
                        )
                if target_node is None:
                    target_node = nws.node
                else:
                    _merge_leaf_page_into_node_metadata(nws.node, target_node)
                node_id = getattr(target_node, "node_id", None) or leaf_meta.get("node_id")
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
        if not passages:
            return await _keyword_fallback_passages(
                session, project_id, user_id, query_text, k
            )
        return passages
    except Exception as e:
        logger.error("search_relevant_passages (notes): %s", e, exc_info=True)
        return []


async def search_relevant_notes(
    session: Session,
    project_id: int,
    query_text: str,
    user_id: int,
    k: int = 10,
) -> List[Dict]:
    project = get_project_by_id(session, project_id, user_id)
    if not project:
        return []
    if not query_text or not query_text.strip():
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
                results.append({"note": note, "score": float(passage.get("score", 0.0))})
        return results
    except Exception as e:
        logger.error("search_relevant_notes: %s", e, exc_info=True)
        return []
