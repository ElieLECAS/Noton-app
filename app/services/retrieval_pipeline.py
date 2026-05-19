"""
Pipeline retrieval lean : vectoriel + match exact refs + boost KAG + gating adaptatif.

Remplace RRF, lexical fuzzy, MMR, parent enrichment et multi-hop orchestrator.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

from llama_index.core.schema import NodeWithScore, TextNode
from sqlalchemy import or_, text
from sqlmodel import Session, select

from app.config import settings
from app.models.chunk_entity_relation import ChunkEntityRelation
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.models.knowledge_entity import KnowledgeEntity
from app.models.note import Note
from app.models.note_chunk import NoteChunk
from app.services.chunk_metadata_utils import (
    apply_row_metadata_defaults,
    merged_chunk_metadata,
)
from app.services.kag_extraction_service import normalize_entity_name

logger = logging.getLogger(__name__)

# --- Constantes pipeline ---
VECTOR_RETRIEVE_MAX = 20
EXACT_REF_LIMIT = 8
GATE_TAU = 0.70
GATE_MAX = 12
MIN_VECTOR_PREFILTER = 0.40
REF_EXACT_BOOST = 0.20
KAG_PIVOT_BOOST = 0.15
TITLE_OVERLAP_BOOST = 0.10
NUMERICAL_BOOST = 0.05
SMART_PARENT_SHORT_CHARS = 200
ADAPTIVE_GAP_MARGIN = 0.15
ONE_HOP_EXTRA_CAP = 4
MIN_ENTITY_CONFIDENCE = float(settings.MIN_ENTITY_CONFIDENCE)

_COMPARATIVE_KW = frozenset({
    "comparaison", "différence", "difference", "versus", "vs", "entre", "ou",
    "impact", "incompatible", "remplace",
})
_FACTUAL_MAX_TOKENS = 18
_REF_TOKEN = r"\b[A-Z]{1,5}\d{3,}[A-Z0-9]*\b"
_TECHNICAL_REF_PATTERN = re.compile(
    rf"{_REF_TOKEN}|\b\d{{2,5}}\s*(?:mm|cm|kg|kn)\b",
    re.IGNORECASE,
)
_PRODUCT_REF_PATTERN = re.compile(_REF_TOKEN, re.IGNORECASE)
_SHORT_COMPARATIVE_KW = frozenset({"ou", "vs"})
_DIMENSION_PATTERN = re.compile(
    r"\b(\d{1,4})\s*(?:à|-|–|/)\s*(\d{1,4})\s*(?:mm|cm)?\b|\b(\d{1,3})\s*mm\b",
    re.IGNORECASE,
)
_FALLBACK_STOPWORDS = {
    "the", "and", "for", "with", "dans", "avec", "pour", "une", "des", "les",
    "est", "sur", "pas", "plus", "que", "qui", "parle", "moi", "quel", "quelle",
}


@dataclass
class QueryAnalysis:
    text: str
    intent: Literal["factual", "comparative", "exploratory"]
    product_refs: List[str] = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    pivot_entities: List[str] = field(default_factory=list)
    needs_one_hop: bool = False


@dataclass
class RetrievalCandidate:
    node: TextNode
    chunk_id: int
    vector_similarity: float
    composite_score: float
    ref_exact_match: bool = False
    kag_match: Optional[str] = None

    def to_node_with_score(self) -> NodeWithScore:
        meta = dict(getattr(self.node, "metadata", {}) or {})
        meta["vector_similarity"] = self.vector_similarity
        meta["composite_score"] = self.composite_score
        if self.ref_exact_match:
            meta["ref_exact_match"] = True
        if self.kag_match:
            meta["kag_matched_entity"] = self.kag_match
        self.node.metadata = meta
        return NodeWithScore(node=self.node, score=self.composite_score)


@dataclass
class RetrievalStats:
    pool_after_merge: int = 0
    pool_after_gate: int = 0
    top1_vector_similarity: float = 0.0
    ref_match_used: bool = False
    kag_boost_used: bool = False
    one_hop_used: bool = False
    final_k: int = 0


def parse_chunk_id(node: TextNode) -> Optional[int]:
    meta = dict(getattr(node, "metadata", {}) or {})
    db_id = meta.get("chunk_db_id")
    if db_id is not None:
        try:
            return int(db_id)
        except (TypeError, ValueError):
            pass
    nid = getattr(node, "id_", None) or ""
    if isinstance(nid, str) and nid.startswith("chunk-"):
        try:
            return int(nid.split("-", 1)[1])
        except ValueError:
            return None
    return None


def _has_comparative_intent(q: str) -> bool:
    for kw in _COMPARATIVE_KW:
        if kw in _SHORT_COMPARATIVE_KW:
            if re.search(rf"\b{re.escape(kw)}\b", q, flags=re.IGNORECASE):
                return True
        elif kw in q:
            return True
    return False


def classify_query_intent(query_text: str) -> Literal["factual", "comparative", "exploratory"]:
    if not query_text or not query_text.strip():
        return "exploratory"
    q = query_text.strip().lower()
    if _TECHNICAL_REF_PATTERN.search(query_text):
        return "factual"
    if _has_comparative_intent(q):
        return "comparative"
    if len(query_text.split()) <= _FACTUAL_MAX_TOKENS:
        return "factual"
    return "exploratory"


def extract_product_refs(query_text: str) -> List[str]:
    refs = _PRODUCT_REF_PATTERN.findall(query_text or "")
    seen: Set[str] = set()
    out: List[str] = []
    for r in refs:
        key = r.upper()
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out[:8]


def extract_dimensions(query_text: str) -> List[str]:
    dims: List[str] = []
    for m in _DIMENSION_PATTERN.finditer(query_text or ""):
        if m.group(1) and m.group(2):
            dims.append(f"{m.group(1)}-{m.group(2)}")
        elif m.group(3):
            dims.append(m.group(3))
    return dims[:6]


def _normalize_pivots(names: Optional[List[str]]) -> List[str]:
    if not names:
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for name in names:
        value = normalize_entity_name(name) if name else ""
        if value and value not in seen:
            seen.add(value)
            out.append(value)
        if len(out) >= 12:
            break
    return out


def analyze_query(query_text: str, pivot_entities: Optional[List[str]] = None) -> QueryAnalysis:
    intent = classify_query_intent(query_text)
    refs = extract_product_refs(query_text)
    dims = extract_dimensions(query_text)
    pivots = _normalize_pivots(pivot_entities)
    if not pivots and settings.KAG_ENABLED:
        try:
            from app.services.kag_extraction_service import extract_entities_from_query_sync

            pivots = _normalize_pivots(extract_entities_from_query_sync(query_text))
        except Exception as e:
            logger.debug("Extraction entités requête ignorée: %s", e)
    needs_one_hop = intent == "comparative" and len(pivots) >= 2
    return QueryAnalysis(
        text=query_text.strip(),
        intent=intent,
        product_refs=refs,
        dimensions=dims,
        pivot_entities=pivots,
        needs_one_hop=needs_one_hop,
    )


def _title_overlap_ratio(query_text: str, document_title: str) -> float:
    if not query_text or not document_title:
        return 0.0

    def _words(s: str) -> Set[str]:
        n = unicodedata.normalize("NFD", s.lower())
        n = "".join(c for c in n if unicodedata.category(c) != "Mn")
        return {w for w in re.findall(r"[a-z0-9]+", n) if len(w) > 3 and w not in _FALLBACK_STOPWORDS}

    qw, tw = _words(query_text), _words(document_title)
    if not qw or not tw:
        return 0.0
    return len(qw & tw) / len(qw)


def _content_has_dimension(content: str, dimensions: List[str]) -> bool:
    if not content or not dimensions:
        return False
    lowered = content.lower()
    return any(d.lower() in lowered for d in dimensions)


def _content_has_ref(content: str, ref: str) -> bool:
    return ref.upper() in (content or "").upper()


def compute_composite_score(
    vector_similarity: float,
    qa: QueryAnalysis,
    *,
    ref_exact: bool = False,
    kag_match: bool = False,
    document_title: str = "",
    content: str = "",
) -> float:
    score = float(vector_similarity or 0.0)
    if ref_exact:
        score += REF_EXACT_BOOST
    if kag_match:
        score += KAG_PIVOT_BOOST
    overlap = _title_overlap_ratio(qa.text, document_title)
    score += TITLE_OVERLAP_BOOST * overlap
    if _content_has_dimension(content, qa.dimensions):
        score += NUMERICAL_BOOST
    return score


def _node_from_space_row(row) -> Tuple[TextNode, int, float]:
    metadata = merged_chunk_metadata(row.metadata_json, row.metadata_)
    apply_row_metadata_defaults(
        metadata,
        document_id=row.document_id,
        document_title=row.document_title or "Document sans titre",
        chunk_index=row.chunk_index,
        source=getattr(row, "chunk_source", None),
    )
    node = TextNode(
        id_=f"chunk-{row.id}",
        text=row.content or row.text or "",
        metadata=metadata,
    )
    return node, int(row.id), 1.0


def _node_from_note_row(row) -> Tuple[TextNode, int, float]:
    metadata = merged_chunk_metadata(row.metadata_json, row.metadata_)
    apply_row_metadata_defaults(
        metadata,
        note_id=row.note_id,
        note_title=row.note_title or "Note sans titre",
        node_id=row.node_id,
        parent_node_id=row.parent_node_id,
    )
    node = TextNode(
        id_=row.node_id or f"chunk-{row.id}",
        text=row.content or row.text or "",
        metadata=metadata,
    )
    return node, int(row.id), 1.0


def retrieve_exact_refs_space(
    session: Session,
    space_id: int,
    refs: List[str],
    limit: int = EXACT_REF_LIMIT,
) -> List[NodeWithScore]:
    if not refs:
        return []
    conditions = [
        or_(
            DocumentChunk.content.ilike(f"%{ref}%"),
            DocumentChunk.text.ilike(f"%{ref}%"),
        )
        for ref in refs[:6]
    ]
    stmt = (
        select(DocumentChunk, Document.title)
        .join(Document, Document.id == DocumentChunk.document_id)
        .join(DocumentSpace, DocumentSpace.document_id == Document.id)
        .where(
            DocumentSpace.space_id == space_id,
            DocumentChunk.is_leaf == True,
            or_(*conditions),
        )
        .limit(limit)
    )
    nodes: List[NodeWithScore] = []
    for chunk, title in session.exec(stmt).all():
        node, cid, _ = _node_from_space_row(
            type("R", (), {
                "id": chunk.id,
                "content": chunk.content,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "document_id": chunk.document_id,
                "metadata_json": chunk.metadata_json,
                "metadata_": chunk.metadata_,
                "chunk_source": chunk.source,
                "document_title": title,
            })()
        )
        nodes.append(NodeWithScore(node=node, score=0.85))
    if nodes:
        logger.info("Exact ref (space): %d chunks pour refs %s", len(nodes), refs[:3])
    return nodes


def retrieve_exact_refs_note(
    session: Session,
    project_id: int,
    user_id: int,
    refs: List[str],
    limit: int = EXACT_REF_LIMIT,
) -> List[NodeWithScore]:
    if not refs:
        return []
    conditions = [
        or_(
            NoteChunk.content.ilike(f"%{ref}%"),
            NoteChunk.text.ilike(f"%{ref}%"),
        )
        for ref in refs[:6]
    ]
    stmt = (
        select(NoteChunk, Note.title)
        .join(Note, Note.id == NoteChunk.note_id)
        .where(
            Note.project_id == project_id,
            Note.user_id == user_id,
            NoteChunk.is_leaf == True,
            or_(*conditions),
        )
        .limit(limit)
    )
    nodes: List[NodeWithScore] = []
    for chunk, title in session.exec(stmt).all():
        metadata = merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        apply_row_metadata_defaults(
            metadata,
            note_id=chunk.note_id,
            note_title=title or "Note sans titre",
            node_id=chunk.node_id,
            parent_node_id=chunk.parent_node_id,
        )
        metadata["chunk_db_id"] = chunk.id
        node = TextNode(
            id_=chunk.node_id or f"chunk-{chunk.id}",
            text=chunk.content or chunk.text or "",
            metadata=metadata,
        )
        nodes.append(NodeWithScore(node=node, score=0.85))
    if nodes:
        logger.info("Exact ref (note): %d chunks pour refs %s", len(nodes), refs[:3])
    return nodes


def merge_vector_and_exact(
    vector_nodes: List[NodeWithScore],
    exact_nodes: List[NodeWithScore],
    qa: QueryAnalysis,
) -> List[RetrievalCandidate]:
    by_id: Dict[int, RetrievalCandidate] = {}

    for nws in vector_nodes:
        cid = parse_chunk_id(nws.node)
        if cid is None:
            continue
        vec = float(nws.score or 0.0)
        if vec < MIN_VECTOR_PREFILTER and not qa.product_refs:
            continue
        meta = dict(getattr(nws.node, "metadata", {}) or {})
        content = nws.node.get_content() if hasattr(nws.node, "get_content") else ""
        ref_hit = any(_content_has_ref(content, r) for r in qa.product_refs)
        title = meta.get("document_title") or meta.get("note_title") or ""
        comp = compute_composite_score(
            vec, qa, ref_exact=ref_hit, document_title=title, content=content
        )
        by_id[cid] = RetrievalCandidate(
            node=nws.node,
            chunk_id=cid,
            vector_similarity=vec,
            composite_score=comp,
            ref_exact_match=ref_hit,
        )

    for nws in exact_nodes:
        cid = parse_chunk_id(nws.node)
        if cid is None:
            continue
        content = nws.node.get_content() if hasattr(nws.node, "get_content") else ""
        meta = dict(getattr(nws.node, "metadata", {}) or {})
        title = meta.get("document_title") or meta.get("note_title") or ""
        ref_hit = True
        vec = float(by_id[cid].vector_similarity) if cid in by_id else 0.55
        comp = compute_composite_score(
            vec, qa, ref_exact=True, document_title=title, content=content
        )
        if cid in by_id:
            prev = by_id[cid]
            prev.ref_exact_match = True
            prev.composite_score = max(prev.composite_score, comp)
        else:
            by_id[cid] = RetrievalCandidate(
                node=nws.node,
                chunk_id=cid,
                vector_similarity=vec,
                composite_score=comp,
                ref_exact_match=True,
            )

    merged = sorted(by_id.values(), key=lambda c: c.composite_score, reverse=True)
    return merged[:VECTOR_RETRIEVE_MAX]


def annotate_kag_matches_space(
    session: Session,
    space_id: int,
    candidates: List[RetrievalCandidate],
    pivot_entities: List[str],
) -> bool:
    if not pivot_entities or not candidates:
        return False
    chunk_ids = [c.chunk_id for c in candidates]
    stmt = (
        select(ChunkEntityRelation.chunk_id, KnowledgeEntity.name)
        .join(KnowledgeEntity, KnowledgeEntity.id == ChunkEntityRelation.entity_id)
        .where(
            KnowledgeEntity.space_id == space_id,
            ChunkEntityRelation.chunk_id.in_(chunk_ids),
            KnowledgeEntity.name_normalized.in_(pivot_entities),
            KnowledgeEntity.confidence_score >= MIN_ENTITY_CONFIDENCE,
        )
    )
    matches = {row[0]: row[1] for row in session.exec(stmt).all()}
    used = False
    for cand in candidates:
        if cand.chunk_id in matches:
            cand.kag_match = matches[cand.chunk_id]
            cand.composite_score += KAG_PIVOT_BOOST
            used = True
    return used


def annotate_kag_matches_note(
    session: Session,
    project_id: int,
    user_id: int,
    candidates: List[RetrievalCandidate],
    pivot_entities: List[str],
) -> bool:
    if not pivot_entities or not candidates:
        return False
    from app.services.kag_graph_service import get_chunks_by_entity_names

    try:
        results = get_chunks_by_entity_names(
            session=session,
            entity_names=pivot_entities,
            project_id=project_id,
            user_id=user_id,
            limit=len(candidates) + 10,
        )
    except Exception as e:
        logger.debug("KAG boost note ignoré: %s", e)
        return False
    match_ids = {r["chunk"].id: r.get("entity_name", "") for r in results}
    used = False
    for cand in candidates:
        if cand.chunk_id in match_ids:
            cand.kag_match = match_ids[cand.chunk_id]
            cand.composite_score += KAG_PIVOT_BOOST
            used = True
    return used


def adaptive_gate(
    candidates: List[RetrievalCandidate],
    tau: float = GATE_TAU,
    max_n: int = GATE_MAX,
) -> List[RetrievalCandidate]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c.composite_score, reverse=True)
    top = ordered[0].composite_score
    if top <= 0:
        return ordered[:max_n]
    floor = top * tau
    kept = [c for c in ordered if c.composite_score >= floor]
    return kept[:max_n]


def expand_one_hop_space(
    session: Session,
    space_id: int,
    candidates: List[RetrievalCandidate],
    qa: QueryAnalysis,
    extra_cap: int = ONE_HOP_EXTRA_CAP,
) -> List[RetrievalCandidate]:
    if not qa.needs_one_hop or len(qa.pivot_entities) < 2:
        return candidates
    pivots = qa.pivot_entities[:6]
    if len(pivots) < 2:
        return candidates

    conditions = []
    for p in pivots:
        conditions.append(
            or_(
                DocumentChunk.content.ilike(f"%{p}%"),
                DocumentChunk.text.ilike(f"%{p}%"),
            )
        )
    stmt = (
        select(DocumentChunk, Document.title)
        .join(Document, Document.id == DocumentChunk.document_id)
        .join(DocumentSpace, DocumentSpace.document_id == Document.id)
        .where(
            DocumentSpace.space_id == space_id,
            DocumentChunk.is_leaf == True,
            or_(*conditions),
        )
        .limit(extra_cap * 4)
    )
    seen = {c.chunk_id for c in candidates}
    extras: List[RetrievalCandidate] = []
    pivot_lower = [p.lower() for p in pivots]

    for chunk, title in session.exec(stmt).all():
        if chunk.id in seen:
            continue
        content = (chunk.content or chunk.text or "").lower()
        hits = sum(1 for p in pivot_lower if p in content)
        if hits < 2:
            continue
        metadata = merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        apply_row_metadata_defaults(
            metadata,
            document_id=chunk.document_id,
            document_title=title or "Document sans titre",
            chunk_index=chunk.chunk_index,
            source=chunk.source,
        )
        node = TextNode(id_=f"chunk-{chunk.id}", text=chunk.content or chunk.text or "", metadata=metadata)
        avg_vec = sum(c.vector_similarity for c in candidates) / max(len(candidates), 1)
        comp = compute_composite_score(
            avg_vec * 0.9,
            qa,
            document_title=title or "",
            content=content,
        )
        extras.append(
            RetrievalCandidate(
                node=node,
                chunk_id=int(chunk.id),
                vector_similarity=avg_vec * 0.85,
                composite_score=comp,
            )
        )
        seen.add(chunk.id)
        if len(extras) >= extra_cap:
            break

    if extras:
        logger.info("One-hop (space): +%d chunks multi-pivot", len(extras))
    merged = sorted(candidates + extras, key=lambda c: c.composite_score, reverse=True)
    return merged[:GATE_MAX + extra_cap]


def sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def adaptive_top_n(
    reranked: List[NodeWithScore],
    k_max: int = 6,
    gap_margin: float = ADAPTIVE_GAP_MARGIN,
) -> List[NodeWithScore]:
    if not reranked:
        return []
    ranked = sorted(reranked, key=lambda n: float(n.score or 0), reverse=True)
    norm_scores = [sigmoid(float(n.score or 0)) for n in ranked]
    if not norm_scores:
        return []
    top_score = norm_scores[0]
    kept = [ranked[0]]
    for i in range(1, min(len(ranked), k_max)):
        if top_score - norm_scores[i] > gap_margin:
            break
        kept.append(ranked[i])
    return kept


def smart_parent_or_leaf(
    leaf_node: TextNode,
    parent_node_dict: Dict[str, TextNode],
    *,
    resolve_parent_fn=None,
) -> TextNode:
    meta = dict(getattr(leaf_node, "metadata", {}) or {})
    content_type = (meta.get("content_type") or "").lower()
    content = (
        leaf_node.get_content()
        if hasattr(leaf_node, "get_content")
        else getattr(leaf_node, "text", "") or ""
    )
    parent_node_id = meta.get("parent_node_id")

    promote = content_type in ("table_row", "table_cell", "table_summary")
    if not promote and len(content.strip()) < SMART_PARENT_SHORT_CHARS:
        promote = True

    if not promote or not parent_node_id:
        return leaf_node

    target = parent_node_dict.get(str(parent_node_id))
    if target is None and resolve_parent_fn is not None:
        target = resolve_parent_fn(parent_node_id, leaf_node)
    return target if target is not None else leaf_node


def log_retrieval_stats(stats: RetrievalStats, scope: str) -> None:
    logger.info(
        "Retrieval [%s]: pool_merge=%d gate=%d top1_vec=%.3f ref=%s kag=%s hop=%s final_k=%d",
        scope,
        stats.pool_after_merge,
        stats.pool_after_gate,
        stats.top1_vector_similarity,
        stats.ref_match_used,
        stats.kag_boost_used,
        stats.one_hop_used,
        stats.final_k,
    )
