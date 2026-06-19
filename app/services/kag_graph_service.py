"""Construction des données graphe KAG pour l'UI (modale chunks + espace)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from sqlalchemy import text
from sqlmodel import Session, select

from app.config import settings
from app.models.document_chunk import DocumentChunk
from app.models.knowledge_entity import (
    ChunkEntityRelation,
    EntityEntityRelation,
    KnowledgeEntity,
)


_ENTITY_TYPE_COLORS = {
    "product": "#6366f1",
    "material": "#10b981",
    "tool": "#f59e0b",
    "norm": "#ef4444",
    "dimension": "#8b5cf6",
    "process": "#06b6d4",
    "organization": "#ec4899",
    "location": "#14b8a6",
    "reference": "#64748b",
    "other": "#9ca3af",
}


def _entity_color(entity_type: str) -> str:
    return _ENTITY_TYPE_COLORS.get((entity_type or "other").lower(), "#9ca3af")


def get_document_chunk_entities(session: Session, document_id: int) -> Dict[str, Any]:
    """
    Entités KAG liées aux chunks d'un document, indexées par chunk_id.
  """
    if not settings.KAG_ENABLED:
        return {
            "entity_count": 0,
            "relation_count": 0,
            "chunk_entities": {},
            "document_relations": [],
        }

    rows = session.execute(
        text(
            """
            SELECT
                cer.chunk_id,
                ke.id AS entity_id,
                ke.name,
                ke.entity_type,
                cer.relation_role,
                cer.relevance_score,
                cer.context_snippet
            FROM chunkentityrelation cer
            INNER JOIN documentchunk dc ON dc.id = cer.chunk_id
            INNER JOIN knowledgeentity ke ON ke.id = cer.entity_id
            WHERE dc.document_id = :document_id
            ORDER BY cer.chunk_id, ke.name
            """
        ),
        {"document_id": document_id},
    ).all()

    chunk_entities: Dict[int, List[dict]] = defaultdict(list)
    seen_per_chunk: Dict[int, Set[int]] = defaultdict(set)
    entity_ids: Set[int] = set()

    for chunk_id, entity_id, name, entity_type, role, score, snippet in rows:
        entity_ids.add(int(entity_id))
        if int(entity_id) in seen_per_chunk[int(chunk_id)]:
            continue
        seen_per_chunk[int(chunk_id)].add(int(entity_id))
        chunk_entities[int(chunk_id)].append(
            {
                "entity_id": int(entity_id),
                "name": name,
                "entity_type": entity_type,
                "relation_role": role,
                "relevance_score": float(score or 0.0),
                "context_snippet": snippet,
            }
        )

    rel_rows = session.execute(
        text(
            """
            SELECT DISTINCT
                eer.entity_a_id,
                eer.entity_b_id,
                eer.relation_type,
                eer.relation_label,
                eer.confidence,
                ke_a.name AS entity_a_name,
                ke_b.name AS entity_b_name
            FROM entityentityrelation eer
            INNER JOIN knowledgeentity ke_a ON ke_a.id = eer.entity_a_id
            INNER JOIN knowledgeentity ke_b ON ke_b.id = eer.entity_b_id
            WHERE eer.source_chunk_id IN (
                SELECT id FROM documentchunk WHERE document_id = :document_id
            )
            ORDER BY eer.relation_type, ke_a.name
            LIMIT 200
            """
        ),
        {"document_id": document_id},
    ).all()

    document_relations = [
        {
            "entity_a_id": int(a_id),
            "entity_b_id": int(b_id),
            "entity_a_name": a_name,
            "entity_b_name": b_name,
            "relation_type": rel_type,
            "relation_label": label,
            "confidence": float(conf or 0.0) if conf is not None else None,
        }
        for a_id, b_id, rel_type, label, conf, a_name, b_name in rel_rows
    ]

    return {
        "entity_count": len(entity_ids),
        "relation_count": len(document_relations),
        "chunk_entities": {str(k): v for k, v in chunk_entities.items()},
        "document_relations": document_relations,
    }


def count_document_kag_stats(session: Session, document_id: int) -> Dict[str, int]:
    """Compteurs KAG pour le snapshot document."""
    if not settings.KAG_ENABLED:
        return {"knowledge_entity_count": 0, "entity_relation_count": 0}

    entity_count = session.execute(
        text(
            """
            SELECT COUNT(DISTINCT cer.entity_id)
            FROM chunkentityrelation cer
            INNER JOIN documentchunk dc ON dc.id = cer.chunk_id
            WHERE dc.document_id = :document_id
            """
        ),
        {"document_id": document_id},
    ).scalar()

    relation_count = session.execute(
        text(
            """
            SELECT COUNT(*)
            FROM entityentityrelation eer
            WHERE eer.source_chunk_id IN (
                SELECT id FROM documentchunk WHERE document_id = :document_id
            )
            """
        ),
        {"document_id": document_id},
    ).scalar()

    return {
        "knowledge_entity_count": int(entity_count or 0),
        "entity_relation_count": int(relation_count or 0),
    }


def build_space_kag_graph(
    session: Session,
    space_id: int,
    *,
    max_nodes: int = 150,
    max_edges: int = 300,
) -> Dict[str, Any]:
    """
    Graphe entités/relations pour visualisation dans un espace.
    Limite le nombre de nœuds/arêtes pour rester fluide côté UI.
    """
    if not settings.KAG_ENABLED:
        return {
            "space_id": space_id,
            "node_count": 0,
            "edge_count": 0,
            "nodes": [],
            "edges": [],
            "status": "disabled",
        }

    entities = list(
        session.exec(
            select(KnowledgeEntity)
            .where(KnowledgeEntity.space_id == space_id)
            .order_by(KnowledgeEntity.mention_count.desc())
            .limit(max_nodes)
        ).all()
    )

    if not entities:
        return {
            "space_id": space_id,
            "node_count": 0,
            "edge_count": 0,
            "nodes": [],
            "edges": [],
            "status": "empty",
        }

    entity_ids = {e.id for e in entities if e.id is not None}
    nodes = [
        {
            "id": str(e.id),
            "label": e.name,
            "entity_type": e.entity_type,
            "mention_count": e.mention_count,
            "description": e.description,
            "color": _entity_color(e.entity_type),
        }
        for e in entities
        if e.id is not None
    ]

    rels = list(
        session.exec(
            select(EntityEntityRelation)
            .where(EntityEntityRelation.space_id == space_id)
            .order_by(EntityEntityRelation.weight.desc())
            .limit(max_edges * 2)
        ).all()
    )

    edges: List[dict] = []
    seen_edges: Set[str] = set()
    for rel in rels:
        if rel.entity_a_id not in entity_ids or rel.entity_b_id not in entity_ids:
            continue
        key = f"{rel.entity_a_id}:{rel.entity_b_id}:{rel.relation_type}"
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append(
            {
                "id": str(rel.id),
                "source": str(rel.entity_a_id),
                "target": str(rel.entity_b_id),
                "relation_type": rel.relation_type,
                "label": rel.relation_label or rel.relation_type.replace("_", " "),
                "weight": rel.weight,
                "confidence": rel.confidence,
            }
        )
        if len(edges) >= max_edges:
            break

    total_entities = session.execute(
        text("SELECT COUNT(*) FROM knowledgeentity WHERE space_id = :space_id"),
        {"space_id": space_id},
    ).scalar()
    total_relations = session.execute(
        text("SELECT COUNT(*) FROM entityentityrelation WHERE space_id = :space_id"),
        {"space_id": space_id},
    ).scalar()

    return {
        "space_id": space_id,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "total_entity_count": int(total_entities or 0),
        "total_relation_count": int(total_relations or 0),
        "nodes": nodes,
        "edges": edges,
        "status": "ok",
    }
