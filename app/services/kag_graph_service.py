"""
Service de gestion du graphe de connaissances KAG.

Gère les opérations CRUD sur les entités et relations chunk-entité,
ainsi que le lookup pour enrichir le retrieval RAG.
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Set
from sqlmodel import Session, select
from sqlalchemy import func, delete
from app.models.knowledge_entity import KnowledgeEntity
from app.models.chunk_entity_relation import ChunkEntityRelation
from app.models.note_chunk import NoteChunk
from app.models.note import Note
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.services.kag_extraction_service import (
    normalize_entity_name,
    SUPPORTED_ENTITY_TYPE_IDS,
    extract_entities_sync,
)

logger = logging.getLogger(__name__)


def _canonicalize_entities(entities: List[Dict]) -> List[Dict]:
    """
    Normalise/déduplique/sort les entités pour garantir un résultat stable.
    """
    canonical: Dict[tuple[str, str], Dict] = {}
    for entity_data in entities or []:
        raw_name = (entity_data.get("name") or "").strip()
        raw_type = (entity_data.get("type") or "concept_technique").strip()
        if not raw_name or len(raw_name) < 2:
            continue
        key = (normalize_entity_name(raw_name), raw_type)
        if not key[0]:
            continue

        importance = entity_data.get("importance", 1.0)
        try:
            importance = float(importance)
        except Exception:
            importance = 1.0
        importance = max(0.0, min(1.0, importance))

        existing = canonical.get(key)
        if existing is None or importance > float(existing.get("importance", 0.0)):
            canonical[key] = {
                "name": raw_name,
                "type": raw_type,
                "importance": importance,
            }

    return sorted(
        canonical.values(),
        key=lambda x: (normalize_entity_name(x.get("name", "")), x.get("type", "")),
    )


def _get_or_compute_chunk_entities(
    session: Session,
    chunk: DocumentChunk,
    content: str,
) -> List[Dict]:
    """
    Récupère les entités canoniques depuis metadata_json si présentes,
    sinon les calcule une seule fois puis les persiste.
    """
    metadata = dict(chunk.metadata_json or {})
    cached_entities = metadata.get("kag_entities")

    if isinstance(cached_entities, list) and cached_entities:
        return _canonicalize_entities(cached_entities)

    entities = _canonicalize_entities(extract_entities_sync(content))
    metadata["kag_entities"] = entities
    chunk.metadata_json = metadata
    chunk.metadata_ = metadata
    session.add(chunk)
    return entities


def get_or_create_entity(
    session: Session,
    name: str,
    entity_type: str,
    project_id: int,
) -> KnowledgeEntity:
    """
    Récupère ou crée une entité dans le graphe de connaissances.
    La déduplication se fait sur (project_id, name_normalized).
    """
    name_normalized = normalize_entity_name(name)
    
    statement = select(KnowledgeEntity).where(
        KnowledgeEntity.project_id == project_id,
        KnowledgeEntity.name_normalized == name_normalized,
    )
    entity = session.exec(statement).first()
    
    if entity:
        entity.mention_count += 1
        entity.updated_at = datetime.utcnow()
        session.add(entity)
        return entity
    
    entity = KnowledgeEntity(
        name=name,
        name_normalized=name_normalized,
        entity_type=entity_type,
        project_id=project_id,
        mention_count=1,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    session.add(entity)
    session.flush()
    return entity


def save_entities_for_chunk(
    session: Session,
    chunk: NoteChunk,
    entities: List[Dict],
    project_id: int,
) -> int:
    """
    Sauvegarde les entités extraites pour un chunk et crée les relations.
    
    Args:
        session: Session SQLModel
        chunk: Le chunk source
        entities: Liste de dicts {"name": str, "type": str, "importance": float}
        project_id: ID du projet
        
    Returns:
        Nombre de relations créées
    """
    if not entities or not chunk.id:
        return 0
    
    relations_created = 0
    
    for entity_data in entities:
        name = entity_data.get("name", "").strip()
        entity_type = entity_data.get("type", "composant")
        importance = entity_data.get("importance", 1.0)
        
        if not name or len(name) < 2:
            continue
        
        entity = get_or_create_entity(session, name, entity_type, project_id)
        
        existing_rel = session.exec(
            select(ChunkEntityRelation).where(
                ChunkEntityRelation.chunk_id == chunk.id,
                ChunkEntityRelation.entity_id == entity.id,
            )
        ).first()
        
        if not existing_rel:
            relation = ChunkEntityRelation(
                chunk_id=chunk.id,
                entity_id=entity.id,
                relevance_score=importance,
                project_id=project_id,
                created_at=datetime.utcnow(),
            )
            session.add(relation)
            relations_created += 1
    
    return relations_created


def delete_entities_for_note(session: Session, note_id: int) -> int:
    """
    Supprime les relations chunk-entité pour une note.
    Les entités elles-mêmes sont conservées (peuvent être référencées ailleurs).
    
    Args:
        session: Session SQLModel
        note_id: ID de la note
        
    Returns:
        Nombre de relations supprimées
    """
    chunk_ids_stmt = select(NoteChunk.id).where(NoteChunk.note_id == note_id)
    chunk_ids = [row for row in session.exec(chunk_ids_stmt).all()]
    
    if not chunk_ids:
        return 0
    
    delete_stmt = delete(ChunkEntityRelation).where(
        ChunkEntityRelation.chunk_id.in_(chunk_ids)
    )
    result = session.execute(delete_stmt)
    deleted_count = result.rowcount
    
    logger.debug("Supprimé %d relations KAG pour note_id=%d", deleted_count, note_id)
    return deleted_count


def delete_entities_for_project(session: Session, project_id: int) -> Dict[str, int]:
    """
    Supprime toutes les entités et relations KAG pour un projet.
    
    Args:
        session: Session SQLModel
        project_id: ID du projet
        
    Returns:
        Dict avec le nombre d'entités et relations supprimées
    """
    relations_deleted = session.execute(
        delete(ChunkEntityRelation).where(ChunkEntityRelation.project_id == project_id)
    ).rowcount
    
    entities_deleted = session.execute(
        delete(KnowledgeEntity).where(KnowledgeEntity.project_id == project_id)
    ).rowcount
    
    logger.info(
        "Supprimé KAG pour project_id=%d: %d entités, %d relations",
        project_id,
        entities_deleted,
        relations_deleted,
    )
    return {"entities_deleted": entities_deleted, "relations_deleted": relations_deleted}


def rebuild_kag_for_project(session: Session, project_id: int) -> Dict[str, int]:
    """
    Purge et reconstruit le graphe KAG pour un projet donné.

    - Supprime toutes les entités et relations existantes du projet
    - Recrée les chunks + embeddings + entités KAG pour toutes les notes du projet
    """
    from app.models.note import Note
    from app.services.chunk_service import reindex_notes

    # Purge complète des entités / relations KAG
    deletion_stats = delete_entities_for_project(session, project_id)

    # Réindexer toutes les notes du projet ; le pipeline de chunking ré‑utilise
    # la configuration actuelle (nouvelle taxonomie KAG)
    reindexed_notes_count = reindex_notes(session, project_id=project_id)

    return {
        "entities_deleted": deletion_stats.get("entities_deleted", 0),
        "relations_deleted": deletion_stats.get("relations_deleted", 0),
        "notes_reindexed": reindexed_notes_count,
    }


def get_chunks_by_entity_names(
    session: Session,
    entity_names: List[str],
    project_id: int,
    user_id: int,
    limit: int = 20,
) -> List[Dict]:
    """
    Récupère les chunks liés aux entités spécifiées.
    
    Args:
        session: Session SQLModel
        entity_names: Liste des noms d'entités (seront normalisés)
        project_id: ID du projet
        user_id: ID de l'utilisateur (pour vérification)
        limit: Nombre max de chunks à retourner
        
    Returns:
        Liste de dicts {"chunk": NoteChunk, "entity_name": str, "relevance_score": float}
    """
    if not entity_names:
        return []
    
    normalized_names = [normalize_entity_name(name) for name in entity_names if name]
    normalized_names = [n for n in normalized_names if n]
    
    if not normalized_names:
        return []
    
    statement = (
        select(
            NoteChunk,
            KnowledgeEntity.name,
            ChunkEntityRelation.relevance_score,
        )
        .join(ChunkEntityRelation, ChunkEntityRelation.chunk_id == NoteChunk.id)
        .join(KnowledgeEntity, KnowledgeEntity.id == ChunkEntityRelation.entity_id)
        .join(Note, Note.id == NoteChunk.note_id)
        .where(
            KnowledgeEntity.project_id == project_id,
            KnowledgeEntity.name_normalized.in_(normalized_names),
            Note.user_id == user_id,
            # Inclut les chunks parents enrichis (is_leaf=False) ET les leaves classiques.
            # Les parents trouvés via KAG contiennent le résumé + questions de leur section
            # et sont passés directement au LLM comme contexte de section.
        )
        .order_by(ChunkEntityRelation.relevance_score.desc())
        .limit(limit)
    )
    
    results = session.exec(statement).all()
    
    return [
        {
            "chunk": chunk,
            "entity_name": entity_name,
            "relevance_score": score,
        }
        for chunk, entity_name, score in results
    ]


def get_entities_for_project(
    session: Session,
    project_id: int,
    entity_type: Optional[str] = None,
    limit: int = 100,
) -> List[KnowledgeEntity]:
    """
    Récupère les entités d'un projet, triées par nombre de mentions.
    
    Args:
        session: Session SQLModel
        project_id: ID du projet
        entity_type: Filtrer par type (optionnel)
        limit: Nombre max d'entités
        
    Returns:
        Liste des entités
    """
    statement = select(KnowledgeEntity).where(
        KnowledgeEntity.project_id == project_id
    )
    
    if entity_type:
        statement = statement.where(KnowledgeEntity.entity_type == entity_type)
    
    statement = statement.order_by(
        KnowledgeEntity.mention_count.desc()
    ).limit(limit)
    
    return list(session.exec(statement).all())


def get_related_chunks_via_graph(
    session: Session,
    seed_chunk_ids: List[int],
    project_id: int,
    user_id: int,
    max_hops: int = 1,
    limit: int = 10,
) -> List[NoteChunk]:
    """
    Trouve des chunks liés aux chunks seeds via le graphe d'entités (traversée).
    
    Args:
        session: Session SQLModel
        seed_chunk_ids: IDs des chunks de départ
        project_id: ID du projet
        user_id: ID de l'utilisateur
        max_hops: Nombre de sauts max dans le graphe (1 = entités directes)
        limit: Nombre max de chunks à retourner
        
    Returns:
        Liste des chunks liés (hors seeds)
    """
    if not seed_chunk_ids:
        return []
    
    entity_ids_stmt = select(ChunkEntityRelation.entity_id).where(
        ChunkEntityRelation.chunk_id.in_(seed_chunk_ids)
    ).distinct()
    entity_ids = list(session.exec(entity_ids_stmt).all())
    
    if not entity_ids:
        return []
    
    related_stmt = (
        select(NoteChunk)
        .join(ChunkEntityRelation, ChunkEntityRelation.chunk_id == NoteChunk.id)
        .join(Note, Note.id == NoteChunk.note_id)
        .where(
            ChunkEntityRelation.entity_id.in_(entity_ids),
            ChunkEntityRelation.chunk_id.notin_(seed_chunk_ids),
            Note.project_id == project_id,
            Note.user_id == user_id,
            NoteChunk.is_leaf == True,
        )
        .order_by(ChunkEntityRelation.relevance_score.desc())
        .limit(limit)
    )
    
    return list(session.exec(related_stmt).all())


def get_kag_stats(session: Session, project_id: int) -> Dict:
    """
    Retourne les statistiques KAG pour un projet.
    """
    entity_count = session.exec(
        select(func.count(KnowledgeEntity.id)).where(
            KnowledgeEntity.project_id == project_id
        )
    ).one()
    
    relation_count = session.exec(
        select(func.count(ChunkEntityRelation.id)).where(
            ChunkEntityRelation.project_id == project_id
        )
    ).one()
    
    type_counts: Dict[str, int] = {}
    for entity_type in SUPPORTED_ENTITY_TYPE_IDS:
        count = session.exec(
            select(func.count(KnowledgeEntity.id)).where(
                KnowledgeEntity.project_id == project_id,
                KnowledgeEntity.entity_type == entity_type,
            )
        ).one()
        type_counts[entity_type] = count
    
    return {
        "total_entities": entity_count,
        "total_relations": relation_count,
        "entities_by_type": type_counts,
    }


def get_project_bipartite_graph(
    session: Session,
    project_id: int,
    user_id: int,
    max_entities: int = 50,
    max_relations: int = 400,
) -> Dict[str, List[Dict]]:
    """
    Construit un graphe biparti (entités <-> chunks) pour un projet.

    Le graphe est limité à un nombre raisonnable d'entités et de relations
    pour rester lisible dans une visualisation front.
    """
    # Récupérer les entités les plus mentionnées du projet
    entities = get_entities_for_project(
        session=session,
        project_id=project_id,
        limit=max_entities,
    )
    if not entities:
        return {"nodes": [], "edges": []}

    entity_ids = [e.id for e in entities]
    if not entity_ids:
        return {"nodes": [], "edges": []}

    # Récupérer les relations chunk-entité + note associée, filtrées par utilisateur
    stmt = (
        select(
            ChunkEntityRelation,
            NoteChunk,
            KnowledgeEntity,
            Note,
        )
        .join(NoteChunk, NoteChunk.id == ChunkEntityRelation.chunk_id)
        .join(KnowledgeEntity, KnowledgeEntity.id == ChunkEntityRelation.entity_id)
        .join(Note, Note.id == NoteChunk.note_id)
        .where(
            ChunkEntityRelation.project_id == project_id,
            Note.user_id == user_id,
            ChunkEntityRelation.entity_id.in_(entity_ids),
        )
        .order_by(ChunkEntityRelation.relevance_score.desc())
        .limit(max_relations)
    )

    rows = session.exec(stmt).all()
    if not rows:
        return {"nodes": [], "edges": []}

    nodes: Dict[str, Dict] = {}
    edges: List[Dict] = []

    def add_entity_node(entity: KnowledgeEntity) -> str:
        node_id = f"entity-{entity.id}"
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "kind": "entity",
                "entity_id": entity.id,
                "label": entity.name,
                "entity_type": entity.entity_type,
                "mention_count": entity.mention_count,
            }
        return node_id

    def add_chunk_node(chunk: NoteChunk, note: Note) -> str:
        node_id = f"chunk-{chunk.id}"
        if node_id not in nodes:
            title = note.title or f"Note {note.id}"
            label = title
            # Prévisualisation courte du contenu du chunk pour l'UI
            preview = (chunk.content or "").strip()
            if preview:
                # On tronque pour éviter un payload trop lourd
                max_len = 320
                if len(preview) > max_len:
                    cut = preview[:max_len]
                    # Couper proprement sur un espace si possible
                    last_space = cut.rfind(" ")
                    if last_space > 40:
                        cut = cut[:last_space]
                    preview = cut + "…"
            nodes[node_id] = {
                "id": node_id,
                "kind": "chunk",
                "chunk_id": chunk.id,
                "note_id": note.id,
                "note_title": title,
                "label": label,
                "chunk_index": chunk.chunk_index,
                # On garde seulement un extrait court pour l'UI
                "preview": preview or None,
            }
        return node_id

    for rel, chunk, entity, note in rows:
        entity_node_id = add_entity_node(entity)
        chunk_node_id = add_chunk_node(chunk, note)

        edges.append(
            {
                "id": f"rel-{rel.id}",
                "from": entity_node_id,
                "to": chunk_node_id,
                "relevance": rel.relevance_score,
            }
        )

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# Compatibilité architecture "Document / Space"
# ---------------------------------------------------------------------------

def _get_or_create_entity_for_space(
    session: Session,
    name: str,
    entity_type: str,
    space_id: int,
) -> KnowledgeEntity:
    """Version espace de get_or_create_entity (déduplication par space_id + name_normalized)."""
    name_normalized = normalize_entity_name(name)
    statement = select(KnowledgeEntity).where(
        KnowledgeEntity.space_id == space_id,
        KnowledgeEntity.name_normalized == name_normalized,
    )
    entity = session.exec(statement).first()
    if entity:
        entity.mention_count += 1
        entity.updated_at = datetime.utcnow()
        session.add(entity)
        return entity

    entity = KnowledgeEntity(
        name=name,
        name_normalized=name_normalized,
        entity_type=entity_type,
        space_id=space_id,
        mention_count=1,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    session.add(entity)
    session.flush()
    return entity


def process_kag_for_document_space(session: Session, document_id: int, space_id: int) -> Dict[str, int]:
    """
    Extrait et sauvegarde les entités KAG pour un document dans un espace donné.
    """
    document = session.get(Document, document_id)
    if not document:
        logger.warning("Document introuvable pour KAG: document_id=%s", document_id)
        return {"entities": 0, "relations": 0, "chunks": 0}

    association = session.exec(
        select(DocumentSpace).where(
            DocumentSpace.document_id == document_id,
            DocumentSpace.space_id == space_id,
        )
    ).first()
    if not association:
        logger.info(
            "Aucune association document-espace, skip KAG document_id=%s space_id=%s",
            document_id,
            space_id,
        )
        return {"entities": 0, "relations": 0, "chunks": 0}

    delete_entities_for_document(session, document_id, space_id)

    chunks_stmt = select(DocumentChunk).where(
        DocumentChunk.document_id == document_id,
        DocumentChunk.is_leaf == True,
    )
    chunks = list(session.exec(chunks_stmt).all())
    if not chunks:
        return {"entities": 0, "relations": 0, "chunks": 0}

    total_entities = 0
    total_relations = 0
    processed_chunks = 0

    for chunk in chunks:
        content = (chunk.content or "").strip()
        if len(content) < 20:
            continue

        entities = _get_or_compute_chunk_entities(session, chunk, content)
        if not entities:
            continue

        processed_chunks += 1
        for entity_data in entities:
            name = (entity_data.get("name") or "").strip()
            entity_type = (entity_data.get("type") or "concept_technique").strip()
            importance = float(entity_data.get("importance", 1.0) or 1.0)
            if not name or len(name) < 2:
                continue

            entity = _get_or_create_entity_for_space(
                session=session,
                name=name,
                entity_type=entity_type,
                space_id=space_id,
            )
            total_entities += 1

            existing_rel = session.exec(
                select(ChunkEntityRelation).where(
                    ChunkEntityRelation.chunk_id == chunk.id,
                    ChunkEntityRelation.entity_id == entity.id,
                    ChunkEntityRelation.space_id == space_id,
                )
            ).first()
            if not existing_rel:
                relation = ChunkEntityRelation(
                    chunk_id=chunk.id,
                    entity_id=entity.id,
                    relevance_score=importance,
                    space_id=space_id,
                    created_at=datetime.utcnow(),
                )
                session.add(relation)
                total_relations += 1

    session.commit()
    logger.info(
        "✅ KAG document-space terminé document_id=%s space_id=%s chunks=%s entités=%s relations=%s",
        document_id,
        space_id,
        processed_chunks,
        total_entities,
        total_relations,
    )
    return {
        "entities": total_entities,
        "relations": total_relations,
        "chunks": processed_chunks,
    }


def delete_entities_for_document(session: Session, document_id: int, space_id: int) -> int:
    """
    Supprime les relations/entités KAG liées à un document dans un espace.
    """
    chunk_ids_stmt = select(DocumentChunk.id).where(DocumentChunk.document_id == document_id)
    chunk_ids = list(session.exec(chunk_ids_stmt).all())
    if not chunk_ids:
        return 0

    entity_ids_stmt = select(ChunkEntityRelation.entity_id).where(
        ChunkEntityRelation.chunk_id.in_(chunk_ids),
        ChunkEntityRelation.space_id == space_id,
    )
    entity_ids = set(session.exec(entity_ids_stmt).all())

    deleted_relations = session.execute(
        delete(ChunkEntityRelation).where(
            ChunkEntityRelation.chunk_id.in_(chunk_ids),
            ChunkEntityRelation.space_id == space_id,
        )
    ).rowcount or 0

    if entity_ids:
        remaining_stmt = select(ChunkEntityRelation.entity_id).where(
            ChunkEntityRelation.entity_id.in_(entity_ids),
            ChunkEntityRelation.space_id == space_id,
        )
        still_used = set(session.exec(remaining_stmt).all())
        orphan_ids = [eid for eid in entity_ids if eid not in still_used]
        if orphan_ids:
            session.execute(
                delete(KnowledgeEntity).where(
                    KnowledgeEntity.id.in_(orphan_ids),
                    KnowledgeEntity.space_id == space_id,
                )
            )

    session.commit()
    logger.debug(
        "Suppression KAG document-space document_id=%s space_id=%s relations=%s",
        document_id,
        space_id,
        deleted_relations,
    )
    return deleted_relations


def get_space_kag_stats(session: Session, space_id: int) -> Dict:
    """
    Retourne les statistiques KAG pour un espace.
    """
    entity_count = session.exec(
        select(func.count(KnowledgeEntity.id)).where(
            KnowledgeEntity.space_id == space_id
        )
    ).one()

    relation_count = session.exec(
        select(func.count(ChunkEntityRelation.id)).where(
            ChunkEntityRelation.space_id == space_id
        )
    ).one()

    type_rows = session.exec(
        select(
            KnowledgeEntity.entity_type,
            func.count(KnowledgeEntity.id),
        )
        .where(KnowledgeEntity.space_id == space_id)
        .group_by(KnowledgeEntity.entity_type)
    ).all()
    type_counts = {entity_type: count for entity_type, count in type_rows}

    return {
        "total_entities": entity_count,
        "total_relations": relation_count,
        "entities_by_type": type_counts,
    }


def get_space_bipartite_graph(
    session: Session,
    space_id: int,
    max_entities: int = 80,
    max_relations: int = 600,
) -> Dict[str, List[Dict]]:
    """
    Construit un graphe biparti (entités <-> chunks) pour un espace.
    """
    entities_stmt = (
        select(KnowledgeEntity)
        .where(KnowledgeEntity.space_id == space_id)
        .order_by(KnowledgeEntity.mention_count.desc())
        .limit(max_entities)
    )
    entities = list(session.exec(entities_stmt).all())
    if not entities:
        return {"nodes": [], "edges": []}

    entity_ids = [e.id for e in entities]
    if not entity_ids:
        return {"nodes": [], "edges": []}

    rows = session.exec(
        select(
            ChunkEntityRelation,
            DocumentChunk,
            KnowledgeEntity,
            Document,
        )
        .join(DocumentChunk, DocumentChunk.id == ChunkEntityRelation.chunk_id)
        .join(KnowledgeEntity, KnowledgeEntity.id == ChunkEntityRelation.entity_id)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            ChunkEntityRelation.space_id == space_id,
            ChunkEntityRelation.entity_id.in_(entity_ids),
        )
        .order_by(ChunkEntityRelation.relevance_score.desc())
        .limit(max_relations)
    ).all()

    if not rows:
        return {"nodes": [], "edges": []}

    nodes: Dict[str, Dict] = {}
    edges: List[Dict] = []

    def add_entity_node(entity: KnowledgeEntity) -> str:
        node_id = f"entity-{entity.id}"
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "kind": "entity",
                "entity_id": entity.id,
                "label": entity.name,
                "entity_type": entity.entity_type,
                "mention_count": entity.mention_count,
            }
        return node_id

    def add_chunk_node(chunk: DocumentChunk, document: Document) -> str:
        node_id = f"chunk-{chunk.id}"
        if node_id not in nodes:
            title = document.title or f"Document {document.id}"
            preview = (chunk.content or "").strip()
            if preview and len(preview) > 320:
                cut = preview[:320]
                last_space = cut.rfind(" ")
                if last_space > 40:
                    cut = cut[:last_space]
                preview = cut + "…"
            nodes[node_id] = {
                "id": node_id,
                "kind": "chunk",
                "chunk_id": chunk.id,
                "document_id": document.id,
                "document_title": title,
                "label": title,
                "chunk_index": chunk.chunk_index,
                "preview": preview or None,
            }
        return node_id

    for rel, chunk, entity, document in rows:
        entity_node_id = add_entity_node(entity)
        chunk_node_id = add_chunk_node(chunk, document)
        edges.append(
            {
                "id": f"rel-{rel.id}",
                "from": entity_node_id,
                "to": chunk_node_id,
                "relevance": rel.relevance_score,
            }
        )

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
    }


def rebuild_kag_for_space(session: Session, space_id: int) -> Dict[str, int]:
    """
    Purge puis reconstruit le KAG pour tous les documents d'un espace.
    """
    doc_ids_stmt = select(DocumentSpace.document_id).where(DocumentSpace.space_id == space_id)
    document_ids = list(session.exec(doc_ids_stmt).all())

    deleted_relations = 0
    for document_id in document_ids:
        deleted_relations += delete_entities_for_document(session, document_id, space_id)

    rebuilt_documents = 0
    total_entities = 0
    total_relations = 0
    for document_id in document_ids:
        stats = process_kag_for_document_space(session, document_id, space_id)
        if stats.get("chunks", 0) > 0:
            rebuilt_documents += 1
        total_entities += stats.get("entities", 0)
        total_relations += stats.get("relations", 0)

    return {
        "documents_in_space": len(document_ids),
        "documents_rebuilt": rebuilt_documents,
        "relations_deleted": deleted_relations,
        "entities_extracted": total_entities,
        "relations_created": total_relations,
    }
