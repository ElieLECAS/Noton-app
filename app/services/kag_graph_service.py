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
from app.services.kag_extraction_service import normalize_entity_name

logger = logging.getLogger(__name__)


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
    
    type_counts = {}
    for entity_type in ["equipement", "procedure", "parametre", "composant", "reference", "lieu"]:
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
