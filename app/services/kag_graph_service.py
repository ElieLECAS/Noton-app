"""
Service de gestion du graphe de connaissances KAG.

Gère les opérations CRUD sur les entités et relations chunk-entité,
ainsi que le lookup pour enrichir le retrieval RAG.
"""

import logging
import math
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple
from sqlmodel import Session, select
from sqlalchemy import func, delete, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.models.knowledge_entity import KnowledgeEntity
from app.models.chunk_entity_relation import ChunkEntityRelation
from app.models.note_chunk import NoteChunk
from app.models.note import Note
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.models.entity_alias import EntityAlias
from app.models.entity_entity_relation import EntityEntityRelation
from app.config import settings
from app.services.kag_extraction_service import (
    normalize_entity_name,
    SUPPORTED_ENTITY_TYPE_IDS,
    extract_entities_sync,
    extract_typed_relations_sync,
    extract_entities_batch_async,
    extract_typed_relations_batch_async,
)
from app.services.document_service_new import (
    LIBRARY_USER_STOPPED_STATUSES,
    _should_abort_processing,
    _finalize_pipeline_abort,
)
import asyncio

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
    Utilise matching exact puis fallback ILIKE partiel si peu de résultats.
    
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
    
    # Matching exact sur name_normalized
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
        )
        .order_by(ChunkEntityRelation.relevance_score.desc())
        .limit(limit)
    )
    
    results = list(session.exec(statement).all())
    
    # Fallback ILIKE partiel si peu de résultats exacts
    if len(results) < limit // 2 and normalized_names:
        logger.debug(
            "KAG retrieval (projet): fallback ILIKE (résultats exacts=%d)", 
            len(results)
        )
        from sqlalchemy import or_
        
        # Limiter aux 5 premiers termes
        ilike_terms = normalized_names[:5]
        ilike_conditions = [
            KnowledgeEntity.name_normalized.ilike(f"%{term}%") 
            for term in ilike_terms
        ]
        
        stmt_ilike = (
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
                or_(*ilike_conditions),
                Note.user_id == user_id,
            )
            .order_by(ChunkEntityRelation.relevance_score.desc())
            .limit(limit - len(results))
        )
        
        ilike_results = list(session.exec(stmt_ilike).all())
        
        # Déduplication par chunk.id
        existing_chunk_ids = {chunk.id for chunk, _, _ in results}
        for chunk, entity_name, score in ilike_results:
            if chunk.id not in existing_chunk_ids:
                results.append((chunk, entity_name, score))
                existing_chunk_ids.add(chunk.id)
    
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

def register_entity_alias(
    session: Session,
    space_id: int,
    entity_id: int,
    alias_normalized: str,
) -> None:
    """Enregistre un alias normalisé pour une entité (expansion requête)."""
    if not alias_normalized or len(alias_normalized) < 2:
        return
    ent = session.get(KnowledgeEntity, entity_id)
    if ent and alias_normalized == ent.name_normalized:
        return
    stmt = (
        pg_insert(EntityAlias)
        .values(
            space_id=space_id,
            entity_id=entity_id,
            alias_normalized=alias_normalized,
            created_at=datetime.utcnow(),
        )
        .on_conflict_do_nothing(index_elements=["space_id", "alias_normalized"])
    )
    session.execute(stmt)


def expand_kag_query_terms_for_space(
    session: Session,
    space_id: int,
    terms: List[str],
) -> List[str]:
    """
    Étend les termes normalisés avec les alias et formes canoniques liés en base.
    """
    if not terms:
        return []
    lowered = [t.strip().lower() for t in terms if t and len(t.strip()) >= 2]
    if not lowered:
        return []
    out: Set[str] = set(lowered)
    alias_rows = session.exec(
        select(EntityAlias).where(
            EntityAlias.space_id == space_id,
            EntityAlias.alias_normalized.in_(lowered),
        )
    ).all()
    entity_ids: Set[int] = {a.entity_id for a in alias_rows}
    direct = session.exec(
        select(KnowledgeEntity).where(
            KnowledgeEntity.space_id == space_id,
            KnowledgeEntity.name_normalized.in_(lowered),
        )
    ).all()
    entity_ids |= {e.id for e in direct}
    if not entity_ids:
        return list(out)
    all_aliases = session.exec(
        select(EntityAlias).where(
            EntityAlias.space_id == space_id,
            EntityAlias.entity_id.in_(entity_ids),
        )
    ).all()
    for a in all_aliases:
        out.add(a.alias_normalized)
    ents = session.exec(
        select(KnowledgeEntity).where(KnowledgeEntity.id.in_(entity_ids))
    ).all()
    for e in ents:
        out.add(e.name_normalized)
    return list(out)


def refresh_entity_entity_relations_for_space(session: Session, space_id: int) -> int:
    """
    Reconstruit les arêtes co_occurs à partir des relations chunk-entité de l'espace.
    """
    from collections import defaultdict

    session.execute(
        delete(EntityEntityRelation).where(
            EntityEntityRelation.space_id == space_id,
            EntityEntityRelation.relation_type == "co_occurs",
        )
    )
    stmt = (
        select(ChunkEntityRelation.chunk_id, ChunkEntityRelation.entity_id)
        .join(DocumentChunk, DocumentChunk.id == ChunkEntityRelation.chunk_id)
        .join(DocumentSpace, DocumentSpace.document_id == DocumentChunk.document_id)
        .where(
            DocumentSpace.space_id == space_id,
            ChunkEntityRelation.space_id == space_id,
        )
    )
    chunk_to_entities: Dict[int, List[int]] = defaultdict(list)
    for chunk_id, eid in session.exec(stmt).all():
        chunk_to_entities[int(chunk_id)].append(int(eid))
    pair_weights: Dict[Tuple[int, int], float] = {}
    for eids in chunk_to_entities.values():
        uniq = sorted(set(eids))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = uniq[i], uniq[j]
                if a > b:
                    a, b = b, a
                pair_weights[(a, b)] = pair_weights.get((a, b), 0.0) + 1.0
    for (a, b), w in pair_weights.items():
        session.add(
            EntityEntityRelation(
                space_id=space_id,
                entity_a_id=a,
                entity_b_id=b,
                relation_type="co_occurs",
                weight=float(w),
                created_at=datetime.utcnow(),
            )
        )
    session.commit()
    return len(pair_weights)


def _neighbor_entity_ids_for_entities(
    session: Session,
    space_id: int,
    entity_ids: Set[int],
    limit: int = 40,
) -> Set[int]:
    if not entity_ids:
        return set()
    eid_list = list(entity_ids)[:80]
    n1 = session.exec(
        select(EntityEntityRelation.entity_b_id).where(
            EntityEntityRelation.space_id == space_id,
            EntityEntityRelation.entity_a_id.in_(eid_list),
        )
    ).all()
    n2 = session.exec(
        select(EntityEntityRelation.entity_a_id).where(
            EntityEntityRelation.space_id == space_id,
            EntityEntityRelation.entity_b_id.in_(eid_list),
        )
    ).all()
    out = {int(x) for x in n1 + n2 if x is not None}
    out -= entity_ids
    return set(list(out)[:limit])


def _get_or_create_entity_for_space(
    session: Session,
    name: str,
    entity_type: str,
    space_id: int,
) -> KnowledgeEntity:
    """
    Version espace de get_or_create_entity (déduplication par space_id + name_normalized).
    Résolution avancée par embedding si disponible.
    """
    name_normalized = normalize_entity_name(name)
    
    # Déduplication par name_normalized exact
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

    # Résolution par embedding sémantique (cosine > 0.92)
    try:
        from app.services.embedding_service import generate_embeddings_batch
        
        entity_embeddings = generate_embeddings_batch([name], batch_size=1)
        if entity_embeddings and entity_embeddings[0]:
            new_entity_embedding = entity_embeddings[0]
            embedding_str = "[" + ",".join(map(str, new_entity_embedding)) + "]"
            
            # Chercher entités existantes similaires (même type, même espace)
            similar_query = text(f"""
                SELECT id, name, name_normalized, mention_count,
                       1 - (embedding <=> '{embedding_str}'::vector) AS similarity
                FROM knowledgeentity
                WHERE space_id = :space_id
                  AND entity_type = :entity_type
                  AND embedding IS NOT NULL
                  AND 1 - (embedding <=> '{embedding_str}'::vector) > 0.92
                ORDER BY similarity DESC
                LIMIT 1
            """)
            
            result = session.execute(
                similar_query,
                {"space_id": space_id, "entity_type": entity_type},
            ).first()
            
            if result:
                # Entité similaire trouvée → fusionner
                existing_entity = session.get(KnowledgeEntity, result.id)
                if existing_entity:
                    existing_entity.mention_count += 1
                    existing_entity.updated_at = datetime.utcnow()
                    session.add(existing_entity)
                    logger.debug(
                        "Entité fusionnée (embedding): '%s' → '%s' (sim=%.3f)",
                        name,
                        existing_entity.name,
                        result.similarity,
                    )
                    try:
                        register_entity_alias(
                            session,
                            space_id,
                            existing_entity.id,
                            name_normalized,
                        )
                    except Exception:
                        pass
                    return existing_entity
            
            # Créer nouvelle entité avec embedding
            entity = KnowledgeEntity(
                name=name,
                name_normalized=name_normalized,
                entity_type=entity_type,
                space_id=space_id,
                mention_count=1,
                embedding=new_entity_embedding,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(entity)
            session.flush()
            logger.debug("Nouvelle entité créée avec embedding: '%s'", name)
            return entity
    
    except Exception as emb_err:
        logger.warning(
            "Erreur résolution embedding entité '%s': %s", 
            name, 
            emb_err,
        )
    
    # Fallback: création sans embedding
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


def _resolve_entity_id_for_space(
    session: Session,
    space_id: int,
    name: str,
) -> Optional[int]:
    """Résout un nom d'entité vers l'ID KnowledgeEntity (canonique ou alias)."""
    nn = normalize_entity_name(name)
    if not nn:
        return None
    ent = session.exec(
        select(KnowledgeEntity).where(
            KnowledgeEntity.space_id == space_id,
            KnowledgeEntity.name_normalized == nn,
        )
    ).first()
    if ent and ent.id is not None:
        return int(ent.id)
    alias = session.exec(
        select(EntityAlias).where(
            EntityAlias.space_id == space_id,
            EntityAlias.alias_normalized == nn,
        )
    ).first()
    if alias:
        return int(alias.entity_id)
    return None


def _update_entity_confidence_score(
    session: Session,
    entity_id: int,
    space_id: int,
) -> None:
    """
    Met à jour confidence_score = min(1, avg_importance * (1 + 0.15 * log1p(mention_count))).
    """
    rows = list(
        session.exec(
            select(ChunkEntityRelation.relevance_score).where(
                ChunkEntityRelation.entity_id == entity_id,
                ChunkEntityRelation.space_id == space_id,
            )
        ).all()
    )
    if not rows:
        return
    scores = [float(r) for r in rows]
    avg = sum(scores) / len(scores)
    entity = session.get(KnowledgeEntity, entity_id)
    if not entity:
        return
    mc = max(1, int(entity.mention_count or 1))
    conf = min(1.0, avg * (1.0 + 0.15 * math.log1p(float(mc))))
    entity.confidence_score = conf
    entity.updated_at = datetime.utcnow()
    session.add(entity)


def save_typed_relations_for_chunk(
    session: Session,
    space_id: int,
    chunk_id: int,
    relations: List[Dict],
    min_confidence: float = 0.35,
) -> int:
    """
    Persiste les relations entité-entité typées extraites par LLM pour un chunk feuille.

    Les arêtes ``co_occurs`` globales restent gérées par ``refresh_entity_entity_relations_for_space`` ;
    on n'enregistre pas ici les lignes ``relation_type == co_occurs`` pour éviter les doublons.
    """
    if not relations:
        return 0
    saved = 0
    for rel in relations:
        rt = str(rel.get("relation_type", "")).strip().lower()
        if not rt or rt == "co_occurs":
            continue
        conf = float(rel.get("confidence", 0) or 0)
        if conf < min_confidence:
            continue
        ea = _resolve_entity_id_for_space(session, space_id, str(rel.get("entity_a", "")))
        eb = _resolve_entity_id_for_space(session, space_id, str(rel.get("entity_b", "")))
        if ea is None or eb is None or ea == eb:
            continue
        existing = session.exec(
            select(EntityEntityRelation).where(
                EntityEntityRelation.space_id == space_id,
                EntityEntityRelation.entity_a_id == ea,
                EntityEntityRelation.entity_b_id == eb,
                EntityEntityRelation.relation_type == rt,
            )
        ).first()
        if existing:
            prev = float(existing.confidence or 0.0)
            if conf > prev:
                existing.confidence = conf
                existing.weight = float(conf)
                existing.source_chunk_id = chunk_id
                session.add(existing)
                saved += 1
        else:
            session.add(
                EntityEntityRelation(
                    space_id=space_id,
                    entity_a_id=ea,
                    entity_b_id=eb,
                    relation_type=rt,
                    weight=float(conf),
                    confidence=conf,
                    source_chunk_id=chunk_id,
                    created_at=datetime.utcnow(),
                )
            )
            saved += 1
    return saved


def process_kag_for_document_space(session: Session, document_id: int, space_id: int) -> Dict[str, int]:
    """
    Extrait et sauvegarde les entités KAG pour un document dans un espace donné.
    VERSION OPTIMISÉE : Extraction LLM parallélisée.
    """
    document = session.get(Document, document_id)
    if not document:
        logger.warning("Document introuvable pour KAG: document_id=%s", document_id)
        return {"entities": 0, "relations": 0, "chunks": 0, "typed_entity_relations": 0}

    association = session.exec(
        select(DocumentSpace).where(
            DocumentSpace.document_id == document_id,
            DocumentSpace.space_id == space_id,
        )
    ).first()
    if not association:
        return {"entities": 0, "relations": 0, "chunks": 0, "typed_entity_relations": 0}

    # Status guard before starting
    if document.processing_status in LIBRARY_USER_STOPPED_STATUSES:
        logger.info("[KAG] document_id=%s space_id=%s — abandon : déjà arrêté.", document_id, space_id)
        return {"entities": 0, "relations": 0, "chunks": 0, "typed_entity_relations": 0}

    delete_entities_for_document(session, document_id, space_id)
    session.flush()

    chunks_stmt = select(DocumentChunk).where(
        DocumentChunk.document_id == document_id,
        DocumentChunk.is_leaf == True,
    )
    chunks = list(session.exec(chunks_stmt).all())
    if not chunks:
        return {"entities": 0, "relations": 0, "chunks": 0, "typed_entity_relations": 0}

    # --- ÉTAPE 1 : Extraction des entités en parallèle ---
    valid_chunks = [c for c in chunks if (c.content or "").strip()]
    contents = [c.content for c in valid_chunks]

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, extract_entities_batch_async(contents))
                all_entities = future.result(timeout=300)
        else:
            all_entities = loop.run_until_complete(extract_entities_batch_async(contents))
    except Exception as e:
        logger.error("Erreur extraction entités batch: %s", e)
        all_entities = [[] for _ in valid_chunks]

    total_entities = 0
    total_relations = 0
    total_typed_relations = 0
    processed_chunks = 0
    
    # Stockage pour l'étape 2 (relations)
    chunks_for_relations = []

    # --- ÉTAPE 2 : Sauvegarde des entités et préparation des relations ---
    with session.no_autoflush:
        for chunk, entities in zip(valid_chunks, all_entities):
            if _should_abort_processing(document_id):
                logger.info("[KAG] document_id=%s — interrompu pendant sauvegarde entités.", document_id)
                session.rollback()
                return {"entities": total_entities, "relations": total_relations, "chunks": processed_chunks, "typed_entity_relations": 0}
            
            if not entities:
                continue
                
            processed_chunks += 1
            entity_ids_touched: Set[int] = set()
            
            # Déduplication locale des entités pour ce chunk unique (évite UniqueViolation)
            unique_entities_for_chunk = {}
            for ed in entities:
                 # Utilisation du nom normalisé comme clé de déduplication pour ce chunk
                 norm_name = normalize_entity_name(ed.get("name", ""))
                 etype = (ed.get("type") or "concept_technique").strip()
                 key = (norm_name, etype)
                 if key not in unique_entities_for_chunk or ed.get("importance", 0) > unique_entities_for_chunk[key].get("importance", 0):
                     unique_entities_for_chunk[key] = ed

            # On persiste les entités dédupliquées
            relation_entity_ids = set() # Pour éviter les doublons ID réels apres lookup DB
            for entity_data in unique_entities_for_chunk.values():
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
                
                # Ultra-safe check : si cet ID d'entité est déjà lié à ce chunk, on skip
                eid = int(entity.id)
                if eid in relation_entity_ids:
                    continue
                
                relation_entity_ids.add(eid)
                entity_ids_touched.add(eid)

                relation = ChunkEntityRelation(
                    chunk_id=chunk.id,
                    entity_id=entity.id,
                    relevance_score=importance,
                    space_id=space_id,
                    created_at=datetime.utcnow(),
                )
                session.add(relation)
                total_relations += 1

            for eid in entity_ids_touched:
                _update_entity_confidence_score(session, eid, space_id)
                
            if settings.KAG_TYPED_RELATIONS_ENABLED and len(entities) >= 2:
                chunks_for_relations.append({"chunk_id": chunk.id, "content": chunk.content, "entities": entities})

    session.flush()

    # --- ÉTAPE 3 : Extraction des relations typées en parallèle ---
    if chunks_for_relations:
        if _should_abort_processing(document_id):
            logger.info(
                "[KAG] document_id=%s — interrompu avant relations typées.",
                document_id,
            )
            # Persister entités / relations chunk déjà flushées dans cette transaction.
            session.commit()
            return {
                "entities": total_entities,
                "relations": total_relations,
                "chunks": processed_chunks,
                "typed_entity_relations": 0,
            }

        try:
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, extract_typed_relations_batch_async(chunks_for_relations))
                    all_typed = future.result(timeout=300)
            else:
                all_typed = loop.run_until_complete(extract_typed_relations_batch_async(chunks_for_relations))
                
            for fr, typed_rels in zip(chunks_for_relations, all_typed):
                if typed_rels:
                    n = save_typed_relations_for_chunk(
                        session, space_id, int(fr["chunk_id"]), typed_rels
                    )
                    total_typed_relations += n
        except Exception as e:
            logger.error("Erreur extraction relations batch: %s", e)

    session.commit()

    try:
        refresh_entity_entity_relations_for_space(session, space_id)
    except Exception as rel_err:
        logger.warning("Rafraîchissement relations co_occurs échoué: %s", rel_err)
    
    logger.info(
        "✅ KAG parallélisé terminé : %d chunks, %d entités, %d relations, %d typed",
        processed_chunks, total_entities, total_relations, total_typed_relations
    )
    return {
        "entities": total_entities,
        "relations": total_relations,
        "chunks": processed_chunks,
        "typed_entity_relations": total_typed_relations,
    }


def delete_entities_for_document(session: Session, document_id: int, space_id: int) -> int:
    """
    Supprime les relations/entités KAG liées à un document dans un espace.
    """
    chunk_ids_stmt = select(DocumentChunk.id).where(DocumentChunk.document_id == document_id)
    chunk_ids = list(session.exec(chunk_ids_stmt).all())
    if not chunk_ids:
        return 0

    session.execute(
        delete(EntityEntityRelation).where(
            EntityEntityRelation.source_chunk_id.in_(chunk_ids)
        )
    )

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

    entity_entity_edges = session.exec(
        select(func.count(EntityEntityRelation.id)).where(
            EntityEntityRelation.space_id == space_id
        )
    ).one()

    return {
        "total_entities": entity_count,
        "total_relations": relation_count,
        "entities_by_type": type_counts,
        "entity_entity_edges": entity_entity_edges,
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
        "graph_mode": "bipartite",
    }


def get_space_entity_relation_graph(
    session: Session,
    space_id: int,
    max_edges: int = 600,
) -> Dict[str, List[Dict]]:
    """
    Graphe entité–entité (co-occurrences et autres relation_type) pour la visualisation.
    """
    rows = list(
        session.exec(
            select(EntityEntityRelation)
            .where(EntityEntityRelation.space_id == space_id)
            .order_by(EntityEntityRelation.weight.desc())
            .limit(max_edges)
        ).all()
    )
    if not rows:
        return {"nodes": [], "edges": [], "graph_mode": "entity_links"}

    entity_ids: Set[int] = set()
    for r in rows:
        entity_ids.add(int(r.entity_a_id))
        entity_ids.add(int(r.entity_b_id))

    entities = list(
        session.exec(
            select(KnowledgeEntity).where(
                KnowledgeEntity.space_id == space_id,
                KnowledgeEntity.id.in_(entity_ids),
            )
        ).all()
    )
    ent_by_id = {int(e.id): e for e in entities if e.id is not None}

    nodes: List[Dict] = []
    for eid in sorted(entity_ids):
        ent = ent_by_id.get(eid)
        if not ent:
            continue
        nodes.append(
            {
                "id": f"entity-{eid}",
                "kind": "entity",
                "entity_id": eid,
                "label": ent.name,
                "entity_type": ent.entity_type,
                "mention_count": ent.mention_count,
            }
        )

    edges: List[Dict] = []
    for r in rows:
        if int(r.entity_a_id) not in ent_by_id or int(r.entity_b_id) not in ent_by_id:
            continue
        w = float(r.weight or 1.0)
        edges.append(
            {
                "id": f"eer-{r.id}",
                "from": f"entity-{int(r.entity_a_id)}",
                "to": f"entity-{int(r.entity_b_id)}",
                "relevance": min(1.0, w / (w + 5.0)),
                "relation_type": r.relation_type or "co_occurs",
                "weight": w,
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "graph_mode": "entity_links",
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


def _process_parent_enrichment_for_document_space(
    session: Session, document_id: int, space_id: int
):
    """
    DÉSACTIVÉ : L'enrichissement des parents est désactivé pour optimiser la performance KAG.
    """
    return
    try:
        from app.services.kag_extraction_service import (
            generate_parent_summary_questions_sync,
            extract_entities_sync,
        )
        from app.services.embedding_service import generate_embeddings_batch

        statement = select(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.is_leaf == False,
        )
        parent_chunks = list(session.exec(statement).all())

        if not parent_chunks:
            logger.debug(
                "Aucun chunk parent pour enrichissement document_id=%s",
                document_id,
            )
            return

        logger.info(
            "Démarrage enrichissement parents document_id=%s: %d parents",
            document_id,
            len(parent_chunks),
        )

        total_parents_enriched = 0
        total_parent_entities = 0
        total_parent_relations = 0

        for chunk in parent_chunks:
            if not chunk.content or len(chunk.content.strip()) < 30:
                continue

            try:
                result = generate_parent_summary_questions_sync(chunk.content)
                if not result:
                    continue

                metadata = dict(chunk.metadata_json or {})
                metadata["summary"] = result["summary"]
                metadata["generated_questions"] = result["generated_questions"]
                chunk.metadata_json = metadata
                chunk.metadata_ = metadata

                # Embedder le summary+questions et stocker dans l'embedding du parent
                enrichment_text = result["summary"]
                if result["generated_questions"]:
                    enrichment_text += " " + " ".join(result["generated_questions"])

                # Générer l'embedding pour ce parent (intention de section)
                parent_embeddings = generate_embeddings_batch([enrichment_text], batch_size=1)
                if parent_embeddings and parent_embeddings[0]:
                    chunk.embedding = parent_embeddings[0]
                    logger.debug(
                        "Embedding parent généré pour chunk_id=%s (summary+questions)",
                        chunk.id,
                    )

                session.add(chunk)
                total_parents_enriched += 1

                # Extraction d'entités sur le summary+questions
                entities = extract_entities_sync(enrichment_text)
                if entities:
                    parent_entity_ids: Set[int] = set()
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
                        total_parent_entities += 1
                        if entity.id is not None:
                            parent_entity_ids.add(int(entity.id))

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
                            total_parent_relations += 1
                    for peid in parent_entity_ids:
                        _update_entity_confidence_score(session, peid, space_id)

                    if settings.KAG_TYPED_RELATIONS_ENABLED and len(entities) >= 2:
                        try:
                            typed = extract_typed_relations_sync(enrichment_text, entities)
                            if typed:
                                save_typed_relations_for_chunk(
                                    session, space_id, int(chunk.id), typed
                                )
                        except Exception as tre:
                            logger.warning(
                                "Relations typées KAG (parent) chunk_id=%s: %s",
                                chunk.id,
                                tre,
                            )

            except Exception as e:
                logger.warning(
                    "Erreur enrichissement parent chunk_id=%s: %s",
                    chunk.id,
                    e,
                )
                continue

        session.commit()
        logger.info(
            "✅ Enrichissement parents terminé document_id=%s: %d/%d parents enrichis, %d entités, %d relations",
            document_id,
            total_parents_enriched,
            len(parent_chunks),
            total_parent_entities,
            total_parent_relations,
        )

    except Exception as e:
        logger.error(
            "Erreur enrichissement parents document_id=%s: %s",
            document_id,
            e,
            exc_info=True,
        )
