from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.database import get_session
from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.services.project_service import get_project_by_id
from app.services.kag_graph_service import (
    get_kag_stats,
    get_project_bipartite_graph,
    rebuild_kag_for_project,
)


router = APIRouter(prefix="/api/kag", tags=["kag"])


def _ensure_project_access(
    session: Session,
    project_id: int,
    current_user: UserRead,
):
    project = get_project_by_id(session, project_id, current_user.id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Projet non trouvé",
        )


@router.get("/projects/{project_id}/stats", response_model=Dict)
async def get_project_kag_stats(
    project_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Statistiques KAG pour un projet (nombre d'entités, de relations, etc.).
    """
    _ensure_project_access(session, project_id, current_user)
    return get_kag_stats(session, project_id)


@router.get("/projects/{project_id}/graph", response_model=Dict)
async def get_project_kag_graph(
    project_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Graphe biparti (entités <-> chunks) pour visualisation front.
    """
    _ensure_project_access(session, project_id, current_user)
    return get_project_bipartite_graph(
        session=session,
        project_id=project_id,
        user_id=current_user.id,
    )


def _process_parent_enrichment_for_document_space(
    session: Session, document_id: int, space_id: int
):
    """
    Enrichit les chunks parents (is_leaf=False) d'un document avec summary+questions+embedding.
    
    Génère via LLM puis embedde le texte combiné pour créer un signal sémantique
    représentant l'intention de la section (utilisable lors du retrieval).
    """
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


def resolve_entity_duplicates_for_space(session: Session, space_id: int) -> Dict[str, int]:
    """
    Résout les doublons d'entités dans un espace en utilisant l'embedding sémantique.
    
    Pour chaque entité sans embedding:
    - Génère son embedding
    - Cherche des entités similaires (cosine > 0.92)
    - Fusionne les entités (met à jour les relations, supprime les doublons)
    
    Args:
        session: Session SQLModel
        space_id: ID de l'espace
    
    Returns:
        Statistiques de fusion
    """
    try:
        from app.services.embedding_service import generate_embeddings_batch
        from sqlalchemy import text as sql_text
        
        # Étape 1: générer les embeddings manquants
        entities_without_embedding = list(
            session.exec(
                select(KnowledgeEntity).where(
                    KnowledgeEntity.space_id == space_id,
                    KnowledgeEntity.embedding.is_(None),
                )
            ).all()
        )
        
        if entities_without_embedding:
            logger.info(
                "Génération embeddings pour %d entités sans embedding (space_id=%s)",
                len(entities_without_embedding),
                space_id,
            )
            names = [e.name for e in entities_without_embedding]
            embeddings = generate_embeddings_batch(names, batch_size=settings.EMBEDDING_BATCH_SIZE)
            
            for entity, embedding in zip(entities_without_embedding, embeddings):
                if embedding:
                    entity.embedding = embedding
                    session.add(entity)
            
            session.commit()
        
        # Étape 2: résolution des doublons par similarité
        all_entities = list(
            session.exec(
                select(KnowledgeEntity)
                .where(
                    KnowledgeEntity.space_id == space_id,
                    KnowledgeEntity.embedding.isnot(None),
                )
                .order_by(KnowledgeEntity.mention_count.desc())
            ).all()
        )
        
        merged_count = 0
        entities_to_delete: Set[int] = set()
        
        for i, entity in enumerate(all_entities):
            if entity.id in entities_to_delete:
                continue
            
            embedding_str = "[" + ",".join(map(str, entity.embedding)) + "]"
            
            # Chercher entités similaires (même type, cosine > 0.92)
            similar_query = sql_text(f"""
                SELECT id, name, mention_count,
                       1 - (embedding <=> '{embedding_str}'::vector) AS similarity
                FROM knowledgeentity
                WHERE space_id = :space_id
                  AND entity_type = :entity_type
                  AND id != :entity_id
                  AND embedding IS NOT NULL
                  AND 1 - (embedding <=> '{embedding_str}'::vector) > 0.92
                ORDER BY similarity DESC
            """)
            
            similar_results = list(
                session.execute(
                    similar_query,
                    {
                        "space_id": space_id,
                        "entity_type": entity.entity_type,
                        "entity_id": entity.id,
                    },
                ).all()
            )
            
            if similar_results:
                # Fusionner toutes les entités similaires dans celle-ci
                for similar_row in similar_results:
                    duplicate_id = similar_row.id
                    if duplicate_id in entities_to_delete:
                        continue
                    
                    # Transférer toutes les relations vers l'entité principale
                    session.execute(
                        sql_text("""
                            UPDATE chunkentityrelation
                            SET entity_id = :target_id
                            WHERE entity_id = :source_id
                              AND space_id = :space_id
                              AND NOT EXISTS (
                                  SELECT 1 FROM chunkentityrelation
                                  WHERE entity_id = :target_id
                                    AND chunk_id = chunkentityrelation.chunk_id
                                    AND space_id = :space_id
                              )
                        """),
                        {
                            "target_id": entity.id,
                            "source_id": duplicate_id,
                            "space_id": space_id,
                        },
                    )
                    
                    # Supprimer les relations dupliquées
                    session.execute(
                        delete(ChunkEntityRelation).where(
                            ChunkEntityRelation.entity_id == duplicate_id,
                            ChunkEntityRelation.space_id == space_id,
                        )
                    )
                    
                    # Mettre à jour le mention_count
                    entity.mention_count += similar_row.mention_count
                    entity.updated_at = datetime.utcnow()
                    
                    entities_to_delete.add(duplicate_id)
                    merged_count += 1
                    
                    logger.debug(
                        "Fusion entité: '%s' (id=%s) → '%s' (id=%s, sim=%.3f)",
                        similar_row.name,
                        duplicate_id,
                        entity.name,
                        entity.id,
                        similar_row.similarity,
                    )
                
                session.add(entity)
        
        # Supprimer les entités fusionnées
        if entities_to_delete:
            session.execute(
                delete(KnowledgeEntity).where(
                    KnowledgeEntity.id.in_(list(entities_to_delete)),
                    KnowledgeEntity.space_id == space_id,
                )
            )
        
        session.commit()
        
        logger.info(
            "✅ Résolution doublons terminée (space_id=%s): %d entités fusionnées, %d supprimées",
            space_id,
            merged_count,
            len(entities_to_delete),
        )
        
        return {
            "entities_merged": merged_count,
            "entities_deleted": len(entities_to_delete),
            "embeddings_generated": len(entities_without_embedding),
        }
    
    except Exception as e:
        logger.error(
            "Erreur résolution doublons (space_id=%s): %s",
            space_id,
            e,
            exc_info=True,
        )
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la résolution des doublons: {str(e)}",
        )


@router.post("/spaces/{space_id}/resolve-entities", response_model=Dict)
async def resolve_space_entities(
    space_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Résout les doublons d'entités dans un espace en utilisant l'embedding sémantique.
    
    Fusionne les entités similaires (cosine > 0.92) pour réduire les doublons.
    Endpoint admin / maintenance.
    """
    from app.services.space_service import get_space_by_id
    
    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Espace non trouvé",
        )
    
    from app.services.kag_graph_service import resolve_entity_duplicates_for_space
    stats = resolve_entity_duplicates_for_space(session, space_id)
    return stats

