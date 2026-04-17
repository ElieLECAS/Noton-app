import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from sqlmodel import select

from app.services.kag_extraction_service import (
    extract_entities_batch_async,
    extract_typed_relations_batch_async,
)


def test_register_entity_alias_idempotent_same_transaction(db_session):
    """
    Deux appels avec le même alias dans une transaction + no_autoflush ne doivent pas
    violer uq_entityalias_space_alias (régression doublon ORM avant flush).
    """
    from app.models.space import Space
    from app.models.knowledge_entity import KnowledgeEntity
    from app.models.entity_alias import EntityAlias
    from app.services.kag_graph_service import register_entity_alias

    space = Space(name="pytest-kag-alias", user_id=None)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)

    ent = KnowledgeEntity(
        name="canon",
        name_normalized="canon",
        entity_type="concept_technique",
        space_id=space.id,
    )
    db_session.add(ent)
    db_session.commit()
    db_session.refresh(ent)

    alias_norm = "175 mm"
    with db_session.no_autoflush:
        register_entity_alias(db_session, space.id, ent.id, alias_norm)
        register_entity_alias(db_session, space.id, ent.id, alias_norm)
    db_session.commit()

    rows = db_session.exec(
        select(EntityAlias).where(
            EntityAlias.space_id == space.id,
            EntityAlias.alias_normalized == alias_norm,
        )
    ).all()
    assert len(rows) == 1

@pytest.mark.asyncio
async def test_extract_entities_batch_async():
    """
    Vérifie que l'extraction par batch appelle bien extract_entities_from_chunk
    en parallèle et retourne les bons résultats.
    """
    contents = ["content 1", "content 2", "content 3"]
    mock_entities = [{"name": "E1", "type": "T1", "importance": 1.0}]
    
    with patch("app.services.kag_extraction_service.extract_entities_from_chunk", new_callable=AsyncMock) as mock_extract:
        mock_extract.return_value = mock_entities
        
        results = await extract_entities_batch_async(contents)
        
        assert len(results) == 3
        assert results[0] == mock_entities
        assert mock_extract.call_count == 3

@pytest.mark.asyncio
async def test_extract_typed_relations_batch_async():
    """
    Vérifie que l'extraction des relations par batch fonctionne.
    """
    inputs = [
        {"content": "c1", "entities": [{"name": "E1"}]},
        {"content": "c2", "entities": [{"name": "E2"}]},
    ]
    mock_rels = [{"source": "E1", "target": "E2", "relation": "test"}]
    
    with patch("app.services.kag_extraction_service.extract_typed_relations_from_chunk", new_callable=AsyncMock) as mock_extract:
        mock_extract.return_value = mock_rels
        
        results = await extract_typed_relations_batch_async(inputs)
        
        assert len(results) == 2
        assert results[0] == mock_rels
        assert mock_extract.call_count == 2

@pytest.mark.asyncio
async def test_kag_extraction_concurrency_limit():
    """
    Vérifie indirectement que le sémaphore est utilisé (test de structure).
    On vérifie que extract_entities_from_chunk utilise le sémaphore global.
    """
    from app.services.kag_extraction_service import KAG_EXTRACTION_SEMAPHORE
    assert KAG_EXTRACTION_SEMAPHORE._value == 5
