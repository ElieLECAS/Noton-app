import pytest
from sqlmodel import Session, select
from app.models.knowledge_entity import KnowledgeEntity
from app.models.entity_alias import EntityAlias
from app.models.space import Space
from app.services.kag_graph_service import register_entity_alias

def test_register_entity_alias_deduplication_in_session(db_session: Session):
    """
    Vérifie que register_entity_alias ne crée pas de doublons dans la session
    avant même que les objets ne soient flushés en base.
    """
    # 1. Setup
    space = Space(name="Test KAG Fix Space")
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)
    
    entity = KnowledgeEntity(
        name="Main Entity",
        name_normalized="main entity",
        entity_type="type",
        space_id=space.id,
        mention_count=1
    )
    db_session.add(entity)
    db_session.commit()
    db_session.refresh(entity)
    
    # 2. Premier enregistrement d'alias
    register_entity_alias(db_session, space.id, entity.id, "my alias")
    
    # Vérifier qu'il est dans session.new
    new_aliases = [obj for obj in db_session.new if isinstance(obj, EntityAlias)]
    assert len(new_aliases) == 1
    assert new_aliases[0].alias_normalized == "my alias"
    
    # 3. Second enregistrement du MÊME alias (doit être ignoré car déjà dans session.new)
    register_entity_alias(db_session, space.id, entity.id, "my alias")
    
    new_aliases = [obj for obj in db_session.new if isinstance(obj, EntityAlias)]
    assert len(new_aliases) == 1, "Il ne devrait y avoir qu'un seul objet EntityAlias dans la session"
    
    # 4. Flush (ne doit pas lever de UniqueViolation)
    db_session.flush()
    
    # 5. Vérifier en base
    count = db_session.exec(
        select(EntityAlias).where(
            EntityAlias.space_id == space.id,
            EntityAlias.alias_normalized == "my alias"
        )
    ).all()
    assert len(count) == 1
    
    # 6. Troisième enregistrement après flush (doit être ignoré car déjà en DB)
    register_entity_alias(db_session, space.id, entity.id, "my alias")
    new_aliases = [obj for obj in db_session.new if isinstance(obj, EntityAlias)]
    assert len(new_aliases) == 0
