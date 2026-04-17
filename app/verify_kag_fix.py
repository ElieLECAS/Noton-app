import os
import sys
from datetime import datetime

# Add app to path
sys.path.append("/app")

from sqlmodel import Session, create_engine, SQLModel, select
from app.models.knowledge_entity import KnowledgeEntity
from app.models.entity_alias import EntityAlias
from app.models.space import Space
from app.services.kag_graph_service import register_entity_alias

# Use in-memory SQLite for testing
sqlite_url = "sqlite://"
engine = create_engine(sqlite_url)

def test_duplicate_alias_session():
    SQLModel.metadata.create_all(engine)
    
    with Session(engine) as session:
        # 1. Setup space and entity
        space = Space(name="Test Space")
        session.add(space)
        session.flush()
        
        entity = KnowledgeEntity(
            name="Test Entity",
            name_normalized="test entity",
            entity_type="type",
            space_id=space.id,
            mention_count=1
        )
        session.add(entity)
        session.flush()
        
        print(f"Space ID: {space.id}, Entity ID: {entity.id}")
        
        # 2. Register alias first time
        register_entity_alias(session, space.id, entity.id, "alias1")
        
        # Check if it's in session.new
        new_aliases = [obj for obj in session.new if isinstance(obj, EntityAlias)]
        print(f"Aliases in session.new after first call: {len(new_aliases)}")
        assert len(new_aliases) == 1
        
        # 3. Register SAME alias second time (should NOT add another one)
        register_entity_alias(session, space.id, entity.id, "alias1")
        
        new_aliases = [obj for obj in session.new if isinstance(obj, EntityAlias)]
        print(f"Aliases in session.new after second call: {len(new_aliases)}")
        assert len(new_aliases) == 1
        
        # 4. Flush and check DB
        session.flush()
        db_aliases = session.exec(select(EntityAlias).where(EntityAlias.alias_normalized == "alias1")).all()
        print(f"Aliases in DB after flush: {len(db_aliases)}")
        assert len(db_aliases) == 1
        
        # 5. Register SAME alias third time (should NOT add another one)
        register_entity_alias(session, space.id, entity.id, "alias1")
        new_aliases = [obj for obj in session.new if isinstance(obj, EntityAlias)]
        print(f"Aliases in session.new after third call: {len(new_aliases)}")
        assert len(new_aliases) == 0 # None in new because it's already in DB
        
        print("\nTest passed successfully!")

if __name__ == "__main__":
    try:
        test_duplicate_alias_session()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
