from sqlmodel import Session, create_engine, text
from app.database import DB_URL

engine = create_engine(DB_URL)

def add_source_columns():
    with Session(engine) as session:
        print("Ajout de la colonne 'source' à la table 'document'...")
        try:
            session.exec(text("ALTER TABLE document ADD COLUMN source VARCHAR"))
            session.commit()
            print("OK.")
        except Exception as e:
            print(f"La colonne existe déjà ou erreur : {e}")
            session.rollback()

        print("Ajout de la colonne 'source' à la table 'documentchunk'...")
        try:
            session.exec(text("ALTER TABLE documentchunk ADD COLUMN source VARCHAR"))
            session.commit()
            print("OK.")
        except Exception as e:
            print(f"La colonne existe déjà ou erreur : {e}")
            session.rollback()
            
        print("Création des index pour la colonne 'source'...")
        try:
            session.exec(text("CREATE INDEX ix_document_source ON document (source)"))
            session.exec(text("CREATE INDEX ix_documentchunk_source ON documentchunk (source)"))
            session.commit()
            print("OK.")
        except Exception as e:
            print(f"Les index existent déjà ou erreur : {e}")
            session.rollback()

if __name__ == "__main__":
    add_source_columns()
