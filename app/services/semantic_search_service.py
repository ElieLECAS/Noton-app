from typing import List, Optional, Dict, Tuple
from sqlmodel import Session, select
from app.models.note import Note
from app.models.note_chunk import NoteChunk
from app.services.embedding_service import generate_embedding
from app.services.faiss_service import get_faiss_manager
from app.services.project_service import get_project_by_id
from app.services.chunk_service import get_chunks_by_ids
import logging

logger = logging.getLogger(__name__)


def search_relevant_chunks(
    session: Session,
    project_id: int,
    query_text: str,
    user_id: int,
    k: int = 10
) -> List[Dict]:
    """
    Recherche sémantique RAG au niveau des chunks.
    Retourne les chunks pertinents avec leur contexte (titre de la note).
    
    Args:
        session: Session SQLModel
        project_id: ID du projet
        query_text: Texte de la requête
        user_id: ID de l'utilisateur (pour vérification de sécurité)
        k: Nombre de chunks à retourner
        
    Returns:
        Liste de dictionnaires contenant le chunk et son contexte (note)
    """
    # Vérifier que le projet appartient à l'utilisateur
    project = get_project_by_id(session, project_id, user_id)
    if not project:
        logger.warning(f"Projet {project_id} non trouvé ou n'appartient pas à l'utilisateur {user_id}")
        return []
    
    if not query_text or not query_text.strip():
        logger.warning("Requête vide fournie")
        return []
    
    try:
        # Générer l'embedding de la requête
        query_embedding = generate_embedding(query_text)
        if query_embedding is None:
            logger.warning("Impossible de générer l'embedding de la requête")
            return []
        
        # Obtenir le gestionnaire FAISS
        faiss_manager = get_faiss_manager()
        
        # Si l'index est vide, charger depuis la DB
        if faiss_manager.index is None or faiss_manager.index.ntotal == 0:
            logger.info("Chargement de l'index FAISS depuis la base de données")
            faiss_manager.load_from_db(session, project_id)
        
        # Rechercher dans FAISS au niveau des chunks
        search_results = faiss_manager.search(query_embedding, project_id=project_id, k=k)
        
        if not search_results:
            logger.info(f"Aucun résultat trouvé pour la requête dans le projet {project_id}")
            return []
        
        # Récupérer les chunks depuis la DB
        chunk_ids = [chunk_id for chunk_id, _ in search_results]
        chunks = get_chunks_by_ids(session, chunk_ids)
        
        # Récupérer les notes correspondantes pour le contexte
        note_ids = list(set(chunk.note_id for chunk in chunks))
        statement = select(Note).where(
            Note.id.in_(note_ids),
            Note.project_id == project_id,
            Note.user_id == user_id
        )
        notes = list(session.exec(statement).all())
        note_dict = {note.id: note for note in notes}
        
        # Créer un mapping chunk_id -> chunk
        chunk_dict = {chunk.id: chunk for chunk in chunks}
        
        # Construire les résultats avec contexte
        results = []
        for chunk_id, score in search_results:
            if chunk_id in chunk_dict:
                chunk = chunk_dict[chunk_id]
                note = note_dict.get(chunk.note_id)
                if note:
                    results.append({
                        'chunk': chunk,
                        'note': note,
                        'score': score
                    })
        
        logger.info(f"Trouvé {len(results)} chunks pertinents pour la requête")
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de la recherche sémantique: {e}")
        # Fallback: retourner les chunks des notes les plus récentes
        logger.info("Fallback sur recherche par date de modification")
        statement = select(Note).where(
            Note.project_id == project_id,
            Note.user_id == user_id
        ).order_by(Note.updated_at.desc()).limit(5)
        notes = list(session.exec(statement).all())
        
        results = []
        for note in notes:
            chunk_statement = select(NoteChunk).where(NoteChunk.note_id == note.id).limit(2)
            for chunk in session.exec(chunk_statement):
                results.append({
                    'chunk': chunk,
                    'note': note,
                    'score': 0.0
                })
        return results[:k]

