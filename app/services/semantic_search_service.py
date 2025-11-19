from typing import List, Dict, Tuple
from sqlmodel import Session, select, func
from app.models.note import Note
from app.models.note_chunk import NoteChunk
from app.services.embedding_service import generate_embedding
from app.services.project_service import get_project_by_id
import logging
import numpy as np

logger = logging.getLogger(__name__)


def search_relevant_notes(
    session: Session,
    project_id: int,
    query_text: str,
    user_id: int,
    k: int = 10
) -> List[Dict]:
    """
    Recherche sémantique RAG sur les chunks de notes.
    Retourne les notes les plus pertinentes par rapport à la requête, basées sur leurs chunks.
    
    Args:
        session: Session SQLModel
        project_id: ID du projet
        query_text: Texte de la requête
        user_id: ID de l'utilisateur (pour vérification de sécurité)
        k: Nombre de notes à retourner
        
    Returns:
        Liste de dictionnaires contenant la note et son score de similarité
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
        
        # Recherche directe dans la table NoteChunk avec pgvector
        from sqlalchemy import text
        
        # Convertir l'embedding en format PostgreSQL array
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        connection = session.connection()
        
        # Rechercher les chunks les plus pertinents, puis grouper par note
        sql_query = f"""
            SELECT DISTINCT ON (nc.note_id)
                n.id, 
                n.title, 
                n.content, 
                n.note_type, 
                n.project_id, 
                n.user_id, 
                n.created_at, 
                n.updated_at,
                1 - (nc.embedding <=> '{embedding_str}'::vector) as similarity_score
            FROM notechunk nc
            INNER JOIN note n ON nc.note_id = n.id
            WHERE n.project_id = {project_id}
                AND n.user_id = {user_id}
                AND nc.embedding IS NOT NULL
            ORDER BY nc.note_id, nc.embedding <=> '{embedding_str}'::vector
            LIMIT {k}
        """
        
        # Exécuter la requête directement
        result = connection.execute(text(sql_query))
        
        # Convertir les résultats en liste de dictionnaires
        results = []
        seen_note_ids = set()
        for row in result:
            # Éviter les doublons (au cas où DISTINCT ON ne fonctionne pas comme prévu)
            if row.id in seen_note_ids:
                continue
            seen_note_ids.add(row.id)
            
            note = Note(
                id=row.id,
                title=row.title,
                content=row.content,
                note_type=row.note_type,
                project_id=row.project_id,
                user_id=row.user_id,
                created_at=row.created_at,
                updated_at=row.updated_at
            )
            results.append({
                'note': note,
                'score': float(row.similarity_score)
            })
        
        # Trier par score décroissant
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Formater les scores pour le log
        score_strs = [f"{r['score']:.3f}" for r in results[:3]]
        logger.info(f"✅ Trouvé {len(results)} notes pertinentes via recherche sémantique sur chunks (scores: {score_strs}...)")
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche sémantique: {e}", exc_info=True)
        return []


def search_relevant_passages(
    session: Session,
    project_id: int,
    query_text: str,
    user_id: int,
    k: int = 15,
    passage_size: int = 500
) -> List[Dict]:
    """
    Recherche sémantique avancée qui cherche directement dans les CHUNKS des notes
    et retourne les chunks les plus pertinents (passages).
    
    Cette approche utilise le nouveau système RAG basé sur les chunks :
    - Recherche directement dans les chunks avec embeddings
    - Plus précis car chaque chunk a son propre embedding
    - Respecte la structure markdown (paragraphes, tableaux)
    - Plus efficace que l'ancien système
    
    Args:
        session: Session SQLModel
        project_id: ID du projet
        query_text: Texte de la requête
        user_id: ID de l'utilisateur (pour vérification de sécurité)
        k: Nombre de passages (chunks) à retourner
        passage_size: Ignoré (les chunks ont déjà une taille optimale)
    
    Returns:
        Liste de dictionnaires contenant le passage (chunk), la note source et le score
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
        
        # Recherche directe dans les chunks avec pgvector
        from sqlalchemy import text
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        connection = session.connection()
        
        # Rechercher les k chunks les plus pertinents directement
        sql_query = f"""
            SELECT 
                nc.id as chunk_id,
                nc.content as chunk_content,
                nc.chunk_index,
                nc.start_char,
                nc.end_char,
                n.id as note_id,
                n.title as note_title,
                n.note_type,
                n.project_id,
                n.user_id,
                1 - (nc.embedding <=> '{embedding_str}'::vector) as similarity_score
            FROM notechunk nc
            INNER JOIN note n ON nc.note_id = n.id
            WHERE n.project_id = {project_id}
                AND n.user_id = {user_id}
                AND nc.embedding IS NOT NULL
            ORDER BY nc.embedding <=> '{embedding_str}'::vector
            LIMIT {k}
        """
        
        result = connection.execute(text(sql_query))
        passages = []
        
        for row in result:
            # Construire le passage avec le titre de la note et le contenu du chunk
            passage_text = f"**{row.note_title}**\n{row.chunk_content}"
            
            passages.append({
                'passage': passage_text,
                'note_title': row.note_title,
                'note_id': row.note_id,
                'chunk_id': row.chunk_id,
                'chunk_index': row.chunk_index,
                'score': float(row.similarity_score)
            })
        
        # Formater les scores pour le log
        score_strs = [f"{p['score']:.3f}" for p in passages[:3]]
        logger.info(f"✅ Trouvé {len(passages)} chunks pertinents via recherche sémantique sur chunks (scores: {score_strs}...)")
        return passages
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche de passages: {e}", exc_info=True)
        return []

