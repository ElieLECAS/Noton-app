from typing import List, Dict, Tuple
from sqlmodel import Session, select, func
from app.models.note import Note
from app.models.document_chunk import DocumentChunk
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
    Recherche sémantique RAG directement sur les notes.
    Retourne les notes les plus pertinentes par rapport à la requête.
    
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
        
        # Recherche directe dans la base de données avec pgvector
        # On utilise l'opérateur <=> pour la distance cosinus (1 - similarité cosinus)
        # Plus le score est petit, plus c'est similaire (0 = identique)
        from sqlalchemy import text
        
        # Convertir l'embedding en format PostgreSQL array
        # Format: '[0.1,0.2,0.3,...]'
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Utiliser la connexion directement pour exécuter du SQL brut avec pgvector
        # On insère directement l'embedding dans la requête car c'est une valeur contrôlée
        # (pas d'entrée utilisateur directe, donc pas de risque d'injection SQL)
        connection = session.connection()
        
        # Construire la requête SQL avec pgvector
        # On utilise <=> pour distance cosinus (0 = identique, 2 = opposé)
        # Le score retourné sera : 1 - distance (donc 1 = identique, -1 = opposé)
        # Note: On insère directement les valeurs dans la requête car ce sont des valeurs contrôlées
        # (project_id, user_id sont des entiers, embedding est généré par notre code)
        sql_query = f"""
            SELECT 
                id, 
                title, 
                content, 
                note_type, 
                project_id, 
                user_id, 
                created_at, 
                updated_at,
                1 - (embedding <=> '{embedding_str}'::vector) as similarity_score
            FROM note
            WHERE project_id = {project_id}
                AND user_id = {user_id}
                AND embedding IS NOT NULL
            ORDER BY embedding <=> '{embedding_str}'::vector
            LIMIT {k}
        """
        
        # Exécuter la requête directement
        result = connection.execute(text(sql_query))
        
        # Convertir les résultats en liste de dictionnaires
        results = []
        for row in result:
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
        
        # Formater les scores pour le log
        score_strs = [f"{r['score']:.3f}" for r in results[:3]]
        logger.info(f"✅ Trouvé {len(results)} notes pertinentes via recherche sémantique (scores: {score_strs}...)")
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche sémantique: {e}", exc_info=True)
        return []


def search_relevant_chunks(
    session: Session,
    project_id: int,
    query_text: str,
    user_id: int,
    k: int = 15
) -> List[Dict]:
    """
    Recherche sémantique dans les DocumentChunk du projet.
    Utilise le nouveau système de chunking intelligent basé sur Docling.
    
    Cette fonction remplace search_relevant_passages() en utilisant
    directement les chunks précalculés avec leurs embeddings.
    
    Args:
        session: Session SQLModel
        project_id: ID du projet
        query_text: Texte de la requête
        user_id: ID de l'utilisateur (pour vérification de sécurité)
        k: Nombre de chunks à retourner
        
    Returns:
        Liste de dictionnaires contenant le chunk, la note source et le score
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
        
        from sqlalchemy import text
        
        # Convertir l'embedding en format PostgreSQL array
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Rechercher dans les DocumentChunk via join avec Note pour filtrer par projet
        connection = session.connection()
        
        sql_query = f"""
            SELECT 
                dc.id,
                dc.note_id,
                dc.chunk_index,
                dc.content,
                dc.chunk_type,
                dc.page_number,
                dc.section_title,
                dc.start_char,
                dc.end_char,
                dc.created_at,
                n.title as note_title,
                n.note_type as note_type,
                n.source_file_type as source_file_type,
                1 - (dc.embedding <=> '{embedding_str}'::vector) as similarity_score
            FROM document_chunk dc
            INNER JOIN note n ON dc.note_id = n.id
            WHERE n.project_id = {project_id}
                AND n.user_id = {user_id}
                AND dc.embedding IS NOT NULL
            ORDER BY dc.embedding <=> '{embedding_str}'::vector
            LIMIT {k}
        """
        
        result = connection.execute(text(sql_query))
        
        # Convertir les résultats
        chunks_results = []
        for row in result:
            chunk_dict = {
                'chunk_id': row.id,
                'note_id': row.note_id,
                'note_title': row.note_title,
                'note_type': row.note_type,
                'source_file_type': row.source_file_type,
                'content': row.content,
                'chunk_type': row.chunk_type,
                'page_number': row.page_number,
                'section_title': row.section_title,
                'score': float(row.similarity_score)
            }
            chunks_results.append(chunk_dict)
        
        top_scores = [f"{c['score']:.3f}" for c in chunks_results[:3]]
        logger.info(f"✅ Trouvé {len(chunks_results)} chunks pertinents (scores: {top_scores})")
        return chunks_results
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche dans les chunks: {e}", exc_info=True)
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
    DEPRECATED: Utiliser search_relevant_chunks() à la place.
    
    Recherche sémantique avancée qui cherche dans TOUTES les notes du projet
    et retourne les PASSAGES les plus pertinents (pas les notes complètes).
    
    Cette fonction est conservée pour compatibilité mais redirige vers
    search_relevant_chunks() qui utilise le nouveau système.
    
    Args:
        session: Session SQLModel
        project_id: ID du projet
        query_text: Texte de la requête
        user_id: ID de l'utilisateur (pour vérification de sécurité)
        k: Nombre de passages à retourner
        passage_size: Taille approximative des passages en caractères (ignoré)
        
    Returns:
        Liste de dictionnaires contenant le passage, la note source et le score
    """
    # Rediriger vers la nouvelle fonction
    chunks = search_relevant_chunks(session, project_id, query_text, user_id, k)
    
    # Convertir au format attendu par l'ancien système
    passages = []
    for chunk in chunks:
        # Formater le contenu en markdown si nécessaire
        passage_content = chunk['content']
        
        # Ajouter les métadonnées au passage
        meta_parts = []
        if chunk.get('page_number'):
            meta_parts.append(f"Page {chunk['page_number']}")
        if chunk.get('section_title'):
            meta_parts.append(f"Section: {chunk['section_title']}")
        if chunk.get('chunk_type') and chunk['chunk_type'] != 'text':
            meta_parts.append(f"Type: {chunk['chunk_type']}")
        
        if meta_parts:
            passage_content = f"*[{', '.join(meta_parts)}]*\n\n{passage_content}"
        
        passages.append({
            'passage': f"**{chunk['note_title']}**\n{passage_content}",
            'note_title': chunk['note_title'],
            'note_id': chunk['note_id'],
            'score': chunk['score']
        })
    
    return passages

