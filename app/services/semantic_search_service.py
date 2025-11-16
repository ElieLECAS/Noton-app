from typing import List, Dict
from sqlmodel import Session, select, func
from app.models.note import Note
from app.services.embedding_service import generate_embedding
from app.services.project_service import get_project_by_id
import logging

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

