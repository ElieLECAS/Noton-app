from typing import List, Dict, Tuple
from sqlmodel import Session, select, func
from app.models.note import Note
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


def search_relevant_passages(
    session: Session,
    project_id: int,
    query_text: str,
    user_id: int,
    k: int = 15,
    passage_size: int = 500
) -> List[Dict]:
    """
    Recherche sémantique avancée qui cherche dans TOUTES les notes du projet
    et retourne les PASSAGES les plus pertinents (pas les notes complètes).
    
    Cette approche permet de :
    - Analyser 100% du projet
    - Extraire seulement les informations pertinentes
    - Combiner des informations de différentes notes
    
    Args:
        session: Session SQLModel
        project_id: ID du projet
        query_text: Texte de la requête
        user_id: ID de l'utilisateur (pour vérification de sécurité)
        k: Nombre de passages à retourner
        passage_size: Taille approximative des passages en caractères
        
    Returns:
        Liste de dictionnaires contenant le passage, la note source et le score
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
        
        # Récupérer TOUTES les notes du projet avec leurs embeddings
        from sqlalchemy import text
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        connection = session.connection()
        
        # Récupérer toutes les notes du projet avec leur similarité
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
        """
        
        result = connection.execute(text(sql_query))
        all_notes = []
        
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
            all_notes.append({
                'note': note,
                'score': float(row.similarity_score)
            })
        
        logger.info(f"📚 Analyse de {len(all_notes)} notes du projet pour extraction de passages")
        
        # Extraire les passages pertinents de toutes les notes
        passages = []
        
        for note_data in all_notes:
            note = note_data['note']
            note_score = note_data['score']
            
            # Si la note n'a pas de contenu, utiliser seulement le titre
            if not note.content or not note.content.strip():
                passages.append({
                    'passage': f"**{note.title}**",
                    'note_title': note.title,
                    'note_id': note.id,
                    'score': note_score
                })
                continue
            
            # Découper le contenu en passages
            content = note.content
            sentences = content.split('\n')
            
            current_passage = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Ajouter la phrase au passage actuel
                if current_length + len(sentence) <= passage_size:
                    current_passage.append(sentence)
                    current_length += len(sentence)
                else:
                    # Sauvegarder le passage actuel
                    if current_passage:
                        passage_text = '\n'.join(current_passage)
                        passages.append({
                            'passage': f"**{note.title}**\n{passage_text}",
                            'note_title': note.title,
                            'note_id': note.id,
                            'score': note_score  # Utiliser le score de la note
                        })
                    
                    # Commencer un nouveau passage
                    current_passage = [sentence]
                    current_length = len(sentence)
            
            # Ajouter le dernier passage
            if current_passage:
                passage_text = '\n'.join(current_passage)
                passages.append({
                    'passage': f"**{note.title}**\n{passage_text}",
                    'note_title': note.title,
                    'note_id': note.id,
                    'score': note_score
                })
        
        # Trier les passages par score et prendre les k meilleurs
        passages.sort(key=lambda x: x['score'], reverse=True)
        top_passages = passages[:k]
        
        logger.info(f"✅ Extrait {len(top_passages)} passages pertinents sur {len(passages)} passages totaux")
        return top_passages
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche de passages: {e}", exc_info=True)
        return []

