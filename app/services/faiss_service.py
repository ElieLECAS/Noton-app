from typing import List, Dict, Optional, Tuple
import faiss
import numpy as np
from sqlmodel import Session, select
from app.models.note_chunk import NoteChunk
import logging

logger = logging.getLogger(__name__)

# Dimension des embeddings (doit correspondre au modèle utilisé)
from app.embedding_config import EMBEDDING_DIMENSION


class FaissIndexManager:
    """Gestionnaire d'index FAISS pour la recherche sémantique des chunks"""
    
    def __init__(self):
        # Index FAISS (L2 distance pour similarité cosinus)
        self.index: Optional[faiss.Index] = None
        # Mapping chunk_id -> position dans l'index
        self.chunk_id_to_position: Dict[int, int] = {}
        # Mapping position -> chunk_id (inverse)
        self.position_to_chunk_id: Dict[int, int] = {}
        # Mapping chunk_id -> note_id pour récupérer la note complète
        self.chunk_id_to_note_id: Dict[int, int] = {}
        # Mapping chunk_id -> project_id pour filtrage
        self.chunk_id_to_project_id: Dict[int, int] = {}
        # Compteur de position actuelle
        self.current_position = 0
    
    def initialize(self, dimension: int = EMBEDDING_DIMENSION):
        """Initialiser l'index FAISS vide"""
        # IndexFlatL2 pour recherche par distance L2 (peut être converti en cosinus)
        # On utilise IndexFlatIP (Inner Product) pour similarité cosinus si les vecteurs sont normalisés
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product pour cosinus similarity
        self.chunk_id_to_position = {}
        self.position_to_chunk_id = {}
        self.chunk_id_to_note_id = {}
        self.chunk_id_to_project_id = {}
        self.current_position = 0
        logger.info(f"Index FAISS initialisé avec dimension {dimension}")
    
    def load_from_db(self, session: Session, project_id: Optional[int] = None):
        """
        Charger tous les chunks depuis la base de données dans FAISS.
        
        Args:
            session: Session SQLModel
            project_id: Optionnel, charger seulement les chunks d'un projet spécifique
        """
        try:
            # Construire la requête pour les chunks
            from app.models.note import Note
            statement = select(NoteChunk).join(Note).where(NoteChunk.embedding.isnot(None))
            if project_id:
                statement = statement.where(Note.project_id == project_id)
            
            chunks = list(session.exec(statement).all())
            
            if not chunks:
                logger.info("Aucun chunk avec embedding trouvé")
                return
            
            # Initialiser l'index si nécessaire
            if self.index is None:
                self.initialize()
            
            # Préparer les embeddings et les mappings
            embeddings = []
            chunk_ids = []
            note_ids = []
            project_ids = []
            
            for chunk in chunks:
                # Vérifier explicitement que l'embedding n'est pas None et a la bonne dimension
                if chunk.embedding is not None:
                    # Convertir en liste si c'est un array numpy
                    embedding_list = chunk.embedding.tolist() if hasattr(chunk.embedding, 'tolist') else chunk.embedding
                    if len(embedding_list) == EMBEDDING_DIMENSION:
                        # Normaliser le vecteur pour utiliser Inner Product comme cosinus similarity
                        embedding_array = np.array(embedding_list, dtype=np.float32)
                        norm = np.linalg.norm(embedding_array)
                        if norm > 0:
                            embedding_array = embedding_array / norm
                        
                        embeddings.append(embedding_array)
                        chunk_ids.append(chunk.id)
                        note_ids.append(chunk.note_id)
                        # Récupérer le project_id depuis la note
                        note = session.get(Note, chunk.note_id)
                        if note:
                            project_ids.append(note.project_id)
                        else:
                            project_ids.append(None)
            
            if not embeddings:
                logger.warning("Aucun embedding valide trouvé")
                return
            
            # Convertir en numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Ajouter à l'index
            self.index.add(embeddings_array)
            
            # Mettre à jour les mappings
            start_pos = self.current_position
            for i, chunk_id in enumerate(chunk_ids):
                pos = start_pos + i
                self.chunk_id_to_position[chunk_id] = pos
                self.position_to_chunk_id[pos] = chunk_id
                self.chunk_id_to_note_id[chunk_id] = note_ids[i]
                if project_ids[i]:
                    self.chunk_id_to_project_id[chunk_id] = project_ids[i]
            
            self.current_position += len(embeddings)
            logger.info(f"Chargé {len(embeddings)} chunks dans FAISS")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement depuis la DB: {e}")
            raise
    
    def add_chunk(self, chunk_id: int, embedding: List[float], project_id: int, note_id: int):
        """
        Ajouter un chunk à l'index FAISS.
        
        Args:
            chunk_id: ID du chunk
            embedding: Liste de floats représentant l'embedding
            project_id: ID du projet
            note_id: ID de la note parente
        """
        if self.index is None:
            self.initialize()
        
        if not embedding or len(embedding) != EMBEDDING_DIMENSION:
            logger.warning(f"Embedding invalide pour chunk {chunk_id}")
            return
        
        try:
            # Normaliser le vecteur
            embedding_array = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm
            else:
                logger.warning(f"Vecteur nul pour chunk {chunk_id}")
                return
            
            # Ajouter à l'index
            self.index.add(embedding_array.reshape(1, -1))
            
            # Mettre à jour les mappings
            pos = self.current_position
            self.chunk_id_to_position[chunk_id] = pos
            self.position_to_chunk_id[pos] = chunk_id
            self.chunk_id_to_note_id[chunk_id] = note_id
            self.chunk_id_to_project_id[chunk_id] = project_id
            self.current_position += 1
            
            logger.debug(f"Ajouté chunk {chunk_id} (note {note_id}) à la position {pos} dans FAISS")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du chunk {chunk_id}: {e}")
            raise
    
    def remove_chunk(self, chunk_id: int):
        """
        Supprimer un chunk de l'index FAISS.
        
        Args:
            chunk_id: ID du chunk à supprimer
        """
        if chunk_id in self.chunk_id_to_position:
            pos = self.chunk_id_to_position[chunk_id]
            del self.chunk_id_to_position[chunk_id]
            if pos in self.position_to_chunk_id:
                del self.position_to_chunk_id[pos]
            if chunk_id in self.chunk_id_to_note_id:
                del self.chunk_id_to_note_id[chunk_id]
            if chunk_id in self.chunk_id_to_project_id:
                del self.chunk_id_to_project_id[chunk_id]
            logger.debug(f"Marqué chunk {chunk_id} comme supprimé de FAISS")
    
    def remove_chunks_for_note(self, note_id: int):
        """
        Supprimer tous les chunks d'une note de l'index FAISS.
        
        Args:
            note_id: ID de la note
        """
        chunks_to_remove = [chunk_id for chunk_id, nid in self.chunk_id_to_note_id.items() if nid == note_id]
        for chunk_id in chunks_to_remove:
            self.remove_chunk(chunk_id)
        logger.debug(f"Supprimé {len(chunks_to_remove)} chunks pour la note {note_id}")
    
    def search(self, query_embedding: List[float], project_id: Optional[int] = None, k: int = 10) -> List[Tuple[int, float]]:
        """
        Recherche sémantique dans l'index FAISS au niveau des chunks.
        
        Args:
            query_embedding: Embedding de la requête
            project_id: Optionnel, filtrer par projet
            k: Nombre de chunks à retourner
            
        Returns:
            Liste de tuples (chunk_id, score) triés par score décroissant
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index FAISS vide ou non initialisé")
            return []
        
        if not query_embedding or len(query_embedding) != EMBEDDING_DIMENSION:
            logger.warning("Embedding de requête invalide")
            return []
        
        try:
            # Normaliser le vecteur de requête
            query_array = np.array(query_embedding, dtype=np.float32)
            norm = np.linalg.norm(query_array)
            if norm > 0:
                query_array = query_array / norm
            else:
                return []
            
            # Rechercher dans l'index
            query_array = query_array.reshape(1, -1)
            scores, indices = self.index.search(query_array, k * 2)  # Chercher plus pour filtrer
            
            # Filtrer par projet si nécessaire et mapper les positions aux chunk_ids
            results = []
            for score, pos in zip(scores[0], indices[0]):
                if pos == -1:  # FAISS retourne -1 pour les résultats vides
                    continue
                
                if pos not in self.position_to_chunk_id:
                    continue
                
                chunk_id = self.position_to_chunk_id[pos]
                
                # Filtrer par projet si nécessaire
                if project_id is not None:
                    if chunk_id not in self.chunk_id_to_project_id:
                        continue
                    if self.chunk_id_to_project_id[chunk_id] != project_id:
                        continue
                
                results.append((chunk_id, float(score)))
            
            # Trier par score décroissant et limiter à k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche FAISS: {e}")
            return []


# Instance globale du gestionnaire FAISS
_faiss_manager: Optional[FaissIndexManager] = None


def get_faiss_manager() -> FaissIndexManager:
    """Obtenir l'instance globale du gestionnaire FAISS (singleton)"""
    global _faiss_manager
    if _faiss_manager is None:
        _faiss_manager = FaissIndexManager()
    return _faiss_manager

