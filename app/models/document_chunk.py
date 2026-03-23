from sqlmodel import SQLModel, Field, Relationship, Column, Text
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from pgvector.sqlalchemy import Vector
from app.embedding_config import EMBEDDING_DIMENSION
import json

if TYPE_CHECKING:
    from .note import Note


class DocumentChunk(SQLModel, table=True):
    """
    Chunk de document pour la recherche sémantique RAG unifiée.
    Gère tous les types de contenu : notes manuelles, PDF, Excel, Word, images, etc.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    note_id: int = Field(foreign_key="note.id", index=True)
    
    # Position et ordre du chunk
    chunk_index: int = Field(default=0, index=True)  # Position du chunk dans le document (0, 1, 2...)
    
    # Contenu textuel du chunk (extrait ou original)
    content: str = Field(sa_column=Column(Text))  # Le texte du chunk
    
    # Métadonnées enrichies provenant de Docling
    chunk_type: str = Field(default="text")  # text, table, image, list, title, etc.
    page_number: Optional[int] = None  # Numéro de page (pour PDF, Word, etc.)
    section_title: Optional[str] = None  # Titre de la section parente
    metadata_json: Optional[str] = Field(default=None, sa_column=Column(Text))  # JSON avec métadonnées Docling
    
    # Position dans le document original (pour les notes manuelles)
    start_char: int = Field(default=0)  # Position de début dans le texte original
    end_char: int = Field(default=0)  # Position de fin dans le texte original
    
    # Embedding vectoriel pour recherche sémantique
    embedding: Optional[List[float]] = Field(
        default=None, 
        sa_column=Column(Vector(EMBEDDING_DIMENSION), nullable=True, index=True)
    )
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relations
    note: Optional["Note"] = Relationship(back_populates="chunks")
    
    def get_metadata_dict(self) -> dict:
        """Récupérer les métadonnées sous forme de dictionnaire"""
        if self.metadata_json:
            try:
                return json.loads(self.metadata_json)
            except:
                return {}
        return {}
    
    def set_metadata_dict(self, metadata: dict):
        """Définir les métadonnées à partir d'un dictionnaire"""
        self.metadata_json = json.dumps(metadata, ensure_ascii=False)
    
    def to_markdown(self) -> str:
        """
        Convertir le chunk en format markdown pour le contexte RAG.
        Ajoute les métadonnées pertinentes pour aider le LLM.
        """
        parts = []
        
        # En-tête avec métadonnées
        meta_parts = []
        if self.page_number:
            meta_parts.append(f"Page {self.page_number}")
        if self.section_title:
            meta_parts.append(f"Section: {self.section_title}")
        if self.chunk_type and self.chunk_type != "text":
            meta_parts.append(f"Type: {self.chunk_type}")
        
        if meta_parts:
            parts.append(f"*[{', '.join(meta_parts)}]*")
        
        # Contenu
        parts.append(self.content)
        
        return "\n".join(parts)


class DocumentChunkRead(SQLModel):
    """Modèle de lecture pour un chunk"""
    id: int
    note_id: int
    chunk_index: int
    content: str
    chunk_type: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    start_char: int
    end_char: int
    created_at: datetime


class DocumentChunkWithScore(SQLModel):
    """Modèle pour un chunk avec son score de similarité"""
    chunk: DocumentChunkRead
    score: float
    note_title: str

