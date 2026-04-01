"""
Service de chunking intelligent de documents.
Découpe les documents en chunks optimisés pour la recherche sémantique RAG.
"""
from typing import List, Optional, Dict, Any
from app.models.document_chunk import DocumentChunk
from app.models.note import Note
from app.services.docling_service import ParsedDocument, get_docling_service
from app.services.embedding_service import generate_embeddings_batch
import logging

logger = logging.getLogger(__name__)

# Configuration du chunking
DEFAULT_CHUNK_SIZE = 500  # Caractères par chunk
DEFAULT_OVERLAP = 50  # Chevauchement entre chunks


class DocumentChunkingService:
    """Service de chunking intelligent pour tous types de documents"""
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.docling_service = get_docling_service()
    
    def chunk_note(self, note: Note) -> List[DocumentChunk]:
        """
        Créer des chunks pour une note (manuelle ou document).
        
        Args:
            note: La note à chunker
            
        Returns:
            Liste de DocumentChunk (sans embeddings, à générer séparément)
        """
        # Si c'est un document uploadé, utiliser Docling
        if note.source_file_path and note.source_file_type:
            return self._chunk_document(note)
        else:
            # Note manuelle : chunking simple
            return self._chunk_text_note(note)
    
    def _chunk_document(self, note: Note) -> List[DocumentChunk]:
        """
        Chunker un document uploadé via Docling.
        Utilise la structure du document pour créer des chunks intelligents.
        """
        try:
            # Parser le document avec Docling
            parsed_doc = self.docling_service.parse_document(
                note.source_file_path, 
                note.source_file_type
            )
            
            if not parsed_doc:
                logger.warning(f"Impossible de parser le document {note.source_file_path}")
                return self._chunk_text_fallback(note)
            
            chunks = []
            chunk_index = 0
            
            # Si Docling a extrait des éléments structurés, les utiliser
            if parsed_doc.elements:
                chunks = self._chunk_from_elements(note, parsed_doc.elements)
            else:
                # Sinon, utiliser le contenu textuel complet
                chunks = self._chunk_text_content(
                    note, 
                    parsed_doc.content,
                    chunk_index_start=0
                )
            
            logger.info(f"📦 Document {note.id} découpé en {len(chunks)} chunks intelligents")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chunking du document: {e}", exc_info=True)
            return self._chunk_text_fallback(note)
    
    def _chunk_from_elements(self, note: Note, elements: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Créer des chunks à partir des éléments structurés extraits par Docling.
        Préserve la structure (paragraphes, tableaux, listes, etc.)
        """
        chunks = []
        chunk_index = 0
        current_section = None
        
        for element in elements:
            element_type = element.get('type', 'text')
            content = element.get('content', '').strip()
            metadata = element.get('metadata', {})
            
            if not content:
                continue
            
            # Déterminer le titre de section si c'est un titre
            if element_type == 'title':
                current_section = content
                # Créer un chunk pour le titre aussi
                chunk = DocumentChunk(
                    note_id=note.id,
                    chunk_index=chunk_index,
                    content=content,
                    chunk_type='title',
                    page_number=metadata.get('page'),
                    section_title=current_section,
                    start_char=0,
                    end_char=len(content)
                )
                chunk.set_metadata_dict(metadata)
                chunks.append(chunk)
                chunk_index += 1
                continue
            
            # Pour les tableaux et listes courtes, créer un seul chunk
            if element_type in ['table', 'list'] and len(content) <= self.chunk_size * 1.5:
                chunk = DocumentChunk(
                    note_id=note.id,
                    chunk_index=chunk_index,
                    content=content,
                    chunk_type=element_type,
                    page_number=metadata.get('page'),
                    section_title=current_section,
                    start_char=0,
                    end_char=len(content)
                )
                chunk.set_metadata_dict(metadata)
                chunks.append(chunk)
                chunk_index += 1
            
            # Pour les longs paragraphes, découper en chunks avec overlap
            elif len(content) > self.chunk_size:
                text_chunks = self._split_text_with_overlap(content)
                for i, text_chunk in enumerate(text_chunks):
                    chunk = DocumentChunk(
                        note_id=note.id,
                        chunk_index=chunk_index,
                        content=text_chunk,
                        chunk_type=element_type,
                        page_number=metadata.get('page'),
                        section_title=current_section,
                        start_char=i * (self.chunk_size - self.overlap),
                        end_char=i * (self.chunk_size - self.overlap) + len(text_chunk)
                    )
                    chunk.set_metadata_dict(metadata)
                    chunks.append(chunk)
                    chunk_index += 1
            
            # Pour les paragraphes courts, créer un chunk simple
            else:
                chunk = DocumentChunk(
                    note_id=note.id,
                    chunk_index=chunk_index,
                    content=content,
                    chunk_type=element_type,
                    page_number=metadata.get('page'),
                    section_title=current_section,
                    start_char=0,
                    end_char=len(content)
                )
                chunk.set_metadata_dict(metadata)
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _chunk_text_note(self, note: Note) -> List[DocumentChunk]:
        """
        Chunker une note textuelle manuelle.
        """
        # Combiner titre et contenu
        full_text = note.title
        if note.content and note.content.strip():
            full_text += f"\n\n{note.content}"
        
        return self._chunk_text_content(note, full_text)
    
    def _chunk_text_content(
        self, 
        note: Note, 
        text: str, 
        chunk_index_start: int = 0
    ) -> List[DocumentChunk]:
        """
        Découper un texte en chunks avec overlap.
        """
        if not text or not text.strip():
            # Créer un chunk vide si pas de contenu
            return [DocumentChunk(
                note_id=note.id,
                chunk_index=0,
                content=note.title,
                chunk_type='text',
                start_char=0,
                end_char=len(note.title)
            )]
        
        chunks = []
        text_chunks = self._split_text_with_overlap(text)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk = DocumentChunk(
                note_id=note.id,
                chunk_index=chunk_index_start + i,
                content=chunk_text,
                chunk_type='text',
                start_char=i * (self.chunk_size - self.overlap),
                end_char=i * (self.chunk_size - self.overlap) + len(chunk_text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_text_with_overlap(self, text: str) -> List[str]:
        """
        Découper un texte en chunks avec overlap, en essayant de couper
        aux limites de phrases ou de mots.
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Si on est à la fin, prendre tout ce qui reste
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
            
            # Essayer de couper à une fin de phrase
            chunk_text = text[start:end]
            
            # Chercher le dernier délimiteur de phrase
            last_sentence_end = max(
                chunk_text.rfind('. '),
                chunk_text.rfind('! '),
                chunk_text.rfind('? '),
                chunk_text.rfind('\n\n')
            )
            
            # Si on trouve une fin de phrase dans les 100 derniers caractères
            if last_sentence_end > self.chunk_size - 100 and last_sentence_end > 0:
                end = start + last_sentence_end + 1
                chunks.append(text[start:end].strip())
            else:
                # Sinon, chercher le dernier espace
                last_space = chunk_text.rfind(' ')
                if last_space > self.chunk_size - 50 and last_space > 0:
                    end = start + last_space
                    chunks.append(text[start:end].strip())
                else:
                    # En dernier recours, couper brutalement
                    chunks.append(chunk_text.strip())
            
            # Avancer avec overlap
            start = end - self.overlap
            
            # Éviter les boucles infinies
            if start >= end:
                start = end
        
        # Filtrer les chunks vides
        return [c for c in chunks if c]
    
    def _chunk_text_fallback(self, note: Note) -> List[DocumentChunk]:
        """
        Fallback : utiliser le contenu de la note directement si le parsing échoue.
        """
        logger.warning(f"Utilisation du fallback textuel pour la note {note.id}")
        return self._chunk_text_note(note)
    
    def generate_embeddings_for_chunks(
        self, 
        chunks: List[DocumentChunk],
        batch_size: int = 8
    ) -> List[DocumentChunk]:
        """
        Générer les embeddings pour une liste de chunks.
        
        Args:
            chunks: Liste de chunks sans embeddings
            batch_size: Taille du batch pour la génération
            
        Returns:
            Liste de chunks avec embeddings
        """
        if not chunks:
            return []
        
        try:
            # Extraire les contenus
            contents = [chunk.content for chunk in chunks]
            
            # Générer les embeddings en batch
            logger.info(f"🔢 Génération de {len(contents)} embeddings en batch...")
            embeddings = generate_embeddings_batch(contents, batch_size=batch_size)
            
            # Assigner les embeddings aux chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            # Compter les succès
            success_count = sum(1 for c in chunks if c.embedding is not None)
            logger.info(f"✅ {success_count}/{len(chunks)} embeddings générés avec succès")
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération des embeddings: {e}", exc_info=True)
            return chunks


# Instance singleton
_chunking_service: Optional[DocumentChunkingService] = None


def get_chunking_service() -> DocumentChunkingService:
    """Récupérer l'instance singleton du service de chunking"""
    global _chunking_service
    if _chunking_service is None:
        _chunking_service = DocumentChunkingService()
    return _chunking_service

