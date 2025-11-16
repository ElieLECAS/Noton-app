from typing import List
from app.models.note import Note
from app.models.note_chunk import NoteChunk
import logging

logger = logging.getLogger(__name__)

# Configuration du chunking
CHUNK_SIZE = 500  # Nombre de caractères par chunk
CHUNK_OVERLAP = 50  # Nombre de caractères de chevauchement entre chunks


def chunk_note(note: Note) -> List[NoteChunk]:
    """
    Découper une note en chunks pour la recherche sémantique RAG.
    
    Args:
        note: La note à découper
        
    Returns:
        Liste de NoteChunk
    """
    chunks = []
    
    # Si la note n'a pas de contenu, créer un chunk avec juste le titre
    if not note.content or not note.content.strip():
        chunk = NoteChunk(
            note_id=note.id,
            chunk_index=0,
            content=note.title,
            start_char=0,
            end_char=len(note.title) if note.title else 0
        )
        chunks.append(chunk)
        return chunks
    
    # Combiner titre et contenu pour le chunking
    full_text = f"{note.title}\n\n{note.content}"
    
    # Découper le texte en chunks avec overlap
    start = 0
    chunk_index = 0
    
    while start < len(full_text):
        # Calculer la fin du chunk
        end = start + CHUNK_SIZE
        
        # Si on est à la fin du texte, prendre tout ce qui reste
        if end >= len(full_text):
            chunk_text = full_text[start:]
            end = len(full_text)
        else:
            # Essayer de couper à la fin d'une phrase ou d'un mot
            chunk_text = full_text[start:end]
            
            # Chercher le dernier point, point d'exclamation ou point d'interrogation
            last_sentence_end = max(
                chunk_text.rfind('.'),
                chunk_text.rfind('!'),
                chunk_text.rfind('?'),
                chunk_text.rfind('\n')
            )
            
            # Si on trouve une fin de phrase dans les 100 derniers caractères, couper là
            if last_sentence_end > CHUNK_SIZE - 100:
                chunk_text = chunk_text[:last_sentence_end + 1]
                end = start + last_sentence_end + 1
            else:
                # Sinon, chercher le dernier espace
                last_space = chunk_text.rfind(' ')
                if last_space > CHUNK_SIZE - 50:
                    chunk_text = chunk_text[:last_space]
                    end = start + last_space
        
        # Créer le chunk
        chunk = NoteChunk(
            note_id=note.id,
            chunk_index=chunk_index,
            content=chunk_text.strip(),
            start_char=start,
            end_char=end
        )
        chunks.append(chunk)
        
        # Passer au chunk suivant avec overlap
        start = end - CHUNK_OVERLAP
        chunk_index += 1
        
        # Éviter les boucles infinies
        if start >= end:
            break
    
    logger.debug(f"Note {note.id} découpée en {len(chunks)} chunks")
    return chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Découper un texte en chunks (fonction utilitaire).
    
    Args:
        text: Le texte à découper
        chunk_size: Taille des chunks
        overlap: Chevauchement entre chunks
        
    Returns:
        Liste de strings (chunks)
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Chercher une fin de phrase ou un espace
        chunk_text = text[start:end]
        last_sentence_end = max(
            chunk_text.rfind('.'),
            chunk_text.rfind('!'),
            chunk_text.rfind('?'),
            chunk_text.rfind('\n')
        )
        
        if last_sentence_end > chunk_size - 100:
            chunks.append(text[start:start + last_sentence_end + 1])
            start = start + last_sentence_end + 1 - overlap
        else:
            last_space = chunk_text.rfind(' ')
            if last_space > chunk_size - 50:
                chunks.append(text[start:start + last_space])
                start = start + last_space - overlap
            else:
                chunks.append(chunk_text)
                start = end - overlap
        
        if start >= end:
            break
    
    return chunks

