from typing import List
from app.models.note import Note
from app.models.note_chunk import NoteChunk
import logging

logger = logging.getLogger(__name__)

# Configuration du chunking
CHUNK_SIZE = 1000  # Nombre de caractères par chunk (augmenté pour meilleure qualité)
CHUNK_OVERLAP = 100  # Nombre de caractères de chevauchement entre chunks


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
    full_text = f"{note.title}\n\n{note.content}" if note.content else note.title
    
    # Utiliser le chunking adaptatif qui respecte les paragraphes
    text_chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Créer les objets NoteChunk
    current_pos = 0
    for chunk_index, chunk_content in enumerate(text_chunks):
        chunk_length = len(chunk_content)
        start_char = current_pos
        end_char = current_pos + chunk_length
        
        chunk = NoteChunk(
            note_id=note.id,
            chunk_index=chunk_index,
            content=chunk_content.strip(),
            start_char=start_char,
            end_char=end_char
        )
        chunks.append(chunk)
        
        # Position pour le prochain chunk (avec overlap)
        current_pos = end_char - CHUNK_OVERLAP if chunk_index < len(text_chunks) - 1 else end_char
    
    logger.debug(f"Note {note.id} découpée en {len(chunks)} chunks")
    return chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Découper un texte en chunks adaptatifs qui respectent les paragraphes markdown.
    Ne coupe jamais au milieu d'un paragraphe (délimité par \n\n).
    
    Args:
        text: Le texte à découper (peut être du markdown)
        chunk_size: Taille cible des chunks en caractères
        overlap: Chevauchement entre chunks
        
    Returns:
        Liste de strings (chunks)
    """
    if not text:
        return []
    
    # Séparer le texte en paragraphes (délimités par \n\n ou \n\n\n)
    # On utilise une regex pour gérer plusieurs sauts de ligne consécutifs
    import re
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Filtrer les paragraphes vides
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if not paragraphs:
        return []
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para)
        
        # Si le paragraphe seul dépasse la taille du chunk, on doit le découper
        # mais on essaie de respecter les phrases
        if para_length > chunk_size:
            # Si on a déjà du contenu dans le chunk actuel, le sauvegarder
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Découper le grand paragraphe en respectant les phrases
            para_chunks = _chunk_large_paragraph(para, chunk_size, overlap)
            chunks.extend(para_chunks)
            continue
        
        # Si ajouter ce paragraphe dépasse la taille du chunk
        if current_length + para_length + 2 > chunk_size:  # +2 pour "\n\n"
            # Sauvegarder le chunk actuel s'il n'est pas vide
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            
            # Commencer un nouveau chunk avec ce paragraphe
            # Si overlap > 0, on garde les derniers paragraphes du chunk précédent
            if overlap > 0 and chunks:
                # Prendre les derniers caractères du dernier chunk pour l'overlap
                last_chunk = chunks[-1]
                overlap_text = last_chunk[-overlap:] if len(last_chunk) > overlap else last_chunk
                # Chercher le dernier paragraphe complet dans l'overlap
                overlap_para = overlap_text.split('\n\n')[-1] if '\n\n' in overlap_text else overlap_text
                if overlap_para.strip():
                    current_chunk = [overlap_para.strip()]
                    current_length = len(overlap_para)
                else:
                    current_chunk = [para]
                    current_length = para_length
            else:
                current_chunk = [para]
                current_length = para_length
        else:
            # Ajouter le paragraphe au chunk actuel
            current_chunk.append(para)
            current_length += para_length + 2  # +2 pour "\n\n"
    
    # Ajouter le dernier chunk s'il n'est pas vide
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def _chunk_large_paragraph(paragraph: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Découpe un paragraphe très long en respectant les phrases.
    
    Args:
        paragraph: Le paragraphe à découper
        chunk_size: Taille cible des chunks
        overlap: Chevauchement entre chunks
        
    Returns:
        Liste de chunks
    """
    chunks = []
    start = 0
    
    while start < len(paragraph):
        end = start + chunk_size
        
        if end >= len(paragraph):
            chunks.append(paragraph[start:])
            break
        
        # Chercher une fin de phrase dans les 200 derniers caractères
        chunk_text = paragraph[start:end]
        
        # Chercher la dernière fin de phrase (., !, ?, ou \n)
        last_sentence_end = max(
            chunk_text.rfind('.'),
            chunk_text.rfind('!'),
            chunk_text.rfind('?'),
            chunk_text.rfind('\n')
        )
        
        # Si on trouve une fin de phrase dans une zone raisonnable, couper là
        if last_sentence_end > chunk_size - 200:
            chunks.append(paragraph[start:start + last_sentence_end + 1].strip())
            start = start + last_sentence_end + 1 - overlap
        else:
            # Sinon, chercher le dernier espace
            last_space = chunk_text.rfind(' ')
            if last_space > chunk_size - 100:
                chunks.append(paragraph[start:start + last_space].strip())
                start = start + last_space - overlap
            else:
                # Dernier recours : couper au caractère
                chunks.append(chunk_text.strip())
                start = end - overlap
        
        if start >= end:
            break
    
    return chunks

