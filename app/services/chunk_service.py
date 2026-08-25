import logging
import re
import threading
from typing import List, Optional
from sqlmodel import Session
from sqlalchemy import delete

from app.models.document_chunk import DocumentChunk
from app.models.document import Document
from app.config import settings
from app.library_document_logging import get_library_document_logger, log_chunk_inventory

from app.services.chunking_service import (
    chunk_markdown_hierarchical_with_tables,
    chunk_markdown_structured,
    specs_to_document_chunks,
    CHUNKING_VERSION_MARKDOWN_H2,
    CHUNKING_VERSION_ADAPTIVE,
    CHUNKING_VERSION_MARKDOWN_STRUCTURED,
    resolve_adaptive_chunk_params,
    _detect_content_type,
    _build_page_marker_index,
    _page_no_from_char_offset,
)

logger = logging.getLogger(__name__)


def delete_chunks_for_document(
    session: Session, document_id: int, commit: bool = True
):
    """Supprime tous les chunks d'un document bibliothèque."""
    session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    if commit:
        session.commit()
    # Delete from LanceDB
    try:
        from app.services.lancedb_service import delete_chunks_lancedb
        delete_chunks_lancedb(document_id)
    except Exception as e:
        logger.error(f"Failed to delete from LanceDB for document_id={document_id}: {e}", exc_info=True)
    logger.debug("Supprimé chunks pour document_id=%s", document_id)


def _try_markdown_h2_sections(text: str) -> Optional[List[dict]]:
    """
    Découpe intermédiaire par sections Markdown (##) si le texte en contient plusieurs.
    Retourne None si non applicable (meilleur que la seule fenêtre glissante).
    """
    if not text or "\n## " not in text:
        return None
    parts = text.split("\n## ")
    if len(parts) < 2:
        return None
    chunks: List[dict] = []
    offset = 0
    for i, part in enumerate(parts):
        block = (f"## {part}" if i else part).strip()
        if not block:
            continue
        start = offset
        end = offset + len(block)
        chunks.append({"content": block, "start_char": start, "end_char": end})
        offset = end + 2
    return chunks if len(chunks) >= 2 else None


def _split_text_for_document(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[dict]:
    """Découpe simple d'un texte en chunks avec overlap."""
    if not text:
        return []
    chunks: List[dict] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "content": chunk_text,
                    "start_char": start,
                    "end_char": end,
                }
            )
        if end >= text_len:
            break
        start = max(0, end - overlap)
    return chunks


def _split_text_adaptive_for_document(text: str) -> List[dict]:
    """
    Découpe par paragraphe avec taille/overlap selon le type (procédure, normatif, description).
    """
    if not text or not text.strip():
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return _split_text_for_document(text.strip())
    chunks: List[dict] = []
    for para in paragraphs:
        ct = _detect_content_type(para)
        chunk_size, overlap = resolve_adaptive_chunk_params(ct)
        sub = _split_text_for_document(para, chunk_size=chunk_size, overlap=overlap)
        for item in sub:
            item = dict(item)
            item["content_type"] = ct
            chunks.append(item)
    return chunks if chunks else _split_text_for_document(text.strip())


def create_chunks_for_document(
    session: Session, document: Document
) -> List[DocumentChunk]:
    """Créer les chunks pour un document (nouvelle architecture)."""
    ld = get_library_document_logger()
    ld.info(
        "[Chunking] document_id=%s — stratégie FALLBACK (markdown H2 ou fenêtre adaptative). "
        "Raison typique : markdown vide ou échec du chunking hiérarchique. "
        "Conséquence : is_leaf=True partout, node_id/parent_node_id NULL, pas de parents RAG.",
        document.id,
    )
    session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
    session.commit()

    source_text = f"{document.title}\n\n{document.content or ''}".strip()
    page_index = _build_page_marker_index(source_text)
    raw_chunks = _try_markdown_h2_sections(source_text)
    chunking_version = CHUNKING_VERSION_MARKDOWN_H2
    if raw_chunks is None:
        raw_chunks = _split_text_adaptive_for_document(source_text)
        chunking_version = CHUNKING_VERSION_ADAPTIVE
    chunks: List[DocumentChunk] = []
    for idx, item in enumerate(raw_chunks):
        ct = item.get("content_type") or _detect_content_type(item.get("content", ""))
        metadata = {
            "document_id": document.id,
            "library_id": document.library_id,
            "user_id": document.user_id,
            "document_title": document.title or "",
            "chunk_index": idx,
            "chunking_version": chunking_version,
            "content_type": ct,
        }
        page_no = _page_no_from_char_offset(int(item.get("start_char", 0) or 0), page_index)
        if page_no is not None:
            metadata["page_no"] = page_no
        chunks.append(
            DocumentChunk(
                document_id=document.id,
                chunk_index=idx,
                source=document.source,
                content=item["content"],
                text=item["content"],
                start_char=item["start_char"],
                end_char=item["end_char"],
                is_leaf=True,
                hierarchy_level=0,
                metadata_json=metadata,
                metadata_=metadata,
            )
        )

    if chunks:
        session.add_all(chunks)
        session.commit()
    log_chunk_inventory(ld, document.id, chunks, "Chunking fallback terminé")
    return chunks


def create_chunks_for_document_from_markdown(
    session: Session,
    document: Document,
    markdown: str,
) -> List[DocumentChunk]:
    """Chunking depuis markdown Mistral OCR (hiérarchique ou structurel selon config)."""
    ld = get_library_document_logger()
    use_structured = settings.USE_MARKDOWN_STRUCTURED_CHUNKING
    
    if use_structured:
        ld.info(
            "[Chunking] document_id=%s — markdown structurel (MarkdownNodeParser) + expansion tableaux.",
            document.id,
        )
    else:
        ld.info(
            "[Chunking] document_id=%s — markdown hiérarchique (HierarchicalNodeParser) + expansion tableaux.",
            document.id,
        )
    
    session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
    session.commit()

    metadata_base = {
        "document_id": document.id,
        "library_id": document.library_id,
        "user_id": document.user_id,
        "document_title": document.title or "",
    }
    
    if use_structured:
        specs = chunk_markdown_structured(markdown, metadata_base)
    else:
        specs = chunk_markdown_hierarchical_with_tables(markdown, metadata_base)
    
    if not specs:
        return create_chunks_for_document(session=session, document=document)

    chunks = specs_to_document_chunks(document, specs)

    for c in chunks:
        c.source = document.source

    session.add_all(chunks)
    session.commit()

    chunking_method = "markdown structurel" if use_structured else "markdown hiérarchique"
    log_chunk_inventory(ld, document.id, chunks, f"Chunking {chunking_method} (persisté)")
    return chunks


_embedding_workers_lock = threading.Lock()
_embedding_workers_initialized = False


def _ensure_embedding_workers() -> None:
    """
    Compatibilité démarrage (app.main).

    Les documents bibliothèque sont indexés par le pipeline multimodal unifié
    (`document_indexing_service.process_document_indexing`, sans embeddings texte
    depuis le 2026-08-25). Il n'y a pas de file thread dédiée ici.
    """
    global _embedding_workers_initialized
    with _embedding_workers_lock:
        if not _embedding_workers_initialized:
            logger.debug(
                "Workers embeddings (threads) : non requis — pipeline document worker"
            )
            _embedding_workers_initialized = True
