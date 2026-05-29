import logging
import re
import threading
import time
from typing import List, Optional
from datetime import datetime
from sqlmodel import Session, select
from sqlalchemy import or_, delete

from app.models.document_chunk import DocumentChunk
from app.models.document import Document
from app.database import engine
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
    session: Session, document: Document, generate_embeddings: bool = False
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

    if generate_embeddings and chunks:
        from app.services.embedding_service import generate_embeddings_batch
        embeddings = generate_embeddings_batch(
            [c.content for c in chunks], batch_size=settings.EMBEDDING_BATCH_SIZE
        )
        model_name = settings.EMBEDDING_MODEL
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:
                chunk.embedding = embedding
                meta = dict(chunk.metadata_json or {})
                meta["embedding_model"] = model_name
                chunk.metadata_json = meta
                chunk.metadata_ = meta

    if chunks:
        session.add_all(chunks)
        session.commit()
        # Sync to LanceDB
        try:
            from app.services.lancedb_service import insert_or_update_chunks_lancedb
            lancedb_data = [
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "vector": chunk.embedding
                }
                for chunk in chunks if chunk.embedding
            ]
            insert_or_update_chunks_lancedb(lancedb_data)
        except Exception as e:
            logger.error(f"Failed to sync to LanceDB for document_id={document.id}: {e}", exc_info=True)
    log_chunk_inventory(ld, document.id, chunks, "Chunking fallback terminé")
    return chunks


def create_chunks_for_document_from_markdown(
    session: Session,
    document: Document,
    markdown: str,
    generate_embeddings: bool = False,
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
        return create_chunks_for_document(
            session=session, document=document, generate_embeddings=generate_embeddings
        )

    chunks = specs_to_document_chunks(document, specs)

    if generate_embeddings and chunks:
        from app.services.embedding_service import generate_embeddings_batch

        leafs = [c for c in chunks if c.is_leaf]
        if leafs:
            embeddings = generate_embeddings_batch(
                [c.content for c in leafs], batch_size=settings.EMBEDDING_BATCH_SIZE
            )
            model_name = settings.EMBEDDING_MODEL
            for chunk, embedding in zip(leafs, embeddings):
                if embedding:
                    chunk.embedding = embedding
                    meta = dict(chunk.metadata_json or {})
                    meta["embedding_model"] = model_name
                    chunk.metadata_json = meta
                    chunk.metadata_ = meta

    for c in chunks:
        c.source = document.source

    session.add_all(chunks)
    session.commit()
    # Sync to LanceDB
    try:
        from app.services.lancedb_service import insert_or_update_chunks_lancedb
        lancedb_data = [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "vector": chunk.embedding
            }
            for chunk in chunks if chunk.embedding
        ]
        insert_or_update_chunks_lancedb(lancedb_data)
    except Exception as e:
        logger.error(f"Failed to sync to LanceDB for document_id={document.id}: {e}", exc_info=True)
    
    chunking_method = "markdown structurel" if use_structured else "markdown hiérarchique"
    log_chunk_inventory(ld, document.id, chunks, f"Chunking {chunking_method} (persisté)")
    return chunks


def _process_embeddings_for_document(
    document_id: int, run_id: Optional[str] = None
):
    """Génère les embeddings des seuls chunks feuilles (parents exclus), puis marque le document completed."""
    from app.services.document_service_new import (
        LIBRARY_USER_STOPPED_STATUSES,
        _finalize_pipeline_abort,
        _should_abort_processing,
    )
    from app.services.document_run import is_processing_run_current

    ld = get_library_document_logger()
    if run_id is not None and not is_processing_run_current(document_id, run_id):
        ld.info(
            "[Embeddings] document_id=%s — abandon : run_id obsolète.",
            document_id,
        )
        return
    ld.info(
        "[Embeddings] document_id=%s — début : vectorisation des feuilles uniquement "
        "(is_leaf=True ; les parents ne reçoivent pas d'embedding).",
        document_id,
    )
    t_embed = time.perf_counter()
    try:
        with Session(engine) as session:
            document = session.get(Document, document_id)
            if not document:
                ld.warning(
                    "[Embeddings] document_id=%s — document introuvable, arrêt.",
                    document_id,
                )
                return

            if _should_abort_processing(document_id):
                _finalize_pipeline_abort(document_id)
                return

            # Feuilles uniquement (parents hiérarchiques sans vecteur)
            statement = select(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.embedding.is_(None),
                or_(DocumentChunk.is_leaf == True, DocumentChunk.is_leaf.is_(None)),
            )
            chunks = list(session.exec(statement).all())
            ld.info(
                "[Embeddings] document_id=%s — %d chunk(s) feuille sans embedding à traiter.",
                document_id,
                len(chunks),
            )
            if not chunks:
                document.processing_status = "completed"
                document.processing_progress = 100
                document.updated_at = datetime.utcnow()
                session.add(document)
                session.commit()
                ld.info(
                    "[Embeddings] document_id=%s — rien à embedder (déjà fait ou aucune feuille), statut completed.",
                    document_id,
                )
                return

            document.processing_progress = max(document.processing_progress or 0, 90)
            document.updated_at = datetime.utcnow()
            session.add(document)
            session.commit()

            logger.info(
                "document_id=%s génération embeddings pour %s chunks feuilles",
                document_id,
                len(chunks),
            )
            from app.services.embedding_service import generate_embeddings_batch

            model_name = settings.EMBEDDING_MODEL
            ok = 0
            batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
            for i in range(0, len(chunks), batch_size):
                if _should_abort_processing(document_id):
                    ld.warning(
                        "[Embeddings] document_id=%s — interrompu pendant embeddings (lot %s/%s).",
                        document_id,
                        i // batch_size + 1,
                        (len(chunks) + batch_size - 1) // batch_size,
                    )
                    _finalize_pipeline_abort(document_id)
                    return
                batch = chunks[i : i + batch_size]
                embeddings = generate_embeddings_batch(
                    [c.content for c in batch], batch_size=len(batch)
                )
                for chunk, embedding in zip(batch, embeddings):
                    if embedding:
                        chunk.embedding = embedding
                        meta = dict(chunk.metadata_json or {})
                        meta["embedding_model"] = model_name
                        chunk.metadata_json = meta
                        chunk.metadata_ = meta
                        session.add(chunk)
                        ok += 1
                session.commit()
                # Sync batch to LanceDB
                try:
                    from app.services.lancedb_service import insert_or_update_chunks_lancedb
                    lancedb_data = [
                        {
                            "id": chunk.id,
                            "document_id": chunk.document_id,
                            "vector": chunk.embedding
                        }
                        for chunk in batch if chunk.embedding
                    ]
                    insert_or_update_chunks_lancedb(lancedb_data)
                except Exception as e:
                    logger.error(f"Failed to sync batch to LanceDB for document_id={document_id}: {e}", exc_info=True)

            if ok == 0:
                logger.warning("Aucun embedding valide pour document_id=%s", document_id)
                ld.error(
                    "[Embeddings] document_id=%s — aucun vecteur valide écrit, statut failed.",
                    document_id,
                )
                document.processing_status = "failed"
                document.processing_progress = max(document.processing_progress or 0, 90)
                document.updated_at = datetime.utcnow()
                session.add(document)
                session.commit()
                return

            session.refresh(document)
            logger.info(
                "document_id=%s embeddings OK en %.2fs (%s vecteurs)",
                document_id,
                time.perf_counter() - t_embed,
                ok,
            )
            ld.info(
                "[Embeddings] document_id=%s — %d vecteur(s) écrits en %.2fs.",
                document_id,
                ok,
                time.perf_counter() - t_embed,
            )

            document.processing_status = "completed"
            document.processing_progress = 100
            document.updated_at = datetime.utcnow()
            session.add(document)
            session.commit()
            ld.info(
                "[Pipeline] document_id=%s — traitement terminé (completed), progress=100.",
                document_id,
            )
    except Exception as e:
        logger.error("Erreur embeddings document %s: %s", document_id, e, exc_info=True)
        get_library_document_logger().error(
            "[Embeddings] document_id=%s — erreur globale : %s",
            document_id,
            e,
            exc_info=True,
        )
        try:
            with Session(engine) as session:
                document = session.get(Document, document_id)
                if document and document.processing_status not in LIBRARY_USER_STOPPED_STATUSES:
                    document.processing_status = "failed"
                    document.processing_progress = max(document.processing_progress or 0, 85)
                    document.updated_at = datetime.utcnow()
                    session.add(document)
                    session.commit()
        except Exception as upd:
            logger.error(
                "Impossible de marquer le document %s en échec: %s", document_id, upd
            )


def complete_document_embeddings_sync(
    document_id: int, run_id: Optional[str] = None
) -> None:
    """Finalise l'indexation : embeddings des feuilles puis statut completed."""
    _process_embeddings_for_document(document_id, run_id)


# Alias rétrocompatibilité (imports existants)
complete_document_embeddings_and_kag_sync = complete_document_embeddings_sync

_embedding_workers_lock = threading.Lock()
_embedding_workers_initialized = False


def _ensure_embedding_workers() -> None:
    """
    Compatibilité démarrage (app.main).

    Les embeddings des documents bibliothèque sont finalisés dans le pipeline
    du worker document (`complete_document_embeddings_sync`) ou via Celery
    (`process_document_embeddings`). Il n'y a plus de file thread dédiée aux embeddings.
    """
    global _embedding_workers_initialized
    with _embedding_workers_lock:
        if not _embedding_workers_initialized:
            logger.debug(
                "Workers embeddings (threads) : non requis — pipeline document worker ou Celery"
            )
            _embedding_workers_initialized = True
