from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, select
from app.database import get_session
from app.models.library import LibraryRead, LibraryStats
from app.models.folder import FolderCreate, FolderRead, FolderUpdate, FolderWithContents
from app.models.document import (
    DocumentRead,
    DocumentListItem,
    DocumentListItemWithSnapshot,
    DocumentCreate,
    DocumentUpdate,
    Document,
)
from app.models.document_chunk import DocumentChunk
from app.models.folder import Folder
from app.models.space import SpaceRead
from app.models.user import UserRead
from app.routers.auth import get_current_user, require_permission, require_role
from app.services.library_service import (
    get_or_create_user_library,
    get_library_stats,
)
from app.services.folder_service import (
    create_folder, get_folder_by_id, get_folders_by_parent,
    get_folder_path, get_folder_with_contents, rename_folder,
    move_folder, delete_folder
)
from app.services.document_service_new import (
    LIBRARY_QUEUE_ACTIVE_STATUSES,
    create_document,
    get_document_by_id,
    get_documents_by_folder,
    get_documents_by_library,
    save_uploaded_file,
    process_document_async,
    mark_document_reindex_queued,
    move_document,
    delete_document,
    update_document,
    skip_all_library_documents_processing,
    skip_library_document_processing,
    stop_all_library_documents_processing,
    stop_library_document_processing,
)
from app.config import settings
from app.services.task_dispatch import (
    dispatch_document_spaces_update,
    dispatch_reindex_all_library,
    dispatch_reindex_library,
)
from app.services.document_space_service import get_spaces_for_document
from app.services.document_processing_snapshot import (
    build_document_diagnostic,
    build_document_processing_snapshot,
)
from app.services.slot_catalog import (
    compute_classification_status,
    normalize_document_classification,
    serialize_classification_options,
)
from app.services.admin_audit_service import log_admin_action
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/library", tags=["library"])


def _parse_json_string_list(raw: Optional[str]) -> List[str]:
    if not raw or not str(raw).strip():
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(v).strip().lower() for v in parsed if str(v).strip()]
    except json.JSONDecodeError:
        pass
    return []


def _build_document_create_with_classification(
    *,
    title: str,
    content: str,
    document_type: str,
    source_file_path: str,
    processing_status: str,
    processing_progress: int,
    is_paid: bool,
    folder_id: Optional[int],
    product_types_raw: Optional[str] = None,
    materials_raw: Optional[str] = None,
    proferm_gammes_raw: Optional[str] = None,
    source_slug: Optional[str] = None,
) -> DocumentCreate:
    product_types = _parse_json_string_list(product_types_raw)
    materials = _parse_json_string_list(materials_raw)
    proferm_gammes = _parse_json_string_list(proferm_gammes_raw)
    normalized = normalize_document_classification(
        product_types=product_types,
        materials=materials,
        source=source_slug,
        proferm_gammes=proferm_gammes,
    )
    status = compute_classification_status(
        product_types=normalized["product_types"],
        materials=normalized["materials"],
        source=normalized["source"],
        proferm_gammes=normalized["proferm_gammes"],
    )
    return DocumentCreate(
        title=title,
        content=content,
        document_type=document_type,
        source_file_path=source_file_path,
        processing_status=processing_status,
        processing_progress=processing_progress,
        is_paid=is_paid,
        folder_id=folder_id,
        source=normalized["source"],
        product_types=normalized["product_types"],
        materials=normalized["materials"],
        proferm_gammes=normalized["proferm_gammes"],
        classification_status=status,
    )


class LibraryStopDocumentResponse(BaseModel):
    """Réponse enrichie après stop/skip sur un document."""
    document: DocumentRead
    processing_snapshot: dict
    revoked_count: int = 0
    revoked_task_ids: List[str] = Field(default_factory=list)
    processing_run_id: Optional[str] = None


class DocumentSpacesManageRequest(BaseModel):
    add_space_ids: List[int] = Field(default_factory=list)
    remove_space_ids: List[int] = Field(default_factory=list)


class DocumentPageChunkItem(BaseModel):
    chunk_id: int
    chunk_index: int
    is_leaf: bool
    node_id: Optional[str] = None
    parent_node_id: Optional[str] = None
    start_char: int
    end_char: int
    content_preview: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentPageChunksCompare(BaseModel):
    page: int
    chunk_count: int
    chunks: List[DocumentPageChunkItem]


class DocumentChunksByPageResponse(BaseModel):
    document_id: int
    document_title: str
    filename: Optional[str] = None
    total_chunks: int
    pages: List[DocumentPageChunksCompare]


class DocumentChunkMonitorItem(BaseModel):
    chunk_id: int
    chunk_index: int
    page: int
    content_type: str
    token_count: Optional[int] = None
    is_leaf: bool
    node_id: Optional[str] = None
    parent_node_id: Optional[str] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunkEntityItem(BaseModel):
    entity_id: int
    name: str
    entity_type: str
    relation_role: str = "mention"
    relevance_score: float = 1.0
    context_snippet: Optional[str] = None


class DocumentEntityRelationItem(BaseModel):
    entity_a_id: int
    entity_b_id: int
    entity_a_name: str
    entity_b_name: str
    relation_type: str
    relation_label: Optional[str] = None
    confidence: Optional[float] = None


class DocumentKagSummary(BaseModel):
    entity_count: int = 0
    relation_count: int = 0
    chunk_entities: Dict[str, List[DocumentChunkEntityItem]] = Field(default_factory=dict)
    document_relations: List[DocumentEntityRelationItem] = Field(default_factory=list)


class DocumentChunksMonitorResponse(BaseModel):
    document_id: int
    document_title: str
    raw_chunks_count: int
    report_chunks_count: int
    semantic_chunks_count: int = 0
    raw_chunks: List[DocumentChunkMonitorItem]
    report_chunks: List[DocumentChunkMonitorItem]
    semantic_chunks: List[DocumentChunkMonitorItem] = Field(default_factory=list)
    kag: DocumentKagSummary = Field(default_factory=DocumentKagSummary)


@router.get("", response_model=LibraryRead)
async def get_library(
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupère la bibliothèque de l'utilisateur."""
    library = get_or_create_user_library(session, current_user.id)
    return LibraryRead.model_validate(library)


@router.get("/stats", response_model=LibraryStats)
async def get_library_statistics(
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupère les statistiques de la bibliothèque."""
    library = get_or_create_user_library(session, current_user.id)
    stats = get_library_stats(session, library.id, current_user.id)
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Statistiques non disponibles"
        )
    return stats


@router.get("/folders", response_model=List[FolderRead])
async def list_root_folders(
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Liste tous les dossiers racine de la bibliothèque."""
    library = get_or_create_user_library(session, current_user.id)
    folders = session.exec(
        select(Folder).where(
            Folder.parent_folder_id.is_(None),
            Folder.library_id == library.id,
        ).order_by(Folder.name)
    ).all()
    return [FolderRead.model_validate(f) for f in folders]


@router.post("/folders", response_model=FolderRead, status_code=status.HTTP_201_CREATED)
async def create_new_folder(
    folder_create: FolderCreate,
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """Crée un nouveau dossier."""
    library = get_or_create_user_library(session, current_user.id)
    folder = create_folder(session, folder_create, library.id, current_user.id)
    return FolderRead.model_validate(folder)


@router.get("/folders/{folder_id}", response_model=FolderWithContents)
async def get_folder(
    folder_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupère un dossier avec ses sous-dossiers et documents."""
    folder = get_folder_with_contents(session, folder_id, current_user.id)
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dossier non trouvé"
        )
    return folder


@router.get("/folders/{folder_id}/path", response_model=List[FolderRead])
async def get_folder_breadcrumb(
    folder_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupère le chemin complet d'un dossier (breadcrumb)."""
    path = get_folder_path(session, folder_id, current_user.id)
    return path


@router.put("/folders/{folder_id}", response_model=FolderRead)
async def update_folder(
    folder_id: int,
    folder_update: FolderUpdate,
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """Renomme un dossier."""
    folder = rename_folder(session, folder_id, folder_update, current_user.id)
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dossier non trouvé"
        )
    return FolderRead.model_validate(folder)


@router.post("/folders/{folder_id}/move", response_model=FolderRead)
async def move_folder_to_parent(
    folder_id: int,
    new_parent_id: Optional[int] = None,
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """Déplace un dossier vers un nouveau parent."""
    folder = move_folder(session, folder_id, new_parent_id, current_user.id)
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de déplacer le dossier"
        )
    return FolderRead.model_validate(folder)


@router.delete("/folders/{folder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_folder_recursive(
    folder_id: int,
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """Supprime un dossier et tout son contenu (récursif)."""
    success = delete_folder(session, folder_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dossier non trouvé"
        )


@router.get("/documents", response_model=List[DocumentListItemWithSnapshot])
async def list_documents(
    folder_id: Optional[int] = None,
    include_all: bool = False,
    include_processing_snapshot: bool = False,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Liste les documents d'un dossier, de la racine, ou de toute la bibliothèque."""
    library = get_or_create_user_library(session, current_user.id)
    if include_all:
        documents = get_documents_by_library(session, library.id, current_user.id)
    else:
        from app.services.document_service_new import _exclude_feedback_corrective_where

        documents = session.exec(
            select(Document).where(
                Document.library_id == library.id,
                Document.folder_id == folder_id,
                *_exclude_feedback_corrective_where(),
            ).order_by(Document.created_at.desc())
        ).all()
    out: list[DocumentListItemWithSnapshot] = []
    for d in documents:
        base = DocumentListItem.model_validate(d)
        snap = None
        if include_processing_snapshot:
            snap = build_document_processing_snapshot(session, d.id)
        out.append(
            DocumentListItemWithSnapshot(
                **base.model_dump(),
                processing_snapshot=snap,
            )
        )
    return out


@router.get("/documents/{document_id}/chunks-by-page", response_model=DocumentChunksByPageResponse)
async def get_document_chunks_by_page(
    document_id: int,
    include_non_leaf: bool = False,
    preview_chars: int = 300,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Interface de comparaison page-document vs chunks BDD (groupement par page Docling)."""
    document = get_document_by_id(session, document_id, current_user.id)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document non trouvé")

    query = select(DocumentChunk).where(DocumentChunk.document_id == document_id)
    if not include_non_leaf:
        query = query.where(DocumentChunk.is_leaf == True)  # noqa: E712
    query = query.order_by(DocumentChunk.chunk_index.asc())

    chunks = session.exec(query).all()
    grouped: Dict[int, List[DocumentPageChunkItem]] = {}

    for chunk in chunks:
        metadata = dict(chunk.metadata_json or chunk.metadata_ or {})
        page_no = metadata.get("page_no") or metadata.get("page") or metadata.get("page_start") or 0
        try:
            page = int(page_no)
        except (TypeError, ValueError):
            page = 0
        preview = (chunk.content or chunk.text or "")[: max(50, preview_chars)]
        item = DocumentPageChunkItem(
            chunk_id=chunk.id,
            chunk_index=chunk.chunk_index,
            is_leaf=bool(chunk.is_leaf),
            node_id=chunk.node_id,
            parent_node_id=chunk.parent_node_id,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            content_preview=preview,
            metadata=metadata,
        )
        grouped.setdefault(page, []).append(item)

    pages = [
        DocumentPageChunksCompare(page=page, chunk_count=len(items), chunks=items)
        for page, items in sorted(grouped.items(), key=lambda x: x[0])
    ]

    return DocumentChunksByPageResponse(
        document_id=document.id,
        document_title=document.title,
        filename=document.filename,
        total_chunks=len(chunks),
        pages=pages,
    )


@router.get("/documents/{document_id}/chunks-monitor", response_model=DocumentChunksMonitorResponse)
async def get_document_chunks_monitor(
    document_id: int,
    max_chars: int = 6000,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Retourne les chunks d'un document pour monitoring UI :
    - semantic : chunks vision Ministral 3B (semantic_leaf, page_anchor)
    - raw : legacy page_raw_enriched / page_multimodal_section
    - report : legacy page_window_report / page_multimodal_summary
    """
    document = get_document_by_id(session, document_id, current_user.id)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document non trouvé")

    safe_max_chars = max(500, min(max_chars, 20000))
    chunks = session.exec(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
    ).all()

    raw_types = {"page_raw_enriched", "page_multimodal_section"}
    report_types = {
        "page_window_report",
        "page_section_report",
        "page_multimodal_summary",
    }
    semantic_types = {"semantic_leaf", "semantic_section", "document_header"}

    raw_items: List[DocumentChunkMonitorItem] = []
    report_items: List[DocumentChunkMonitorItem] = []
    semantic_items: List[DocumentChunkMonitorItem] = []

    for chunk in chunks:
        metadata = dict(chunk.metadata_json or chunk.metadata_ or {})
        content_type = str(metadata.get("content_type") or "unknown")
        if (
            content_type not in raw_types
            and content_type not in report_types
            and content_type not in semantic_types
        ):
            continue
        page_no = metadata.get("page_no") or metadata.get("page") or metadata.get("page_start") or 0
        try:
            page = int(page_no)
        except (TypeError, ValueError):
            page = 0
        content = (chunk.content or chunk.text or "")[:safe_max_chars]
        item = DocumentChunkMonitorItem(
            chunk_id=chunk.id,
            chunk_index=chunk.chunk_index,
            page=page,
            content_type=content_type,
            token_count=metadata.get("token_count"),
            is_leaf=bool(chunk.is_leaf),
            node_id=chunk.node_id,
            parent_node_id=chunk.parent_node_id,
            content=content,
            metadata=metadata,
        )
        if content_type in semantic_types:
            semantic_items.append(item)
        elif content_type in raw_types:
            raw_items.append(item)
        else:
            report_items.append(item)

    raw_items.sort(key=lambda x: (x.page, x.chunk_index))
    report_items.sort(key=lambda x: (x.page, x.chunk_index))
    semantic_items.sort(key=lambda x: (x.page, x.chunk_index, x.is_leaf))

    from app.services.kag_graph_service import get_document_chunk_entities

    kag_raw = get_document_chunk_entities(session, document_id)
    chunk_entities_parsed: Dict[str, List[DocumentChunkEntityItem]] = {}
    for chunk_key, items in (kag_raw.get("chunk_entities") or {}).items():
        chunk_entities_parsed[chunk_key] = [
            DocumentChunkEntityItem.model_validate(item) for item in items
        ]
    kag_summary = DocumentKagSummary(
        entity_count=int(kag_raw.get("entity_count") or 0),
        relation_count=int(kag_raw.get("relation_count") or 0),
        chunk_entities=chunk_entities_parsed,
        document_relations=[
            DocumentEntityRelationItem.model_validate(r)
            for r in (kag_raw.get("document_relations") or [])
        ],
    )

    return DocumentChunksMonitorResponse(
        document_id=document.id,
        document_title=document.title,
        raw_chunks_count=len(raw_items),
        report_chunks_count=len(report_items),
        semantic_chunks_count=len(semantic_items),
        raw_chunks=raw_items,
        report_chunks=report_items,
        semantic_chunks=semantic_items,
        kag=kag_summary,
    )


@router.post("/documents/stop-all", status_code=status.HTTP_200_OK)
async def stop_all_library_documents_endpoint(
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    """Annule tous les documents en file (admin uniquement)."""
    body = stop_all_library_documents_processing(session, current_user.id)
    log_admin_action(
        user_id=current_user.id,
        action="library.stop_all",
        detail=body,
    )
    return body


@router.post("/documents/skip-all", status_code=status.HTTP_200_OK)
async def skip_all_library_documents_endpoint(
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    """Ignore les documents en attente et arrête celui en cours (admin uniquement)."""
    body = skip_all_library_documents_processing(session, current_user.id)
    log_admin_action(
        user_id=current_user.id,
        action="library.skip_all",
        detail=body,
    )
    return body


@router.get("/documents/{document_id}", response_model=DocumentRead)
async def get_document(
    document_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupère un document par son ID."""
    library = get_or_create_user_library(session, current_user.id)
    document = session.exec(
        select(Document).where(
            Document.id == document_id,
            Document.library_id == library.id,
        )
    ).first()
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document non trouvé"
        )
    return DocumentRead.model_validate(document)


@router.post(
    "/documents/{document_id}/stop",
    response_model=LibraryStopDocumentResponse,
)
async def stop_single_library_document(
    document_id: int,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    """Arrête le traitement d'un document sans le supprimer (admin uniquement)."""
    library = get_or_create_user_library(session, current_user.id)
    document = session.exec(
        select(Document).where(
            Document.id == document_id,
            Document.library_id == library.id,
        )
    ).first()
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document non trouvé",
        )
    if document.processing_status not in LIBRARY_QUEUE_ACTIVE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Aucun traitement en cours ou en attente pour ce document",
        )
    updated, revoke_info = stop_library_document_processing(
        session, document_id, current_user.id
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossible d'arrêter le traitement",
        )
    session.refresh(updated)
    snap = build_document_processing_snapshot(session, document_id)
    log_admin_action(
        user_id=current_user.id,
        action="library.stop_document",
        detail={
            "document_id": document_id,
            **revoke_info,
        },
    )
    return LibraryStopDocumentResponse(
        document=DocumentRead.model_validate(updated),
        processing_snapshot=snap,
        revoked_count=int(revoke_info.get("revoked_count") or 0),
        revoked_task_ids=list(revoke_info.get("revoked_task_ids") or []),
        processing_run_id=updated.processing_run_id,
    )


@router.post(
    "/documents/{document_id}/skip",
    response_model=LibraryStopDocumentResponse,
)
async def skip_single_library_document(
    document_id: int,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    """Ignore un document en attente ou arrête s'il est déjà en cours (admin uniquement)."""
    library = get_or_create_user_library(session, current_user.id)
    document = session.exec(
        select(Document).where(
            Document.id == document_id,
            Document.library_id == library.id,
        )
    ).first()
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document non trouvé",
        )
    if document.processing_status not in LIBRARY_QUEUE_ACTIVE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Aucun traitement en cours ou en attente pour ce document",
        )
    updated, revoke_info = skip_library_document_processing(
        session, document_id, current_user.id
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossible d'ignorer le document",
        )
    session.refresh(updated)
    snap = build_document_processing_snapshot(session, document_id)
    log_admin_action(
        user_id=current_user.id,
        action="library.skip_document",
        detail={"document_id": document_id, **revoke_info},
    )
    return LibraryStopDocumentResponse(
        document=DocumentRead.model_validate(updated),
        processing_snapshot=snap,
        revoked_count=int(revoke_info.get("revoked_count") or 0),
        revoked_task_ids=list(revoke_info.get("revoked_task_ids") or []),
        processing_run_id=updated.processing_run_id,
    )


@router.get("/documents/{document_id}/processing-health")
async def get_document_processing_health(
    document_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Statut pipeline : chunks, embeddings, compteurs KAG (lecture seule)."""
    library = get_or_create_user_library(session, current_user.id)
    document = session.exec(
        select(Document).where(
            Document.id == document_id,
            Document.library_id == library.id,
        )
    ).first()
    if document is None:
        raise HTTPException(status_code=404, detail="Document non trouvé")
    return build_document_processing_snapshot(session, document_id)


@router.get("/documents/{document_id}/diagnostic")
async def get_document_diagnostic(
    document_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Diagnostic : checks chunks, embeddings, entités, relations."""
    library = get_or_create_user_library(session, current_user.id)
    document = session.exec(
        select(Document).where(
            Document.id == document_id,
            Document.library_id == library.id,
        )
    ).first()
    if document is None:
        raise HTTPException(status_code=404, detail="Document non trouvé")
    return build_document_diagnostic(session, document_id)


@router.put("/documents/{document_id}", response_model=DocumentRead)
async def update_library_document(
    document_id: int,
    document_update: DocumentUpdate,
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """Met à jour les métadonnées d'un document."""
    document = update_document(session, document_id, document_update, current_user.id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document non trouvé"
        )
    return DocumentRead.model_validate(document)


class ReindexRequest(BaseModel):
    """Corps de la requête de retraitement."""
    mode: str = Field(default="full", description="full | text_only | colpali_only")


@router.post("/documents/{document_id}/reindex", status_code=status.HTTP_200_OK)
async def reindex_library_document_endpoint(
    document_id: int,
    body: ReindexRequest = ReindexRequest(),
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    """
    Enfile le retraitement d'un document sur Celery.

    Modes disponibles :
    - full        : Vision Ministral 3B + mistral-embed + ColPali (pipeline complet, défaut)
    - text_only   : Vision Ministral 3B + mistral-embed uniquement (ColPali inchangé)
    - colpali_only: re-sync visuel ColPali uniquement (chunks texte inchangés)

    Marque le document en reindex_queued (chunks encore disponibles pour le RAG).
    """
    from app.services.document_indexing_service import IndexingMode

    valid_modes = {m.value for m in IndexingMode}
    if body.mode not in valid_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Mode invalide '{body.mode}'. Valeurs acceptées : {', '.join(sorted(valid_modes))}",
        )
    if body.mode in ("full", "text_only") and not settings.MISTRAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MISTRAL_API_KEY requise pour les modes full et text_only.",
        )
    if body.mode in ("full", "colpali_only") and not settings.COLPALI_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ColPali désactivé (COLPALI_ENABLED=false). Modes full et colpali_only nécessitent ColPali.",
        )

    library = get_or_create_user_library(session, current_user.id)
    document = session.exec(
        select(Document).where(
            Document.id == document_id,
            Document.library_id == library.id,
        )
    ).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document non trouvé",
        )
    if document.document_type != "document" or not document.source_file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Réindexation réservée aux documents avec fichier source.",
        )
    mark_document_reindex_queued(session, document_id, current_user.id)
    try:
        celery_task_id = dispatch_reindex_library(document_id, current_user.id, mode=body.mode)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    log_admin_action(
        user_id=current_user.id,
        action="library.reindex_document",
        detail={"document_id": document_id, "celery_task_id": celery_task_id, "mode": body.mode},
    )
    return {
        "status": "queued",
        "celery_task_id": celery_task_id,
        "document_id": document_id,
        "mode": body.mode,
    }


class ReindexAllRequest(BaseModel):
    """Corps de la requête de retraitement global."""
    mode: str = Field(default="full", description="full | text_only | colpali_only")


@router.post("/reindex-all", status_code=status.HTTP_200_OK)
async def reindex_all_library_endpoint(
    body: ReindexAllRequest = ReindexAllRequest(),
    current_user: UserRead = Depends(require_role("admin")),
):
    """
    Enfile le retraitement de tous les documents de la bibliothèque.

    Modes disponibles :
    - full        : Vision Ministral 3B + mistral-embed + ColPali (défaut)
    - text_only   : Vision Ministral 3B + mistral-embed uniquement
    - colpali_only: re-sync visuel ColPali uniquement
    """
    from app.services.document_indexing_service import IndexingMode

    valid_modes = {m.value for m in IndexingMode}
    if body.mode not in valid_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Mode invalide '{body.mode}'. Valeurs acceptées : {', '.join(sorted(valid_modes))}",
        )
    if body.mode in ("full", "text_only") and not settings.MISTRAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MISTRAL_API_KEY requise pour les modes full et text_only.",
        )
    if body.mode in ("full", "colpali_only") and not settings.COLPALI_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ColPali désactivé. Modes full et colpali_only nécessitent ColPali.",
        )
    try:
        celery_task_id = dispatch_reindex_all_library(current_user.id, mode=body.mode)
        log_admin_action(
            user_id=current_user.id,
            action="library.reindex_all",
            detail={"celery_task_id": celery_task_id, "mode": body.mode},
        )
        return {
            "status": "queued",
            "celery_task_id": celery_task_id,
            "mode": body.mode,
        }
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )


@router.get("/documents/{document_id}/file")
async def get_document_file(
    document_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupère le fichier source d'un document."""
    library = get_or_create_user_library(session, current_user.id)
    document = session.exec(
        select(Document).where(
            Document.id == document_id,
            Document.library_id == library.id,
        )
    ).first()
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document non trouvé"
        )
    
    if not document.source_file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ce document n'a pas de fichier source"
        )
    
    file_path = Path(document.source_file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Le fichier source n'existe plus"
        )
    
    media_type = "application/pdf" if file_path.suffix.lower() == ".pdf" else "application/octet-stream"
    
    response = FileResponse(
        path=str(file_path),
        filename=f"{document.title}{file_path.suffix}",
        media_type=media_type
    )
    response.headers["Accept-Ranges"] = "bytes"
    response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Range, Accept-Ranges"
    
    return response


@router.get("/classification-options")
async def get_classification_options(
    current_user: UserRead = Depends(get_current_user),
):
    """Taxonomie partagée document / chat (familles, matériaux, gammes, fournisseurs)."""
    return serialize_classification_options()


@router.post("/upload", response_model=List[DocumentRead], status_code=status.HTTP_201_CREATED)
async def upload_documents(
    files: List[UploadFile] = File(...),
    space_ids: str = Form("[]"),
    is_paid: bool = Form(False),
    folder_id: Optional[int] = Form(None),
    product_types: str = Form("[]"),
    materials: str = Form("[]"),
    proferm_gammes: str = Form("[]"),
    source: Optional[str] = Form(None),
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """
    Upload unifié de documents avec sélection d'espaces.
    
    - files: Liste de fichiers à uploader
    - space_ids: JSON array d'IDs d'espaces (ex: "[1,2,3]")
    - folder_id: ID du dossier destination (optionnel, None = racine)
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Aucun fichier fourni"
        )
    
    try:
        space_ids_list = json.loads(space_ids)
        if not isinstance(space_ids_list, list):
            raise ValueError("space_ids doit être un tableau")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Format space_ids invalide: {str(e)}"
        )
    
    library = get_or_create_user_library(session, current_user.id)
    
    created_documents = []
    errors = []
    
    import os

    single_file_upload = len(files) == 1

    # Traitement strictement séquentiel, fichier par fichier.
    for file in files:
        filename = file.filename or "fichier_inconnu"
        try:
            file_content = await file.read()

            file_path = save_uploaded_file(file_content, filename)

            if not file_path:
                errors.append(f"Erreur lors de la sauvegarde du fichier '{filename}'")
                continue

            filename_without_ext = os.path.splitext(filename)[0]

            apply_metadata = single_file_upload
            document_create = _build_document_create_with_classification(
                title=filename_without_ext,
                content="⏳ Traitement en cours...",
                document_type="document",
                source_file_path=file_path,
                processing_status="pending",
                processing_progress=0,
                is_paid=is_paid,
                folder_id=folder_id,
                product_types_raw=product_types if apply_metadata else None,
                materials_raw=materials if apply_metadata else None,
                proferm_gammes_raw=proferm_gammes if apply_metadata else None,
                source_slug=source if apply_metadata else None,
            )
            
            document = create_document(
                session,
                document_create,
                library.id,
                current_user.id,
                space_ids_list
            )
            
            if not document:
                errors.append(f"Impossible de créer le document pour '{filename}'")
                continue
            
            process_document_async(document.id, file_path)
            
            created_documents.append(document)
            logger.info(f"Document '{filename}' uploadé, traitement en arrière-plan (document ID: {document.id})")
        except Exception as e:
            logger.error(
                f"Erreur lors du traitement du document '{filename}': {e}",
                exc_info=True,
            )
            errors.append(f"Erreur lors du traitement de '{filename}': {str(e)}")
    
    if not created_documents:
        error_message = "; ".join(errors) if errors else "Aucun fichier n'a pu être uploadé"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_message
        )
    
    if errors:
        logger.warning(f"Certains fichiers n'ont pas pu être uploadés: {errors}")
    
    return [DocumentRead.model_validate(doc) for doc in created_documents]


@router.get("/documents/{document_id}/spaces", response_model=List[SpaceRead])
async def list_document_spaces(
    document_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Liste les espaces ayant accès à un document."""
    document = get_document_by_id(session, document_id, current_user.id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document non trouvé"
        )
    
    spaces = get_spaces_for_document(session, document_id, current_user.id)
    return [SpaceRead.model_validate(s) for s in spaces]


@router.post("/documents/{document_id}/spaces", status_code=status.HTTP_200_OK)
async def manage_document_spaces(
    document_id: int,
    payload: DocumentSpacesManageRequest,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Ajoute ou retire un document de plusieurs espaces via worker."""
    document = get_document_by_id(session, document_id, current_user.id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document non trouvé"
        )

    # Vérification des autorisations sur les espaces concernés
    is_admin = "admin" in (current_user.roles or [])
    has_library_write = "library.write" in (current_user.permissions or [])
    
    all_target_space_ids = set((payload.add_space_ids or []) + (payload.remove_space_ids or []))
    if all_target_space_ids:
        from app.models.space import Space
        statement = select(Space).where(Space.id.in_(list(all_target_space_ids)))
        target_spaces = session.exec(statement).all()
        target_spaces_dict = {s.id: s for s in target_spaces}

        for space_id in all_target_space_ids:
            space = target_spaces_dict.get(space_id)
            if not space:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Espace {space_id} non trouvé"
                )
            
            if space.is_shared:
                # Seuls les utilisateurs avec library.write peuvent modifier les associations des espaces communs
                if not has_library_write:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Les lecteurs ne peuvent pas ajouter ou retirer de documents dans des espaces communs."
                    )
            else:
                # Pour les espaces privés, l'utilisateur doit en être le propriétaire
                if space.user_id != current_user.id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Vous n'êtes pas autorisé à modifier l'espace privé {space_id}."
                    )

    if not payload.add_space_ids and not payload.remove_space_ids:
        return {"status": "noop", "message": "Aucun changement à appliquer"}

    task_id = dispatch_document_spaces_update(
        document_id=document_id,
        add_space_ids=payload.add_space_ids,
        remove_space_ids=payload.remove_space_ids,
        user_id=current_user.id,
    )

    return {
        "status": "queued",
        "message": "Mise à jour des espaces planifiée",
        "task_id": task_id,
        "document_id": document_id,
    }


@router.post("/documents/{document_id}/move", response_model=DocumentRead)
async def move_document_to_folder(
    document_id: int,
    new_folder_id: Optional[int] = None,
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """Déplace un document vers un nouveau dossier."""
    document = move_document(session, document_id, new_folder_id, current_user.id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de déplacer le document"
        )
    return DocumentRead.model_validate(document)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library_document(
    document_id: int,
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """Supprime un document de la bibliothèque et ses données associées."""
    success = delete_document(session, document_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document non trouvé"
        )


@router.get("/documents/{document_id}/export")
async def export_library_document(
    document_id: int,
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """
    Exporte un document sous forme d'archive ZIP contenant :
    - Le fichier PDF source.
    - Les métadonnées Postgres (Document, DocumentChunks) au format JSON.
    - Les vecteurs d'embeddings de patchs ColPali de LanceDB au format JSON.
    """
    import io
    import zipfile
    from fastapi.responses import StreamingResponse

    library = get_or_create_user_library(session, current_user.id)
    document = session.exec(
        select(Document).where(
            Document.id == document_id,
            Document.library_id == library.id,
        )
    ).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document non trouvé"
        )

    if not document.source_file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ce document n'a pas de fichier source"
        )

    file_path = Path(document.source_file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Le fichier source n'existe plus"
        )

    chunks = session.exec(
        select(DocumentChunk).where(DocumentChunk.document_id == document_id)
    ).all()

    patches_data = []
    try:
        from app.services.lancedb_service import get_colpali_table
        table = get_colpali_table()
        if table:
            rows = table.search().where(f"document_id = {document_id}").to_list()
            for r in rows:
                vec = r["vector"]
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()
                patches_data.append({
                    "chunk_id": int(r["chunk_id"]),
                    "patch_index": int(r["patch_index"]),
                    "vector": vec
                })
    except Exception as ex:
        logger.warning(f"Impossible de récupérer les patchs ColPali pour l'export : {ex}")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        zip_file.write(file_path, arcname=file_path.name)

        metadata = {
            "document": {
                "title": document.title,
                "filename": Path(document.source_file_path).name if document.source_file_path else None,
                "document_type": document.document_type,
                "content": document.content,
                "processing_status": document.processing_status,
                "processing_progress": document.processing_progress,
                "is_paid": document.is_paid,
                "source": document.source,
            },
            "chunks": [
                {
                    "id": c.id,
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "text": c.text,
                    "is_leaf": c.is_leaf,
                    "hierarchy_level": c.hierarchy_level,
                    "node_id": c.node_id,
                    "parent_node_id": c.parent_node_id,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "metadata_json": c.metadata_json or c.metadata_ or {},
                    "embedding": c.embedding.tolist() if hasattr(c.embedding, "tolist") else c.embedding,
                    "source": c.source
                }
                for c in chunks
            ]
        }
        zip_file.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))

        patches_json = {
            "patches": patches_data
        }
        zip_file.writestr("colpali_patches.json", json.dumps(patches_json, ensure_ascii=False))

    zip_buffer.seek(0)
    headers = {
        'Content-Disposition': f'attachment; filename="export_doc_{document_id}.zip"'
    }
    return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)


@router.get("/export-all")
async def export_all_library_documents(
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """
    Exporte tous les documents de la bibliothèque sous forme d'une archive ZIP globale.
    Chaque document est stocké dans un sous-dossier contenant son fichier source, ses métadonnées et ses patchs ColPali.
    """
    import io
    import zipfile
    from fastapi.responses import StreamingResponse
    from app.services.lancedb_service import get_colpali_table

    library = get_or_create_user_library(session, current_user.id)
    documents = get_documents_by_library(session, library.id, current_user.id)
    
    zip_buffer = io.BytesIO()
    table = get_colpali_table()
    
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for document in documents:
            if not document.source_file_path:
                continue
                
            file_path = Path(document.source_file_path)
            if not file_path.exists():
                continue
                
            doc_dir = f"doc_{document.id}"
            
            # Ecrire le fichier PDF
            zip_file.write(file_path, arcname=f"{doc_dir}/{file_path.name}")
            
            # Récupérer les chunks
            chunks = session.exec(
                select(DocumentChunk).where(DocumentChunk.document_id == document.id)
            ).all()
            
            # Récupérer les patchs ColPali
            patches_data = []
            if table:
                try:
                    rows = table.search().where(f"document_id = {document.id}").to_list()
                    for r in rows:
                        vec = r["vector"]
                        if hasattr(vec, "tolist"):
                            vec = vec.tolist()
                        patches_data.append({
                            "chunk_id": int(r["chunk_id"]),
                            "patch_index": int(r["patch_index"]),
                            "vector": vec
                        })
                except Exception as ex:
                    logger.warning(f"Impossible de récupérer les patchs ColPali pour l'export du doc {document.id} : {ex}")
            
            # Ecrire metadata.json
            metadata = {
                "document": {
                    "title": document.title,
                    "filename": file_path.name,
                    "document_type": document.document_type,
                    "content": document.content,
                    "processing_status": document.processing_status,
                    "processing_progress": document.processing_progress,
                    "is_paid": document.is_paid,
                    "source": document.source,
                },
                "chunks": [
                    {
                        "id": c.id,
                        "chunk_index": c.chunk_index,
                        "content": c.content,
                        "text": c.text,
                        "is_leaf": c.is_leaf,
                        "hierarchy_level": c.hierarchy_level,
                        "node_id": c.node_id,
                        "parent_node_id": c.parent_node_id,
                        "start_char": c.start_char,
                        "end_char": c.end_char,
                        "metadata_json": c.metadata_json or c.metadata_ or {},
                        "embedding": c.embedding.tolist() if hasattr(c.embedding, "tolist") else c.embedding,
                        "source": c.source
                    }
                    for c in chunks
                ]
            }
            zip_file.writestr(f"{doc_dir}/metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
            
            # Ecrire colpali_patches.json
            patches_json = {"patches": patches_data}
            zip_file.writestr(f"{doc_dir}/colpali_patches.json", json.dumps(patches_json, ensure_ascii=False))
            
    zip_buffer.seek(0)
    headers = {
        'Content-Disposition': 'attachment; filename="export_library_all.zip"'
    }
    return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)


@router.post("/documents/import", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def import_library_document(
    file: UploadFile = File(...),
    folder_id: Optional[int] = Form(None),
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """
    Importe un package ZIP contenant un document ou plusieurs documents (bulk export) :
    - Recrée l'enregistrement du Document et des DocumentChunks associés dans Postgres.
    - Sauvegarde le fichier source (PDF) dans le dossier media local.
    - Ré-injecte les vecteurs d'embeddings ColPali dans LanceDB avec les nouveaux IDs de chunks.
    """
    import io
    import zipfile
    import os

    zip_contents = await file.read()
    zip_buffer = io.BytesIO(zip_contents)

    try:
        with zipfile.ZipFile(zip_buffer, "r") as zip_file:
            namelist = zip_file.namelist()
            
            # Mode 1 : ZIP contenant un seul document à la racine
            if "metadata.json" in namelist:
                metadata_str = zip_file.read("metadata.json").decode("utf-8")
                metadata = json.loads(metadata_str)

                doc_data = metadata["document"]
                chunks_data = metadata.get("chunks", [])

                pdf_filename = None
                for name in namelist:
                    if name != "metadata.json" and name != "colpali_patches.json":
                        pdf_filename = name
                        break

                if not pdf_filename:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Archive invalide : fichier source manquant"
                    )

                pdf_content = zip_file.read(pdf_filename)
                file_path = save_uploaded_file(pdf_content, pdf_filename)
                if not file_path:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Impossible de sauvegarder le fichier PDF importé"
                    )

                library = get_or_create_user_library(session, current_user.id)
                from app.services.document_service_new import infer_document_source
                inferred_source = doc_data.get("source") or infer_document_source(file_path, doc_data.get("content"))

                new_doc = Document(
                    title=doc_data["title"],
                    filename=doc_data.get("filename") or pdf_filename,
                    content=doc_data.get("content", ""),
                    document_type=doc_data.get("document_type", "document"),
                    source_file_path=file_path,
                    processing_status=doc_data.get("processing_status", "completed"),
                    processing_progress=doc_data.get("processing_progress", 100),
                    is_paid=doc_data.get("is_paid", False),
                    source=inferred_source,
                    library_id=library.id,
                    user_id=current_user.id,
                    folder_id=folder_id
                )
                session.add(new_doc)
                session.commit()
                session.refresh(new_doc)

                old_chunk_to_new_chunk_id = {}
                for chunk_data in chunks_data:
                    new_chunk = DocumentChunk(
                        document_id=new_doc.id,
                        chunk_index=chunk_data["chunk_index"],
                        content=chunk_data.get("content", ""),
                        text=chunk_data.get("text", ""),
                        is_leaf=chunk_data.get("is_leaf", True),
                        hierarchy_level=chunk_data.get("hierarchy_level", 0),
                        node_id=chunk_data.get("node_id"),
                        parent_node_id=chunk_data.get("parent_node_id"),
                        start_char=chunk_data.get("start_char", 0),
                        end_char=chunk_data.get("end_char", 0),
                        metadata_json=chunk_data.get("metadata_json", {}),
                        metadata_=chunk_data.get("metadata_json", {}),
                        embedding=chunk_data.get("embedding"),
                        source=chunk_data.get("source") or inferred_source
                    )
                    session.add(new_chunk)
                    session.commit()
                    session.refresh(new_chunk)

                    old_chunk_to_new_chunk_id[chunk_data["id"]] = new_chunk.id

                if "colpali_patches.json" in namelist:
                    patches_str = zip_file.read("colpali_patches.json").decode("utf-8")
                    patches_json = json.loads(patches_str)
                    patches_list = patches_json.get("patches", [])

                    from collections import defaultdict
                    chunk_id_to_vectors = defaultdict(list)
                    for p in patches_list:
                        chunk_id_to_vectors[p["chunk_id"]].append(p)

                    chunk_patches_list = []
                    for old_chunk_id, p_items in chunk_id_to_vectors.items():
                        new_chunk_id = old_chunk_to_new_chunk_id.get(old_chunk_id)
                        if new_chunk_id:
                            p_items.sort(key=lambda x: x["patch_index"])
                            vectors = [item["vector"] for item in p_items]
                            chunk_patches_list.append((new_chunk_id, vectors))

                    if chunk_patches_list:
                        from app.services.lancedb_service import insert_colpali_patches_batch_lancedb
                        insert_colpali_patches_batch_lancedb(new_doc.id, chunk_patches_list)

                logger.info(f"Document ID {new_doc.id} ('{new_doc.title}') importé avec succès.")
                return new_doc

            # Mode 2 : ZIP global contenant plusieurs dossiers de documents
            else:
                from collections import defaultdict
                folders = defaultdict(list)
                for name in namelist:
                    parts = name.split("/")
                    if len(parts) > 1:
                        folder_name = parts[0]
                        folders[folder_name].append(name)

                imported_docs = []
                for folder, files_in_folder in folders.items():
                    metadata_file = next((f for f in files_in_folder if f.endswith("metadata.json")), None)
                    if not metadata_file:
                        continue

                    metadata_str = zip_file.read(metadata_file).decode("utf-8")
                    metadata = json.loads(metadata_str)

                    doc_data = metadata["document"]
                    chunks_data = metadata.get("chunks", [])

                    pdf_filename = None
                    for name in files_in_folder:
                        if not name.endswith("metadata.json") and not name.endswith("colpali_patches.json") and not name.endswith("/"):
                            pdf_filename = name
                            break

                    if not pdf_filename:
                        continue

                    pdf_content = zip_file.read(pdf_filename)
                    base_pdf_name = os.path.basename(pdf_filename)
                    file_path = save_uploaded_file(pdf_content, base_pdf_name)
                    if not file_path:
                        continue

                    library = get_or_create_user_library(session, current_user.id)
                    from app.services.document_service_new import infer_document_source
                    inferred_source = doc_data.get("source") or infer_document_source(file_path, doc_data.get("content"))

                    new_doc = Document(
                        title=doc_data["title"],
                        filename=doc_data.get("filename") or base_pdf_name,
                        content=doc_data.get("content", ""),
                        document_type=doc_data.get("document_type", "document"),
                        source_file_path=file_path,
                        processing_status=doc_data.get("processing_status", "completed"),
                        processing_progress=doc_data.get("processing_progress", 100),
                        is_paid=doc_data.get("is_paid", False),
                        source=inferred_source,
                        library_id=library.id,
                        user_id=current_user.id,
                        folder_id=folder_id
                    )
                    session.add(new_doc)
                    session.commit()
                    session.refresh(new_doc)

                    old_chunk_to_new_chunk_id = {}
                    for chunk_data in chunks_data:
                        new_chunk = DocumentChunk(
                            document_id=new_doc.id,
                            chunk_index=chunk_data["chunk_index"],
                            content=chunk_data.get("content", ""),
                            text=chunk_data.get("text", ""),
                            is_leaf=chunk_data.get("is_leaf", True),
                            hierarchy_level=chunk_data.get("hierarchy_level", 0),
                            node_id=chunk_data.get("node_id"),
                            parent_node_id=chunk_data.get("parent_node_id"),
                            start_char=chunk_data.get("start_char", 0),
                            end_char=chunk_data.get("end_char", 0),
                            metadata_json=chunk_data.get("metadata_json", {}),
                            metadata_=chunk_data.get("metadata_json", {}),
                            embedding=chunk_data.get("embedding"),
                            source=chunk_data.get("source") or inferred_source
                        )
                        session.add(new_chunk)
                        session.commit()
                        session.refresh(new_chunk)

                        old_chunk_to_new_chunk_id[chunk_data["id"]] = new_chunk.id

                    patches_file = next((f for f in files_in_folder if f.endswith("colpali_patches.json")), None)
                    if patches_file:
                        patches_str = zip_file.read(patches_file).decode("utf-8")
                        patches_json = json.loads(patches_str)
                        patches_list = patches_json.get("patches", [])

                        chunk_id_to_vectors = defaultdict(list)
                        for p in patches_list:
                            chunk_id_to_vectors[p["chunk_id"]].append(p)

                        chunk_patches_list = []
                        for old_chunk_id, p_items in chunk_id_to_vectors.items():
                            new_chunk_id = old_chunk_to_new_chunk_id.get(old_chunk_id)
                            if new_chunk_id:
                                p_items.sort(key=lambda x: x["patch_index"])
                                vectors = [item["vector"] for item in p_items]
                                chunk_patches_list.append((new_chunk_id, vectors))

                        if chunk_patches_list:
                            from app.services.lancedb_service import insert_colpali_patches_batch_lancedb
                            insert_colpali_patches_batch_lancedb(new_doc.id, chunk_patches_list)

                    imported_docs.append(new_doc)
                    logger.info(f"Document ID {new_doc.id} ('{new_doc.title}') importé avec succès du package global.")

                if not imported_docs:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Aucun document valide n'a pu être importé de l'archive globale."
                    )
                return imported_docs[0]

    except Exception as e:
        logger.error(f"Erreur lors de l'import du document : {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de l'import du document : {str(e)}"
        )
