"""
Retraitement multimodal additif : pymupdf4llm (texte) + mistral-small (vision) → 1 chunk / page.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from typing import List, Optional, Tuple

import httpx
from sqlalchemy import delete
from sqlmodel import Session, select

from app.config import settings
from app.database import engine
from app.library_document_logging import get_library_document_logger
from app.models.document import Document
from app.models.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)

PAGE_MULTIMODAL_CONTENT_TYPE = "page_multimodal_summary"
CHUNKING_VERSION_MULTIMODAL = "multimodal_page_v1"

_SYSTEM_PROMPT = """Tu es un expert technique en documentation industrielle et normative.
Tu reçois le texte extrait automatiquement (pymupdf) et l'image de la page.
Produis une synthèse professionnelle en markdown structuré :
- ## Texte — structure et conserve le contenu pymupdf (ne le réécris pas inutilement)
- ## Tableaux — tableaux en markdown si présents
- ## Schémas et figures — décris uniquement ce qui est visible et non couvert par le texte pymupdf
- ## Normes et contraintes — listes à puces des exigences, références normatives, limites
- ## Synthèse page — 5 à 8 phrases factuelles
Règles : français ; n'invente rien ; si le texte pymupdf est vide, base-toi sur l'image seule ;
signale « illisible » pour les zones floues."""

_USER_PROMPT_TEMPLATE = """Document : {title}
Page : {page_no}

Texte extrait automatiquement (pymupdf) :
---
{pymupdf_text}
---

Synthétise cette page en t'appuyant sur le texte ci-dessus et sur l'image jointe."""


def _multimodal_page_model() -> str:
    return (settings.MULTIMODAL_PAGE_MODEL or "mistral-small-latest").strip()


def delete_multimodal_chunks_for_document(
    session: Session, document_id: int, commit: bool = True
) -> int:
    """Supprime les chunks page_multimodal_summary d'un document."""
    col = DocumentChunk.metadata_json["content_type"].as_string()
    result = session.execute(
        delete(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            col == PAGE_MULTIMODAL_CONTENT_TYPE,
        )
    )
    if commit:
        session.commit()
    deleted = result.rowcount if result.rowcount is not None else 0
    logger.debug(
        "Supprimé %s chunk(s) multimodal pour document_id=%s",
        deleted,
        document_id,
    )
    return deleted


def render_pdf_page_png(pdf_path: str, page_index: int, dpi: Optional[int] = None) -> bytes:
    """Rend une page PDF (0-based) en PNG."""
    from pdf2image import convert_from_path

    dpi_val = dpi or settings.MULTIMODAL_PAGE_DPI or 200
    images = convert_from_path(
        pdf_path,
        dpi=dpi_val,
        first_page=page_index + 1,
        last_page=page_index + 1,
    )
    if not images:
        raise ValueError(f"Impossible de rendre la page {page_index + 1}")
    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return buf.getvalue()


def synthesize_page_with_mistral_small(
    image_png: bytes,
    page_no: int,
    pymupdf_text: str,
    document_title: str,
) -> str:
    """Appel synchrone mistral-small avec image + contexte pymupdf."""
    api_key = settings.MISTRAL_API_KEY
    if not api_key:
        raise ValueError("MISTRAL_API_KEY n'est pas configurée")

    b64 = base64.b64encode(image_png).decode("ascii")
    pymupdf_block = pymupdf_text.strip() if pymupdf_text else "(aucun texte extractible sur cette page)"
    user_text = _USER_PROMPT_TEMPLATE.format(
        title=document_title or "Document",
        page_no=page_no,
        pymupdf_text=pymupdf_block,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{b64}",
                },
            ],
        },
    ]

    payload = {
        "model": _multimodal_page_model(),
        "messages": messages,
        "stream": False,
        "max_tokens": settings.MULTIMODAL_PAGE_MAX_TOKENS,
        "temperature": 0.2,
    }
    base_url = (settings.MISTRAL_BASE_URL or "https://api.mistral.ai").rstrip("/")
    timeout = float(getattr(settings, "MISTRAL_OCR_TIMEOUT", 300) or 300)

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = (msg.get("content") or "").strip()
    if not content:
        raise RuntimeError(f"Synthèse vide pour la page {page_no}")
    return content


def _next_chunk_index(session: Session, document_id: int) -> int:
    rows = session.exec(
        select(DocumentChunk.chunk_index).where(
            DocumentChunk.document_id == document_id
        )
    ).all()
    if not rows:
        return 0
    return max(int(r) for r in rows) + 1


def append_multimodal_page_chunks(
    session: Session,
    document: Document,
    page_contents: List[Tuple[int, str]],
) -> List[DocumentChunk]:
    """Persiste les chunks multimodal (sans embeddings)."""
    base_index = _next_chunk_index(session, document.id)
    model_name = _multimodal_page_model()
    chunks: List[DocumentChunk] = []

    for offset, (page_no, content) in enumerate(page_contents):
        if not content or not content.strip():
            continue
        meta = {
            "content_type": PAGE_MULTIMODAL_CONTENT_TYPE,
            "chunking_version": CHUNKING_VERSION_MULTIMODAL,
            "page_no": page_no,
            "page_start": page_no,
            "page_end": page_no,
            "generation_method": "pymupdf+mistral_small",
            "llm_model": model_name,
            "document_id": document.id,
            "document_title": document.title or "",
        }
        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=base_index + offset,
            content=content.strip(),
            text=content.strip(),
            start_char=0,
            end_char=len(content),
            node_id=f"multimodal-page-{document.id}-{page_no}",
            parent_node_id=None,
            is_leaf=True,
            hierarchy_level=0,
            metadata_json=meta,
            metadata_=meta,
            source=document.source,
        )
        chunks.append(chunk)

    if chunks:
        session.add_all(chunks)
        session.commit()
    return chunks


def build_multimodal_pages_for_pdf(
    pdf_path: str,
    document_title: str,
    *,
    max_pages: Optional[int] = None,
) -> List[Tuple[int, str]]:
    """
    Pour chaque page : pymupdf + rendu PNG + synthèse mistral-small.
    """
    from app.services.pdf_extraction_service import extract_page_texts_from_pdf

    ld = get_library_document_logger()
    page_texts = extract_page_texts_from_pdf(pdf_path)
    if not page_texts:
        from pdf2image import convert_from_path

        dpi_val = settings.MULTIMODAL_PAGE_DPI or 200
        images = convert_from_path(pdf_path, dpi=dpi_val)
        page_texts = [(i + 1, "") for i in range(len(images))]

    limit = settings.VISION_MAX_IMAGES_PER_DOCUMENT
    if limit is not None and limit > 0:
        page_texts = page_texts[:limit]
    if max_pages is not None and max_pages > 0:
        page_texts = page_texts[:max_pages]

    results: List[Tuple[int, str]] = []
    total = len(page_texts)
    for idx, (page_no, pymupdf_text) in enumerate(page_texts):
        page_index = page_no - 1 if page_no > 0 else idx
        ld.info(
            "[Multimodal] page %s/%s (page_no=%s)",
            idx + 1,
            total,
            page_no,
        )
        t0 = time.perf_counter()
        png = render_pdf_page_png(pdf_path, page_index)
        synthesis = synthesize_page_with_mistral_small(
            png, page_no, pymupdf_text, document_title
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            "multimodal page_no=%s synthèse %.2fs (%d chars)",
            page_no,
            elapsed,
            len(synthesis),
        )
        header = f"# Page {page_no}"
        if document_title:
            header += f" — {document_title}"
        full_content = f"{header}\n\n{synthesis}"
        results.append((page_no, full_content))
    return results


def embed_new_multimodal_chunks(document_id: int) -> int:
    """Embeddings mistral-embed pour les chunks multimodal sans vecteur."""
    from app.services.embedding_service import generate_embeddings_batch

    with Session(engine) as session:
        statement = select(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.embedding.is_(None),
            DocumentChunk.metadata_json["content_type"].as_string()
            == PAGE_MULTIMODAL_CONTENT_TYPE,
        )
        chunks = list(session.exec(statement).all())
        if not chunks:
            return 0

        batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
        model_name = settings.EMBEDDING_MODEL
        ok = 0
        for i in range(0, len(chunks), batch_size):
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
                    ok += 1
            session.add_all(batch)
            session.commit()
        return ok
