"""
Construction du contexte de génération RAG (chat espace + évaluation admin).

Enrichit les passages ColPali (pymupdf), attache les PNG des pages rerankées
au message utilisateur si le modèle est vision-capable.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from sqlmodel import Session

from app.config import settings
from app.models.document import Document

logger = logging.getLogger(__name__)

COLPALI_PLACEHOLDER_MARKER = "[ColPali Indexed Page"
DEFAULT_RAG_PAGE_IMAGE_DPI = 150
DEFAULT_RAG_MAX_PAGE_IMAGES = 3


def is_vision_model(model_name: str) -> bool:
    """Indique si le modèle accepte des images en entrée."""
    name_lower = (model_name or "").lower()
    return (
        "pixtral" in name_lower
        or "vision" in name_lower
        or "large-latest" in name_lower
        or "gpt-4o" in name_lower
    )


def _is_colpali_placeholder(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return True
    if COLPALI_PLACEHOLDER_MARKER in text:
        return True
    if text.startswith("[Page ") and "contenu visuel uniquement" in text:
        return True
    return False


def extract_page_text_from_pdf(pdf_path: str, page_no: int) -> str:
    """Extrait le texte natif pymupdf d'une page (1-based)."""
    import fitz

    with fitz.open(pdf_path) as pdf:
        if page_no < 1 or page_no > len(pdf):
            return ""
        return pdf[page_no - 1].get_text("text").strip()


def enrich_colpali_passages_with_pymupdf(
    session: Session,
    passages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remplace les placeholders ColPali par le texte pymupdf de la page."""
    for passage in passages:
        raw = passage.get("passage_raw") or ""
        body = passage.get("passage") or ""
        if not _is_colpali_placeholder(raw or body):
            continue

        doc_id = passage.get("document_id")
        page_no = passage.get("page_no") or passage.get("page_start")
        if doc_id is None or page_no is None:
            continue

        try:
            page_no_int = int(page_no)
        except (TypeError, ValueError):
            continue

        doc = session.get(Document, doc_id)
        if not doc or not doc.source_file_path or not os.path.exists(doc.source_file_path):
            continue

        try:
            text = extract_page_text_from_pdf(doc.source_file_path, page_no_int)
        except Exception as exc:
            logger.warning(
                "Enrichissement pymupdf échoué (doc=%s, page=%s): %s",
                doc_id,
                page_no_int,
                exc,
            )
            continue

        if not text:
            continue

        title = passage.get("document_title") or "Document sans titre"
        passage["passage_raw"] = text
        passage["passage"] = f"**{title}**\n[Page {page_no_int}]\n{text}"

    return passages


def collect_unique_page_keys(
    passages: List[Dict[str, Any]],
    *,
    max_pages: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Retourne les couples (document_id, page_no) uniques, ordre de pertinence."""
    unique_pages: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for passage in passages:
        doc_id = passage.get("document_id")
        page_no = passage.get("page_no") or passage.get("page_start")
        if doc_id is None or page_no is None:
            continue
        try:
            key = (int(doc_id), int(page_no))
        except (TypeError, ValueError):
            continue
        if key in seen:
            continue
        seen.add(key)
        unique_pages.append(key)
        if max_pages is not None and len(unique_pages) >= max_pages:
            break
    return unique_pages


def render_page_images_for_passages(
    session: Session,
    passages: List[Dict[str, Any]],
    *,
    max_pages: int = DEFAULT_RAG_MAX_PAGE_IMAGES,
    dpi: int = DEFAULT_RAG_PAGE_IMAGE_DPI,
) -> List[str]:
    """Rend les pages candidates en PNG base64 (ordre de pertinence des passages)."""
    from app.services.multimodal_page_service import render_page_png_cached

    images_b64: List[str] = []
    for doc_id, page_no in collect_unique_page_keys(passages, max_pages=max_pages):
        doc = session.get(Document, doc_id)
        if not doc or not doc.source_file_path or not os.path.exists(doc.source_file_path):
            logger.warning(
                "Rendu PNG ignoré : document %s introuvable (page %s)",
                doc_id,
                page_no,
            )
            continue
        try:
            png_bytes = render_page_png_cached(doc.source_file_path, page_no, dpi=dpi)
            images_b64.append(base64.b64encode(png_bytes).decode("utf-8"))
        except Exception as exc:
            logger.error(
                "Erreur rendu PNG page %s (document %s): %s",
                page_no,
                doc_id,
                exc,
                exc_info=True,
            )
    return images_b64


async def render_page_images_for_passages_async(
    session: Session,
    passages: List[Dict[str, Any]],
    *,
    max_pages: int = DEFAULT_RAG_MAX_PAGE_IMAGES,
    dpi: int = DEFAULT_RAG_PAGE_IMAGE_DPI,
) -> List[str]:
    """Version async (thread pool) du rendu PNG."""
    return await asyncio.to_thread(
        render_page_images_for_passages,
        session,
        passages,
        max_pages=max_pages,
        dpi=dpi,
    )


def build_rag_user_message(
    question: str,
    *,
    images_b64: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Construit le message utilisateur avec images optionnelles."""
    user_msg: Dict[str, Any] = {"role": "user", "content": question}
    if images_b64:
        user_msg["images"] = images_b64
    return user_msg


async def build_rag_generation_messages(
    session: Session,
    passages: List[Dict[str, Any]],
    question: str,
    *,
    model: Optional[str] = None,
    max_page_images: int = DEFAULT_RAG_MAX_PAGE_IMAGES,
    page_image_dpi: int = DEFAULT_RAG_PAGE_IMAGE_DPI,
) -> List[Dict[str, Any]]:
    """
    Pipeline unifié chat + éval : enrichissement pymupdf, system prompt, images PNG.
    """
    from app.routers.chat import build_space_context_from_passages

    model_name = model or settings.MODEL_FAST
    enriched = enrich_colpali_passages_with_pymupdf(session, list(passages))
    space_context = build_space_context_from_passages(enriched)

    images_b64: List[str] = []
    if is_vision_model(model_name):
        images_b64 = await render_page_images_for_passages_async(
            session,
            enriched,
            max_pages=max_page_images,
            dpi=page_image_dpi,
        )
        logger.info(
            "build_rag_generation_messages: %d image(s) PNG pour modèle %s",
            len(images_b64),
            model_name,
        )
    else:
        logger.info(
            "build_rag_generation_messages: pas d'images (modèle non vision: %s)",
            model_name,
        )

    user_msg = build_rag_user_message(question, images_b64=images_b64 or None)
    return [space_context, user_msg]
