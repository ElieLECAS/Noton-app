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
DEFAULT_RAG_MAX_PAGE_IMAGES = settings.RAG_MAX_IMAGES


def is_vision_model(model_name: str) -> bool:
    """Indique si le modèle accepte des images en entrée."""
    name_lower = (model_name or "").lower()
    return (
        "pixtral" in name_lower
        or "vision" in name_lower
        or "large-latest" in name_lower
        or "ministral" in name_lower
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
    needs_image_only: bool = False,
) -> List[Tuple[int, int]]:
    """
    Retourne les couples (document_id, page_no) uniques, ordre de pertinence.
    Si needs_image_only=True, ne retient que les pages marquées needs_page_image / image_pages.
    """
    unique_pages: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()

    if needs_image_only:
        for passage in passages:
            for doc_id, page_no in passage.get("image_pages") or []:
                try:
                    key = (int(doc_id), int(page_no))
                except (TypeError, ValueError):
                    continue
                if key in seen:
                    continue
                seen.add(key)
                unique_pages.append(key)
                if max_pages is not None and len(unique_pages) >= max_pages:
                    return unique_pages
        return unique_pages

    for passage in passages:
        doc_id = passage.get("document_id")
        page_start = passage.get("page_start") or passage.get("page_no")
        page_end = passage.get("page_end") or page_start
        if doc_id is None or page_start is None:
            continue
        try:
            doc_id_i = int(doc_id)
            start_i = int(page_start)
            end_i = int(page_end) if page_end is not None else start_i
        except (TypeError, ValueError):
            continue
        for page_no in range(start_i, end_i + 1):
            key = (doc_id_i, page_no)
            if key in seen:
                continue
            seen.add(key)
            unique_pages.append(key)
            if max_pages is not None and len(unique_pages) >= max_pages:
                return unique_pages
    return unique_pages


def build_page_passage_from_l1_chunks(
    session: Session,
    document_id: int,
    page_nos: List[int],
) -> str:
    """Concatène les chunks L1 vision pour une ou plusieurs pages (small-to-big)."""
    from app.services.page_retrieval_service import (
        build_consolidated_page_text,
        load_l1_chunks_for_page,
    )

    all_chunks = []
    seen_ids: set[int] = set()
    for page_no in sorted(page_nos):
        for chunk in load_l1_chunks_for_page(session, document_id, page_no):
            if chunk.id in seen_ids:
                continue
            seen_ids.add(chunk.id)
            all_chunks.append(chunk)
    all_chunks.sort(key=lambda c: (c.chunk_index or 0, c.id or 0))
    return build_consolidated_page_text(all_chunks)


def render_page_images_for_passages(
    session: Session,
    passages: List[Dict[str, Any]],
    *,
    max_pages: int = DEFAULT_RAG_MAX_PAGE_IMAGES,
    dpi: int = DEFAULT_RAG_PAGE_IMAGE_DPI,
    needs_image_only: bool = True,
) -> List[str]:
    """Rend les pages candidates en PNG base64 (ordre de pertinence des passages)."""
    from app.services.multimodal_page_service import render_page_png_cached

    max_pages = max_pages or settings.GENERATION_MAX_PAGE_IMAGES
    images_b64: List[str] = []
    for doc_id, page_no in collect_unique_page_keys(
        passages, max_pages=max_pages, needs_image_only=needs_image_only
    ):
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
    needs_image_only: bool = True,
) -> List[str]:
    """Version async (thread pool) du rendu PNG."""
    return await asyncio.to_thread(
        render_page_images_for_passages,
        session,
        passages,
        max_pages=max_pages,
        dpi=dpi,
        needs_image_only=needs_image_only,
    )


def build_rag_user_message(
    question: str,
    *,
    images_b64: Optional[List[str]] = None,
    image_captions: Optional[List[Dict[str, Any]]] = None,
    task_reminder: Optional[str] = None,
) -> Dict[str, Any]:
    """Construit le message utilisateur avec images optionnelles.

    ``image_captions`` relie chaque PNG joint à son bloc texte du contexte CAG
    (Image k = DOCUMENT i, page p) — sans ces légendes le modèle ne sait pas quelle
    image correspond à quel document. ``task_reminder`` est le rappel final de tâche
    placé APRÈS tout le contexte (anti « lost in the middle » sur ~100k tokens).
    """
    content = question
    if image_captions:
        lines = [
            f"Image {c.get('image_index')} = DOCUMENT {c.get('document_index')} "
            f"« {c.get('document_title') or 'Sans titre'} », page {c.get('page_no')}"
            for c in image_captions
        ]
        content += "\n\n[Correspondance des images jointes]\n" + "\n".join(lines)
    if task_reminder:
        content += "\n\n" + task_reminder
    user_msg: Dict[str, Any] = {"role": "user", "content": content}
    if images_b64:
        user_msg["images"] = images_b64
    return user_msg


def build_cag_task_reminder(
    original_message: str,
    *,
    standalone_question: Optional[str] = None,
) -> str:
    """Rappel final de tâche, placé en fin de message user (donc en tout dernier dans le
    contexte) : après ~100k tokens de documents, c'est la consigne que le modèle suit le
    plus fidèlement."""
    bits: List[str] = ["RAPPEL FINAL — Réponds UNIQUEMENT à ma question ci-dessus."]
    sq = (standalone_question or "").strip()
    if sq and sq.lower() != (original_message or "").strip().lower():
        bits.append(f"Question autonome reformulée : « {sq} ».")
    bits.append(
        "Si la réponse dépend d'un produit, d'une gamme ou d'une version que je n'ai pas "
        "précisés et que les documents en couvrent plusieurs, demande-moi d'abord lequel "
        "(ou présente brièvement les cas) au lieu de choisir à ma place."
    )
    return " ".join(bits)


async def build_rag_generation_messages(
    session: Session,
    passages: List[Dict[str, Any]],
    question: str,
    *,
    model: Optional[str] = None,
    max_page_images: Optional[int] = None,
    page_image_dpi: int = DEFAULT_RAG_PAGE_IMAGE_DPI,
) -> List[Dict[str, Any]]:
    """
    Pipeline unifié chat + éval : passages L1 vision consolidés, PNG uniquement si ColPali seul.
    """
    from app.routers.chat import build_space_context_from_passages

    model_name = model or settings.MODEL_FAST
    max_images = max_page_images if max_page_images is not None else settings.RAG_MAX_IMAGES

    # Fallback pymupdf uniquement pour placeholders ColPali legacy sans texte L1
    enriched = enrich_colpali_passages_with_pymupdf(session, list(passages))
    space_context = build_space_context_from_passages(enriched)

    images_b64: List[str] = []
    render_all = settings.RAG_RENDER_ALL_IMAGES
    any_needs_image = any(p.get("needs_page_image") for p in enriched) or render_all
    if is_vision_model(model_name) and any_needs_image:
        images_b64 = await render_page_images_for_passages_async(
            session,
            enriched,
            max_pages=max_images,
            dpi=page_image_dpi,
            needs_image_only=not render_all,
        )
        logger.info(
            "build_rag_generation_messages: %d image(s) PNG pour modèle %s (render_all=%s)",
            len(images_b64),
            model_name,
            render_all,
        )
    elif any_needs_image and not is_vision_model(model_name):
        logger.warning(
            "build_rag_generation_messages: pages ColPali-only détectées mais modèle non vision (%s)",
            model_name,
        )
    else:
        logger.info(
            "build_rag_generation_messages: pas d'images (accord texte+visuel ou pas de ColPali seul)",
        )

    user_msg = build_rag_user_message(question, images_b64=images_b64 or None)
    return [space_context, user_msg]
