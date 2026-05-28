"""
Extraction PDF hybride : markdown natif (pymupdf4llm) puis fallback OCR (Mistral).

Stratégie :
- PDF texte (extractible) → pymupdf4llm → markdown structuré (headers, tables, listes)
- PDF scanné ou échec → Mistral OCR (fallback)

Le markdown produit est directement compatible MarkdownNodeParser de LlamaIndex.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

MIN_CHARS_THRESHOLD = 100  # Seuil minimal de texte pour considérer un PDF comme textuel


def has_extractable_text(pdf_path: str, min_chars: int = MIN_CHARS_THRESHOLD) -> bool:
    """
    Détecte rapidement si le PDF contient du texte extractible (vs scanné).
    Vérifie uniquement les 3 premières pages pour la rapidité.
    """
    try:
        import fitz  # PyMuPDF (inclus dans pymupdf4llm)

        doc = fitz.open(pdf_path)
        total_text = ""
        for page in doc[: min(3, len(doc))]:
            total_text += page.get_text()
            if len(total_text) > min_chars:
                doc.close()
                return True
        doc.close()
        return len(total_text.strip()) > min_chars
    except Exception as exc:
        logger.debug("Détection texte PDF échouée pour %s: %s", pdf_path, exc)
        return False


def _normalize_page_markers(markdown: str) -> str:
    """
    Normalise les marqueurs de page pymupdf4llm vers le format <!-- page:N -->
    utilisé dans le reste du pipeline.

    pymupdf4llm produit : "-----\n\n[page N]\n\n-----" ou variantes.
    On convertit en : "<!-- page:N -->"
    """
    # Format "-----\n\n[page N]\n\n-----" produit par pymupdf4llm
    markdown = re.sub(
        r"-{4,}\s*\[page\s+(\d+)\]\s*-{4,}",
        r"<!-- page:\1 -->",
        markdown,
        flags=re.IGNORECASE,
    )
    # Format simplifié "[page N]" seul sur sa ligne
    markdown = re.sub(
        r"^\[page\s+(\d+)\]\s*$",
        r"<!-- page:\1 -->",
        markdown,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return markdown


def extract_markdown_pymupdf4llm(pdf_path: str) -> str:
    """
    Extraction markdown structuré via pymupdf4llm.

    Produit du markdown optimisé pour LlamaIndex/MarkdownNodeParser :
    - Headers automatiques (# ## ###) détectés via font size
    - Tables en markdown pipes
    - Listes formatées (- item)
    - Multi-colonnes gérées (ordre de lecture reconstruit)
    - Compatible direct avec chunk_markdown_structured()
    """
    import pymupdf4llm

    chunks = pymupdf4llm.to_markdown(
        pdf_path,
        page_chunks=True,    # Récupère une liste de dictionnaires par page
        write_images=False,  # Pas d'export images (RAG texte seul)
        show_progress=False,
    )

    if isinstance(chunks, list):
        pages_md = []
        for i, chunk in enumerate(chunks):
            page_text = ""
            page_num = i + 1
            if isinstance(chunk, dict):
                page_text = chunk.get("text", "") or ""
                meta = chunk.get("metadata")
                if isinstance(meta, dict):
                    # page_number est 1-based d'après pymupdf4llm
                    p_num = meta.get("page_number") or meta.get("page")
                    if p_num is not None:
                        try:
                            page_num = int(p_num)
                        except (TypeError, ValueError):
                            pass
            elif isinstance(chunk, str):
                page_text = chunk
            
            # Injection explicite du marqueur standardisé
            pages_md.append(f"<!-- page:{page_num} -->\n\n{page_text}")
        
        markdown = "\n\n".join(pages_md)
    elif isinstance(chunks, str):
        markdown = chunks
    else:
        markdown = ""

    markdown = _normalize_page_markers(markdown)
    return markdown.strip()


def extract_page_texts_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extrait le markdown pymupdf4llm page par page.

    Returns:
        Liste de (page_no 1-based, markdown_page).
    """
    import pymupdf4llm

    chunks = pymupdf4llm.to_markdown(
        pdf_path,
        page_chunks=True,
        write_images=False,
        show_progress=False,
    )
    pages: List[Tuple[int, str]] = []
    if not isinstance(chunks, list):
        return pages

    for i, chunk in enumerate(chunks):
        page_text = ""
        page_num = i + 1
        if isinstance(chunk, dict):
            page_text = (chunk.get("text") or "").strip()
            meta = chunk.get("metadata")
            if isinstance(meta, dict):
                p_num = meta.get("page_number") or meta.get("page")
                if p_num is not None:
                    try:
                        page_num = int(p_num)
                    except (TypeError, ValueError):
                        pass
        elif isinstance(chunk, str):
            page_text = chunk.strip()
        pages.append((page_num, page_text))
    return pages


def extract_markdown_from_pdf(pdf_path: str) -> Tuple[str, str]:
    """
    Extraction PDF : OCR Mistral page par page.

    Args:
        pdf_path: Chemin absolu ou relatif vers le fichier PDF.

    Returns:
        (markdown, method_used)
        - method_used: "ocr"
    """
    logger.info("Extraction OCR Mistral démarrée pour %s", Path(pdf_path).name)
    from app.services.mistral_ocr_service import _ocr_pdf_page_by_page

    markdown = _ocr_pdf_page_by_page(pdf_path)
    return markdown, "ocr"
