"""
Extraction markdown via API Mistral OCR (remplace Docling pour l'ingestion).
PDF / images : OCR Mistral ; autres formats bureautique : LibreOffice → PDF puis OCR.
"""
from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".xml", ".html", ".htm"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".tiff", ".tif", ".bmp"}
_PDF_EXTENSIONS = {".pdf"}
_OFFICE_EXTENSIONS = {
    ".doc",
    ".docx",
    ".odt",
    ".ppt",
    ".pptx",
    ".odp",
    ".xls",
    ".xlsx",
    ".ods",
    ".rtf",
}


def _read_text_file(file_path: str) -> str:
    path = Path(file_path)
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _file_to_base64_data_url(file_path: str, mime: str) -> str:
    raw = Path(file_path).read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _ocr_document_via_api(
    *,
    document_payload: dict,
    pages: Optional[List[int]] = None,
    max_retries: int = 3,
) -> str:
    """Appelle POST /v1/ocr et retourne le markdown agrégé."""
    import httpx

    api_key = settings.MISTRAL_API_KEY
    if not api_key:
        raise ValueError("MISTRAL_API_KEY n'est pas configurée pour l'OCR")

    model = getattr(settings, "MISTRAL_OCR_MODEL", None) or "mistral-ocr-latest"
    base_url = (settings.MISTRAL_BASE_URL or "https://api.mistral.ai").rstrip("/")
    timeout = float(getattr(settings, "MISTRAL_OCR_TIMEOUT", 300) or 300)

    body: dict = {
        "model": model,
        "document": document_payload,
    }
    if pages is not None:
        body["pages"] = pages

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    f"{base_url}/v1/ocr",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
            return _markdown_from_ocr_response(data)
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "Mistral OCR tentative %s/%s échouée (%s), nouvel essai dans %ss",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                break
    raise RuntimeError(f"Mistral OCR échoué après {max_retries} tentatives: {last_err}") from last_err


def _markdown_from_ocr_response(data: dict) -> str:
    """Extrait le markdown des pages renvoyées par l'API OCR."""
    pages = data.get("pages") or []
    parts: List[str] = []
    for idx, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        md = (page.get("markdown") or page.get("text") or "").strip()
        if md:
            page_num = page.get("index")
            if page_num is None:
                page_num = page.get("page")
            if page_num is None:
                page_num = idx + 1
            try:
                page_num = int(page_num)
            except (TypeError, ValueError):
                page_num = idx + 1
            if page_num > 0:
                parts.append(f"<!-- page:{page_num} -->\n\n{md}")
            else:
                parts.append(md)
    if parts:
        aggregated = "\n\n".join(parts).strip()
        from app.services.chunking_service import normalize_markdown_tables

        return normalize_markdown_tables(aggregated)

    # Fallback : certains payloads exposent un champ racine
    for key in ("markdown", "text", "content"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            from app.services.chunking_service import normalize_markdown_tables

            return normalize_markdown_tables(val.strip())
    return ""


def _ocr_pdf_page_by_page(file_path: str) -> str:
    """Convertit le PDF en images et OCR page par page (page_no dans le markdown)."""
    from pdf2image import convert_from_path

    images = convert_from_path(file_path, dpi=200)
    if not images:
        return ""

    sections: List[str] = []
    for page_idx, image in enumerate(images):
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        payload = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }
        page_md = _ocr_document_via_api(document_payload=payload)
        if page_md:
            if "<!-- page:" not in page_md[:80].lower():
                page_md = f"<!-- page:{page_idx + 1} -->\n\n{page_md}"
            sections.append(page_md)
    result = "\n\n".join(sections).strip()
    if result:
        from app.services.chunking_service import normalize_markdown_tables

        return normalize_markdown_tables(result)
    return result


def _ocr_file(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in _TEXT_EXTENSIONS:
        return _read_text_file(file_path).strip()

    if suffix in _IMAGE_EXTENSIONS:
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".bmp": "image/bmp",
        }.get(suffix, "image/png")
        payload = {
            "type": "image_url",
            "image_url": {"url": _file_to_base64_data_url(file_path, mime)},
        }
        return _ocr_document_via_api(document_payload=payload)

    if suffix in _PDF_EXTENSIONS:
        # Déléguer au service hybride : pymupdf4llm (natif) → Mistral OCR (fallback)
        from app.services.pdf_extraction_service import extract_markdown_from_pdf
        md, method = extract_markdown_from_pdf(file_path)
        return md

    raise ValueError(f"Format non supporté pour Mistral OCR: {suffix}")


def extract_markdown_from_file(file_path: str) -> str:
    """
    Extrait le contenu markdown d'un fichier.
    - PDF : pymupdf4llm (natif, rapide) → fallback Mistral OCR si scanné
    - Images : Mistral OCR direct
    - Office (docx, pptx…) : LibreOffice → PDF → pipeline PDF ci-dessus
    - Texte brut : lecture directe
    """
    from app.services.file_conversion import ensure_pdf_for_ocr

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Fichier introuvable: {file_path}")

    suffix = Path(file_path).suffix.lower()
    ocr_path = file_path

    if suffix in _OFFICE_EXTENSIONS or suffix == ".epub":
        ocr_path = ensure_pdf_for_ocr(file_path)
    elif suffix not in (_TEXT_EXTENSIONS | _IMAGE_EXTENSIONS | _PDF_EXTENSIONS):
        try:
            ocr_path = ensure_pdf_for_ocr(file_path)
        except Exception:
            logger.warning(
                "Conversion PDF impossible pour %s, tentative OCR direct",
                file_path,
            )

    t0 = time.perf_counter()
    markdown = _ocr_file(ocr_path)
    logger.info(
        "Mistral OCR terminé pour %s en %.2fs (%d caractères)",
        file_path,
        time.perf_counter() - t0,
        len(markdown),
    )
    return markdown
