"""Conversion de fichiers bureautique vers PDF (LibreOffice headless)."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_CONVERTIBLE_EXTS = {
    ".odt",
    ".odm",
    ".odg",
    ".odp",
    ".ods",
    ".odf",
    ".doc",
    ".docx",
    ".rtf",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".epub",
}


def ensure_pdf_for_ocr(file_path: str) -> str:
    """
    Retourne un chemin PDF utilisable par Mistral OCR.
    Les formats bureautique sont convertis via LibreOffice.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return file_path

    if suffix not in _CONVERTIBLE_EXTS:
        return file_path

    pdf_path = Path(file_path).with_suffix(".pdf")
    try:
        if pdf_path.exists():
            pdf_path.unlink()
    except Exception:
        pass

    logger.info("Conversion LibreOffice vers PDF: %s", file_path)
    cmd = [
        "libreoffice",
        "--headless",
        "--nologo",
        "--nolockcheck",
        "--convert-to",
        "pdf",
        "--outdir",
        str(pdf_path.parent),
        str(file_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-2000:]
        stdout_tail = (proc.stdout or "").strip()[-2000:]
        details = stderr_tail or stdout_tail or f"code={proc.returncode}"
        raise RuntimeError(f"LibreOffice a échoué pour {file_path}: {details}")

    if not pdf_path.exists():
        candidates = sorted(
            pdf_path.parent.glob(f"{Path(file_path).stem}*.pdf"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"Aucun PDF généré pour {file_path}")
        pdf_path = candidates[0]

    return str(pdf_path)
