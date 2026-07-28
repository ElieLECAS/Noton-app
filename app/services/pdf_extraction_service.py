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
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

MIN_CHARS_THRESHOLD = 100  # Seuil minimal de texte pour considérer un PDF comme textuel

# Marqueurs et artefacts pymupdf4llm à nettoyer
_PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*\d+\s*-->", re.IGNORECASE)
_PICTURE_OMITTED_RE = re.compile(
    r"\*{0,2}\s*==>\s*picture\s*\[[^\]]*\]\s*intentionally omitted\s*<==\s*\*{0,2}",
    re.IGNORECASE,
)
# Bloc « texte d'image » de pymupdf4llm. DEUX formes existent selon la version :
#   --- Start of picture text ---      (ancienne, à tirets)
#   <!-- Start of picture text -->     (actuelle, commentaire HTML)
# Le motif d'origine n'acceptait que la première (`-{3,}`) : `<!--` ne porte que DEUX
# tirets, donc le nettoyage ne se déclenchait PLUS du tout. Conséquence observée le
# 2026-07-28 sur une plaquette : les puces de « Les avantages » et deux sections
# entières concaténées en un chunk de 450 tokens, marqueurs bruts inclus.
_PICTURE_TEXT_BLOCK_RE = re.compile(
    r"\*{0,2}\s*(?:-{3,}|<!--)\s*Start of picture text\s*(?:-{3,}|-->)\s*\*{0,2}\s*"
    r"(?:<br\s*/?>\s*)*"
    r"(.*?)"
    r"(?:<br\s*/?>\s*)*"
    r"\*{0,2}\s*(?:-{3,}|<!--)\s*End of picture text\s*(?:-{3,}|-->)\s*\*{0,2}",
    re.DOTALL | re.IGNORECASE,
)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_BOLD_WRAP_RE = re.compile(r"\*\*(.+?)\*\*")


def _unwrap_bold_line(line: str) -> str:
    """Retire le gras markdown récursif sur une ligne."""
    s = (line or "").strip()
    for _ in range(5):
        if s.startswith("**") and s.endswith("**") and len(s) > 4:
            s = s[2:-2].strip()
        else:
            break
    return s


# Glyphes de puce rencontrés dans les PDF (carrés, ronds, tirets typographiques) :
# pymupdf4llm les rend tels quels, ils doivent devenir des puces markdown.
_BULLET_GLYPHS = "•▪▫◦‣·−–—□■☐✓✔»›"
_BULLET_PREFIX_RE = re.compile(rf"^[{re.escape(_BULLET_GLYPHS)}]\s*")

# Un titre de section d'une plaquette est typiquement une ligne ENTIÈREMENT en
# capitales, courte, sans ponctuation finale (« UN DESIGN EXCLUSIF », « LES OUVERTURES »).
# Sans promotion, ces titres restent noyés dans la liste et deux sections distinctes
# finissent dans le même chunk.
_ALLCAPS_TITLE_RE = re.compile(r"^[^a-z]{3,60}$")

# Mobilier de page à la forme NON ambiguë : « Page 3 sur 53 », « 3/53 ».
# Un nombre nu n'est PAS listé ici : « 9016 » (RAL), « 300 » (code CSTB) ou « 1500 »
# (cote) ont exactement la même forme qu'un folio. Le nombre nu n'est retiré que s'il
# ÉGALE le numéro de page — voir le paramètre page_no de clean_pymupdf4llm_markdown.
_PAGE_FURNITURE_RE = re.compile(
    r"^(?:[Pp]age\s+\d+\s*(?:sur|/)\s*\d+|\d{1,4}\s*/\s*\d{1,4})$"
)


def _is_page_folio(line: str, page_no: Optional[int]) -> bool:
    """Nombre nu identique au numéro de page (« 4 », « .4 », « 4. »)."""
    if page_no is None:
        return False
    stripped = line.strip().strip(".").strip()
    return stripped.isdigit() and int(stripped) == page_no


def _looks_like_section_title(line: str) -> bool:
    stripped = line.strip().rstrip(":").strip()
    if not (3 <= len(stripped) <= 60):
        return False
    if stripped.endswith((".", ";", ",")):
        return False
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) < 3:
        return False
    if not all(c.isupper() for c in letters):
        return False
    # Un simple sigle isolé (« PVC ») n'est pas un titre de section.
    return len(stripped.split()) >= 2


def _picture_text_to_list(match: re.Match) -> str:
    """Convertit un bloc « picture text » pymupdf4llm en markdown structuré.

    Les lignes tout en capitales deviennent des TITRES (####) pour que la découpe par
    section les voie ; les autres deviennent des puces. Sans ça, un encart de plaquette
    mêlant une liste d'avantages et deux titres de section produisait un seul bloc.
    """
    inner = _BR_RE.sub("\n", match.group(1) or "")
    out: List[str] = []
    for raw in inner.splitlines():
        line = _unwrap_bold_line(raw)
        line = _BULLET_PREFIX_RE.sub("", line).strip()
        if not line:
            continue
        if _looks_like_section_title(line):
            out.append(f"#### {line}")
        elif line.startswith(("-", "#")):
            out.append(line)
        else:
            out.append(f"- {line}")
    if not out:
        return ""
    # Sauts de ligne encadrants OBLIGATOIRES : le `\s*` du motif consomme les retours
    # à la ligne autour du bloc, ce qui collerait la première puce à la ligne
    # précédente et la ligne suivante à la dernière puce.
    return "\n" + "\n".join(out) + "\n"


def clean_pymupdf4llm_markdown(text: str, page_no: Optional[int] = None) -> str:
    """
    Nettoie le markdown brut pymupdf4llm pour stockage et embedding.

    - Supprime les marqueurs <!-- page:N --> et le mobilier de page (folios)
    - Supprime les placeholders d'images omises
    - Convertit les blocs « picture text » en titres + listes à puces
    - Normalise les glyphes de puce (•, ▪, ☐, –) en puces markdown
    - Convertit <br> en retours à la ligne
    - Normalise le gras excessif (**...** sur chaque ligne)
    - Compresse les lignes vides multiples
    """
    if not text or not text.strip():
        return ""

    t = text
    t = _PAGE_MARKER_RE.sub("", t)
    t = _PICTURE_TEXT_BLOCK_RE.sub(_picture_text_to_list, t)
    t = _PICTURE_OMITTED_RE.sub("", t)
    t = _BR_RE.sub("\n", t)

    cleaned_lines: List[str] = []
    for line in t.splitlines():
        stripped = _unwrap_bold_line(line)
        # Mobilier de page : présent sur CHAQUE page, il pollue tous les chunks sans
        # rien apporter. Le nombre nu n'est retiré que s'il ÉGALE le numéro de page —
        # sinon on supprimerait les RAL (9016), codes CSTB (300) et cotes (1500).
        if _PAGE_FURNITURE_RE.match(stripped.strip()) or _is_page_folio(stripped, page_no):
            continue
        # Puce à glyphe → puce markdown (hors tableaux, dont les lignes commencent par |).
        if not stripped.lstrip().startswith("|"):
            bullet = _BULLET_PREFIX_RE.match(stripped.lstrip())
            if bullet:
                stripped = "- " + stripped.lstrip()[bullet.end():].strip()
        cleaned_lines.append(stripped)

    t = "\n".join(cleaned_lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def extract_first_heading(markdown: str) -> str:
    """Retourne le premier titre ## ou la première ligne non vide."""
    cleaned = clean_pymupdf4llm_markdown(markdown)
    if not cleaned:
        return ""
    m = re.search(r"(?m)^##\s+(.+)$", cleaned)
    if m:
        return _unwrap_bold_line(m.group(1))
    for line in cleaned.splitlines():
        line = line.strip()
        if line:
            return _unwrap_bold_line(line)
    return ""


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
        page_text = clean_pymupdf4llm_markdown(page_text)
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
