"""
Retraitement multimodal v4 : pymupdf + mistral-small vision (raw) + Pass 2 texte (rapports fenêtre).

- Pass 1 (vision/page) : raw_text uniquement (native: pymupdf canonique + [Image:…] ; scanned: OCR vision).
- Pass 2 (texte/fenêtre) : page_window_report explicites multi-pages (≤480 tokens via split_text_rag_friendly).
- Tous les chunks indexés sont des leaves (is_leaf=True), liés par window_id.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
from sqlalchemy import delete, or_
from sqlmodel import Session, select

from app.config import settings
from app.database import engine
from app.library_document_logging import get_library_document_logger
from app.models.document import Document
from app.models.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)

PAGE_RAW_ENRICHED_CONTENT_TYPE = "page_raw_enriched"
PAGE_WINDOW_REPORT_CONTENT_TYPE = "page_window_report"
# Legacy v3 (suppression au reindex, lecture retrieval)
PAGE_SECTION_REPORT_CONTENT_TYPE = "page_section_report"
PAGE_GROUP_SUMMARY_CONTENT_TYPE = "page_group_summary"
PAGE_MULTIMODAL_SECTION_CONTENT_TYPE = "page_multimodal_section"
PAGE_MULTIMODAL_SUMMARY_CONTENT_TYPE = "page_multimodal_summary"
CHUNKING_VERSION_MULTIMODAL = "multimodal_page_v4"
MAX_REPORTS_PER_WINDOW = 10
MAX_CHUNK_CHARS = 1500  # Limite caractères (fallback si token count échoue)
MAX_CHUNK_TOKENS = 450  # Limite stricte en tokens (marge plus conservative vs 512)

MULTIMODAL_CONTENT_TYPES = (
    PAGE_RAW_ENRICHED_CONTENT_TYPE,
    PAGE_WINDOW_REPORT_CONTENT_TYPE,
)
LEGACY_V3_MULTIMODAL_CONTENT_TYPES = (
    PAGE_SECTION_REPORT_CONTENT_TYPE,
    PAGE_GROUP_SUMMARY_CONTENT_TYPE,
)
LEGACY_MULTIMODAL_CONTENT_TYPES = (
    PAGE_MULTIMODAL_SECTION_CONTENT_TYPE,
    PAGE_MULTIMODAL_SUMMARY_CONTENT_TYPE,
)
ALL_MULTIMODAL_CONTENT_TYPES = (
    MULTIMODAL_CONTENT_TYPES
    + LEGACY_V3_MULTIMODAL_CONTENT_TYPES
    + LEGACY_MULTIMODAL_CONTENT_TYPES
)
EMBEDDABLE_MULTIMODAL_CONTENT_TYPES = MULTIMODAL_CONTENT_TYPES

_IMAGE_BLOCK_RE = re.compile(r"\[Image:\s*[^\]]*\]", re.IGNORECASE)
_DEICTIC_RE = re.compile(
    r"\b(ce profil|ce produit|cette section|ci-dessus|ci-dessous|celui-ci|celle-ci)\b",
    re.IGNORECASE,
)

# Tokenizer lazy-loaded pour comptage tokens
_tokenizer = None
_tokenizer_lock = None


def _get_tokenizer():
    """Charge le tokenizer BERT ou Mistral pour comptage tokens."""
    global _tokenizer, _tokenizer_lock
    if _tokenizer is not None:
        return _tokenizer
    
    import threading
    if _tokenizer_lock is None:
        _tokenizer_lock = threading.Lock()
    
    with _tokenizer_lock:
        if _tokenizer is not None:
            return _tokenizer
        try:
            from transformers import AutoTokenizer
            if getattr(settings, "RERANKER_PROVIDER", "local") == "mistral":
                model_name = "mistralai/Mistral-7B-v0.1"
            else:
                model_name = settings.RERANKER_MODEL or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Tokenizer chargé pour comptage tokens : %s", model_name)
        except Exception as e:
            logger.warning("Échec chargement tokenizer, fallback approximation : %s", e)
            _tokenizer = None
    return _tokenizer


def count_tokens(text: str) -> int:
    """
    Compte le nombre de tokens dans un texte avec le tokenizer approprié.
    Fallback : approximation 3.5 chars/token si tokenizer indisponible.
    """
    if not text:
        return 0
    
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        try:
            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=False)
            return len(tokens)
        except Exception as e:
            logger.warning("Échec comptage tokens avec tokenizer, fallback approximation : %s", e)
    
    # Fallback : approximation basée sur ratio chars/tokens observé (français technique)
    return max(1, int(len(text) / 3.5))


def split_text_by_tokens(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> List[str]:
    """
    Découpe un texte en sous-parties de max_tokens tokens maximum.
    Préserve les paragraphes quand possible.
    
    Args:
        text: Texte à découper
        max_tokens: Nombre maximum de tokens par chunk
        
    Returns:
        Liste de sous-textes (1 élément si déjà < max_tokens)
    """
    if not text or not text.strip():
        return []
    
    token_count = count_tokens(text)
    if token_count <= max_tokens:
        return [text]
    
    # Split sur paragraphes (double saut de ligne)
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = count_tokens(para)
        
        # Si un seul paragraphe dépasse la limite, le découper par phrases
        if para_tokens > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Split par phrases (points + espace/newline)
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if current_tokens + sent_tokens > max_tokens and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [sent]
                    current_tokens = sent_tokens
                else:
                    current_chunk.append(sent)
                    current_tokens += sent_tokens
            continue
        
        # Si ajouter ce paragraphe dépasse la limite, flush current chunk
        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens
    
    # Flush dernier chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks if chunks else [text]


def _tokenize_rag_units(text: str) -> List[str]:
    """Découpe le texte en unités atomiques (Image, tableau markdown, paragraphes)."""
    if not text or not text.strip():
        return []

    units: List[str] = []
    pos = 0
    while pos < len(text):
        img_match = _IMAGE_BLOCK_RE.search(text, pos)
        next_pos = len(text)
        chunk_end = len(text)

        if img_match and img_match.start() == pos:
            units.append(img_match.group(0))
            pos = img_match.end()
            continue

        if img_match:
            chunk_end = img_match.start()
        else:
            chunk_end = len(text)

        segment = text[pos:chunk_end]
        if segment.strip():
            lines = segment.split("\n")
            table_buf: List[str] = []
            para_buf: List[str] = []

            def flush_table():
                nonlocal table_buf
                if table_buf:
                    units.append("\n".join(table_buf))
                    table_buf = []

            def flush_para():
                nonlocal para_buf
                if para_buf:
                    units.append("\n".join(para_buf))
                    para_buf = []

            for line in lines:
                if "|" in line and line.strip():
                    flush_para()
                    table_buf.append(line)
                else:
                    flush_table()
                    if line.strip() == "" and para_buf:
                        flush_para()
                    elif line.strip():
                        para_buf.append(line)
                    elif not line.strip() and not para_buf:
                        pass
            flush_table()
            flush_para()

        pos = chunk_end if img_match else len(text)

    return [u for u in units if u and u.strip()]


def _pack_rag_units(
    units: List[str],
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = 0,
) -> List[str]:
    """Regroupe les unités atomiques en chunks ≤ max_tokens."""
    if not units:
        return []

    overlap_tokens = max(0, overlap_tokens or 0)
    chunks: List[str] = []
    current_parts: List[str] = []
    current_tokens = 0

    def flush():
        nonlocal current_parts, current_tokens
        if current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_tokens = 0

    for unit in units:
        ut = count_tokens(unit)
        if ut > max_tokens:
            flush()
            if "|" in unit and "\n" in unit:
                lines = unit.split("\n")
                line_buf: List[str] = []
                lt = 0
                for line in lines:
                    line_t = count_tokens(line)
                    if lt + line_t > max_tokens and line_buf:
                        chunks.append("\n".join(line_buf))
                        line_buf = [line]
                        lt = line_t
                    else:
                        line_buf.append(line)
                        lt += line_t
                if line_buf:
                    chunks.append("\n".join(line_buf))
            else:
                sub = split_text_by_tokens(unit, max_tokens)
                chunks.extend(sub)
            continue

        if current_tokens + ut > max_tokens and current_parts:
            flush()
        current_parts.append(unit)
        current_tokens += ut

    flush()

    if overlap_tokens > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = overlapped[-1]
            current = chunks[i]
            prefix = _tail_sentences_by_tokens(prev, overlap_tokens)
            if not prefix:
                overlapped.append(current)
                continue

            sep = "\n\n"
            combined = f"{prefix}{sep}{current}"
            if count_tokens(combined) <= max_tokens:
                overlapped.append(combined)
                continue

            # Budget strict: on réduit le préfixe overlap pour garantir <= max_tokens.
            low, high = 0, overlap_tokens
            best = current
            while low <= high:
                mid = (low + high) // 2
                test_prefix = _tail_sentences_by_tokens(prev, mid)
                test_combined = (
                    f"{test_prefix}{sep}{current}" if test_prefix else current
                )
                if count_tokens(test_combined) <= max_tokens:
                    best = test_combined
                    low = mid + 1
                else:
                    high = mid - 1
            overlapped.append(best)
        return overlapped

    return chunks if chunks else ["\n\n".join(units)]


def _tail_sentences_by_tokens(text: str, target_tokens: int) -> str:
    """
    Extrait un suffixe composé de phrases entières de l'ordre de target_tokens tokens.
    Évite les coupures de phrases au milieu lors du glissement de l'overlap.
    """
    if not text or target_tokens <= 0:
        return ""
    
    # Découper en phrases
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return ""
        
    collected_sentences = []
    tokens_count = 0
    
    # Parcourir à l'envers depuis la fin
    for i in range(len(sentences) - 1, -1, -1):
        sent = sentences[i].strip()
        if not sent:
            continue
        sent_tokens = count_tokens(sent)
        if not collected_sentences:
            collected_sentences.insert(0, sent)
            tokens_count += sent_tokens
            if tokens_count >= target_tokens:
                break
        else:
            if tokens_count + sent_tokens > target_tokens:
                break
            collected_sentences.insert(0, sent)
            tokens_count += sent_tokens
            
    return " ".join(collected_sentences)


def split_text_rag_friendly(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: Optional[int] = None,
) -> List[str]:
    """Découpe RAG-friendly : unités Image/tableau indivisibles + overlap optionnel."""
    if not text or not text.strip():
        return []
    if count_tokens(text) <= max_tokens:
        return [text]
    overlap = (
        overlap_tokens
        if overlap_tokens is not None
        else getattr(settings, "RAG_CHUNK_OVERLAP_TOKENS", 40)
    )
    units = _tokenize_rag_units(text)
    if not units:
        return split_text_by_tokens(text, max_tokens)
    return _pack_rag_units(units, max_tokens=max_tokens, overlap_tokens=overlap)


def assert_chunk_rag_quality(
    content: str,
    *,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> Tuple[str, dict]:
    """Valide un chunk leaf ; retourne contenu éventuellement corrigé + flags."""
    flags: dict = {"split_warning": False}
    if not content or not content.strip():
        return content, flags

    if not content.strip().startswith("Document:"):
        flags["split_warning"] = True

    tc = count_tokens(content)
    if tc > max_tokens:
        flags["split_warning"] = True
        # Troncature token-aware de dernier recours.
        words = content.split()
        lo, hi = 0, len(words)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = " ".join(words[:mid])
            if count_tokens(candidate) <= max_tokens:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        content = best if best else content[: int(max_tokens * 3.5)]
        logger.warning("Chunk tronqué d'urgence : %d tokens > %d", tc, max_tokens)

    if _IMAGE_BLOCK_RE.search(content) and content.rstrip().endswith("[Image:"):
        flags["split_warning"] = True

    return content, flags


def page_text_quality_score(pymupdf_text: str) -> float:
    """Score 0–1 de qualité du texte natif extrait."""
    text = (pymupdf_text or "").strip()
    if not text:
        return 0.0
    length = len(text)
    alnum = sum(1 for c in text if c.isalnum())
    ratio = alnum / max(length, 1)
    has_structure = 1.0 if ("|" in text or re.search(r"^#+\s", text, re.M)) else 0.0
    length_score = min(1.0, length / 500.0)
    return min(1.0, 0.5 * length_score + 0.3 * ratio + 0.2 * has_structure)


def resolve_extraction_mode(pymupdf_text: str) -> str:
    """native si assez de texte extractible, sinon scanned."""
    min_chars = getattr(settings, "MULTIMODAL_NATIVE_TEXT_MIN_CHARS", 100)
    if len((pymupdf_text or "").strip()) >= min_chars:
        return "native"
    return "scanned"


def page_has_significant_visuals(pdf_path: str, page_index: int) -> bool:
    """Heuristique : ratio zone image élevé sur la page."""
    try:
        import fitz

        doc = fitz.open(pdf_path)
        if page_index < 0 or page_index >= len(doc):
            doc.close()
            return False
        page = doc[page_index]
        img_area = 0.0
        for img in page.get_images():
            try:
                rects = page.get_image_rects(img[0])
                for r in rects:
                    img_area += abs(r.width * r.height)
            except Exception:
                pass
        page_area = abs(page.rect.width * page.rect.height) or 1.0
        doc.close()
        return (img_area / page_area) > 0.25
    except Exception as exc:
        logger.debug("page_has_significant_visuals: %s", exc)
        return False


def count_image_blocks(text: str) -> int:
    return len(_IMAGE_BLOCK_RE.findall(text or ""))


_RAW_SYSTEM_PROMPT_NATIVE = """Tu es un expert en extraction documentaire pour RAG technique.
Tu reçois le texte pymupdf (source de vérité) et l'image PNG de la même page.

Réponds UNIQUEMENT avec un JSON valide :
{"page_no": <int>, "raw_text": "<texte brut enrichi>"}

Règles STRICTES pour raw_text :
- Le bloc pymupdf fourni est la source de vérité pour tout texte déjà extractible : NE PAS le réécrire ni le paraphraser.
- Parcourir le PNG et insérer à l'emplacement logique des blocs [Image: description technique détaillée] pour chaque schéma, photo, plan, dessin technique ou tableau visuel ABSENT du pymupdf.
- Pour chaque schéma ou image technique, la description dans `[Image: ...]` doit impérativement transcrire et lister TOUS les textes, légendes, références, valeurs numériques, cotes, cibles et annotations textuelles visibles dans l'image (ex: 'NF EN 1991', '55mm', 'PVC-76', etc.) afin de lier parfaitement le texte et le visuel pour le RAG.
- Ne pas supprimer de contenu pymupdf. Tableaux texte : markdown pipes.
- Français ; n'invente rien ; « illisible » si zone floue."""

_RAW_SYSTEM_PROMPT_SCANNED = """Tu es un expert en OCR et extraction documentaire pour RAG technique.
Tu reçois l'image PNG d'une page PDF scannée ou sans texte extractible.

Réponds UNIQUEMENT avec un JSON valide :
{"page_no": <int>, "raw_text": "<texte brut enrichi>"}

Règles pour raw_text :
- OCR complet et ordonné de la page.
- Pour chaque schéma/image : [Image: description technique détaillée transcrivant toutes les annotations textuelles, cotes, valeurs et unités visibles au sein du dessin/schéma].
- Tableaux : markdown pipes si possible.
- Français ; n'invente rien."""

_RAW_SCHEMA_RETRY_PROMPT = """IMPORTANT: Retourne UNIQUEMENT un JSON strict {"page_no": int, "raw_text": "..."}."""

_SCHEMAS_ONLY_PROMPT = """Le texte pymupdf est déjà fourni et doit rester inchangé.
Analyse UNIQUEMENT le PNG et ajoute les blocs [Image: …] manquants pour schémas/photos non décrits dans le texte actuel.
Retourne JSON {"page_no": int, "raw_text": "<texte pymupdf + nouveaux [Image:…]>"} sans réécrire le texte existant."""

_WINDOW_PRO_SYSTEM_PROMPT = """Tu es un expert en documentation technique et normative (RAG).
Tu reçois le texte brut concaténé de plusieurs pages d'une fenêtre.

Réponds UNIQUEMENT avec un JSON valide :
{
  "window_start": <int>,
  "window_end": <int>,
  "pro_reports": [
    {
      "theme": "<titre thème>",
      "report": "<rapport dense, explicite>",
      "references": ["<référence nommée>", ...],
      "keywords": ["..."],
      "norms": ["NF EN ...", "DTU ..."],
      "constraints": ["<contrainte chiffrée avec unité>", ...],
      "dependencies": ["..."]
    }
  ]
}

Règles STRICTES :
- Produire 1 à 6 rapports selon la complexité (thèmes distincts).
- INTERDIT : « ce profil », « cette section », « ci-dessus », pronoms sans antécédent nommé.
- OBLIGATION : nommer produits, références catalogue, normes complètes, valeurs + unités + pages sources.
- Chaque assertion technique doit être explicite et actionnable.
- Français ; n'invente rien hors du texte fourni."""

_USER_PROMPT_NATIVE = """Document : {title}
Page : {page_no}

Texte pymupdf (source de vérité — ne pas réécrire) :
---
{pymupdf_text}
---

Analyse le PNG. Retourne JSON avec raw_text = pymupdf inchangé + [Image: …] pour tout visuel absent."""

_USER_PROMPT_SCANNED = """Document : {title}
Page : {page_no}

Page scannée ou sans texte extractible. Analyse le PNG.
Retourne JSON avec raw_text = OCR complet + [Image: …] pour schémas."""


@dataclass
class MultimodalChunkSpec:
    """Spécification d'un chunk multimodal à persister."""

    page_no: int
    content: str
    content_type: str
    node_id: str
    metadata: dict = field(default_factory=dict)


def _multimodal_page_model() -> str:
    return (settings.MULTIMODAL_PAGE_MODEL or "mistral-small-latest").strip()


def _content_type_col():
    return DocumentChunk.metadata_json["content_type"].as_string()


def delete_multimodal_chunks_for_document(
    session: Session, document_id: int, commit: bool = True
) -> int:
    """Supprime tous les chunks multimodal (v1/v2 legacy + v3 raw/report)."""
    col = _content_type_col()
    result = session.execute(
        delete(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            or_(*[col == ct for ct in ALL_MULTIMODAL_CONTENT_TYPES]),
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


def _parse_json_from_llm_content(raw: str) -> dict:
    """Extrait un objet JSON depuis la réponse LLM."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Réponse LLM vide")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))
    raise ValueError("JSON introuvable dans la réponse LLM")


def _repair_common_json_issues(raw: str) -> str:
    """
    Répare des erreurs JSON fréquentes de sorties LLM :
    - wrapper markdown ```json
    - guillemets typographiques
    - virgules terminales avant } ou ]
    """
    text = (raw or "").strip()
    if not text:
        return text

    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        text = match.group(0)

    # Supprime les virgules terminales non valides
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()


def _parse_json_with_repair(raw: str) -> dict:
    """Parse JSON avec tentative de réparation avant échec."""
    try:
        return _parse_json_from_llm_content(raw)
    except (ValueError, json.JSONDecodeError):
        repaired = _repair_common_json_issues(raw)
        if not repaired:
            raise
        return _parse_json_from_llm_content(repaired)


def _as_str_list(val: Any) -> List[str]:
    if not val:
        return []
    if isinstance(val, str):
        return [val.strip()] if val.strip() else []
    if isinstance(val, list):
        return [str(x).strip() for x in val if x and str(x).strip()]
    return []


def _normalize_report_dict(rep: dict, index: int, document_title: str, page_no: int) -> Optional[dict]:
    """Normalise un rapport technique de thème (pro_report)."""
    if not isinstance(rep, dict):
        return None

    theme = (rep.get("theme") or f"Thème {index + 1}").strip()
    content = (rep.get("report") or rep.get("content") or "").strip()

    if not content:
        return None

    return {
        "theme": theme,
        "report": content,
        "references": _as_str_list(rep.get("references")),
        "keywords": _as_str_list(rep.get("keywords")),
        "norms": _as_str_list(rep.get("norms")),
        "constraints": _as_str_list(rep.get("constraints")),
        "dependencies": _as_str_list(rep.get("dependencies")),
    }


def parse_multimodal_page_response(
    data: dict,
    page_no: int,
    pymupdf_text: str,
    document_title: str,
    *,
    extraction_mode: str = "native",
) -> dict:
    """Valide et normalise la réponse JSON Pass 1 (raw_text uniquement)."""
    raw_text = (data.get("raw_text") or "").strip()
    if not raw_text:
        raw_text = pymupdf_text.strip() or f"(contenu non extractible — page {page_no})"

    return {
        "page_no": page_no,
        "page_start": page_no,
        "page_end": page_no,
        "raw_text": raw_text,
        "extraction_mode": extraction_mode,
        "pymupdf_char_count": len((pymupdf_text or "").strip()),
        "image_block_count": count_image_blocks(raw_text),
    }


def validate_raw_page(
    parsed: dict,
    pymupdf_text: str,
    *,
    pdf_path: Optional[str] = None,
    page_index: Optional[int] = None,
) -> Tuple[dict, str]:
    """
    Valide le raw_text après Pass 1.
    Retourne (parsed mis à jour, status: ok|warning|retry_schemas|vision_failed).
    """
    mode = parsed.get("extraction_mode", "native")
    raw = (parsed.get("raw_text") or "").strip()
    pymupdf = (pymupdf_text or "").strip()
    status = "ok"

    if mode == "native" and pymupdf:
        if len(pymupdf) > 20:
            ratio = len(pymupdf) / max(len(raw), 1)
            if ratio > 1.1 or ratio < 0.35:
                status = "warning"
                logger.warning(
                    "page %s ratio pymupdf/raw=%.2f (réécriture suspecte)",
                    parsed.get("page_no"),
                    ratio,
                )

    if (
        pdf_path
        and page_index is not None
        and page_has_significant_visuals(pdf_path, page_index)
        and count_image_blocks(raw) == 0
    ):
        return parsed, "retry_schemas"

    if "|" in pymupdf and "|" not in raw and mode == "native":
        parsed["raw_text"] = pymupdf + ("\n\n" + raw if raw else "")
        parsed["image_block_count"] = count_image_blocks(parsed["raw_text"])
        status = "warning"

    if not raw and pymupdf:
        parsed["raw_text"] = pymupdf
        status = "vision_failed"

    parsed["raw_validation_status"] = status
    return parsed, status


def _format_raw_enriched_chunk(
    document_title: str,
    page_no: int,
    raw_text: str,
    page_start: int,
    page_end: int,
) -> str:
    """Formate un chunk de texte brut enrichi (page complète ou plage de pages)."""
    page_str = f"Page {page_no}" if page_start == page_end else f"Pages {page_start}-{page_end}"
    return f"[{page_str}]\n{raw_text}"


def _format_pro_report_chunk(
    document_title: str,
    page_no: int,
    report: dict,
    page_start: int,
    page_end: int,
) -> str:
    """Formate un chunk rapport pro en prose naturelle."""
    page_str = f"page {page_no}" if page_start == page_end else f"pages {page_start} à {page_end}"
    parts = []
    theme = (report.get("theme") or "").strip()
    if theme:
        parts.append(f"Rapport technique sur le thème : {theme} ({page_str}).")
    else:
        parts.append(f"Rapport technique ({page_str}).")
        
    refs = report.get("references") or []
    if refs:
        parts.append(f"Références : {', '.join(refs)}.")
    norms = report.get("norms") or []
    if norms:
        parts.append(f"Normes applicables : {', '.join(norms)}.")
    constraints = report.get("constraints") or []
    if constraints:
        parts.append(f"Contraintes techniques : {', '.join(constraints)}.")
    deps = report.get("dependencies") or []
    if deps:
        parts.append(f"Dépendances : {', '.join(deps)}.")
        
    report_text = (report.get("report") or "").strip()
    parts.append(f"\n{report_text}")
    return "\n".join(parts).strip()


def _enforce_token_limit(content: str, page_no: int, label: str, sec_idx: int, part_idx: int) -> str:
    """Validation finale : troncature d'urgence si split insuffisant."""
    final_token_count = count_tokens(content)
    if final_token_count > MAX_CHUNK_TOKENS:
        logger.error(
            "VALIDATION ÉCHOUÉE : chunk %s page_no=%s section_idx=%s part=%d contient %d tokens > %d",
            label,
            page_no,
            sec_idx,
            part_idx + 1,
            final_token_count,
            MAX_CHUNK_TOKENS,
        )
        return content[: int(MAX_CHUNK_TOKENS * 3.5)]
    return content


def build_raw_chunk_specs_from_page(
    document_id: int,
    page_no: int,
    parsed: dict,
    document_title: str,
) -> List[MultimodalChunkSpec]:
    """Construit les specs page_raw_enriched (split RAG-friendly ≤480 tokens)."""
    specs: List[MultimodalChunkSpec] = []
    model_name = _multimodal_page_model()
    page_start = int(parsed.get("page_start", page_no))
    page_end = int(parsed.get("page_end", page_no))
    raw_text = (parsed.get("raw_text") or "").strip()
    if not raw_text:
        return specs

    formatted_raw = _format_raw_enriched_chunk(
        document_title, page_no, raw_text, page_start, page_end
    )
    raw_parts = split_text_rag_friendly(formatted_raw, MAX_CHUNK_TOKENS)
    for part_idx, part_content in enumerate(raw_parts):
        part_content, quality_flags = assert_chunk_rag_quality(part_content)
        node_suffix = "raw" if len(raw_parts) == 1 else f"raw-part{part_idx + 1}"
        node_id = f"multimodal-page-{document_id}-{page_no}-{node_suffix}"
        meta = {
            "content_type": PAGE_RAW_ENRICHED_CONTENT_TYPE,
            "chunking_version": CHUNKING_VERSION_MULTIMODAL,
            "page_no": page_no,
            "page_start": page_start,
            "page_end": page_end,
            "generation_method": "pymupdf+mistral_small_v4",
            "llm_model": model_name,
            "document_id": document_id,
            "document_title": document_title or "",
            "extraction_mode": parsed.get("extraction_mode", "native"),
            "pymupdf_char_count": parsed.get("pymupdf_char_count", 0),
            "image_block_count": parsed.get("image_block_count", 0),
            "raw_validation_status": parsed.get("raw_validation_status", "ok"),
            "is_leaf": True,
            "is_split": len(raw_parts) > 1,
            "split_part": part_idx + 1 if len(raw_parts) > 1 else None,
            "split_total": len(raw_parts) if len(raw_parts) > 1 else None,
            "token_count": count_tokens(part_content),
            **quality_flags,
        }
        specs.append(
            MultimodalChunkSpec(
                page_no=page_no,
                content=part_content,
                content_type=PAGE_RAW_ENRICHED_CONTENT_TYPE,
                node_id=node_id,
                metadata=meta,
            )
        )

    logger.info(
        "Page %s : %d raw chunk(s) (≤%d tokens)",
        page_no,
        len(specs),
        MAX_CHUNK_TOKENS,
    )
    return specs


def _format_window_pro_report_chunk(
    document_title: str,
    window_start: int,
    window_end: int,
    report: dict,
    window_id: str,
) -> str:
    """Formate un chunk rapport pro fenêtre en prose naturelle pour éviter le bruit d'en-tête."""
    page_str = (
        f"pages {window_start} à {window_end}"
        if window_start != window_end
        else f"page {window_start}"
    )
    
    parts = []
    theme = (report.get("theme") or "").strip()
    if theme:
        parts.append(f"Rapport technique sur le thème : {theme} ({page_str}).")
    else:
        parts.append(f"Rapport technique ({page_str}).")
        
    refs = report.get("references") or []
    if refs:
        parts.append(f"Références : {', '.join(refs)}.")
        
    norms = report.get("norms") or []
    if norms:
        parts.append(f"Normes applicables : {', '.join(norms)}.")
        
    constraints = report.get("constraints") or []
    if constraints:
        parts.append(f"Contraintes techniques : {', '.join(constraints)}.")
        
    deps = report.get("dependencies") or []
    if deps:
        parts.append(f"Dépendances : {', '.join(deps)}.")
        
    report_text = (report.get("report") or "").strip()
    parts.append(f"\n{report_text}")
    
    return "\n".join(parts).strip()


def build_window_report_chunk_specs(
    document_id: int,
    window: dict,
    pro_reports: List[dict],
    document_title: str,
) -> List[MultimodalChunkSpec]:
    """Construit les specs page_window_report pour une fenêtre."""
    specs: List[MultimodalChunkSpec] = []
    model_name = _multimodal_page_model()
    window_id = window["window_id"]
    w_start = int(window["window_start"])
    w_end = int(window["window_end"])

    for rep_idx, report in enumerate(pro_reports):
        formatted = _format_window_pro_report_chunk(
            document_title, w_start, w_end, report, window_id
        )
        parts = split_text_rag_friendly(formatted, MAX_CHUNK_TOKENS)
        for part_idx, part_content in enumerate(parts):
            part_content, quality_flags = assert_chunk_rag_quality(part_content)
            suffix = f"win-{w_start}-{w_end}-r{rep_idx + 1}"
            if len(parts) > 1:
                suffix = f"{suffix}-part{part_idx + 1}"
            node_id = f"multimodal-{document_id}-{suffix}"
            meta = {
                "content_type": PAGE_WINDOW_REPORT_CONTENT_TYPE,
                "chunking_version": CHUNKING_VERSION_MULTIMODAL,
                "window_id": window_id,
                "window_start": w_start,
                "window_end": w_end,
                "page_no": w_start,
                "page_start": w_start,
                "page_end": w_end,
                "report_index": rep_idx + 1,
                "theme": report.get("theme") or "",
                "references": report.get("references") or [],
                "keywords": report.get("keywords") or [],
                "norms": report.get("norms") or [],
                "constraints": report.get("constraints") or [],
                "dependencies": report.get("dependencies") or [],
                "generation_method": "mistral_text_window_v4",
                "llm_model": model_name,
                "document_id": document_id,
                "document_title": document_title or "",
                "is_leaf": True,
                "is_split": len(parts) > 1,
                "split_part": part_idx + 1 if len(parts) > 1 else None,
                "split_total": len(parts) if len(parts) > 1 else None,
                "token_count": count_tokens(part_content),
                "paired_content_types": [
                    PAGE_RAW_ENRICHED_CONTENT_TYPE,
                    PAGE_WINDOW_REPORT_CONTENT_TYPE,
                ],
                **quality_flags,
            }
            specs.append(
                MultimodalChunkSpec(
                    page_no=w_start,
                    content=part_content,
                    content_type=PAGE_WINDOW_REPORT_CONTENT_TYPE,
                    node_id=node_id,
                    metadata=meta,
                )
            )
    return specs


def validate_pro_reports(pro_reports: List[dict]) -> Tuple[List[dict], bool]:
    """Retourne (reports, needs_retry) si déictiques détectés."""
    needs_retry = False
    for rep in pro_reports:
        text = (rep.get("report") or "") + " ".join(rep.get("references") or [])
        if _DEICTIC_RE.search(text):
            needs_retry = True
            break
    return pro_reports, needs_retry


def _mistral_chat_completion(
    messages: list,
    *,
    page_no: int,
    max_tokens: Optional[int] = None,
    temperature: float = 0.2,
    response_format_json: bool = True,
    timeout_seconds: Optional[float] = None,
) -> str:
    """Appel Mistral chat/completions avec retries."""
    api_key = settings.MISTRAL_API_KEY
    if not api_key:
        raise ValueError("MISTRAL_API_KEY n'est pas configurée")

    payload: dict = {
        "model": _multimodal_page_model(),
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens or settings.MULTIMODAL_PAGE_MAX_TOKENS,
        "temperature": temperature,
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    base_url = (settings.MISTRAL_BASE_URL or "https://api.mistral.ai").rstrip("/")
    timeout = float(
        timeout_seconds
        if timeout_seconds is not None
        else getattr(settings, "MISTRAL_OCR_TIMEOUT", 300) or 300
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    import random

    max_attempts = 4
    backoff_base = 2.0
    retryable_codes = {429, 500, 502, 503, 504}
    last_exception = None
    for attempt in range(1, max_attempts + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    f"{base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            raw_out = (msg.get("content") or "").strip()
            if not raw_out:
                raise RuntimeError(f"Réponse vide pour la page {page_no}")
            return raw_out
        except httpx.HTTPStatusError as e:
            last_exception = e
            if e.response.status_code in retryable_codes and attempt < max_attempts:
                wait = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                time.sleep(wait)
                continue
            raise
        except (httpx.RequestError, RuntimeError) as e:
            last_exception = e
            if attempt < max_attempts:
                wait = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                time.sleep(wait)
                continue
            raise
    if last_exception:
        raise last_exception
    raise RuntimeError("Échec appel Mistral après retries")


def synthesize_page_with_mistral_small(
    image_png: bytes,
    page_no: int,
    pymupdf_text: str,
    document_title: str,
    *,
    extraction_mode: Optional[str] = None,
    pdf_path: Optional[str] = None,
    page_index: Optional[int] = None,
) -> dict:
    """Pass 1 : mistral-small vision → JSON { raw_text } uniquement."""
    mode = extraction_mode or resolve_extraction_mode(pymupdf_text)
    b64 = base64.b64encode(image_png).decode("ascii")

    if mode == "native":
        pymupdf_block = pymupdf_text.strip() if pymupdf_text else "(aucun texte extractible)"
        user_text = _USER_PROMPT_NATIVE.format(
            title=document_title or "Document",
            page_no=page_no,
            pymupdf_text=pymupdf_block,
        )
        system_prompt = _RAW_SYSTEM_PROMPT_NATIVE
    else:
        user_text = _USER_PROMPT_SCANNED.format(
            title=document_title or "Document",
            page_no=page_no,
        )
        system_prompt = _RAW_SYSTEM_PROMPT_SCANNED

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
            ],
        },
    ]

    parsed_raw: dict = {}
    try:
        raw_content = _mistral_chat_completion(messages, page_no=page_no)
        parsed_raw = _parse_json_with_repair(raw_content)
    except (ValueError, json.JSONDecodeError) as exc_first:
        logger.warning("page_no=%s JSON invalide (%s), retry strict", page_no, exc_first)
        retry_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_text + "\n\n" + _RAW_SCHEMA_RETRY_PROMPT,
                    },
                    {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
                ],
            },
        ]
        try:
            retry_raw = _mistral_chat_completion(
                retry_messages, page_no=page_no, temperature=0.0
            )
            parsed_raw = _parse_json_with_repair(retry_raw)
        except (ValueError, json.JSONDecodeError) as exc_retry:
            logger.warning("page_no=%s JSON invalide après retry (%s)", page_no, exc_retry)
            parsed_raw = {}

    parsed = parse_multimodal_page_response(
        parsed_raw, page_no, pymupdf_text, document_title, extraction_mode=mode
    )
    parsed, status = validate_raw_page(
        parsed,
        pymupdf_text,
        pdf_path=pdf_path,
        page_index=page_index,
    )

    if status == "retry_schemas" and mode == "native":
        schema_messages = [
            {"role": "system", "content": _RAW_SYSTEM_PROMPT_NATIVE},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Document: {document_title}\nPage: {page_no}\n\n"
                            f"Texte actuel:\n{parsed.get('raw_text', '')}\n\n"
                            + _SCHEMAS_ONLY_PROMPT
                        ),
                    },
                    {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
                ],
            },
        ]
        try:
            schema_raw = _mistral_chat_completion(schema_messages, page_no=page_no)
            schema_parsed = _parse_json_with_repair(schema_raw)
            if schema_parsed.get("raw_text"):
                parsed["raw_text"] = schema_parsed["raw_text"]
                parsed["image_block_count"] = count_image_blocks(parsed["raw_text"])
                parsed["raw_validation_status"] = "ok"
        except Exception as exc:
            logger.warning("page_no=%s retry schémas échoué: %s", page_no, exc)

    return parsed


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
    chunk_specs: List[MultimodalChunkSpec],
) -> List[DocumentChunk]:
    """
    Persiste les chunks multimodal (feuilles et résumés parents).
    """
    base_index = _next_chunk_index(session, document.id)
    chunks: List[DocumentChunk] = []

    for offset, spec in enumerate(chunk_specs):
        if not spec.content or not spec.content.strip():
            continue
        
        ct = spec.metadata.get("content_type")
        assert ct in MULTIMODAL_CONTENT_TYPES, (
            f"Type de chunk multimodal invalide: {ct}"
        )
        
        is_leaf = spec.metadata.get("is_leaf", True)
        parent_node_id = spec.metadata.get("parent_node_id")
        hierarchy_level = spec.metadata.get("hierarchy_level", 0)

        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=base_index + offset,
            content=spec.content.strip(),
            text=spec.content.strip(),
            start_char=0,
            end_char=len(spec.content),
            node_id=spec.node_id,
            parent_node_id=parent_node_id,
            is_leaf=is_leaf,
            hierarchy_level=hierarchy_level,
            metadata_json=spec.metadata,
            metadata_=spec.metadata,
            source=document.source,
        )
        chunks.append(chunk)

    if chunks:
        session.add_all(chunks)
        session.commit()
    return chunks


def _process_single_multimodal_page(
    pdf_path: str,
    page_no: int,
    page_index: int,
    pymupdf_text: str,
    document_title: str,
    document_id: int,
    *,
    page_idx: int,
    total_pages: int,
) -> Tuple[int, dict]:
    """Traite une page : PNG + mistral-small → JSON dict (appelable en parallèle)."""
    ld = get_library_document_logger()
    ld.info(
        "[Multimodal] page %s/%s (page_no=%s)",
        page_idx + 1,
        total_pages,
        page_no,
    )
    t0 = time.perf_counter()
    mode = resolve_extraction_mode(pymupdf_text)
    png = render_pdf_page_png(pdf_path, page_index)
    parsed = synthesize_page_with_mistral_small(
        png,
        page_no,
        pymupdf_text,
        document_title,
        extraction_mode=mode,
        pdf_path=pdf_path,
        page_index=page_index,
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "multimodal page_no=%s %.2fs",
        page_no,
        elapsed,
    )
    return page_no, parsed


def merge_cut_sections_across_pages(parsed_pages: List[dict]) -> List[dict]:
    """
    Fusionne le raw_text des pages successives si coupure de paragraphe ou tableau (v4 : raw only).
    """
    if len(parsed_pages) <= 1:
        return parsed_pages

    parsed_pages = sorted(parsed_pages, key=lambda x: x["page_no"])

    # 1. Fusion des raw_text
    for i in range(len(parsed_pages) - 1):
        curr_page = parsed_pages[i]
        next_page = parsed_pages[i + 1]

        raw_curr = curr_page.get("raw_text", "").strip()
        raw_next = next_page.get("raw_text", "").strip()

        if not raw_curr or not raw_next:
            continue

        ends_with_table = raw_curr.endswith("|") or " |" in raw_curr[-20:]
        starts_with_table = raw_next.startswith("|") or "| " in raw_next[:20]
        both_tables = ends_with_table and starts_with_table

        ends_abruptly = raw_curr[-1] not in (".", "!", "?", '"', '»') if raw_curr else False

        should_merge = both_tables or ends_abruptly

        if should_merge:
            logger.info(
                "Fusion du raw_text de la page %s avec la page %s (tables: %s, abrupt: %s)",
                curr_page["page_no"],
                next_page["page_no"],
                both_tables,
                ends_abruptly,
            )
            curr_page["page_start"] = curr_page.get("page_start", curr_page["page_no"])
            curr_page["page_end"] = next_page.get("page_end", next_page["page_no"])

            if both_tables:
                curr_page["raw_text"] = _merge_markdown_tables(raw_curr, raw_next)
            else:
                curr_page["raw_text"] = raw_curr + "\n" + raw_next

            next_page["raw_text"] = ""
            next_page["page_start"] = curr_page["page_end"]

    return [p for p in parsed_pages if (p.get("raw_text") or "").strip()]


def _merge_markdown_tables(table_a: str, table_b: str) -> str:
    """
    Fusionne deux tableaux markdown en conservant l'en-tête du premier et en concaténant les lignes.
    """
    lines_a = [l.strip() for l in table_a.split("\n") if l.strip()]
    lines_b = [l.strip() for l in table_b.split("\n") if l.strip()]

    if not lines_a:
        return table_b
    if not lines_b:
        return table_a

    # Trouver l'en-tête du tableau B pour le sauter
    start_idx = 0
    if len(lines_b) > 1 and "|" in lines_b[0] and "|" in lines_b[1]:
        if re.search(r"\|?\s*:?-+:?\s*\|", lines_b[1]):
            start_idx = 2
        elif "|" in lines_b[0]:
            start_idx = 1

    merged_lines = lines_a + lines_b[start_idx:]
    return "\n".join(merged_lines)


def _is_major_heading_line(line: str) -> bool:
    s = line.strip()
    return bool(re.match(r"^#{1,2}\s+\S", s)) or bool(re.match(r"^[A-Z][A-Z0-9\s\-]{8,}$", s))


def build_dynamic_windows(
    document_id: int,
    ordered_pages: List[dict],
) -> List[dict]:
    """
    Fenêtres dynamiques : tokens max, rupture thème (headings), max pages, overlap 1.
    """
    if not ordered_pages:
        return []

    max_tokens = getattr(settings, "WINDOW_MAX_INPUT_TOKENS", 7000)
    max_pages = getattr(settings, "WINDOW_MAX_PAGES", 12)
    overlap_pages = max(0, getattr(settings, "WINDOW_PAGE_OVERLAP", 1))

    windows: List[dict] = []
    i = 0
    while i < len(ordered_pages):
        batch: List[dict] = []
        token_acc = 0
        j = i
        while j < len(ordered_pages):
            page = ordered_pages[j]
            page_text = (page.get("raw_text") or "").strip()
            part = f"--- PAGE {page['page_no']} ---\n{page_text}" if page_text else ""
            part_tokens = count_tokens(part) if part else 0

            if batch and (
                token_acc + part_tokens > max_tokens
                or len(batch) >= max_pages
                or (
                    j > i
                    and page_text
                    and any(_is_major_heading_line(ln) for ln in page_text.split("\n")[:3])
                )
            ):
                break

            if page_text:
                batch.append(page)
                token_acc += part_tokens
            j += 1

        if not batch:
            i += 1
            continue

        start_page = batch[0]["page_no"]
        end_page = batch[-1]["page_no"]
        window_id = f"win-{document_id}-{start_page}-{end_page}"
        context_parts = [
            f"--- PAGE {p['page_no']} ---\n{p['raw_text'].strip()}"
            for p in batch
            if (p.get("raw_text") or "").strip()
        ]
        windows.append(
            {
                "window_id": window_id,
                "window_start": start_page,
                "window_end": end_page,
                "pages": batch,
                "raw_concat": "\n\n".join(context_parts),
            }
        )

        if j >= len(ordered_pages):
            break
        i = max(i + 1, j - overlap_pages)

    return windows


def generate_window_pro_reports(
    document_id: int,
    document_title: str,
    windows: List[dict],
) -> Dict[str, List[dict]]:
    """
    Pass 2 texte : rapports pro par fenêtre.
    Retourne { window_id: [pro_reports...] }.
    """
    result: Dict[str, List[dict]] = {}
    if not windows:
        return result
    if not settings.MISTRAL_API_KEY:
        logger.warning("MISTRAL_API_KEY manquante, Pass 2 fenêtres ignoré.")
        return result

    max_reports = getattr(settings, "MAX_REPORTS_PER_WINDOW", MAX_REPORTS_PER_WINDOW)

    total_windows = len(windows)
    logger.info(
        "Pass 2 démarrage doc=%s fenêtres=%s timeout=%ss",
        document_id,
        total_windows,
        getattr(settings, "MISTRAL_PASS2_TIMEOUT", 90),
    )

    for idx_window, window in enumerate(windows, start=1):
        context = (window.get("raw_concat") or "").strip()
        if not context:
            logger.info(
                "Pass 2 fenêtre %s/%s ignorée (contexte vide)",
                idx_window,
                total_windows,
            )
            continue
        w_start = window["window_start"]
        w_end = window["window_end"]
        t0_window = time.perf_counter()
        logger.info(
            "Pass 2 fenêtre %s/%s start pages=%s-%s",
            idx_window,
            total_windows,
            w_start,
            w_end,
        )
        user_prompt = f"""Document : {document_title}
Fenêtre pages {w_start} à {w_end}.

Texte brut des pages :
{context}

Génère des rapports techniques pro_reports explicites et concis pour cette fenêtre (JSON uniquement).
Maximum {max_reports} thèmes.
Chaque thème doit nommer explicitement les références/produits/normes, sans tournures vagues."""

        messages = [
            {"role": "system", "content": _WINDOW_PRO_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        pro_reports: List[dict] = []
        try:
            raw = _mistral_chat_completion(
                messages,
                page_no=w_start,
                max_tokens=min(settings.MULTIMODAL_PAGE_MAX_TOKENS, 8000),
                temperature=0.15,
                timeout_seconds=float(getattr(settings, "MISTRAL_PASS2_TIMEOUT", 90)),
            )
            data = _parse_json_with_repair(raw)
            reports_in = data.get("pro_reports") or []
            if isinstance(reports_in, list):
                for idx, rep in enumerate(reports_in[:max_reports]):
                    normalized = _normalize_report_dict(rep, idx, document_title, w_start)
                    if normalized:
                        pro_reports.append(normalized)
        except Exception as exc:
            logger.warning(
                "Pass 2 fenêtre %s-%s doc %s échoué: %s",
                w_start,
                w_end,
                document_id,
                exc,
            )

        if not pro_reports:
            pro_reports.append(
                {
                    "theme": f"Pages {w_start}-{w_end}",
                    "report": (
                        f"Synthèse technique des pages {w_start} à {w_end} "
                        f"du document {document_title or 'Document'}."
                    ),
                    "references": [],
                    "keywords": [],
                    "norms": [],
                    "constraints": [],
                    "dependencies": [],
                }
            )

        pro_reports, needs_retry = validate_pro_reports(pro_reports)
        if needs_retry:
            logger.info(
                "Pass 2 fenêtre %s-%s: déictiques détectés, conservé sans retry auto (mode light)",
                w_start,
                w_end,
            )

        result[window["window_id"]] = pro_reports
        elapsed = time.perf_counter() - t0_window
        logger.info(
            "Pass 2 fenêtre %s/%s done pages=%s-%s reports=%s elapsed=%.2fs",
            idx_window,
            total_windows,
            w_start,
            w_end,
            len(pro_reports),
            elapsed,
        )

    return result


def build_multimodal_pages_for_pdf(
    pdf_path: str,
    document_title: str,
    document_id: int,
    *,
    max_pages: Optional[int] = None,
) -> List[MultimodalChunkSpec]:
    """
    Pipeline v4 : Pass 1 vision (raw) → merge → fenêtres → Pass 2 rapports → chunks leaves.
    """
    from app.services.pdf_extraction_service import extract_page_texts_from_pdf

    page_texts = extract_page_texts_from_pdf(pdf_path)
    if not page_texts:
        import fitz
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
        except Exception as e:
            logger.warning("Échec comptage pages avec PyMuPDF pour %s : %s, fallback pdf2image", pdf_path, e)
            from pdf2image import convert_from_path
            dpi_val = settings.MULTIMODAL_PAGE_DPI or 200
            images = convert_from_path(pdf_path, dpi=dpi_val)
            total_pages = len(images)

        page_texts = [(i + 1, "") for i in range(total_pages)]

    limit = settings.VISION_MAX_IMAGES_PER_DOCUMENT
    if limit is not None and limit > 0:
        page_texts = page_texts[:limit]
    if max_pages is not None and max_pages > 0:
        page_texts = page_texts[:max_pages]

    total = len(page_texts)
    if total == 0:
        return []

    workers = max(1, min(settings.MULTIMODAL_PAGE_CONCURRENCY, total))
    page_jsons_by_no: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for idx, (page_no, pymupdf_text) in enumerate(page_texts):
            page_index = page_no - 1 if page_no > 0 else idx
            futures.append(
                executor.submit(
                    _process_single_multimodal_page,
                    pdf_path,
                    page_no,
                    page_index,
                    pymupdf_text,
                    document_title,
                    document_id,
                    page_idx=idx,
                    total_pages=total,
                )
            )
        for future in futures:
            page_no, parsed = future.result()
            page_jsons_by_no[page_no] = parsed

    # 1. Trier les pages par ordre chronologique
    ordered_pages = []
    for page_no, _ in page_texts:
        if page_no in page_jsons_by_no:
            ordered_pages.append(page_jsons_by_no[page_no])

    ordered_pages = merge_cut_sections_across_pages(ordered_pages)

    windows = build_dynamic_windows(document_id, ordered_pages)
    logger.info(
        "Pass 2 préparation doc=%s pages_raw=%s fenêtres=%s",
        document_id,
        len(ordered_pages),
        len(windows),
    )
    window_reports = generate_window_pro_reports(
        document_id, document_title, windows
    )

    all_specs: List[MultimodalChunkSpec] = []

    window_by_page: Dict[int, str] = {}
    for w in windows:
        for p in w.get("pages") or []:
            window_by_page[int(p["page_no"])] = w["window_id"]

    for page_data in ordered_pages:
        page_no = int(page_data["page_no"])
        raw_specs = build_raw_chunk_specs_from_page(
            document_id, page_no, page_data, document_title
        )
        wid = window_by_page.get(page_no)
        if wid:
            for spec in raw_specs:
                spec.metadata["window_id"] = wid
        all_specs.extend(raw_specs)

    for window in windows:
        wid = window["window_id"]
        reports = window_reports.get(wid) or []
        all_specs.extend(
            build_window_report_chunk_specs(
                document_id, window, reports, document_title
            )
        )

    logger.info(
        "Document %s v4 : %d pages, %d fenêtres, %d chunks",
        document_id,
        len(ordered_pages),
        len(windows),
        len(all_specs),
    )
    return all_specs


def embed_new_multimodal_chunks(document_id: int) -> int:
    """Embeddings mistral-embed pour les feuilles multimodal v4 sans vecteur."""
    from app.services.embedding_service import generate_embeddings_batch

    col = _content_type_col()
    with Session(engine) as session:
        statement = select(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.embedding.is_(None),
            DocumentChunk.is_leaf == True,  # noqa: E712
            or_(*[col == ct for ct in EMBEDDABLE_MULTIMODAL_CONTENT_TYPES]),
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
