"""
Retraitement multimodal v3 : pymupdf + mistral-small vision (JSON)
→ par section : chunk texte brut enrichi + chunk rapport pro (≤480 tokens chacun).

STRATÉGIE PARENT/LEAF (Option A - multimodal flat contrôlé):
- Tous les chunks multimodaux sont des LEAFS autonomes (is_leaf=True, parent_node_id=None, hierarchy_level=0)
- Pas de hiérarchie parent/leaf pour les chunks multimodaux (optimisé pour MiniLM 512 tokens)
- Chaque chunk doit être sémantiquement complet et auto-suffisant
- Compatibilité reranker cross-encoder/ms-marco-MiniLM-L-6-v2 (max_length=512)
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
from typing import Any, List, Optional, Tuple

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
PAGE_SECTION_REPORT_CONTENT_TYPE = "page_section_report"
PAGE_GROUP_SUMMARY_CONTENT_TYPE = "page_group_summary"
# Legacy (retraitement / suppression)
PAGE_MULTIMODAL_SECTION_CONTENT_TYPE = "page_multimodal_section"
PAGE_MULTIMODAL_SUMMARY_CONTENT_TYPE = "page_multimodal_summary"
CHUNKING_VERSION_MULTIMODAL = "multimodal_page_v3"
MAX_SECTIONS_PER_PAGE = 10
MAX_CHUNK_CHARS = 1500  # Limite caractères (fallback si token count échoue)
MAX_CHUNK_TOKENS = 480  # Limite stricte en tokens (marge sécurité vs 512)

MULTIMODAL_CONTENT_TYPES = (
    PAGE_RAW_ENRICHED_CONTENT_TYPE,
    PAGE_SECTION_REPORT_CONTENT_TYPE,
    PAGE_GROUP_SUMMARY_CONTENT_TYPE,
)
LEGACY_MULTIMODAL_CONTENT_TYPES = (
    PAGE_MULTIMODAL_SECTION_CONTENT_TYPE,
    PAGE_MULTIMODAL_SUMMARY_CONTENT_TYPE,
)
ALL_MULTIMODAL_CONTENT_TYPES = MULTIMODAL_CONTENT_TYPES + LEGACY_MULTIMODAL_CONTENT_TYPES

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

_SYSTEM_PROMPT = """Tu es un expert en RAG et en analyse technique de documentation industrielle et normative.
Tu reçois le texte pymupdf d'une page PDF et son image.

Tu dois répondre UNIQUEMENT avec un objet JSON valide (pas de markdown autour), selon ce schéma :
{
  "page_no": <int>,
  "raw_text": "<texte brut enrichi de la page complète>",
  "pro_reports": [
    {
      "theme": "<titre du thème technique>",
      "report": "<rapport technique professionnel RAG-friendly pour ce thème>",
      "references": ["NF EN ...", "DTU ...", ...],
      "keywords": ["mot-clé", ...],
      "norms": ["..."],
      "constraints": ["..."],
      "dependencies": ["..."]
    }
  ]
}

Règles pour le champ raw_text (texte brut enrichi de la page) :
- Extraire l'intégralité du texte de la page de manière fidèle et ordonnée.
- Pour chaque image, schéma ou tableau complexe visible, insérer une description technique précise inline à l'endroit correspondant : [Image: description concise avec valeurs, dimensions, composants].
- Ne pas interpréter, rester fidèle au document. Si tableau textuel simple, le transcrire en tableau markdown standard.
- Ce champ doit contenir toutes les données brutes cherchables de la page.
- Vise une extraction complète et claire de toute la page.

Règles pour la liste pro_reports (1 ou plusieurs rapports professionnels par thème) :
- Découper la page en thèmes ou chapitres logiques (produire entre 1 et 4 rapports selon la complexité). Si la page est simple ou traite d'un sujet unique, produire 1 seul rapport.
- Pour chaque thème, rédiger un rapport technique professionnel RAG-friendly : réinterpréter et structurer de façon actionnable les normes applicables, contraintes chiffrées, procédures, compatibilités, exigences et risques.
- Répéter les termes techniques et utiliser un vocabulaire précis.
- Chaque rapport doit être dense, sans blabla d'introduction.

Général : français ; n'invente rien ; « illisible » si zone floue ; si pymupdf vide, base-toi sur l'image."""

_USER_PROMPT_TEMPLATE = """Document : {title}
Page : {page_no}

Texte pymupdf :
---
{pymupdf_text}
---

Analyse l'image et le texte de la page. Retourne le JSON structuré contenant l'extraction brute 'raw_text' (avec images inline) et la liste de rapports techniques par thème 'pro_reports'."""


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
) -> dict:
    """
    Valide et normalise la réponse JSON du LLM.
    """
    raw_text = (data.get("raw_text") or "").strip()
    if not raw_text:
        raw_text = pymupdf_text.strip() or f"(contenu non extractible — page {page_no})"

    reports_in = data.get("pro_reports") or data.get("reports")
    reports: List[dict] = []
    
    if isinstance(reports_in, list) and reports_in:
        for i, rep in enumerate(reports_in[:MAX_SECTIONS_PER_PAGE]):
            normalized = _normalize_report_dict(rep, i, document_title, page_no)
            if normalized:
                reports.append(normalized)

    if not reports:
        reports.append({
            "theme": "Général",
            "report": (
                f"Rapport technique page {page_no} — {document_title or 'Document'}. "
                f"Synthèse documentaire et analyse des contraintes de la page."
            ),
            "references": [],
            "keywords": [],
            "norms": [],
            "constraints": [],
            "dependencies": [],
        })

    return {
        "page_no": page_no,
        "raw_text": raw_text,
        "pro_reports": reports,
    }


def _format_raw_enriched_chunk(
    document_title: str,
    page_no: int,
    raw_text: str,
    page_start: int,
    page_end: int,
) -> str:
    """Formate un chunk de texte brut enrichi (page complète ou plage de pages)."""
    page_str = f"Page: {page_no}" if page_start == page_end else f"Pages: {page_start}-{page_end}"
    lines = [
        f"Document: {document_title or 'Document'} | {page_str}",
        "",
        raw_text,
    ]
    return "\n".join(lines).strip()


def _format_pro_report_chunk(
    document_title: str,
    page_no: int,
    report: dict,
    page_start: int,
    page_end: int,
) -> str:
    """Formate un chunk rapport pro (page complète ou plage de pages)."""
    page_str = f"Page: {page_no}" if page_start == page_end else f"Pages: {page_start}-{page_end}"
    lines = [
        f"Document: {document_title or 'Document'} | {page_str} | Rapport technique",
    ]
    if report.get("theme"):
        lines.append(f"Thème: {report['theme']}")
    if report.get("references"):
        lines.append(f"Références: {', '.join(report['references'])}")
    if report.get("norms"):
        lines.append(f"Normes: {', '.join(report['norms'])}")
    if report.get("constraints"):
        lines.append(f"Contraintes: {', '.join(report['constraints'])}")
    if report.get("dependencies"):
        lines.append(f"Dépendances: {', '.join(report['dependencies'])}")
    if report.get("keywords"):
        lines.append(f"Mots-clés: {', '.join(report['keywords'])}")
    lines.append("")
    lines.append(report.get("report", "").strip())
    return "\n".join(lines).strip()


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


def build_chunk_specs_from_page(
    document_id: int,
    page_no: int,
    parsed: dict,
    document_title: str,
) -> List[MultimodalChunkSpec]:
    """
    Construit les specs de chunks pour une page :
    - 1 ou plusieurs chunks raw enrichis (texte de la page complète, splitté si >480 tokens)
    - 1 ou plusieurs chunks rapport pro (un par thème/rapport dans pro_reports)
    """
    specs: List[MultimodalChunkSpec] = []
    model_name = _multimodal_page_model()
    
    page_start = parsed.get("page_start", page_no)
    page_end = parsed.get("page_end", page_no)

    # 1. Traitement du raw_text de la page (si non vide)
    raw_text = parsed.get("raw_text", "").strip()
    if raw_text:
        formatted_raw = _format_raw_enriched_chunk(document_title, page_no, raw_text, page_start, page_end)
        raw_parts = split_text_by_tokens(formatted_raw, MAX_CHUNK_TOKENS)
        for part_idx, part_content in enumerate(raw_parts):
            part_content = _enforce_token_limit(
                part_content, page_no, "raw", 0, part_idx
            )
            
            node_suffix = "raw"
            if len(raw_parts) > 1:
                node_suffix = f"raw-part{part_idx + 1}"
                
            node_id = f"multimodal-page-{document_id}-{page_no}-{node_suffix}"
            
            meta = {
                "content_type": PAGE_RAW_ENRICHED_CONTENT_TYPE,
                "chunking_version": CHUNKING_VERSION_MULTIMODAL,
                "page_no": page_no,
                "page_start": page_start,
                "page_end": page_end,
                "generation_method": "pymupdf+mistral_small",
                "llm_model": model_name,
                "document_id": document_id,
                "document_title": document_title or "",
                "is_split": len(raw_parts) > 1,
                "split_part": part_idx + 1 if len(raw_parts) > 1 else None,
                "split_total": len(raw_parts) if len(raw_parts) > 1 else None,
                "token_count": count_tokens(part_content),
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

    # 2. Traitement des pro_reports de la page
    for rep_idx, report in enumerate(parsed.get("pro_reports") or []):
        formatted_report = _format_pro_report_chunk(document_title, page_no, report, page_start, page_end)
        report_parts = split_text_by_tokens(formatted_report, MAX_CHUNK_TOKENS)
        for part_idx, part_content in enumerate(report_parts):
            part_content = _enforce_token_limit(
                part_content, page_no, "report", rep_idx + 1, part_idx
            )
            
            node_suffix = f"report-r{rep_idx + 1}"
            if len(report_parts) > 1:
                node_suffix = f"report-r{rep_idx + 1}-part{part_idx + 1}"
                
            node_id = f"multimodal-page-{document_id}-{page_no}-{node_suffix}"
            
            meta = {
                "content_type": PAGE_SECTION_REPORT_CONTENT_TYPE,
                "chunking_version": CHUNKING_VERSION_MULTIMODAL,
                "page_no": page_no,
                "page_start": page_start,
                "page_end": page_end,
                "report_index": rep_idx + 1,
                "theme": report.get("theme") or "",
                "references": report.get("references") or [],
                "keywords": report.get("keywords") or [],
                "norms": report.get("norms") or [],
                "constraints": report.get("constraints") or [],
                "dependencies": report.get("dependencies") or [],
                "generation_method": "pymupdf+mistral_small",
                "llm_model": model_name,
                "document_id": document_id,
                "document_title": document_title or "",
                "is_split": len(report_parts) > 1,
                "split_part": part_idx + 1 if len(report_parts) > 1 else None,
                "split_total": len(report_parts) if len(report_parts) > 1 else None,
                "token_count": count_tokens(part_content),
            }
            
            specs.append(
                MultimodalChunkSpec(
                    page_no=page_no,
                    content=part_content,
                    content_type=PAGE_SECTION_REPORT_CONTENT_TYPE,
                    node_id=node_id,
                    metadata=meta,
                )
            )

    logger.info(
        "Page %s : %d raw chunk(s) + %d report chunk(s) générés (tous ≤%d tokens)",
        page_no,
        len([s for s in specs if s.content_type == PAGE_RAW_ENRICHED_CONTENT_TYPE]),
        len([s for s in specs if s.content_type == PAGE_SECTION_REPORT_CONTENT_TYPE]),
        MAX_CHUNK_TOKENS,
    )

    return specs


def synthesize_page_with_mistral_small(
    image_png: bytes,
    page_no: int,
    pymupdf_text: str,
    document_title: str,
) -> dict:
    """Appel mistral-small vision → JSON parsé (sections raw_text + pro_report)."""
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
        "response_format": {"type": "json_object"},
    }
    base_url = (settings.MISTRAL_BASE_URL or "https://api.mistral.ai").rstrip("/")
    timeout = float(getattr(settings, "MISTRAL_OCR_TIMEOUT", 300) or 300)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def _call_once(call_payload: dict) -> str:
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
                        json=call_payload,
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
                    logger.warning(
                        "Mistral vision page_no=%s tentative %s/%s: HTTP %s, retry dans %.1fs",
                        page_no, attempt, max_attempts, e.response.status_code, wait
                    )
                    time.sleep(wait)
                    continue
                raise
            except (httpx.RequestError, RuntimeError) as e:
                last_exception = e
                if attempt < max_attempts:
                    wait = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    logger.warning(
                        "Mistral vision page_no=%s tentative %s/%s: Erreur (%s), retry dans %.1fs",
                        page_no, attempt, max_attempts, e, wait
                    )
                    time.sleep(wait)
                    continue
                raise
        if last_exception:
            raise last_exception
        raise RuntimeError("Échec appel Mistral après retries")

    parsed_raw: dict = {}
    raw_content = _call_once(payload)
    try:
        parsed_raw = _parse_json_with_repair(raw_content)
    except (ValueError, json.JSONDecodeError) as exc_first:
        logger.warning(
            "page_no=%s JSON invalide après réparation (%s), retry strict",
            page_no,
            exc_first,
        )
        retry_messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            user_text
                            + "\n\nIMPORTANT: Retourne UNIQUEMENT un JSON strict valide RFC8259. "
                            "Aucun texte hors JSON, pas de trailing comma, guillemets correctement échappés."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{b64}",
                    },
                ],
            },
        ]
        retry_payload = {
            **payload,
            "messages": retry_messages,
            "temperature": 0.0,
        }
        try:
            retry_raw = _call_once(retry_payload)
            parsed_raw = _parse_json_with_repair(retry_raw)
        except (ValueError, json.JSONDecodeError) as exc_retry:
            logger.warning(
                "page_no=%s JSON invalide après retry strict (%s), fallback section unique",
                page_no,
                exc_retry,
            )
            parsed_raw = {}

    return parse_multimodal_page_response(
        parsed_raw, page_no, pymupdf_text, document_title
    )


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
    png = render_pdf_page_png(pdf_path, page_index)
    parsed = synthesize_page_with_mistral_small(
        png, page_no, pymupdf_text, document_title
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
    Fusionne le raw_text et les pro_reports des pages successives si coupure de paragraphe,
    de tableau ou de thème.
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

    # 2. Fusion des pro_reports par thèmes identiques
    for i in range(len(parsed_pages) - 1):
        curr_page = parsed_pages[i]
        next_page = parsed_pages[i + 1]

        reports_curr = curr_page.get("pro_reports") or []
        reports_next = next_page.get("pro_reports") or []

        if not reports_curr or not reports_next:
            continue

        last_report = reports_curr[-1]
        first_report = reports_next[0]

        theme_curr = last_report.get("theme", "").strip().lower()
        theme_next = first_report.get("theme", "").strip().lower()

        if theme_curr == theme_next and theme_curr != "":
            logger.info(
                "Fusion des pro_reports de même thème '%s' entre page %s et %s",
                last_report.get("theme"),
                curr_page["page_no"],
                next_page["page_no"],
            )
            last_report["report"] = last_report.get("report", "").strip() + "\n\n" + first_report.get("report", "").strip()

            for key in ("references", "keywords", "norms", "constraints", "dependencies"):
                combined = list(set(last_report.get(key, []) + first_report.get(key, [])))
                last_report[key] = combined

            next_page["pro_reports"].pop(0)

    return parsed_pages


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


def generate_page_group_summaries(
    document_id: int,
    document_title: str,
    ordered_pages: List[dict]
) -> List[MultimodalChunkSpec]:
    """
    Pass 2 : Génère des résumés contextuels pour des fenêtres glissantes de 3 pages.
    Rejoint l'API Mistral en mode texte.
    """
    specs: List[MultimodalChunkSpec] = []
    if not ordered_pages:
        return specs

    api_key = settings.MISTRAL_API_KEY
    if not api_key:
        logger.warning("MISTRAL_API_KEY manquante, Pass 2 résumé ignoré.")
        return specs

    base_url = (settings.MISTRAL_BASE_URL or "https://api.mistral.ai").rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    window_size = 3
    for i in range(0, len(ordered_pages), window_size):
        batch = ordered_pages[i : i + window_size]
        if not batch:
            continue
        start_page = batch[0]["page_no"]
        end_page = batch[-1]["page_no"]
        
        # Concaténer le texte brut de ce batch
        context_parts = []
        for page in batch:
            page_text = page.get("raw_text", "").strip()
            if page_text:
                context_parts.append(f"--- PAGE {page['page_no']} ---\n{page_text}")
        
        context = "\n\n".join(context_parts).strip()
        if not context:
            continue

        prompt = f"""Tu es un expert en RAG et en analyse technique.
Résume de manière synthétique et dense le contenu technique des pages {start_page} à {end_page} du document "{document_title}".
Concentre-toi sur les thèmes principaux, règles, normes, valeurs numériques clés ou tableaux présentés.
Ce résumé sera utilisé comme contexte additionnel pour aider le RAG à comprendre ce que contient cette section.
Rédige le résumé en 100 à 150 tokens maximum.

Texte des pages :
{context}

Résumé technique (sois concis, pas de blabla d'introduction, réponds en français) :"""

        payload = {
            "model": "mistral-small-latest",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "max_tokens": 300,
            "temperature": 0.1,
        }

        summary_text = ""
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                res_json = resp.json()
                choice = (res_json.get("choices") or [{}])[0]
                summary_text = (choice.get("message", {}).get("content") or "").strip()
        except Exception as e:
            logger.warning(
                "Échec génération résumé Pass 2 pour pages %s-%s du doc %s: %s",
                start_page, end_page, document_id, e
            )
            # Fallback simple
            themes = []
            for p in batch:
                for r in p.get("pro_reports") or []:
                    if r.get("theme"):
                        themes.append(r.get("theme"))
            summary_text = (
                f"Résumé technique des pages {start_page} à {end_page} de {document_title}. "
                f"Sujets couverts : {', '.join(themes[:10])}."
            )

        if summary_text:
            node_id = f"multimodal-parent-{document_id}-p{start_page}-p{end_page}"
            meta = {
                "content_type": PAGE_GROUP_SUMMARY_CONTENT_TYPE,
                "chunking_version": CHUNKING_VERSION_MULTIMODAL,
                "page_start": start_page,
                "page_end": end_page,
                "is_leaf": False,
                "hierarchy_level": 1,
                "document_id": document_id,
                "document_title": document_title,
                "token_count": count_tokens(summary_text)
            }
            specs.append(
                MultimodalChunkSpec(
                    page_no=start_page,
                    content=summary_text,
                    content_type=PAGE_GROUP_SUMMARY_CONTENT_TYPE,
                    node_id=node_id,
                    metadata=meta
                )
            )
            
    return specs


def build_multimodal_pages_for_pdf(
    pdf_path: str,
    document_title: str,
    document_id: int,
    *,
    max_pages: Optional[int] = None,
) -> List[MultimodalChunkSpec]:
    """
    Pour chaque page : PNG + mistral-small JSON → specs de chunks.
    Jusqu'à MULTIMODAL_PAGE_CONCURRENCY pages en parallèle par document.
    Puis fusionne heuristiquement les coupures de page et génère les parents (Pass 2).
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

    # 2. Appliquer la fusion heuristique des sections coupées
    ordered_pages = merge_cut_sections_across_pages(ordered_pages)

    # 3. Générer les résumés parent Pass 2
    parent_specs = generate_page_group_summaries(document_id, document_title, ordered_pages)

    # 4. Générer les specs de chunks feuilles et injecter parent_node_id
    leaf_specs = []
    for page_data in ordered_pages:
        page_no = page_data["page_no"]
        page_specs = build_chunk_specs_from_page(
            document_id, page_no, page_data, document_title
        )
        
        # Associer les feuilles à leur parent correspondant (contenant page_no dans sa plage)
        for spec in page_specs:
            for p_spec in parent_specs:
                start_p = p_spec.metadata.get("page_start")
                end_p = p_spec.metadata.get("page_end")
                if start_p <= page_no <= end_p:
                    spec.metadata["parent_node_id"] = p_spec.node_id
                    break
        
        leaf_specs.extend(page_specs)

    # Retourner les parents suivis des feuilles
    return parent_specs + leaf_specs


def embed_new_multimodal_chunks(document_id: int) -> int:
    """Embeddings mistral-embed pour les chunks multimodal sans vecteur."""
    from app.services.embedding_service import generate_embeddings_batch

    col = _content_type_col()
    with Session(engine) as session:
        statement = select(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.embedding.is_(None),
            or_(*[col == ct for ct in MULTIMODAL_CONTENT_TYPES]),
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
