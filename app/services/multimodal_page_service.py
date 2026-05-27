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
# Legacy (retraitement / suppression)
PAGE_MULTIMODAL_SECTION_CONTENT_TYPE = "page_multimodal_section"
PAGE_MULTIMODAL_SUMMARY_CONTENT_TYPE = "page_multimodal_summary"
CHUNKING_VERSION_MULTIMODAL = "multimodal_page_v3"
MAX_SECTIONS_PER_PAGE = 5
MAX_CHUNK_CHARS = 1500  # Limite caractères (fallback si token count échoue)
MAX_CHUNK_TOKENS = 480  # Limite stricte en tokens (marge sécurité vs 512)

MULTIMODAL_CONTENT_TYPES = (
    PAGE_RAW_ENRICHED_CONTENT_TYPE,
    PAGE_SECTION_REPORT_CONTENT_TYPE,
)
LEGACY_MULTIMODAL_CONTENT_TYPES = (
    PAGE_MULTIMODAL_SECTION_CONTENT_TYPE,
    PAGE_MULTIMODAL_SUMMARY_CONTENT_TYPE,
)
ALL_MULTIMODAL_CONTENT_TYPES = MULTIMODAL_CONTENT_TYPES + LEGACY_MULTIMODAL_CONTENT_TYPES

# Tokenizer lazy-loaded pour comptage tokens compatible MiniLM (BERT-based)
_tokenizer = None
_tokenizer_lock = None


def _get_tokenizer():
    """Charge le tokenizer BERT pour comptage tokens compatible MiniLM."""
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
            # Utiliser le tokenizer du modèle exact pour compatibilité
            model_name = settings.RERANKER_MODEL or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Tokenizer chargé pour comptage tokens : %s", model_name)
        except Exception as e:
            logger.warning("Échec chargement tokenizer %s, fallback approximation : %s", model_name, e)
            _tokenizer = None
    return _tokenizer


def count_tokens(text: str) -> int:
    """
    Compte le nombre de tokens dans un texte avec le tokenizer MiniLM.
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

_SYSTEM_PROMPT = """Tu es un expert en documentation technique, normes et RAG documentaire.
Tu reçois le texte pymupdf d'une page PDF et son image.

Tu dois répondre UNIQUEMENT avec un objet JSON valide (pas de markdown autour), selon ce schéma :
{
  "page_no": <int>,
  "sections": [
    {
      "section_index": <int 1..5>,
      "heading": "<titre de section>",
      "section_kind": "text|table|figure|mixed",
      "raw_text": "<texte brut enrichi : texte source + [Image N: description] inline>",
      "pro_report": "<rapport technique professionnel RAG-friendly pour cette section>",
      "references": ["NF EN ...", "DTU ...", ...],
      "keywords": ["mot-clé", ...],
      "norms": ["..."],
      "constraints": ["..."],
      "dependencies": ["..."],
      "linked_figures": ["Figure N — ...", ...]
    }
  ]
}

Règles de découpage (sections) :
- Produire entre 1 et 5 sections selon la complexité de la page.
- Page simple (un seul bloc) → 1 section.
- Découper par sous-chapitres, blocs logiques distincts ; chaque tableau important = section dédiée (section_kind table).
- Chaque section produit DEUX champs distincts : raw_text ET pro_report.

Champ raw_text (texte brut enrichi — PRIORITÉ retrieval sur contenu source) :
- Extraire TOUT le texte de la section dans l'ordre de lecture, fidèle au document.
- Pour chaque image/schéma/figure significatif, insérer inline une description :
  [Image N: description technique concise avec valeurs, cotes, composants]
- NE PAS ajouter d'interprétation narrative : rester factuel et proche du document source.
- Si tableau : inclure en markdown pipes fidèle.
- Objectif : préserver le contenu original + rendre les visuels cherchables en texte.
- Limite : ~350 tokens par section.

Champ pro_report (rapport technique professionnel par section) :
- RÉINTERPRÉTER la section pour usage RAG (pas un simple résumé).
- Structurer : normes applicables, contraintes chiffrées, procédures, compatibilités, risques.
- Vocabulaire technique précis, formulations actionnables, termes-clés répétés pour retrieval.
- Format rapport pro avec listes à puces, valeurs numériques + unités.
- Limite : ~350 tokens par section.
- Un pro_report PAR section (pas de synthèse globale de page).

Général : français ; n'invente rien ; « illisible » si zone floue ; si pymupdf vide, base-toi sur l'image."""

_USER_PROMPT_TEMPLATE = """Document : {title}
Page : {page_no}

Texte pymupdf :
---
{pymupdf_text}
---

Analyse l'image et le texte. Retourne le JSON structuré (sections avec raw_text + pro_report par section)."""


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


def _normalize_section_dict(sec: dict, index: int, pymupdf_text: str, document_title: str, page_no: int) -> Optional[dict]:
    """Normalise une section LLM (v3 raw_text/pro_report ou legacy content)."""
    if not isinstance(sec, dict):
        return None

    raw_text = (sec.get("raw_text") or sec.get("content") or "").strip()
    pro_report = (sec.get("pro_report") or "").strip()
    heading = (sec.get("heading") or f"Section {index + 1}").strip()

    if not raw_text and not pro_report:
        return None

    if not raw_text:
        raw_text = pymupdf_text.strip() or f"(contenu non extractible — section {heading})"

    if not pro_report:
        pro_report = (
            f"Rapport technique — {heading} (page {page_no}, {document_title or 'Document'}). "
            f"Section documentaire à consulter pour les détails techniques."
        )

    return {
        "section_index": int(sec.get("section_index") or index + 1),
        "heading": heading,
        "section_kind": (sec.get("section_kind") or "text").strip() or "text",
        "raw_text": raw_text,
        "pro_report": pro_report,
        "references": _as_str_list(sec.get("references")),
        "keywords": _as_str_list(sec.get("keywords")),
        "norms": _as_str_list(sec.get("norms")),
        "constraints": _as_str_list(sec.get("constraints")),
        "dependencies": _as_str_list(sec.get("dependencies")),
        "linked_figures": _as_str_list(sec.get("linked_figures")),
    }


def parse_multimodal_page_response(
    data: dict,
    page_no: int,
    pymupdf_text: str,
    document_title: str,
) -> dict:
    """
    Valide et normalise la réponse JSON du LLM.
    Fallback : 1 section avec raw_text + pro_report minimaux si invalide.
    """
    sections_in = data.get("sections")
    if not isinstance(sections_in, list) or not sections_in:
        fallback_content = pymupdf_text.strip() or "(contenu non extractible — voir image)"
        return {
            "page_no": page_no,
            "sections": [
                {
                    "section_index": 1,
                    "heading": f"Page {page_no}",
                    "section_kind": "mixed",
                    "raw_text": fallback_content,
                    "pro_report": (
                        f"Rapport technique page {page_no} — {document_title or 'Document'}. "
                        f"Contenu principal extrait de la page."
                    ),
                    "references": [],
                    "keywords": [],
                    "norms": [],
                    "constraints": [],
                    "dependencies": [],
                    "linked_figures": [],
                }
            ],
        }

    sections: List[dict] = []
    for i, sec in enumerate(sections_in[:MAX_SECTIONS_PER_PAGE]):
        normalized = _normalize_section_dict(sec, i, pymupdf_text, document_title, page_no)
        if normalized:
            sections.append(normalized)

    if len(sections_in) > MAX_SECTIONS_PER_PAGE:
        logger.warning(
            "page_no=%s : %s sections proposées, tronqué à %s",
            page_no,
            len(sections_in),
            MAX_SECTIONS_PER_PAGE,
        )

    if not sections:
        return parse_multimodal_page_response({}, page_no, pymupdf_text, document_title)

    return {
        "page_no": page_no,
        "sections": sections,
    }


def _format_raw_enriched_chunk(
    document_title: str,
    page_no: int,
    section: dict,
) -> str:
    """Formate un chunk de texte brut enrichi (minimal metadata overhead)."""
    lines = [
        f"Document: {document_title or 'Document'} | Page: {page_no}",
        f"Section: {section.get('heading', 'Section')}",
        "",
        section.get("raw_text", ""),
    ]
    formatted = "\n".join(lines).strip()
    token_count = count_tokens(formatted)
    if token_count > MAX_CHUNK_TOKENS:
        logger.warning(
            "Raw enriched page_no=%s section_index=%s dépasse MAX_CHUNK_TOKENS (%d > %d), sera splitté",
            page_no,
            section.get("section_index"),
            token_count,
            MAX_CHUNK_TOKENS,
        )
    return formatted


def _format_pro_report_chunk(
    document_title: str,
    page_no: int,
    section: dict,
) -> str:
    """Formate un chunk rapport pro (métadonnées riches)."""
    lines = [
        f"Document: {document_title or 'Document'} | Page: {page_no} | Rapport technique",
        f"Section: {section.get('heading', 'Section')}",
    ]
    if section.get("references"):
        lines.append(f"Références: {', '.join(section['references'])}")
    if section.get("norms"):
        lines.append(f"Normes: {', '.join(section['norms'])}")
    if section.get("constraints"):
        lines.append(f"Contraintes: {', '.join(section['constraints'])}")
    if section.get("dependencies"):
        lines.append(f"Dépendances: {', '.join(section['dependencies'])}")
    if section.get("keywords"):
        lines.append(f"Mots-clés: {', '.join(section['keywords'])}")
    lines.append("")
    lines.append(section.get("pro_report", ""))
    formatted = "\n".join(lines).strip()
    token_count = count_tokens(formatted)
    if token_count > MAX_CHUNK_TOKENS:
        logger.warning(
            "Pro report page_no=%s section_index=%s dépasse MAX_CHUNK_TOKENS (%d > %d), sera splitté",
            page_no,
            section.get("section_index"),
            token_count,
            MAX_CHUNK_TOKENS,
        )
    return formatted


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


def _append_split_specs(
    specs: List[MultimodalChunkSpec],
    *,
    document_id: int,
    page_no: int,
    section: dict,
    sec_idx: int,
    content: str,
    content_type: str,
    node_suffix: str,
    model_name: str,
    document_title: str,
    extra_meta: Optional[dict] = None,
) -> None:
    """Ajoute les specs d'un contenu splitté à 480 tokens."""
    parts = split_text_by_tokens(content, MAX_CHUNK_TOKENS)
    label = "raw" if content_type == PAGE_RAW_ENRICHED_CONTENT_TYPE else "report"

    for part_idx, content_part in enumerate(parts):
        content_part = _enforce_token_limit(
            content_part, page_no, label, sec_idx, part_idx
        )
        effective_sec_idx = (
            f"{sec_idx}" if len(parts) == 1 else f"{sec_idx}.{part_idx + 1}"
        )
        node_part = "" if len(parts) == 1 else f"-part{part_idx + 1}"
        meta = {
            "content_type": content_type,
            "chunking_version": CHUNKING_VERSION_MULTIMODAL,
            "page_no": page_no,
            "page_start": page_no,
            "page_end": page_no,
            "section_index": sec_idx,
            "section_heading": section.get("heading") or "",
            "section_kind": section.get("section_kind") or "text",
            "references": section.get("references") or [],
            "keywords": section.get("keywords") or [],
            "linked_figures": section.get("linked_figures") or [],
            "generation_method": "pymupdf+mistral_small",
            "llm_model": model_name,
            "document_id": document_id,
            "document_title": document_title or "",
            "is_split": len(parts) > 1,
            "split_part": part_idx + 1 if len(parts) > 1 else None,
            "split_total": len(parts) if len(parts) > 1 else None,
            "token_count": count_tokens(content_part),
        }
        if content_type == PAGE_SECTION_REPORT_CONTENT_TYPE:
            meta["norms"] = section.get("norms") or []
            meta["constraints"] = section.get("constraints") or []
            meta["dependencies"] = section.get("dependencies") or []
        if extra_meta:
            meta.update(extra_meta)

        specs.append(
            MultimodalChunkSpec(
                page_no=page_no,
                content=content_part,
                content_type=content_type,
                node_id=(
                    f"multimodal-page-{document_id}-{page_no}-s{effective_sec_idx}-{node_suffix}{node_part}"
                ),
                metadata=meta,
            )
        )


def build_chunk_specs_from_page(
    document_id: int,
    page_no: int,
    parsed: dict,
    document_title: str,
) -> List[MultimodalChunkSpec]:
    """
    Construit les specs de chunks pour une page :
    - 1 chunk raw enrichi par section (texte source + images inline)
    - 1 chunk rapport pro par section
    Garantit que tous les chunks sont ≤ MAX_CHUNK_TOKENS (480 tokens).
    """
    specs: List[MultimodalChunkSpec] = []
    model_name = _multimodal_page_model()

    for section in parsed.get("sections") or []:
        sec_idx = int(section.get("section_index") or len(specs) + 1)

        raw_content = _format_raw_enriched_chunk(document_title, page_no, section)
        _append_split_specs(
            specs,
            document_id=document_id,
            page_no=page_no,
            section=section,
            sec_idx=sec_idx,
            content=raw_content,
            content_type=PAGE_RAW_ENRICHED_CONTENT_TYPE,
            node_suffix="raw",
            model_name=model_name,
            document_title=document_title,
        )

        report_content = _format_pro_report_chunk(document_title, page_no, section)
        _append_split_specs(
            specs,
            document_id=document_id,
            page_no=page_no,
            section=section,
            sec_idx=sec_idx,
            content=report_content,
            content_type=PAGE_SECTION_REPORT_CONTENT_TYPE,
            node_suffix="report",
            model_name=model_name,
            document_title=document_title,
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
    Persiste les chunks multimodal (sans embeddings).
    
    STRATÉGIE PARENT/LEAF verrouillée (Option A):
    - Tous les chunks sont des LEAFS autonomes (is_leaf=True, parent_node_id=None, hierarchy_level=0)
    - Optimisation pour reranker MiniLM (max_length=512 tokens)
    """
    base_index = _next_chunk_index(session, document.id)
    chunks: List[DocumentChunk] = []

    for offset, spec in enumerate(chunk_specs):
        if not spec.content or not spec.content.strip():
            continue
        
        # ASSERT: Stratégie Option A - tous les chunks multimodaux doivent être des leafs
        assert spec.metadata.get("content_type") in MULTIMODAL_CONTENT_TYPES, (
            f"Type de chunk multimodal invalide: {spec.metadata.get('content_type')}"
        )
        
        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=base_index + offset,
            content=spec.content.strip(),
            text=spec.content.strip(),
            start_char=0,
            end_char=len(spec.content),
            node_id=spec.node_id,
            parent_node_id=None,  # Option A: pas de parent hiérarchique
            is_leaf=True,  # Option A: tous les chunks multimodaux sont des leafs
            hierarchy_level=0,  # Option A: pas de hiérarchie
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
) -> Tuple[int, List[MultimodalChunkSpec]]:
    """Traite une page : PNG + mistral-small → specs (appelable en parallèle)."""
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
    page_specs = build_chunk_specs_from_page(
        document_id, page_no, parsed, document_title
    )
    elapsed = time.perf_counter() - t0
    n_raw = sum(
        1 for s in page_specs if s.content_type == PAGE_RAW_ENRICHED_CONTENT_TYPE
    )
    n_report = sum(
        1 for s in page_specs if s.content_type == PAGE_SECTION_REPORT_CONTENT_TYPE
    )
    logger.info(
        "multimodal page_no=%s %.2fs — %s raw + %s report chunk(s)",
        page_no,
        elapsed,
        n_raw,
        n_report,
    )
    return page_no, page_specs


def build_multimodal_pages_for_pdf(
    pdf_path: str,
    document_title: str,
    document_id: int,
    *,
    max_pages: Optional[int] = None,
) -> List[MultimodalChunkSpec]:
    """
    Pour chaque page : pymupdf + PNG + mistral-small JSON → specs de chunks.
    Jusqu'à MULTIMODAL_PAGE_CONCURRENCY pages en parallèle par document.
    """
    from app.services.pdf_extraction_service import extract_page_texts_from_pdf

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

    total = len(page_texts)
    if total == 0:
        return []

    workers = max(1, min(settings.MULTIMODAL_PAGE_CONCURRENCY, total))
    page_specs_by_no: dict[int, List[MultimodalChunkSpec]] = {}

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
        for future in as_completed(futures):
            page_no, specs = future.result()
            page_specs_by_no[page_no] = specs

    all_specs: List[MultimodalChunkSpec] = []
    for page_no, _ in page_texts:
        all_specs.extend(page_specs_by_no[page_no])
    return all_specs


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
