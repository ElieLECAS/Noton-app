"""
Extraction de chunks documentaires via Mistral vision (Ministral 3B).

Pipeline par page :
  1. Rendu PNG via render_page_png_cached
  2. Appel Ministral 3B → JSON structuré { chunks: [...] }
  3. Validation Pydantic + guard tokens (cap 480)
  4. Fallback pymupdf4llm si l'API échoue

Après collecte de toutes les pages :
  5. merge_cross_page_chunks : recollage des passages coupés entre page N et N+1

CHUNKING_VERSION = "vision_page_v1"
"""
from __future__ import annotations

import base64
import logging
import re
import uuid
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError

from app.config import settings

logger = logging.getLogger(__name__)

CHUNKING_VERSION = "vision_page_v1"
EXTRACTION_PROVIDER_VISION = "mistral_vision"
EXTRACTION_PROVIDER_FALLBACK = "pymupdf4llm_fallback"

# Connecteurs de continuité textuelle qui indiquent une coupure de page
_CONTINUITY_CONNECTORS = re.compile(
    r"^(et |puis |afin de |de plus |par ailleurs |également |en outre |or |mais |ainsi |donc |car |"
    r"cependant |toutefois |néanmoins |enfin |d'abord |ensuite |enfin )",
    re.IGNORECASE,
)

# -----------------------------------------------------------------------
# Schémas Pydantic pour la réponse JSON du LLM
# -----------------------------------------------------------------------


class VisionChunk(BaseModel):
    heading: Optional[str] = Field(default=None)
    step_number: Optional[int] = Field(default=None)
    section_type: str = Field(default="section")
    content: str
    continues_on_next_page: bool = Field(default=False)
    continues_from_previous_page: bool = Field(default=False)


class VisionPageResponse(BaseModel):
    page_no: int
    chunks: List[VisionChunk]


# -----------------------------------------------------------------------
# Prompt système
# -----------------------------------------------------------------------

_SYSTEM_PROMPT = """Tu es un assistant d'extraction documentaire technique.
On te donne l'image d'une page de document (notice technique, catalogue, procédure).
Ton travail est de lire TOUTE la page (texte, schémas, tableaux, légendes) et de la découper en chunks sémantiques pour un système RAG.

Règles impératives :
1. Renvoie UNIQUEMENT un objet JSON valide (aucun texte hors JSON).
2. Chaque chunk doit être auto-suffisant pour le RAG : inclure le contexte nécessaire à sa compréhension (ne pas écrire "voir ci-dessus" ou "celui-ci").
3. Chaque "content" doit faire au maximum 480 tokens Mistral (~1800 caractères).
4. Décris les éléments visuels (schémas, assemblages) en texte dans le content du chunk concerné (ex. "le schéma montre Gâche OB Droite à gauche et Gâche OB Gauche à droite").
5. Inclure les légendes des images dans le chunk de l'étape/section associée.
6. Ne pas inclure de bruit markdown : pas de "**texte**", pas de "<!-- page:N -->", pas de "==> picture omitted".
7. Si le texte d'une section est coupé en bas de page (phrase inachevée ou étape non terminée), mettre "continues_on_next_page": true sur ce chunk.
8. Si le premier chunk de la page reprend un texte commencé sur la page précédente, mettre "continues_from_previous_page": true.

Types de section_type possibles : document_header | step | section | table | legend

Format de réponse OBLIGATOIRE :
{
  "page_no": <numéro de page>,
  "chunks": [
    {
      "heading": "<titre court ou null>",
      "step_number": <entier ou null>,
      "section_type": "<type>",
      "content": "<texte auto-suffisant>",
      "continues_on_next_page": false,
      "continues_from_previous_page": false
    }
  ]
}"""

_USER_PROMPT_TEMPLATE = (
    "Document : {title}\nPage : {page_no}\n\n"
    "Extrait tous les chunks sémantiques de cette page selon les règles du système."
)

_VISION_COMPACT_RETRY_SUFFIX = (
    "\n\nIMPORTANT : JSON strictement valide et complet. "
    "Chunks concis (content court, auto-suffisant). Maximum 10 chunks par page."
)


# -----------------------------------------------------------------------
# Appel API vision
# -----------------------------------------------------------------------


def _call_vision_api(
    image_b64: str,
    page_no: int,
    document_title: str,
    *,
    compact_retry: bool = False,
) -> dict:
    """
    Appelle l'API Mistral vision sur une page PNG (base64) et retourne le JSON parsé.
    Lève ValueError si la réponse est vide ou non parsable après retries.
    """
    from app.services.multimodal_page_service import (
        _mistral_chat_completion,
        _parse_json_with_repair,
    )

    user_text = _USER_PROMPT_TEMPLATE.format(
        title=document_title or "Document",
        page_no=page_no,
    )
    if compact_retry:
        user_text += _VISION_COMPACT_RETRY_SUFFIX

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        },
    ]

    raw = _mistral_chat_completion(
        messages,
        page_no=page_no,
        max_tokens=settings.PAGE_EXTRACTION_MAX_TOKENS,
        temperature=0.0,
        response_format_json=True,
        timeout_seconds=settings.PAGE_EXTRACTION_TIMEOUT,
        model=settings.PAGE_EXTRACTION_MODEL,
    )
    return _parse_json_with_repair(raw)


# -----------------------------------------------------------------------
# Validation et normalisation des specs
# -----------------------------------------------------------------------


def _validate_and_normalize(
    raw_response: dict,
    page_no: int,
    metadata_base: dict,
) -> List[dict]:
    """
    Valide la réponse JSON du LLM via Pydantic, cap les chunks > 480 tokens,
    et retourne des specs compatibles DocumentChunk.
    """
    from app.services.multimodal_page_service import count_tokens, split_text_by_tokens

    max_tokens = settings.PAGE_EXTRACTION_MAX_CHUNK_TOKENS
    max_chunks = settings.PAGE_EXTRACTION_MAX_CHUNKS_PER_PAGE

    try:
        response = VisionPageResponse.model_validate(raw_response)
    except ValidationError as exc:
        logger.warning("[VisionExtract] page %s — validation Pydantic échouée : %s", page_no, exc)
        raise ValueError(f"JSON invalide page {page_no}") from exc

    valid_chunks = [c for c in response.chunks if c.content and c.content.strip()]
    if not valid_chunks:
        raise ValueError(f"Aucun chunk valide retourné pour la page {page_no}")

    specs: List[dict] = []
    chunk_counter = 0

    for chunk in valid_chunks[:max_chunks]:
        content = chunk.content.strip()
        token_count = count_tokens(content)

        if token_count > max_tokens:
            parts = split_text_by_tokens(content, max_tokens=max_tokens)
        else:
            parts = [content]

        for part_idx, part in enumerate(parts):
            if not part.strip():
                continue
            if chunk_counter >= max_chunks:
                break

            node_id = str(uuid.uuid4())
            meta = dict(metadata_base)
            meta.update(
                {
                    "chunking_version": CHUNKING_VERSION,
                    "extraction_provider": EXTRACTION_PROVIDER_VISION,
                    "extraction_model": settings.PAGE_EXTRACTION_MODEL,
                    "content_type": "semantic_leaf",
                    "section_type": chunk.section_type or "section",
                    "page_no": page_no,
                    "page_start": page_no,
                    "page_end": page_no,
                    "token_count": count_tokens(part),
                    "is_leaf": True,
                    "node_id": node_id,
                    "parent_node_id": None,
                }
            )
            if chunk.heading:
                meta["heading"] = chunk.heading
                meta["parent_heading"] = chunk.heading
            if chunk.step_number is not None:
                meta["step_number"] = chunk.step_number
            # Flags de coupure (seulement sur le dernier / premier part si split)
            if part_idx == len(parts) - 1:
                meta["continues_on_next_page"] = chunk.continues_on_next_page
            if part_idx == 0:
                meta["continues_from_previous_page"] = chunk.continues_from_previous_page

            specs.append(
                {
                    "chunk_index": 0,
                    "is_leaf": True,
                    "content": part,
                    "text": part,
                    "start_char": 0,
                    "end_char": len(part),
                    "node_id": node_id,
                    "parent_node_id": None,
                    "hierarchy_level": 1,
                    "metadata_json": meta,
                }
            )
            chunk_counter += 1

    return specs


# -----------------------------------------------------------------------
# Fallback pymupdf4llm
# -----------------------------------------------------------------------


def _fallback_pymupdf4llm(
    pdf_path: str,
    page_no: int,
    metadata_base: dict,
) -> List[dict]:
    """
    Fallback par page : extraction pymupdf4llm + chunking sémantique existant.
    Marque les specs avec extraction_provider=pymupdf4llm_fallback.
    """
    from app.services.pdf_extraction_service import extract_page_texts_from_pdf
    from app.services.chunking_service import chunk_pymupdf4llm_page_clean

    pages = extract_page_texts_from_pdf(pdf_path)
    page_text = ""
    for pno, text in pages:
        if pno == page_no:
            page_text = text
            break

    meta_base = dict(metadata_base)
    specs = chunk_pymupdf4llm_page_clean(page_text, page_no, meta_base)
    for spec in specs:
        m = spec.get("metadata_json") or {}
        m["extraction_provider"] = EXTRACTION_PROVIDER_FALLBACK
        m["chunking_version"] = CHUNKING_VERSION
        spec["metadata_json"] = m
    return specs


# -----------------------------------------------------------------------
# Point d'entrée page unique
# -----------------------------------------------------------------------


def extract_page_chunk_specs(
    pdf_path: str,
    page_no: int,
    document_title: str,
    metadata_base: dict,
) -> List[dict]:
    """
    Extrait les chunks sémantiques d'une page PDF via Mistral vision.
    Fallback pymupdf4llm si l'API échoue.

    Retourne une liste de specs DocumentChunk-compatibles (semantic_leaf).
    """
    from app.services.multimodal_page_service import render_page_png_cached

    dpi = settings.PAGE_EXTRACTION_DPI
    try:
        png = render_page_png_cached(pdf_path, page_no, dpi=dpi)
        image_b64 = base64.b64encode(png).decode("ascii")
    except Exception as exc:
        logger.warning(
            "[VisionExtract] page %s — rendu PNG échoué (%s), fallback pymupdf4llm", page_no, exc
        )
        return _fallback_pymupdf4llm(pdf_path, page_no, metadata_base)

    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            raw = _call_vision_api(
                image_b64,
                page_no,
                document_title,
                compact_retry=attempt > 0,
            )
            specs = _validate_and_normalize(raw, page_no, metadata_base)
            logger.info(
                "[VisionExtract] page %s — %s chunks vision (%s)%s",
                page_no,
                len(specs),
                settings.PAGE_EXTRACTION_MODEL,
                " [retry]" if attempt > 0 else "",
            )
            return specs
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                logger.warning(
                    "[VisionExtract] page %s — tentative 1 échouée (%s), retry compact",
                    page_no,
                    exc,
                )

    logger.warning(
        "[VisionExtract] page %s — API vision échouée (%s), fallback pymupdf4llm",
        page_no,
        last_exc,
    )
    return _fallback_pymupdf4llm(pdf_path, page_no, metadata_base)


# -----------------------------------------------------------------------
# Merge inter-pages (post-traitement Python)
# -----------------------------------------------------------------------


def _should_merge(last_spec: dict, first_spec: dict) -> bool:
    """
    Détermine si le dernier chunk de la page N doit être fusionné
    avec le premier chunk de la page N+1.
    """
    last_meta = last_spec.get("metadata_json") or {}
    first_meta = first_spec.get("metadata_json") or {}

    # Signal LLM explicite
    if last_meta.get("continues_on_next_page"):
        return True
    if first_meta.get("continues_from_previous_page"):
        return True

    # Phrase incomplète en bas de page
    last_content = (last_spec.get("content") or "").strip()
    if len(last_content) > 40 and not last_content[-1] in ".!?:":
        return True

    # Même numéro d'étape des deux côtés
    last_step = last_meta.get("step_number")
    first_step = first_meta.get("step_number")
    if last_step is not None and last_step == first_step:
        return True

    # Même section_type + headings compatibles
    last_type = last_meta.get("section_type", "")
    first_type = first_meta.get("section_type", "")
    if last_type == first_type and last_type not in ("document_header",):
        last_heading = (last_meta.get("heading") or "").strip().lower()
        first_heading = (first_meta.get("heading") or "").strip().lower()
        if last_heading and first_heading and (
            last_heading == first_heading
            or first_heading.startswith(last_heading[:20])
        ):
            return True

    # Continuité textuelle : premier mot de la page suivante est connecteur
    first_content = (first_spec.get("content") or "").strip()
    if _CONTINUITY_CONNECTORS.match(first_content):
        return True
    # Commence par minuscule (continuation de phrase)
    if first_content and first_content[0].islower():
        return True

    return False


def _incompatible_types(last_spec: dict, first_spec: dict) -> bool:
    """
    Certaines combinaisons de section_type ne doivent jamais être fusionnées.
    """
    last_meta = last_spec.get("metadata_json") or {}
    first_meta = first_spec.get("metadata_json") or {}
    last_type = last_meta.get("section_type", "")
    first_type = first_meta.get("section_type", "")
    last_step = last_meta.get("step_number")
    first_step = first_meta.get("step_number")

    # En-tête document + quoi que ce soit → pas de fusion
    if "document_header" in (last_type, first_type):
        return True

    # Étapes différentes (sans flag LLM explicite) → pas de fusion
    if (
        last_step is not None
        and first_step is not None
        and last_step != first_step
        and not last_meta.get("continues_on_next_page")
        and not first_meta.get("continues_from_previous_page")
    ):
        return True

    return False


def _merge_two_specs(last_spec: dict, first_spec: dict) -> dict:
    """
    Fusionne deux specs. Conserve le heading/step du chunk amont.
    """
    from app.services.multimodal_page_service import count_tokens

    last_meta = dict(last_spec.get("metadata_json") or {})
    first_meta = dict(first_spec.get("metadata_json") or {})

    merged_content = (last_spec.get("content") or "").rstrip() + "\n\n" + (first_spec.get("content") or "").lstrip()
    merged_content = merged_content.strip()

    page_start = last_meta.get("page_start") or last_meta.get("page_no", 0)
    page_end = first_meta.get("page_end") or first_meta.get("page_no", 0)

    merged_meta = dict(last_meta)
    merged_meta["page_end"] = page_end
    merged_meta["page_no"] = page_start
    merged_meta["cross_page_merge"] = True
    merged_meta["merged_pages"] = [page_start, page_end]
    merged_meta["token_count"] = count_tokens(merged_content)
    merged_meta["continues_on_next_page"] = first_meta.get("continues_on_next_page", False)
    # Nettoyer les flags inter-pages sur le chunk fusionné
    merged_meta.pop("continues_from_previous_page", None)

    return {
        "chunk_index": last_spec.get("chunk_index", 0),
        "is_leaf": True,
        "content": merged_content,
        "text": merged_content,
        "start_char": 0,
        "end_char": len(merged_content),
        "node_id": str(uuid.uuid4()),
        "parent_node_id": last_spec.get("parent_node_id"),
        "hierarchy_level": 1,
        "metadata_json": merged_meta,
    }


def merge_cross_page_chunks(specs: List[dict]) -> List[dict]:
    """
    Post-traitement Python : fusionne les chunks coupés entre page N et page N+1.

    - Travaille en ordre page_no croissant (les specs doivent déjà être ordonnées).
    - Fusionne last(page N) + first(page N+1) si les signaux de coupure le justifient.
    - Re-split si le chunk fusionné dépasse PAGE_EXTRACTION_MAX_CHUNK_TOKENS.
    - Retourne la liste finale (nouveaux node_id pour les merges).
    """
    from app.services.multimodal_page_service import count_tokens, split_text_by_tokens

    if len(specs) <= 1:
        return specs

    max_tokens = settings.PAGE_EXTRACTION_MAX_CHUNK_TOKENS
    result: List[dict] = [specs[0]]

    for i in range(1, len(specs)):
        candidate = specs[i]
        last = result[-1]

        last_page = (last.get("metadata_json") or {}).get("page_no", -1)
        curr_page = (candidate.get("metadata_json") or {}).get("page_no", -1)
        is_adjacent_page = (
            isinstance(last_page, int)
            and isinstance(curr_page, int)
            and curr_page == last_page + 1
        )

        if (
            is_adjacent_page
            and not _incompatible_types(last, candidate)
            and _should_merge(last, candidate)
        ):
            merged = _merge_two_specs(last, candidate)
            token_count = count_tokens(merged["content"])

            if token_count > max_tokens:
                # Re-split le chunk fusionné
                parts = split_text_by_tokens(merged["content"], max_tokens=max_tokens)
                merged_meta = merged.get("metadata_json") or {}
                first_part = True
                for part in parts:
                    if not part.strip():
                        continue
                    part_spec = dict(merged)
                    part_spec["content"] = part
                    part_spec["text"] = part
                    part_spec["end_char"] = len(part)
                    part_spec["node_id"] = str(uuid.uuid4())
                    part_meta = dict(merged_meta)
                    part_meta["token_count"] = count_tokens(part)
                    if not first_part:
                        part_meta["cross_page_merge"] = True
                    part_spec["metadata_json"] = part_meta
                    if first_part:
                        result[-1] = part_spec
                        first_part = False
                    else:
                        result.append(part_spec)
            else:
                result[-1] = merged
        else:
            result.append(candidate)

    return result
