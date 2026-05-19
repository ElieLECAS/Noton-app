"""
Utilitaires partagés pour metadata_json / metadata_ (bibliothèque et notes).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

_CHUNK_T = TypeVar("_CHUNK_T")

PAGE_TECHNICAL_SHEET_CONTENT_TYPE = "page_technical_sheet"
KAG_ENTITIES_SOURCE_PAGE_SHEET = "page_sheet"
PAGE_SHEET_NODE_ID_PREFIX = "page-sheet"


def _coerce_positive_page(value: Any) -> Optional[int]:
    """Convertit une valeur en numéro de page (>0), sinon None."""
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _pages_from_provenance_list(prov: Any) -> List[int]:
    """Extrait les numéros de page d'une liste de provenances Docling."""
    pages: List[int] = []
    if not isinstance(prov, list):
        return pages
    for item in prov:
        if not isinstance(item, dict):
            continue
        for key in ("page_no", "page_idx", "page", "page_number"):
            page = _coerce_positive_page(item.get(key))
            if page is not None:
                pages.append(page)
    return pages


def extract_page_numbers_from_docling_metadata(meta: Optional[dict]) -> List[int]:
    """
    Collecte tous les numéros de page (>0) depuis les métadonnées Docling.
    Parcourt les clés top-level et doc_items[].prov[].
    """
    if not meta:
        return []
    pages: List[int] = []
    for key in ("page_no", "page", "page_number", "page_idx", "page_label"):
        page = _coerce_positive_page(meta.get(key))
        if page is not None:
            pages.append(page)
    for item in meta.get("doc_items") or meta.get("doc_items_refs") or []:
        if not isinstance(item, dict):
            continue
        pages.extend(_pages_from_provenance_list(item.get("prov")))
    return pages


def resolve_page_range_from_metadata(
    meta: Optional[dict],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Retourne (page_no, page_start, page_end) à partir des métadonnées Docling ou dérivées.
    page_no et page_start valent le minimum des pages trouvées ; page_end le maximum.
    """
    if not meta:
        return None, None, None

    existing_start = _coerce_positive_page(meta.get("page_start"))
    existing_end = _coerce_positive_page(meta.get("page_end"))
    existing_no = _coerce_positive_page(meta.get("page_no"))

    extracted = extract_page_numbers_from_docling_metadata(meta)
    if extracted:
        page_start = min(extracted)
        page_end = max(extracted)
        page_no = page_start
    else:
        page_start = existing_start
        page_end = existing_end
        page_no = existing_no or page_start

    if page_start is None and page_no is not None:
        page_start = page_no
    if page_end is None and page_no is not None:
        page_end = page_no
    if page_no is None and page_start is not None:
        page_no = page_start

    return page_no, page_start, page_end


def resolve_page_from_metadata(metadata: Optional[dict]) -> Optional[int]:
    """Première page utile (>0) depuis les métadonnées Docling ou dérivées."""
    page_no, _, _ = resolve_page_range_from_metadata(metadata)
    return page_no


def enrich_docling_page_metadata(meta: dict) -> dict:
    """
    Promouvoit page_no / page_start / page_end au premier niveau si présents
    dans doc_items[].prov[] (sans écraser des valeurs top-level déjà valides).
    """
    if not meta:
        return meta
    page_no, page_start, page_end = resolve_page_range_from_metadata(meta)
    if page_no is not None:
        meta.setdefault("page_no", page_no)
    if page_start is not None:
        meta.setdefault("page_start", page_start)
    if page_end is not None:
        meta.setdefault("page_end", page_end)
    return meta


def page_sheet_node_id(document_id: int, page_no: int) -> str:
    """Identifiant stable d'une fiche technique page (chunk parent KAG)."""
    return f"{PAGE_SHEET_NODE_ID_PREFIX}-{document_id}-{page_no}"


def chunk_page_group_key(metadata: Optional[dict]) -> Optional[int]:
    """
    Clé de regroupement par page pour KAG.
    Utilise page_start (via resolve_page_range) ; les plages multi-pages
    sont rattachées à la page de début uniquement.
    """
    page_no, _, _ = resolve_page_range_from_metadata(metadata)
    return page_no


def _chunk_sort_key(chunk: _CHUNK_T) -> Tuple[int, int, int]:
    idx = getattr(chunk, "chunk_index", 0) or 0
    start = getattr(chunk, "start_char", 0) or 0
    cid = getattr(chunk, "id", 0) or 0
    return (int(idx), int(start), int(cid))


def group_chunks_by_page_no(
    chunks: Sequence[_CHUNK_T],
) -> Tuple[Dict[int, List[_CHUNK_T]], List[_CHUNK_T]]:
    """
    Regroupe les chunks feuilles par numéro de page (>0).

    Returns:
        (by_page, without_page) — feuilles triées par chunk_index dans chaque page.
    """
    by_page: Dict[int, List[_CHUNK_T]] = {}
    without_page: List[_CHUNK_T] = []

    for chunk in chunks:
        meta = merged_chunk_metadata(
            getattr(chunk, "metadata_json", None),
            getattr(chunk, "metadata_", None),
        )
        page_key = chunk_page_group_key(meta)
        if page_key is None:
            without_page.append(chunk)
            continue
        by_page.setdefault(page_key, []).append(chunk)

    for page_no in by_page:
        by_page[page_no] = sorted(by_page[page_no], key=_chunk_sort_key)
    without_page.sort(key=_chunk_sort_key)
    return by_page, without_page


def assemble_page_text_from_chunks(chunks: Sequence[_CHUNK_T]) -> str:
    """Concatène le contenu des feuilles d'une page (ordre chunk_index / start_char)."""
    parts: List[str] = []
    for chunk in sorted(chunks, key=_chunk_sort_key):
        text = (getattr(chunk, "content", None) or getattr(chunk, "text", None) or "").strip()
        if text:
            parts.append(text)
    return "\n\n---\n\n".join(parts)


def dominant_parent_heading_for_chunks(chunks: Sequence[_CHUNK_T]) -> Optional[str]:
    """Heading de section le plus fréquent parmi les feuilles d'une page."""
    counts: Dict[str, int] = {}
    for chunk in chunks:
        meta = merged_chunk_metadata(
            getattr(chunk, "metadata_json", None),
            getattr(chunk, "metadata_", None),
        )
        heading = (meta.get("parent_heading") or meta.get("heading") or "").strip()
        if heading and heading != "__no_heading__":
            counts[heading] = counts.get(heading, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def merged_chunk_metadata(
    primary: Optional[dict],
    legacy: Optional[dict],
) -> Dict[str, Any]:
    """Fusionne metadata_json (primary) et metadata_ (legacy). JSON prime."""
    merged: Dict[str, Any] = {}
    if isinstance(legacy, dict):
        merged.update(legacy)
    if isinstance(primary, dict):
        merged.update(primary)
    return merged


def meta_is_leaf(
    metadata: Optional[dict],
    column_value: Optional[bool] = None,
) -> bool:
    """
    Interprète is_leaf depuis la colonne SQL ou le JSON (bool ou \"true\"/\"false\").
    """
    if column_value is not None:
        return bool(column_value)
    if not metadata:
        return True
    raw = metadata.get("is_leaf")
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "1", "yes", "on")
    return bool(raw)


def apply_row_metadata_defaults(
    metadata: Dict[str, Any],
    *,
    document_id: Optional[int] = None,
    document_title: Optional[str] = None,
    note_id: Optional[int] = None,
    note_title: Optional[str] = None,
    chunk_index: Optional[int] = None,
    node_id: Optional[str] = None,
    parent_node_id: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Complète les métadonnées avec les champs dénormalisés SQL."""
    if document_id is not None:
        metadata.setdefault("document_id", document_id)
    if document_title is not None:
        metadata.setdefault("document_title", document_title)
    if note_id is not None:
        metadata.setdefault("note_id", note_id)
    if note_title is not None:
        metadata.setdefault("note_title", note_title)
    if chunk_index is not None:
        metadata.setdefault("chunk_index", chunk_index)
    if node_id is not None:
        metadata.setdefault("node_id", node_id)
    if parent_node_id is not None:
        metadata.setdefault("parent_node_id", parent_node_id)
    if source:
        metadata.setdefault("source", source)
    return metadata


def mmr_diversity_key(metadata: dict) -> Optional[str]:
    """
    Clé de diversité MMR : un chunk par tableau (table_id), par page KAG, ou par parent section.
    """
    if metadata.get("kag_matched_entity"):
        doc_id = metadata.get("document_id")
        page_no = resolve_page_from_metadata(metadata)
        if doc_id is not None and page_no is not None:
            return f"kag_page:{doc_id}:{page_no}"
    table_id = metadata.get("table_id")
    if table_id:
        return f"table:{table_id}"
    parent_id = metadata.get("parent_node_id")
    if parent_id:
        return f"parent:{parent_id}"
    return None


def mmr_subject_key(metadata: dict) -> Optional[str]:
    """
    Clé de sujet (gamme / document) pour pénaliser les chunks hors-sujet en MMR.
    """
    explicit = metadata.get("mmr_subject_key")
    if explicit:
        return str(explicit)
    entity = metadata.get("kag_matched_entity")
    if entity and isinstance(entity, str) and entity.strip():
        return f"entity:{entity.strip().lower()}"
    doc_id = metadata.get("document_id")
    if doc_id is not None:
        return f"doc:{doc_id}"
    return None


def infer_primary_subject_key(
    pivot_entity_names: Optional[List[str]],
    candidates: Optional[List] = None,
) -> Optional[str]:
    """Déduit le sujet primaire (gamme / entité pivot) pour le MMR."""
    from app.services.kag_extraction_service import normalize_entity_name

    if pivot_entity_names:
        for name in pivot_entity_names:
            if name and len(name.strip()) >= 2:
                norm = normalize_entity_name(name)
                if norm:
                    return f"entity:{norm}"
    if not candidates:
        return None
    counts: Dict[str, int] = {}
    for nws in candidates:
        meta = dict(getattr(getattr(nws, "node", nws), "metadata", None) or {})
        sk = mmr_subject_key(meta)
        if sk:
            counts[sk] = counts.get(sk, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def content_type_score_multiplier(metadata: dict) -> float:
    """Boost léger selon le type structurel de chunk et pénalité lignes suspectes."""
    ct = str(metadata.get("content_type") or "").strip().lower()
    multipliers = {
        "table_summary": 1.08,
        "table_row": 1.02,
        "text_full": 1.0,
        "text_window": 0.98,
        "page_technical_sheet": 1.05,
    }
    mult = multipliers.get(ct, 1.0)
    if metadata.get("suspicious"):
        mult *= 0.92
    suspicious_rows = metadata.get("suspicious_rows")
    if isinstance(suspicious_rows, list) and suspicious_rows:
        mult *= 0.95
    return mult


def build_embedding_input_text(content: str, metadata: dict) -> str:
    """Texte envoyé au modèle d'embedding : section + légende figure + corps."""
    parts: List[str] = []
    heading = metadata.get("parent_heading") or metadata.get("heading")
    if heading and str(heading).strip() and heading != "__no_heading__":
        parts.append(f"[{str(heading).strip()}]")
    figure = metadata.get("figure_title") or metadata.get("image_anchor")
    if figure and str(figure).strip():
        parts.append(f"[Figure: {str(figure).strip()}]")
    if not parts:
        return content
    return "\n".join(parts) + f"\n{content}"


def parent_llm_context_block(metadata: dict) -> str:
    """Bloc résumé / questions KAG parent pour le contexte LLM."""
    summary = (metadata.get("summary") or "").strip()
    questions = metadata.get("generated_questions") or []
    if not summary and not questions:
        return ""
    lines: List[str] = []
    if summary:
        lines.append(f"Résumé de section : {summary}")
    if isinstance(questions, list):
        q_parts = [str(q).strip() for q in questions[:3] if q and str(q).strip()]
        if q_parts:
            lines.append("Questions clés : " + " ; ".join(q_parts))
    return "\n".join(lines)


def table_citation_hint(metadata: dict) -> Optional[Dict[str, Any]]:
    """Indicateurs tabulaires pour citations structurées (chat / sources)."""
    tj = metadata.get("table_json")
    if not isinstance(tj, dict):
        ct = metadata.get("content_type")
        if ct not in ("table_row", "table_full", "table_summary"):
            return None
        hint: Dict[str, Any] = {
            "content_type": ct,
            "table_id": metadata.get("table_id"),
            "row_index": metadata.get("row_index"),
        }
        if metadata.get("table_row_truncated"):
            hint["table_row_truncated"] = True
            hint["table_row_total"] = metadata.get("table_row_total")
        return hint
    hint = {
        "content_type": metadata.get("content_type"),
        "table_id": metadata.get("table_id"),
        "row_index": metadata.get("row_index"),
        "headers": tj.get("headers"),
        "nb_rows": tj.get("nb_rows"),
        "nb_cols": tj.get("nb_cols"),
    }
    if metadata.get("table_row_truncated"):
        hint["table_row_truncated"] = True
        hint["table_row_total"] = metadata.get("table_row_total")
        hint["table_row_indexed"] = metadata.get("table_row_indexed")
    return hint


def enrich_passage_content_for_llm(content: str, metadata: dict) -> str:
    """
    Enrichit le corps pour rerank / LLM : section, figure, résumé parent KAG.
    """
    parent_heading = metadata.get("parent_heading") or metadata.get("heading")
    figure_title = metadata.get("figure_title") or metadata.get("image_anchor")
    parts: List[str] = []
    if parent_heading and str(parent_heading).strip():
        parts.append(f"[Section: {str(parent_heading).strip()}]")
    if figure_title and str(figure_title).strip():
        parts.append(str(figure_title).strip())
    parent_block = parent_llm_context_block(metadata)
    if parent_block:
        parts.append(parent_block)
    if not parts:
        return content
    prefix = "\n\n".join(parts) + "\n\n"
    return prefix + content if content else prefix.strip()
