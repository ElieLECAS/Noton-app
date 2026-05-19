"""
Utilitaires partagés pour metadata_json / metadata_ (bibliothèque et notes).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


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
    Clé de diversité MMR : un chunk par tableau (table_id) ou par parent section.
    """
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
