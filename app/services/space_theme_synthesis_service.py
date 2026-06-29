"""Synthèse "carte mentale" (CAG) d'un nœud d'arbre thématique.

Principe : pour un nœud (racine, famille ou catégorie), on récupère TOUT le texte L1
des documents rattachés dans l'espace — sans images ni enrichissements — on l'assemble
en un seul contexte et on demande au LLM une note de synthèse. C'est du CAG (contexte
bourré) plutôt que du RAG sélectif. Le résultat est mémorisé (``SpaceThemeSynthesis``)
et invalidé si l'ensemble des chunks change.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import bindparam, text
from sqlmodel import Session, select

from app.config import settings
from app.models.space_theme_synthesis import SpaceThemeSynthesis
from app.services import mistral_service
from app.services.category_catalog import AXIS_TASK
from app.services.page_retrieval_service import (
    CONTENT_TYPE_CONTEXTUAL_ENRICHMENT,
    CONTENT_TYPE_PAGE_ANCHOR,
    CONTENT_TYPE_SEMANTIC_LEAF,
)
from app.services.space_theme_tree_service import (
    chunk_ids_for_entity_category,
    parse_entity_node_key,
    resolve_node_to_category_ids,
)

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = (
    "Tu es un expert documentaire. À partir des EXTRAITS fournis (issus de plusieurs "
    "documents d'un même espace), rédige une note de synthèse claire et structurée en "
    "français, comme si tous ces documents n'en formaient qu'un seul.\n\n"
    "Règles strictes :\n"
    "- Fonde-toi UNIQUEMENT sur les extraits fournis ; n'invente rien, n'extrapole pas.\n"
    "- Organise la synthèse en sous-thèmes avec des titres Markdown (## / ###) et des "
    "listes à puces.\n"
    "- Commence par un paragraphe d'introduction qui résume le thème en 2-3 phrases.\n"
    "- Regroupe et réconcilie les informations qui se recoupent entre documents ; "
    "signale explicitement les contradictions éventuelles.\n"
    "- Si une information attendue est absente des extraits, indique-le brièvement "
    "plutôt que de combler le vide.\n"
    "- Reste factuel et concis ; pas de formules commerciales."
)


def _merge_meta(metadata_json: Any, metadata_: Any) -> Dict[str, Any]:
    return dict(metadata_json or metadata_ or {})


def _is_text_chunk(meta: Dict[str, Any]) -> bool:
    """Garde le texte source L1 ; exclut images, ancres et enrichissements."""
    content_type = meta.get("content_type")
    if content_type in (CONTENT_TYPE_PAGE_ANCHOR, CONTENT_TYPE_CONTEXTUAL_ENRICHMENT):
        return False
    if content_type not in (None, CONTENT_TYPE_SEMANTIC_LEAF):
        return False
    return True


def _fetch_chunk_rows(
    session: Session,
    space_id: int,
    category_ids: Optional[Sequence[int]],
    chunk_ids: Optional[Sequence[int]] = None,
) -> List[Tuple]:
    """Rows (chunk_id, document_id, title, content, chunk_index, meta_json, meta_)."""
    if chunk_ids is not None:
        # Nœud entité : ensemble de chunks explicite (entité ∩ catégorie).
        if not chunk_ids:
            return []
        stmt = text(
            """
            SELECT dc.id, dc.document_id, d.title, dc.content, dc.chunk_index,
                   dc.metadata_json, dc.metadata_
            FROM documentchunk dc
            INNER JOIN document d ON d.id = dc.document_id
            WHERE dc.id IN :chunk_ids
            ORDER BY d.title, dc.chunk_index
            """
        ).bindparams(bindparam("chunk_ids", expanding=True))
        return list(session.execute(stmt, {"chunk_ids": list(chunk_ids)}).all())

    if category_ids is None:
        # Racine : tout le texte L1 des documents de l'espace.
        return list(
            session.execute(
                text(
                    """
                    SELECT dc.id, dc.document_id, d.title, dc.content, dc.chunk_index,
                           dc.metadata_json, dc.metadata_
                    FROM documentchunk dc
                    INNER JOIN document_space ds ON ds.document_id = dc.document_id
                    INNER JOIN document d ON d.id = dc.document_id
                    WHERE ds.space_id = :space_id
                      AND dc.is_leaf = true
                    ORDER BY d.title, dc.chunk_index
                    """
                ),
                {"space_id": space_id},
            ).all()
        )

    if not category_ids:
        return []

    stmt = text(
        """
        SELECT dc.id, dc.document_id, d.title, dc.content, dc.chunk_index,
               dc.metadata_json, dc.metadata_
        FROM chunkcategoryrelation ccr
        INNER JOIN documentchunk dc ON dc.id = ccr.chunk_id
        INNER JOIN document_space ds ON ds.document_id = ccr.document_id
        INNER JOIN document d ON d.id = ccr.document_id
        WHERE ds.space_id = :space_id
          AND ccr.category_id IN :category_ids
        GROUP BY dc.id, dc.document_id, d.title, dc.content, dc.chunk_index,
                 dc.metadata_json, dc.metadata_
        ORDER BY d.title, dc.chunk_index
        """
    ).bindparams(bindparam("category_ids", expanding=True))
    return list(
        session.execute(
            stmt,
            {"space_id": space_id, "category_ids": list(category_ids)},
        ).all()
    )


def collect_text_and_sources(
    session: Session,
    space_id: int,
    category_ids: Optional[Sequence[int]],
    chunk_ids: Optional[Sequence[int]] = None,
) -> Tuple[str, List[Dict[str, Any]], List[int], bool]:
    """Assemble le texte (budgété) + sources + ids de chunks inclus + flag tronqué."""
    rows = _fetch_chunk_rows(session, space_id, category_ids, chunk_ids=chunk_ids)

    budget = settings.SYNTHESIS_MAX_CONTEXT_CHARS
    used = 0
    truncated = False
    included_ids: List[int] = []

    # Regroupement par document en conservant l'ordre d'apparition.
    docs: "dict[int, Dict[str, Any]]" = {}
    for chunk_id, document_id, title, content, _chunk_index, meta_json, meta_ in rows:
        meta = _merge_meta(meta_json, meta_)
        if not _is_text_chunk(meta):
            continue
        body = (content or "").strip()
        if not body:
            continue

        heading = meta.get("heading") or meta.get("parent_heading")
        step_no = meta.get("step_number")
        prefix = ""
        if heading:
            prefix = f"### {heading}\n"
        elif step_no is not None:
            prefix = f"### Étape {step_no}\n"
        piece = f"{prefix}{body}"

        if used + len(piece) > budget:
            truncated = True
            break
        used += len(piece) + 2

        doc = docs.setdefault(
            int(document_id),
            {"title": title or f"Document {document_id}", "pieces": [], "pages": set()},
        )
        doc["pieces"].append(piece)
        included_ids.append(int(chunk_id))
        page = meta.get("page_start") or meta.get("page_no")
        if page is not None:
            try:
                doc["pages"].add(int(page))
            except (TypeError, ValueError):
                pass

    parts: List[str] = []
    sources: List[Dict[str, Any]] = []
    for document_id, doc in docs.items():
        parts.append(f"# {doc['title']}\n\n" + "\n\n".join(doc["pieces"]))
        sources.append(
            {
                "document_id": document_id,
                "document_title": doc["title"],
                "pages": sorted(doc["pages"]),
            }
        )

    return "\n\n".join(parts).strip(), sources, included_ids, truncated


def _content_hash(chunk_ids: Sequence[int]) -> str:
    raw = ",".join(str(i) for i in sorted(chunk_ids))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _cache_payload(row: SpaceThemeSynthesis, cached: bool) -> Dict[str, Any]:
    return {
        "space_id": row.space_id,
        "node_key": row.node_key,
        "node_label": row.node_label,
        "axis": row.axis,
        "synthesis_markdown": row.synthesis_markdown,
        "sources": row.sources_json or [],
        "chunk_count": row.chunk_count,
        "truncated": row.truncated,
        "model": row.model,
        "cached": cached,
        "status": "ok" if row.synthesis_markdown else "empty",
    }


async def generate_space_theme_synthesis(
    session: Session,
    space_id: int,
    node_key: str,
    node_label: str,
    axis: str = AXIS_TASK,
) -> Dict[str, Any]:
    """Génère (ou récupère depuis le cache) la synthèse CAG d'un nœud."""
    label = (node_label or "").strip() or ("Tous les documents" if node_key == "root" else "Thème")

    ent = parse_entity_node_key(node_key)
    if ent is not None:
        scope_chunk_ids = chunk_ids_for_entity_category(session, space_id, ent[0], ent[1])
        context_text, sources, chunk_ids, truncated = collect_text_and_sources(
            session, space_id, None, chunk_ids=scope_chunk_ids
        )
    else:
        category_ids = resolve_node_to_category_ids(session, space_id, node_key, axis)
        context_text, sources, chunk_ids, truncated = collect_text_and_sources(
            session, space_id, category_ids
        )
    content_hash = _content_hash(chunk_ids)

    existing = session.exec(
        select(SpaceThemeSynthesis).where(
            SpaceThemeSynthesis.space_id == space_id,
            SpaceThemeSynthesis.node_key == node_key,
            SpaceThemeSynthesis.axis == axis,
        )
    ).first()

    if (
        existing
        and existing.content_hash == content_hash
        and existing.synthesis_markdown
    ):
        return _cache_payload(existing, cached=True)

    if not context_text.strip():
        return {
            "space_id": space_id,
            "node_key": node_key,
            "node_label": label,
            "axis": axis,
            "synthesis_markdown": "",
            "sources": [],
            "chunk_count": 0,
            "truncated": False,
            "cached": False,
            "status": "empty",
            "model": settings.MODEL_FAST,
        }

    user_message = (
        f"Thème à synthétiser : « {label} ».\n\n"
        f"EXTRAITS DES DOCUMENTS :\n\n{context_text}\n\n"
        "Rédige maintenant la note de synthèse en respectant les règles."
    )
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    logger.info(
        "[THEME-SYNTHESIS] space=%s node=%s chars=%d chunks=%d truncated=%s",
        space_id,
        node_key,
        len(context_text),
        len(chunk_ids),
        truncated,
    )

    response = await mistral_service.chat(
        message="",
        model=settings.MODEL_FAST,
        context=messages,
        max_tokens=settings.SYNTHESIS_MAX_TOKENS,
        temperature=settings.SYNTHESIS_TEMPERATURE,
    )
    synthesis = (
        (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    ).strip()

    now = datetime.utcnow()
    if existing:
        existing.content_hash = content_hash
        existing.node_label = label
        existing.synthesis_markdown = synthesis
        existing.sources_json = sources
        existing.chunk_count = len(chunk_ids)
        existing.truncated = truncated
        existing.model = settings.MODEL_FAST
        existing.updated_at = now
        row = existing
    else:
        row = SpaceThemeSynthesis(
            space_id=space_id,
            node_key=node_key,
            axis=axis,
            content_hash=content_hash,
            node_label=label,
            synthesis_markdown=synthesis,
            sources_json=sources,
            chunk_count=len(chunk_ids),
            truncated=truncated,
            model=settings.MODEL_FAST,
            created_at=now,
            updated_at=now,
        )
        session.add(row)

    try:
        session.commit()
        session.refresh(row)
    except Exception:
        session.rollback()
        logger.exception("[THEME-SYNTHESIS] échec persistance cache")

    payload = _cache_payload(row, cached=False)
    return payload
