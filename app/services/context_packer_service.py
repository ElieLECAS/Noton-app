"""CAG post-retriever — packe des DOCUMENTS entiers dans le contexte de génération.

Au lieu d'injecter ~20 passages/pages tronqués, on exploite la fenêtre 256k de Mistral
Large : les passages retrouvés servent à SÉLECTIONNER des documents ; on charge ensuite
chaque document (entier s'il tient, sinon fenêtré autour des pages matchées) sous un budget
de tokens, avec un en-tête explicite (source, gamme, matériau) pour éviter la confusion de
gammes. Le rappel passe du niveau passage (fragile) au niveau document (stable).

Fonction principale : ``build_cag_context`` — remplace ``build_space_context_from_passages``
côté chat quand ``CAG_ENABLED`` est actif. Signature de sortie compatible (dict system).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlmodel import Session, select

from app.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)


def _resolve_chunk_page(chunk: DocumentChunk) -> int:
    """Page d'un chunk depuis metadata_json/metadata_ (0 si inconnue)."""
    for meta in (getattr(chunk, "metadata_json", None), getattr(chunk, "metadata_", None)):
        if isinstance(meta, dict):
            for key in ("page_no", "page_start", "page_label", "page_idx"):
                try:
                    val = int(meta.get(key))
                except (TypeError, ValueError):
                    continue
                if val > 0:
                    return val
    return 0


def _chunk_text(chunk: DocumentChunk) -> str:
    return (chunk.text or chunk.content or "").strip()


def estimate_tokens(text: str) -> int:
    """Estimation grossière FR (chars / CAG_CHARS_PER_TOKEN)."""
    return int(len(text) / max(0.5, settings.CAG_CHARS_PER_TOKEN))


def _load_leaf_chunks(session: Session, document_id: int) -> List[DocumentChunk]:
    """Tous les chunks feuilles (L1) d'un document, dans l'ordre de lecture."""
    rows = list(
        session.exec(
            select(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.is_leaf == True,  # noqa: E712
            )
        ).all()
    )
    return sorted(rows, key=lambda c: (_resolve_chunk_page(c), c.chunk_index or 0, c.id or 0))


def aggregate_documents(
    passages: List[Dict[str, Any]], *, max_documents: int
) -> List[Tuple[int, Dict[str, Any]]]:
    """Regroupe les passages par document et classe par pertinence.

    Score document = score_max + 0.2·(reste des scores) : favorise les documents à la fois
    fortement (une page très pertinente) ET largement (plusieurs pages) matchés.
    Retourne [(document_id, {score, matched_pages, title}), …] trié décroissant.
    """
    agg: Dict[int, Dict[str, Any]] = {}
    for p in passages:
        did = p.get("document_id")
        if did is None:
            continue
        did = int(did)
        entry = agg.setdefault(
            did,
            {"score_sum": 0.0, "score_max": 0.0, "matched_pages": set(), "title": p.get("document_title")},
        )
        score = float(p.get("score") or 0.0)
        entry["score_sum"] += score
        entry["score_max"] = max(entry["score_max"], score)
        for key in ("page_no", "page_start", "page_end"):
            val = p.get(key)
            if isinstance(val, int) and val > 0:
                entry["matched_pages"].add(val)

    ranked = sorted(
        agg.items(),
        key=lambda kv: kv[1]["score_max"] + 0.2 * (kv[1]["score_sum"] - kv[1]["score_max"]),
        reverse=True,
    )
    return ranked[:max_documents]


def _pages_in_window(matched_pages: set, radius: int) -> Optional[set]:
    """Ensemble de pages à conserver autour des pages matchées (None = tout le document)."""
    if not matched_pages:
        return None
    keep: set = set()
    for pg in matched_pages:
        for delta in range(-radius, radius + 1):
            if pg + delta > 0:
                keep.add(pg + delta)
    return keep


def _render_document_block(
    doc: Document,
    chunks: List[DocumentChunk],
    *,
    index: int,
    full: bool,
) -> Tuple[str, List[int]]:
    """Rend un document (ou extrait) avec en-tête métadonnées + marqueurs de page."""
    title = doc.title or "Document sans titre"
    header_bits: List[str] = []
    if doc.source:
        header_bits.append(f"Source : {doc.source}")
    if getattr(doc, "proferm_gammes", None):
        header_bits.append(f"Gamme : {', '.join(doc.proferm_gammes)}")
    if getattr(doc, "materials", None):
        header_bits.append(f"Matériau : {', '.join(doc.materials)}")
    if getattr(doc, "product_types", None):
        header_bits.append(f"Type : {', '.join(doc.product_types)}")

    lines: List[str] = [f"=== DOCUMENT {index} : « {title} » ==="]
    if header_bits:
        lines.append(" | ".join(header_bits))

    pages_included: List[int] = []
    current_page = None
    for chunk in chunks:
        text = _chunk_text(chunk)
        if not text:
            continue
        page = _resolve_chunk_page(chunk)
        if page and page != current_page:
            current_page = page
            pages_included.append(page)
            lines.append(f"\n[page {page}]")
        lines.append(text)

    scope = "document complet" if full else "extrait"
    if pages_included:
        span = f"{min(pages_included)}-{max(pages_included)}"
        lines.insert(1 if not header_bits else 2, f"Pages incluses : {span} ({scope})")

    return "\n".join(lines), pages_included


def build_cag_context(
    session: Session,
    passages: List[Dict[str, Any]],
    *,
    system_prompt: str,
    token_budget: Optional[int] = None,
    max_documents: Optional[int] = None,
    full_doc_max_tokens: Optional[int] = None,
    page_radius: Optional[int] = None,
    anchor_document_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Construit le message système CAG : documents entiers/étendus sous budget de tokens.

    Retourne un dict ``{"role": "system", "content": ..., "cag_documents": [...]}``
    (compatible avec l'ancien build_space_context_from_passages ; la clé cag_documents
    liste les documents réellement inclus, pour les sources côté UI).
    """
    token_budget = token_budget if token_budget is not None else settings.CAG_TOKEN_BUDGET
    max_documents = max_documents if max_documents is not None else settings.CAG_MAX_DOCUMENTS
    full_doc_max_tokens = (
        full_doc_max_tokens if full_doc_max_tokens is not None else settings.CAG_FULL_DOC_MAX_TOKENS
    )
    page_radius = page_radius if page_radius is not None else settings.CAG_PAGE_RADIUS

    system_message: Dict[str, Any] = {"role": "system", "content": system_prompt}

    if not passages:
        system_message["content"] += "\n\nAucun passage trouvé dans cet espace pour cette requête."
        system_message["cag_documents"] = []
        return system_message

    ranked_docs = aggregate_documents(passages, max_documents=max_documents)

    # GARANTIE d'ancrage : les documents du sujet courant de la conversation sont TOUJOURS
    # packés, en tête, même si le retrieval de ce tour ne les a pas fait remonter (ex. suivi
    # « tu as ses dimensions ? » où le mot "dimensions" tire vers un autre manuel). Un boost
    # de ranking ne peut pas repêcher un document absent du pool — l'inclusion ici, si.
    if anchor_document_ids:
        by_id = {did: meta for did, meta in ranked_docs}
        anchored: List[Tuple[int, Dict[str, Any]]] = []
        for aid in anchor_document_ids:
            aid = int(aid)
            meta = by_id.pop(
                aid,
                {"score_sum": 0.0, "score_max": 0.0, "matched_pages": set(), "title": None},
            )
            anchored.append((aid, meta))
        others = [(did, meta) for did, meta in ranked_docs if did in by_id]
        # Jamais tronquer les ancres ; le reste complète jusqu'au plafond documents.
        ranked_docs = anchored + others[: max(0, max_documents - len(anchored))]

    doc_ids = [did for did, _ in ranked_docs]
    docs_by_id = {
        d.id: d
        for d in session.exec(select(Document).where(Document.id.in_(doc_ids))).all()
    }

    blocks: List[str] = []
    cag_documents: List[Dict[str, Any]] = []
    spent_tokens = 0
    position = 0

    for did, meta in ranked_docs:
        doc = docs_by_id.get(did)
        if doc is None:
            continue
        remaining = token_budget - spent_tokens
        if remaining <= 0:
            break

        leaf_chunks = _load_leaf_chunks(session, did)
        if not leaf_chunks:
            continue
        full_tokens = estimate_tokens("\n".join(_chunk_text(c) for c in leaf_chunks))

        # Décision : document entier vs fenêtre autour des pages matchées.
        if full_tokens <= full_doc_max_tokens and full_tokens <= remaining:
            selected = leaf_chunks
            full = True
        else:
            window = _pages_in_window(meta["matched_pages"], page_radius)
            selected = [c for c in leaf_chunks if window is None or _resolve_chunk_page(c) in window]
            # Rogne encore si l'extrait dépasse le budget restant.
            while selected and estimate_tokens("\n".join(_chunk_text(c) for c in selected)) > remaining:
                selected = selected[:-1]
            full = False

        if not selected:
            continue

        position += 1
        block, pages_included = _render_document_block(doc, selected, index=position, full=full)
        block_tokens = estimate_tokens(block)
        if block_tokens > remaining and position > 1:
            # Ne pas dépasser le budget (on garde toujours au moins le 1er document).
            position -= 1
            break

        blocks.append(block)
        spent_tokens += block_tokens
        cag_documents.append(
            {
                "index": position,
                "document_id": did,
                "document_title": doc.title,
                "pages": sorted(set(pages_included)),
                "full_document": full,
                "score": round(float(meta["score_max"]), 4),
            }
        )

    cag_preamble = (
        "\n\nDOCUMENTS (contexte complet) — chaque document ci-dessous est fourni ENTIER ou en "
        "extrait étendu, avec un en-tête (source, gamme, matériau) et ses numéros de page.\n"
        "IMPÉRATIF : avant d'attribuer une valeur, une cote ou une consigne à une gamme/produit, "
        "vérifie l'en-tête du document concerné. Ne transfère JAMAIS une information d'un document "
        "vers une autre gamme (ex. Perform 70 ≠ Perform 76). Les documents sont classés par "
        "pertinence décroissante.\n\n"
    )
    system_message["content"] += cag_preamble + "\n\n".join(blocks)
    system_message["content"] += f"\n\n({len(blocks)} document(s), ~{spent_tokens} tokens de contexte.)"
    system_message["cag_documents"] = cag_documents

    logger.info(
        "[CAG] %d document(s) packé(s), ~%d tokens (budget %d) — %s",
        len(blocks),
        spent_tokens,
        token_budget,
        ", ".join(f"doc={d['document_id']}{'(complet)' if d['full_document'] else ''}" for d in cag_documents),
    )
    return system_message
