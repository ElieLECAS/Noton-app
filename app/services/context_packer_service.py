"""CAG post-retriever — packe des DOCUMENTS entiers dans le contexte de génération.

Au lieu d'injecter ~20 passages/pages tronqués, on exploite la fenêtre 256k de Mistral
Large : les passages retrouvés servent à SÉLECTIONNER des documents ; on charge ensuite
chaque document (entier s'il tient, sinon fenêtré autour des pages matchées) sous un budget
de tokens, avec un en-tête explicite (source, gamme, matériau) pour éviter la confusion de
gammes. Le rappel passe du niveau passage (fragile) au niveau document (stable).

Fonctions principales :
  * ``build_cag_context`` — remplace ``build_space_context_from_passages`` côté chat quand
    ``CAG_ENABLED`` est actif. Budget/max_documents adaptatifs par intent
    (``CAG_BUDGET_BY_INTENT``). Signature de sortie compatible (dict system).
  * ``select_cag_images`` — PNG UNIQUEMENT pour des pages réellement packées dans le
    contexte (alignement texte/visuel), avec légendes pour relier image ↔ document.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from sqlmodel import Session, select

from app.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)

# Enregistrement d'un chunk feuille aplati : (page, chunk_index, texte).
LeafRecord = Tuple[int, int, str]

# Cache TTL du texte des feuilles par document (évite ~60 lignes SQL + concat par requête).
# Valeurs PLATES (pas d'objets ORM : ils seraient détachés de leur session d'origine).
_leaf_cache: Dict[int, Tuple[float, List[LeafRecord]]] = {}


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


def budget_for_intent(intent: Optional[str]) -> Tuple[int, int]:
    """(token_budget, max_documents) selon l'intent de la requête.

    Une question de spécification ponctuelle ne paie pas le prefill (coût + latence
    1er token) d'un diagnostic SAV. Table ``CAG_BUDGET_BY_INTENT`` ; la clé "default"
    couvre les intents absents/inconnus.

    CAG_TOKEN_BUDGET et CAG_MAX_DOCUMENTS sont des PLAFONDS DURS : la table par intent
    module en dessous, jamais au-dessus. Sans ce clamp, la table par défaut du code
    (product_selection=100000/8) rendait les variables d'env inopérantes — une prod
    configurée à 50000/3 packait quand même 100000/8 (constaté logs prod 2026-07-22).
    """
    table = settings.cag_budget_by_intent or {}
    key = (intent or "").strip().lower()
    entry = table.get(key) or table.get("default") or {}
    token_budget = int(entry.get("budget") or settings.CAG_TOKEN_BUDGET)
    max_documents = int(entry.get("max_documents") or settings.CAG_MAX_DOCUMENTS)
    token_budget = min(token_budget, int(settings.CAG_TOKEN_BUDGET))
    max_documents = min(max_documents, int(settings.CAG_MAX_DOCUMENTS))
    return max(1000, token_budget), max(1, max_documents)


def invalidate_document_fulltext_cache(document_id: Optional[int] = None) -> None:
    """Invalide le cache des feuilles (un document, ou tout si None) — à appeler après réindexation."""
    if document_id is None:
        _leaf_cache.clear()
    else:
        _leaf_cache.pop(int(document_id), None)


def _load_leaf_records(session: Session, document_id: int) -> List[LeafRecord]:
    """Chunks feuilles (L1) d'un document, aplatis en (page, index, texte), ordre de lecture.

    Mise en cache TTL (``CAG_FULLTEXT_CACHE_TTL``, 0 = off) : un réindex peut mettre
    jusqu'à TTL secondes à se refléter ici — acceptable, et ``invalidate_document_fulltext_cache``
    permet l'invalidation immédiate côté indexation.
    """
    ttl = settings.CAG_FULLTEXT_CACHE_TTL
    now = time.monotonic()
    if ttl > 0:
        cached = _leaf_cache.get(document_id)
        if cached and cached[0] > now:
            return cached[1]

    rows = list(
        session.exec(
            select(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.is_leaf == True,  # noqa: E712
            )
        ).all()
    )
    rows.sort(key=lambda c: (_resolve_chunk_page(c), c.chunk_index or 0, c.id or 0))
    records = [
        (_resolve_chunk_page(c), c.chunk_index or 0, text)
        for c in rows
        if (text := _chunk_text(c))
    ]
    if ttl > 0:
        _leaf_cache[document_id] = (now + ttl, records)
    return records


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


def _records_tokens(records: List[LeafRecord]) -> int:
    return estimate_tokens("\n".join(text for _, _, text in records))


def _trim_records_to_budget(
    records: List[LeafRecord], matched_pages: set, remaining: int
) -> List[LeafRecord]:
    """Rogne un extrait qui dépasse le budget en retirant d'abord les pages les plus
    ÉLOIGNÉES des pages matchées (et non les dernières du document : la réponse est
    souvent juste après le match, pas avant)."""

    def dist(page: int) -> int:
        if not matched_pages:
            return 0
        return min(abs(page - m) for m in matched_pages)

    trimmed = list(records)
    while trimmed and _records_tokens(trimmed) > remaining:
        pages = {page for page, _, _ in trimmed}
        worst = max(pages, key=lambda p: (dist(p), p))
        trimmed = [r for r in trimmed if r[0] != worst]
    return trimmed


def _render_document_block(
    doc: Document,
    records: List[LeafRecord],
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
    for page, _, text in records:
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
    intent: Optional[str] = None,
    emit_sources_tag: bool = True,
) -> Dict[str, Any]:
    """Construit le message système CAG : documents entiers/étendus sous budget de tokens.

    Budget et nombre de documents résolus par priorité : paramètre explicite >
    table par intent (``CAG_BUDGET_BY_INTENT``) > plafonds globaux.

    Retourne un dict ``{"role": "system", "content": ..., "cag_documents": [...]}``
    (compatible avec l'ancien build_space_context_from_passages ; la clé cag_documents
    liste les documents réellement inclus, pour les sources côté UI et les images).
    """
    intent_budget, intent_max_docs = budget_for_intent(intent)
    token_budget = token_budget if token_budget is not None else intent_budget
    max_documents = max_documents if max_documents is not None else intent_max_docs
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

        leaf_records = _load_leaf_records(session, did)
        if not leaf_records:
            continue
        full_tokens = _records_tokens(leaf_records)

        # Décision : document entier vs fenêtre autour des pages matchées.
        if full_tokens <= full_doc_max_tokens and full_tokens <= remaining:
            selected = leaf_records
            full = True
        else:
            window = _pages_in_window(meta["matched_pages"], page_radius)
            selected = [r for r in leaf_records if window is None or r[0] in window]
            # Rogne encore si l'extrait dépasse le budget restant (pages les plus
            # éloignées des pages matchées d'abord).
            selected = _trim_records_to_budget(selected, meta["matched_pages"], remaining)
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
                "matched_pages": sorted(meta["matched_pages"]),
                "has_source_file": bool(getattr(doc, "source_file_path", None)),
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
    if emit_sources_tag:
        system_message["content"] += (
            "\n\nFIN DE RÉPONSE OBLIGATOIRE : termine ta réponse par une ligne EXACTEMENT au format "
            '<sources>{"used":[{"doc":1,"pages":[3,4]}]}</sources> listant les index de DOCUMENT '
            "et les pages que tu as réellement utilisés pour répondre (liste vide si aucun). "
            "Cette ligne est masquée à l'utilisateur — n'en parle jamais dans le corps de la réponse."
        )
    system_message["cag_documents"] = cag_documents

    logger.info(
        "[CAG] %d document(s) packé(s), ~%d tokens (budget %d, intent=%s) — %s",
        len(blocks),
        spent_tokens,
        token_budget,
        intent or "n/a",
        ", ".join(f"doc={d['document_id']}{'(complet)' if d['full_document'] else ''}" for d in cag_documents),
    )
    return system_message


def _pages_span_label(pages: List[int]) -> str:
    """Libellé humain d'un ensemble de pages ("page 5" / "pages 1-12")."""
    if not pages:
        return "pages n/a"
    lo, hi = min(pages), max(pages)
    return f"page {lo}" if lo == hi else f"pages {lo}-{hi}"


def build_document_sources(
    cag_documents: List[Dict[str, Any]], used_pages_by_index: Optional[Dict[Any, Any]] = None
) -> List[Dict[str, Any]]:
    """Sources UI par DOCUMENT packé — reflète le contexte que le modèle a réellement lu.

    ``used_pages_by_index`` vient du bloc final <sources> émis par le modèle : quand il est
    exploitable, seuls les documents réellement UTILISÉS sont affichés (avec leurs pages) ;
    sinon (None / vide / inexploitable) on retombe sur tous les documents packés. La forme
    des entrées reste compatible avec les badges front (index, document_title, page_no,
    excerpt, score, has_source_file, cag_document)."""
    used_pages_by_index = used_pages_by_index or {}
    entries: List[Dict[str, Any]] = []
    for d in cag_documents:
        idx = d.get("index")
        if used_pages_by_index and idx not in used_pages_by_index:
            continue
        used_pages = [p for p in (used_pages_by_index.get(idx) or []) if isinstance(p, int)]
        pages = [p for p in (d.get("pages") or []) if isinstance(p, int)]
        matched = [p for p in (d.get("matched_pages") or []) if isinstance(p, int)]
        landing_candidates = used_pages or matched or pages
        landing = landing_candidates[0] if landing_candidates else None

        scope = ("Document complet" if d.get("full_document") else "Extrait") + f" ({_pages_span_label(pages)})"
        if used_pages:
            scope += " — pages utilisées : " + ", ".join(str(p) for p in used_pages)

        entries.append(
            {
                "index": idx,
                "document_id": d.get("document_id"),
                "document_title": d.get("document_title") or "Document sans titre",
                "excerpt": scope,
                "passage_full": scope,
                "score": float(d.get("score") or 0.0),
                "page_no": landing,
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
                "section": None,
                "has_source_file": bool(d.get("has_source_file")),
                "cag_document": True,
                "pages": pages,
                "used_pages": used_pages,
                "full_document": bool(d.get("full_document")),
            }
        )

    if not entries and cag_documents:
        # Le bloc <sources> du modèle a tout écarté (ou est inexploitable) → afficher
        # tous les documents packés plutôt que rien.
        return build_document_sources(cag_documents, {})
    return entries


def select_cag_images(
    session: Session,
    cag_documents: List[Dict[str, Any]],
    passages: List[Dict[str, Any]],
    *,
    max_images: Optional[int] = None,
    dpi: int = 150,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Sélectionne et rend les PNG de pages pour la génération vision, ALIGNÉS sur le
    contexte packé : uniquement des pages incluses dans un document CAG.

    Priorité : pages à besoin visuel (needs_page_image) puis score de passage décroissant.
    Retourne (images_b64, captions) — captions = [{image_index, document_index,
    document_title, page_no}, …] pour légender les images dans le message user.
    """
    import base64
    import os

    from app.services.multimodal_page_service import render_page_png_cached

    max_images = max_images if max_images is not None else settings.CAG_MAX_IMAGES
    if max_images <= 0 or not cag_documents:
        return [], []

    included: Dict[int, Dict[str, Any]] = {
        int(d["document_id"]): d for d in cag_documents if d.get("document_id") is not None
    }

    # Candidats (doc_id, page) depuis les passages, page ancre uniquement (pas les voisins :
    # le texte des voisins est déjà dans le contexte, le PNG n'apporte que pour le match).
    candidates: List[Tuple[int, int, int]] = []  # (need_rank, doc_id, page)
    seen: set = set()
    for p in sorted(passages, key=lambda x: float(x.get("score") or 0.0), reverse=True):
        did = p.get("document_id")
        if did is None or int(did) not in included:
            continue
        did = int(did)
        page = p.get("page_no") or p.get("page_start")
        if not isinstance(page, int) or page <= 0:
            continue
        if page not in (included[did].get("pages") or []):
            continue
        if (did, page) in seen:
            continue
        seen.add((did, page))
        candidates.append((0 if p.get("needs_page_image") else 1, did, page))

    # Tri stable : besoin visuel d'abord, ordre score conservé au sein de chaque classe.
    candidates.sort(key=lambda t: t[0])

    images_b64: List[str] = []
    captions: List[Dict[str, Any]] = []
    doc_cache: Dict[int, Optional[Document]] = {}
    for _, did, page in candidates:
        if len(images_b64) >= max_images:
            break
        if did not in doc_cache:
            doc_cache[did] = session.get(Document, did)
        doc = doc_cache[did]
        if not doc or not doc.source_file_path or not os.path.exists(doc.source_file_path):
            continue
        try:
            png_bytes = render_page_png_cached(doc.source_file_path, page, dpi=dpi)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CAG] rendu PNG échoué (doc=%s page=%s): %s", did, page, exc)
            continue
        images_b64.append(base64.b64encode(png_bytes).decode("utf-8"))
        captions.append(
            {
                "image_index": len(images_b64),
                "document_index": included[did].get("index"),
                "document_title": included[did].get("document_title"),
                "page_no": page,
            }
        )

    logger.info(
        "[CAG] %d image(s) alignée(s) sur le contexte packé (max %d) — %s",
        len(images_b64),
        max_images,
        ", ".join(f"doc{c['document_index']}:p{c['page_no']}" for c in captions),
    )
    return images_b64, captions
