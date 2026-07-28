"""
Pipeline d'indexation documentaire unifié.

Quatre modes, pensés pour pouvoir séparer un passage RAPIDE d'un passage LOURD :
  - full        : extraction + KAG + enrichissement + embeddings + ColPali (tout)
  - text_only   : extraction + embeddings SEULEMENT (ni KAG, ni enrichissement,
                  ColPali inchangé) → rapide, utilisable en journée sur tout le corpus
  - kag_only    : KAG (entités/relations/catégories) + enrichissement contextuel +
                  ré-embedding, sur les chunks EXISTANTS → lent (appels LLM par batch),
                  à lancer séparément (typiquement la nuit)
  - colpali_only: re-sync ColPali uniquement (chunks texte inchangés)

text_only puis kag_only aboutit au MÊME état final que full : kag_only ré-embarque
les chunks, donc le préfixe d'embedding récupère catégories et entités.

Deux voies d'extraction du texte (paramètre ``extractor``) :
  - vision : rendu PNG + mistral-small (gère les pages sans couche texte)
  - text   : couche texte native pymupdf4llm, avec repli vision par page
"""
from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

from sqlalchemy import delete, text
from sqlmodel import Session, select

from app.config import settings
from app.database import engine
from app.library_document_logging import get_library_document_logger
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.vision_page_extraction_service import (
    CHUNKING_VERSION,
    extract_page_chunk_specs,
    merge_cross_page_chunks,
)

logger = logging.getLogger(__name__)

CONTENT_TYPE_PAGE_ANCHOR = "page_anchor"
CONTENT_TYPE_SEMANTIC_LEAF = "semantic_leaf"
CONTENT_TYPE_CONTEXTUAL_ENRICHMENT = "contextual_enrichment"


class IndexingMode(str, Enum):
    FULL = "full"
    TEXT_ONLY = "text_only"
    COLPALI_ONLY = "colpali_only"
    # KAG seul : re-extrait entités + relations + catégories sur les chunks EXISTANTS
    # (ni texte, ni ColPali, ni embedding vision re-faits). Répare les documents dont
    # les chunks ont été re-créés (liens KAG orphelins) sans tout retraiter.
    KAG_ONLY = "kag_only"


class TextExtractor(str, Enum):
    """Voie d'extraction du texte (modes full et text_only uniquement).

    VISION : rendu PNG 300 dpi + mistral-small. Gère les pages sans couche texte et
        produit la structure (étapes, sections), mais transcrit les chiffres depuis
        des pixels et ne peut pas sortir un grand tableau entier (plafond de tokens).
    TEXT   : pymupdf4llm sur la couche texte native. Chiffres et références exacts,
        tableaux complets en chunks-lignes, coût nul, déterministe. Bascule
        automatiquement sur la vision pour les pages sans texte.
    """

    VISION = "vision"
    TEXT = "text"


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------


def process_document_indexing(
    document_id: int,
    file_path: str,
    user_id: int,
    mode: IndexingMode = IndexingMode.FULL,
    run_id: Optional[str] = None,
    extractor: TextExtractor = TextExtractor.VISION,
) -> dict:
    """
    Orchestrateur principal d'indexation documentaire.

    Gère les 3 modes de traitement et la progression en base.
    ``extractor`` sélectionne la voie d'extraction du texte (vision ou texte natif) ;
    il n'a d'effet que sur les modes full et text_only.
    Lève une exception en cas d'échec après avoir mis le document en status=failed.
    """
    from app.services.document_run import is_processing_run_current
    from app.services.file_conversion import ensure_pdf_for_ocr

    ld = get_library_document_logger()

    def _aborted() -> bool:
        return run_id is not None and not is_processing_run_current(document_id, run_id)

    if _aborted():
        return {"document_id": document_id, "status": "aborted", "reason": "stale_run"}

    ld.info(
        "[Indexing] Démarrage document_id=%s mode=%s extractor=%s file=%s",
        document_id,
        mode.value,
        extractor.value,
        file_path,
    )

    # --- Nettoyage initial & passage en processing ---
    with Session(engine) as session:
        document = session.get(Document, document_id)
        if not document:
            raise ValueError(f"Document introuvable : {document_id}")

        if mode == IndexingMode.FULL:
            ld.info("[Indexing] mode=full — suppression chunks + LanceDB")
            _delete_all_chunks(session, document_id)
        elif mode == IndexingMode.TEXT_ONLY:
            ld.info("[Indexing] mode=text_only — suppression chunks texte L1 (L0 page_anchor conservés, LanceDB inchangé)")
            _delete_text_chunks(session, document_id)
        # COLPALI_ONLY : aucune suppression de chunks

        document = session.get(Document, document_id)
        document.processing_status = "processing"
        document.processing_progress = 10
        document.updated_at = datetime.utcnow()
        session.add(document)
        session.commit()
        title = document.title or ""

    # --- Conversion PDF ---
    try:
        pdf_path = ensure_pdf_for_ocr(file_path)
    except Exception as exc:
        _mark_failed(document_id, str(exc))
        raise

    _set_progress(document_id, 10)

    chunk_count = 0
    embed_count = 0
    kag_stats: dict = {"entities": 0, "relations": 0, "status": "disabled"}

    enrichment_stats: dict = {"chunks": 0, "status": "disabled"}

    try:
        if mode in (IndexingMode.FULL, IndexingMode.TEXT_ONLY, IndexingMode.KAG_ONLY):
            if _aborted():
                return {"document_id": document_id, "status": "aborted", "reason": "stale_run"}

            if mode in (IndexingMode.FULL, IndexingMode.TEXT_ONLY):
                # --- 1. Extraction texte → L0 page_anchor + L1 semantic_leaf ---
                _set_progress(document_id, 30)
                ld.info(
                    "[Indexing] Extraction %s document_id=%s",
                    extractor.value,
                    document_id,
                )
                with Session(engine) as session:
                    document = session.get(Document, document_id)
                    chunk_count = _extract_and_persist_chunks(
                        session,
                        document,
                        pdf_path,
                        preserve_page_anchors=(mode == IndexingMode.TEXT_ONLY),
                        extractor=extractor,
                    )
            elif mode == IndexingMode.KAG_ONLY:
                # Chunks EXISTANTS : compter (pour finalize) + purger l'ancien KAG du doc
                # avant ré-extraction (répare les liens orphelins sans re-créer les chunks,
                # donc sans casser l'index ColPali).
                _set_progress(document_id, 30)
                ld.info("[Indexing] KAG seul — chunks existants document_id=%s", document_id)
                with Session(engine) as session:
                    chunk_count = session.execute(
                        text("SELECT count(*) FROM documentchunk WHERE document_id = :d AND is_leaf = true"),
                        {"d": document_id},
                    ).scalar() or 0
                    if settings.KAG_ENABLED:
                        from app.services.kag_extraction_service import cleanup_kag_for_document
                        cleanup_kag_for_document(session, document_id)
                        session.commit()

            if _aborted():
                return {"document_id": document_id, "status": "aborted", "reason": "stale_run"}

            # Couches sémantiques lourdes (KAG + enrichissement contextuel) : plusieurs
            # appels LLM par batch de pages, soit l'essentiel du temps de traitement.
            # Elles sont SAUTÉES en text_only — qui devient un passage rapide « extraction
            # + embeddings » utilisable en journée sur tout le corpus — et portées par
            # kag_only, lancé séparément (typiquement la nuit). Enchaîner text_only puis
            # kag_only aboutit au même état final que full : kag_only ré-embarque les
            # chunks, donc le préfixe d'embedding récupère bien catégories et entités.
            runs_semantic_layers = mode in (IndexingMode.FULL, IndexingMode.KAG_ONLY)

            # --- 2. KAG : entités + relations + catégories (écrites dans les métadonnées L1) ---
            # NB : tourne AVANT l'embedding pour que le vecteur intègre catégories/entités.
            if runs_semantic_layers and settings.KAG_ENABLED:
                _set_progress(document_id, 45)
                ld.info("[Indexing] Extraction KAG entités/relations document_id=%s", document_id)
                try:
                    from app.services.kag_extraction_service import (
                        embed_kag_entities_for_document,
                        extract_kag_for_document,
                    )

                    kag_stats = extract_kag_for_document(document_id, pdf_path)
                    ld.info("[Indexing] Embedding entités KAG document_id=%s", document_id)
                    embed_kag_entities_for_document(document_id)
                except Exception as exc:
                    logger.error(
                        "[Indexing] KAG échoué document_id=%s (non bloquant) : %s",
                        document_id,
                        exc,
                        exc_info=True,
                    )
                    kag_stats = {"entities": 0, "relations": 0, "status": "failed"}

            if _aborted():
                return {"document_id": document_id, "status": "aborted", "reason": "stale_run"}

            # --- 3. Enrichissement contextuel inter-pages → L2 contextual_enrichment ---
            # Idempotent : run_contextual_enrichment_for_document purge les L2 existants
            # avant de régénérer, donc relancer kag_only ne duplique rien.
            if runs_semantic_layers and settings.CONTEXTUAL_ENRICHMENT_ENABLED:
                _set_progress(document_id, 60)
                ld.info(
                    "[Indexing] Enrichissement contextuel document_id=%s",
                    document_id,
                )
                try:
                    from app.services.contextual_enrichment_service import (
                        run_contextual_enrichment_for_document,
                    )

                    enrichment_stats = run_contextual_enrichment_for_document(document_id)
                except Exception as exc:
                    logger.error(
                        "[Indexing] Enrichissement échoué document_id=%s (non bloquant) : %s",
                        document_id,
                        exc,
                        exc_info=True,
                    )
                    enrichment_stats = {"chunks": 0, "status": "failed"}

            if _aborted():
                return {"document_id": document_id, "status": "aborted", "reason": "stale_run"}

            # --- 4. Embedding mistral-embed EN DERNIER (L1 + L2, métadonnées enrichies) ---
            _set_progress(document_id, 80)
            ld.info("[Indexing] Embeddings mistral-embed (L1+L2) document_id=%s", document_id)
            embed_count = _embed_text_chunks(document_id)

        if mode in (IndexingMode.FULL, IndexingMode.COLPALI_ONLY):
            if _aborted():
                return {"document_id": document_id, "status": "aborted", "reason": "stale_run"}

            _set_progress(document_id, 90)
            ld.info("[Indexing] ColPali sync document_id=%s", document_id)
            _sync_colpali_for_pages(document_id, pdf_path)

        _finalize_document(document_id, chunk_count)
        semantic_ran = mode in (IndexingMode.FULL, IndexingMode.KAG_ONLY)
        ld.info(
            "[Indexing] FIN OK document_id=%s mode=%s chunks=%s embeds=%s kag=%s enrichment=%s",
            document_id,
            mode.value,
            chunk_count,
            embed_count,
            kag_stats if semantic_ran else "n/a (text_only)",
            enrichment_stats if semantic_ran else "n/a (text_only)",
        )
        result = {"document_id": document_id, "chunks": chunk_count, "status": "completed"}
        if semantic_ran:
            result["kag"] = kag_stats
            result["enrichment"] = enrichment_stats
        return result

    except Exception as exc:
        logger.error("[Indexing] Échec document_id=%s: %s", document_id, exc, exc_info=True)
        _mark_failed(document_id, str(exc))
        raise


# ---------------------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------------------


def _get_deletable_text_chunk_ids(session: Session, document_id: int) -> List[int]:
    """IDs des chunks texte supprimables en mode text_only (hors page_anchor)."""
    rows = session.execute(
        text(
            """
            SELECT id FROM documentchunk
            WHERE document_id = :doc_id
              AND COALESCE(
                  metadata_json->>'content_type',
                  metadata_->>'content_type',
                  ''
              ) != :page_anchor
            """
        ),
        {"doc_id": document_id, "page_anchor": CONTENT_TYPE_PAGE_ANCHOR},
    ).all()
    return [int(row[0]) for row in rows]


def _delete_chunk_foreign_relations(session: Session, chunk_ids: List[int]) -> None:
    """Supprime les relations FK bloquant la suppression de chunks (KAG + enrichissement)."""
    if not chunk_ids:
        return

    if settings.KAG_ENABLED:
        from app.services.kag_extraction_service import (
            delete_chunk_kag_relations,
            prune_kag_entities_after_chunk_removal,
        )

        affected = delete_chunk_kag_relations(session, chunk_ids)
        try:
            with session.begin_nested():
                prune_kag_entities_after_chunk_removal(session, affected)
        except Exception as exc:
            logger.warning(
                "[Indexing] Élagage entités KAG ignoré (%s chunk(s)) : %s",
                len(chunk_ids),
                exc,
            )


def _delete_all_chunks(session: Session, document_id: int) -> None:
    """Supprime tous les chunks PostgreSQL, relations KAG et patches LanceDB."""
    all_chunk_ids = [
        int(row[0])
        for row in session.execute(
            text("SELECT id FROM documentchunk WHERE document_id = :doc_id"),
            {"doc_id": document_id},
        ).all()
    ]
    _delete_chunk_foreign_relations(session, all_chunk_ids)

    session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    session.commit()
    from app.services.context_packer_service import invalidate_document_fulltext_cache

    invalidate_document_fulltext_cache(document_id)
    try:
        from app.services.lancedb_service import delete_colpali_patches_for_document
        delete_colpali_patches_for_document(document_id)
    except Exception as exc:
        logger.warning("[Indexing] LanceDB delete échoué pour document_id=%s : %s", document_id, exc)


def _delete_text_chunks(session: Session, document_id: int) -> None:
    """
    Supprime tous les chunks texte (L1 + sections) sans toucher les L0 page_anchor ni LanceDB.
    """
    chunk_ids = _get_deletable_text_chunk_ids(session, document_id)
    _delete_chunk_foreign_relations(session, chunk_ids)

    # Feuilles texte (toutes versions du pipeline pymupdf4llm)
    session.execute(
        delete(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.is_leaf == True,  # noqa: E712
            DocumentChunk.metadata_json["content_type"].astext != CONTENT_TYPE_PAGE_ANCHOR,
        )
    )
    # Parents intermédiaires éventuels (ancien pipeline hiérarchique)
    session.execute(
        delete(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.is_leaf == False,  # noqa: E712
            DocumentChunk.metadata_json["content_type"].astext != CONTENT_TYPE_PAGE_ANCHOR,
        )
    )
    session.commit()
    from app.services.context_packer_service import invalidate_document_fulltext_cache

    invalidate_document_fulltext_cache(document_id)


# ---------------------------------------------------------------------------
# Extraction et chunking
# ---------------------------------------------------------------------------


def _get_pdf_page_count(pdf_path: str) -> int:
    import fitz
    doc = fitz.open(pdf_path)
    n = len(doc)
    doc.close()
    return n


def _extract_pages_vision(
    pdf_path: str,
    page_numbers: List[int],
    doc_title: str,
    metadata_base: dict,
) -> dict[int, List[dict]]:
    """Extraction vision (rendu PNG + mistral-small), en parallèle par page."""
    concurrency = settings.PAGE_EXTRACTION_CONCURRENCY
    specs_by_page: dict[int, List[dict]] = {}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(extract_page_chunk_specs, pdf_path, pno, doc_title, metadata_base): pno
            for pno in page_numbers
        }
        for future in as_completed(futures):
            pno = futures[future]
            try:
                specs_by_page[pno] = future.result()
            except Exception as exc:
                logger.error("[Indexing] Extraction page %s échouée : %s", pno, exc)
                specs_by_page[pno] = []

    return specs_by_page


def _extract_and_persist_chunks(
    session: Session,
    document: "Document",
    pdf_path: str,
    preserve_page_anchors: bool = False,
    extractor: TextExtractor = TextExtractor.VISION,
) -> int:
    """
    Extrait le texte (voie vision ou voie texte natif selon ``extractor``),
    crée les chunks L0 (page_anchor) + L1 (semantic_leaf) et les persiste.

    Si preserve_page_anchors=True (mode text_only), met à jour les anchors existants
    au lieu de les recréer — préserve les chunk_id liés aux patches ColPali LanceDB.
    """
    doc_id = document.id
    doc_title = document.title or ""

    page_count = _get_pdf_page_count(pdf_path)
    if page_count == 0:
        logger.warning("[Indexing] PDF vide ou illisible : %s", pdf_path)
        return 0

    page_numbers = list(range(1, page_count + 1))

    metadata_base = {
        "document_id": doc_id,
        "document_title": doc_title,
        "library_id": document.library_id,
        "user_id": document.user_id,
    }

    # --- Extraction ---
    pages_from_text = 0
    pages_from_vision = 0

    if extractor == TextExtractor.TEXT:
        from app.services.text_page_extraction_service import (
            extract_document_chunk_specs_text,
        )

        page_specs_by_page, pages_without_text = extract_document_chunk_specs_text(
            pdf_path, metadata_base
        )
        pages_from_text = len(page_specs_by_page)

        # Pages sans couche texte exploitable (scan, texte vectorisé) : la voie texte
        # ne peut rien produire — bascule sur la vision pour ne pas les perdre.
        missing = [p for p in page_numbers if p not in page_specs_by_page]
        if missing and settings.TEXT_EXTRACTION_VISION_FALLBACK:
            logger.info(
                "[Indexing] %d page(s) sans couche texte → repli vision : %s",
                len(missing),
                missing[:20],
            )
            vision_specs = _extract_pages_vision(
                pdf_path, missing, doc_title, metadata_base
            )
            for pno, specs in vision_specs.items():
                if specs:
                    page_specs_by_page[pno] = specs
                    pages_from_vision += 1
        elif missing:
            logger.warning(
                "[Indexing] %d page(s) sans couche texte et repli vision désactivé "
                "→ pages VIDES : %s",
                len(missing),
                missing[:20],
            )
    else:
        page_specs_by_page = _extract_pages_vision(
            pdf_path, page_numbers, doc_title, metadata_base
        )
        pages_from_vision = sum(1 for s in page_specs_by_page.values() if s)

    logger.info(
        "[Indexing] document_id=%s extracteur=%s — %d page(s) texte natif, %d page(s) vision",
        doc_id,
        extractor.value,
        pages_from_text,
        pages_from_vision,
    )

    # Assemblage en ordre strict page_no croissant
    all_specs_ordered: List[dict] = []
    for pno in sorted(page_specs_by_page):
        all_specs_ordered.extend(page_specs_by_page[pno])

    # --- Merge inter-pages ---
    # Uniquement en voie vision : les heuristiques de recollage (dernier caractère
    # hors .!?:, première lettre minuscule) fusionneraient des lignes de tableau
    # adjacentes, alors que la découpe markdown fournit déjà des frontières nettes.
    if extractor == TextExtractor.VISION:
        all_specs_ordered = merge_cross_page_chunks(all_specs_ordered)

    # --- L0 : un anchor minimal par page ---
    existing_anchors: dict[int, "DocumentChunk"] = {}
    if preserve_page_anchors:
        stmt = select(DocumentChunk).where(
            DocumentChunk.document_id == doc_id,
            DocumentChunk.is_leaf == False,  # noqa: E712
            DocumentChunk.metadata_json["content_type"].astext == CONTENT_TYPE_PAGE_ANCHOR,
        )
        for anchor in session.exec(stmt).all():
            meta = anchor.metadata_json or {}
            pno = meta.get("page_no")
            if pno is not None:
                existing_anchors[int(pno)] = anchor

    # Résumé L0 = heading du premier chunk vision de la page
    first_heading_by_page: dict[int, str] = {}
    for spec in all_specs_ordered:
        meta = spec.get("metadata_json") or {}
        pno = meta.get("page_no")
        if pno is not None and pno not in first_heading_by_page:
            heading = meta.get("heading") or ""
            first_heading_by_page[pno] = heading

    page_anchor_by_page: dict[int, "DocumentChunk"] = {}
    l0_chunks: List["DocumentChunk"] = []
    l0_updated = 0

    for pno in page_numbers:
        node_id = f"page-anchor-{doc_id}-{pno}"
        heading = first_heading_by_page.get(pno, "")
        content = heading or f"Page {pno} — contenu visuel uniquement"
        meta = {
            "document_id": doc_id,
            "document_title": doc_title,
            "page_no": pno,
            "page_start": pno,
            "page_end": pno,
            "content_type": CONTENT_TYPE_PAGE_ANCHOR,
            "chunking_version": CHUNKING_VERSION,
            "is_leaf": False,
        }

        existing = existing_anchors.get(pno)
        if existing is not None:
            existing.content = content
            existing.text = content
            existing.end_char = len(content)
            existing.metadata_json = meta
            existing.metadata_ = meta
            existing.source = document.source
            session.add(existing)
            page_anchor_by_page[pno] = existing
            l0_updated += 1
            continue

        chunk = DocumentChunk(
            document_id=doc_id,
            chunk_index=pno - 1,
            content=content,
            text=content,
            start_char=0,
            end_char=len(content),
            node_id=node_id,
            parent_node_id=None,
            is_leaf=False,
            hierarchy_level=0,
            metadata_json=meta,
            metadata_=meta,
            source=document.source,
        )
        l0_chunks.append(chunk)
        page_anchor_by_page[pno] = chunk

    if l0_chunks:
        session.add_all(l0_chunks)
    session.flush()

    # --- L1 : chunks sémantiques ---
    l1_chunks: List["DocumentChunk"] = []
    chunk_index_offset = page_count

    for i, spec in enumerate(all_specs_ordered):
        content = (spec.get("content") or "").strip()
        if not content:
            continue

        spec_meta = dict(spec.get("metadata_json") or {})
        # La voie texte pose sa propre chunking_version (text_page_v1) : ne l'écrase
        # pas, elle sert à distinguer les deux voies en base pour la comparaison.
        spec_meta.setdefault("chunking_version", CHUNKING_VERSION)
        spec_meta["content_type"] = CONTENT_TYPE_SEMANTIC_LEAF

        # Rattacher au page_anchor de la page de départ
        page_no_spec = spec_meta.get("page_no") or spec_meta.get("page_start")
        anchor = page_anchor_by_page.get(page_no_spec) if page_no_spec else None
        parent_node_id = anchor.node_id if anchor else spec.get("parent_node_id")
        if anchor:
            spec_meta["page_anchor_node_id"] = anchor.node_id

        chunk = DocumentChunk(
            document_id=doc_id,
            chunk_index=chunk_index_offset + i,
            content=content,
            text=content,
            start_char=0,
            end_char=len(content),
            node_id=spec.get("node_id") or str(uuid.uuid4()),
            parent_node_id=parent_node_id,
            is_leaf=True,
            hierarchy_level=1,
            metadata_json=spec_meta,
            metadata_=spec_meta,
            source=document.source,
        )
        l1_chunks.append(chunk)

    session.add_all(l1_chunks)
    session.commit()

    total = len(l0_chunks) + l0_updated + len(l1_chunks)
    logger.info(
        "[Indexing] Chunks persistés document_id=%s : %s L0 créés, %s L0 màj, %s L1 = %s total",
        doc_id,
        len(l0_chunks),
        l0_updated,
        len(l1_chunks),
        total,
    )
    return total


# ---------------------------------------------------------------------------
# Embeddings mistral-embed
# ---------------------------------------------------------------------------


# Types de chunks texte embeddés pour le retrieval dense/lexical (L1 + L2).
_EMBEDDABLE_TEXT_CONTENT_TYPES = (
    CONTENT_TYPE_SEMANTIC_LEAF,
    CONTENT_TYPE_CONTEXTUAL_ENRICHMENT,
)


def _category_labels(slugs: List[str]) -> List[str]:
    """Mappe les slugs de catégorie vers leurs labels lisibles (fallback slug)."""
    if not slugs:
        return []
    try:
        from app.services.category_catalog import DEFAULT_CATEGORY_LABELS

        return [DEFAULT_CATEGORY_LABELS.get(s, s) for s in slugs if s]
    except Exception:
        return [s for s in slugs if s]


def _build_embed_text(
    chunk: DocumentChunk,
    *,
    doc_source: Optional[str] = None,
    doc_materials: Optional[List[str]] = None,
) -> str:
    """
    Construit le texte à embedder avec un préfixe contextuel déterministe (Contextual
    Retrieval) : titre + section + catégories + entités + matériau/source.

    Les métadonnées (catégories/entités) sont injectées par la passe KAG AVANT l'embedding,
    de sorte que le vecteur dense « voit » nativement les signaux de catégorie et d'entité.
    Le préfixe n'est pas stocké dans content — seul le vecteur en bénéficie.
    """
    meta = chunk.metadata_json or {}
    doc_title = meta.get("document_title", "")
    page_no = meta.get("page_no")
    heading = meta.get("heading") or meta.get("parent_heading") or ""
    step_no = meta.get("step_number")
    theme = meta.get("theme") or ""  # chunks L2 contextual_enrichment

    categories = meta.get("categories") or []
    if not categories and meta.get("category_slug"):
        categories = [meta["category_slug"]]
    category_labels = _category_labels(categories if isinstance(categories, list) else [])

    entities = meta.get("entities") or []
    if not isinstance(entities, list):
        entities = []

    parts: List[str] = []
    if doc_title:
        parts.append(f"Document : {doc_title}.")
    if heading:
        parts.append(f"Section : {heading}.")
    elif theme:
        parts.append(f"Thème : {theme}.")
    if step_no is not None:
        parts.append(f"Étape {step_no}.")
    elif page_no is not None:
        parts.append(f"Page {page_no}.")
    if category_labels:
        parts.append(f"Catégories : {', '.join(category_labels[:4])}.")
    if entities:
        parts.append(f"Éléments : {', '.join(str(e) for e in entities[:8])}.")
    if doc_materials:
        parts.append(f"Matériau : {', '.join(doc_materials)}.")
    if doc_source:
        parts.append(f"Source : {doc_source}.")

    prefix = " ".join(parts)
    content = chunk.content or ""

    return f"{prefix}\n\n{content}".strip() if prefix else content


def _embed_text_chunks(document_id: int) -> int:
    """
    Génère les embeddings mistral-embed pour les chunks texte retrievables (L1 semantic_leaf
    + L2 contextual_enrichment) du document et les persiste en base.

    Tourne EN DERNIER (après KAG + enrichissement) afin que `_build_embed_text` intègre
    les catégories et entités dans le texte embeddé.
    Renvoie le nombre de chunks embeddés.
    """
    from app.services.embedding_service import generate_embeddings_batch

    with Session(engine) as session:
        document = session.get(Document, document_id)
        doc_source = document.source if document else None
        doc_materials = list(document.materials or []) if document else []

        statement = select(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.is_leaf == True,  # noqa: E712
        )
        all_leaves = list(session.exec(statement).all())
        chunks = [
            c
            for c in all_leaves
            if (c.metadata_json or {}).get("content_type", CONTENT_TYPE_SEMANTIC_LEAF)
            in _EMBEDDABLE_TEXT_CONTENT_TYPES
        ]

        if not chunks:
            logger.warning("[Indexing] Aucun chunk texte à embedder pour document_id=%s", document_id)
            return 0

        texts = [
            _build_embed_text(c, doc_source=doc_source, doc_materials=doc_materials)
            for c in chunks
        ]
        embeddings = generate_embeddings_batch(texts, batch_size=settings.EMBEDDING_BATCH_SIZE)

        embedded = 0
        for chunk, vector in zip(chunks, embeddings):
            if vector:
                chunk.embedding = vector
                meta = dict(chunk.metadata_json or {})
                meta["embedding_model"] = settings.EMBEDDING_MODEL
                chunk.metadata_json = meta
                chunk.metadata_ = meta
                session.add(chunk)
                embedded += 1

        session.commit()
        logger.info(
            "[Indexing] %s/%s chunks texte (L1+L2) embeddés pour document_id=%s",
            embedded,
            len(chunks),
            document_id,
        )
        return embedded


# ---------------------------------------------------------------------------
# ColPali sync ciblant les L0 page_anchor
# ---------------------------------------------------------------------------


def _sync_colpali_for_pages(document_id: int, pdf_path: str) -> int:
    """
    Génère les embeddings ColPali pour chaque page et les lie aux chunks L0 page_anchor.
    Renvoie le nombre de pages traitées.

    Contrairement à sync_document_colpali_embeddings (qui cible is_leaf=True), cette
    fonction cible les chunks content_type="page_anchor" pour le nouveau pipeline.
    """
    if not settings.COLPALI_ENABLED:
        logger.info("[Indexing] ColPali désactivé, sync ignoré.")
        return 0

    if not Path(pdf_path).is_file():
        logger.warning("[Indexing] ColPali sync : fichier introuvable %s", pdf_path)
        return 0

    from app.services.colpali_service import embed_pdf_pages_colpali
    from app.services.lancedb_service import insert_colpali_patches_batch_lancedb

    # Générer les embeddings image par page
    page_embeddings = embed_pdf_pages_colpali(pdf_path, document_id=document_id)

    with Session(engine) as session:
        statement = select(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.is_leaf == False,
            DocumentChunk.metadata_json["content_type"].astext == CONTENT_TYPE_PAGE_ANCHOR,
        )
        anchors = list(session.exec(statement).all())

    if not anchors:
        logger.warning(
            "[Indexing] Aucun chunk page_anchor trouvé pour document_id=%s — "
            "fallback sur is_leaf=True",
            document_id,
        )
        from app.services.colpali_service import sync_document_colpali_embeddings
        sync_document_colpali_embeddings(document_id)
        return len(page_embeddings)

    chunk_patches_list = []
    for anchor in anchors:
        meta = anchor.metadata_json or {}
        page_no = meta.get("page_no")
        if page_no is not None:
            page_idx = int(page_no) - 1
            if 0 <= page_idx < len(page_embeddings):
                chunk_patches_list.append((anchor.id, page_embeddings[page_idx]))

    if chunk_patches_list:
        insert_colpali_patches_batch_lancedb(document_id, chunk_patches_list)

    logger.info(
        "[Indexing] ColPali sync terminé : %s pages pour document_id=%s",
        len(chunk_patches_list),
        document_id,
    )
    return len(chunk_patches_list)


# ---------------------------------------------------------------------------
# Helpers statut document
# ---------------------------------------------------------------------------


def _set_progress(document_id: int, progress: int) -> None:
    try:
        with Session(engine) as session:
            doc = session.get(Document, document_id)
            if doc:
                doc.processing_progress = progress
                doc.updated_at = datetime.utcnow()
                session.add(doc)
                session.commit()
    except Exception as exc:
        logger.debug("[Indexing] _set_progress échoué document_id=%s: %s", document_id, exc)


def _finalize_document(document_id: int, chunk_count: int) -> None:
    try:
        with Session(engine) as session:
            doc = session.get(Document, document_id)
            if doc:
                doc.processing_status = "completed"
                doc.processing_progress = 100
                doc.updated_at = datetime.utcnow()
                session.add(doc)
                session.commit()
        # Les chunks viennent d'être réécrits : purger le cache fulltext du packer CAG
        # pour que la génération voie immédiatement le nouveau contenu.
        from app.services.context_packer_service import invalidate_document_fulltext_cache

        invalidate_document_fulltext_cache(document_id)
        try:
            from app.services.discord_service import notify_document_status
            with Session(engine) as session:
                doc = session.get(Document, document_id)
                title = doc.title if doc else "Sans titre"
            notify_document_status(
                document_id=document_id,
                document_title=title,
                status="completed",
                chunks_count=chunk_count,
            )
        except Exception:
            pass
    except Exception as exc:
        logger.error("[Indexing] _finalize_document échoué document_id=%s: %s", document_id, exc)


def _mark_failed(document_id: int, error_message: str) -> None:
    try:
        with Session(engine) as session:
            doc = session.get(Document, document_id)
            if doc and doc.processing_status != "failed":
                doc.processing_status = "failed"
                doc.last_processing_error = error_message[:500]
                doc.updated_at = datetime.utcnow()
                session.add(doc)
                session.commit()
                try:
                    from app.services.discord_service import notify_document_status
                    notify_document_status(
                        document_id=document_id,
                        document_title=doc.title or "Sans titre",
                        status="failed",
                        error_message=error_message,
                    )
                except Exception:
                    pass
    except Exception as exc:
        logger.error("[Indexing] _mark_failed échoué document_id=%s: %s", document_id, exc)
