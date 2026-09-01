"""
Pipeline d'indexation documentaire unifié.

Quatre modes, pensés pour pouvoir séparer un passage RAPIDE d'un passage LOURD :
  - full            : extraction + chunks contextuels + ColPali (tout)
  - text_only       : extraction SEULEMENT (ni chunks contextuels, ColPali inchangé)
                      → rapide, en journée sur tout le corpus
  - enrichment_only : chunks contextuels (fenêtres de 3 pages, texte + vision) sur les
                      chunks EXISTANTS → lent (appels LLM par batch), à lancer
                      séparément (typiquement la nuit)
  - colpali_only    : re-sync ColPali uniquement (chunks texte inchangés)

text_only puis enrichment_only aboutit au MÊME état final que full.

Deux voies d'extraction du texte (paramètre ``extractor``) :
  - vision : rendu PNG + mistral-small (gère les pages sans couche texte)
  - text   : couche texte native pymupdf4llm, avec repli vision par page
"""
from __future__ import annotations

import logging
import re
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

# Postgres/psycopg refuse tout octet NUL (0x00) dans une colonne texte (String/Text) —
# vérification CLIENT-SIDE avant même l'envoi au serveur. JSONB n'a pas ce problème
# (json.dumps échappe le NUL en séquence JSON valide), donc seuls `content`/`text`
# sont concernés. Certains PDF convertis ou mal encodés (police corrompue, table CID
# cassée) embarquent des NUL dans leur couche texte native : rare par page, mais
# quasi certain sur un document de plusieurs centaines de pages — et SANS ce
# nettoyage, l'unique octet fautif fait échouer le commit de TOUS les chunks du
# document, y compris ceux des centaines de pages déjà extraites avec succès.
_CONTROL_CHARS_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _strip_db_unsafe_chars(value: str) -> str:
    """Retire le NUL et les autres caractères de contrôle non imprimables avant
    insertion en base. Conserve \\n, \\t, \\r (hors de la plage exclue)."""
    if not value:
        return value
    return _CONTROL_CHARS_RE.sub("", value)


class IndexingMode(str, Enum):
    FULL = "full"
    TEXT_ONLY = "text_only"
    COLPALI_ONLY = "colpali_only"
    # Chunks contextuels seuls : régénère les synthèses L2 (fenêtres de 3 pages,
    # texte + vision) sur les chunks EXISTANTS, puis ré-embarque. Ni extraction texte,
    # ni ColPali. C'est le passage LOURD, séparé du passage rapide text_only.
    ENRICHMENT_ONLY = "enrichment_only"


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
    enrichment_stats: dict = {"chunks": 0, "status": "disabled"}

    try:
        if mode in (IndexingMode.FULL, IndexingMode.TEXT_ONLY, IndexingMode.ENRICHMENT_ONLY):
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
                        extractor=extractor,
                    )
            elif mode == IndexingMode.ENRICHMENT_ONLY:
                # Chunks EXISTANTS : on compte seulement (pour finalize). Les synthèses L2
                # sont purgées puis régénérées par run_contextual_enrichment_for_document,
                # donc rien à nettoyer ici — et l'index ColPali reste intact.
                _set_progress(document_id, 30)
                ld.info(
                    "[Indexing] Chunks contextuels seuls — chunks existants document_id=%s",
                    document_id,
                )
                with Session(engine) as session:
                    chunk_count = session.execute(
                        text("SELECT count(*) FROM documentchunk WHERE document_id = :d AND is_leaf = true"),
                        {"d": document_id},
                    ).scalar() or 0

            if _aborted():
                return {"document_id": document_id, "status": "aborted", "reason": "stale_run"}

            # Couche sémantique lourde (chunks contextuels) : plusieurs appels LLM par
            # batch de 3 pages, soit l'essentiel du temps de traitement. SAUTÉE en
            # text_only — qui devient un passage rapide « extraction seule »
            # utilisable en journée sur tout le corpus — et portée par enrichment_only,
            # lancé séparément (typiquement la nuit). Enchaîner les deux aboutit au même
            # état final que full.
            runs_semantic_layers = mode in (
                IndexingMode.FULL,
                IndexingMode.ENRICHMENT_ONLY,
            )

            # --- 2. Chunks contextuels inter-pages → L2 contextual_enrichment ---
            # Idempotent : run_contextual_enrichment_for_document purge les L2 existants
            # avant de régénérer, donc relancer enrichment_only ne duplique rien.
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

        if mode in (IndexingMode.FULL, IndexingMode.COLPALI_ONLY):
            if _aborted():
                return {"document_id": document_id, "status": "aborted", "reason": "stale_run"}

            _set_progress(document_id, 90)
            ld.info("[Indexing] ColPali sync document_id=%s", document_id)
            sync_colpali_page_anchors(document_id, pdf_path)

        _finalize_document(document_id, chunk_count)
        semantic_ran = mode in (IndexingMode.FULL, IndexingMode.ENRICHMENT_ONLY)
        ld.info(
            "[Indexing] FIN OK document_id=%s mode=%s chunks=%s enrichment=%s",
            document_id,
            mode.value,
            chunk_count,
            enrichment_stats if semantic_ran else "n/a (text_only)",
        )
        result = {"document_id": document_id, "chunks": chunk_count, "status": "completed"}
        if semantic_ran:
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
    """Supprime les lignes qui référencent ces chunks par clé étrangère.

    INCONDITIONNEL : `chunkentityrelation.chunk_id` et `chunkcategoryrelation.chunk_id`
    n'ont PAS de `ON DELETE CASCADE`. Conditionner cette purge à un flag (c'était le cas
    de `KAG_ENABLED` avant le 2026-07-28) fait échouer toute suppression de chunk avec
    une violation de contrainte dès qu'il reste d'anciennes lignes en base — donc tout
    retraitement du document.

    Les tables KAG sont conservées le temps de la transition : cette purge est ce qui
    permet de retraiter des documents encore porteurs d'anciennes relations.
    """
    if not chunk_ids:
        return

    chunk_ids_tuple = tuple(chunk_ids)
    for table in ("chunkentityrelation", "chunkcategoryrelation", "entityentityrelation"):
        column = "source_chunk_id" if table == "entityentityrelation" else "chunk_id"
        try:
            with session.begin_nested():
                session.execute(
                    text(f"DELETE FROM {table} WHERE {column} IN :chunk_ids"),
                    {"chunk_ids": chunk_ids_tuple},
                )
        except Exception as exc:  # table absente (déjà supprimée) → sans objet
            logger.debug(
                "[Indexing] Purge %s ignorée (%s chunk(s)) : %s",
                table,
                len(chunk_ids_tuple),
                exc,
            )


def _all_chunk_ids_for_document(session: Session, document_id: int) -> List[int]:
    """Tous les chunk_id d'un document (utilisé avant purge des relations FK)."""
    return [
        int(row[0])
        for row in session.execute(
            text("SELECT id FROM documentchunk WHERE document_id = :doc_id"),
            {"doc_id": document_id},
        ).all()
    ]


def _delete_all_chunks(session: Session, document_id: int) -> None:
    """Supprime tous les chunks PostgreSQL, leurs relations FK et les patches LanceDB."""
    _delete_chunk_foreign_relations(
        session, _all_chunk_ids_for_document(session, document_id)
    )

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


def ensure_page_anchors(
    session: Session,
    document: "Document",
    page_count: int,
    *,
    headings_by_page: Optional[dict] = None,
    update_existing: bool = False,
) -> tuple[dict, int, int]:
    """Garantit un chunk L0 ``page_anchor`` par page — cible UNIQUE des patches ColPali.

    Charge les anchors existants, crée les manquants, et ne réécrit le contenu des
    existants que si ``update_existing`` (pipeline texte : le heading de page vient
    d'être ré-extrait). Les chunk_id existants sont TOUJOURS préservés — ce sont eux
    que les patches LanceDB référencent.

    Retourne ``(anchors_par_page, nb_créés, nb_mis_à_jour)``. ``session.flush()`` est
    appelé pour matérialiser les ids ; le commit reste à la charge de l'appelant.
    """
    doc_id = document.id
    doc_title = document.title or ""
    headings_by_page = headings_by_page or {}

    existing_anchors: dict[int, DocumentChunk] = {}
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

    anchors_by_page: dict[int, DocumentChunk] = {}
    created: List[DocumentChunk] = []
    updated = 0

    for pno in range(1, page_count + 1):
        heading = _strip_db_unsafe_chars(headings_by_page.get(pno, "") or "")
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
            if update_existing:
                existing.content = content
                existing.text = content
                existing.end_char = len(content)
                existing.metadata_json = meta
                existing.metadata_ = meta
                existing.source = document.source
                session.add(existing)
                updated += 1
            anchors_by_page[pno] = existing
            continue

        chunk = DocumentChunk(
            document_id=doc_id,
            chunk_index=pno - 1,
            content=content,
            text=content,
            start_char=0,
            end_char=len(content),
            node_id=f"page-anchor-{doc_id}-{pno}",
            parent_node_id=None,
            is_leaf=False,
            hierarchy_level=0,
            metadata_json=meta,
            metadata_=meta,
            source=document.source,
        )
        created.append(chunk)
        anchors_by_page[pno] = chunk

    if created:
        session.add_all(created)
    session.flush()
    return anchors_by_page, len(created), updated


def _extract_and_persist_chunks(
    session: Session,
    document: "Document",
    pdf_path: str,
    extractor: TextExtractor = TextExtractor.VISION,
) -> int:
    """
    Extrait le texte (voie vision ou voie texte natif selon ``extractor``),
    crée les chunks L0 (page_anchor) + L1 (semantic_leaf) et les persiste.

    Les anchors existants sont mis à jour (jamais recréés) — préserve les chunk_id
    liés aux patches ColPali LanceDB, quel que soit le mode d'indexation.
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

    # Résumé L0 = heading du premier chunk vision de la page
    first_heading_by_page: dict[int, str] = {}
    for spec in all_specs_ordered:
        meta = spec.get("metadata_json") or {}
        pno = meta.get("page_no")
        if pno is not None and pno not in first_heading_by_page:
            heading = meta.get("heading") or ""
            first_heading_by_page[pno] = heading

    # --- L0 : un anchor minimal par page (création + mise à jour idempotentes) ---
    page_anchor_by_page, l0_created, l0_updated = ensure_page_anchors(
        session,
        document,
        page_count,
        headings_by_page=first_heading_by_page,
        update_existing=True,
    )

    # --- L1 : chunks sémantiques ---
    l1_chunks: List["DocumentChunk"] = []
    chunk_index_offset = page_count

    for i, spec in enumerate(all_specs_ordered):
        content = _strip_db_unsafe_chars((spec.get("content") or "").strip())
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

    total = l0_created + l0_updated + len(l1_chunks)
    logger.info(
        "[Indexing] Chunks persistés document_id=%s : %s L0 créés, %s L0 màj, %s L1 = %s total",
        doc_id,
        l0_created,
        l0_updated,
        len(l1_chunks),
        total,
    )
    return total




# ---------------------------------------------------------------------------
# ColPali : sync + réparation de topologie (1 page = 1 jeu de patches sur l'anchor)
# ---------------------------------------------------------------------------


def sync_colpali_page_anchors(document_id: int, pdf_path: str) -> int:
    """
    Génère les embeddings ColPali de chaque page et les lie aux chunks L0 page_anchor.
    Renvoie le nombre de pages traitées.

    Topologie UNIQUE du visuel : 1 page = 1 jeu de patches, rattaché à l'anchor de la
    page. Les anchors manquants sont CRÉÉS (document jamais passé par l'extraction
    texte) — plus aucun repli vers les chunks feuilles, qui dupliquait chaque page.
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
    if not page_embeddings:
        logger.warning("[Indexing] ColPali sync : aucun embedding produit pour %s", pdf_path)
        return 0

    with Session(engine) as session:
        document = session.get(Document, document_id)
        if not document:
            logger.warning("[Indexing] ColPali sync : document %s introuvable", document_id)
            return 0
        anchors_by_page, created, _ = ensure_page_anchors(
            session, document, len(page_embeddings)
        )
        session.commit()
        anchor_id_by_page = {pno: chunk.id for pno, chunk in anchors_by_page.items()}

    if created:
        logger.info(
            "[Indexing] %d anchor(s) page créés pour ColPali document_id=%s",
            created,
            document_id,
        )

    chunk_patches_list = [
        (anchor_id_by_page[pno], page_embeddings[pno - 1])
        for pno in sorted(anchor_id_by_page)
        if 0 <= pno - 1 < len(page_embeddings)
    ]
    if chunk_patches_list:
        insert_colpali_patches_batch_lancedb(document_id, chunk_patches_list)

    logger.info(
        "[Indexing] ColPali sync terminé : %s pages pour document_id=%s",
        len(chunk_patches_list),
        document_id,
    )
    return len(chunk_patches_list)


def repair_colpali_topology(document_id: int) -> dict:
    """Ré-attache les patches ColPali existants aux anchors de page, SANS ré-embedding.

    Répare l'héritage de l'ancien pipeline feuille (chaque chunk texte d'une page
    portait une copie complète des patches de la page) : pour chaque page on garde UN
    jeu de patches — les copies sont identiques entre elles — et on le réinsère sous
    l'anchor. Coût : I/O LanceDB uniquement, aucun passage du modèle.

    Statuts retournés :
      - ok         : topologie déjà saine, rien à faire ;
      - repaired   : réécriture effectuée, toutes les pages à patches couvertes ;
      - incomplete : réécriture faite mais pages sans patches → colpali_only requis ;
      - empty      : aucun patch en base pour ce document ;
      - error      : scan LanceDB ou document indisponible.
    """
    from app.services.lancedb_service import (
        fetch_colpali_patch_vectors_for_chunks,
        get_colpali_chunk_ids_by_document,
        insert_colpali_patches_batch_lancedb,
    )

    lancedb_ids = get_colpali_chunk_ids_by_document([document_id]).get(document_id)
    if lancedb_ids is None:
        return {"document_id": document_id, "status": "error", "reason": "lancedb_scan_failed"}
    if not lancedb_ids:
        return {"document_id": document_id, "status": "empty"}

    with Session(engine) as session:
        document = session.get(Document, document_id)
        if not document:
            return {"document_id": document_id, "status": "error", "reason": "document_missing"}

        rows = session.execute(
            text(
                """
                SELECT id,
                       COALESCE(
                           (metadata_json->>'page_no')::int,
                           (metadata_json->>'page_start')::int,
                           (metadata_->>'page_no')::int,
                           (metadata_->>'page_start')::int
                       ) AS page_no,
                       COALESCE(metadata_json->>'content_type', metadata_->>'content_type', '') AS content_type
                FROM documentchunk
                WHERE id IN :ids
                """
            ),
            {"ids": tuple(lancedb_ids)},
        ).all()
        page_by_chunk = {int(r.id): int(r.page_no) for r in rows if r.page_no is not None}
        anchor_chunk_ids = {
            int(r.id) for r in rows if r.content_type == CONTENT_TYPE_PAGE_ANCHOR
        }
        existing_ids = {int(r.id) for r in rows}
        orphan_ids = set(lancedb_ids) - existing_ids
        unresolved_ids = {int(r.id) for r in rows if r.page_no is None}

        page_count = max(page_by_chunk.values(), default=0)
        if document.source_file_path and Path(document.source_file_path).is_file():
            real_count = _get_pdf_page_count(document.source_file_path)
            if real_count:
                page_count = max(page_count, real_count)

        if page_count:
            anchors_by_page, created, _ = ensure_page_anchors(session, document, page_count)
            session.commit()
        else:
            anchors_by_page, created = {}, 0
        anchor_id_by_page = {pno: chunk.id for pno, chunk in anchors_by_page.items()}
        anchor_id_set = set(anchor_id_by_page.values())

    # Topologie déjà saine : toutes les cibles LanceDB sont des anchors, zéro orphelin.
    if not orphan_ids and not unresolved_ids and set(lancedb_ids) <= anchor_id_set:
        return {"document_id": document_id, "status": "ok", "pages": len(lancedb_ids)}

    # UN chunk source par page : l'anchor lui-même s'il porte déjà des patches,
    # sinon le plus petit id (déterministe — les copies sont identiques).
    source_by_page: dict[int, int] = {}
    for c_id, pno in page_by_chunk.items():
        current = source_by_page.get(pno)
        if current is None:
            source_by_page[pno] = c_id
            continue
        current_is_anchor = current in anchor_chunk_ids
        candidate_is_anchor = c_id in anchor_chunk_ids
        if candidate_is_anchor and not current_is_anchor:
            source_by_page[pno] = c_id
        elif candidate_is_anchor == current_is_anchor and c_id < current:
            source_by_page[pno] = c_id

    vectors_by_chunk = fetch_colpali_patch_vectors_for_chunks(
        sorted(set(source_by_page.values()))
    )

    chunk_patches_list = []
    unreadable_pages: List[int] = []
    for pno in sorted(source_by_page):
        anchor_id = anchor_id_by_page.get(pno)
        # `patches` est un tableau NumPy : tester sa longueur, pas sa vérité booléenne.
        patches = vectors_by_chunk.get(source_by_page[pno])
        if anchor_id is None or patches is None or len(patches) == 0:
            unreadable_pages.append(pno)
            continue
        chunk_patches_list.append((anchor_id, patches))

    stats = {
        "document_id": document_id,
        "targets_before": len(lancedb_ids),
        "orphan_targets": len(orphan_ids),
        "unresolved_targets": len(unresolved_ids),
        "anchors_created": created,
        "pages_with_patches": len(chunk_patches_list),
        "expected_pages": len(anchor_id_by_page),
    }

    if not chunk_patches_list:
        # Rien de récupérable (ex. uniquement des orphelins) : on purge pour que le
        # health signale « missing » plutôt qu'un faux desync éternel.
        from app.services.lancedb_service import delete_colpali_patches_for_document

        delete_colpali_patches_for_document(document_id)
        return {**stats, "status": "incomplete", "reason": "no_resolvable_patches"}

    # delete document + insert : la réécriture est portée par l'insert batch.
    insert_colpali_patches_batch_lancedb(document_id, chunk_patches_list)

    missing_pages = sorted(set(anchor_id_by_page) - set(source_by_page)) + unreadable_pages
    status = "incomplete" if missing_pages else "repaired"
    logger.info(
        "[ColPali repair] document_id=%s : %d cible(s) → %d page(s) sur anchors "
        "(orphelins=%d, pages manquantes=%d)",
        document_id,
        len(lancedb_ids),
        len(chunk_patches_list),
        len(orphan_ids),
        len(missing_pages),
    )
    return {**stats, "status": status, "missing_pages": missing_pages[:20]}


def ensure_colpali_page_sync(document_id: int, pdf_path: str) -> dict:
    """Garantit la topologie ColPali d'un document SANS ré-embedding inutile.

    Appelé par le retraitement multimodal : le PDF n'ayant pas changé, les patches
    existants restent valides. Ordre :
      1. patches déjà tous sur les anchors et complets → rien à faire ;
      2. patches présents mais mal rattachés → réparation in-place (I/O seulement) ;
      3. pages manquantes après réparation, ou aucun patch → embedding complet.
    """
    if not settings.COLPALI_ENABLED:
        return {"document_id": document_id, "status": "disabled"}

    from app.services.lancedb_service import get_colpali_chunk_ids_by_document

    lancedb_ids = get_colpali_chunk_ids_by_document([document_id]).get(document_id)
    page_count = _get_pdf_page_count(pdf_path) if Path(pdf_path).is_file() else 0

    if lancedb_ids:
        repair = repair_colpali_topology(document_id)
        if repair.get("status") in ("ok", "repaired"):
            after = get_colpali_chunk_ids_by_document([document_id]).get(document_id) or set()
            if len(after) >= page_count and after:
                logger.info(
                    "[Indexing] ColPali déjà à jour document_id=%s (%d pages, %s) — pas de ré-embedding",
                    document_id,
                    len(after),
                    repair["status"],
                )
                return {"document_id": document_id, "status": repair["status"], "pages": len(after)}
        logger.info(
            "[Indexing] ColPali incomplet après réparation (%s) — re-sync complet document_id=%s",
            repair.get("status"),
            document_id,
        )

    pages = sync_colpali_page_anchors(document_id, pdf_path)
    return {"document_id": document_id, "status": "resynced", "pages": pages}


def repair_all_colpali_topologies() -> dict:
    """Répare la topologie ColPali de tous les documents ayant des patches LanceDB.

    Réparation in-place uniquement (aucun embedding) : les documents ``incomplete``
    restent listés pour un passage ``colpali_only`` explicite.
    """
    from app.services.lancedb_service import get_colpali_document_ids

    doc_ids = get_colpali_document_ids()
    if doc_ids is None:
        return {"status": "error", "reason": "lancedb_scan_failed"}

    ld = get_library_document_logger()
    summary = {
        "status": "completed",
        "documents": len(doc_ids),
        "ok": 0,
        "repaired": 0,
        "incomplete": 0,
        "empty": 0,
        "error": 0,
        "details": [],
    }
    for doc_id in doc_ids:
        result = repair_colpali_topology(doc_id)
        status = result.get("status", "error")
        summary[status] = summary.get(status, 0) + 1
        if status not in ("ok", "empty"):
            summary["details"].append(result)

    # Les réécritures ont laissé des vecteurs hors index : on les y intègre UNE fois
    # ici, jamais par document (une reconstruction complète coûte ~90 s sur 1,6 M).
    if summary.get("repaired") or summary.get("incomplete"):
        from app.services.lancedb_service import optimize_colpali_index

        summary["index_optimize"] = optimize_colpali_index()

    ld.info(
        "[ColPali repair] passe globale : %d document(s) — ok=%d réparés=%d "
        "incomplets=%d vides=%d erreurs=%d",
        summary["documents"],
        summary["ok"],
        summary["repaired"],
        summary["incomplete"],
        summary["empty"],
        summary["error"],
    )
    return summary


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
