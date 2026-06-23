"""
Enrichissement contextuel inter-pages : synthèse factuelle par thème/catégorie.

Troisième passe après KAG, fenêtre glissante de 3 pages :
  1. Texte L1 transcrit + catégories/entités déjà extraites
  2. Appel LLM texte → chunks de synthèse documentaire (1 par notion/thème)
  3. Persistance content_type=contextual_enrichment + embedding

Ces chunks servent au retrieval (vectoriel + BM25) comme contexte complémentaire,
jamais comme preuve absolue — toujours rattachés aux pages sources.
"""

from __future__ import annotations

import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import text
from sqlmodel import Session, select

from app.config import settings
from app.database import engine
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.models.chunk_category_relation import ChunkCategoryRelation
from app.services.kag_extraction_service import build_kag_batches

logger = logging.getLogger(__name__)

CONTEXTUAL_ENRICHMENT_VERSION = "contextual_enrichment_v1"
CONTENT_TYPE_CONTEXTUAL_ENRICHMENT = "contextual_enrichment"

_ENRICHMENT_SYSTEM_PROMPT = """Tu es rédacteur technique documentaire pour documents menuiserie / profilés PVC-alu.
On te donne le texte transcrit de plusieurs pages consécutives d'un document technique,
les catégories détectées et les entités nommées extraites.

Produis des chunks de synthèse factuels, un par thème ou catégorie pertinent détecté dans le batch.

Règles absolues :
1. Renvoie UNIQUEMENT un objet JSON valide (aucun texte hors JSON).
2. Utilise les noms de produits, gammes, références et normes EXACTEMENT tels qu'ils apparaissent dans le texte source.
3. Jamais de déictiques : interdiction de "ce schéma", "ce profil", "celui-ci", "l'image montre", "voir ci-dessus".
   Remplace par le nom précis ("le profilé PVC Kömmerling 76 AD", "la gâche OB Droite", "la gamme Perform", etc.).
4. Texte directement réutilisable par un LLM RAG : autonome, dense, factuel, style notice technique.
5. Chaque chunk = 1 notion/thème. Maximum 600 tokens (~2200 caractères) par chunk.
6. Ne pas inventer d'information absente du texte source.
7. Croise les informations réparties sur les pages du batch quand elles concernent le même thème.
8. Le category_slug doit correspondre à une catégorie détectée dans le batch (slug exact).
9. source_page = numéro de la page principale où le thème est le plus documenté.

Format de réponse OBLIGATOIRE :
{{
  "enrichment_chunks": [
    {{
      "category_slug": "mounting",
      "theme": "Pose du profil seuil Profine 76",
      "content": "...",
      "source_page": 2
    }}
  ]
}}"""

_ENRICHMENT_USER_TEMPLATE = (
    "Document : {title}\nPages du batch : {page_range}\n\n"
    "Texte transcrit par page :\n{page_text}\n\n"
    "Catégories détectées par page :\n{categories_text}\n\n"
    "Entités extraites par page :\n{entities_text}\n\n"
    "Produis les chunks de synthèse factuels selon les règles du système."
)


class EnrichmentChunkItem(BaseModel):
    category_slug: str
    theme: str
    content: str
    source_page: int = Field(ge=1)


class BatchEnrichmentResponse(BaseModel):
    enrichment_chunks: List[EnrichmentChunkItem] = Field(default_factory=list)


def _enrichment_model() -> str:
    return settings.CONTEXTUAL_ENRICHMENT_MODEL or settings.MODEL_FAST


def _load_semantic_chunks_by_page(session: Session, document_id: int) -> Dict[int, List[DocumentChunk]]:
    stmt = select(DocumentChunk).where(
        DocumentChunk.document_id == document_id,
        DocumentChunk.is_leaf == True,  # noqa: E712
    )
    by_page: Dict[int, List[DocumentChunk]] = {}
    for chunk in session.exec(stmt).all():
        meta = chunk.metadata_json or {}
        if meta.get("content_type") != "semantic_leaf":
            continue
        page_no = meta.get("page_no") or meta.get("page_start")
        if page_no is None:
            continue
        by_page.setdefault(int(page_no), []).append(chunk)
    for pno in by_page:
        by_page[pno].sort(key=lambda c: (c.chunk_index or 0, c.id or 0))
    return by_page


def _load_page_anchors(session: Session, document_id: int) -> Dict[int, DocumentChunk]:
    stmt = select(DocumentChunk).where(
        DocumentChunk.document_id == document_id,
        DocumentChunk.is_leaf == False,  # noqa: E712
    )
    anchors: Dict[int, DocumentChunk] = {}
    for chunk in session.exec(stmt).all():
        meta = chunk.metadata_json or {}
        if meta.get("content_type") != "page_anchor":
            continue
        page_no = meta.get("page_no")
        if page_no is not None:
            anchors[int(page_no)] = chunk
    return anchors


def _load_categories_by_page(
    session: Session,
    document_id: int,
) -> Dict[int, List[str]]:
    rows = session.execute(
        text(
            """
            SELECT ccr.page_no, dc.slug
            FROM chunkcategoryrelation ccr
            INNER JOIN documentcategory dc ON dc.id = ccr.category_id
            WHERE ccr.document_id = :doc_id
            ORDER BY ccr.page_no, dc.slug
            """
        ),
        {"doc_id": document_id},
    ).all()
    by_page: Dict[int, List[str]] = {}
    for page_no, slug in rows:
        pno = int(page_no or 0)
        by_page.setdefault(pno, [])
        if slug and slug not in by_page[pno]:
            by_page[pno].append(slug)
    return by_page


def _load_entities_by_page(
    session: Session,
    document_id: int,
) -> Dict[int, List[str]]:
    rows = session.execute(
        text(
            """
            SELECT
                COALESCE(
                    (dc2.metadata_json->>'page_no')::int,
                    (dc2.metadata_->>'page_no')::int
                ) AS page_no,
                ke.name
            FROM chunkentityrelation cer
            INNER JOIN documentchunk dc2 ON dc2.id = cer.chunk_id
            INNER JOIN knowledgeentity ke ON ke.id = cer.entity_id
            WHERE dc2.document_id = :doc_id
            """
        ),
        {"doc_id": document_id},
    ).all()
    by_page: Dict[int, List[str]] = {}
    for page_no, name in rows:
        if not page_no or not name:
            continue
        pno = int(page_no)
        by_page.setdefault(pno, [])
        if name not in by_page[pno]:
            by_page[pno].append(name)
    return by_page


def _format_batch_context(
    batch_pages: List[int],
    chunks_by_page: Dict[int, List[DocumentChunk]],
    categories_by_page: Dict[int, List[str]],
    entities_by_page: Dict[int, List[str]],
) -> Tuple[str, str, str]:
    page_text_parts: List[str] = []
    cat_parts: List[str] = []
    ent_parts: List[str] = []

    for pno in batch_pages:
        chunks = chunks_by_page.get(pno) or []
        texts = []
        for idx, chunk in enumerate(chunks):
            content = (chunk.content or chunk.text or "").strip()
            if content:
                meta = chunk.metadata_json or {}
                heading = meta.get("heading") or "null"
                texts.append(f"[chunk_index={idx}] {heading}\n{content}")
        page_text_parts.append(f"--- PAGE {pno} ---\n" + ("\n\n".join(texts) if texts else "(vide)"))

        cats = categories_by_page.get(pno) or []
        cat_parts.append(f"Page {pno}: {', '.join(cats) if cats else 'aucune'}")

        ents = entities_by_page.get(pno) or []
        ent_parts.append(f"Page {pno}: {', '.join(ents[:15]) if ents else 'aucune'}")

    return (
        "\n\n".join(page_text_parts),
        "\n".join(cat_parts),
        "\n".join(ent_parts),
    )


def _call_enrichment_api(
    document_title: str,
    batch_pages: List[int],
    page_text: str,
    categories_text: str,
    entities_text: str,
) -> dict:
    from app.services.multimodal_page_service import (
        _mistral_chat_completion,
        _parse_json_with_repair,
    )

    page_range = f"{batch_pages[0]}-{batch_pages[-1]}" if len(batch_pages) > 1 else str(batch_pages[0])
    user_text = _ENRICHMENT_USER_TEMPLATE.format(
        title=document_title or "Document",
        page_range=page_range,
        page_text=page_text[:20000],
        categories_text=categories_text[:4000],
        entities_text=entities_text[:4000],
    )

    messages = [
        {"role": "system", "content": _ENRICHMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    raw = _mistral_chat_completion(
        messages,
        page_no=batch_pages[0],
        max_tokens=settings.CONTEXTUAL_ENRICHMENT_MAX_TOKENS,
        temperature=0.0,
        response_format_json=True,
        timeout_seconds=settings.CONTEXTUAL_ENRICHMENT_TIMEOUT,
        model=_enrichment_model(),
    )
    return _parse_json_with_repair(raw)


def _coerce_enrichment_response(
    raw: dict,
    batch_pages: List[int],
    valid_category_slugs: frozenset[str],
) -> BatchEnrichmentResponse:
    payload = dict(raw or {})
    items_in = payload.get("enrichment_chunks")
    if not isinstance(items_in, list):
        raise ValueError("Réponse enrichissement invalide : champ 'enrichment_chunks' absent")

    items: List[EnrichmentChunkItem] = []
    for item in items_in:
        if not isinstance(item, dict):
            continue
        slug = (item.get("category_slug") or "").strip().lower().replace(" ", "_")
        if slug and slug not in valid_category_slugs:
            continue
        content = (item.get("content") or "").strip()
        theme = (item.get("theme") or "").strip()
        if not content or not theme or not slug:
            continue
        source_page = int(item.get("source_page") or batch_pages[len(batch_pages) // 2])
        if source_page not in batch_pages:
            source_page = batch_pages[len(batch_pages) // 2]
        items.append(
            EnrichmentChunkItem(
                category_slug=slug,
                theme=theme,
                content=content,
                source_page=source_page,
            )
        )

    if not items:
        raise ValueError(f"Aucun chunk d'enrichissement valide pour batch {batch_pages}")
    return BatchEnrichmentResponse(enrichment_chunks=items)


def extract_batch_enrichment_response(
    batch_pages: List[int],
    document_title: str,
    chunks_by_page: Dict[int, List[DocumentChunk]],
    categories_by_page: Dict[int, List[str]],
    entities_by_page: Dict[int, List[str]],
    valid_category_slugs: frozenset[str],
) -> Optional[BatchEnrichmentResponse]:
    if not batch_pages:
        return None

    page_text, categories_text, entities_text = _format_batch_context(
        batch_pages,
        chunks_by_page,
        categories_by_page,
        entities_by_page,
    )
    if not page_text.strip():
        return None

    try:
        for attempt in range(2):
            try:
                raw = _call_enrichment_api(
                    document_title,
                    batch_pages,
                    page_text,
                    categories_text,
                    entities_text,
                )
                return _coerce_enrichment_response(raw, batch_pages, valid_category_slugs)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                if attempt == 0:
                    logger.warning(
                        "[Enrichment] Validation batch %s échouée (tentative 1) : %s — retry",
                        batch_pages,
                        exc,
                    )
                    continue
                logger.warning("[Enrichment] Validation batch %s échouée : %s", batch_pages, exc)
                return None
    except Exception as exc:
        logger.warning("[Enrichment] Extraction batch %s échouée : %s", batch_pages, exc)
        return None
    return None


def _get_document_space_ids(session: Session, document_id: int) -> List[int]:
    stmt = select(DocumentSpace.space_id).where(DocumentSpace.document_id == document_id)
    return list(session.exec(stmt).all())


def _next_chunk_index(session: Session, document_id: int) -> int:
    row = session.execute(
        text(
            "SELECT COALESCE(MAX(chunk_index), -1) FROM documentchunk WHERE document_id = :doc_id"
        ),
        {"doc_id": document_id},
    ).first()
    return int(row[0] if row else -1) + 1


def _persist_enrichment_chunks(
    session: Session,
    document: Document,
    space_ids: List[int],
    batch_pages: List[int],
    enrichment_items: List[EnrichmentChunkItem],
    page_anchors: Dict[int, DocumentChunk],
    category_id_by_slug: Dict[str, int],
    start_chunk_index: int,
) -> int:
    created = 0
    chunk_index = start_chunk_index
    central_page = batch_pages[len(batch_pages) // 2]
    anchor = page_anchors.get(central_page)
    parent_node_id = anchor.node_id if anchor else None

    for item in enrichment_items:
        source_page = item.source_page if item.source_page in batch_pages else central_page
        node_id = str(uuid.uuid4())
        meta = {
            "document_id": document.id,
            "document_title": document.title or "",
            "content_type": CONTENT_TYPE_CONTEXTUAL_ENRICHMENT,
            "chunk_type": CONTENT_TYPE_CONTEXTUAL_ENRICHMENT,
            "is_contextual_enrichment": True,
            "enrichment_version": CONTEXTUAL_ENRICHMENT_VERSION,
            "enrichment_model": _enrichment_model(),
            "category_slug": item.category_slug,
            "theme": item.theme,
            "source_pages": batch_pages,
            "source_page": source_page,
            "page_no": source_page,
            "page_start": min(batch_pages),
            "page_end": max(batch_pages),
            "page_anchor_node_id": parent_node_id,
            "is_leaf": True,
            "node_id": node_id,
            "parent_node_id": parent_node_id,
        }

        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=chunk_index,
            content=item.content,
            text=item.content,
            start_char=0,
            end_char=len(item.content),
            node_id=node_id,
            parent_node_id=parent_node_id,
            is_leaf=True,
            hierarchy_level=2,
            metadata_json=meta,
            metadata_=meta,
            source=document.source,
        )
        session.add(chunk)
        session.flush()

        slug_meta = dict(meta)
        slug_meta["categories"] = [item.category_slug]
        chunk.metadata_json = slug_meta
        chunk.metadata_ = slug_meta
        session.add(chunk)

        category_id = category_id_by_slug.get(item.category_slug)
        if category_id is not None:
            for space_id in space_ids:
                session.add(
                    ChunkCategoryRelation(
                        chunk_id=chunk.id,
                        category_id=category_id,
                        space_id=space_id,
                        document_id=document.id,
                        page_no=source_page,
                        confidence=0.9,
                    )
                )

        chunk_index += 1
        created += 1

    return created


def cleanup_enrichment_for_document(session: Session, document_id: int) -> None:
    """Supprime les chunks d'enrichissement contextuel d'un document."""
    chunk_ids = [
        row[0]
        for row in session.execute(
            text(
                """
                SELECT id FROM documentchunk
                WHERE document_id = :doc_id
                  AND COALESCE(
                      metadata_json->>'content_type',
                      metadata_->>'content_type',
                      ''
                  ) = :content_type
                """
            ),
            {"doc_id": document_id, "content_type": CONTENT_TYPE_CONTEXTUAL_ENRICHMENT},
        ).all()
    ]
    if not chunk_ids:
        return

    chunk_ids_tuple = tuple(chunk_ids)
    session.execute(
        text("DELETE FROM chunkcategoryrelation WHERE chunk_id IN :chunk_ids"),
        {"chunk_ids": chunk_ids_tuple},
    )
    session.execute(
        text("DELETE FROM documentchunk WHERE id IN :chunk_ids"),
        {"chunk_ids": chunk_ids_tuple},
    )
    session.commit()
    logger.info(
        "[Enrichment] Nettoyage document_id=%s — %s chunks supprimés",
        document_id,
        len(chunk_ids),
    )


def run_contextual_enrichment_for_document(document_id: int) -> dict:
    """
    Génère et persiste les chunks d'enrichissement contextuel pour un document.
    Non bloquant : retourne des compteurs même si certains batches échouent.
    """
    if not settings.CONTEXTUAL_ENRICHMENT_ENABLED:
        return {"chunks": 0, "batches": 0, "status": "disabled"}

    with Session(engine) as session:
        document = session.get(Document, document_id)
        if not document:
            raise ValueError(f"Document introuvable : {document_id}")

        space_ids = _get_document_space_ids(session, document_id)
        if not space_ids:
            logger.warning(
                "[Enrichment] document_id=%s sans espace associé — ignoré",
                document_id,
            )
            return {"chunks": 0, "batches": 0, "status": "no_space"}

        chunks_by_page = _load_semantic_chunks_by_page(session, document_id)
        if not chunks_by_page:
            return {"chunks": 0, "batches": 0, "status": "no_chunks"}

        cleanup_enrichment_for_document(session, document_id)

        page_anchors = _load_page_anchors(session, document_id)
        categories_by_page = _load_categories_by_page(session, document_id)
        entities_by_page = _load_entities_by_page(session, document_id)

        from app.services.category_catalog import get_category_id_by_slug

        category_id_by_slug = get_category_id_by_slug(session)
        valid_category_slugs = frozenset(category_id_by_slug.keys())

        page_numbers = sorted(chunks_by_page.keys())
        batches = build_kag_batches(
            page_numbers,
            batch_size=settings.CONTEXTUAL_ENRICHMENT_BATCH_SIZE,
            overlap=settings.CONTEXTUAL_ENRICHMENT_BATCH_OVERLAP,
        )

        batch_responses: List[Tuple[List[int], BatchEnrichmentResponse]] = []
        concurrency = settings.CONTEXTUAL_ENRICHMENT_CONCURRENCY

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    extract_batch_enrichment_response,
                    batch,
                    document.title or "",
                    chunks_by_page,
                    categories_by_page,
                    entities_by_page,
                    valid_category_slugs,
                ): batch
                for batch in batches
            }
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    response = future.result()
                    if response and response.enrichment_chunks:
                        batch_responses.append((batch, response))
                except Exception as exc:
                    logger.error("[Enrichment] Batch %s échoué : %s", batch, exc)

        chunk_index = _next_chunk_index(session, document_id)
        total_created = 0
        batches_ok = 0

        for batch, response in sorted(batch_responses, key=lambda x: x[0][0]):
            created = _persist_enrichment_chunks(
                session,
                document,
                space_ids,
                batch,
                response.enrichment_chunks,
                page_anchors,
                category_id_by_slug,
                chunk_index,
            )
            chunk_index += created
            total_created += created
            batches_ok += 1

        session.commit()

        logger.info(
            "[Enrichment] Terminé document_id=%s batches=%s/%s chunks=%s model=%s",
            document_id,
            batches_ok,
            len(batches),
            total_created,
            _enrichment_model(),
        )
        return {
            "chunks": total_created,
            "batches": batches_ok,
            "status": "completed" if total_created else "empty",
        }


def embed_enrichment_chunks_for_document(document_id: int) -> int:
    """Embed les chunks contextual_enrichment sans embedding."""
    if not settings.CONTEXTUAL_ENRICHMENT_ENABLED:
        return 0

    from app.services.document_indexing_service import _build_embed_text
    from app.services.embedding_service import generate_embeddings_batch

    with Session(engine) as session:
        stmt = select(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.is_leaf == True,  # noqa: E712
        )
        chunks = [
            c
            for c in session.exec(stmt).all()
            if (c.metadata_json or {}).get("content_type") == CONTENT_TYPE_CONTEXTUAL_ENRICHMENT
            and c.embedding is None
        ]
        if not chunks:
            return 0

        texts = [_build_embed_text(c) for c in chunks]
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
            "[Enrichment] %s/%s chunks enrichissement embeddés document_id=%s",
            embedded,
            len(chunks),
            document_id,
        )
        return embedded
