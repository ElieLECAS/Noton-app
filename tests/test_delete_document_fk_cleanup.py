"""Purge des relations FK à la suppression d'un document.

L'invariant CRITIQUE, seul conservé après le retrait du KAG (2026-07-28) :
supprimer un document dont les chunks sont référencés par
``chunkcategoryrelation`` / ``chunkentityrelation`` / ``entityentityrelation``
ne doit PAS lever de violation de contrainte — aucune de ces FK n'a d'
``ON DELETE CASCADE``, la purge applicative est donc obligatoire.

Les assertions sur l'élagage des entités orphelines et la décrémentation de
``mention_count`` ont été retirées : ce comportement appartenait au KAG. Les lignes
``knowledgeentity`` résiduelles sont désormais inertes (plus rien ne les lit) et
partiront avec le DROP des tables.
"""
from __future__ import annotations

import uuid

from sqlmodel import Session, select

from app.models.chunk_category_relation import ChunkCategoryRelation
from app.models.document import Document
from app.models.document_category import DocumentCategory
from app.models.document_chunk import DocumentChunk
from app.models.knowledge_entity import (
    ChunkEntityRelation,
    EntityEntityRelation,
    KnowledgeEntity,
)
from app.models.library import Library
from app.models.space import Space
from app.services.document_service_new import delete_document
from tests.conftest import create_test_user


def _make_document(session: Session, *, user_id: int, library_id: int, title: str) -> Document:
    doc = Document(
        title=title,
        document_type="written",
        processing_status="completed",
        library_id=library_id,
        user_id=user_id,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    return doc


def _make_leaf_chunk(session: Session, *, document_id: int, node_id: str) -> DocumentChunk:
    chunk = DocumentChunk(
        document_id=document_id,
        chunk_index=0,
        content="pose du seuil PMR",
        text="pose du seuil PMR",
        is_leaf=True,
        hierarchy_level=0,
        node_id=node_id,
        start_char=0,
        end_char=17,
        metadata_json={"page_no": 1},
    )
    session.add(chunk)
    session.commit()
    session.refresh(chunk)
    return chunk


def test_delete_document_prunes_orphan_entity_and_category(db_session: Session):
    user = create_test_user(db_session, "responsable")
    library = Library(name="Lib", user_id=user.id, is_global=False)
    db_session.add(library)
    db_session.commit()
    db_session.refresh(library)

    space = Space(name=f"Espace {uuid.uuid4().hex[:8]}", user_id=user.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)

    doc = _make_document(db_session, user_id=user.id, library_id=library.id, title="Notice A")
    chunk = _make_leaf_chunk(db_session, document_id=doc.id, node_id=f"n_{uuid.uuid4().hex[:8]}")

    category = DocumentCategory(slug=f"mounting-{uuid.uuid4().hex[:8]}", label="Pose", axis="task")
    db_session.add(category)
    db_session.commit()
    db_session.refresh(category)

    entity = KnowledgeEntity(
        space_id=space.id,
        name="Seuil 76180",
        name_normalized="seuil 76180",
        entity_type="reference",
        mention_count=1,
    )
    db_session.add(entity)
    db_session.commit()
    db_session.refresh(entity)

    db_session.add(
        ChunkEntityRelation(chunk_id=chunk.id, entity_id=entity.id, space_id=space.id)
    )
    db_session.add(
        ChunkCategoryRelation(
            chunk_id=chunk.id,
            category_id=category.id,
            document_id=doc.id,
            space_id=space.id,
        )
    )
    # Relation entité-entité ancrée sur ce chunk : source_chunk_id (ON DELETE SET NULL)
    other = KnowledgeEntity(
        space_id=space.id,
        name="Vis TF",
        name_normalized="vis tf",
        entity_type="tool",
        mention_count=1,
    )
    db_session.add(other)
    db_session.commit()
    db_session.refresh(other)
    db_session.add(
        EntityEntityRelation(
            space_id=space.id,
            entity_a_id=entity.id,
            entity_b_id=other.id,
            relation_type="utilise",
            source_chunk_id=chunk.id,
        )
    )
    db_session.commit()

    doc_id, chunk_id, entity_id = doc.id, chunk.id, entity.id

    # Ne doit PAS lever de violation FK (chunkcategoryrelation sans cascade).
    assert delete_document(db_session, doc_id, user.id) is True

    db_session.expire_all()
    assert db_session.get(Document, doc_id) is None
    assert db_session.get(DocumentChunk, chunk_id) is None
    # Relations du chunk supprimées (c'est ce qui rend la suppression possible).
    assert (
        db_session.exec(
            select(ChunkEntityRelation).where(ChunkEntityRelation.chunk_id == chunk_id)
        ).first()
        is None
    )
    assert (
        db_session.exec(
            select(ChunkCategoryRelation).where(ChunkCategoryRelation.chunk_id == chunk_id)
        ).first()
        is None
    )
    # La catégorie (référentiel global) n'est pas supprimée, seulement la relation.
    assert db_session.get(DocumentCategory, category.id) is not None


def test_delete_document_preserve_les_liens_des_autres_documents(db_session: Session):
    user = create_test_user(db_session, "responsable")
    library = Library(name="Lib", user_id=user.id, is_global=False)
    db_session.add(library)
    db_session.commit()
    db_session.refresh(library)

    space = Space(name=f"Espace {uuid.uuid4().hex[:8]}", user_id=user.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)

    doc_a = _make_document(db_session, user_id=user.id, library_id=library.id, title="Notice A")
    doc_b = _make_document(db_session, user_id=user.id, library_id=library.id, title="Notice B")
    chunk_a = _make_leaf_chunk(db_session, document_id=doc_a.id, node_id=f"na_{uuid.uuid4().hex[:8]}")
    chunk_b = _make_leaf_chunk(db_session, document_id=doc_b.id, node_id=f"nb_{uuid.uuid4().hex[:8]}")

    # Une seule entité, référencée par les deux documents (mention_count = 2).
    entity = KnowledgeEntity(
        space_id=space.id,
        name="Perform 76",
        name_normalized="perform 76",
        entity_type="product",
        mention_count=2,
    )
    db_session.add(entity)
    db_session.commit()
    db_session.refresh(entity)

    db_session.add(ChunkEntityRelation(chunk_id=chunk_a.id, entity_id=entity.id, space_id=space.id))
    db_session.add(ChunkEntityRelation(chunk_id=chunk_b.id, entity_id=entity.id, space_id=space.id))
    db_session.commit()

    doc_a_id, chunk_b_id, entity_id = doc_a.id, chunk_b.id, entity.id

    assert delete_document(db_session, doc_a_id, user.id) is True

    db_session.expire_all()
    # Seuls les liens du document SUPPRIMÉ partent : ceux du document B restent intacts.
    assert (
        db_session.exec(
            select(ChunkEntityRelation).where(ChunkEntityRelation.chunk_id == chunk_b_id)
        ).first()
        is not None
    )
