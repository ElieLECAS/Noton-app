"""CAG Phase 1 — packer document-centric (build_cag_context).

  * aggregate_documents : classement des documents par pertinence.
  * document entier chargé quand il tient sous le budget (en-tête + marqueurs de page).
  * fenêtrage autour des pages matchées quand le document dépasse le plafond.
"""
from __future__ import annotations

import uuid

from sqlmodel import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.library import Library
from app.services.context_packer_service import aggregate_documents, build_cag_context
from tests.conftest import create_test_user

SYS = "SYSTEME"


def _make_doc(session: Session, *, user_id: int, library_id: int, **kwargs) -> Document:
    doc = Document(
        title=kwargs.pop("title", "Notice"),
        document_type="written",
        processing_status="completed",
        library_id=library_id,
        user_id=user_id,
        **kwargs,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    return doc


def _add_leaf(session: Session, *, document_id: int, page: int, idx: int, text: str) -> None:
    session.add(
        DocumentChunk(
            document_id=document_id,
            chunk_index=idx,
            content=text,
            text=text,
            is_leaf=True,
            hierarchy_level=0,
            node_id=f"n_{uuid.uuid4().hex[:8]}",
            start_char=0,
            end_char=len(text),
            metadata_json={"page_no": page, "content_type": "semantic_leaf"},
        )
    )


def _library(session: Session, user_id: int) -> Library:
    lib = Library(name="Lib", user_id=user_id, is_global=False)
    session.add(lib)
    session.commit()
    session.refresh(lib)
    return lib


# ---------------------------------------------------------------------------
# aggregate_documents (pur)
# ---------------------------------------------------------------------------


def test_aggregate_documents_ranks_and_collects_pages():
    passages = [
        {"document_id": 1, "page_no": 2, "score": 0.9},
        {"document_id": 1, "page_no": 3, "score": 0.4},
        {"document_id": 2, "page_no": 5, "score": 0.6},
    ]
    ranked = aggregate_documents(passages, max_documents=8)
    assert [did for did, _ in ranked] == [1, 2]  # doc 1 : 0.9 + 0.2·0.4 > doc 2 : 0.6
    assert ranked[0][1]["matched_pages"] == {2, 3}


# ---------------------------------------------------------------------------
# build_cag_context (DB)
# ---------------------------------------------------------------------------


def test_build_cag_loads_full_small_document(db_session: Session):
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)
    doc = _make_doc(
        db_session,
        user_id=user.id,
        library_id=lib.id,
        title="Notice KSR PVC",
        source="Roto",
        proferm_gammes=["KSR PVC"],
        materials=["PVC"],
    )
    for page in (1, 2, 3):
        _add_leaf(db_session, document_id=doc.id, page=page, idx=page, text=f"CONTENU_PAGE_{page}")
    db_session.commit()

    passages = [{"document_id": doc.id, "document_title": doc.title, "page_no": 2, "score": 0.9}]
    msg = build_cag_context(db_session, passages, system_prompt=SYS, token_budget=100000)

    content = msg["content"]
    # Document entier : les 3 pages présentes.
    assert "CONTENU_PAGE_1" in content and "CONTENU_PAGE_2" in content and "CONTENU_PAGE_3" in content
    # En-tête métadonnées (évite la confusion de gammes).
    assert "Source : Roto" in content and "KSR PVC" in content
    # Marqueurs de page + mention document complet.
    assert "[page 1]" in content and "document complet" in content
    assert msg["cag_documents"][0]["full_document"] is True
    assert msg["cag_documents"][0]["pages"] == [1, 2, 3]


def test_build_cag_windows_large_document(db_session: Session):
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)
    doc = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Gros doc")
    for page in range(1, 7):  # pages 1..6
        _add_leaf(db_session, document_id=doc.id, page=page, idx=page, text=f"PAGE_{page}_TEXTE")
    db_session.commit()

    passages = [{"document_id": doc.id, "document_title": doc.title, "page_no": 3, "score": 0.9}]
    # full_doc_max_tokens=1 → force le fenêtrage ; radius=1 → pages {2,3,4}.
    msg = build_cag_context(
        db_session,
        passages,
        system_prompt=SYS,
        token_budget=100000,
        full_doc_max_tokens=1,
        page_radius=1,
    )
    content = msg["content"]
    assert "PAGE_2_TEXTE" in content and "PAGE_3_TEXTE" in content and "PAGE_4_TEXTE" in content
    assert "PAGE_1_TEXTE" not in content and "PAGE_5_TEXTE" not in content and "PAGE_6_TEXTE" not in content
    assert msg["cag_documents"][0]["full_document"] is False
    assert "extrait" in content


def test_build_cag_empty_passages(db_session: Session):
    msg = build_cag_context(db_session, [], system_prompt=SYS)
    assert "Aucun passage" in msg["content"]
    assert msg["cag_documents"] == []


def test_build_cag_force_includes_anchored_document(db_session: Session):
    """GARANTIE d'ancrage : le document du sujet courant est packé EN TÊTE même si le
    retrieval du tour ne l'a pas fait remonter (scénario « tu as ses dimensions ? » qui
    dérive vers un autre manuel)."""
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)

    doc_retrieved = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Manuel Roto NX")
    _add_leaf(db_session, document_id=doc_retrieved.id, page=1, idx=1, text="ROTO_NX_DIMENSIONS")

    doc_anchored = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Notice dormant 6101")
    _add_leaf(db_session, document_id=doc_anchored.id, page=1, idx=1, text="DORMANT_6101_CONTENU")
    db_session.commit()

    # Le retrieval de ce tour ne renvoie QUE le manuel Roto.
    passages = [
        {"document_id": doc_retrieved.id, "document_title": doc_retrieved.title, "page_no": 1, "score": 0.9}
    ]
    msg = build_cag_context(
        db_session,
        passages,
        system_prompt=SYS,
        anchor_document_ids=[doc_anchored.id],
    )

    content = msg["content"]
    # Les deux documents sont présents, l'ancré en premier (DOCUMENT 1).
    assert "DORMANT_6101_CONTENU" in content
    assert "ROTO_NX_DIMENSIONS" in content
    packed_ids = [d["document_id"] for d in msg["cag_documents"]]
    assert packed_ids[0] == doc_anchored.id
    assert doc_retrieved.id in packed_ids


def test_build_cag_anchor_never_truncated_by_max_documents(db_session: Session):
    """Le plafond max_documents ne fait jamais sauter le document ancré."""
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)

    docs = []
    for i in range(3):
        d = _make_doc(db_session, user_id=user.id, library_id=lib.id, title=f"Doc {i}")
        _add_leaf(db_session, document_id=d.id, page=1, idx=1, text=f"TEXTE_DOC_{i}")
        docs.append(d)
    anchored = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Doc ancré")
    _add_leaf(db_session, document_id=anchored.id, page=1, idx=1, text="TEXTE_ANCRE")
    db_session.commit()

    passages = [
        {"document_id": d.id, "document_title": d.title, "page_no": 1, "score": 0.9 - 0.1 * i}
        for i, d in enumerate(docs)
    ]
    # max_documents=2 : 1 ancre + 1 seul doc retrieval.
    msg = build_cag_context(
        db_session,
        passages,
        system_prompt=SYS,
        max_documents=2,
        anchor_document_ids=[anchored.id],
    )
    packed_ids = [d["document_id"] for d in msg["cag_documents"]]
    assert packed_ids[0] == anchored.id
    assert len(packed_ids) == 2
