"""CAG — packer document-centric (build_cag_context).

  * aggregate_documents : classement des documents par pertinence.
  * document entier chargé quand il tient sous le budget (en-tête + marqueurs de page).
  * fenêtrage autour des pages matchées quand le document dépasse le plafond.
  * budget adaptatif par intent, rognage des pages les plus éloignées, cache fulltext,
    sélection d'images alignée sur le contexte packé.
"""
from __future__ import annotations

import uuid
from unittest import mock

from sqlmodel import Session, select

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.library import Library
from app.services.context_packer_service import (
    aggregate_documents,
    budget_for_intent,
    build_cag_context,
    invalidate_document_fulltext_cache,
    select_cag_images,
)
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


def test_budget_for_intent_uses_table_and_default(monkeypatch):
    # Plafonds globaux au niveau des défauts code : la table par intent s'exprime pleinement.
    from app.config import settings

    monkeypatch.setattr(settings, "CAG_TOKEN_BUDGET", 100000)
    monkeypatch.setattr(settings, "CAG_MAX_DOCUMENTS", 8)
    # Table par défaut : specification = petit budget, troubleshooting = gros budget.
    assert budget_for_intent("specification") == (30000, 4)
    assert budget_for_intent("troubleshooting") == (100000, 8)
    # Intent inconnu / absent → clé "default".
    assert budget_for_intent("intent_inconnu") == (60000, 6)
    assert budget_for_intent(None) == (60000, 6)


def test_budget_for_intent_clamped_by_global_ceilings(monkeypatch):
    """CAG_TOKEN_BUDGET / CAG_MAX_DOCUMENTS sont des plafonds DURS : une prod configurée
    à 50000/3 ne doit plus packer 100000/8 parce que l'intent est product_selection
    (bug constaté en prod 2026-07-22 : variables d'env ignorées par la table par intent)."""
    from app.config import settings

    monkeypatch.setattr(settings, "CAG_TOKEN_BUDGET", 50000)
    monkeypatch.setattr(settings, "CAG_MAX_DOCUMENTS", 3)
    # Les gros intents sont bornés par les plafonds globaux.
    assert budget_for_intent("product_selection") == (50000, 3)
    assert budget_for_intent("troubleshooting") == (50000, 3)
    # Les petits intents restent en dessous du plafond, inchangés.
    assert budget_for_intent("specification") == (30000, 3)


def test_windowed_trim_drops_farthest_pages_first(db_session: Session):
    """Quand l'extrait dépasse le budget, on rogne les pages les plus ÉLOIGNÉES des pages
    matchées — pas les dernières du document (la réponse est souvent juste après le match)."""
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)
    doc = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Gros doc")
    for page in range(1, 7):  # pages 1..6, ~70 chars chacune (~20 tokens à 3.5 chars/token)
        _add_leaf(
            db_session,
            document_id=doc.id,
            page=page,
            idx=page,
            text=f"PAGE_{page}_" + "x" * 60,
        )
    db_session.commit()

    passages = [{"document_id": doc.id, "document_title": doc.title, "page_no": 2, "score": 0.9}]
    # Fenêtrage forcé (full_doc_max_tokens=1), rayon large (toutes les pages), budget ~2 pages.
    msg = build_cag_context(
        db_session,
        passages,
        system_prompt=SYS,
        token_budget=45,
        full_doc_max_tokens=1,
        page_radius=10,
    )
    content = msg["content"]
    # Pages proches du match (2 et sa voisine 1) conservées ; les lointaines rognées.
    assert "PAGE_2_" in content and "PAGE_1_" in content
    assert "PAGE_5_" not in content and "PAGE_6_" not in content


def test_fulltext_cache_serves_stale_until_invalidated(db_session: Session):
    """TTL > 0 : le packer relit depuis le cache ; invalidate_document_fulltext_cache purge."""
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)
    doc = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Doc cache")
    _add_leaf(db_session, document_id=doc.id, page=1, idx=1, text="VERSION_A")
    db_session.commit()

    passages = [{"document_id": doc.id, "document_title": doc.title, "page_no": 1, "score": 0.9}]
    msg1 = build_cag_context(db_session, passages, system_prompt=SYS)
    assert "VERSION_A" in msg1["content"]

    # Modifier le chunk : le cache TTL sert encore l'ancien texte…
    chunk = db_session.exec(
        select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
    ).first()
    chunk.text = "VERSION_B"
    chunk.content = "VERSION_B"
    db_session.add(chunk)
    db_session.commit()

    msg2 = build_cag_context(db_session, passages, system_prompt=SYS)
    assert "VERSION_A" in msg2["content"]

    # …jusqu'à l'invalidation explicite (appelée par l'indexation).
    invalidate_document_fulltext_cache(doc.id)
    msg3 = build_cag_context(db_session, passages, system_prompt=SYS)
    assert "VERSION_B" in msg3["content"]


def test_select_cag_images_only_packed_pages(db_session: Session, tmp_path):
    """Les PNG joints sont UNIQUEMENT des pages incluses dans le contexte packé, avec
    priorité aux pages à besoin visuel, et légendes reliant image ↔ document."""
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    doc = _make_doc(
        db_session,
        user_id=user.id,
        library_id=lib.id,
        title="Notice packée",
        source_file_path=str(pdf_path),
    )
    other = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Doc hors contexte")
    db_session.commit()

    cag_documents = [
        {
            "index": 1,
            "document_id": doc.id,
            "document_title": doc.title,
            "pages": [1, 2, 3],
            "full_document": True,
        }
    ]
    passages = [
        # Page packée SANS besoin visuel (score max)
        {"document_id": doc.id, "page_no": 2, "score": 0.9, "needs_page_image": False},
        # Page packée AVEC besoin visuel (score moindre) → prioritaire
        {"document_id": doc.id, "page_no": 3, "score": 0.5, "needs_page_image": True},
        # Page HORS contexte packé → jamais rendue
        {"document_id": doc.id, "page_no": 9, "score": 0.95, "needs_page_image": True},
        # Document non packé → jamais rendu
        {"document_id": other.id, "page_no": 1, "score": 0.99, "needs_page_image": True},
    ]

    with mock.patch(
        "app.services.multimodal_page_service.render_page_png_cached",
        return_value=b"PNGDATA",
    ) as render_mock:
        images, captions = select_cag_images(
            db_session, cag_documents, passages, max_images=1
        )

    assert len(images) == 1 and len(captions) == 1
    # La page à besoin visuel (3) passe devant la page à plus fort score (2).
    assert captions[0]["page_no"] == 3
    assert captions[0]["document_index"] == 1
    assert captions[0]["document_title"] == "Notice packée"
    rendered_pages = [c.args[1] for c in render_mock.call_args_list]
    assert rendered_pages == [3]


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
