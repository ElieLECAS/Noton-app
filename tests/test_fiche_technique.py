"""Fiche technique refondue — réponse NATURELLE (prose sur contexte CAG) + sources
au niveau document. Plus de gabarit rigide, plus de dump KAG, plus de pied « Fiabilité »."""
import uuid
from unittest import mock

import pytest
from sqlmodel import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.library import Library
from app.services.fiche_technique_service import (
    ReferenceQuery,
    build_fiche_technique,
    detect_reference_query,
)
from tests.conftest import create_test_user


# ---------------------------------------------------------------------------
# Détection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected_primary",
    [
        ("profil 6101", "6101"),
        ("notice de montage seuil 76180", "76180"),
        ("parle-moi de la gamme Textural", ""),  # présentation sans code chiffré
    ],
)
def test_detect_positive(message, expected_primary):
    r = detect_reference_query(message)
    assert r is not None
    assert r.primary == expected_primary


@pytest.mark.parametrize(
    "message",
    [
        "comment poser le seuil 76180 ?",  # question procédurale → RAG/guidé
        "pourquoi le joint fuit",
        "bonjour",
        "",
    ],
)
def test_detect_negative(message):
    assert detect_reference_query(message) is None


# ---------------------------------------------------------------------------
# Génération de la fiche (prose naturelle + sources doc-level)
# ---------------------------------------------------------------------------


def _make_doc_with_ref(session: Session, user_id: int, lib_id: int) -> int:
    doc = Document(
        title="Catalogue général PROFERM",
        document_type="written",
        processing_status="completed",
        library_id=lib_id,
        user_id=user_id,
        source="PROFERM",
        source_file_path="/tmp/cat.pdf",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    for page, txt in ((1, "Le profil 6101 est la traverse haute du dormant."),
                      (2, "Cote associée 120 mm, matériau aluminium.")):
        session.add(
            DocumentChunk(
                document_id=doc.id, chunk_index=page, content=txt, text=txt,
                is_leaf=True, hierarchy_level=0, node_id=f"n_{uuid.uuid4().hex[:8]}",
                start_char=0, end_char=len(txt),
                metadata_json={"page_no": page, "content_type": "semantic_leaf"},
            )
        )
    session.commit()
    return doc.id


@pytest.mark.asyncio
async def test_build_fiche_produces_natural_prose_and_doc_sources(db_session: Session):
    user = create_test_user(db_session, "responsable")
    lib = Library(name="Lib fiche", user_id=user.id, is_global=False)
    db_session.add(lib)
    db_session.commit()
    db_session.refresh(lib)
    doc_id = _make_doc_with_ref(db_session, user.id, lib.id)

    passages = [
        {"document_id": doc_id, "document_title": "Catalogue général PROFERM", "page_no": 1, "score": 0.9},
        {"document_id": doc_id, "document_title": "Catalogue général PROFERM", "page_no": 2, "score": 0.7},
    ]

    prose = (
        "Le profil 6101 est la traverse haute du dormant, en aluminium. "
        "Sa cote associée est de 120 mm."
    )
    fake_chat = mock.AsyncMock(
        return_value={"choices": [{"message": {"content": prose}}]}
    )

    with mock.patch(
        "app.services.fiche_technique_service._resolve_passages",
        new=mock.AsyncMock(return_value=passages),
    ), mock.patch("app.services.fiche_technique_service.chat", fake_chat):
        result = await build_fiche_technique(
            session=db_session,
            space_id=1,
            user_id=user.id,
            ref_query=ReferenceQuery(references=["6101"], primary="6101", raw_message="profil 6101"),
        )

    assert result is not None
    # Prose naturelle, PAS de gabarit rigide ni de dump KAG.
    assert "traverse haute" in result.markdown
    assert "Entités liées" not in result.markdown
    assert "Fiabilité" not in result.markdown
    assert "🗂️" not in result.markdown
    # Sources au niveau DOCUMENT (badge « Documents consultés »).
    assert len(result.sources) == 1
    src = result.sources[0]
    assert src["cag_document"] is True
    assert src["document_id"] == doc_id
    assert src["has_source_file"] is True
    assert result.topic == "6101"
    # Le contexte fourni au LLM NE demande PAS le bloc <sources> (affichage doc-level direct).
    system_content = fake_chat.call_args.kwargs["context"][0]["content"]
    assert "<sources>" not in system_content


@pytest.mark.asyncio
async def test_build_fiche_abstains_without_passages(db_session: Session):
    user = create_test_user(db_session, "responsable")
    with mock.patch(
        "app.services.fiche_technique_service._resolve_passages",
        new=mock.AsyncMock(return_value=[]),
    ):
        result = await build_fiche_technique(
            session=db_session,
            space_id=1,
            user_id=user.id,
            ref_query=ReferenceQuery(references=["9999"], primary="9999", raw_message="profil 9999"),
        )
    # Aucun passage → None → le routeur retombe sur le RAG normal.
    assert result is None
