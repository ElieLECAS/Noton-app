"""P0.2/P0.3 — robustesse du flux de génération : contexte de secours (fallback 400),
persistance avec retry, marqueur de troncature, budgets temps."""
import json
from unittest import mock

import pytest
from sqlmodel import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.library import Library
from app.routers.chat import (
    _build_eco_context,
    _persist_reply_with_retry,
    _stream_llm_to_sse,
)
from tests.conftest import create_test_user, extract_sse_message_text


def _mk_docs(session: Session, user_id: int, lib_id: int, n_docs: int) -> list:
    ids = []
    for d in range(n_docs):
        doc = Document(
            title=f"Notice {d}",
            document_type="written",
            processing_status="completed",
            library_id=lib_id,
            user_id=user_id,
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)
        for p in (1, 2):
            session.add(
                DocumentChunk(
                    document_id=doc.id, chunk_index=p, content=f"DOC{d}_PAGE{p} " * 50,
                    text=f"DOC{d}_PAGE{p} " * 50, is_leaf=True, hierarchy_level=0,
                    node_id=f"n{d}_{p}", start_char=0, end_char=100,
                    metadata_json={"page_no": p, "content_type": "semantic_leaf"},
                )
            )
        session.commit()
        ids.append(doc.id)
    return ids


# ---------------------------------------------------------------------------
# Contexte de secours (fallback Mistral 400) — CAG re-packé PLUS PETIT
# ---------------------------------------------------------------------------


def test_eco_context_caps_documents(db_session: Session):
    user = create_test_user(db_session, "responsable")
    lib = Library(name="L", user_id=user.id, is_global=False)
    db_session.add(lib)
    db_session.commit()
    db_session.refresh(lib)
    doc_ids = _mk_docs(db_session, user.id, lib.id, n_docs=8)

    passages = [
        {"document_id": did, "document_title": f"Notice {i}", "page_no": 1, "score": 0.9 - i * 0.05}
        for i, did in enumerate(doc_ids)
    ]
    with mock.patch("app.config.settings.CAG_ENABLED", True), mock.patch(
        "app.config.settings.CAG_ECO_MAX_DOCUMENTS", 3
    ):
        eco = _build_eco_context(db_session, passages, None, "quelle vis ?")

    assert eco[-1] == {"role": "user", "content": "quelle vis ?"}
    eco_system = eco[0]["content"]
    # Le contexte de secours plafonne le nombre de documents packés (CAG_ECO_MAX_DOCUMENTS)
    # là où le plein budget en prendrait 8.
    assert eco_system.count("=== DOCUMENT") <= 3


def test_eco_context_without_cag_is_prompt_plus_user(db_session: Session):
    with mock.patch("app.config.settings.CAG_ENABLED", False):
        eco = _build_eco_context(db_session, [], None, "bonjour")
    assert len(eco) == 2
    assert eco[0]["role"] == "system"
    assert eco[1] == {"role": "user", "content": "bonjour"}


# ---------------------------------------------------------------------------
# Persistance avec retry (P0.2) — pas de perte silencieuse
# ---------------------------------------------------------------------------


def test_persist_retry_succeeds_second_attempt():
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("DB indisponible")
        return 4242

    with mock.patch("app.routers.chat._persist_assistant_reply", side_effect=flaky):
        mid = _persist_reply_with_retry(1, "texte", "m", "p", None, attempts=2)
    assert mid == 4242
    assert calls["n"] == 2


def test_persist_retry_returns_none_after_exhaustion():
    with mock.patch(
        "app.routers.chat._persist_assistant_reply",
        side_effect=RuntimeError("DB morte"),
    ):
        mid = _persist_reply_with_retry(1, "texte", "m", "p", None, attempts=2)
    assert mid is None  # jamais d'exception propagée, None signale l'échec


# ---------------------------------------------------------------------------
# _stream_llm_to_sse — filtre <sources> appliqué, texte accumulé dans le sink
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Régression : le contexte "eco" ne doit être construit QUE si la tentative
# "full" échoue réellement — pas à chaque requête (bug introduit puis corrigé
# le jour même : la liste des tentatives construisait _build_eco_context AVANT
# la boucle, donc à chaque appel, même en cas de succès immédiat).
# ---------------------------------------------------------------------------


async def _fake_success_stream(*args, **kwargs):
    yield json.dumps({"message": {"content": "Réponse complète."}})


def test_eco_context_not_built_when_full_attempt_succeeds(
    client, responsable_headers, db_session: Session
):
    user = create_test_user(db_session, "responsable")
    lib = Library(name="Lib eco", user_id=user.id, is_global=False)
    db_session.add(lib)
    db_session.commit()
    db_session.refresh(lib)
    doc = Document(
        title="Notice", document_type="written", processing_status="completed",
        library_id=lib.id, user_id=user.id,
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    db_session.add(
        DocumentChunk(
            document_id=doc.id, chunk_index=1, content="Contenu de la notice",
            text="Contenu de la notice", is_leaf=True, hierarchy_level=0, node_id="n1",
            start_char=0, end_char=20, metadata_json={"page_no": 1, "content_type": "semantic_leaf"},
        )
    )
    db_session.commit()

    sp = client.post("/api/spaces", headers=responsable_headers, json={"name": "Espace eco pytest"})
    space_id = sp.json()["id"]

    passages = [
        {"document_id": doc.id, "document_title": doc.title, "page_no": 1, "score": 0.9}
    ]
    try:
        with mock.patch("app.config.settings.QUERY_UNDERSTANDING_ENABLED", False), mock.patch(
            "app.config.settings.FICHE_TECHNIQUE_ENABLED", False
        ), mock.patch(
            "app.services.query_reasoning_service.decide_retrieval_route",
            new=mock.AsyncMock(
                return_value=mock.Mock(decision="rag", reasoning="technique")
            ),
        ), mock.patch(
            "app.services.space_search_service.search_technical_passages",
            new=mock.AsyncMock(return_value={"passages": passages, "status": "ok", "reason": None}),
        ), mock.patch(
            "app.routers.chat.mistral_chat_stream", _fake_success_stream
        ), mock.patch(
            "app.routers.chat._build_eco_context"
        ) as eco_mock:
            r = client.post(
                f"/api/spaces/{space_id}/chat/stream",
                headers=responsable_headers,
                json={
                    "message": "Quelle est la référence de cette pièce ?",
                    "model": "mistral-small-latest",
                    "provider": "mistral",
                    "conversation_id": None,
                },
            )
        assert r.status_code == 200
        assert "Réponse complète." in extract_sse_message_text(r.text)
        # Cœur de la régression : succès du 1er coup → JAMAIS de repackaging eco.
        eco_mock.assert_not_called()
    finally:
        client.delete(f"/api/spaces/{space_id}", headers=responsable_headers)


@pytest.mark.asyncio
async def test_stream_helper_filters_sources_and_fills_sink():
    import json as _json

    from app.services.stream_source_filter import SourcesTagStreamFilter

    async def fake_stream(*args, **kwargs):
        for piece in [
            'La réponse. ',
            '<sources>{"used":[{"doc":1,"pages":[2]}]}</sources>',
        ]:
            yield _json.dumps({"message": {"content": piece}})

    sink: list = []
    sf = SourcesTagStreamFilter()
    with mock.patch("app.routers.chat.chat_stream_wrapper", fake_stream):
        emitted = []
        async for sse in _stream_llm_to_sse(
            [{"role": "user", "content": "q"}],
            model="m", max_tokens=None, source_filter=sf, sink=sink,
        ):
            emitted.append(sse)

    # Le bloc <sources> ne fuit pas dans le texte affiché.
    joined = "".join(sink)
    assert "La réponse." in joined
    assert "<sources>" not in joined
    assert sf.used_documents == [{"doc": 1, "pages": [2]}]
