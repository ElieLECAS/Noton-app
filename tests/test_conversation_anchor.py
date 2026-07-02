"""Ancrage documentaire conversationnel : continuité du sujet entre les tours.

  * compute_anchor_documents : documents dominants d'un lot de passages (top-N par score).
  * apply_anchor_boost_to_fused_hits : une page d'un document ancré remonte avant la coupe.
  * _update_conversation_documents : persiste l'ancre dans query_context sans écraser le reste.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest import mock

from sqlmodel import Session

from app.models.conversation import Conversation
from app.models.space import Space
from app.routers.chat import _save_conversation_query_context, _update_conversation_documents
from app.services.retrieval_boost_service import (
    apply_anchor_boost_to_fused_hits,
    compute_anchor_documents,
)
from tests.conftest import create_test_user


# ---------------------------------------------------------------------------
# compute_anchor_documents
# ---------------------------------------------------------------------------


def test_compute_anchor_documents_ranks_by_summed_score():
    passages = [
        {"document_id": 1, "score": 0.5},
        {"document_id": 2, "score": 0.9},
        {"document_id": 1, "score": 0.5},  # doc 1 → total 1.0 > doc 2 (0.9)
        {"document_id": 3, "score": 0.2},
    ]
    assert compute_anchor_documents(passages, max_docs=2) == [1, 2]


def test_compute_anchor_documents_empty_or_no_doc_id():
    assert compute_anchor_documents([], max_docs=3) == []
    assert compute_anchor_documents([{"score": 1.0}], max_docs=3) == []
    assert compute_anchor_documents([{"document_id": 5, "score": 1.0}], max_docs=0) == []


def test_fiche_sources_yield_dominant_document_anchor():
    """Sources d'une fiche technique (mêmes forme que le RAG) → document dominant ancré.

    C'est le chemin qui garde le suivi (« tu as ses dimensions ? ») sur le produit de la
    fiche (ex. dormant 6101) au lieu de dériver.
    """
    fiche_sources = [
        {"document_id": 42, "score": 0.9, "page_no": 7},
        {"document_id": 42, "score": 0.6, "page_no": 13},
        {"document_id": 99, "score": 0.2, "page_no": 3},
    ]
    anchor = compute_anchor_documents(fiche_sources, max_docs=3)
    assert anchor[0] == 42
    assert set(anchor) == {42, 99}


# ---------------------------------------------------------------------------
# apply_anchor_boost_to_fused_hits
# ---------------------------------------------------------------------------


def _hit(doc_id: int, rrf: float, rank: int) -> SimpleNamespace:
    return SimpleNamespace(document_id=doc_id, rrf_score=rrf, final_rank=rank)


def test_anchor_boost_floats_anchored_doc_above():
    # doc 2 (rrf 1.0) en tête ; doc 1 (rrf 0.8) juste derrière.
    hits = [_hit(2, 1.0, 1), _hit(1, 0.8, 2)]
    with mock.patch("app.config.settings.CONVERSATION_ANCHOR_BOOST", 0.5):
        apply_anchor_boost_to_fused_hits(hits, anchor_document_ids=[1])
    # doc 1 boosté ×1.5 → 1.2 > 1.0 → passe en tête, rangs recalculés.
    assert hits[0].document_id == 1
    assert hits[0].final_rank == 1
    assert abs(hits[0].rrf_score - 1.2) < 1e-9
    assert hits[1].document_id == 2


def test_anchor_boost_noop_without_anchor():
    hits = [_hit(2, 1.0, 1), _hit(1, 0.8, 2)]
    apply_anchor_boost_to_fused_hits(hits, anchor_document_ids=None)
    assert hits[0].document_id == 2  # ordre inchangé


# ---------------------------------------------------------------------------
# _update_conversation_documents — persistance dans query_context
# ---------------------------------------------------------------------------


def test_update_conversation_documents_merges_without_clobber(db_session: Session):
    user = create_test_user(db_session, "responsable")
    space = Space(name=f"Espace {uuid.uuid4().hex[:8]}", user_id=user.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)

    conv = Conversation(
        title="Test",
        user_id=user.id,
        space_id=space.id,
        query_context={"current_topic": "seuil PMR 76100", "topic_shift": False},
    )
    db_session.add(conv)
    db_session.commit()
    db_session.refresh(conv)

    _update_conversation_documents(conv.id, [7, 3])

    db_session.expire_all()
    reloaded = db_session.get(Conversation, conv.id)
    assert reloaded.query_context["current_documents"] == [7, 3]
    # Les clés existantes ne sont pas écrasées.
    assert reloaded.query_context["current_topic"] == "seuil PMR 76100"


def test_save_query_context_preserves_anchor(db_session: Session):
    """La sauvegarde du query_context (compréhension) ne doit PAS effacer l'ancre.

    Scénario réel : T1 fiche technique pose l'ancre ; T2 (clarification, ou n'importe quel
    tour qui n'atteint pas le retrieval) sauvegarde son query_context — sans préservation,
    current_documents disparaissait et T3 dérivait.
    """
    user = create_test_user(db_session, "responsable")
    space = Space(name=f"Espace {uuid.uuid4().hex[:8]}", user_id=user.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)

    conv = Conversation(
        title="Test",
        user_id=user.id,
        space_id=space.id,
        query_context={"current_documents": [42], "current_topic": "dormant 6101"},
    )
    db_session.add(conv)
    db_session.commit()
    db_session.refresh(conv)

    # Le query_context produit par la compréhension ne contient jamais current_documents.
    _save_conversation_query_context(
        conv.id, {"current_topic": "dormant 6101", "topic_shift": False, "signals": {}}
    )

    db_session.expire_all()
    reloaded = db_session.get(Conversation, conv.id)
    assert reloaded.query_context["current_documents"] == [42], "l'ancre doit survivre à la sauvegarde"
    assert reloaded.query_context["topic_shift"] is False
