"""Ancre de conversation — la recherche ne doit plus se dégrader aux tours suivants.

Régression prod du 27/07 : tour 1 « quelle référence de crémone 4 points ? » → correcte
depuis le catalogue CONCEPTION ; tour 2 « comment l'installer ? » → fausse, alors que la
même question posée en conversation NEUVE donnait la bonne réponse depuis le catalogue
FABRICATION. La question autonome était identique dans les deux cas : seule l'ancre
différait. Elle appliquait ×1,5 à TOUTES les pages du catalogue conception, soit
gratuitement le boost catégorie MAXIMAL (plafonné à 1,5, et qui exige lui des
correspondances fortes) — le signal de pertinence était purement annulé.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

from sqlmodel import Session

from app.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.library import Library
from app.services.context_packer_service import build_cag_context
from app.services.retrieval_boost_service import apply_anchor_boost_to_fused_hits
from tests.conftest import create_test_user

SYS = "SYSTEME"


def _hit(doc_id: int, page: int, rrf: float) -> SimpleNamespace:
    return SimpleNamespace(document_id=doc_id, page_no=page, rrf_score=rrf, final_rank=0)


# ---------------------------------------------------------------------------
# A1 — Boost proportionné et limité aux meilleures pages
# ---------------------------------------------------------------------------


def test_anchor_boost_is_capped_to_best_pages(monkeypatch):
    """Seules les N meilleures pages du document ancré sont boostées : le but est qu'une
    page pertinente survive à la coupe, pas qu'un document entier s'installe en tête."""
    monkeypatch.setattr(settings, "CONVERSATION_ANCHOR_BOOST", 0.15)
    monkeypatch.setattr(settings, "CONVERSATION_ANCHOR_BOOST_MAX_PAGES", 2)

    hits = [_hit(1, p, 0.05 - p * 0.001) for p in range(1, 8)]
    info = apply_anchor_boost_to_fused_hits(hits, anchor_document_ids=[1])

    assert info is not None
    assert len(info["boosted_pages"]) == 2
    assert [p["page_no"] for p in info["boosted_pages"]] == [1, 2]  # les mieux classées
    boosted = {h.page_no: h.rrf_score for h in hits}
    assert abs(boosted[1] - 0.049 * 1.15) < 1e-9
    assert abs(boosted[3] - 0.047) < 1e-9  # au-delà du plafond : intact


def test_anchor_boost_no_longer_drowns_a_more_relevant_document(monkeypatch):
    """Cas mesuré : la page 111 du catalogue fabrication (0,0523) doit rester devant les
    pages médiocres du catalogue conception ancré (0,035). À ×1,5 elles passaient devant."""
    monkeypatch.setattr(settings, "CONVERSATION_ANCHOR_BOOST", 0.15)
    monkeypatch.setattr(settings, "CONVERSATION_ANCHOR_BOOST_MAX_PAGES", 3)

    hits = [_hit(157, 111, 0.0523)] + [_hit(152, p, 0.035) for p in range(40, 50)]
    apply_anchor_boost_to_fused_hits(hits, anchor_document_ids=[152])

    assert hits[0].document_id == 157 and hits[0].page_no == 111


def test_legacy_boost_would_have_buried_it(monkeypatch):
    """Contraste : avec les anciens réglages (×1,5 sur toutes les pages), la page
    pertinente était enterrée — c'est le comportement que A1 corrige."""
    monkeypatch.setattr(settings, "CONVERSATION_ANCHOR_BOOST", 0.5)
    monkeypatch.setattr(settings, "CONVERSATION_ANCHOR_BOOST_MAX_PAGES", 0)  # toutes

    hits = [_hit(157, 111, 0.0523)] + [_hit(152, p, 0.035) for p in range(40, 50)]
    apply_anchor_boost_to_fused_hits(hits, anchor_document_ids=[152])

    assert hits[0].document_id == 152


def test_anchor_boost_returns_none_without_anchor():
    hits = [_hit(2, 1, 1.0), _hit(1, 1, 0.8)]
    assert apply_anchor_boost_to_fused_hits(hits, anchor_document_ids=None) is None
    assert hits[0].document_id == 2


# ---------------------------------------------------------------------------
# A3 — Garantir la présence n'est pas garantir la priorité
# ---------------------------------------------------------------------------


def _library(session: Session, user_id: int) -> Library:
    lib = Library(name="Lib", user_id=user_id, is_global=False)
    session.add(lib)
    session.commit()
    session.refresh(lib)
    return lib


def _doc_with_page(session: Session, *, user_id: int, library_id: int, title: str, text: str) -> Document:
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
    session.add(
        DocumentChunk(
            document_id=doc.id,
            chunk_index=1,
            content=text,
            text=text,
            is_leaf=True,
            hierarchy_level=0,
            node_id=f"n_{uuid.uuid4().hex[:8]}",
            start_char=0,
            end_char=len(text),
            metadata_json={"page_no": 1, "content_type": "semantic_leaf"},
        )
    )
    session.commit()
    return doc


def test_anchor_is_packed_but_ranked_by_its_real_score(db_session: Session):
    """L'ancre reste packée (jamais tronquée) mais ne capte plus la part de budget du
    rang 1 : le document réellement pertinent passe devant."""
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)
    ancre = _doc_with_page(
        db_session, user_id=user.id, library_id=lib.id, title="Conception", text="ANCRE_CONCEPTION"
    )
    trouve = _doc_with_page(
        db_session, user_id=user.id, library_id=lib.id, title="Fabrication", text="PROCEDURE_MONTAGE"
    )

    passages = [
        {"document_id": trouve.id, "document_title": trouve.title, "page_no": 1, "score": 0.9}
    ]
    msg = build_cag_context(
        db_session, passages, system_prompt=SYS, anchor_document_ids=[ancre.id]
    )

    packed = [d["document_id"] for d in msg["cag_documents"]]
    assert packed[0] == trouve.id, "le document pertinent prend le rang 1"
    assert ancre.id in packed, "l'ancre reste néanmoins packée"
    assert "PROCEDURE_MONTAGE" in msg["content"] and "ANCRE_CONCEPTION" in msg["content"]


def test_anchor_first_when_rank_by_score_disabled(db_session: Session, monkeypatch):
    """Repli : l'ancien comportement (ancre en tête) reste disponible par configuration."""
    monkeypatch.setattr(settings, "CAG_ANCHOR_RANK_BY_SCORE", False)
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)
    ancre = _doc_with_page(
        db_session, user_id=user.id, library_id=lib.id, title="Conception", text="ANCRE_CONCEPTION"
    )
    trouve = _doc_with_page(
        db_session, user_id=user.id, library_id=lib.id, title="Fabrication", text="PROCEDURE_MONTAGE"
    )

    passages = [
        {"document_id": trouve.id, "document_title": trouve.title, "page_no": 1, "score": 0.9}
    ]
    msg = build_cag_context(
        db_session, passages, system_prompt=SYS, anchor_document_ids=[ancre.id]
    )
    assert [d["document_id"] for d in msg["cag_documents"]][0] == ancre.id


# ---------------------------------------------------------------------------
# Lot J — les blocs documents sont exposés séparément du prompt système
# ---------------------------------------------------------------------------


def test_cag_exposes_document_blocks_without_system_prompt(db_session: Session):
    """Le juge consomme ces blocs : ils ne doivent contenir NI le prompt système NI le
    préambule CAG, qui n'ont aucune valeur probante et coûtaient 35 % de son budget."""
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)
    doc = _doc_with_page(
        db_session, user_id=user.id, library_id=lib.id, title="Notice", text="CONTENU_UTILE"
    )

    passages = [{"document_id": doc.id, "document_title": doc.title, "page_no": 1, "score": 0.9}]
    msg = build_cag_context(db_session, passages, system_prompt=SYS)

    blocks = msg["cag_document_blocks"]
    assert len(blocks) == 1
    assert "CONTENU_UTILE" in blocks[0]
    assert SYS not in blocks[0]
    assert "IMPÉRATIF" not in blocks[0]  # préambule CAG absent
    assert SYS in msg["content"]  # …mais toujours présent dans le message système
