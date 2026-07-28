"""Élection & packing — le meilleur passage doit survivre jusqu'au modèle.

Chaque test reproduit une pathologie mesurée sur le corpus réel, et vérifie qu'elle est
corrigée (avec, quand c'est parlant, l'ancien comportement en contraste via les flags) :

  * T1 « catalogue vs notice » — le volume de pages moyennes ne doit plus battre la page
    qui contient la réponse (aggregate_documents).
  * T2 « saturation du pool » — un gros document ne peut plus occuper tous les slots du
    top-K, ni évincer une page que seul ColPali sait voir (select_final_hits).
  * T3 « budget confisqué » — le document n°1 ne peut plus avaler tout le budget CAG.
  * T4 « fenêtres disjointes » — l'en-tête annonce les vraies plages, pas min-max.
  * T5 « rognage aveugle » — sous budget serré, ce sont les pages les plus FAIBLES qui
    sautent, pas les plus éloignées géographiquement.
  * T6 « ancre gloutonne » — les documents ancrés ne consomment plus tous les slots.
"""
from __future__ import annotations

import uuid

from sqlmodel import Session

from app.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.library import Library
from app.services.context_packer_service import (
    _pages_span_summary,
    aggregate_documents,
    build_cag_context,
    invalidate_document_fulltext_cache,
)
from app.services.page_retrieval_service import UnifiedPageHit, select_final_hits
from tests.conftest import create_test_user

SYS = "SYSTEME"


# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------


def _library(session: Session, user_id: int) -> Library:
    lib = Library(name="Lib", user_id=user_id, is_global=False)
    session.add(lib)
    session.commit()
    session.refresh(lib)
    return lib


def _make_doc(session: Session, *, user_id: int, library_id: int, title: str) -> Document:
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
    invalidate_document_fulltext_cache(doc.id)
    return doc


def _add_page(session: Session, *, document_id: int, page: int, text: str) -> None:
    session.add(
        DocumentChunk(
            document_id=document_id,
            chunk_index=page,
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


def _fill_pages(session: Session, document_id: int, pages: range, *, filler: int = 60) -> None:
    """Pages calibrées à ~19 tokens chacune (≈ 67 chars / CAG_CHARS_PER_TOKEN=3.5)."""
    for page in pages:
        _add_page(
            session,
            document_id=document_id,
            page=page,
            text=f"PAGE_{page}_" + "x" * filler,
        )


def _hit(doc_id: int, page: int, rrf: float, **kwargs) -> UnifiedPageHit:
    sources = kwargs.pop("retrieval_sources", ["pgvector", "bm25"])
    return UnifiedPageHit(
        document_id=doc_id,
        page_no=page,
        rrf_score=rrf,
        retrieval_sources=list(sources),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# T1 — Le volume ne bat plus le meilleur passage
# ---------------------------------------------------------------------------


def test_election_prefers_best_passage_over_bulky_catalogue(monkeypatch):
    """Cas mesuré : un catalogue qui place 6 pages moyennes battait arithmétiquement la
    notice contenant LA bonne page (0.056 vs 0.049). L'élection doit désormais suivre la
    meilleure page, les bonus n'étant que des départages bornés."""
    monkeypatch.setattr(settings, "CAG_ELECTION_MODE", "best_passage")

    catalogue = [
        {
            "document_id": 1,
            "document_title": "Catalogue 200 pages",
            "page_no": 40 + i,
            "score": 0.031,
            # Une seule FAMILLE : pgvector et BM25 lisent la même évidence textuelle.
            "retrieval_sources": ["pgvector", "bm25"],
        }
        for i in range(6)
    ]
    notice = [
        {
            "document_id": 2,
            "document_title": "Notice 18 pages",
            "page_no": 13,
            "score": 0.049,
            "retrieval_sources": ["colpali", "pgvector", "bm25"],
        }
    ]

    ranked = aggregate_documents(catalogue + notice, max_documents=3)
    assert [did for did, _ in ranked] == [2, 1], "la notice doit être élue avant le catalogue"
    assert ranked[0][1]["families"] == {"texte", "visuel"}
    assert ranked[1][1]["families"] == {"texte"}


def test_election_legacy_mode_still_rewards_volume(monkeypatch):
    """Contraste : l'ancienne formule (flag legacy) élisait bien le catalogue — c'est
    exactement le comportement que le nouveau mode corrige."""
    monkeypatch.setattr(settings, "CAG_ELECTION_MODE", "legacy")

    passages = [
        {"document_id": 1, "page_no": 40 + i, "score": 0.031, "retrieval_sources": ["pgvector", "bm25"]}
        for i in range(6)
    ] + [
        {"document_id": 2, "page_no": 13, "score": 0.049, "retrieval_sources": ["colpali", "kag"]}
    ]
    ranked = aggregate_documents(passages, max_documents=3)
    assert [did for did, _ in ranked] == [1, 2]


def test_election_bonus_never_overturns_a_clearly_better_passage(monkeypatch):
    """Garde-fou : les bonus sont bornés (×1,30 max) — ils départagent des documents
    proches, ils ne renversent jamais un meilleur passage net."""
    monkeypatch.setattr(settings, "CAG_ELECTION_MODE", "best_passage")

    passages = [
        # Document « large » : 4 familles, 5 pages → bonus maximal.
        *[
            {
                "document_id": 1,
                "page_no": 10 + i,
                "score": 0.030,
                "retrieval_sources": ["pgvector", "bm25", "colpali", "kag"],
            }
            for i in range(5)
        ],
        # Document « net » : une seule page, un seul canal, mais bien meilleure.
        {"document_id": 2, "page_no": 4, "score": 0.060, "retrieval_sources": ["colpali"]},
    ]
    ranked = aggregate_documents(passages, max_documents=3)
    assert ranked[0][0] == 2


# ---------------------------------------------------------------------------
# T2 — La coupe du top-K ne se laisse plus saturer
# ---------------------------------------------------------------------------


def test_per_doc_quota_keeps_room_for_other_documents():
    """Un document qui domine le pool RRF ne doit plus occuper les K slots — sinon il
    remporte ensuite l'élection grâce à un volume qu'il vient lui-même de fabriquer."""
    pool = [_hit(1, page, 0.05 - page * 0.001) for page in range(1, 26)]
    pool.append(_hit(2, 7, 0.012))  # le petit document, dernier au RRF

    final, protected = select_final_hits(pool, 20, per_doc_quota_ratio=0.4, colpali_slots=0)

    assert len(final) == 20
    assert 2 in {h.document_id for h in final}, "le second document doit survivre à la coupe"
    assert protected == []
    # Quota souple : les slots non réclamés reviennent au gros document.
    assert sum(1 for h in final if h.document_id == 1) == 19


def test_cut_without_quota_lets_one_document_take_everything():
    """Contraste : la coupe brute historique (ratio=0) laisse tout au gros document."""
    pool = [_hit(1, page, 0.05 - page * 0.001) for page in range(1, 26)]
    pool.append(_hit(2, 7, 0.012))

    final, _ = select_final_hits(pool, 20, per_doc_quota_ratio=0.0, colpali_slots=0)

    assert {h.document_id for h in final} == {1}


def test_colpali_only_page_gets_a_reserved_slot_without_reranker():
    """Sans reranker, protect_colpali_visual_hits n'est jamais appelé : une page « muette »
    (dessin coté) que seul ColPali sait voir pouvait être éjectée par le consensus
    pgvector+BM25, qui vote deux fois la même évidence. Elle doit garder un slot."""
    pool = [_hit(doc, 1, 0.05 - doc * 0.001) for doc in range(1, 6)]
    visual = UnifiedPageHit(
        document_id=9,
        page_no=88,
        rrf_score=0.010,  # dernier du pool
        colpali_score=0.80,
        retrieval_sources=["colpali"],
    )
    pool.append(visual)

    final, protected = select_final_hits(pool, 3, per_doc_quota_ratio=0.0, colpali_slots=2)

    assert [h.document_id for h in protected] == [9]
    assert 9 in {h.document_id for h in final}
    assert [h.final_rank for h in final] == [1, 2, 3, 4]


def test_select_final_hits_handles_empty_pool():
    assert select_final_hits([], 10) == ([], [])


# ---------------------------------------------------------------------------
# T3 — Le budget est partagé entre les documents élus
# ---------------------------------------------------------------------------


def test_budget_shares_leave_room_for_the_second_document(db_session: Session):
    """Le document n°1 ne doit plus avaler tout le budget : le n°2 a toujours sa part."""
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)
    gros = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Gros catalogue")
    _fill_pages(db_session, gros.id, range(1, 21))
    petit = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Notice")
    _add_page(db_session, document_id=petit.id, page=1, text="REPONSE_ATTENDUE_ICI")
    db_session.commit()

    passages = [
        {"document_id": gros.id, "document_title": gros.title, "page_no": 10, "score": 0.9},
        {"document_id": petit.id, "document_title": petit.title, "page_no": 1, "score": 0.5},
    ]
    msg = build_cag_context(
        db_session,
        passages,
        system_prompt=SYS,
        token_budget=300,
        full_doc_max_tokens=1,  # force le fenêtrage des deux documents
        page_radius=10,
    )

    assert "REPONSE_ATTENDUE_ICI" in msg["content"]
    assert [d["document_id"] for d in msg["cag_documents"]] == [gros.id, petit.id]


def test_without_shares_first_document_starves_the_second(db_session, monkeypatch):
    """Contraste : sans partage, le premier document consomme tout et le second saute."""
    monkeypatch.setattr(settings, "CAG_DOC_BUDGET_SHARES", "")
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)
    gros = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Gros catalogue")
    _fill_pages(db_session, gros.id, range(1, 21))
    petit = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Notice")
    _add_page(db_session, document_id=petit.id, page=1, text="REPONSE_ATTENDUE_ICI")
    db_session.commit()

    passages = [
        {"document_id": gros.id, "document_title": gros.title, "page_no": 10, "score": 0.9},
        {"document_id": petit.id, "document_title": petit.title, "page_no": 1, "score": 0.5},
    ]
    msg = build_cag_context(
        db_session,
        passages,
        system_prompt=SYS,
        token_budget=300,
        full_doc_max_tokens=1,
        page_radius=10,
    )

    assert "REPONSE_ATTENDUE_ICI" not in msg["content"]


# ---------------------------------------------------------------------------
# T4 — L'en-tête dit la vérité sur les fenêtres disjointes
# ---------------------------------------------------------------------------


def test_pages_span_summary_reports_real_runs():
    assert _pages_span_summary([2, 3, 4, 37, 38, 39, 88]) == "2-4, 37-39, 88"
    assert _pages_span_summary([5]) == "5"
    assert _pages_span_summary([]) == ""
    assert _pages_span_summary([3, 1, 2]) == "1-3"


def test_disjoint_window_header_and_seed_markers(db_session: Session):
    """Deux zones matchées éloignées : l'en-tête ne doit pas annoncer « 1-30 », et les
    pages retrouvées doivent être signalées au modèle."""
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)
    doc = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Catalogue")
    _fill_pages(db_session, doc.id, range(1, 31))
    db_session.commit()

    passages = [
        {"document_id": doc.id, "document_title": doc.title, "page_no": 5, "score": 0.9},
        {"document_id": doc.id, "document_title": doc.title, "page_no": 25, "score": 0.85},
    ]
    msg = build_cag_context(
        db_session,
        passages,
        system_prompt=SYS,
        token_budget=200,
        full_doc_max_tokens=1,
        page_radius=1,
    )
    content = msg["content"]

    assert "Pages incluses : 4-6, 24-26 (extrait)" in content
    assert "[page 5 — ★ page retrouvée par la recherche]" in content
    assert "[page 25 — ★ page retrouvée par la recherche]" in content
    assert "[page 4]" in content  # une voisine reste un marqueur simple
    assert msg["cag_documents"][0]["seed_pages"] == [5, 25]


# ---------------------------------------------------------------------------
# T5 — Sous budget serré, ce sont les pages faibles qui sautent
# ---------------------------------------------------------------------------


def test_greedy_window_drops_weakest_pages_not_farthest(db_session: Session):
    """Deux zones matchées, l'une très forte (p.2, score 0.9), l'autre faible (p.20,
    score 0.1). Le rognage historique gardait p.20 (distance nulle à un match) ; le
    remplissage par valeur garde les voisines immédiates de la zone forte."""
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)
    doc = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Catalogue")
    _fill_pages(db_session, doc.id, range(1, 26))
    db_session.commit()

    passages = [
        {"document_id": doc.id, "document_title": doc.title, "page_no": 2, "score": 0.9},
        {"document_id": doc.id, "document_title": doc.title, "page_no": 20, "score": 0.1},
    ]
    msg = build_cag_context(
        db_session,
        passages,
        system_prompt=SYS,
        token_budget=70,  # ≈ 3 pages
        full_doc_max_tokens=1,
        page_radius=3,
    )
    content = msg["content"]

    assert "PAGE_2_" in content and "PAGE_1_" in content and "PAGE_3_" in content
    assert "PAGE_20_" not in content, "la zone faiblement matchée doit céder la place"
    assert msg["cag_documents"][0]["seed_pages"] == [2]


def test_distant_matched_page_survives_the_seed_cap(db_session: Session):
    """Régression prod (27/07, crémone TGY3704) : dans un catalogue de 180 pages, la page
    89 — trouvée par 3 canaux et reclassée 0,901 — était écartée parce que les 3 meilleures
    pages matchées étaient groupées 40 pages plus tôt, alors que 72 % du budget restait
    inutilisé. Une page MATCHÉE ne doit jamais céder la place à la voisine d'une autre."""
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)
    doc = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Catalogue conception")
    _fill_pages(db_session, doc.id, range(40, 96))
    _add_page(db_session, document_id=doc.id, page=89, text="RALLONGE_4E_POINT_TGY3704")
    db_session.commit()

    passages = [
        {"document_id": doc.id, "document_title": doc.title, "page_no": 46, "score": 0.95},
        {"document_id": doc.id, "document_title": doc.title, "page_no": 48, "score": 0.94},
        {"document_id": doc.id, "document_title": doc.title, "page_no": 52, "score": 0.93},
        # 4e page matchée, isolée mais excellente — au-delà du plafond de seeds.
        {"document_id": doc.id, "document_title": doc.title, "page_no": 89, "score": 0.90},
    ]
    msg = build_cag_context(
        db_session,
        passages,
        system_prompt=SYS,
        token_budget=100000,  # budget large : rien ne justifie de sacrifier une page matchée
        full_doc_max_tokens=1,
        page_radius=3,
    )

    assert "RALLONGE_4E_POINT_TGY3704" in msg["content"]
    assert 89 in msg["cag_documents"][0]["pages"]
    assert 89 in msg["cag_documents"][0]["seed_pages"]
    assert "[page 89 — ★ page retrouvée par la recherche]" in msg["content"]


def test_seed_cap_still_limits_halo_expansion(db_session: Session):
    """Le plafond de seeds continue de brider le HALO (anti « fenêtre pieuvre ») : les
    voisines d'une 4e page matchée n'entrent pas, seule la page matchée elle-même."""
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)
    doc = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Catalogue")
    _fill_pages(db_session, doc.id, range(1, 100))
    db_session.commit()

    passages = [
        {"document_id": doc.id, "document_title": doc.title, "page_no": p, "score": s}
        for p, s in ((10, 0.95), (11, 0.94), (12, 0.93), (80, 0.90))
    ]
    msg = build_cag_context(
        db_session,
        passages,
        system_prompt=SYS,
        token_budget=100000,
        full_doc_max_tokens=1,
        page_radius=3,
    )
    pages = msg["cag_documents"][0]["pages"]

    assert 80 in pages, "la page matchée isolée est packée"
    assert 79 not in pages and 81 not in pages, "mais pas son halo (plafond de seeds)"
    assert {7, 8, 9, 13, 14, 15} <= set(pages), "le halo des 3 meilleures seeds reste entier"


def test_greedy_window_never_trims_the_winning_page(db_session: Session):
    """Budget si serré qu'une seule page tient : ce doit être la page matchée."""
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)
    doc = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Catalogue")
    _fill_pages(db_session, doc.id, range(1, 26))
    db_session.commit()

    passages = [{"document_id": doc.id, "document_title": doc.title, "page_no": 17, "score": 0.9}]
    msg = build_cag_context(
        db_session,
        passages,
        system_prompt=SYS,
        token_budget=20,
        full_doc_max_tokens=1,
        page_radius=3,
    )

    assert "PAGE_17_" in msg["content"]
    assert msg["cag_documents"][0]["pages"] == [17]


# ---------------------------------------------------------------------------
# T6 — L'ancre ne confisque plus tous les slots
# ---------------------------------------------------------------------------


def test_anchor_slots_leave_room_for_the_retrieved_document(db_session: Session):
    """Trois documents ancrés (issus d'un tour raté) ne doivent plus occuper les trois
    slots CAG : le document réellement retrouvé ce tour doit être packé."""
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)

    anchors = []
    for i in range(3):
        d = _make_doc(db_session, user_id=user.id, library_id=lib.id, title=f"Ancre {i}")
        _add_page(db_session, document_id=d.id, page=1, text=f"ANCRE_{i}_CONTENU")
        anchors.append(d)
    trouve = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Bon document")
    _add_page(db_session, document_id=trouve.id, page=1, text="BONNE_REPONSE")
    db_session.commit()

    passages = [
        {"document_id": trouve.id, "document_title": trouve.title, "page_no": 1, "score": 0.9}
    ]
    msg = build_cag_context(
        db_session,
        passages,
        system_prompt=SYS,
        max_documents=3,
        anchor_document_ids=[d.id for d in anchors],
    )

    packed = [d["document_id"] for d in msg["cag_documents"]]
    assert trouve.id in packed
    assert "BONNE_REPONSE" in msg["content"]
    # Une seule ancre packée de force (CAG_ANCHOR_SLOTS=1) ; depuis A3 elle est classée à
    # son score réel, donc le document réellement retrouvé prend le rang 1.
    assert packed[0] == trouve.id
    assert sum(1 for d in packed if d in {a.id for a in anchors}) == 1


def test_anchor_slots_zero_disables_forced_packing(db_session: Session, monkeypatch):
    monkeypatch.setattr(settings, "CAG_ANCHOR_SLOTS", 0)
    user = create_test_user(db_session, "responsable")
    lib = _library(db_session, user.id)
    ancre = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Ancre")
    _add_page(db_session, document_id=ancre.id, page=1, text="ANCRE_CONTENU")
    trouve = _make_doc(db_session, user_id=user.id, library_id=lib.id, title="Bon document")
    _add_page(db_session, document_id=trouve.id, page=1, text="BONNE_REPONSE")
    db_session.commit()

    passages = [
        {"document_id": trouve.id, "document_title": trouve.title, "page_no": 1, "score": 0.9}
    ]
    msg = build_cag_context(
        db_session,
        passages,
        system_prompt=SYS,
        anchor_document_ids=[ancre.id],
    )

    assert [d["document_id"] for d in msg["cag_documents"]] == [trouve.id]
