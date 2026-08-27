"""Exploration bornée au document élu (phase B du retriever).

Pathologie mesurée en production (2026-08-27) qui motive ce lot : sur la question
« Comment se fait la décompression pour la gamme Perform 76 ? », l'élection ET le juge
de suffisance désignaient tous deux le SEUL document 438. Le contexte de génération
contenait pourtant 3 documents, dont une notice de quincaillerie Roto écartée par les
deux mécanismes — parce que la coupe globale laissait tous les documents dans le pool et
que le packer remplissait ensuite ses slots avec les meilleurs candidats RRF restants.

  * X1 « les non élus ne passent plus » — le pool final ne contient QUE des pages des
    documents élus (le cœur du lot).
  * X2 « recherche bornée » — les retrievers sont bien rappelés avec doc_ids=[élu].
  * X3 « plus de quota par document » — un document seul peut occuper tout le top_k
    (l'ancien quota le plafonnait à 8 pages).
  * X4 « scores recalés » — un score borné, mécaniquement plus élevé faute de
    concurrents, ne doit pas faire remonter un complément au-dessus du dominant.
  * X5 « profil » — les trois archétypes de document sont reconnus.
"""
from __future__ import annotations

import uuid
from unittest import mock

import pytest
from sqlmodel import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.library import Library
from app.services import space_search_service as sss
from app.services.context_packer_service import (
    MODE_FULL_TEXT,
    MODE_IMAGE_FIRST,
    MODE_WINDOWED,
    invalidate_document_fulltext_cache,
    profile_document,
)
from app.services.document_election_service import ElectedDocument, ElectionResult
from app.services.page_retrieval_service import UnifiedPageHit
from tests.conftest import create_test_user


def _hit(doc_id: int, page: int, rrf: float, *, sources=("bm25",), colpali=None, bm25=None):
    return UnifiedPageHit(
        document_id=doc_id,
        page_no=page,
        rrf_score=rrf,
        retrieval_sources=list(sources),
        document_title=f"Doc {doc_id}",
        colpali_score=colpali,
        bm25_score=bm25,
    )


def _elected(doc_id: int, score: float, *, role="dominant") -> ElectedDocument:
    return ElectedDocument(
        document_id=doc_id,
        title=f"Doc {doc_id}",
        election_score=score,
        score_max=score,
        role=role,
    )


# ---------------------------------------------------------------------------
# X3 — répartition des slots
# ---------------------------------------------------------------------------


def test_x3_un_document_seul_prend_tout_le_top_k():
    """L'ancien quota plafonnait à ceil(20 × 0.4) = 8 pages par document."""
    assert sss._page_slots_for_roles(1, 20) == [20]


def test_slots_repartis_quand_plusieurs_elus():
    assert sss._page_slots_for_roles(2, 20) == [13, 7]
    assert sss._page_slots_for_roles(3, 20) == [12, 5, 3]
    # Chaque élu garde au moins une page, même avec un top_k minuscule.
    assert sss._page_slots_for_roles(3, 1) == [1, 1, 1]


def test_slots_degrade_proprement_au_dela_de_trois():
    assert sss._page_slots_for_roles(5, 20) == [4, 4, 4, 4, 4]


# ---------------------------------------------------------------------------
# X4 — recalage des scores bornés
# ---------------------------------------------------------------------------


def test_x4_le_score_borne_ne_depasse_pas_le_plafond_de_phase_a():
    """Un pool borné donne de meilleurs rangs, donc des RRF plus élevés.

    Sans recalage, ces scores gonflés remonteraient dans le packer et pourraient faire
    passer un document de complément devant le dominant.
    """
    phase_a = [_hit(1, 5, 0.030)]
    scoped = [_hit(1, 7, 0.0320), _hit(1, 5, 0.0310)]  # scores bornés > phase A

    merged = sss._merge_scoped_into_phase_a(phase_a, scoped, image_first=False)

    assert max(float(h.rrf_score) for h in merged) == pytest.approx(0.030)
    # L'ordre INTERNE vient bien de la recherche bornée : p.7 devant p.5.
    assert [h.page_no for h in merged] == [7, 5]


def test_merge_conserve_les_pages_que_la_recherche_bornee_na_pas_retrouvees():
    phase_a = [_hit(1, 5, 0.030), _hit(1, 9, 0.020)]
    scoped = [_hit(1, 7, 0.031)]

    merged = sss._merge_scoped_into_phase_a(phase_a, scoped, image_first=False)

    assert {h.page_no for h in merged} == {5, 7, 9}


def test_merge_conserve_le_boost_dancre_et_les_canaux_de_phase_a():
    """Le boost d'ancre conversationnelle est appliqué AVANT l'élection : il ne doit pas
    être perdu par la passe bornée, qui ne le connaît pas."""
    ancre = _hit(1, 5, 0.050, sources=("bm25",), bm25=0.9)  # score déjà boosté
    scoped = [_hit(1, 5, 0.020, sources=("colpali",), colpali=0.7)]

    merged = sss._merge_scoped_into_phase_a([ancre], scoped, image_first=False)

    assert merged[0].rrf_score == pytest.approx(0.050)
    assert set(merged[0].retrieval_sources) == {"bm25", "colpali"}
    assert merged[0].bm25_score == pytest.approx(0.9)
    assert merged[0].colpali_score == pytest.approx(0.7)


def test_merge_marque_image_first_sur_document_muet():
    merged = sss._merge_scoped_into_phase_a(
        [], [_hit(1, 3, 0.02)], image_first=True
    )
    assert merged[0].image_first is True


def test_merge_sans_resultat_borne_retombe_sur_phase_a():
    phase_a = [_hit(1, 5, 0.030)]
    assert sss._merge_scoped_into_phase_a(phase_a, [], image_first=False) == phase_a


# ---------------------------------------------------------------------------
# X1 + X2 — exploration : seuls les élus survivent, avec recherche bornée
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_x1_les_documents_non_elus_disparaissent_du_pool():
    """Reproduit le cas prod : 438 élu, 391 (Roto) et 413 (TROCAL) non élus.

    Avant ce lot, 391 et 413 restaient dans le pool final et le packer les repêchait.
    """
    fused = [
        _hit(438, 7, 0.0311),
        _hit(438, 4, 0.0296),
        _hit(391, 73, 0.0164),   # notice quincaillerie — écartée par l'élection
        _hit(413, 48, 0.0159),   # TROCAL — écarté par l'élection
    ]
    election = ElectionResult(elected=[_elected(438, 0.0404)], decision="mono_document")

    with mock.patch.object(sss, "_run_retrievers", return_value=([], [])) as run, \
         mock.patch(
             "app.services.context_packer_service.profile_document",
             return_value=mock.MagicMock(
                 mode=MODE_FULL_TEXT,
                 to_trace=lambda: {"mode": MODE_FULL_TEXT},
             ),
         ):
        final_hits, trace = await sss._explore_elected_documents(
            mock.MagicMock(),
            election,
            space_id=28,
            colpali_q="décompression Perform 76",
            lexical_q="décompression Perform 76",
            use_colpali=True,
            embed_cache={},
            fused_hits=fused,
            top_k=20,
            pool_size=40,
        )

    assert {h.document_id for h in final_hits} == {438}, (
        "un document non élu subsiste dans le pool final"
    )
    assert [h.page_no for h in final_hits] == [7, 4]
    assert trace["documents"][0]["document_id"] == 438
    # X2 : la recherche bornée ne porte que sur le document élu.
    assert run.call_count == 1
    assert run.call_args.args[2] == [438]


@pytest.mark.asyncio
async def test_x2_la_recherche_bornee_reutilise_lencodage_colpali():
    """Le ré-encodage de la requête coûte jusqu'à 23 s sur CPU : le cache doit circuler."""
    election = ElectionResult(elected=[_elected(1, 0.04)], decision="mono_document")
    cache = {"colpali_query": [[0.1, 0.2]]}

    with mock.patch.object(sss, "_run_retrievers", return_value=([], [])) as run, \
         mock.patch(
             "app.services.context_packer_service.profile_document",
             return_value=mock.MagicMock(mode=MODE_WINDOWED, to_trace=lambda: {}),
         ):
        await sss._explore_elected_documents(
            mock.MagicMock(),
            election,
            space_id=1,
            colpali_q="q",
            lexical_q="q",
            use_colpali=True,
            embed_cache=cache,
            fused_hits=[_hit(1, 2, 0.03)],
            top_k=20,
            pool_size=40,
        )

    assert run.call_args.kwargs["embed_cache"] is cache


@pytest.mark.asyncio
async def test_mode_texte_integral_ne_relance_pas_bm25_borne():
    """Le texte partant EN ENTIER au packer, chercher dedans n'apporterait rien.

    ColPali borné tourne quand même : c'est lui qui choisit les pages envoyées en image,
    et le plafond de 8 images rend ce choix déterminant sur les planches cotées.
    """
    election = ElectionResult(elected=[_elected(1, 0.04)], decision="mono_document")
    captured = {}

    async def _fake_run(session, space_id, doc_ids, cq, lq, pool, **kwargs):
        captured["doc_ids"] = doc_ids
        return ([_hit(1, 3, 0.02, sources=("colpali",), colpali=0.6)], [_hit(1, 9, 0.01)])

    with mock.patch.object(sss, "_run_retrievers", side_effect=_fake_run), \
         mock.patch(
             "app.services.context_packer_service.profile_document",
             return_value=mock.MagicMock(mode=MODE_FULL_TEXT, to_trace=lambda: {}),
         ):
        final_hits, _ = await sss._explore_elected_documents(
            mock.MagicMock(),
            election,
            space_id=1,
            colpali_q="q",
            lexical_q="q",
            use_colpali=True,
            embed_cache={},
            fused_hits=[_hit(1, 3, 0.03)],
            top_k=20,
            pool_size=40,
        )

    assert captured["doc_ids"] == [1]
    # La page BM25 bornée (p.9) est ignorée en mode texte-intégral ; p.3 (ColPali) reste.
    assert 9 not in {h.page_no for h in final_hits}


@pytest.mark.asyncio
async def test_exploration_repartit_les_pages_entre_deux_elus():
    election = ElectionResult(
        elected=[_elected(1, 0.05), _elected(2, 0.04, role="complement")],
        decision="comparative:2",
    )
    fused = [_hit(1, p, 0.05 - p * 0.001) for p in range(1, 20)]
    fused += [_hit(2, p, 0.03 - p * 0.001) for p in range(1, 20)]

    with mock.patch.object(sss, "_run_retrievers", return_value=([], [])), \
         mock.patch(
             "app.services.context_packer_service.profile_document",
             return_value=mock.MagicMock(mode=MODE_FULL_TEXT, to_trace=lambda: {}),
         ):
        final_hits, trace = await sss._explore_elected_documents(
            mock.MagicMock(),
            election,
            space_id=1,
            colpali_q="q",
            lexical_q="q",
            use_colpali=True,
            embed_cache={},
            fused_hits=fused,
            top_k=20,
            pool_size=40,
        )

    per_doc = {}
    for hit in final_hits:
        per_doc[hit.document_id] = per_doc.get(hit.document_id, 0) + 1
    assert per_doc == {1: 13, 2: 7}
    assert len(trace["documents"]) == 2
    # final_rank réaffecté, dominant d'abord.
    assert final_hits[0].document_id == 1
    assert final_hits[0].final_rank == 1


# ---------------------------------------------------------------------------
# X5 — profilage des trois archétypes (base réelle)
# ---------------------------------------------------------------------------


def _doc_with_pages(
    session: Session, *, user_id: int, library_id: int, title: str, pages: int, chars: int
) -> Document:
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
    for page in range(1, pages + 1):
        body = f"PAGE_{page}_" + "x" * chars
        session.add(
            DocumentChunk(
                document_id=doc.id,
                chunk_index=page,
                content=body,
                text=body,
                is_leaf=True,
                hierarchy_level=0,
                node_id=f"n_{uuid.uuid4().hex[:8]}",
                start_char=0,
                end_char=len(body),
                metadata_json={"page_no": page, "content_type": "semantic_leaf"},
            )
        )
    session.commit()
    invalidate_document_fulltext_cache(doc.id)
    return doc


@pytest.fixture
def _lib(db_session: Session):
    user = create_test_user(db_session, "responsable")
    lib = Library(name="Lib", user_id=user.id, is_global=False)
    db_session.add(lib)
    db_session.commit()
    db_session.refresh(lib)
    return user, lib


def test_x5_document_texte_riche_qui_tient_en_entier(db_session: Session, _lib):
    user, lib = _lib
    doc = _doc_with_pages(
        db_session, user_id=user.id, library_id=lib.id,
        title="Notice courte", pages=10, chars=400,
    )
    profile = profile_document(db_session, doc.id)
    assert profile.mode == MODE_FULL_TEXT
    assert profile.page_count == 10
    assert profile.pages_with_text == 10


def test_x5_document_trop_gros_passe_en_fenetre(db_session: Session, _lib):
    user, lib = _lib
    doc = _doc_with_pages(
        db_session, user_id=user.id, library_id=lib.id,
        title="Gros DTA", pages=60, chars=3000,
    )
    profile = profile_document(db_session, doc.id, full_doc_max_tokens=1000)
    assert profile.mode == MODE_WINDOWED


def test_x5_planche_muette_passe_en_image_first(db_session: Session, _lib):
    """Cas réel : la page des parcloses du cahier Perform 76 est transcrite en simple
    liste d'étiquettes (145 caractères, aucune cote) — le texte ne peut pas répondre."""
    user, lib = _lib
    doc = _doc_with_pages(
        db_session, user_id=user.id, library_id=lib.id,
        title="Planches CAO", pages=24, chars=8,
    )
    profile = profile_document(db_session, doc.id)
    assert profile.mode == MODE_IMAGE_FIRST


def test_profil_document_vide_ne_leve_pas(db_session: Session, _lib):
    user, lib = _lib
    doc = Document(
        title="Vide",
        document_type="written",
        processing_status="completed",
        library_id=lib.id,
        user_id=user.id,
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    invalidate_document_fulltext_cache(doc.id)

    profile = profile_document(db_session, doc.id)
    assert profile.mode == MODE_IMAGE_FIRST
    assert profile.text_tokens == 0
    assert profile.page_count == 0
