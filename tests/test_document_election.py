"""Élection du document (phase A du retriever) — « dans QUEL document est la réponse ? ».

Chaque test reproduit une pathologie mesurée sur le corpus réel :

  * E1 « preuve détruite » — un document qui place beaucoup de pages doit pouvoir
    gagner sur la distribution, signal que la coupe top_k + le quota par document
    effaçaient avant même que l'élection le voie.
  * E2 « catalogue vs notice » — mais le volume ne doit JAMAIS renverser une meilleure
    page nette (parité avec T1 de test_election_packing).
  * E3 « empilement par défaut » — trois documents qui scorent ne donnent plus trois
    documents dans le contexte : sans preuve, le dominant part seul (c'est la cause de
    la réponse qui recollait des références de gammes différentes).
  * E4 « comparaison légitime » — une question comparative admet bien un second document.
  * E5 « complémentarité d'archétypes » — notice texte + planche muette vue par ColPali.
  * E6 « versions concurrentes » — CC01 et CC02 du même cahier technique ne coexistent
    plus (cotes contradictoires, budget gaspillé).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from sqlmodel import Session

from app.models.document import Document
from app.models.library import Library
from app.services.document_election_service import (
    DOMINANCE_RATIO,
    ElectedDocument,
    dedupe_versions,
    elect_documents,
    election_candidate_passages,
    format_election_log,
    is_comparative_query,
    normalize_title_family,
    score_documents,
)
from app.services.page_retrieval_service import UnifiedPageHit
from tests.conftest import create_test_user


def _hit(
    doc_id: int,
    page: int,
    rrf: float,
    *,
    sources=("bm25",),
    title: str | None = None,
    colpali: float | None = None,
    bm25: float | None = None,
) -> UnifiedPageHit:
    return UnifiedPageHit(
        document_id=doc_id,
        page_no=page,
        rrf_score=rrf,
        retrieval_sources=list(sources),
        document_title=title or f"Doc {doc_id}",
        colpali_score=colpali,
        bm25_score=bm25,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_e1_distribution_departage_a_score_max_egal():
    """À meilleure page identique, le document qui en place plusieurs gagne.

    C'est exactement le signal que le quota (8 pages/doc) et la coupe top-20
    supprimaient avant que l'élection ne l'observe.
    """
    hits = [
        _hit(1, 4, 0.50),
        _hit(2, 4, 0.50),
        _hit(2, 5, 0.45),
        _hit(2, 6, 0.40),
        _hit(2, 7, 0.35),
    ]
    ranked = score_documents(hits)
    assert [d.document_id for d in ranked] == [2, 1]
    assert ranked[0].page_count == 4
    assert ranked[0].election_score > ranked[1].election_score


def test_e2_meilleure_page_nette_bat_le_volume():
    """Parité avec T1 : six pages moyennes ne battent pas LA bonne page."""
    catalogue = [_hit(1, p, 0.30) for p in range(1, 7)]
    notice = [_hit(2, 3, 0.90)]
    ranked = score_documents(catalogue + notice)
    assert ranked[0].document_id == 2, "le catalogue volumineux a repris l'avantage"


def test_bonus_familles_recompense_accord_des_canaux():
    """Deux canaux d'accord sur un document valent mieux qu'un seul, à score égal."""
    un_canal = [_hit(1, 2, 0.40, sources=("bm25",))]
    deux_canaux = [_hit(2, 2, 0.40, sources=("bm25", "colpali"))]
    ranked = score_documents(un_canal + deux_canaux)
    assert ranked[0].document_id == 2
    assert ranked[0].families == {"texte", "visuel"}


def test_pool_vide_rend_une_decision_explicite():
    result = elect_documents(None, [])
    assert result.decision == "empty"
    assert result.elected == []
    assert result.elected_ids == []
    assert "aucun document" in format_election_log(result)


def test_score_non_positif_ne_recoit_pas_de_bonus_multiplicatif():
    """Garde-fou : un score ≤ 0 ne doit pas être « amélioré » par le bonus.

    Les scores RRF sont structurellement positifs (1/(k+rang), et le boost d'ancre est
    un facteur > 1), donc ce chemin est défensif : on vérifie surtout qu'un pool
    dégénéré ne produit ni exception ni score gonflé.
    """
    ranked = score_documents([_hit(1, 1, 0.0), _hit(1, 2, -0.4)])
    assert len(ranked) == 1
    assert ranked[0].election_score <= 0.0


# ---------------------------------------------------------------------------
# Règle de décision 1 / 2 / 3 documents
# ---------------------------------------------------------------------------


def test_e3_sans_preuve_le_dominant_part_seul():
    """Trois documents aux scores PROCHES → un seul élu (pas d'empilement)."""
    hits = [
        _hit(1, 3, 0.50, sources=("bm25",)),
        _hit(2, 3, 0.48, sources=("bm25",)),
        _hit(3, 3, 0.47, sources=("bm25",)),
    ]
    result = elect_documents(None, hits, query_text="hauteur de poignee")
    assert result.decision == "mono_document"
    assert result.elected_ids == [1]
    # Les autres restent CANDIDATS : le juge doit pouvoir contredire l'élection.
    assert [c.document_id for c in result.candidates] == [1, 2, 3]
    assert result.margin == pytest.approx(0.48 / 0.50, rel=1e-3)


def test_e4_question_comparative_admet_un_second_document():
    hits = [
        _hit(1, 3, 0.50, sources=("bm25",)),
        _hit(2, 3, 0.48, sources=("bm25",)),
    ]
    result = elect_documents(
        None, hits, query_text="quelle difference entre le Perform 76 et le Kommerling 70 ?"
    )
    assert result.decision == "comparative:2"
    assert result.elected_ids == [1, 2]
    assert result.elected[1].role == "complement"
    assert result.elected[1].reason.startswith("comparative_query:")


def test_question_comparative_mais_score_trop_ecarte_reste_mono():
    """La preuve ne suffit pas : le second doit aussi être proche du dominant."""
    hits = [
        _hit(1, 3, 0.90, sources=("bm25",)),
        _hit(2, 3, 0.10, sources=("bm25",)),
    ]
    result = elect_documents(None, hits, query_text="comparer les deux gammes")
    assert result.decision == "mono_document"
    assert result.elected_ids == [1]


def test_e5_complementarite_archetypes_notice_plus_planche(monkeypatch):
    """Notice retrouvée par son TEXTE + planche muette que seul ColPali voit."""
    from app.services import document_election_service as svc

    monkeypatch.setattr(svc.settings, "COLPALI_DOMINANCE_MIN_SCORE", 0.55)
    hits = [
        _hit(1, 8, 0.50, sources=("bm25",), bm25=0.7),
        _hit(2, 14, 0.46, sources=("colpali",), colpali=0.80),
    ]
    result = elect_documents(None, hits, query_text="cote du dormant")
    assert result.decision == "complement:2"
    assert result.elected_ids == [1, 2]
    assert result.elected[1].reason == "channel_complement"


def test_complementarite_refusee_si_colpali_trop_faible(monkeypatch):
    from app.services import document_election_service as svc

    monkeypatch.setattr(svc.settings, "COLPALI_DOMINANCE_MIN_SCORE", 0.55)
    hits = [
        _hit(1, 8, 0.50, sources=("bm25",), bm25=0.7),
        _hit(2, 14, 0.46, sources=("colpali",), colpali=0.20),
    ]
    result = elect_documents(None, hits, query_text="cote du dormant")
    assert result.decision == "mono_document"
    assert result.elected_ids == [1]


def test_meme_famille_de_canaux_nest_pas_une_complementarite():
    """Deux documents texte proches ne sont pas complémentaires : ils sont concurrents."""
    hits = [
        _hit(1, 8, 0.50, sources=("bm25",), bm25=0.7),
        _hit(2, 9, 0.48, sources=("bm25",), bm25=0.6),
    ]
    result = elect_documents(None, hits, query_text="cote du dormant")
    assert result.elected_ids == [1]


def test_max_docs_plafonne_le_nombre_delus():
    hits = [
        _hit(1, 1, 0.50, sources=("bm25",)),
        _hit(2, 1, 0.49, sources=("bm25",)),
        _hit(3, 1, 0.48, sources=("bm25",)),
    ]
    result = elect_documents(
        None, hits, query_text="comparer 1 et 2", max_docs=2
    )
    assert len(result.elected) == 2


def test_seuil_de_dominance_est_bien_relatif():
    """Juste sous le seuil → écarté ; juste au-dessus → admis (question comparative)."""
    juste_sous = DOMINANCE_RATIO * 0.50 - 0.01
    juste_au_dessus = DOMINANCE_RATIO * 0.50 + 0.01
    q = "comparer les deux"

    r1 = elect_documents(
        None, [_hit(1, 1, 0.50, sources=("bm25",)), _hit(2, 1, juste_sous, sources=("bm25",))],
        query_text=q,
    )
    assert r1.elected_ids == [1]

    r2 = elect_documents(
        None, [_hit(1, 1, 0.50, sources=("bm25",)), _hit(2, 1, juste_au_dessus, sources=("bm25",))],
        query_text=q,
    )
    assert r2.elected_ids == [1, 2]


# ---------------------------------------------------------------------------
# Détection de comparaison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "quelle difference entre le 76177 et le 76185 ?",
        "peux-tu comparer les deux gammes",
        "Perform 76 vs Kommerling 70",
        "le 76185 par rapport au 76177",
        "quel est l'equivalent chez Technal ?",
    ],
)
def test_questions_comparatives_detectees(question):
    ok, marker = is_comparative_query(question)
    assert ok, f"non détectée : {question}"
    assert marker


@pytest.mark.parametrize(
    "question",
    [
        # Le piège central : une ÉNUMÉRATION cite beaucoup de références mais veut UN
        # document. La compter comme comparative rouvrirait le recollage inter-documents.
        "pour les dormants standards donne moi pour chaque seuil les embouts compatibles",
        "quelle est la charge maximale par ouvrant sur le pivot bas ?",
        "liste des references de parclose pour les dormants",
        "",
    ],
)
def test_questions_non_comparatives(question):
    ok, _ = is_comparative_query(question)
    assert not ok, f"faussement comparative : {question}"


# ---------------------------------------------------------------------------
# Déduplication de versions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (
            "Dossier technique PERFORM 76 24 08 2026",
            "Dossier technique PERFORM 76 26 08 2026",
        ),
        ("Cahier technique Perform 76 CC01", "Cahier technique Perform76 CC02"),
        ("DTA_6_16-2335_V4.pdf", "DTA_6_16-2335_V5.pdf"),
        ("Notice pose (1).pdf", "Notice pose.pdf"),
        ("Notice 2025-04-24", "Notice 2026-08-26"),
    ],
)
def test_titres_de_meme_famille_convergent(a, b):
    assert normalize_title_family(a) == normalize_title_family(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # Deux PRODUITS distincts ne doivent jamais fusionner (perte de rappel).
        ("Cahier technique Perform 76", "Cahier technique Perform 70"),
        ("DTA Kommerling 70", "DTA Trocal 88"),
        ("Notice pose fenetre", "Notice reglage volet"),
    ],
)
def test_titres_de_produits_distincts_ne_fusionnent_pas(a, b):
    assert normalize_title_family(a) != normalize_title_family(b)


def test_e6_deux_versions_ne_sont_pas_toutes_deux_candidates():
    """CC01 + CC02 → une seule survit, l'autre est tracée comme écartée."""
    hits = [
        _hit(1, 3, 0.50, title="Cahier technique Perform 76 CC01"),
        _hit(2, 3, 0.49, title="Cahier technique Perform 76 CC02"),
    ]
    result = elect_documents(None, hits, query_text="hauteur de poignee")
    assert len(result.candidates) == 1, "les deux révisions coexistent encore"
    assert len(result.rejected_versions) == 1
    assert result.rejected_versions[0]["reason"] == "version_superseded"
    assert result.rejected_versions[0]["kept_document_id"] == result.elected_ids[0]


def test_deux_versions_meme_sur_question_comparative_restent_une_seule():
    """Une question comparative ne doit pas ressusciter la version périmée."""
    hits = [
        _hit(1, 3, 0.50, title="Cahier technique Perform 76 CC01"),
        _hit(2, 3, 0.49, title="Cahier technique Perform 76 CC02"),
    ]
    result = elect_documents(None, hits, query_text="quelle difference entre les deux")
    assert len(result.elected) == 1


def test_dedupe_sans_titre_ne_fusionne_pas_par_defaut():
    """Titres vides : chaque document garde sa propre famille (pas de fusion aveugle)."""
    docs = [
        ElectedDocument(document_id=1, title="", election_score=0.5, score_max=0.5),
        ElectedDocument(document_id=2, title="", election_score=0.4, score_max=0.4),
    ]
    kept, rejected = dedupe_versions(None, docs)
    assert [d.document_id for d in kept] == [1, 2]
    assert rejected == []


def test_dedupe_arbitre_sur_updated_at(db_session: Session):
    """Avec la base : c'est la révision la plus récemment mise à jour qui survit."""
    user = create_test_user(db_session, "responsable")
    lib = Library(name="Lib", user_id=user.id, is_global=False)
    db_session.add(lib)
    db_session.commit()
    db_session.refresh(lib)

    titre = "Cahier technique Perform 76"
    vieux = Document(
        title=f"{titre} CC01",
        document_type="written",
        processing_status="completed",
        library_id=lib.id,
        user_id=user.id,
        updated_at=datetime(2026, 8, 24, 10, 0, 0),
    )
    recent = Document(
        title=f"{titre} CC02",
        document_type="written",
        processing_status="completed",
        library_id=lib.id,
        user_id=user.id,
        updated_at=datetime(2026, 8, 26, 16, 0, 0),
    )
    db_session.add_all([vieux, recent])
    db_session.commit()
    db_session.refresh(vieux)
    db_session.refresh(recent)

    # La VIEILLE version score mieux : seule la date doit la faire perdre.
    hits = [
        _hit(vieux.id, 3, 0.60, title=vieux.title),
        _hit(recent.id, 3, 0.55, title=recent.title),
    ]
    result = elect_documents(db_session, hits, query_text="hauteur de poignee")
    assert result.elected_ids == [recent.id]
    assert result.rejected_versions[0]["document_id"] == vieux.id


# ---------------------------------------------------------------------------
# Pack du juge & trace
# ---------------------------------------------------------------------------


def test_candidate_passages_couvre_tous_les_candidats_pas_seulement_les_elus():
    """Le juge doit garder une vue LARGE, sinon il ne peut jamais contredire l'élection."""
    hits = [
        _hit(1, 3, 0.50, sources=("bm25",)),
        _hit(1, 4, 0.45, sources=("bm25",)),
        _hit(2, 9, 0.40, sources=("bm25",)),
        _hit(3, 2, 0.35, sources=("bm25",)),
    ]
    result = elect_documents(None, hits, query_text="hauteur de poignee")
    assert result.elected_ids == [1]

    passages = election_candidate_passages(result)
    doc_ids = {p["document_id"] for p in passages}
    assert doc_ids == {1, 2, 3}, "le juge ne verrait que le document élu"
    # Passages LÉGERS : build_cag_context recharge le texte depuis la base.
    assert all("passage" not in p for p in passages)
    assert all(p["page_no"] > 0 for p in passages)


def test_candidate_passages_plafonne_les_pages_par_document():
    from app.services.document_election_service import CANDIDATE_PAGES_PER_DOC

    hits = [_hit(1, p, 0.50 - p * 0.01, sources=("bm25",)) for p in range(1, 20)]
    result = elect_documents(None, hits, query_text="test")
    passages = election_candidate_passages(result)
    assert len(passages) == CANDIDATE_PAGES_PER_DOC
    # Les pages retenues sont les mieux notées (ici les premières).
    assert [p["page_no"] for p in passages] == list(range(1, CANDIDATE_PAGES_PER_DOC + 1))


def test_trace_est_json_serialisable():
    """La trace part en colonne JSON et dans le SSE : aucun set/dataclass ne doit fuir."""
    import json

    hits = [
        _hit(1, 3, 0.50, sources=("bm25", "colpali")),
        _hit(2, 3, 0.20, sources=("bm25",)),
    ]
    result = elect_documents(None, hits, query_text="test")
    payload = json.dumps(result.to_trace())
    assert '"decision"' in payload
    assert '"elected"' in payload
    assert '"candidates"' in payload
