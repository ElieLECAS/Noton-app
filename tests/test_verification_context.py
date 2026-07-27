"""Vérification post-génération — le juge doit voir le contexte RÉELLEMENT donné.

Régression prod du 27/07 : sur une réponse CORRECTE, le juge déclarait « la réponse invente
tout, le contexte concerne exclusivement la gamme SOLEAL FY ». Il recevait
``context_text[:20000]`` d'un texte qui commençait par 5 044 caractères de prompt système —
soit 8 à 30 % du contexte réel, c'est-à-dire la tête du document élu n°1 (un catalogue FY),
sans jamais voir le document 2 (le catalogue GY qui contenait la réponse).
"""
from __future__ import annotations

import pytest

import app.services.response_verification_service as rvs
from app.config import settings
from app.services.response_verification_service import (
    _document_manifest,
    _split_block_into_pages,
    build_verification_context,
)

# Reproduction du contexte du tour litigieux : le bon document est le SECOND.
BLOCK_FY = (
    "DOCUMENT 1 : SOLEAL-FY-55-65-QC-conception\n"
    "Pages incluses : 1-2 (extrait)\n\n"
    "[page 1]\nFenetres francaises FY 55 et 65, generalites.\n\n"
    "[page 2]\nSuite du catalogue FY, dormants battants.\n"
)
BLOCK_GY = (
    "DOCUMENT 2 : SOLEAL-GY-55-Catalogue-conception\n"
    "Pages incluses : 89 (extrait)\n\n"
    "[page 89 — ★ page retrouvée par la recherche]\n"
    "Choix des fermetures : TGY3702 fermeture 3 points, TGY3704 rallonge 4eme point.\n"
)
CAG_DOCUMENTS = [
    {
        "index": 1,
        "document_id": 10,
        "document_title": "SOLEAL-FY-55-65-QC-conception",
        "pages": [1, 2],
        "seed_pages": [1],
        "full_document": False,
    },
    {
        "index": 2,
        "document_id": 20,
        "document_title": "SOLEAL-GY-55-Catalogue-conception",
        "pages": [89],
        "seed_pages": [89],
        "full_document": False,
    },
]


# ---------------------------------------------------------------------------
# Manifeste — le garde-fou structurel
# ---------------------------------------------------------------------------


def test_manifest_lists_every_packed_document():
    manifest = _document_manifest(CAG_DOCUMENTS)
    assert "SOLEAL-FY-55-65-QC-conception" in manifest
    assert "SOLEAL-GY-55-Catalogue-conception" in manifest
    assert "pages retrouvées par la recherche : 89" in manifest


def test_manifest_survives_even_when_content_is_truncated():
    """LE correctif : même si le texte du document 2 ne tient pas, son existence est
    annoncée — le juge ne peut plus conclure « le contexte ne parle que de FY »."""
    text, report = build_verification_context(
        [BLOCK_FY, BLOCK_GY], CAG_DOCUMENTS, max_chars=120
    )
    assert report["truncated"] is True
    assert "SOLEAL-GY-55-Catalogue-conception" in text


def test_manifest_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "VERIFICATION_INCLUDE_MANIFEST", False)
    text, _ = build_verification_context([BLOCK_FY], CAG_DOCUMENTS[:1], max_chars=0)
    assert "manifeste COMPLET" not in text


# ---------------------------------------------------------------------------
# Contexte complet et couverture
# ---------------------------------------------------------------------------


def test_full_context_passes_whole_under_cap():
    text, report = build_verification_context(
        [BLOCK_FY, BLOCK_GY], CAG_DOCUMENTS, max_chars=60000
    )
    assert report["truncated"] is False
    assert report["coverage"] == 1.0
    assert "TGY3704 rallonge 4eme point" in text
    assert "dormants battants" in text


def test_zero_cap_means_unlimited():
    _, report = build_verification_context([BLOCK_FY, BLOCK_GY], CAG_DOCUMENTS, max_chars=0)
    assert report["truncated"] is False
    assert report["coverage"] == 1.0


def test_coverage_never_exceeds_one():
    _, report = build_verification_context([BLOCK_FY], CAG_DOCUMENTS[:1], max_chars=0)
    assert 0.0 <= report["coverage"] <= 1.0


# ---------------------------------------------------------------------------
# Remplissage par les preuves d'abord
# ---------------------------------------------------------------------------


def test_cited_page_wins_over_head_of_context_when_truncated():
    """Sous plafond serré, la page CITÉE par la réponse (doc 2 p.89) doit être montrée
    avant le corps du document 1 — l'inverse exact de la troncature historique."""
    text, report = build_verification_context(
        [BLOCK_FY, BLOCK_GY],
        CAG_DOCUMENTS,
        cited_pages={20: [89]},
        max_chars=340,
    )
    assert report["truncated"] is True
    assert "TGY3704 rallonge 4eme point" in text
    assert "dormants battants" not in text


def test_pages_are_restored_in_reading_order():
    text, _ = build_verification_context(
        [BLOCK_FY], [CAG_DOCUMENTS[0]], cited_pages={10: [2]}, max_chars=0
    )
    assert text.index("[page 1]") < text.index("[page 2]")


def test_split_block_isolates_pages_and_header():
    units = _split_block_into_pages(BLOCK_GY)
    assert units[0][0] is None and "DOCUMENT 2" in units[0][1]
    assert units[1][0] == 89
    assert "TGY3704" in units[1][1]


def test_empty_blocks_are_safe():
    text, report = build_verification_context([], [], max_chars=0)
    assert report["coverage"] == 1.0
    assert text == ""


# ---------------------------------------------------------------------------
# Verdict LLM marqué suspect (J6)
# ---------------------------------------------------------------------------


async def _judge(**verdict):
    async def _fake(question, response_text, context_text, *, model):
        return {"answers_question": True, "grounded": True, "issues": [], "parse_error": False, **verdict}

    return _fake


@pytest.mark.asyncio
async def test_negative_verdict_on_truncated_context_is_marked_suspect(monkeypatch):
    """Un juge qui n'a pas tout vu ne peut pas conclure à une invention : son verdict est
    signalé et ne condamne plus seul la réponse."""
    monkeypatch.setattr(settings, "VERIFICATION_CONTEXT_MAX_CHARS", 120)
    monkeypatch.setattr(
        rvs, "judge_relevance", await _judge(grounded=False, issues=["contexte hors sujet"])
    )
    result = await rvs.verify_response(
        question="quelle rallonge pour TGY3702 ?",
        response_text="TGY3704.",
        context_text=BLOCK_FY + BLOCK_GY,
        model="test",
        document_blocks=[BLOCK_FY, BLOCK_GY],
        cag_documents=CAG_DOCUMENTS,
    )
    assert result["judge_suspect"] is True
    assert "% du contexte" in result["judge_suspect_reason"]
    assert result["ok"] is True  # le programmatique, qui a tout lu, ne signale rien
    assert result["grounded"] is False  # le verdict brut reste consultable


@pytest.mark.asyncio
async def test_verdict_contradicting_programmatic_check_is_marked_suspect(monkeypatch):
    """Contexte entier vu, mais le juge dit « valeurs inventées » alors que le contrôle
    programmatique confirme chaque cote citée : contradiction → verdict suspect."""
    monkeypatch.setattr(
        rvs, "judge_relevance", await _judge(grounded=False, issues=["cotes inventées"])
    )
    context = "Le seuil 9F67 mesure 20 mm de haut."
    result = await rvs.verify_response(
        question="dimensions du 9F67 ?",
        response_text="Le seuil 9F67 fait 20 mm.",
        context_text=context,
        model="test",
        document_blocks=[context],
        cag_documents=[{"index": 1, "document_id": 1, "document_title": "Doc", "pages": [1]}],
    )
    assert result["unsupported_claims"] == []
    assert result["judge_suspect"] is True
    assert "programmatique" in result["judge_suspect_reason"]
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_real_invention_still_fails(monkeypatch):
    """Garde-fou inverse : une cote réellement absente du contexte reste détectée, et le
    verdict n'est PAS blanchi."""
    monkeypatch.setattr(rvs, "judge_relevance", await _judge(grounded=False))
    context = "Le seuil 9F67 est livré avec ses embouts."
    result = await rvs.verify_response(
        question="dimensions du 9F67 ?",
        response_text="Le seuil 9F67 fait 125 mm.",
        context_text=context,
        model="test",
        document_blocks=[context],
        cag_documents=[{"index": 1, "document_id": 1, "document_title": "Doc", "pages": [1]}],
    )
    assert result["unsupported_claims"] == ["125 mm"]
    assert result["ok"] is False


@pytest.mark.asyncio
async def test_clean_verdict_stays_clean(monkeypatch):
    monkeypatch.setattr(rvs, "judge_relevance", await _judge())
    result = await rvs.verify_response(
        question="q",
        response_text="Le seuil 9F67 fait 20 mm.",
        context_text="Le seuil 9F67 mesure 20 mm.",
        model="test",
        document_blocks=["Le seuil 9F67 mesure 20 mm."],
        cag_documents=[{"index": 1, "document_id": 1, "document_title": "Doc", "pages": [1]}],
    )
    assert result["ok"] is True
    assert result["judge_suspect"] is False
    assert result["context"]["coverage"] == 1.0
