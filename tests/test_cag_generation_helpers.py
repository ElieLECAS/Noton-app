"""Helpers de génération CAG : sandwich (rappel final de tâche), légendes d'images,
sources UI par document (_build_document_sources)."""
from __future__ import annotations

from app.routers.chat import _build_document_sources
from app.services.rag_generation_service import (
    build_cag_task_reminder,
    build_rag_user_message,
)


# ---------------------------------------------------------------------------
# build_rag_user_message (sandwich + légendes)
# ---------------------------------------------------------------------------


def test_user_message_plain():
    msg = build_rag_user_message("Quelle vis pour le seuil ?")
    assert msg == {"role": "user", "content": "Quelle vis pour le seuil ?"}


def test_user_message_with_reminder_and_captions():
    captions = [
        {"image_index": 1, "document_index": 2, "document_title": "Notice seuil", "page_no": 5}
    ]
    msg = build_rag_user_message(
        "Quelle vis ?",
        images_b64=["QUJD"],
        image_captions=captions,
        task_reminder="RAPPEL FINAL — Réponds UNIQUEMENT à ma question ci-dessus.",
    )
    content = msg["content"]
    # Ordre : question → légendes → rappel (le rappel est TOUJOURS en dernier).
    assert content.startswith("Quelle vis ?")
    assert "Image 1 = DOCUMENT 2 « Notice seuil », page 5" in content
    assert content.rstrip().endswith("RAPPEL FINAL — Réponds UNIQUEMENT à ma question ci-dessus.")
    assert msg["images"] == ["QUJD"]


def test_task_reminder_includes_standalone_question_when_different():
    reminder = build_cag_task_reminder(
        "et le 76180 ?", standalone_question="Quelles sont les dimensions du seuil 76180 ?"
    )
    assert "Quelles sont les dimensions du seuil 76180 ?" in reminder
    assert "demande-moi d'abord lequel" in reminder


def test_task_reminder_skips_identical_standalone():
    reminder = build_cag_task_reminder("Quelle vis ?", standalone_question="quelle vis ?")
    assert "Question autonome" not in reminder


# ---------------------------------------------------------------------------
# _build_document_sources (sources UI par document packé)
# ---------------------------------------------------------------------------

_CAG_DOCS = [
    {
        "index": 1,
        "document_id": 11,
        "document_title": "Notice seuil PMR",
        "pages": [1, 2, 3],
        "matched_pages": [2],
        "full_document": True,
        "score": 0.91,
        "has_source_file": True,
    },
    {
        "index": 2,
        "document_id": 22,
        "document_title": "DTA",
        "pages": [5, 6, 7],
        "matched_pages": [6],
        "full_document": False,
        "score": 0.55,
        "has_source_file": False,
    },
]


def test_document_sources_without_used_block_lists_all():
    sources = _build_document_sources(_CAG_DOCS, {})
    assert [s["index"] for s in sources] == [1, 2]
    s1 = sources[0]
    assert s1["cag_document"] is True
    assert s1["document_title"] == "Notice seuil PMR"
    assert "Document complet" in s1["excerpt"]
    assert s1["page_no"] == 2  # atterrissage PDF = première page matchée
    assert s1["has_source_file"] is True
    s2 = sources[1]
    assert "Extrait" in s2["excerpt"] and "pages 5-7" in s2["excerpt"]


def test_document_sources_filtered_by_model_used_block():
    used = {2: [6, 7]}
    sources = _build_document_sources(_CAG_DOCS, used)
    assert len(sources) == 1
    s = sources[0]
    assert s["index"] == 2
    assert s["used_pages"] == [6, 7]
    assert s["page_no"] == 6  # atterrissage = première page utilisée
    assert "pages utilisées : 6, 7" in s["excerpt"]


def test_document_sources_fallback_when_used_block_matches_nothing():
    # Le modèle a renvoyé des index inconnus → on retombe sur tous les documents packés.
    sources = _build_document_sources(_CAG_DOCS, {99: [1]})
    assert [s["index"] for s in sources] == [1, 2]
