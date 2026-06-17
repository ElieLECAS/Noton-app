"""Tests taxonomie et filtre classification document."""
from __future__ import annotations

from sqlalchemy import text

from app.config import settings
from app.models.document import Document
from app.models.document_space import DocumentSpace
from app.models.space import Space
from app.catalog.taxonomy import (
    compute_classification_status,
    resolve_proferm_gammes_from_text,
    slot_material_to_document,
    supplier_uses_proferm_gammes,
    validate_classification,
)
from app.services.document_classification_filter import build_slot_classification_filter
from app.services.slot_filling_service import SlotState
from tests.conftest import create_test_user


def test_validate_classification_complete():
    ok, errors = validate_classification(
        supplier="Profine",
        product_types=["fenetre"],
        materials=["pvc"],
        coulissant_galandage=None,
    )
    assert ok is True
    assert errors == []


def test_validate_classification_requires_galandage_for_coulissant():
    ok, errors = validate_classification(
        supplier="Technal",
        product_types=["coulissant"],
        materials=["alu"],
        coulissant_galandage=None,
    )
    assert ok is False
    assert any("galandage" in e.lower() for e in errors)


def test_slot_material_mixte_maps_to_hybride():
    assert slot_material_to_document("mixte") == "hybride"


def test_resolve_proferm_gammes_perform():
    assert "perform" in resolve_proferm_gammes_from_text("fenêtre Perform 70")


def test_build_slot_filter_pvc_fenetre():
    state = SlotState(type="fenetre", material="pvc", intent_type="diagnostic_probleme")
    clause, params = build_slot_classification_filter(state)
    assert "classification_status" in clause
    assert params["filter_product_types"] == ["fenetre"]
    assert params["filter_materials"] == ["pvc"]


def test_build_slot_filter_galandage_coulissant():
    state = SlotState(
        type="coulissant",
        material="alu",
        galandage="oui",
        intent_type="diagnostic_probleme",
    )
    _, params = build_slot_classification_filter(state)
    assert params["filter_galandage_values"] == ["oui", "both"]


def test_build_slot_filter_proferm_gamme_from_range():
    state = SlotState(
        type="fenetre",
        material="pvc",
        range_or_model="Perform 70",
        intent_type="recherche_reference",
    )
    _, params = build_slot_classification_filter(state)
    assert params["filter_proferm_gammes"] == ["perform"]


def test_build_slot_filter_skips_proferm_gamme_for_alu_without_profine():
    state = SlotState(
        type="fenetre",
        material="alu",
        supplier_brand="Technal",
        range_or_model="Soleal 55",
        intent_type="recherche_reference",
    )
    _, params = build_slot_classification_filter(state)
    assert "filter_proferm_gammes" not in params


def test_build_slot_filter_proferm_gamme_for_profine_supplier():
    state = SlotState(
        type="fenetre",
        material="alu",
        supplier_brand="Profine",
        range_or_model="Perform 70",
        intent_type="recherche_reference",
    )
    _, params = build_slot_classification_filter(state)
    assert params.get("filter_proferm_gammes") == ["perform"]


def test_supplier_profile_profine_uses_proferm_gammes():
    assert supplier_uses_proferm_gammes("Profine") is True
    assert supplier_uses_proferm_gammes("Technal") is False


def test_compute_classification_status_incomplete_without_supplier():
    status = compute_classification_status(
        supplier=None,
        product_types=["fenetre"],
        materials=["pvc"],
        coulissant_galandage=None,
    )
    assert status == "incomplete"


def _classified_doc(
    db_session,
    user_id: int,
    library_id: int,
    *,
    title: str,
    materials: list[str],
    product_types: list[str] | None = None,
    classification_status: str = "complete",
) -> Document:
    doc = Document(
        title=title,
        content="Notice technique",
        document_type="document",
        processing_status="completed",
        processing_progress=100,
        library_id=library_id,
        user_id=user_id,
        source="Profine",
        product_types=product_types or ["fenetre"],
        materials=materials,
        classification_status=classification_status,
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    return doc


def _link_doc_to_space(db_session, doc: Document, space: Space, user_id: int) -> None:
    db_session.add(
        DocumentSpace(document_id=doc.id, space_id=space.id, user_id=user_id)
    )
    db_session.commit()


def test_sql_prefilter_excludes_alu_doc_for_pvc_fenetre_slots(db_session, monkeypatch):
    """Scénario poignée fenêtre PVC : un catalogue alu-only est exclu du pré-filtre SQL."""
    monkeypatch.setattr(settings, "RAG_REQUIRE_CLASSIFICATION", True)
    user = create_test_user(db_session, "responsable")
    from app.services.library_service import get_or_create_user_library

    library = get_or_create_user_library(db_session, user.id)
    space = Space(name="Espace RAG test", user_id=user.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)

    pvc_doc = _classified_doc(
        db_session, user.id, library.id, title="Notice PVC poignée", materials=["pvc"]
    )
    alu_doc = _classified_doc(
        db_session, user.id, library.id, title="LUMEAL GA catalogue", materials=["alu"]
    )
    incomplete_doc = _classified_doc(
        db_session,
        user.id,
        library.id,
        title="Doc non classé",
        materials=["pvc"],
        classification_status="incomplete",
    )
    _link_doc_to_space(db_session, pvc_doc, space, user.id)
    _link_doc_to_space(db_session, alu_doc, space, user.id)
    _link_doc_to_space(db_session, incomplete_doc, space, user.id)

    slot_state = SlotState(
        type="fenetre",
        material="pvc",
        problem_symptom="poignée ferme mal",
        intent_type="diagnostic_probleme",
    )
    clause, params = build_slot_classification_filter(slot_state)
    sql = text(
        f"""
        SELECT DISTINCT d.id
        FROM document d
        INNER JOIN document_space ds ON ds.document_id = d.id
        WHERE ds.space_id = :space_id
          AND {clause}
        """
    )
    doc_ids = [
        row[0]
        for row in db_session.execute(sql, {"space_id": space.id, **params})
    ]

    assert pvc_doc.id in doc_ids
    assert alu_doc.id not in doc_ids
    assert incomplete_doc.id not in doc_ids


def test_slot_state_mixte_maps_to_hybride_for_filter():
    state = SlotState(material="mixte")
    assert state.document_filter_material() == "hybride"


def test_slot_state_resolves_proferm_gamme_from_range():
    state = SlotState(range_or_model="Perform 70")
    assert "perform" in state.resolved_proferm_gammes()
