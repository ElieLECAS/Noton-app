"""
Construction du filtre SQL document à partir des slots validés (slot filling).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.catalog.taxonomy import (
    resolve_proferm_gammes_from_text,
    slot_material_to_document,
)
from app.catalog.slot_config import should_apply_proferm_gamme_filter
from app.config import settings
from app.services.slot_filling_service import SlotState

logger = logging.getLogger(__name__)


def _slot_galandage_to_doc_values(galandage: Optional[str]) -> Optional[List[str]]:
    if not galandage or galandage == "inconnu":
        return None
    if galandage == "oui":
        return ["oui", "both"]
    if galandage == "non":
        return ["non", "both"]
    return None


def build_slot_classification_filter(
    slot_state: Optional[SlotState],
) -> Tuple[str, Dict[str, Any]]:
    """
    Retourne (clause_sql, params) à injecter dans la requête document_ids.
    Chaîne vide si filtre désactivé ou aucun critère slot.
    """
    if not settings.RAG_REQUIRE_CLASSIFICATION:
        return "", {}

    if slot_state is None:
        return "", {}

    params: Dict[str, Any] = {}
    clauses: List[str] = ["d.classification_status = 'complete'"]

    if slot_state.type:
        params["filter_product_types"] = [slot_state.type]
        clauses.append("d.product_types && CAST(:filter_product_types AS varchar[])")

    doc_material = slot_material_to_document(slot_state.material)
    if doc_material:
        params["filter_materials"] = [doc_material]
        clauses.append("d.materials && CAST(:filter_materials AS varchar[])")

    if slot_state.type == "coulissant":
        galandage_values = _slot_galandage_to_doc_values(slot_state.galandage)
        if galandage_values:
            params["filter_galandage_values"] = galandage_values
            clauses.append("d.coulissant_galandage = ANY(:filter_galandage_values)")

    if slot_state.supplier_brand:
        params["filter_supplier"] = slot_state.supplier_brand
        clauses.append("LOWER(d.source) = LOWER(:filter_supplier)")

    proferm_gammes: List[str] = []
    if slot_state.range_or_model:
        proferm_gammes.extend(resolve_proferm_gammes_from_text(slot_state.range_or_model))
    if not proferm_gammes and slot_state.problem_symptom:
        proferm_gammes.extend(resolve_proferm_gammes_from_text(slot_state.problem_symptom))

    if should_apply_proferm_gamme_filter(
        material=slot_state.material,
        supplier_brand=slot_state.supplier_brand,
        resolved_gammes=proferm_gammes,
    ):
        params["filter_proferm_gammes"] = proferm_gammes
        clauses.append("d.proferm_gammes && CAST(:filter_proferm_gammes AS varchar[])")

    if len(clauses) <= 1:
        # classification_status seul — toujours appliquer en mode strict
        pass

    clause_sql = " AND ".join(clauses)
    logger.info(
        "Filtre classification document: %s (params=%s)",
        clause_sql,
        {k: v for k, v in params.items()},
    )
    return clause_sql, params


def build_no_documents_message(slot_state: Optional[SlotState]) -> str:
    """Message utilisateur quand aucun document classé compatible."""
    parts: List[str] = []
    if slot_state and slot_state.type:
        type_labels = {"fenetre": "fenêtre", "porte": "porte", "coulissant": "coulissant"}
        parts.append(type_labels.get(slot_state.type, slot_state.type))
    if slot_state and slot_state.material and slot_state.material != "inconnu":
        mat_labels = {"pvc": "PVC", "alu": "aluminium", "mixte": "hybride", "bois": "bois"}
        parts.append(mat_labels.get(slot_state.material, slot_state.material))
    if parts:
        criteria = " ".join(parts)
        return (
            f"Aucun document classé compatible ({criteria}) n'est disponible dans cet espace. "
            "Vérifiez la classification des documents dans la bibliothèque."
        )
    return (
        "Aucun document classé compatible n'est disponible dans cet espace. "
        "Vérifiez la classification des documents dans la bibliothèque."
    )
