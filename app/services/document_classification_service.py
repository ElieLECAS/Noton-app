"""Helpers classification document (bibliothèque)."""

from __future__ import annotations

from typing import List, Optional

from app.catalog.taxonomy import compute_classification_status, validate_classification
from app.models.document import Document, DocumentCreate, DocumentUpdate


def apply_classification_to_document(
    document: Document,
    *,
    supplier: Optional[str] = None,
    product_types: Optional[List[str]] = None,
    materials: Optional[List[str]] = None,
    coulissant_galandage: Optional[str] = None,
    proferm_gammes: Optional[List[str]] = None,
) -> None:
    """Applique les champs classification et recalcule classification_status."""
    if supplier is not None:
        document.source = supplier.strip() if supplier else None
    if product_types is not None:
        document.product_types = list(product_types)
    if materials is not None:
        document.materials = list(materials)
    if coulissant_galandage is not None:
        document.coulissant_galandage = coulissant_galandage or None
    if proferm_gammes is not None:
        document.proferm_gammes = list(proferm_gammes)

    if "coulissant" not in (document.product_types or []):
        document.coulissant_galandage = None

    document.classification_status = compute_classification_status(
        supplier=document.source,
        product_types=document.product_types,
        materials=document.materials,
        coulissant_galandage=document.coulissant_galandage,
        proferm_gammes=document.proferm_gammes,
    )


def classification_from_create(document_create: DocumentCreate) -> dict:
    return {
        "supplier": document_create.supplier,
        "product_types": document_create.product_types,
        "materials": document_create.materials,
        "coulissant_galandage": document_create.coulissant_galandage,
        "proferm_gammes": document_create.proferm_gammes,
    }


def validate_classification_payload(
    *,
    supplier: Optional[str],
    product_types: Optional[List[str]],
    materials: Optional[List[str]],
    coulissant_galandage: Optional[str],
    proferm_gammes: Optional[List[str]] = None,
    require_complete: bool = True,
) -> None:
    """Lève ValueError si classification invalide ou incomplète."""
    complete, errors = validate_classification(
        supplier=supplier,
        product_types=product_types,
        materials=materials,
        coulissant_galandage=coulissant_galandage,
        proferm_gammes=proferm_gammes,
    )
    if errors and require_complete:
        raise ValueError("; ".join(errors))
    if require_complete and not complete:
        raise ValueError("Classification incomplète.")


def apply_classification_update(document: Document, document_update: DocumentUpdate) -> None:
    """Fusionne DocumentUpdate classification sur le document."""
    data = document_update.model_dump(exclude_unset=True)
    apply_classification_to_document(
        document,
        supplier=data.get("supplier", document.source),
        product_types=data.get("product_types", document.product_types),
        materials=data.get("materials", document.materials),
        coulissant_galandage=data.get("coulissant_galandage", document.coulissant_galandage),
        proferm_gammes=data.get("proferm_gammes", document.proferm_gammes),
    )
