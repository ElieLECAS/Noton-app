"""Catalogue centralisé des catégories de contenu documentaire."""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from sqlmodel import Session, select

from app.models.document_category import DocumentCategory

CONTENT_CATEGORY_SLUGS: Tuple[str, ...] = (
    "mounting",
    "hardware_adjustment",
    "sealing",
    "drilling_constraints",
    "dimensions_tolerances",
    "load_capacity",
    "material_profile",
    "glazing",
    "parts_references",
    "product_range",
    "regulatory",
    "warranty",
    "certification",
    "commercial",
    "product_comparison",
    "troubleshooting",
)

# Fallback statique si BDD vide (tests, bootstrap)
DEFAULT_CATEGORY_LABELS: Dict[str, str] = {
    "mounting": "Pose / montage",
    "hardware_adjustment": "Réglage quincaillerie",
    "sealing": "Étanchéité / calfeutrement",
    "drilling_constraints": "Perçage / interdictions",
    "dimensions_tolerances": "Cotes / tolérances",
    "load_capacity": "Charge / limites structurelles",
    "material_profile": "Matériau / profilé",
    "glazing": "Vitrage / performances",
    "parts_references": "Références pièces / codes",
    "product_range": "Gamme / produit",
    "regulatory": "Normes / conformité",
    "warranty": "Garanties",
    "certification": "Certifications / marquage",
    "commercial": "Contenu commercial",
    "product_comparison": "Comparatif / aide au choix",
    "troubleshooting": "Dépannage / diagnostic",
}

DEFAULT_CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "mounting": "Séquences de pose, assemblage, fixation, calage, pattes, gondage",
    "hardware_adjustment": "Roulettes, gâches, réglages de manœuvre, alignement ouvrant",
    "sealing": "Bavettes, joints, remontées, infiltration, étanchéité air/eau",
    "drilling_constraints": "Ce qu'on peut/ne peut pas percer ; formulations « interdit »",
    "dimensions_tolerances": "Faux aplomb, mm/m, cotes chiffrées, tolérances de pose",
    "load_capacity": "Poids max, report de charge, entraxe pattes, limites vantail",
    "material_profile": "PVC, alu, hybride, coupe de profil, composition matériau",
    "glazing": "Vitrages, thermique, acoustique, spécifications de vitrage",
    "parts_references": "Codes Txxx, SEC-xxx, nomenclature, liste de composants",
    "product_range": "Identification SOLEAL, LUMEAL, Perform, variante, famille",
    "regulatory": "DTU, NF EN, obligations normatives, PMR (exigences réglementaires)",
    "warranty": "Durée, conditions de garantie, exclusions, couverture",
    "certification": "CE, labels, attestations, conformité produit",
    "commercial": "Dépliants, arguments design/performance, contenu marketing",
    "product_comparison": "LUMEAL vs SOLEAL, différences produit, aide à la décision",
    "troubleshooting": "Symptômes client, causes probables, SAV, diagnostic",
}

INTENT_TO_CATEGORIES: Dict[str, List[str]] = {
    "installation": [
        "mounting",
        "hardware_adjustment",
        "sealing",
        "drilling_constraints",
    ],
    "specification": [
        "dimensions_tolerances",
        "load_capacity",
        "parts_references",
        "material_profile",
        "glazing",
    ],
    "regulatory": ["regulatory", "warranty", "certification"],
    "product_selection": ["product_comparison", "commercial", "product_range"],
    "troubleshooting": ["troubleshooting", "hardware_adjustment", "sealing"],
    "documentation": [],
}


def get_active_categories(session: Session) -> List[DocumentCategory]:
    """Retourne les catégories actives triées par slug."""
    stmt = (
        select(DocumentCategory)
        .where(DocumentCategory.is_active == True)  # noqa: E712
        .order_by(DocumentCategory.slug)
    )
    return list(session.exec(stmt).all())


def get_category_id_by_slug(session: Session, *, active_only: bool = True) -> Dict[str, int]:
    """Map slug → id pour persistance."""
    stmt = select(DocumentCategory)
    if active_only:
        stmt = stmt.where(DocumentCategory.is_active == True)  # noqa: E712
    rows = session.exec(stmt).all()
    return {row.slug: row.id for row in rows}


def is_valid_category_slug(slug: str, session: Optional[Session] = None) -> bool:
    """Valide un slug contre la BDD ou le catalogue statique."""
    if session is not None:
        row = session.exec(
            select(DocumentCategory).where(
                DocumentCategory.slug == slug,
                DocumentCategory.is_active == True,  # noqa: E712
            )
        ).first()
        return row is not None
    return slug in CONTENT_CATEGORY_SLUGS


def get_active_categories_for_prompt(session: Session) -> str:
    """JSON slug+description pour le prompt LLM d'extraction."""
    categories = get_active_categories(session)
    if not categories:
        payload = [
            {"slug": slug, "description": DEFAULT_CATEGORY_DESCRIPTIONS.get(slug, "")}
            for slug in CONTENT_CATEGORY_SLUGS
        ]
    else:
        payload = [{"slug": c.slug, "description": c.description} for c in categories]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def get_category_choices_for_slot(session: Optional[Session] = None) -> Dict[str, str]:
    """Retourne slug → label pour le slot filling UI."""
    if session is not None:
        rows = get_active_categories(session)
        if rows:
            return {row.slug: row.label for row in rows}
    return dict(DEFAULT_CATEGORY_LABELS)


def suggested_categories_for_intent(intent: Optional[str]) -> List[str]:
    """Catégories suggérées selon l'intent chat."""
    if not intent:
        return []
    return list(INTENT_TO_CATEGORIES.get(intent.strip().lower(), []))
