"""Hiérarchie thématique (carte mentale) : regroupement curé des catégories.

Niveau 1 = familles de thèmes (paquets), niveau 2 = catégories de l'axe ``task``.
La source de vérité vit en code (versionnée, ajustable produit). Les autres axes
(``doc_type``, ``lifecycle_phase``, ``symptom``) restent plats : la carte mentale les
expose en deux niveaux (racine → catégorie) sans familles intermédiaires.

Cf. :mod:`app.services.category_catalog` pour le vocabulaire des catégories.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.services.category_catalog import (
    AXIS_DOC_TYPE,
    AXIS_LIFECYCLE_PHASE,
    AXIS_SYMPTOM,
    AXIS_TASK,
    DOC_TYPE_LABELS,
    LIFECYCLE_PHASE_LABELS,
    SYMPTOM_LABELS,
)

# Famille de thèmes (niveau 1) regroupant des catégories `task` (niveau 2).
# `children` = slugs de l'axe `task` (cf. category_catalog.CONTENT_CATEGORY_SLUGS).
THEME_FAMILIES: Tuple[Dict[str, object], ...] = (
    {
        "slug": "produit_gamme",
        "label": "Produit & gamme",
        "icon": "ti-package",
        "children": ("product_range", "product_comparison", "commercial"),
    },
    {
        "slug": "technique",
        "label": "Caractéristiques techniques",
        "icon": "ti-ruler-2",
        "children": (
            "material_profile",
            "glazing",
            "dimensions_tolerances",
            "load_capacity",
            "parts_references",
        ),
    },
    {
        "slug": "pose",
        "label": "Pose & mise en œuvre",
        "icon": "ti-tools",
        "children": ("mounting", "hardware_adjustment", "sealing", "drilling_constraints"),
    },
    {
        "slug": "normes_garanties",
        "label": "Normes & garanties",
        "icon": "ti-certificate",
        "children": ("regulatory", "certification", "warranty"),
    },
    {
        "slug": "sav",
        "label": "SAV & dépannage",
        "icon": "ti-lifebuoy",
        "children": ("troubleshooting",),
    },
)

# Famille fourre-tout pour les catégories `task` présentes mais non rattachées
# (garantit qu'aucune catégorie ne disparaisse de l'arbre si la taxonomie évolue).
FALLBACK_FAMILY = {
    "slug": "autres",
    "label": "Autres thèmes",
    "icon": "ti-dots",
}

# Axes proposés dans le sélecteur de regroupement de la carte mentale.
# `task` est hiérarchique (familles) ; les autres sont plats.
THEME_AXES: Tuple[Dict[str, str], ...] = (
    {"key": AXIS_TASK, "label": "Thèmes"},
    {"key": AXIS_DOC_TYPE, "label": "Type de document"},
    {"key": AXIS_LIFECYCLE_PHASE, "label": "Phase"},
    {"key": AXIS_SYMPTOM, "label": "Symptômes"},
)

# Map axe plat -> labels connus (pour filtrer les catégories d'un axe donné).
FLAT_AXIS_LABELS: Dict[str, Dict[str, str]] = {
    AXIS_DOC_TYPE: DOC_TYPE_LABELS,
    AXIS_LIFECYCLE_PHASE: LIFECYCLE_PHASE_LABELS,
    AXIS_SYMPTOM: SYMPTOM_LABELS,
}

# Croisement catégorie précise (task) → type(s) d'entité KAG à afficher sous le nœud.
# Les catégories listées ici sont filtrées sur un/des type(s) curé(s) (effet « colonne
# produits » à la NotebookLM, sans bruit). Les catégories NON listées tombent sur un
# fallback « tout type sauf other » (cf. _entity_children_for_category) : aucune
# catégorie non vide ne reste sans entités.
CATEGORY_ENTITY_TYPES: Dict[str, Tuple[str, ...]] = {
    "product_range": ("product",),
    "product_comparison": ("product",),
    "commercial": ("product",),
    "material_profile": ("material",),
    "glazing": ("material",),
    "parts_references": ("reference",),
    "certification": ("norm", "organization"),
    "regulatory": ("norm",),
}

# Icône tabler par type d'entité (pour l'affichage des feuilles entités).
ENTITY_TYPE_ICON: Dict[str, str] = {
    "product": "ti-cube",
    "material": "ti-stack-2",
    "reference": "ti-hash",
    "norm": "ti-shield-check",
    "organization": "ti-building",
    "tool": "ti-tool",
    "process": "ti-route",
    "dimension": "ti-ruler",
    "location": "ti-map-pin",
}


def entity_types_for_category(slug: str) -> Tuple[str, ...]:
    return CATEGORY_ENTITY_TYPES.get(slug, ())


def entity_icon(entity_type: str) -> str:
    return ENTITY_TYPE_ICON.get(entity_type, "ti-point")


_FAMILY_BY_TASK_SLUG: Dict[str, str] = {
    child: family["slug"]  # type: ignore[index]
    for family in THEME_FAMILIES
    for child in family["children"]  # type: ignore[union-attr]
}

_FAMILY_ORDER: Tuple[str, ...] = tuple(f["slug"] for f in THEME_FAMILIES) + (  # type: ignore[misc]
    FALLBACK_FAMILY["slug"],
)


def family_for_task_slug(slug: str) -> str:
    """Retourne le slug de la famille d'une catégorie `task` (fourre-tout sinon)."""
    return _FAMILY_BY_TASK_SLUG.get(slug, FALLBACK_FAMILY["slug"])


def family_meta(family_slug: str) -> Dict[str, object]:
    """Métadonnées (label, icon) d'une famille."""
    for family in THEME_FAMILIES:
        if family["slug"] == family_slug:
            return family
    return FALLBACK_FAMILY


def family_order_index(family_slug: str) -> int:
    """Index d'ordre d'affichage d'une famille."""
    try:
        return _FAMILY_ORDER.index(family_slug)
    except ValueError:
        return len(_FAMILY_ORDER)


def is_valid_axis(axis: Optional[str]) -> bool:
    return axis in {a["key"] for a in THEME_AXES}


def axes_for_payload() -> List[Dict[str, str]]:
    return [dict(a) for a in THEME_AXES]
