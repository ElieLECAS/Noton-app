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

# ---------------------------------------------------------------------------
# Taxonomie multi-axes (facettes)
# ---------------------------------------------------------------------------
# Les 16 slugs ci-dessus appartiennent à l'axe `task` (DEFAULT_CATEGORY_LABELS).
# On ajoute trois axes orthogonaux, chacun avec son propre vocabulaire fermé.
# `DEFAULT_CATEGORY_LABELS` reste volontairement l'axe `task` SEUL : c'est lui qui
# alimente le slot UI `content_categories` (slot_catalog) — pas de pollution.

AXIS_TASK = "task"
AXIS_DOC_TYPE = "doc_type"
AXIS_LIFECYCLE_PHASE = "lifecycle_phase"
AXIS_SYMPTOM = "symptom"
CONTENT_AXES: Tuple[str, ...] = (AXIS_TASK, AXIS_DOC_TYPE, AXIS_LIFECYCLE_PHASE, AXIS_SYMPTOM)

DOC_TYPE_LABELS: Dict[str, str] = {
    "notice_pose": "Notice de pose",
    "fiche_technique": "Fiche technique",
    "doc_commerciale": "Document commercial",
    "pv_certification": "PV / certification",
    "conditions_garantie": "Conditions de garantie",
    "guide_sav": "Guide SAV",
    "nomenclature": "Nomenclature / pièces",
    "dtu_norme": "DTU / norme",
}
DOC_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "notice_pose": "Procédure de pose pas à pas, séquence de montage sur chantier",
    "fiche_technique": "Caractéristiques techniques, cotes, performances produit",
    "doc_commerciale": "Dépliant, argumentaire, brochure marketing",
    "pv_certification": "Procès-verbal d'essai, certificat, marquage CE, attestation",
    "conditions_garantie": "Durée, conditions, exclusions de garantie",
    "guide_sav": "Guide de dépannage, diagnostic, intervention après-vente",
    "nomenclature": "Liste de composants, références pièces détachées, éclatés",
    "dtu_norme": "Document normatif, DTU, NF EN, réglementation",
}

LIFECYCLE_PHASE_LABELS: Dict[str, str] = {
    "avant_vente": "Avant-vente",
    "chantier_pose": "Chantier / pose",
    "apres_vente_sav": "Après-vente / SAV",
}
LIFECYCLE_PHASE_DESCRIPTIONS: Dict[str, str] = {
    "avant_vente": "Choix produit, devis, conseil avant achat",
    "chantier_pose": "Phase de pose et mise en œuvre sur chantier",
    "apres_vente_sav": "Usage, maintenance, dépannage après installation",
}

SYMPTOM_LABELS: Dict[str, str] = {
    "infiltration_eau": "Infiltration d'eau",
    "condensation": "Condensation",
    "blocage_manoeuvre": "Blocage de manœuvre",
    "deformation": "Déformation",
    "defaut_etancheite_air": "Défaut d'étanchéité à l'air",
    "casse_quincaillerie": "Casse quincaillerie",
    "bruit": "Bruit",
    "desalignement_ouvrant": "Désalignement d'ouvrant",
}
SYMPTOM_DESCRIPTIONS: Dict[str, str] = {
    "infiltration_eau": "Entrée d'eau, fuite, défaut d'étanchéité à l'eau",
    "condensation": "Buée, condensation sur vitrage ou profilé",
    "blocage_manoeuvre": "Ouvrant dur, bloqué, manœuvre difficile",
    "deformation": "Profilé déformé, voilé, gauchi",
    "defaut_etancheite_air": "Courant d'air, sifflement, perméabilité à l'air",
    "casse_quincaillerie": "Pièce cassée : gâche, roulette, charnière défaillante",
    "bruit": "Grincement, claquement, bruit de manœuvre ou au vent",
    "desalignement_ouvrant": "Ouvrant désaligné, frottement, mauvais affleurement",
}

# Vocabulaire statique par axe (fallback si BDD vide) : axis -> {slug: description}
DEFAULT_DESCRIPTIONS_BY_AXIS: Dict[str, Dict[str, str]] = {
    AXIS_TASK: DEFAULT_CATEGORY_DESCRIPTIONS,
    AXIS_DOC_TYPE: DOC_TYPE_DESCRIPTIONS,
    AXIS_LIFECYCLE_PHASE: LIFECYCLE_PHASE_DESCRIPTIONS,
    AXIS_SYMPTOM: SYMPTOM_DESCRIPTIONS,
}
DEFAULT_LABELS_BY_AXIS: Dict[str, Dict[str, str]] = {
    AXIS_TASK: DEFAULT_CATEGORY_LABELS,
    AXIS_DOC_TYPE: DOC_TYPE_LABELS,
    AXIS_LIFECYCLE_PHASE: LIFECYCLE_PHASE_LABELS,
    AXIS_SYMPTOM: SYMPTOM_LABELS,
}

INTENT_TO_CATEGORIES: Dict[str, List[str]] = {
    "installation": [
        "mounting",
        "hardware_adjustment",
        "sealing",
        "drilling_constraints",
        "notice_pose",
        "chantier_pose",
    ],
    "specification": [
        "dimensions_tolerances",
        "load_capacity",
        "parts_references",
        "material_profile",
        "glazing",
        "fiche_technique",
    ],
    "regulatory": ["regulatory", "warranty", "certification", "dtu_norme", "pv_certification"],
    "product_selection": [
        "product_comparison",
        "commercial",
        "product_range",
        "doc_commerciale",
        "avant_vente",
    ],
    "troubleshooting": [
        "troubleshooting",
        "hardware_adjustment",
        "sealing",
        "guide_sav",
        "apres_vente_sav",
    ],
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
    """JSON slug+description (axe `task` uniquement) pour le prompt LLM d'extraction.

    Restreint à l'axe `task` : les axes doc_type/lifecycle_phase/symptom sont exposés
    séparément via :func:`get_categories_for_prompt_grouped_by_axis`.
    """
    categories = get_active_categories_by_axis(session, AXIS_TASK)
    if not categories:
        payload = [
            {"slug": slug, "description": DEFAULT_CATEGORY_DESCRIPTIONS.get(slug, "")}
            for slug in CONTENT_CATEGORY_SLUGS
        ]
    else:
        payload = [{"slug": c.slug, "description": c.description} for c in categories]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def get_category_choices_for_slot(session: Optional[Session] = None) -> Dict[str, str]:
    """Retourne slug → label pour le slot filling UI (axe `task` uniquement)."""
    if session is not None:
        rows = get_active_categories_by_axis(session, AXIS_TASK)
        if rows:
            return {row.slug: row.label for row in rows}
    return dict(DEFAULT_CATEGORY_LABELS)


def suggested_categories_for_intent(intent: Optional[str]) -> List[str]:
    """Catégories suggérées selon l'intent chat."""
    if not intent:
        return []
    return list(INTENT_TO_CATEGORIES.get(intent.strip().lower(), []))


# ---------------------------------------------------------------------------
# Accès par axe (taxonomie multi-facettes)
# ---------------------------------------------------------------------------


def get_active_categories_by_axis(session: Session, axis: str) -> List[DocumentCategory]:
    """Retourne les catégories actives d'un axe donné, triées par slug."""
    stmt = (
        select(DocumentCategory)
        .where(
            DocumentCategory.is_active == True,  # noqa: E712
            DocumentCategory.axis == axis,
        )
        .order_by(DocumentCategory.slug)
    )
    return list(session.exec(stmt).all())


def get_allowed_slugs_by_axis(
    session: Optional[Session] = None,
    axes: Optional[Tuple[str, ...]] = None,
) -> Dict[str, frozenset]:
    """Map axis -> frozenset(slugs actifs). Fallback statique si BDD vide/absente."""
    wanted = axes or CONTENT_AXES
    if session is not None:
        rows = session.exec(
            select(DocumentCategory).where(DocumentCategory.is_active == True)  # noqa: E712
        ).all()
        result: Dict[str, set] = {axis: set() for axis in wanted}
        for row in rows:
            if row.axis in result:
                result[row.axis].add(row.slug)
        # Si un axe n'a aucune ligne en base, retomber sur le vocabulaire statique
        out: Dict[str, frozenset] = {}
        for axis in wanted:
            slugs = result.get(axis) or set()
            if not slugs:
                slugs = set(DEFAULT_DESCRIPTIONS_BY_AXIS.get(axis, {}).keys())
            out[axis] = frozenset(slugs)
        return out
    return {
        axis: frozenset(DEFAULT_DESCRIPTIONS_BY_AXIS.get(axis, {}).keys())
        for axis in wanted
    }


def get_categories_for_prompt_grouped_by_axis(
    session: Session,
    axes: Optional[Tuple[str, ...]] = None,
) -> str:
    """JSON {axis: [{slug, description}]} pour les prompts LLM (extraction / requête).

    Présente le vocabulaire fermé groupé par axe afin que le LLM choisisse PAR AXE
    (formater, jamais créer). Fallback statique par axe si la BDD est vide.
    """
    wanted = axes or CONTENT_AXES
    payload: Dict[str, List[Dict[str, str]]] = {}
    for axis in wanted:
        rows = get_active_categories_by_axis(session, axis)
        if rows:
            payload[axis] = [{"slug": c.slug, "description": c.description} for c in rows]
        else:
            descriptions = DEFAULT_DESCRIPTIONS_BY_AXIS.get(axis, {})
            payload[axis] = [
                {"slug": slug, "description": desc} for slug, desc in descriptions.items()
            ]
    return json.dumps(payload, ensure_ascii=False, indent=2)
