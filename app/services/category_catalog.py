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
    "mounting": "Séquence de pose et fixation de la menuiserie sur chantier : calage, pattes, vissage, mise en place du dormant. À utiliser pour le GESTE de pose. NE PAS confondre avec hardware_adjustment (réglage après pose) ni sealing (étanchéité).",
    "hardware_adjustment": "Réglage et entretien de la quincaillerie : roulettes, gâches, compas, paumelles, alignement de l'ouvrant. À utiliser pour AJUSTER un organe mécanique. NE PAS confondre avec mounting (pose initiale) ni casse_quincaillerie (pièce cassée, axe symptom).",
    "sealing": "Étanchéité air/eau : joints, bavettes, fonds de joint, remontées, calfeutrement périphérique. À utiliser pour ce qui REND ÉTANCHE. NE PAS confondre avec infiltration_eau (le symptôme constaté, axe symptom).",
    "drilling_constraints": "Règles et interdictions de perçage : où l'on peut/ne peut pas percer, fixations autorisées, « interdit de percer ». À utiliser pour une CONTRAINTE de perçage. NE PAS confondre avec mounting (pose générale).",
    "dimensions_tolerances": "Cotes chiffrées, tolérances, faux aplomb, mm/m, jeux admissibles. À utiliser quand des VALEURS dimensionnelles encadrent la pose ou le produit. NE PAS confondre avec load_capacity (limites de charge).",
    "load_capacity": "Limites structurelles : poids max d'un vantail, report de charge, entraxe des pattes, capacité portante. À utiliser pour une LIMITE mécanique. NE PAS confondre avec dimensions_tolerances (cotes).",
    "material_profile": "Nature et composition du matériau/profilé : PVC, aluminium, bois, hybride, coupe de profil, traitements. À utiliser pour la MATIÈRE. NE PAS confondre avec glazing (vitrage) ni product_range (gamme commerciale).",
    "glazing": "Vitrage et ses performances directes : double/triple vitrage, Ug, intercalaire, acoustique du vitrage. À utiliser pour le VERRE. NE PAS confondre avec material_profile (le profilé).",
    "parts_references": "Codes et références de pièces détachées, nomenclatures, éclatés (Txxx, SEC-xxx). À utiliser quand le contenu LISTE des références. NE PAS confondre avec product_range (la gamme).",
    "product_range": "Identification d'une gamme/produit commercial (SOLEAL, LUMEAL, Perform, Lumine…), variantes, familles. À utiliser pour NOMMER le produit. NE PAS confondre avec product_comparison (choix entre gammes) ni commercial (argumentaire).",
    "regulatory": "Exigences réglementaires/normatives : DTU, NF EN, PMR, obligations de mise en œuvre. À utiliser pour une OBLIGATION normative. NE PAS confondre avec certification (preuve/marquage d'un produit).",
    "warranty": "Conditions, durées et exclusions de garantie. À utiliser pour la GARANTIE contractuelle. NE PAS confondre avec certification (conformité produit).",
    "certification": "Preuves de conformité d'un produit : marquage CE, PV d'essai, attestations, labels. À utiliser pour une PREUVE/marquage. NE PAS confondre avec regulatory (l'exigence) ni warranty (la garantie).",
    "commercial": "Contenu marketing : argumentaires, dépliants, mises en avant design/performance à but commercial. À utiliser pour du MARKETING. NE PAS confondre avec product_range (identification) ni product_comparison (aide au choix factuelle).",
    "product_comparison": "Comparaison factuelle entre gammes/options pour aider au choix (LUMEAL vs SOLEAL, différences). À utiliser pour COMPARER. NE PAS confondre avec commercial (argumentaire) ni product_range (simple identification).",
    "troubleshooting": "Diagnostic SAV : symptômes client, causes probables, démarche de dépannage. À utiliser pour un RAISONNEMENT de panne. Les symptômes précis vont AUSSI sur l'axe symptom (infiltration_eau, blocage_manoeuvre…).",
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
