"""Catalogue centralisé des slots pour le query understanding (labels, enums, ordre de collecte)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.services.category_catalog import DEFAULT_CATEGORY_LABELS, suggested_categories_for_intent

# --- Champs obligatoires (ordre de collecte) ---

REQUIRED_FIELDS: Tuple[str, ...] = ("intent", "product_family", "material")

OPTIONAL_FIELDS: Tuple[str, ...] = ("product_range", "supplier", "content_categories")

DOCUMENT_CLASSIFICATION_FIELDS: Tuple[str, ...] = (
    "product_types",
    "materials",
    "source",
    "proferm_gammes",
)

INTENT_CHOICES: Dict[str, str] = {
    "specification": "Spécifications / dimensions",
    "installation": "Pose / montage / réglage",
    "regulatory": "Normes / conformité",
    "product_selection": "Choix de gamme / comparatif",
    "troubleshooting": "Dépannage / SAV",
    "documentation": "Retrouver une notice / fiche",
}

PRODUCT_FAMILY_CHOICES: Dict[str, str] = {
    "fenetres": "Fenêtres",
    "portes": "Portes",
    "coulissants": "Coulissants",
}

MATERIAL_CHOICES: Dict[str, str] = {
    "pvc": "PVC",
    "aluminium": "Aluminium",
    "hybride": "Matériau hybride",
}

PRODUCT_RANGE_CHOICES: Dict[str, str] = {
    "perform": "Perform",
    "lumine": "Lumine",
    "hybride": "Gamme Hybride Proferm",
    "textural": "Textural",
}

SUPPLIER_CHOICES: Dict[str, str] = {
    "kommerling": "Kommerling",
    "profine": "Profine",
    "askey": "Askey",
    "roto": "ROTO",
    "technal": "Technal",
    "soprofen": "Soprofen",
    "proferm": "Proferm",
}

FIELD_CHOICES: Dict[str, Dict[str, str]] = {
    "intent": INTENT_CHOICES,
    "product_family": PRODUCT_FAMILY_CHOICES,
    "material": MATERIAL_CHOICES,
    "product_range": PRODUCT_RANGE_CHOICES,
    "supplier": SUPPLIER_CHOICES,
    "content_categories": DEFAULT_CATEGORY_LABELS,
}

DOCUMENT_FIELD_CHOICES: Dict[str, Dict[str, str]] = {
    "product_types": PRODUCT_FAMILY_CHOICES,
    "materials": MATERIAL_CHOICES,
    "proferm_gammes": PRODUCT_RANGE_CHOICES,
    "source": SUPPLIER_CHOICES,
}

FIELD_QUESTIONS: Dict[str, str] = {
    "intent": "Quel type d'information recherchez-vous ?",
    "product_family": "Quelle famille de produit concerne votre question ?",
    "material": "Quel matériau concerne votre question ?",
    "product_range": "Quelle gamme Proferm souhaitez-vous cibler ?",
    "supplier": "Quel fournisseur ou marque souhaitez-vous cibler ?",
    "content_categories": "Quel type de contenu recherchez-vous ?",
}

INTENT_DESCRIPTIONS: Dict[str, str] = {
    "specification": "cotes, dimensions, performances",
    "installation": "pose, montage, réglage",
    "regulatory": "normes (DTU, NF EN), conformité",
    "product_selection": "choix gamme / comparatif",
    "troubleshooting": "dépannage, SAV",
    "documentation": "retrouver une notice / fiche",
}

# Type de CONTENU attendu par intention (B1, plan boucle agentique 2026-07-29) : sert au
# juge de suffisance pré-génération. La leçon du cas TGY3702/3704 : une page qui mentionne
# la référence sans porter le TYPE d'information demandé (nomenclature quand on demande une
# pose) doit être jugée insuffisante, quel que soit son score de similarité.
INTENT_EXPECTED_CONTENT: Dict[str, str] = {
    "specification": (
        "des valeurs techniques précises (fiche technique, tableau de caractéristiques, cotes)"
    ),
    "installation": (
        "des étapes de pose/montage/réglage numérotées ou des schémas de montage "
        "(notice de pose, catalogue de fabrication) — un tableau de composition ou une "
        "nomenclature qui cite la référence NE suffit PAS"
    ),
    "regulatory": (
        "des exigences normatives explicites (DTU, NF EN, PV d'essai, conditions de garantie)"
    ),
    "product_selection": (
        "des références, tableaux de composition ou nomenclatures (catalogue de conception)"
    ),
    "troubleshooting": (
        "un diagnostic ou une procédure de réglage/SAV (guide SAV, notice) — pas une simple "
        "fiche produit"
    ),
    "documentation": "la notice ou fiche demandée, identifiable par son titre",
}

_EXPECTED_CONTENT_DEFAULT = "toute page qui contient LITTÉRALEMENT l'information demandée"


def expected_content_for_intent(intent: Optional[str]) -> str:
    """Description du type de contenu qui peut répondre, pour le prompt du juge (B1)."""
    return INTENT_EXPECTED_CONTENT.get((intent or "").strip().lower(), _EXPECTED_CONTENT_DEFAULT)


UNKNOWN_OPTIONAL_PHRASES: Tuple[str, ...] = (
    "je ne sais pas",
    "je sais pas",
    "aucune idée",
    "pas sûr",
    "pas sur",
    "je ne sais",
)

SLOT_TO_DOCUMENT_FIELD: Dict[str, str] = {
    "product_family": "product_types",
    "material": "materials",
    "product_range": "proferm_gammes",
    "supplier": "source",
}


@dataclass
class ClassificationFilters:
    """Filtres actifs dérivés des slots chat pour le pré-filtre SQL document."""

    product_family: Optional[str] = None
    material: Optional[str] = None
    product_range: Optional[str] = None
    supplier_source: Optional[str] = None
    content_categories: Optional[List[str]] = None

    def has_any(self) -> bool:
        return any(
            v is not None
            for v in (
                self.product_family,
                self.material,
                self.product_range,
                self.supplier_source,
                self.content_categories,
            )
        )


def empty_slots() -> Dict[str, Optional[str]]:
    return {field: None for field in REQUIRED_FIELDS + OPTIONAL_FIELDS}


def parse_content_categories(value: Optional[str]) -> List[str]:
    """Parse une valeur slot content_categories (csv ou slug unique)."""
    if not value or not str(value).strip():
        return []
    raw = str(value).strip()
    if raw.startswith("["):
        import json

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(v).strip().lower() for v in parsed if str(v).strip()]
        except json.JSONDecodeError:
            pass
    parts = [p.strip().lower() for p in raw.replace(";", ",").split(",") if p.strip()]
    return list(dict.fromkeys(parts))


def serialize_content_categories(slugs: List[str]) -> Optional[str]:
    cleaned = list(dict.fromkeys(s.strip().lower() for s in slugs if s and str(s).strip()))
    return ",".join(cleaned) if cleaned else None


def is_valid_content_categories_value(value: str) -> bool:
    slugs = parse_content_categories(value)
    if not slugs:
        return False
    return all(slug in DEFAULT_CATEGORY_LABELS for slug in slugs)


def is_valid_slot_value(field: str, value: str) -> bool:
    if field == "content_categories":
        return is_valid_content_categories_value(value)
    choices = FIELD_CHOICES.get(field, {})
    return value in choices


def is_valid_document_slug(field: str, value: str) -> bool:
    choices = DOCUMENT_FIELD_CHOICES.get(field, {})
    return value in choices


def get_label(field: str, value: str) -> str:
    return FIELD_CHOICES.get(field, {}).get(value, value)


def supplier_slug_to_source(slug: Optional[str]) -> Optional[str]:
    if not slug:
        return None
    return SUPPLIER_CHOICES.get(slug.strip().lower(), slug.strip().capitalize())


def supplier_to_primary_source(supplier: Optional[str]) -> Optional[str]:
    return supplier_slug_to_source(supplier)


def next_missing_required(slots: Dict[str, Optional[str]]) -> Optional[str]:
    for field in REQUIRED_FIELDS:
        if not slots.get(field):
            return field
    return None


def next_missing_optional(
    slots: Dict[str, Optional[str]],
    skipped_optional: List[str],
) -> Optional[str]:
    for field in OPTIONAL_FIELDS:
        if field in skipped_optional:
            continue
        if not slots.get(field):
            return field
    return None


def is_unknown_optional_response(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not normalized:
        return False
    return any(phrase in normalized for phrase in UNKNOWN_OPTIONAL_PHRASES)


def detect_optional_skip_from_message(
    user_message: str,
    pending_field: Optional[str],
) -> Optional[str]:
    """Retourne le champ optionnel à skipper si l'utilisateur dit « je ne sais pas »."""
    if not pending_field or pending_field not in OPTIONAL_FIELDS:
        return None
    if is_unknown_optional_response(user_message):
        return pending_field
    return None


def build_slot_prompt(
    field: str,
    *,
    phase: str,
    allow_skip: bool = False,
) -> Dict[str, Any]:
    choices_map = FIELD_CHOICES.get(field, {})
    choices = [
        {"label": label, "field": field, "value": value}
        for value, label in choices_map.items()
    ]
    prompt: Dict[str, Any] = {
        "phase": phase,
        "field": field,
        "question": FIELD_QUESTIONS.get(field, f"Précisez : {field}"),
        "choices": choices,
        "allow_skip": allow_skip,
    }
    if allow_skip:
        prompt["skip_label"] = "Passer"
        prompt["unknown_label"] = "Je ne sais pas"
    return prompt


def serialize_classification_options() -> Dict[str, Any]:
    """Payload API pour alimenter les modales bibliothèque et le chat."""
    return {
        "product_types": [
            {"value": k, "label": v} for k, v in PRODUCT_FAMILY_CHOICES.items()
        ],
        "materials": [
            {"value": k, "label": v} for k, v in MATERIAL_CHOICES.items()
        ],
        "proferm_gammes": [
            {"value": k, "label": v} for k, v in PRODUCT_RANGE_CHOICES.items()
        ],
        "suppliers": [
            {"value": k, "label": v} for k, v in SUPPLIER_CHOICES.items()
        ],
        "slots": {
            field: [{"value": k, "label": v} for k, v in choices.items()]
            for field, choices in FIELD_CHOICES.items()
        },
        "content_categories": [
            {"value": k, "label": v} for k, v in DEFAULT_CATEGORY_LABELS.items()
        ],
    }


def validate_document_classification(
    *,
    product_types: Optional[List[str]] = None,
    materials: Optional[List[str]] = None,
    source: Optional[str] = None,
    proferm_gammes: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    pt = product_types or []
    mat = materials or []
    pg = proferm_gammes or []

    if not pt:
        errors.append("Au moins une famille de produit est requise.")
    else:
        for v in pt:
            if not is_valid_document_slug("product_types", v):
                errors.append(f"Famille produit invalide : {v}")

    if not mat:
        errors.append("Au moins un matériau est requis.")
    else:
        for v in mat:
            if not is_valid_document_slug("materials", v):
                errors.append(f"Matériau invalide : {v}")

    if not source or not str(source).strip():
        errors.append("Le fournisseur est requis.")
    elif not is_valid_document_slug("source", str(source).strip().lower()):
        errors.append(f"Fournisseur invalide : {source}")

    if not pg:
        errors.append("Au moins une gamme Proferm est requise.")
    else:
        for v in pg:
            if not is_valid_document_slug("proferm_gammes", v):
                errors.append(f"Gamme Proferm invalide : {v}")

    return len(errors) == 0, errors


def compute_classification_status(
    *,
    product_types: Optional[List[str]] = None,
    materials: Optional[List[str]] = None,
    source: Optional[str] = None,
    proferm_gammes: Optional[List[str]] = None,
) -> str:
    ok, _ = validate_document_classification(
        product_types=product_types,
        materials=materials,
        source=source,
        proferm_gammes=proferm_gammes,
    )
    return "complete" if ok else "incomplete"


def normalize_document_classification(
    *,
    product_types: Optional[List[str]] = None,
    materials: Optional[List[str]] = None,
    source: Optional[str] = None,
    proferm_gammes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Normalise les slugs et mappe le fournisseur vers la forme canonique."""
    src_slug = (source or "").strip().lower() if source else None
    return {
        "product_types": list(dict.fromkeys(product_types or [])),
        "materials": list(dict.fromkeys(materials or [])),
        "proferm_gammes": list(dict.fromkeys(proferm_gammes or [])),
        "source": supplier_slug_to_source(src_slug) if src_slug else None,
    }


def build_classification_filters(
    slots: Dict[str, Optional[str]],
    skipped_optional: Optional[List[str]] = None,
) -> ClassificationFilters:
    """Construit les filtres SQL actifs à partir des slots chat collectés."""
    skipped = set(skipped_optional or [])
    filters = ClassificationFilters()

    if slots.get("product_family"):
        filters.product_family = slots["product_family"]

    if slots.get("material"):
        filters.material = slots["material"]

    if "product_range" not in skipped and slots.get("product_range"):
        filters.product_range = slots["product_range"]

    if "supplier" not in skipped and slots.get("supplier"):
        filters.supplier_source = supplier_slug_to_source(slots["supplier"])

    if "content_categories" not in skipped and slots.get("content_categories"):
        parsed = parse_content_categories(slots["content_categories"])
        if parsed:
            filters.content_categories = parsed

    return filters
