"""Catalogue centralisé des slots pour le query understanding (labels, enums, ordre de collecte)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# --- Champs obligatoires (ordre de collecte) ---

REQUIRED_FIELDS: Tuple[str, ...] = ("intent", "product_family", "material")

OPTIONAL_FIELDS: Tuple[str, ...] = ("product_range", "supplier")

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
    "profine": "Profine",
    "technal": "Technal",
    "kommerling": "Kommerling",
    "roto": "Roto",
}

FIELD_CHOICES: Dict[str, Dict[str, str]] = {
    "intent": INTENT_CHOICES,
    "product_family": PRODUCT_FAMILY_CHOICES,
    "material": MATERIAL_CHOICES,
    "product_range": PRODUCT_RANGE_CHOICES,
    "supplier": SUPPLIER_CHOICES,
}

FIELD_QUESTIONS: Dict[str, str] = {
    "intent": "Quel type d'information recherchez-vous ?",
    "product_family": "Quelle famille de produit concerne votre question ?",
    "material": "Quel matériau concerne votre question ?",
    "product_range": "Quelle gamme Proferm souhaitez-vous cibler ?",
    "supplier": "Quel fournisseur ou marque souhaitez-vous cibler ?",
}

INTENT_DESCRIPTIONS: Dict[str, str] = {
    "specification": "cotes, dimensions, performances",
    "installation": "pose, montage, réglage",
    "regulatory": "normes (DTU, NF EN), conformité",
    "product_selection": "choix gamme / comparatif",
    "troubleshooting": "dépannage, SAV",
    "documentation": "retrouver une notice / fiche",
}


def empty_slots() -> Dict[str, Optional[str]]:
    return {field: None for field in REQUIRED_FIELDS + OPTIONAL_FIELDS}


def is_valid_slot_value(field: str, value: str) -> bool:
    choices = FIELD_CHOICES.get(field, {})
    return value in choices


def get_label(field: str, value: str) -> str:
    return FIELD_CHOICES.get(field, {}).get(value, value)


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
    return prompt


def supplier_to_primary_source(supplier: Optional[str]) -> Optional[str]:
    mapping = {
        "profine": "Profine",
        "technal": "Technal",
        "kommerling": "Kommerling",
        "roto": "Roto",
    }
    if not supplier:
        return None
    return mapping.get(supplier, supplier.capitalize())
