"""
Taxonomie normalisée pour la classification des documents bibliothèque.

Utilisée par l'UI upload/édition, la validation API et le filtre retriever RAG.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Literal, Optional, Tuple

ProductType = Literal["fenetre", "porte", "coulissant"]
MaterialType = Literal["pvc", "alu", "hybride"]
CoulissantGalandage = Literal["oui", "non", "both"]
ProfermGamme = Literal["perform", "lumine", "hybride", "textural"]
ClassificationStatus = Literal["complete", "incomplete"]

SUPPLIERS: List[str] = [
    "Proferm",
    "Technal",
    "Askey",
    "Profine",
    "Roto",
    "Somfy",
    "Maco",
    "Gu",
    "VBH",
    "KBE",
]

PRODUCT_TYPES: List[ProductType] = ["fenetre", "porte", "coulissant"]
MATERIALS: List[MaterialType] = ["pvc", "alu", "hybride"]
COULISSANT_GALANDAGE_OPTIONS: List[CoulissantGalandage] = ["oui", "non", "both"]
PROFERM_GAMMES: List[ProfermGamme] = ["perform", "lumine", "hybride", "textural"]

# Profils fournisseur pour règles génériques (filtre SQL, validation slots)
SUPPLIER_PROFILES: Dict[str, Dict[str, bool]] = {
    "Roto": {"multi_material_hardware": True},
    "Proferm": {"uses_proferm_gammes": True},
    "Profine": {"uses_proferm_gammes": True},
}

LABELS: Dict[str, Dict[str, str]] = {
    "product_types": {
        "fenetre": "Fenêtre",
        "porte": "Porte",
        "coulissant": "Coulissant",
    },
    "materials": {
        "pvc": "PVC",
        "alu": "Aluminium",
        "hybride": "Hybride",
    },
    "coulissant_galandage": {
        "oui": "Galandage",
        "non": "2 rails",
        "both": "Les deux",
    },
    "proferm_gammes": {
        "perform": "Perform",
        "lumine": "Lumine",
        "hybride": "Hybride",
        "textural": "Textural",
    },
}

# Alias utilisateur / slot filling → gamme Proferm (filtre document, pas requête ColPali)
PROFERM_GAMME_ALIASES: Dict[str, List[str]] = {
    "perform": ["perform"],
    "lumine": ["lumine", "lumeal", "soleal"],
    "hybride": ["hybride", "hybrid"],
    "textural": ["textural", "texture"],
}

# Slot filling material "mixte" → taxonomie document "hybride"
SLOT_MATERIAL_TO_DOCUMENT: Dict[str, str] = {
    "mixte": "hybride",
    "bois": "hybride",  # pas de valeur bois en doc ; conservateur
}


def normalize_token(value: str) -> str:
    lowered = value.lower().strip()
    return unicodedata.normalize("NFD", lowered).encode("ascii", "ignore").decode("ascii")


def slot_material_to_document(material: Optional[str]) -> Optional[str]:
    if not material or material == "inconnu":
        return None
    return SLOT_MATERIAL_TO_DOCUMENT.get(material, material)


def resolve_proferm_gammes_from_text(text: Optional[str]) -> List[str]:
    """Résout les gammes Proferm mentionnées dans un texte utilisateur."""
    if not text or not text.strip():
        return []
    norm = normalize_token(text)
    matched: List[str] = []
    for gamme, aliases in PROFERM_GAMME_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(normalize_token(alias))}\b", norm):
                if gamme not in matched:
                    matched.append(gamme)
                break
    return matched


def supplier_uses_proferm_gammes(supplier: Optional[str]) -> bool:
    if not supplier:
        return False
    return SUPPLIER_PROFILES.get(supplier, {}).get("uses_proferm_gammes", False)


def supplier_skips_material_requirement(
    supplier: Optional[str],
    product_type: Optional[str],
) -> bool:
    if not supplier or product_type != "porte":
        return False
    return SUPPLIER_PROFILES.get(supplier, {}).get("multi_material_hardware", False)


def validate_classification(
    *,
    supplier: Optional[str],
    product_types: Optional[List[str]],
    materials: Optional[List[str]],
    coulissant_galandage: Optional[str],
    proferm_gammes: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Valide une classification document.
    Retourne (is_complete, errors).
    """
    errors: List[str] = []
    pts = [p for p in (product_types or []) if p]
    mats = [m for m in (materials or []) if m]
    gammes = [g for g in (proferm_gammes or []) if g]

    if not supplier or not str(supplier).strip():
        errors.append("Le fournisseur est obligatoire.")
    elif supplier not in SUPPLIERS:
        errors.append(f"Fournisseur invalide : {supplier}")

    if not pts:
        errors.append("Au moins un type de produit est requis.")
    else:
        invalid_pt = [p for p in pts if p not in PRODUCT_TYPES]
        if invalid_pt:
            errors.append(f"Types produit invalides : {', '.join(invalid_pt)}")

    if not mats:
        errors.append("Au moins un matériau est requis.")
    else:
        invalid_mat = [m for m in mats if m not in MATERIALS]
        if invalid_mat:
            errors.append(f"Matériaux invalides : {', '.join(invalid_mat)}")

    if "coulissant" in pts:
        if not coulissant_galandage:
            errors.append("Le galandage est obligatoire pour un document coulissant.")
        elif coulissant_galandage not in COULISSANT_GALANDAGE_OPTIONS:
            errors.append(f"Valeur galandage invalide : {coulissant_galandage}")

    if gammes:
        invalid_g = [g for g in gammes if g not in PROFERM_GAMMES]
        if invalid_g:
            errors.append(f"Gammes Proferm invalides : {', '.join(invalid_g)}")

    return len(errors) == 0, errors


def compute_classification_status(
    *,
    supplier: Optional[str],
    product_types: Optional[List[str]],
    materials: Optional[List[str]],
    coulissant_galandage: Optional[str],
    proferm_gammes: Optional[List[str]] = None,
) -> ClassificationStatus:
    complete, _ = validate_classification(
        supplier=supplier,
        product_types=product_types,
        materials=materials,
        coulissant_galandage=coulissant_galandage,
        proferm_gammes=proferm_gammes,
    )
    return "complete" if complete else "incomplete"


def taxonomy_payload() -> Dict[str, Any]:
    """Payload pour GET /api/library/taxonomy."""
    return {
        "suppliers": SUPPLIERS,
        "product_types": [
            {"value": v, "label": LABELS["product_types"][v]} for v in PRODUCT_TYPES
        ],
        "materials": [
            {"value": v, "label": LABELS["materials"][v]} for v in MATERIALS
        ],
        "coulissant_galandage": [
            {"value": v, "label": LABELS["coulissant_galandage"][v]}
            for v in COULISSANT_GALANDAGE_OPTIONS
        ],
        "proferm_gammes": [
            {"value": v, "label": LABELS["proferm_gammes"][v]} for v in PROFERM_GAMMES
        ],
    }
