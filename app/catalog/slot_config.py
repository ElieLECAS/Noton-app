"""
Configuration déclarative du slot filling menuiserie.

Les règles métier (validation, détection, enrichment retrieval) sont dérivées
de cette taxonomie — pas de cas particuliers codés dans les services.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Literal, Optional, Sequence, Tuple

from app.catalog.taxonomy import (
    PRODUCT_TYPES,
    PROFERM_GAMME_ALIASES,
    SUPPLIERS,
    normalize_token,
    resolve_proferm_gammes_from_text,
    slot_material_to_document,
    supplier_skips_material_requirement,
    supplier_uses_proferm_gammes,
)

IntentType = Literal[
    "diagnostic_probleme",
    "recherche_reference",
    "norme_procedure",
    "comparatif_produits",
]
ProductType = Literal["fenetre", "porte", "coulissant"]
GalandageType = Literal["oui", "non", "inconnu"]
MaterialType = Literal["pvc", "alu", "mixte", "bois", "inconnu"]

SLOT_PRIORITY_ORDER: Tuple[str, ...] = (
    "type",
    "material",
    "context_usage",
    "galandage",
    "problem_symptom",
    "range_or_model",
    "supplier_brand",
)

RETRIEVAL_INSTALLATION_SUFFIX = "schéma coupe pose calfeutrement"
RETRIEVAL_EARLY_PAGE_THRESHOLD = 6
RETRIEVAL_EARLY_PAGE_PENALTY = 0.7
RETRIEVAL_TECHNICAL_KEYWORD_BOOST = 1.2

RETRIEVAL_TECHNICAL_KEYWORDS: Tuple[str, ...] = (
    "pose",
    "coupe",
    "schema",
    "schéma",
    "mise en oeuvre",
    "calfeutrement",
)

RETRIEVAL_ADMIN_KEYWORDS: Tuple[str, ...] = (
    "avis",
    "domaine d'emploi",
    "avant-propos",
    "dossier technique",
)


@dataclass(frozen=True)
class SlotValueDef:
    value: str
    label: str
    patterns: Tuple[str, ...] = ()
    clarification_message: str = ""


@dataclass(frozen=True)
class SlotDef:
    name: str
    question: str
    hint: str
    values: Tuple[SlotValueDef, ...] = ()
    retrieval_suffix: str = ""


SLOT_DEFS: Dict[str, SlotDef] = {
    "type": SlotDef(
        name="type",
        question="Quel type de produit concerne votre demande ?",
        hint="Répondez par exemple : fenêtre, porte ou coulissant (baie coulissante).",
        values=(
            SlotValueDef("fenetre", "Fenêtre", (r"\b(fenetre|fenêtre|chassis|châssis|ouvrant)\b",), "fenêtre"),
            SlotValueDef("porte", "Porte", (r"\b(porte)\b",), "porte"),
            SlotValueDef(
                "coulissant",
                "Coulissant",
                (r"\b(baie vitree|baie coulissante|coulissant|galandage)\b",),
                "coulissant",
            ),
        ),
    ),
    "material": SlotDef(
        name="material",
        question="Quel est le matériau du châssis ?",
        hint="Répondez par exemple : PVC, aluminium (alu), mixte ou bois.",
        values=(
            SlotValueDef("pvc", "PVC", (r"\bpvc\b",), "PVC"),
            SlotValueDef("alu", "Aluminium", (r"\b(alu|aluminium|aluminum)\b",), "aluminium"),
            SlotValueDef("mixte", "Mixte", (r"\b(mixte|bi[- ]?matiere|bi-matière)\b",), "mixte"),
            SlotValueDef("bois", "Bois", (r"\b(bois)\b",), "bois"),
        ),
    ),
    "material_coulissant": SlotDef(
        name="material",
        question="Quel est le matériau du châssis ?",
        hint="Répondez par exemple : PVC ou aluminium.",
        values=(
            SlotValueDef("pvc", "PVC", (r"\bpvc\b",), "PVC"),
            SlotValueDef("alu", "Aluminium", (r"\b(alu|aluminium|aluminum)\b",), "aluminium"),
        ),
    ),
    "galandage": SlotDef(
        name="galandage",
        question="S'agit-il d'un coulissant à galandage ?",
        hint=(
            "Répondez « oui » pour un coulissant à galandage (vantail dans le mur), "
            "« non » pour un coulissant 2 rails classique."
        ),
        values=(
            SlotValueDef("oui", "Oui (galandage)", (r"\b(galandage|a galandage|à galandage)\b",), "oui galandage"),
            SlotValueDef(
                "non",
                "Non (2 rails)",
                (r"\b(sans galandage|2 rails|deux rails|coulissant 2 rails)\b",),
                "non galandage",
            ),
        ),
    ),
    "context_usage": SlotDef(
        name="context_usage",
        question="Quel est le contexte de pose ?",
        hint="Précisez le mode de pose ou le support (applique, monomur, tableau, ITE, rénovation…).",
        values=(
            SlotValueDef(
                "applique_exterieure",
                "applique extérieure",
                (r"\b(applique\s+exterieure|applique\s+exterieur)\b",),
                "applique extérieure",
            ),
            SlotValueDef(
                "applique_interieure",
                "applique intérieure",
                (r"\b(applique\s+interieure|applique\s+interieur)\b",),
                "applique intérieure",
            ),
            SlotValueDef("monomur", "monomur", (r"\bmonomur\b",), "monomur"),
            SlotValueDef("tableau", "tableau", (r"\b(tableau|tunnel)\b",), "tableau"),
            SlotValueDef("ite", "ITE", (r"\b(ite|bardage|enduit)\b",), "ITE bardage"),
            SlotValueDef(
                "renovation",
                "rénovation",
                (r"\b(renovation|dormant existant)\b",),
                "rénovation sur dormant",
            ),
        ),
        retrieval_suffix=RETRIEVAL_INSTALLATION_SUFFIX,
    ),
    "component_part": SlotDef(
        name="component_part",
        question="Quelle pièce ou élément est concerné ?",
        hint="Précisez la pièce ou l'élément technique visé.",
        values=(
            SlotValueDef("membrane", "membrane", (r"\bmembrane\b",)),
            SlotValueDef("crémone", "crémone", (r"\b(cremone|crémone)\b",)),
            SlotValueDef("poignée", "poignée", (r"\bpoignee\b",)),
            SlotValueDef("serrure", "serrure", (r"\b(serrure|digicode|telecommande)\b",)),
            SlotValueDef("rail", "rail", (r"\brail\b",)),
            SlotValueDef("seuil", "seuil", (r"\bseuil\b",)),
            SlotValueDef("capotage", "capotage", (r"\bcapotage\b",)),
            SlotValueDef("joint", "joint", (r"\bjoint\b",)),
            SlotValueDef("galet", "galet", (r"\bgalet\b",)),
        ),
        retrieval_suffix=RETRIEVAL_INSTALLATION_SUFFIX,
    ),
}

INSTALLATION_SLOT_NAMES: FrozenSet[str] = frozenset({"context_usage", "component_part"})

INTENT_DETECTION_ORDER: Tuple[IntentType, ...] = (
    "norme_procedure",
    "comparatif_produits",
    "recherche_reference",
    "diagnostic_probleme",
)

INTENT_PATTERNS: Dict[IntentType, Tuple[str, ...]] = {
    "norme_procedure": (
        r"\b(dtu|norme|nf en|procedure|procédure)\b",
        r"\b(prevoir|prevoir|quelle\s+(cote|retombee|retombée|valeur))\b",
    ),
    "comparatif_produits": (
        r"\b(comparer|comparatif|difference|différence| vs | versus )\b",
    ),
    "recherche_reference": (
        r"\b(gamme|reference|référence|modele|modèle|dimension)\b",
        r"\bcote du\b|\bcote de\b|\bcote d['\u2019]\b|\bdimensionnement\b",
    ),
    "diagnostic_probleme": (
        r"\b(probleme|problème|ferme pas|ferme mal|panne|defaut|défaut|reglage|réglage|bloque|coince)\b",
    ),
}

INTENT_VALIDATION: Dict[IntentType, Dict[str, object]] = {
    "diagnostic_probleme": {
        "required": ("type", "material", "problem_symptom"),
    },
    "recherche_reference": {
        "required": ("type", "material"),
        "one_of": (("range_or_model", "supplier_brand"),),
    },
    "norme_procedure": {
        "required": ("type", "material"),
        "conditional": (("context_usage", "installation_context"),),
    },
    "comparatif_produits": {
        "required": ("type", "material"),
    },
}

PRODUCT_TYPE_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("coulissant", ("galandage",)),
)

MATERIAL_RETRIEVAL_TERMS: Dict[str, str] = {
    "pvc": "châssis PVC menuiserie",
    "alu": "châssis aluminium menuiserie",
    "mixte": "châssis mixte aluminium PVC",
    "bois": "châssis bois menuiserie",
}

MATERIAL_GENERATION_CONSTRAINTS: Dict[str, str] = {
    "pvc": (
        "L'utilisateur a confirmé un châssis PVC. N'utilise que les passages explicitement "
        "liés au PVC dans le contexte. N'applique pas des notices clairement dédiées à "
        "l'aluminium. Ne déduis aucune gamme ni référence non fournie par l'utilisateur."
    ),
    "alu": (
        "L'utilisateur a confirmé un châssis aluminium. N'utilise que les passages "
        "explicitement liés à l'aluminium dans le contexte. N'applique pas des notices "
        "clairement dédiées au PVC. Ne déduis aucune gamme ni référence non fournie par l'utilisateur."
    ),
    "mixte": (
        "L'utilisateur a indiqué un châssis mixte. Distingue explicitement les parties PVC "
        "et aluminium si le contexte contient les deux matériaux."
    ),
    "bois": (
        "L'utilisateur a indiqué un châssis bois. Ne mélange pas avec le PVC ou l'aluminium "
        "sauf si le contexte le justifie explicitement."
    ),
}

SLOT_SUMMARY_LABELS: Dict[str, str] = {
    "type": "Produit",
    "material": "Matériau",
    "galandage": "Galandage",
    "problem_symptom": "Symptôme",
    "range_or_model": "Gamme / référence",
    "supplier_brand": "Marque",
    "component_part": "Pièce concernée",
    "context_usage": "Contexte de pose",
}

CLARIFICATION_QUESTIONS_EXTRA: Dict[str, str] = {
    "problem_symptom": "Quel est le symptôme précis que vous observez ?",
    "range_or_model": "Quelle gamme ou référence produit ?",
    "supplier_brand": "Quelle marque ou fournisseur ?",
}

CLARIFICATION_HINTS_EXTRA: Dict[str, str] = {
    "problem_symptom": (
        "Soyez le plus concret possible. Exemples utiles :\n"
        "• « Le vantail ferme mal côté poignée »\n"
        "• « La crémone bloque, la fermeture 3 points ne s'enclenche pas »\n"
        "• « Infiltration d'air / d'eau en bas de feuillure »\n"
        "• « Le vantail frotte en haut du dormant »"
    ),
    "range_or_model": "Indiquez la gamme ou la référence produit si vous la connaissez.",
    "supplier_brand": "Indiquez la marque ou le fournisseur si vous le connaissez.",
}

INTENT_LABELS: Dict[str, str] = {
    "diagnostic_probleme": "diagnostic",
    "recherche_reference": "référence technique",
    "norme_procedure": "norme procédure",
    "comparatif_produits": "comparatif",
}

_INSTALLATION_QUERY_RE: Optional[re.Pattern] = None
_TECHNICAL_KEYWORD_RE: Optional[re.Pattern] = None
_ADMIN_KEYWORD_RE: Optional[re.Pattern] = None


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    return unicodedata.normalize("NFD", lowered).encode("ascii", "ignore").decode("ascii")


def _installation_triggers() -> Tuple[str, ...]:
    tokens: List[str] = list(RETRIEVAL_TECHNICAL_KEYWORDS)
    for slot_name in INSTALLATION_SLOT_NAMES:
        slot_def = SLOT_DEFS.get(slot_name)
        if not slot_def:
            continue
        for value_def in slot_def.values:
            tokens.append(normalize_text(value_def.label))
            tokens.append(normalize_text(value_def.value.replace("_", " ")))
    return tuple(dict.fromkeys(t for t in tokens if t))


def installation_query_pattern() -> re.Pattern:
    global _INSTALLATION_QUERY_RE
    if _INSTALLATION_QUERY_RE is None:
        triggers = _installation_triggers()
        pattern = "|".join(re.escape(t) for t in triggers)
        _INSTALLATION_QUERY_RE = re.compile(rf"\b({pattern})\b", re.IGNORECASE)
    return _INSTALLATION_QUERY_RE


def technical_keyword_pattern() -> re.Pattern:
    global _TECHNICAL_KEYWORD_RE
    if _TECHNICAL_KEYWORD_RE is None:
        pattern = "|".join(re.escape(normalize_text(k)) for k in RETRIEVAL_TECHNICAL_KEYWORDS)
        _TECHNICAL_KEYWORD_RE = re.compile(rf"\b({pattern})\b", re.IGNORECASE)
    return _TECHNICAL_KEYWORD_RE


def admin_keyword_pattern() -> re.Pattern:
    global _ADMIN_KEYWORD_RE
    if _ADMIN_KEYWORD_RE is None:
        pattern = "|".join(re.escape(normalize_token(k)) for k in RETRIEVAL_ADMIN_KEYWORDS)
        _ADMIN_KEYWORD_RE = re.compile(rf"\b({pattern})\b", re.IGNORECASE)
    return _ADMIN_KEYWORD_RE


def query_suggests_installation_context(text: str) -> bool:
    if not text or not text.strip():
        return False
    return bool(installation_query_pattern().search(normalize_text(text)))


def detect_slot_value(slot_name: str, text_norm: str) -> Optional[str]:
    slot_def = SLOT_DEFS.get(slot_name)
    if not slot_def:
        return None
    for value_def in slot_def.values:
        for pattern in value_def.patterns:
            if re.search(pattern, text_norm):
                return value_def.value
    return None


def slot_value_label(slot_name: str, value: str) -> str:
    slot_def = SLOT_DEFS.get(slot_name)
    if not slot_def:
        return value
    for value_def in slot_def.values:
        if value_def.value == value:
            return value_def.label
    return value


def get_slot_options(slot_name: str, product_type: Optional[str] = None) -> List[str]:
    key = slot_name
    if slot_name == "material" and product_type == "coulissant":
        key = "material_coulissant"
    slot_def = SLOT_DEFS.get(key)
    if not slot_def:
        return []
    return [v.value for v in slot_def.values]


def get_clarification_actions(
    slot_name: str,
    product_type: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    key = slot_name
    if slot_name == "material" and product_type == "coulissant":
        key = "material_coulissant"
    slot_def = SLOT_DEFS.get(key)
    if not slot_def:
        return []
    return [
        (v.value, v.label, v.clarification_message or v.label)
        for v in slot_def.values
    ]


def get_clarification_question(slot_name: str) -> str:
    slot_def = SLOT_DEFS.get(slot_name)
    if slot_def:
        return slot_def.question
    return CLARIFICATION_QUESTIONS_EXTRA.get(slot_name, f"Précisez : {slot_name}")


def get_clarification_hint(slot_name: str) -> str:
    slot_def = SLOT_DEFS.get(slot_name)
    if slot_def:
        return slot_def.hint
    return CLARIFICATION_HINTS_EXTRA.get(slot_name, "")


def detect_intent(text_norm: str) -> IntentType:
    if query_suggests_installation_context(text_norm):
        return "norme_procedure"
    for intent in INTENT_DETECTION_ORDER:
        patterns = INTENT_PATTERNS.get(intent, ())
        if any(re.search(pattern, text_norm) for pattern in patterns):
            return intent
    return "diagnostic_probleme"


def detect_product_type(text_norm: str) -> Optional[ProductType]:
    if re.search(r"\b(porte[- ]?fenetre|porte fenetre)\b", text_norm):
        return "fenetre"
    detected = detect_slot_value("type", text_norm)
    if detected == "porte" and re.search(r"\b(fenetre|fenêtre)\b", text_norm):
        return "fenetre"
    return detected  # type: ignore[return-value]


def detect_material(text_norm: str) -> Optional[MaterialType]:
    value = detect_slot_value("material", text_norm)
    return value  # type: ignore[return-value]


def detect_galandage(text_norm: str) -> GalandageType:
    value = detect_slot_value("galandage", text_norm)
    return value or "inconnu"  # type: ignore[return-value]


def detect_supplier(text: str) -> Optional[str]:
    text_norm = normalize_text(text)
    for supplier in SUPPLIERS:
        if normalize_token(supplier) in text_norm:
            return supplier
    return None


def detect_range_or_model(text: str) -> Optional[str]:
    norm = normalize_text(text)
    for _gamme, aliases in PROFERM_GAMME_ALIASES.items():
        for alias in aliases:
            alias_norm = normalize_token(alias)
            match = re.search(rf"\b{re.escape(alias_norm)}(\s*\d+)?\b", norm)
            if match:
                return match.group(0).strip()
    generic = re.search(r"\b([A-Z]{1,3}[- ]?\d{2,5}[A-Z0-9-]*)\b", text, re.IGNORECASE)
    if generic:
        return generic.group(1).strip()
    return None


def detect_all_context_usages(text_norm: str) -> List[str]:
    found: List[str] = []
    for value_def in SLOT_DEFS["context_usage"].values:
        for pattern in value_def.patterns:
            if re.search(pattern, text_norm):
                found.append(value_def.value)
                break
    return found


def format_context_usage_display(
    value: Optional[str],
    conversation_text: str = "",
) -> str:
    usages = detect_all_context_usages(normalize_text(conversation_text))
    if value:
        norm_val = normalize_text(value.replace("_", " "))
        for value_def in SLOT_DEFS["context_usage"].values:
            token = value_def.value.replace("_", " ")
            if (
                value_def.value in value
                or re.search(rf"\b{re.escape(token)}\b", norm_val)
            ):
                if value_def.value not in usages:
                    usages.append(value_def.value)
    if not usages:
        return slot_value_label("context_usage", value) if value else ""
    return ", ".join(slot_value_label("context_usage", u) for u in dict.fromkeys(usages))


def format_component_part_display(value: Optional[str]) -> str:
    if not value:
        return ""
    detected = detect_slot_value("component_part", normalize_text(value))
    if detected:
        return slot_value_label("component_part", detected)
    return value


def slot_state_sanitization_updates(
    *,
    context_usage: Optional[str],
    component_part: Optional[str],
    material: Optional[str],
    product_type: Optional[str],
    conversation_text: str = "",
) -> Dict[str, str]:
    """Retourne les champs à corriger pour aligner les valeurs LLM sur la taxonomie."""
    text_norm = normalize_text(conversation_text)
    updates: Dict[str, str] = {}

    usages = detect_all_context_usages(text_norm)
    if usages:
        updates["context_usage"] = usages[0]
    elif context_usage:
        for value_def in SLOT_DEFS["context_usage"].values:
            if value_def.value in str(context_usage):
                updates["context_usage"] = value_def.value
                break

    if component_part:
        detected = detect_slot_value("component_part", normalize_text(component_part))
        if detected:
            updates["component_part"] = detected

    if material:
        detected = detect_slot_value("material", normalize_text(str(material)))
        if detected:
            updates["material"] = detected

    if product_type:
        detected = detect_product_type(normalize_text(str(product_type)))
        if detected:
            updates["type"] = detected

    return updates


def needs_installation_context(
    *,
    context_usage: Optional[str],
    component_part: Optional[str],
    conversation_text: str = "",
) -> bool:
    if context_usage or detect_all_context_usages(normalize_text(conversation_text)):
        return False
    if component_part:
        return True
    return query_suggests_installation_context(conversation_text)


def material_required(
    *,
    intent: IntentType,
    product_type: Optional[str],
    supplier_brand: Optional[str],
    conversation_text: str,
) -> bool:
    if intent not in ("diagnostic_probleme", "norme_procedure", "recherche_reference", "comparatif_produits"):
        return False
    if supplier_skips_material_requirement(supplier_brand, product_type):
        return False
    if intent == "comparatif_produits" and _conversation_has_multi_material_comparison(conversation_text):
        return False
    return True


def _conversation_has_multi_material_comparison(text: str) -> bool:
    norm = normalize_text(text)
    has_pvc = bool(re.search(r"\bpvc\b", norm))
    has_alu = bool(re.search(r"\b(alu|aluminium|aluminum)\b", norm))
    return has_pvc and has_alu


def build_retrieval_enrichment(
    *,
    material: Optional[str],
    context_usage: Optional[str],
    component_part: Optional[str],
) -> str:
    parts: List[str] = []
    if material and material != "inconnu":
        terms = MATERIAL_RETRIEVAL_TERMS.get(material, "")
        doc_material = slot_material_to_document(material)
        if doc_material == "hybride" and not terms:
            terms = "châssis hybride aluminium PVC"
        if terms:
            parts.append(terms)

    if context_usage:
        label = slot_value_label("context_usage", context_usage)
        suffix = SLOT_DEFS["context_usage"].retrieval_suffix
        parts.append(f"pose {label} {suffix}".strip())
    elif component_part:
        suffix = SLOT_DEFS["component_part"].retrieval_suffix
        parts.append(f"{component_part} {suffix}".strip())

    return " ".join(parts).strip()


def should_apply_proferm_gamme_filter(
    *,
    material: Optional[str],
    supplier_brand: Optional[str],
    resolved_gammes: Sequence[str],
) -> bool:
    if not resolved_gammes:
        return False
    if slot_material_to_document(material) == "pvc":
        return True
    return supplier_uses_proferm_gammes(supplier_brand)


def passage_has_technical_installation_content(text: str) -> bool:
    return bool(technical_keyword_pattern().search(normalize_text(text)))


def passage_looks_like_admin_front_matter(text: str) -> bool:
    return bool(admin_keyword_pattern().search(normalize_text(text)))


def build_slot_filling_prompt_fields() -> str:
    context_values = " | ".join(v.value for v in SLOT_DEFS["context_usage"].values)
    product_values = " | ".join(PRODUCT_TYPES)
    material_values = " | ".join(v.value for v in SLOT_DEFS["material"].values)
    return (
        f"- type: {product_values}\n"
        f"- material: {material_values} | inconnu\n"
        f"- context_usage: {context_values} ou null\n"
    )
