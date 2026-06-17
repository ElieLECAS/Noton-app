"""
Slot filling menuiserie pour le chat Espace.

Extraction structurée des critères métier, validation obligatoire/facultative,
construction de requête canonique et mémoire conversationnelle des slots.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.config import settings
from app.catalog.taxonomy import resolve_proferm_gammes_from_text, slot_material_to_document
from app.catalog.slot_config import (
    CLARIFICATION_HINTS_EXTRA,
    CLARIFICATION_QUESTIONS_EXTRA,
    INTENT_LABELS,
    INTENT_VALIDATION,
    MATERIAL_GENERATION_CONSTRAINTS,
    PRODUCT_TYPE_RULES,
    SLOT_DEFS,
    SLOT_PRIORITY_ORDER,
    SLOT_SUMMARY_LABELS,
    build_retrieval_enrichment,
    build_slot_filling_prompt_fields,
    detect_galandage,
    detect_intent,
    detect_material,
    detect_product_type,
    detect_range_or_model,
    detect_slot_value,
    detect_supplier,
    format_component_part_display,
    format_context_usage_display,
    get_clarification_actions,
    get_clarification_hint,
    get_clarification_question,
    get_slot_options,
    material_required,
    needs_installation_context,
    normalize_text,
    query_suggests_installation_context,
    slot_state_sanitization_updates,
    slot_value_label,
)
from app.models.message import Message

logger = logging.getLogger(__name__)

IntentType = Literal[
    "diagnostic_probleme",
    "recherche_reference",
    "norme_procedure",
    "comparatif_produits",
]
ProductType = Literal["fenetre", "porte", "coulissant"]
GalandageType = Literal["oui", "non", "inconnu"]
MaterialType = Literal["pvc", "alu", "mixte", "bois", "inconnu"]

# Rétrocompatibilité tests / imports externes
query_has_pose_signals = query_suggests_installation_context

SLOT_OPTIONS: Dict[str, List[str]] = {
    name: get_slot_options(name)
    for name in ("type", "material", "galandage", "context_usage")
}
SLOT_OPTIONS["material_coulissant"] = get_slot_options("material", "coulissant")

SLOT_CLARIFICATION_ACTIONS: Dict[str, List[tuple[str, str, str]]] = {
    name: get_clarification_actions(name)
    for name in ("type", "material", "galandage", "context_usage")
}
SLOT_CLARIFICATION_ACTIONS["material_coulissant"] = get_clarification_actions(
    "material", "coulissant"
)

CLARIFICATION_QUESTIONS: Dict[str, str] = {
    name: get_clarification_question(name)
    for name in SLOT_DEFS
}
CLARIFICATION_QUESTIONS.update(CLARIFICATION_QUESTIONS_EXTRA)

CLARIFICATION_HINTS: Dict[str, str] = {
    name: get_clarification_hint(name)
    for name in SLOT_DEFS
}
CLARIFICATION_HINTS.update(CLARIFICATION_HINTS_EXTRA)

CONTEXT_USAGE_LABELS: Dict[str, str] = {
    v.value: v.label for v in SLOT_DEFS["context_usage"].values
}

# Signaux matériau génériques (pas de noms de gammes / fournisseurs)
_GENERIC_ALU_MATERIAL_RE = re.compile(
    r"\b(aluminium|aluminum)\b|[-\s]alu[-\s]",
    re.IGNORECASE,
)
_GENERIC_PVC_MATERIAL_RE = re.compile(r"\bpvc\b", re.IGNORECASE)

SLOT_FILLING_SYSTEM_PROMPT = f"""Tu es un extracteur de slots pour un assistant RAG menuiserie (PROFERM).
Analyse le message utilisateur et l'historique récent pour remplir les champs structurés.

CHAMPS :
- intent_type: diagnostic_probleme | recherche_reference | norme_procedure | comparatif_produits
{build_slot_filling_prompt_fields()}- galandage: oui | non | inconnu (obligatoire si type=coulissant)
- opening_type: battant | coulissant | fixe | null
- supplier_brand: marque/fournisseur ou null
- range_or_model: gamme/modèle/référence ou null
- problem_symptom: description du symptôme CONCRET ou null (ferme mal, coince, fuit, frotte…)
- component_part: pièce concernée ou null
- confidence: 0.0 à 1.0

RÈGLES SYMPTÔME :
- Ne remplis problem_symptom QUE si l'utilisateur décrit un symptôme observable et précis.
- Phrases vagues (« j'ai un problème avec ma fenêtre », « ça ne va pas ») → problem_symptom = null.
- Les réponses courtes à une question de clarification (ex. « PVC », « oui ») ne sont PAS un symptôme.

NORMALISATION :
- "baie vitrée", "baie coulissante" -> type=coulissant
- "châssis" sans autre indication -> type=fenetre
- "aluminium" -> material=alu
- "à galandage" -> galandage=oui ; "2 rails", "coulissant classique" sans galandage -> galandage=non

Retourne UNIQUEMENT un JSON avec ces clés (null si inconnu).
"""


class SlotState(BaseModel):
    intent_type: Optional[IntentType] = None
    type: Optional[ProductType] = None
    galandage: Optional[GalandageType] = "inconnu"
    material: Optional[MaterialType] = None
    opening_type: Optional[str] = None
    supplier_brand: Optional[str] = None
    range_or_model: Optional[str] = None
    problem_symptom: Optional[str] = None
    component_part: Optional[str] = None
    context_usage: Optional[str] = None
    confidence: float = 0.0

    def model_dump_filled(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None and v != "inconnu"}

    def document_filter_material(self) -> Optional[str]:
        """Matériau aligné sur la taxonomie document (mixte → hybride)."""
        return slot_material_to_document(self.material)

    def resolved_proferm_gammes(self) -> List[str]:
        """Gammes Proferm déduites du slot range_or_model (filtre SQL, pas ColPali)."""
        if not self.range_or_model:
            return []
        return resolve_proferm_gammes_from_text(self.range_or_model)


class SlotValidationResult(BaseModel):
    ready: bool
    missing_required_slots: List[str] = Field(default_factory=list)
    clarification_questions: List[str] = Field(default_factory=list)
    suggested_options: Dict[str, List[str]] = Field(default_factory=dict)
    clarification_actions: List[Dict[str, str]] = Field(default_factory=list)
    slot_state: SlotState
    canonical_query: Optional[str] = None
    conversation_text: Optional[str] = None


def _normalize_text(text: str) -> str:
    return normalize_text(text)


def _detect_intent(text_norm: str) -> IntentType:
    return detect_intent(text_norm)


def _detect_product_type(text_norm: str) -> Optional[ProductType]:
    return detect_product_type(text_norm)


def _detect_material(text_norm: str) -> Optional[MaterialType]:
    return detect_material(text_norm)


def _detect_galandage(text_norm: str) -> GalandageType:
    return detect_galandage(text_norm)


def _detect_context_usage(text_norm: str) -> Optional[str]:
    return detect_slot_value("context_usage", text_norm)


def _detect_component_part(text_norm: str) -> Optional[str]:
    return detect_slot_value("component_part", text_norm)


def _needs_pose_context(state: SlotState, conversation_text: str = "") -> bool:
    return needs_installation_context(
        context_usage=state.context_usage,
        component_part=state.component_part,
        conversation_text=conversation_text,
    )


def _requires_material(state: SlotState, intent: IntentType, text_norm: str) -> bool:
    return material_required(
        intent=intent,
        product_type=state.type,
        supplier_brand=state.supplier_brand,
        conversation_text=text_norm,
    )


def collect_user_messages(
    conversation_context: Optional[List[Dict[str, str]]],
    current_message: str,
) -> str:
    """Agrège les messages utilisateur pour la requête retrieval multi-tours."""
    parts: List[str] = []
    if conversation_context:
        for msg in conversation_context:
            if msg.get("role") == "user" and msg.get("content"):
                content = str(msg["content"]).strip()
                if content:
                    parts.append(content)
    current = (current_message or "").strip()
    if current and (not parts or parts[-1] != current):
        parts.append(current)
    return " ".join(parts)


def _detect_supplier(text: str) -> Optional[str]:
    return detect_supplier(text)


def _detect_range_or_model(text: str) -> Optional[str]:
    return detect_range_or_model(text)


_VAGUE_PROBLEM_INTRO_RE = re.compile(
    r"^(?:j['\u2019]?ai\s+)?(?:un\s+)?probleme\s+(?:avec|sur)\s+"
    r"(?:ma|mon|mes|une|un|le|la|l)\s+"
    r"(?:fenetre|porte|coulissant|chassis|baie|ouvrant|vantail)"
    r"(?:\s+.*)?$",
)

_SYMPTOM_SPECIFIC_RE = re.compile(
    r"\b(?:"
    r"ferme\s+pas|ferme\s+mal|ne\s+ferme\s+pas|n['\u2019]ouvre\s+pas|"
    r"coince|bloque|frotte|fuit|infiltre|infiltration|"
    r"reglage|alignement|desalign|jeu|"
    r"etancheite|etanche|casse|endommage|"
    r"poignee|serrure|cremone|galet|rail|"
    r"vent|air|eau|joint|usure|fissure|"
    r"ouverture|fermeture|vantail|enclenche"
    r")\b",
    re.IGNORECASE,
)


def _is_meaningful_problem_symptom(symptom: Optional[str]) -> bool:
    """Un symptôme vague (« j'ai un problème avec ma fenêtre ») ne suffit pas."""
    if not symptom or not isinstance(symptom, str):
        return False
    cleaned = symptom.strip()
    if not cleaned:
        return False
    text_norm = _normalize_text(cleaned)
    if _VAGUE_PROBLEM_INTRO_RE.match(text_norm):
        return False
    if re.fullmatch(r"probleme(?:\s+(?:avec|sur)\s+\w+)*", text_norm):
        return False
    return bool(_SYMPTOM_SPECIFIC_RE.search(text_norm))


def _detect_problem_symptom(text: str, text_norm: str) -> Optional[str]:
    symptom_patterns = [
        r"(ferme pas[^.?]*)",
        r"(ferme mal[^.?]*)",
        r"(ne ferme pas[^.?]*)",
        r"(coince[^.?]*)",
        r"(bloque[^.?]*)",
        r"(fuit[^.?]*)",
        r"(infiltre[^.?]*)",
        r"(frotte[^.?]*)",
    ]
    for pattern in symptom_patterns:
        match = re.search(pattern, text_norm, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()[:200]
            if _is_meaningful_problem_symptom(candidate):
                return candidate
    return None


def extract_slots_heuristic(query: str, history: Optional[List[Dict[str, str]]] = None) -> SlotState:
    """Extraction déterministe par règles (utilisée en fallback et en tests)."""
    combined_parts = [query]
    if history:
        for msg in history[-4:]:
            if msg.get("role") == "user" and msg.get("content"):
                combined_parts.append(str(msg["content"]))
    combined = " ".join(combined_parts)
    text_norm = _normalize_text(combined)

    intent = _detect_intent(text_norm)
    product_type = _detect_product_type(text_norm)
    material = _detect_material(text_norm)
    galandage = _detect_galandage(text_norm)
    supplier = _detect_supplier(combined)
    range_or_model = _detect_range_or_model(combined)
    problem_symptom = _detect_problem_symptom(combined, text_norm)
    context_usage = _detect_context_usage(text_norm)
    component_part = _detect_component_part(text_norm)

    confidence = 0.4
    filled = sum(
        1
        for v in [
            product_type,
            material,
            problem_symptom,
            range_or_model,
            supplier,
            context_usage,
            component_part,
        ]
        if v
    )
    confidence = min(0.95, 0.35 + filled * 0.12)

    return SlotState(
        intent_type=intent,
        type=product_type,
        galandage=galandage,
        material=material,
        supplier_brand=supplier,
        range_or_model=range_or_model,
        problem_symptom=problem_symptom,
        context_usage=context_usage,
        component_part=component_part,
        confidence=confidence,
    )


async def extract_slots_llm(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[SlotState]:
    """Extraction LLM structurée (JSON). Retourne None si indisponible ou erreur."""
    try:
        from app.services.mistral_service import chat

        context_lines = []
        if history:
            for msg in history[-6:]:
                role = msg.get("role", "user")
                content = str(msg.get("content", "")).strip()
                if content:
                    context_lines.append(f"{role}: {content}")
        user_content = "Historique récent:\n" + "\n".join(context_lines) if context_lines else ""
        user_content += f"\n\nMessage à analyser: '{query}'"

        response = await chat(
            "",
            model=settings.MODEL_FAST,
            context=[
                {"role": "system", "content": SLOT_FILLING_SYSTEM_PROMPT},
                {"role": "user", "content": user_content.strip()},
            ],
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"].get("content", "{}")
        data = json.loads(content)
        return SlotState.model_validate(data)
    except Exception as exc:
        logger.warning("Extraction LLM slots échouée, fallback heuristique: %s", exc)
        return None


def merge_slot_states(base: Optional[SlotState], new: SlotState) -> SlotState:
    """Fusionne deux états : les valeurs non nulles/inconnues de `new` écrasent `base`."""
    if base is None:
        return new

    merged = base.model_dump()
    incoming = new.model_dump()
    for key, value in incoming.items():
        if key == "confidence":
            merged[key] = max(float(merged.get(key) or 0.0), float(value or 0.0))
            continue
        if value is None:
            continue
        if key == "galandage" and value == "inconnu":
            continue
        if key == "material" and value == "inconnu":
            continue
        merged[key] = value
    return SlotState.model_validate(merged)


def _slot_is_filled(state: SlotState, slot_name: str) -> bool:
    value = getattr(state, slot_name, None)
    if value is None:
        return False
    if slot_name in ("galandage", "material") and value == "inconnu":
        return False
    if slot_name == "problem_symptom" and not _is_meaningful_problem_symptom(value):
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def build_clarification_actions(
    state: SlotState,
    missing_slots: List[str],
) -> List[Dict[str, str]]:
    """Boutons proposés pour le slot manquant prioritaire (UI chat)."""
    if not missing_slots:
        return []

    primary = missing_slots[0]
    if primary == "material" and state.type == "coulissant":
        action_key = "material_coulissant"
    elif primary in SLOT_CLARIFICATION_ACTIONS:
        action_key = primary
    else:
        return []

    slot_name = "material" if action_key == "material_coulissant" else primary
    return [
        {
            "slot": slot_name,
            "value": value,
            "label": label,
            "message": message,
        }
        for value, label, message in SLOT_CLARIFICATION_ACTIONS[action_key]
    ]


def _suggested_options_for_missing(
    state: SlotState,
    missing_slots: List[str],
) -> Dict[str, List[str]]:
    suggested: Dict[str, List[str]] = {}
    for slot in missing_slots:
        if slot == "material" and state.type == "coulissant":
            suggested[slot] = list(SLOT_OPTIONS["material_coulissant"])
        elif slot in SLOT_OPTIONS:
            suggested[slot] = list(SLOT_OPTIONS[slot])
    return suggested


def _order_missing_slots(missing: List[str]) -> List[str]:
    priority = {name: idx for idx, name in enumerate(SLOT_PRIORITY_ORDER)}
    seen = set()
    ordered: List[str] = []
    for slot in sorted(missing, key=lambda s: priority.get(s, 99)):
        if slot not in seen:
            seen.add(slot)
            ordered.append(slot)
    return ordered


def _apply_validation_rules(
    state: SlotState,
    intent: IntentType,
    text_norm: str,
    missing: List[str],
) -> None:
    rules = INTENT_VALIDATION.get(intent, {})
    for slot in rules.get("required", ()):
        if slot == "material" and not _requires_material(state, intent, text_norm):
            continue
        if not _slot_is_filled(state, slot):
            missing.append(slot)

    for group in rules.get("one_of", ()):
        if not any(_slot_is_filled(state, slot) for slot in group):
            missing.append(group[0])

    for slot, condition in rules.get("conditional", ()):
        if condition == "installation_context" and _needs_pose_context(state, text_norm):
            if not _slot_is_filled(state, slot):
                missing.append(slot)


def validate_slots(
    state: SlotState,
    conversation_text: Optional[str] = None,
) -> SlotValidationResult:
    """Applique la matrice obligatoire/facultative selon intent_type."""
    if state.problem_symptom and not _is_meaningful_problem_symptom(state.problem_symptom):
        state = state.model_copy(update={"problem_symptom": None})

    missing: List[str] = []
    intent = state.intent_type or "diagnostic_probleme"
    text_norm = _normalize_text(conversation_text or "")

    _apply_validation_rules(state, intent, text_norm, missing)

    for product_type, extra_slots in PRODUCT_TYPE_RULES:
        if state.type == product_type:
            for slot in extra_slots:
                if not _slot_is_filled(state, slot):
                    missing.append(slot)

    ordered_missing = _order_missing_slots(missing)

    questions = [CLARIFICATION_QUESTIONS[s] for s in ordered_missing if s in CLARIFICATION_QUESTIONS][:1]
    suggested = _suggested_options_for_missing(state, ordered_missing)
    actions = build_clarification_actions(state, ordered_missing)

    ready = len(ordered_missing) == 0
    canonical = build_canonical_query(state) if ready else None

    return SlotValidationResult(
        ready=ready,
        missing_required_slots=ordered_missing,
        clarification_questions=questions,
        suggested_options=suggested,
        clarification_actions=actions,
        slot_state=state,
        canonical_query=canonical,
    )


def build_canonical_query(state: SlotState) -> str:
    """Construit la requête canonique à partir des slots validés."""
    parts: List[str] = []

    type_labels = {v.value: v.label for v in SLOT_DEFS["type"].values}
    if state.type:
        parts.append(type_labels.get(state.type, state.type))

    if state.galandage and state.galandage != "inconnu":
        parts.append("galandage" if state.galandage == "oui" else "sans galandage")

    if state.material and state.material != "inconnu":
        parts.append(state.material)

    if state.opening_type:
        parts.append(state.opening_type)

    if state.supplier_brand:
        parts.append(state.supplier_brand)

    if state.range_or_model:
        parts.append(state.range_or_model)

    if state.component_part:
        parts.append(state.component_part)

    if state.problem_symptom:
        parts.append(state.problem_symptom)

    if state.context_usage:
        parts.append(format_context_usage_display(state.context_usage))

    if state.intent_type:
        parts.append(INTENT_LABELS.get(state.intent_type, state.intent_type))

    return " ".join(p for p in parts if p).strip()


def build_retrieval_query(
    canonical_query: str,
    raw_user_message: str,
    slot_state: Optional[SlotState] = None,
) -> str:
    """Requête finale retrieval : canonique + message(s) utilisateur + ancrage métier."""
    canonical = (canonical_query or "").strip()
    raw = (raw_user_message or "").strip()
    if canonical and raw:
        if raw.lower() in canonical.lower():
            base = canonical
        else:
            base = f"{canonical} {raw}".strip()
    else:
        base = canonical or raw

    enrichment = build_retrieval_enrichment(
        material=slot_state.material if slot_state else None,
        context_usage=slot_state.context_usage if slot_state else None,
        component_part=slot_state.component_part if slot_state else None,
    )
    if enrichment and enrichment.lower() not in base.lower():
        base = f"{base} {enrichment}".strip()

    return base


def build_retrieval_query_from_conversation(
    canonical_query: str,
    conversation_context: Optional[List[Dict[str, str]]],
    current_message: str,
    slot_state: Optional[SlotState] = None,
) -> str:
    """Requête retrieval enrichie avec l'historique utilisateur complet."""
    aggregated = collect_user_messages(conversation_context, current_message)
    return build_retrieval_query(
        canonical_query,
        aggregated,
        slot_state=slot_state,
    )


def _document_title(passage: Dict[str, Any]) -> str:
    return str(passage.get("document_title") or "")


def _detect_material_signals(text: str) -> set[str]:
    """Détecte les mentions explicites de matériau dans un texte (sans nom de gamme)."""
    signals: set[str] = set()
    if _GENERIC_PVC_MATERIAL_RE.search(text):
        signals.add("pvc")
    if _GENERIC_ALU_MATERIAL_RE.search(text):
        signals.add("alu")
    return signals


def _passage_material_signals(passage: Dict[str, Any]) -> set[str]:
    """Signaux matériau agrégés depuis titre, extrait et métadonnée source."""
    combined = " ".join(
        part
        for part in (
            _document_title(passage),
            str(passage.get("passage_raw") or ""),
            str(passage.get("source") or ""),
        )
        if part
    )
    return _detect_material_signals(combined)


def _passage_matches_material_doc(passage: Dict[str, Any], material: str) -> bool:
    """
    True si le passage est compatible avec le matériau demandé.
    Sans signal matériau explicite dans le passage → conservé (pas d'exclusion par nom de gamme).
    """
    signals = _passage_material_signals(passage)
    if not signals:
        return True
    if material == "pvc":
        return "alu" not in signals or "pvc" in signals
    if material == "alu":
        return "pvc" not in signals or "alu" in signals
    return True


def filter_passages_by_slot_material(
    passages: List[Dict[str, Any]],
    slot_state: Optional[SlotState],
) -> List[Dict[str, Any]]:
    """Écarte les passages issus de catalogues incompatibles avec le matériau validé."""
    if not passages or not slot_state:
        return passages
    material = slot_state.material
    if not material or material in ("inconnu", "mixte", "bois"):
        return passages

    filtered = [
        p for p in passages if _passage_matches_material_doc(p, material)
    ]
    if filtered:
        if len(filtered) < len(passages):
            logger.info(
                "Filtre matériau %s : %d -> %d passage(s)",
                material,
                len(passages),
                len(filtered),
            )
        return filtered

    logger.warning(
        "Filtre matériau %s : aucun passage compatible (%d écarté(s))",
        material,
        len(passages),
    )
    return []


def _format_slot_display_value(
    slot_name: str,
    value: Any,
    conversation_text: str = "",
) -> str:
    if slot_name == "type":
        return slot_value_label("type", str(value))
    if slot_name == "material":
        return slot_value_label("material", str(value))
    if slot_name == "galandage":
        return "Oui" if value == "oui" else "Non"
    if slot_name == "context_usage":
        return format_context_usage_display(str(value), conversation_text)
    if slot_name == "component_part":
        return format_component_part_display(str(value))
    return str(value)


def _get_filled_slots_summary(
    state: SlotState,
    conversation_text: str = "",
) -> List[tuple[str, str]]:
    summary: List[tuple[str, str]] = []
    for slot_name, label in SLOT_SUMMARY_LABELS.items():
        if _slot_is_filled(state, slot_name):
            summary.append(
                (
                    label,
                    _format_slot_display_value(
                        slot_name,
                        getattr(state, slot_name),
                        conversation_text,
                    ),
                )
            )
    return summary


def _build_optional_precision_hint(state: SlotState, missing: List[str]) -> Optional[str]:
    """Suggère des infos facultatives utiles au retrieval, sans bloquer."""
    if not missing or missing[0] != "problem_symptom":
        return None
    intent = state.intent_type or "diagnostic_probleme"
    if intent != "diagnostic_probleme":
        return None

    extras: List[str] = []
    if not _slot_is_filled(state, "range_or_model"):
        extras.append("la gamme ou la référence produit")
    if not _slot_is_filled(state, "component_part"):
        extras.append("la pièce concernée (crémone, poignée, galet, joint…)")

    if not extras:
        return None
    joined = " et ".join(extras)
    return f"💡 Vous pouvez aussi préciser {joined} dans votre réponse pour affiner la recherche documentaire."


def build_clarification_message(validation: SlotValidationResult) -> str:
    state = validation.slot_state
    conversation_text = validation.conversation_text or ""
    filled = _get_filled_slots_summary(state, conversation_text)
    primary_slot = validation.missing_required_slots[0] if validation.missing_required_slots else None

    lines: List[str] = []
    if filled:
        lines.append(
            "Pour cibler la bonne documentation technique, j'ai besoin de quelques précisions."
        )
        lines.append("")
        lines.append("Ce que j'ai déjà compris :")
        for label, value in filled:
            lines.append(f"  • {label} : {value}")
        lines.append("")
    else:
        lines.append(
            "Pour lancer une recherche précise dans la documentation, j'ai besoin de quelques informations."
        )
        lines.append("")

    if validation.clarification_questions:
        lines.append(validation.clarification_questions[0])
        if primary_slot and primary_slot in CLARIFICATION_HINTS:
            lines.append("")
            lines.append(CLARIFICATION_HINTS[primary_slot])

    optional_hint = _build_optional_precision_hint(state, validation.missing_required_slots)
    if optional_hint:
        lines.append("")
        lines.append(optional_hint)

    return "\n".join(lines)


def build_slot_context_for_generation(state: SlotState) -> str:
    """Bloc structuré injecté dans le prompt de génération."""
    filled = state.model_dump_filled()
    if not filled:
        return ""
    lines = ["CRITÈRES VALIDÉS POUR LA RECHERCHE (à respecter strictement) :"]
    for key, value in filled.items():
        if key == "confidence":
            continue
        if key in SLOT_SUMMARY_LABELS:
            lines.append(f"- {SLOT_SUMMARY_LABELS[key]}: {_format_slot_display_value(key, value)}")
        else:
            lines.append(f"- {key}: {value}")

    material = state.material
    if material and material in MATERIAL_GENERATION_CONSTRAINTS:
        lines.append("")
        lines.append(f"CONTRAINTE MATÉRIAU : {MATERIAL_GENERATION_CONSTRAINTS[material]}")
    lines.append("")
    lines.append(
        "CONTRAINTE COTES : Ne cite aucune valeur en mm sauf si elle apparaît "
        "textuellement dans le passage cité. Si la cote est uniquement sur un schéma "
        "image, réponds « Voir schéma page X » sans inventer de chiffre."
    )
    return "\n".join(lines)


def slot_state_to_sources_payload(validation: SlotValidationResult) -> str:
    """Sérialise l'état slots pour persistance dans Message.sources."""
    return json.dumps(
        {
            "need_clarification": True,
            "slot_state": validation.slot_state.model_dump(),
            "missing_required_slots": validation.missing_required_slots,
            "suggested_options": validation.suggested_options,
            "clarification_actions": validation.clarification_actions,
        },
        ensure_ascii=False,
    )


def load_slot_state_from_session(session: Session, conversation_id: int) -> Optional[SlotState]:
    """Charge le dernier état slots depuis un message assistant de clarification."""
    rows = session.exec(
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.role == "assistant")
        .order_by(Message.id.desc())
        .limit(8)
    ).all()
    for row in rows:
        if not row.sources:
            continue
        try:
            data = json.loads(row.sources)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("need_clarification") and data.get("slot_state"):
            try:
                return SlotState.model_validate(data["slot_state"])
            except Exception:
                continue
    return None


async def process_slot_filling(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    previous_state: Optional[SlotState] = None,
    *,
    use_llm: bool = True,
) -> SlotValidationResult:
    """
    Pipeline complet : extraction -> merge -> validation -> canonical_query.
    """
    heuristic = extract_slots_heuristic(query, history)
    extracted = heuristic

    if use_llm and settings.MISTRAL_API_KEY and settings.LLM_PROVIDER != "ollama":
        llm_state = await extract_slots_llm(query, history)
        if llm_state is not None:
            extracted = merge_slot_states(heuristic, llm_state)

    merged = merge_slot_states(previous_state, extracted)
    conversation_text = collect_user_messages(history, query)
    updates = slot_state_sanitization_updates(
        context_usage=merged.context_usage,
        component_part=merged.component_part,
        material=merged.material,
        product_type=merged.type,
        conversation_text=conversation_text,
    )
    if updates:
        merged = merged.model_copy(update=updates)
    result = validate_slots(merged, conversation_text=conversation_text)
    return result.model_copy(update={"conversation_text": conversation_text})
