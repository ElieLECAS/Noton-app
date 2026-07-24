"""Schémas Pydantic pour l'extraction légère de signaux de requête (entités ouvertes, catégories contraintes)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Self

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from sqlmodel import Session

from app.services.category_catalog import (
    CONTENT_CATEGORY_SLUGS,
    DOC_TYPE_LABELS,
    LIFECYCLE_PHASE_LABELS,
    SYMPTOM_LABELS,
    get_active_categories,
    get_active_categories_for_prompt,
)
from app.services.slot_catalog import (
    INTENT_CHOICES,
    MATERIAL_CHOICES,
    PRODUCT_FAMILY_CHOICES,
    PRODUCT_RANGE_CHOICES,
    SUPPLIER_CHOICES,
    supplier_slug_to_source,
)

logger = logging.getLogger(__name__)

QueryIntentSlug = Literal[
    "specification",
    "installation",
    "regulatory",
    "product_selection",
    "troubleshooting",
    "documentation",
]

MaterialSlug = Literal["pvc", "aluminium", "hybride"]

_MAX_INFERRED_CATEGORIES = 3

_MATERIAL_ALIASES: Dict[str, str] = {
    "alu": "aluminium",
    "aluminium": "aluminium",
    "pvc": "pvc",
    "hybride": "hybride",
}

# Alias vers les slugs de PRODUCT_FAMILY_CHOICES (singulier/pluriel, accents).
_PRODUCT_FAMILY_ALIASES: Dict[str, str] = {
    "coulissant": "coulissants",
    "coulissants": "coulissants",
    "coulisse": "coulissants",
    "coulisses": "coulissants",
    "galandage": "coulissants",
    "baie": "coulissants",
    "baie coulissante": "coulissants",
    "fenetre": "fenetres",
    "fenetres": "fenetres",
    "fenêtre": "fenetres",
    "fenêtres": "fenetres",
    "porte": "portes",
    "portes": "portes",
    "porte-fenetre": "portes",
    "porte-fenêtre": "portes",
}


class ExtractedEntity(BaseModel):
    """Terme tel que présent ou implicite dans la question — pas de type imposé."""

    text: str = Field(min_length=1, description="Forme exacte ou normalisée légère")
    role: Optional[str] = Field(
        default=None,
        description="Hint libre optionnel : product, component, norm, material, process, other",
    )
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()


class QuerySignalsExtraction(BaseModel):
    intent: Optional[QueryIntentSlug] = None
    entities: List[ExtractedEntity] = Field(default_factory=list)
    inferred_categories: List[str] = Field(default_factory=list)
    primary_source: Optional[str] = None
    material_hint: Optional[str] = None
    # Périmètre de recherche : famille de produit (coulissants/fenetres/portes) et
    # gamme Proferm — matchés contre Document.product_types / proferm_gammes au retrieval.
    product_family: Optional[str] = None
    product_range: Optional[str] = None
    detected_references: List[str] = Field(default_factory=list)
    # Facettes multi-axes (validées contre les axes correspondants)
    detected_symptom: Optional[str] = None
    lifecycle_phase: Optional[str] = None
    doc_type_hint: Optional[str] = None
    symptom_freeform: Optional[str] = None  # symptôme hors-liste (non boosté)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    @field_validator("entities", mode="before")
    @classmethod
    def coerce_entities(cls, v: Any) -> Any:
        if not v:
            return []
        result = []
        for item in v:
            if isinstance(item, str) and item.strip():
                result.append({"text": item.strip()})
            elif isinstance(item, dict) and item.get("text"):
                result.append(item)
        return result

    @field_validator("intent", mode="before")
    @classmethod
    def validate_intent(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        slug = str(v).strip().lower()
        return slug if slug in INTENT_CHOICES else None

    @field_validator("material_hint", mode="before")
    @classmethod
    def normalize_material(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        normalized = _MATERIAL_ALIASES.get(str(v).strip().lower())
        return normalized if normalized in MATERIAL_CHOICES else None

    @field_validator("product_family", mode="before")
    @classmethod
    def normalize_product_family(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        raw = str(v).strip().lower()
        normalized = _PRODUCT_FAMILY_ALIASES.get(raw, raw)
        return normalized if normalized in PRODUCT_FAMILY_CHOICES else None

    @field_validator("product_range", mode="before")
    @classmethod
    def normalize_product_range(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        slug = str(v).strip().lower()
        if slug in PRODUCT_RANGE_CHOICES:
            return slug
        # Tolère le libellé ("Perform" → "perform").
        for key, label in PRODUCT_RANGE_CHOICES.items():
            if label.lower() == slug:
                return key
        return None

    @field_validator("primary_source", mode="before")
    @classmethod
    def normalize_source(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        slug = str(v).strip().lower()
        if slug in SUPPLIER_CHOICES:
            return supplier_slug_to_source(slug)
        for key, label in SUPPLIER_CHOICES.items():
            if label.lower() == slug:
                return label
        return str(v).strip().capitalize()

    @field_validator("detected_references", mode="before")
    @classmethod
    def coerce_references(cls, v: Any) -> List[str]:
        if not v:
            return []
        if isinstance(v, list):
            return [str(r).strip() for r in v if r and str(r).strip()]
        return [str(v).strip()] if str(v).strip() else []

    @model_validator(mode="after")
    def sanitize_categories(self, info: ValidationInfo) -> Self:
        allowed: frozenset[str] = info.context.get("allowed_category_slugs") or frozenset()
        if not allowed:
            return self

        valid: List[str] = []
        dropped: List[str] = []
        for slug in self.inferred_categories:
            normalized = slug.strip().lower()
            if normalized in allowed:
                valid.append(normalized)
            else:
                dropped.append(slug)

        if dropped:
            logger.warning("[query_signals] catégories rejetées (hors catalogue): %s", dropped)

        self.inferred_categories = list(dict.fromkeys(valid))[:_MAX_INFERRED_CATEGORIES]
        return self

    @model_validator(mode="after")
    def sanitize_axis_facets(self, info: ValidationInfo) -> Self:
        ctx = info.context or {}

        def _norm(value: Optional[str]) -> Optional[str]:
            if not value:
                return None
            return str(value).strip().lower().replace(" ", "_") or None

        doc_allowed: frozenset[str] = ctx.get("allowed_doc_type_slugs") or frozenset()
        life_allowed: frozenset[str] = ctx.get("allowed_lifecycle_slugs") or frozenset()
        symptom_allowed: frozenset[str] = ctx.get("allowed_symptom_slugs") or frozenset()

        dt = _norm(self.doc_type_hint)
        self.doc_type_hint = dt if (dt and (not doc_allowed or dt in doc_allowed)) else None

        lp = _norm(self.lifecycle_phase)
        self.lifecycle_phase = lp if (lp and (not life_allowed or lp in life_allowed)) else None

        sym = _norm(self.detected_symptom)
        if sym and symptom_allowed and sym not in symptom_allowed:
            # Symptôme hors-liste → conservé en freeform (non boosté), pas en detected_symptom.
            if not self.symptom_freeform:
                self.symptom_freeform = self.detected_symptom
            sym = None
        self.detected_symptom = sym
        if self.symptom_freeform:
            self.symptom_freeform = str(self.symptom_freeform).strip() or None
        return self


class LightweightQuerySignals(BaseModel):
    intent: Optional[str] = None
    entities: List[ExtractedEntity] = Field(default_factory=list)
    entity_texts: List[str] = Field(default_factory=list)
    inferred_categories: List[str] = Field(default_factory=list)
    primary_source: Optional[str] = None
    material_hint: Optional[str] = None
    product_family: Optional[str] = None
    product_range: Optional[str] = None
    detected_references: List[str] = Field(default_factory=list)
    detected_symptom: Optional[str] = None
    lifecycle_phase: Optional[str] = None
    doc_type_hint: Optional[str] = None
    symptom_freeform: Optional[str] = None
    confidence: float = 0.8


def get_allowed_category_slugs(session: Optional[Session] = None) -> frozenset[str]:
    if session is not None:
        rows = get_active_categories(session)
        if rows:
            return frozenset(r.slug for r in rows)
    return frozenset(CONTENT_CATEGORY_SLUGS)


def get_allowed_slugs_context(session: Optional[Session] = None) -> Dict[str, frozenset]:
    """Contexte de validation par axe pour QuerySignalsExtraction."""
    from app.services.category_catalog import (
        AXIS_DOC_TYPE,
        AXIS_LIFECYCLE_PHASE,
        AXIS_SYMPTOM,
        get_allowed_slugs_by_axis,
    )

    by_axis = get_allowed_slugs_by_axis(session)
    all_slugs = frozenset().union(*by_axis.values()) if by_axis else get_allowed_category_slugs(session)
    return {
        "allowed_category_slugs": all_slugs,
        "allowed_doc_type_slugs": by_axis.get(AXIS_DOC_TYPE, frozenset()),
        "allowed_lifecycle_slugs": by_axis.get(AXIS_LIFECYCLE_PHASE, frozenset()),
        "allowed_symptom_slugs": by_axis.get(AXIS_SYMPTOM, frozenset()),
    }


def _format_axis_vocab(labels: Dict[str, str]) -> str:
    return " | ".join(f"{slug} ({label})" for slug, label in labels.items())


def build_extract_signals_prompt(session: Session) -> str:
    category_catalog_json = get_active_categories_for_prompt(session)
    intent_values = " | ".join(INTENT_CHOICES.keys())
    product_family_values = _format_axis_vocab(PRODUCT_FAMILY_CHOICES)
    product_range_values = _format_axis_vocab(PRODUCT_RANGE_CHOICES)
    symptom_values = _format_axis_vocab(SYMPTOM_LABELS)
    lifecycle_values = _format_axis_vocab(LIFECYCLE_PHASE_LABELS)
    doc_type_values = _format_axis_vocab(DOC_TYPE_LABELS)
    return f"""Tu es un analyseur de requêtes pour un assistant documentaire menuiserie (PROFERM).

Extrais les signaux suivants du message utilisateur.

1. INTENT (optionnel, une seule valeur si déductible) :
   {intent_values}

2. ENTITIES (liste ouverte — IMPORTANT) :
   Extrais TOUS les termes techniques, noms propres, composants, gammes, matériaux,
   normes, processus ou concepts métier présents ou clairement implicites dans la question.
   - PAS de liste fermée : si l'utilisateur dit "coulisses", "MONOBLOC", "ALU", "Perform",
     "DTU 36.5", "seuil PMR", etc. → extrais-les.
   - Conserve la forme la plus utile pour la recherche (casse, abréviations).
   - Ne limite pas aux exemples ci-dessus.
   - N'invente rien absent du message.
   - Champ role optionnel (hint libre, pas contraint).
   - Format : liste d'objets {{"text": "...", "role": null ou hint, "confidence": 0.0-1.0}}

3. INFERRED_CATEGORIES (0 à {_MAX_INFERRED_CATEGORIES} slugs MAX, UNIQUEMENT parmi le catalogue ci-dessous) :
   Choisis les catégories de contenu documentaire les plus pertinentes pour orienter la recherche.
   Si aucune ne convient → [].
   Catalogue autorisé (slug + description) :
   {category_catalog_json}

4. PRIMARY_SOURCE (optionnel) : marque/fournisseur explicitement cité, sinon null.

5. MATERIAL_HINT (optionnel) : pvc | aluminium | hybride si mentionné, sinon null.

5b. PRODUCT_FAMILY (optionnel) : famille de produit UNIQUEMENT si déductible de la question,
   UN slug parmi : {product_family_values}. « coulissant/coulisse/galandage/baie » ⇒ coulissants ;
   « fenêtre » ⇒ fenetres ; « porte » ⇒ portes. Sinon null. NE PAS deviner si le type
   d'ouverture n'est pas clair (« frappe » seul est ambigu → null).

5c. PRODUCT_RANGE (optionnel) : gamme Proferm citée, UN slug parmi : {product_range_values}.
   Sinon null. NE PAS inférer depuis un code fournisseur.

6. DETECTED_REFERENCES : codes, modèles, normes cités sous forme exacte (ex. "Perform 70").

7. DETECTED_SYMPTOM (optionnel) : si la question décrit un PROBLÈME/SYMPTÔME SAV, choisis UN slug
   STRICTEMENT parmi : {symptom_values}. Sinon null. Si le symptôme n'existe pas dans cette liste,
   laisse detected_symptom=null et mets le libellé court dans symptom_freeform.

8. LIFECYCLE_PHASE (optionnel) : phase du cycle de vie, UN slug parmi : {lifecycle_values}. Sinon null.

9. DOC_TYPE_HINT (optionnel) : nature de document recherchée, UN slug parmi : {doc_type_values}. Sinon null.

RÈGLES :
- entities : privilégier le rappel (mieux vaut en extraire trop que pas assez)
- inferred_categories : strictement limité au catalogue fourni ; ne jamais inventer de slug
- detected_symptom / lifecycle_phase / doc_type_hint : strictement parmi les listes fournies
- confidence : 0.0-1.0 selon clarté globale de la demande

Retourne UNIQUEMENT un JSON :
{{
  "intent": null,
  "entities": [{{"text": "...", "role": null, "confidence": 0.9}}],
  "inferred_categories": [],
  "primary_source": null,
  "material_hint": null,
  "product_family": null,
  "product_range": null,
  "detected_references": [],
  "detected_symptom": null,
  "lifecycle_phase": null,
  "doc_type_hint": null,
  "symptom_freeform": null,
  "confidence": 0.8
}}
"""


def parse_and_validate_signals(raw: dict, *, session: Optional[Session] = None) -> QuerySignalsExtraction:
    context = get_allowed_slugs_context(session)
    return QuerySignalsExtraction.model_validate(raw, context=context)


def to_lightweight_signals(extraction: QuerySignalsExtraction) -> LightweightQuerySignals:
    """Convertit l'extraction validée en signaux pipeline avec entity_texts dédupliqués.

    Fusionne les facettes d'axe (symptôme, phase, doc_type) dans ``inferred_categories``
    afin que le boost de retrieval (qui matche par slug) les prenne en charge sans
    modification, tout en conservant les champs dédiés pour le matching d'arbres SAV.
    """
    seen: set[str] = set()
    entity_texts: List[str] = []

    for entity in extraction.entities:
        key = entity.text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            entity_texts.append(entity.text.strip())

    for ref in extraction.detected_references:
        key = ref.strip().lower()
        if key and key not in seen:
            seen.add(key)
            entity_texts.append(ref.strip())

    fused_categories = list(extraction.inferred_categories)
    for slug in (extraction.detected_symptom, extraction.lifecycle_phase, extraction.doc_type_hint):
        if slug and slug not in fused_categories:
            fused_categories.append(slug)

    return LightweightQuerySignals(
        intent=extraction.intent,
        entities=extraction.entities,
        entity_texts=entity_texts,
        inferred_categories=fused_categories,
        primary_source=extraction.primary_source,
        material_hint=extraction.material_hint,
        product_family=extraction.product_family,
        product_range=extraction.product_range,
        detected_references=extraction.detected_references,
        detected_symptom=extraction.detected_symptom,
        lifecycle_phase=extraction.lifecycle_phase,
        doc_type_hint=extraction.doc_type_hint,
        symptom_freeform=extraction.symptom_freeform,
        confidence=extraction.confidence,
    )
