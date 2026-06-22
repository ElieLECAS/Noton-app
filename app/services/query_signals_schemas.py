"""Schémas Pydantic pour l'extraction légère de signaux de requête (entités ouvertes, catégories contraintes)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Self

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from sqlmodel import Session

from app.services.category_catalog import CONTENT_CATEGORY_SLUGS, get_active_categories, get_active_categories_for_prompt
from app.services.slot_catalog import INTENT_CHOICES, MATERIAL_CHOICES, SUPPLIER_CHOICES, supplier_slug_to_source

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
    detected_references: List[str] = Field(default_factory=list)
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


class LightweightQuerySignals(BaseModel):
    intent: Optional[str] = None
    entities: List[ExtractedEntity] = Field(default_factory=list)
    entity_texts: List[str] = Field(default_factory=list)
    inferred_categories: List[str] = Field(default_factory=list)
    primary_source: Optional[str] = None
    material_hint: Optional[str] = None
    detected_references: List[str] = Field(default_factory=list)
    confidence: float = 0.8


def get_allowed_category_slugs(session: Optional[Session] = None) -> frozenset[str]:
    if session is not None:
        rows = get_active_categories(session)
        if rows:
            return frozenset(r.slug for r in rows)
    return frozenset(CONTENT_CATEGORY_SLUGS)


def build_extract_signals_prompt(session: Session) -> str:
    category_catalog_json = get_active_categories_for_prompt(session)
    intent_values = " | ".join(INTENT_CHOICES.keys())
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

6. DETECTED_REFERENCES : codes, modèles, normes cités sous forme exacte (ex. "Perform 70").

RÈGLES :
- entities : privilégier le rappel (mieux vaut en extraire trop que pas assez)
- inferred_categories : strictement limité au catalogue fourni ; ne jamais inventer de slug
- confidence : 0.0-1.0 selon clarté globale de la demande

Retourne UNIQUEMENT un JSON :
{{
  "intent": null,
  "entities": [{{"text": "...", "role": null, "confidence": 0.9}}],
  "inferred_categories": [],
  "primary_source": null,
  "material_hint": null,
  "detected_references": [],
  "confidence": 0.8
}}
"""


def parse_and_validate_signals(raw: dict, *, session: Optional[Session] = None) -> QuerySignalsExtraction:
    allowed = get_allowed_category_slugs(session)
    return QuerySignalsExtraction.model_validate(
        raw,
        context={"allowed_category_slugs": allowed},
    )


def to_lightweight_signals(extraction: QuerySignalsExtraction) -> LightweightQuerySignals:
    """Convertit l'extraction validée en signaux pipeline avec entity_texts dédupliqués."""
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

    return LightweightQuerySignals(
        intent=extraction.intent,
        entities=extraction.entities,
        entity_texts=entity_texts,
        inferred_categories=extraction.inferred_categories,
        primary_source=extraction.primary_source,
        material_hint=extraction.material_hint,
        detected_references=extraction.detected_references,
        confidence=extraction.confidence,
    )
