"""Résolveur de périmètre de recherche — décide QUOI filtrer et OÙ chercher AVANT le retrieval.

Étape déterministe (0 appel LLM) qui, à partir des signaux de la compréhension de requête
et des statistiques de classification de l'espace, statue par champ :

  CERTAIN        — valeur connue (explicite dans la requête, héritée du fil, ou mono-valeur
                   dans l'espace) → appliquée au filtre sans rien demander ;
  TO_CONFIRM     — valeur inférée (pré-cochée dans la carte, à valider) ;
  TO_ASK         — valeur absente ET champ discriminant ET « grand-public » → proposée ;
  NON_PERTINENT  — champ non discriminant, non détecté (expert), ou intent comparatif.

Sortie : une décision globale (RETRIEVE_DIRECT / PROPOSE_CARD) + le périmètre résolu
(champs CERTAIN à appliquer) + éventuellement une carte de confirmation.

Le cœur (`resolve_scope`, `build_scope_card`) est PUR (aucune DB) → testable en isolation.
La lecture des stats d'espace (`compute_space_scope_stats`) est séparée et cachée.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from app.config import settings
from app.services.slot_catalog import (
    MATERIAL_CHOICES,
    PRODUCT_FAMILY_CHOICES,
    PRODUCT_RANGE_CHOICES,
    SUPPLIER_CHOICES,
)

logger = logging.getLogger(__name__)

# Statuts par champ.
CERTAIN = "certain"
TO_CONFIRM = "to_confirm"
TO_ASK = "to_ask"
NON_PERTINENT = "non_pertinent"

# Décisions globales.
RETRIEVE_DIRECT = "retrieve_direct"
PROPOSE_CARD = "propose_card"

# Intents pour lesquels la famille de produit ne doit PAS être filtrée (comparaison de
# gammes/types → l'utilisateur veut justement traverser les familles).
_COMPARATIVE_INTENTS = frozenset({"product_selection"})


@dataclass(frozen=True)
class ScopeFieldSpec:
    """Description d'un champ de périmètre : lien signal ↔ document ↔ vocabulaire."""

    key: str                 # clé canonique du périmètre
    signal_attr: str         # attribut correspondant sur LightweightQuerySignals
    document_field: str      # colonne Document filtrée au retrieval
    choices: Dict[str, str]  # slug → libellé
    values_are_slugs: bool   # True: valeurs = slugs (product_types…) ; False: labels (source)


# Ordre = ordre d'affichage dans la carte.
SCOPE_FIELD_SPECS: List[ScopeFieldSpec] = [
    ScopeFieldSpec("product_family", "product_family", "product_types", PRODUCT_FAMILY_CHOICES, True),
    ScopeFieldSpec("material", "material_hint", "materials", MATERIAL_CHOICES, True),
    ScopeFieldSpec("product_range", "product_range", "proferm_gammes", PRODUCT_RANGE_CHOICES, True),
    ScopeFieldSpec("supplier", "primary_source", "source", SUPPLIER_CHOICES, False),
]

_SPEC_BY_KEY: Dict[str, ScopeFieldSpec] = {s.key: s for s in SCOPE_FIELD_SPECS}


def _label_for(spec: ScopeFieldSpec, value: str) -> str:
    """Libellé d'affichage d'une valeur de champ."""
    if spec.values_are_slugs:
        return spec.choices.get(value, value)
    return value  # source : déjà sous forme de libellé canonique


@dataclass
class FieldResolution:
    key: str
    status: str
    value: Optional[str] = None        # valeur retenue (CERTAIN/TO_CONFIRM)
    options: List[str] = field(default_factory=list)  # valeurs proposées (TO_ASK), présentes dans l'espace


@dataclass
class ScopeResolution:
    decision: str
    fields: List[FieldResolution]
    # Périmètre à appliquer au retrieval : clé de champ → valeur (CERTAIN + TO_CONFIRM).
    applied_scope: Dict[str, str] = field(default_factory=dict)

    def asked_fields(self) -> List[FieldResolution]:
        return [f for f in self.fields if f.status in (TO_ASK, TO_CONFIRM)]


def resolve_scope(
    signals: Dict[str, Optional[str]],
    *,
    space_stats: Dict[str, Set[str]],
    inherited_scope: Optional[Dict[str, str]] = None,
    topic_shift: bool = False,
    intent: Optional[str] = None,
    mode: Optional[str] = None,
    askable_fields: Optional[List[str]] = None,
    min_distinct_to_ask: Optional[int] = None,
) -> ScopeResolution:
    """Décision de périmètre (PUR, sans DB).

    - ``signals`` : dict des signaux (clés = attributs LightweightQuerySignals).
    - ``space_stats`` : par clé de champ, ensemble des valeurs distinctes présentes dans
      l'espace (déjà normalisées : slugs pour product_types/materials/proferm_gammes,
      libellés pour source).
    - ``inherited_scope`` : périmètre confirmé au tour précédent (query_context.current_scope).
    """
    mode = (mode or settings.SCOPE_MODE or "off").lower()
    askable = set(askable_fields if askable_fields is not None else settings.scope_askable_fields)
    min_distinct = (
        min_distinct_to_ask if min_distinct_to_ask is not None else settings.SCOPE_MIN_DISTINCT_TO_ASK
    )
    inherited = {} if (topic_shift or not inherited_scope) else dict(inherited_scope)
    intent_norm = (intent or "").strip().lower()

    resolutions: List[FieldResolution] = []
    applied: Dict[str, str] = {}

    for spec in SCOPE_FIELD_SPECS:
        present = {v for v in (space_stats.get(spec.key) or set()) if v}
        signal_value = _clean(signals.get(spec.signal_attr))
        inherited_value = _clean(inherited.get(spec.key))

        # Champ non pertinent si l'espace ne contient rien à filtrer dessus.
        if not present:
            resolutions.append(FieldResolution(spec.key, NON_PERTINENT))
            continue

        # 1) Hérité du fil (sujet inchangé) → CERTAIN.
        if inherited_value and inherited_value in present:
            resolutions.append(FieldResolution(spec.key, CERTAIN, value=inherited_value))
            applied[spec.key] = inherited_value
            continue

        # 2) Détecté dans la requête → CERTAIN si présent dans l'espace.
        if signal_value and signal_value in present:
            resolutions.append(FieldResolution(spec.key, CERTAIN, value=signal_value))
            applied[spec.key] = signal_value
            continue
        if signal_value and signal_value not in present:
            # Détecté mais absent de l'espace : ne pas filtrer sur une valeur qui viderait
            # tout (laisser le retrieval + l'élargissement auto gérer). Non bloquant.
            resolutions.append(FieldResolution(spec.key, NON_PERTINENT, value=signal_value))
            continue

        # 3) Mono-valeur dans l'espace → CERTAIN (ex. espace mono-fournisseur).
        if len(present) == 1:
            only = next(iter(present))
            resolutions.append(FieldResolution(spec.key, CERTAIN, value=only))
            applied[spec.key] = only
            continue

        # 4) Absent + discriminant + demandable + hors intent comparatif → TO_ASK.
        is_comparative_family = spec.key == "product_family" and intent_norm in _COMPARATIVE_INTENTS
        if (
            spec.key in askable
            and len(present) >= min_distinct
            and not is_comparative_family
        ):
            resolutions.append(
                FieldResolution(spec.key, TO_ASK, options=sorted(present))
            )
            continue

        # 5) Sinon non pertinent (champ expert non détecté, ou non discriminant).
        resolutions.append(FieldResolution(spec.key, NON_PERTINENT))

    asked = [r for r in resolutions if r.status in (TO_ASK, TO_CONFIRM)]
    if mode == "confirm" and asked:
        decision = PROPOSE_CARD
    else:
        # off / auto / rien à demander → retrieval direct avec les champs CERTAIN.
        decision = RETRIEVE_DIRECT

    return ScopeResolution(decision=decision, fields=resolutions, applied_scope=applied)


def build_scope_card(resolution: ScopeResolution) -> Dict:
    """Construit le payload de la carte de confirmation (événement SSE `scope_proposal`)."""
    fields_payload: List[Dict] = []
    for res in resolution.fields:
        spec = _SPEC_BY_KEY[res.key]
        if res.status == TO_ASK:
            fields_payload.append(
                {
                    "key": res.key,
                    "label": _field_label(res.key),
                    "status": res.status,
                    "options": [
                        {"value": v, "label": _label_for(spec, v)} for v in res.options
                    ],
                }
            )
        elif res.status in (CERTAIN, TO_CONFIRM) and res.value:
            fields_payload.append(
                {
                    "key": res.key,
                    "label": _field_label(res.key),
                    "status": res.status,
                    "value": res.value,
                    "value_label": _label_for(spec, res.value),
                }
            )
    return {"type": "scope_proposal", "fields": fields_payload}


_FIELD_LABELS: Dict[str, str] = {
    "product_family": "Type de produit",
    "material": "Matériau",
    "product_range": "Gamme",
    "supplier": "Fournisseur",
}


def _field_label(key: str) -> str:
    return _FIELD_LABELS.get(key, key)


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip()
    return v or None


# --------------------------------------------------------------------------- #
# Statistiques de classification par espace (DB, cachées).
# --------------------------------------------------------------------------- #
# Cache plat par espace : {space_id: (expiry_monotonic, {field_key: {values}})}.
_stats_cache: Dict[int, tuple] = {}


def invalidate_space_scope_stats(space_id: Optional[int] = None) -> None:
    """À appeler après (dé)association ou reclassification d'un document."""
    if space_id is None:
        _stats_cache.clear()
    else:
        _stats_cache.pop(int(space_id), None)


def compute_space_scope_stats(session, space_id: int) -> Dict[str, Set[str]]:
    """Valeurs de classification distinctes présentes dans l'espace, par champ de périmètre.

    S'appuie sur les colonnes ARRAY (product_types/materials/proferm_gammes) et scalaire
    (source) de Document, restreintes aux documents associés à l'espace. Mise en cache TTL.
    """
    ttl = settings.SCOPE_STATS_CACHE_TTL
    now = time.monotonic()
    if ttl > 0:
        cached = _stats_cache.get(space_id)
        if cached and cached[0] > now:
            return cached[1]

    stats: Dict[str, Set[str]] = {spec.key: set() for spec in SCOPE_FIELD_SPECS}
    try:
        from sqlalchemy import text
        from sqlmodel import Session

        from app.database import engine

        # Session FRAÎCHE isolée : une lecture qui échouerait ne doit jamais empoisonner
        # la transaction de la requête appelante (InFailedSqlTransaction).
        with Session(engine) as s:
            rows = s.execute(
                text(
                    """
                    SELECT d.product_types, d.materials, d.proferm_gammes, d.source
                    FROM document d
                    JOIN document_space ds ON ds.document_id = d.id
                    WHERE ds.space_id = :space_id
                    """
                ),
                {"space_id": space_id},
            ).all()
    except Exception as exc:  # pragma: no cover - dépend du schéma runtime
        logger.warning("[scope] stats espace %s indisponibles: %s", space_id, exc)
        rows = []

    for product_types, materials, proferm_gammes, source in rows:
        for v in product_types or []:
            if v:
                stats["product_family"].add(str(v).strip().lower())
        for v in materials or []:
            if v:
                stats["material"].add(str(v).strip().lower())
        for v in proferm_gammes or []:
            if v:
                stats["product_range"].add(str(v).strip().lower())
        if source and str(source).strip():
            stats["supplier"].add(str(source).strip())

    if ttl > 0:
        _stats_cache[space_id] = (now + ttl, stats)
    return stats


def compute_allowed_document_ids(session, space_id: int, applied_scope: Dict[str, str]):
    """Documents de l'espace matchant le périmètre, avec POLITIQUE WILDCARD.

    Règle wildcard : un document NON classé sur un champ n'est JAMAIS exclu par ce champ
    (inclusion par défaut). Sinon un filtre dur tuerait le rappel en silence sur tout
    document mal paramétré. Un document ne matche que si, pour CHAQUE champ du périmètre,
    il est soit non classé sur ce champ, soit porte la valeur demandée.

    Retourne ``None`` si ``applied_scope`` est vide (⇒ aucun filtrage). Retourne une liste
    (éventuellement vide) sinon.
    """
    applied = {k: v for k, v in (applied_scope or {}).items() if v}
    if not applied:
        return None

    # (document_field, valeur attendue, valeurs_sont_slugs) par champ appliqué.
    constraints = []
    for key, value in applied.items():
        spec = _SPEC_BY_KEY.get(key)
        if spec is None:
            continue
        constraints.append((spec.document_field, value, spec.values_are_slugs))

    try:
        from sqlalchemy import text
        from sqlmodel import Session

        from app.database import engine

        with Session(engine) as s:
            rows = s.execute(
                text(
                    """
                    SELECT d.id, d.product_types, d.materials, d.proferm_gammes, d.source
                    FROM document d
                    JOIN document_space ds ON ds.document_id = d.id
                    WHERE ds.space_id = :space_id
                    """
                ),
                {"space_id": space_id},
            ).all()
    except Exception as exc:  # pragma: no cover - dépend du schéma runtime
        logger.warning("[scope] allowed_document_ids espace %s indisponible: %s", space_id, exc)
        return None

    by_field = {
        "product_types": lambda r: r[1],
        "materials": lambda r: r[2],
        "proferm_gammes": lambda r: r[3],
        "source": lambda r: r[4],
    }

    allowed: List[int] = []
    wildcard_hits = 0
    for row in rows:
        doc_id = row[0]
        matches = True
        used_wildcard = False
        for document_field, expected, values_are_slugs in constraints:
            raw = by_field[document_field](row)
            if values_are_slugs:
                values = {str(v).strip().lower() for v in (raw or []) if v}
            else:
                values = {str(raw).strip()} if raw and str(raw).strip() else set()
            if not values:
                used_wildcard = True  # non classé sur ce champ → wildcard, non exclu
                continue
            if expected not in values:
                matches = False
                break
        if matches:
            allowed.append(int(doc_id))
            if used_wildcard:
                wildcard_hits += 1

    logger.info(
        "[scope] périmètre %s → %d documents (dont %d par wildcard) dans l'espace %s",
        applied,
        len(allowed),
        wildcard_hits,
        space_id,
    )
    return allowed
