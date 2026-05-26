import json
import logging
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from app.config import settings
from app.services.mistral_service import chat

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns de référence produit détectés LOCALEMENT (sans appel LLM)
# Ordre : du plus spécifique au plus générique
# ---------------------------------------------------------------------------
_REF_PATTERNS: List[re.Pattern] = [
    re.compile(r'\bPerform[\s\-]?\d{2,3}\b', re.IGNORECASE),          # Perform 70, Perform-76
    re.compile(r'\b[A-Z]{2,6}[\-\.]?\d{2,4}(?:[\-\.][A-Z0-9]+)?\b'), # TGY-200, SEC.76, T-660
    re.compile(r'\bT[\-\.]?\d{3,4}\b', re.IGNORECASE),                # T-660, T.700
    re.compile(r'\b\d{2,3}[\-\.]\d{1,2}\b'),                          # 36.5, 24-40 (ex: DTU 36.5)
    re.compile(r'\b[A-Z]{1,3}\d{3,4}\b'),                             # SF200, R500
]


def _extract_detected_refs(query: str) -> List[str]:
    """Extrait les références produit détectées localement dans la requête."""
    found: List[str] = []
    for pattern in _REF_PATTERNS:
        for m in pattern.finditer(query):
            ref = m.group(0).strip()
            # Éviter les doublons (insensible à la casse)
            if not any(r.lower() == ref.lower() for r in found):
                found.append(ref)
    return found


class QueryIntent(BaseModel):
    intent: str  # 'company_info', 'supplier_info', 'generic', 'mixed', 'exact_reference'
    primary_source: Optional[str] = None  # 'Proferm', 'Technal', 'Profine', etc.
    reasoning: str
    confidence: float
    search_terms: List[str] = []  # Termes de recherche additionnels pour query expansion
    detected_refs: List[str] = []  # Références produit détectées localement (regex)

SYSTEM_PROMPT = """Tu es un expert en analyse d'intention pour un système RAG industriel (PROFERM).
Ton rôle est de décoder la question de l'utilisateur pour déterminer quelle source documentaire doit être privilégiée.

SOURCES POSSIBLES :
- 'Proferm' : La documentation interne de l'entreprise (produits 'maison', procédures internes, vos gammes).
- 'Technal', 'Profine', 'Askey', 'Roto', 'Somfy', 'Maco' : Les fournisseurs et partenaires.

LOGIQUE DE DÉCISION :
1. Si l'utilisateur utilise des adjectifs possessifs ('vos' gammes, 'votre' catalogue, chez 'vous'), l'intention est 'Proferm'.
2. Si l'utilisateur cite une marque spécifique ('catalogue Technal', 'dormant Profine'), la source primaire est cette marque.
3. Si l'utilisateur mentionne un CODE ou RÉFÉRENCE PRODUIT EXACTE (ex: 'Perform 70', 'Perform 76', 'TGY-200', 'T-660', 'DTU 36.5'), l'intention est 'exact_reference'. Note bien la référence exacte dans le raisonnement — ne jamais la confondre avec une référence similaire.
4. Si la question est générique ('comment poser une fenêtre', 'norme DTU'), l'intention est 'generic' et aucune source n'est privilégiée.
5. En cas de doute entre Proferm et un fournisseur sur un produit générique, privilégie TOUJOURS 'Proferm'.

TERMES DE RECHERCHE ADDITIONNELS :
Génère une liste de 3-6 termes ou expressions clés pertinents pour enrichir la recherche.
Inclure : synonymes techniques, termes associés, abréviations, noms complets si abrégé, et inversement.
Exemple : pour 'dormant Profine', ajouter ['profilé PVC', 'menuiserie PVC', 'châssis', 'Profine Systems'].
Exemple : pour 'DTU 36.5', ajouter ['norme', 'étanchéité', 'menuiserie extérieure', 'mise en oeuvre'].
Ne PAS répéter les termes déjà présents dans la question originale.

RETOURNE UNIQUEMENT UN JSON avec les champs :
- intent: (company_info | supplier_info | generic | mixed | exact_reference)
- primary_source: (Le nom exact de la marque ou null)
- reasoning: (Explication courte en français, cite la référence exacte si intent=exact_reference)
- confidence: (0.0 à 1.0)
- search_terms: (Liste de 3-6 termes additionnels pour la recherche, ou liste vide si non pertinent)
"""

async def reason_query_intent(query: str, history: Optional[List[Dict[str, str]]] = None) -> QueryIntent:
    """
    Analyse l'intention de la requête utilisateur pour orienter la recherche.
    
    La détection de références produit (intent=exact_reference) est d'abord tentée
    localement par regex, SANS appel LLM, pour garantir la robustesse et la rapidité.
    L'appel LLM affine ensuite le résultat.
    """
    # Détection locale des références par regex (zéro coût LLM)
    detected_refs = _extract_detected_refs(query)

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        messages.append({"role": "user", "content": f"Analyse cette requête : '{query}'"})

        response = await chat(
            "",
            model=settings.MODEL_FAST,
            context=messages,
            response_format={"type": "json_object"}
        )

        content = response["choices"][0]["message"].get("content", "{}")
        data = json.loads(content)

        raw_search_terms = data.get("search_terms", [])
        if not isinstance(raw_search_terms, list):
            raw_search_terms = []
        search_terms = [str(t).strip() for t in raw_search_terms if t and str(t).strip()][:6]

        # Si des références ont été détectées localement, forcer intent=exact_reference
        # même si le LLM a retourné un intent différent
        llm_intent = data.get("intent", "generic")
        final_intent = llm_intent
        if detected_refs and llm_intent not in ("exact_reference",):
            final_intent = "exact_reference"
            logger.info(
                "Intent forcé à 'exact_reference' par regex (LLM=%s, refs=%s)",
                llm_intent,
                detected_refs,
            )

        return QueryIntent(
            intent=final_intent,
            primary_source=data.get("primary_source"),
            reasoning=data.get("reasoning", "Défaut"),
            confidence=data.get("confidence", 0.5),
            search_terms=search_terms,
            detected_refs=detected_refs,
        )
    except Exception as e:
        logger.error(f"Erreur lors du raisonnement de la requête: {e}")
        # Fallback : utiliser seulement la détection regex locale
        fallback_intent = "exact_reference" if detected_refs else "generic"
        return QueryIntent(
            intent=fallback_intent,
            reasoning="Erreur technique — détection regex locale uniquement",
            confidence=0.5 if detected_refs else 0.0,
            search_terms=[],
            detected_refs=detected_refs,
        )
