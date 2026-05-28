import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from app.config import settings
from app.services.mistral_service import chat

logger = logging.getLogger(__name__)

class QueryIntent(BaseModel):
    intent: str  # 'company_info', 'supplier_info', 'generic', 'mixed'
    primary_source: Optional[str] = None  # 'Proferm', 'Technal', 'Profine', etc.
    reasoning: str
    confidence: float
    search_terms: List[str] = []  # Termes de recherche additionnels pour query expansion
    detected_references: List[str] = []  # Références techniques exactes identifiées (CQR)

SYSTEM_PROMPT = """Tu es un expert en analyse d'intention pour un système RAG industriel (PROFERM).
Ton rôle est de décoder la question de l'utilisateur pour déterminer quelle source documentaire doit être privilégiée.

SOURCES POSSIBLES :
- 'Proferm' : La documentation interne de l'entreprise (produits 'maison', procédures internes, vos gammes).
- 'Technal', 'Profine', 'Askey', 'Roto', 'Somfy', 'Maco' : Les fournisseurs et partenaires.

LOGIQUE DE DÉCISION :
1. Si l'utilisateur utilise des adjectifs possessifs ('vos' gammes, 'votre' catalogue, chez 'vous'), l'intention est 'Proferm'.
2. Si l'utilisateur cite une marque spécifique ('catalogue Technal', 'dormant Profine'), la source primaire est cette marque.
3. Attention aux numéros de gammes : Si l'utilisateur mentionne 'Perform 70' ou 'Perform 76', note bien cette distinction dans le raisonnement pour orienter le filtrage.
4. Si la question est générique ('comment poser une fenêtre', 'norme DTU'), l'intention est 'generic' et aucune source n'est privilégiée.
5. En cas de doute entre Proferm et un fournisseur sur un produit générique, privilégie TOUJOURS 'Proferm'.

TERMES DE RECHERCHE ADDITIONNELS :
Génère une liste de 3-6 termes ou expressions clés pertinents pour enrichir la recherche.
Inclure : synonymes techniques, termes associés, abréviations, noms complets si abrégé, et inversement.
Exemple : pour 'dormant Profine', ajouter ['profilé PVC', 'menuiserie PVC', 'châssis', 'Profine Systems'].
Exemple : pour 'DTU 36.5', ajouter ['norme', 'étanchéité', 'menuiserie extérieure', 'mise en oeuvre'].
Ne PAS répéter les termes déjà présents dans la question originale.

RÉFÉRENCES ET CODES TECHNIQUES :
Identifie et extrait sous leur forme exacte toutes les références techniques, codes de produits, modèles de profilés, ou normes cités dans la question (ex: "Perform 70", "DTU 36.5", "NF EN 1991", "Soleal 55", "REF123", "PVC-76").

RETOURNE UNIQUEMENT UN JSON avec les champs :
- intent: (company_info | supplier_info | generic | mixed)
- primary_source: (Le nom exact de la marque ou null)
- reasoning: (Explication courte en français)
- confidence: (0.0 à 1.0)
- search_terms: (Liste de 3-6 termes additionnels pour la recherche, ou liste vide si non pertinent)
- detected_references: (Liste de toutes les références techniques exactes, ou liste vide si aucune)
"""

async def reason_query_intent(query: str, history: Optional[List[Dict[str, str]]] = None) -> QueryIntent:
    """
    Analyse l'intention de la requête utilisateur pour orienter la recherche.
    """
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        
        # On pourrait ajouter l'historique ici pour le contexte, mais pour l'instant on reste sur la query
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

        raw_detected_refs = data.get("detected_references", [])
        if not isinstance(raw_detected_refs, list):
            raw_detected_refs = []
        detected_references = [str(r).strip() for r in raw_detected_refs if r and str(r).strip()]

        return QueryIntent(
            intent=data.get("intent", "generic"),
            primary_source=data.get("primary_source"),
            reasoning=data.get("reasoning", "Défaut"),
            confidence=data.get("confidence", 0.5),
            search_terms=search_terms,
            detected_references=detected_references,
        )
    except Exception as e:
        logger.error(f"Erreur lors du raisonnement de la requête: {e}")
        return QueryIntent(intent="generic", reasoning="Erreur technique", confidence=0.0, search_terms=[], detected_references=[])
