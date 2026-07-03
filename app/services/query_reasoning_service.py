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
            model=settings.MODEL_QUERY_UNDERSTANDING, 
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


class RetrievalDecision(BaseModel):
    decision: str  # 'direct' | 'rag'
    reasoning: str


DECISION_SYSTEM_PROMPT = """Tu es un système d'aiguillage intelligent pour un assistant RAG industriel (PROFERM).
Ton rôle est d'analyser le message de l'utilisateur pour décider s'il est nécessaire de faire une recherche documentaire (RAG) ou s'il faut répondre directement.

LOGIQUE DE DÉCISION :
1. Répondre DIRECTEMENT (decision: 'direct') si :
   - C'est une salutation (ex: "bonjour", "hello", "salut", "bonsoir").
   - C'est un remerciement ou une clôture (ex: "merci", "merci beaucoup", "au revoir", "bye").
   - C'est une question sur ton identité, ton rôle ou tes capacités (ex: "qui es-tu ?", "que peux-tu faire ?", "aide-moi").
   - C'est une phrase de politesse ou de bavardage générique sans lien avec un produit ou document (ex: "comment ça va ?", "bonne journée").
2. Faire appel au RAG (decision: 'rag') si :
   - La question porte sur des caractéristiques techniques, des gammes (ex: "Perform 70", "Perform 76", "Soleal", "Lumeal"), des profilés, des dimensions, des tolérances.
   - La question mentionne un fournisseur ou une marque (ex: Technal, Profine, Somfy, Roto, Maco, Askey, Proferm).
   - La question demande comment poser, monter, régler, réparer ou utiliser un produit ou un composant.
   - La question concerne des normes (ex: DTU, NF EN) ou de la documentation technique.
   - En cas de doute, privilégie TOUJOURS 'rag'.

FORMAT DE RETOUR :
Retourne UNIQUEMENT un objet JSON avec les champs suivants :
- decision: (direct | rag)
- reasoning: (Explication courte en français de la décision)
"""


class GuidedModeDecision(BaseModel):
    is_guided: bool
    flow_kind: str = "howto"  # 'howto' (pose/montage chantier) | 'diagnostic' (SAV)
    topic: str = ""
    detected_symptom: str = ""  # slug axis=symptom si flow_kind=diagnostic, sinon ""
    # Le produit/gamme cible est-il explicitement nommé PAR L'UTILISATEUR (message ou
    # historique utilisateur) ? False → le parcours guidé commence par une étape
    # d'identification du produit au lieu d'en verrouiller un déduit du retrieval.
    # Défaut True = comportement historique (pas d'étape 0) pour les constructions
    # directes ; decide_guided_mode renseigne toujours la valeur explicitement.
    product_named: bool = True
    # L'intention est-elle ambiguë entre un geste (régler/ajuster) et un dimensionnement
    # (définir/choisir une valeur) ? True → on n'entre PAS en guidé : le pipeline one-shot
    # pose la question de clarification.
    needs_intent_clarification: bool = False
    reasoning: str = ""


GUIDED_MODE_SYSTEM_PROMPT = """Tu es un classifieur pour l'assistant technique PROFERM (menuiserie, volets roulants).
Tu détermines si la demande de l'utilisateur nécessite un GUIDAGE PAS-À-PAS interactif (un cheminement
en plusieurs étapes avec aiguillage) plutôt qu'une simple réponse documentaire en un coup.

GUIDAGE = true dans 2 cas :
1. flow_kind = "howto" : l'utilisateur veut être ACCOMPAGNÉ dans un geste sur le chantier
   (ex : « comment poser ce seuil ? », « comment monter l'embout sur le profil alu ? »,
   « comment installer cet accessoire ? », « guide-moi pour fixer la coulisse »).
   → Procédure ordonnée à dérouler étape par étape.
2. flow_kind = "diagnostic" : l'utilisateur signale un PROBLÈME / SYMPTÔME sur un produit posé
   (ex : « mon volet roulant ne fonctionne plus », « déformation du montant coulisse »,
   « non-conformité », « le volet est bloqué », « fuite d'eau »).
   → Diagnostic en plusieurs étapes (vérifications successives).

GUIDAGE = false si :
- C'est une question factuelle ponctuelle (cote, tolérance, référence pièce, dimension, condition de
  garantie, comparaison produit) qui se répond en une fois.
- C'est une question de DIMENSIONNEMENT ou de CHOIX DE VALEUR (« à quelle hauteur poser… »,
  « quelle hauteur choisir… », « quelle taille prendre… ») : c'est une réponse documentaire,
  PAS un accompagnement pas-à-pas.
- C'est une salutation, un remerciement, du bavardage, ou une question sur l'identité de l'assistant.

RÈGLE : en cas de doute entre une question factuelle et un guidage, choisis is_guided=false.

PRODUIT NOMMÉ (product_named) :
- true UNIQUEMENT si l'UTILISATEUR (dans son message ou ses messages précédents, PAS ceux de
  l'assistant) a explicitement nommé le produit, la gamme ou la référence concernés
  (ex : « INNOSLIDE », « Perform 76 », « KSR PVC », « seuil 76180 »).
- false sinon — même si le contexte laisse deviner un produit probable. Ne devine jamais.

AMBIGUÏTÉ D'INTENTION (needs_intent_clarification) :
- true si le message peut se lire de DEUX façons matériellement différentes, typiquement
  RÉGLER/ajuster un élément existant vs DÉFINIR/choisir une valeur ou une position
  (ex : « comment régler la hauteur de poignée ? » = ajuster la poignée posée OU déterminer
  à quelle hauteur la poser). En cas de vraie ambiguïté → true.
- false si le contexte lève clairement l'ambiguïté.

Retourne UNIQUEMENT un JSON :
- is_guided: true | false
- flow_kind: "howto" | "diagnostic" (valeur indicative si is_guided=false)
- topic: courte étiquette de la TÂCHE, sans y injecter un produit que l'utilisateur n'a pas
  nommé (ex : « réglage hauteur poignée », « volet roulant bloqué »)
- product_named: true | false
- needs_intent_clarification: true | false
- reasoning: explication courte en français
"""


async def decide_guided_mode(
    query: str, history: Optional[List[Dict[str, str]]] = None
) -> GuidedModeDecision:
    """Détermine si la demande relève du guidage procédural (how-to / diagnostic SAV).

    Appelé uniquement quand GUIDED_FLOW_ENABLED et hors reprise d'un parcours actif :
    aucun impact sur le pipeline one-shot quand le flag est désactivé.
    """
    logger.info("[guided_mode] Analyse — query=%r", (query or "")[:120])
    try:
        from app.services.category_catalog import SYMPTOM_LABELS

        symptom_vocab = " | ".join(f"{slug} ({label})" for slug, label in SYMPTOM_LABELS.items())
        system_prompt = (
            GUIDED_MODE_SYSTEM_PROMPT
            + "\n\nSYMPTÔMES SAV connus (slugs) : "
            + symptom_vocab
            + "\nSi flow_kind=\"diagnostic\" et qu'un symptôme de cette liste correspond, renseigne "
            "\"detected_symptom\" avec son slug exact ; sinon \"detected_symptom\": \"\"."
        )
        messages = [{"role": "system", "content": system_prompt}]
        history_snippet = ""
        if history:
            lines = []
            for msg in history[-6:]:
                role = msg.get("role", "user")
                content = str(msg.get("content", ""))[:300]
                lines.append(f"{role}: {content}")
            history_snippet = "\n".join(lines)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Historique récent :\n{history_snippet or '(vide)'}\n\n"
                    f"Message à classer : '{query}'"
                ),
            }
        )

        # Décision subtile (is_guided + flow_kind diagnostic/howto + symptôme) : on garde
        # le gros modèle ici, sinon le mode guidé n'est plus détecté (plus de boutons).
        response = await chat(
            "",
            model=settings.MODEL_FAST,
            context=messages,
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"].get("content", "{}")
        data = json.loads(content)

        flow_kind = str(data.get("flow_kind") or "howto").strip().lower()
        if flow_kind not in ("howto", "diagnostic"):
            flow_kind = "howto"

        detected_symptom = str(data.get("detected_symptom") or "").strip().lower()
        if detected_symptom and detected_symptom not in SYMPTOM_LABELS:
            detected_symptom = ""

        decision = GuidedModeDecision(
            is_guided=bool(data.get("is_guided")),
            flow_kind=flow_kind,
            topic=str(data.get("topic") or "").strip()[:300],
            detected_symptom=detected_symptom,
            product_named=bool(data.get("product_named")),
            needs_intent_clarification=bool(data.get("needs_intent_clarification")),
            reasoning=str(data.get("reasoning") or "").strip(),
        )
        logger.info(
            "[guided_mode] is_guided=%s flow_kind=%s topic=%r symptom=%r product_named=%s intent_ambigu=%s",
            decision.is_guided,
            decision.flow_kind,
            decision.topic,
            decision.detected_symptom,
            decision.product_named,
            decision.needs_intent_clarification,
        )
        return decision
    except Exception as e:
        logger.error(f"Erreur lors de la classification guidage: {e}")
        # En cas d'échec, ne pas forcer le mode guidé (repli sur le pipeline standard).
        return GuidedModeDecision(is_guided=False, reasoning=f"Erreur technique: {e}")


async def resolve_guided_mode(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    *,
    fused_guided: Any = None,
    topic: str = "",
) -> GuidedModeDecision:
    """Décision de mode guidé SANS appel LLM dédié quand la compréhension fusionnée l'a
    déjà produite (P0.4 : un seul appel LLM avant le retrieval).

    ``fused_guided`` est un objet type GuidedDecision (attribut ``present``). S'il est
    présent, on en dérive directement la décision (0 appel LLM) ; sinon on retombe sur
    ``decide_guided_mode`` (1 appel LLM, ex. compréhension fusionnée désactivée ou échouée
    — c'est aussi ce chemin que mockent les tests)."""
    if fused_guided is not None and getattr(fused_guided, "present", False):
        return GuidedModeDecision(
            is_guided=bool(getattr(fused_guided, "is_guided", False)),
            flow_kind=str(getattr(fused_guided, "flow_kind", "howto") or "howto"),
            topic=(topic or "").strip()[:300],
            detected_symptom=str(getattr(fused_guided, "detected_symptom", "") or ""),
            product_named=bool(getattr(fused_guided, "product_named", True)),
            needs_intent_clarification=bool(
                getattr(fused_guided, "needs_intent_clarification", False)
            ),
            reasoning="fused",
        )
    return await decide_guided_mode(query, history)


async def decide_retrieval_route(query: str, history: Optional[List[Dict[str, str]]] = None) -> RetrievalDecision:
    """
    Détermine si une requête nécessite une recherche documentaire RAG ou si elle peut être traitée directement.
    """
    logger.info("[query_routing] Analyse routage — query=%r", (query or "")[:120])
    try:
        messages = [
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
        ]
        
        # Pour l'instant, on se base sur la query principale.
        messages.append({"role": "user", "content": f"Analyse cette requête : '{query}'"})
        
        response = await chat(
            "", 
            model=settings.MODEL_QUERY_UNDERSTANDING, 
            context=messages,
            response_format={"type": "json_object"}
        )
        
        content = response["choices"][0]["message"].get("content", "{}")
        data = json.loads(content)
        
        decision = data.get("decision", "rag")
        if decision not in ("direct", "rag"):
            decision = "rag"

        logger.info(
            "[query_routing] decision=%s reason=%s",
            decision,
            data.get("reasoning", "Défaut"),
        )

        return RetrievalDecision(
            decision=decision,
            reasoning=data.get("reasoning", "Défaut"),
        )
    except Exception as e:
        logger.error(f"Erreur lors du choix de routage de la requête: {e}")
        return RetrievalDecision(decision="rag", reasoning=f"Erreur technique: {e}")

