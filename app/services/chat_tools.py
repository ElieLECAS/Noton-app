"""Définitions des tools pour function calling (compatibles OpenAI et Ollama)."""
import json
from typing import List, Dict, Any, Optional
from app.services.brave_search_service import brave_web_search

# Définition du tool Brave Search (format OpenAI/Ollama)
BRAVE_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "brave_web_search",
        "description": (
            "Effectue une recherche web en temps réel pour obtenir des informations à jour. "
            "Tu DOIS appeler cette fonction dès que l'utilisateur demande : actualités, infos politiques, événements récents, "
            "données récentes, ou toute information qui nécessite des sources en ligne. "
            "Ne dis jamais que tu n'as pas accès à Internet : utilise cette fonction à la place."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Requête de recherche (mots-clés en français ou anglais, ex: actualités politique février 2026)",
                }
            },
            "required": ["query"],
        },
    },
}

# Message système à injecter quand la recherche web est disponible (incite le modèle à utiliser le tool)
WEB_SEARCH_SYSTEM_PROMPT = (
    "Tu peux utiliser l'outil brave_web_search pour obtenir des informations récentes. "
    "Si la question concerne des actualités ou des données à jour, appelle cet outil plutôt que de répondre sans source."
)


def get_available_tools(include_brave_search: bool) -> List[Dict]:
    """Retourne la liste des tools à envoyer au modèle (selon la config)."""
    tools = []
    if include_brave_search:
        tools.append(BRAVE_SEARCH_TOOL)
    return tools


def get_web_search_system_prompt(include_brave_search: bool) -> Optional[str]:
    """Retourne le message système pour inciter à la recherche web, ou None."""
    return WEB_SEARCH_SYSTEM_PROMPT if include_brave_search else None


async def run_tool(name: str, arguments: Dict[str, Any]) -> str:
    """
    Exécute un tool par son nom avec les arguments fournis.
    Retourne une chaîne (souvent JSON) à renvoyer au modèle.
    """
    if name == "brave_web_search":
        query = (arguments.get("query") or "").strip()
        if not query:
            return json.dumps({"error": "Le paramètre 'query' est requis."})
        results = await brave_web_search(query)
        if results and "error" in results[0]:
            return json.dumps(results[0])
        return json.dumps(results, ensure_ascii=False)
    return json.dumps({"error": f"Tool inconnu: {name}"})
