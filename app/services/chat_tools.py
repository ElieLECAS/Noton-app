"""Définitions des tools pour function calling (compatibles OpenAI et Ollama)."""
import json
from typing import List, Dict, Any, Optional
from app.services.brave_search_service import brave_web_search

# Définition du tool Brave Search (format OpenAI/Ollama) — description courte pour limiter les tokens
BRAVE_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "brave_web_search",
        "description": (
            "Recherche web à jour. À appeler pour : actualités, événements récents, données récentes. "
            "Ne jamais dire que tu n'as pas accès à Internet ; utiliser cette fonction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Mots-clés (fr/en), ex: actualités février 2026",
                }
            },
            "required": ["query"],
        },
    },
}

# Message système court quand la recherche web est disponible
WEB_SEARCH_SYSTEM_PROMPT = (
    "Utilise brave_web_search pour les actualités ou données à jour."
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
