"""Service de recherche web via l'API Brave Search (pour function calling)."""
import httpx
from typing import List, Dict, Optional
from app.config import settings

BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


async def brave_web_search(query: str, count: int = 8) -> List[Dict]:
    """
    Recherche web via l'API Brave Search.

    Args:
        query: Requête de recherche.
        count: Nombre de résultats à retourner (défaut 8).

    Returns:
        Liste de dicts avec 'title', 'url', 'description' pour chaque résultat.
    """
    if not settings.BRAVE_SEARCH_API_KEY:
        return [{"error": "BRAVE_SEARCH_API_KEY n'est pas configurée"}]

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                BRAVE_WEB_SEARCH_URL,
                params={"q": query, "count": min(count, 20)},
                headers={"X-Subscription-Token": settings.BRAVE_SEARCH_API_KEY},
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        return [{"error": f"Brave Search API: {e.response.status_code} - {e.response.text[:200]}"}]
    except Exception as e:
        return [{"error": f"Erreur Brave Search: {str(e)}"}]

    results = []
    web = data.get("web") or {}
    for item in (web.get("results") or [])[:count]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("description", ""),
        })
    return results
