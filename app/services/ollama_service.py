import httpx
from typing import List, Dict, Optional, Any

from app.config import settings


async def chat(
    message: str,
    model: str,
    context: Optional[List[Dict]] = None,
) -> Dict:
    """
    Appel Ollama (/api/chat) avec réponse normalisée au format OpenAI-like:
    {"choices": [{"message": {"content": "..."}}]}
    """
    messages: List[Dict[str, Any]] = []
    if context:
        messages.extend(context)
    else:
        messages.append({"role": "user", "content": message})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    base_url = (settings.OLLAMA_BASE_URL or "http://host.docker.internal:11434").rstrip("/")
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    content = (data.get("message") or {}).get("content", "")
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                }
            }
        ]
    }
