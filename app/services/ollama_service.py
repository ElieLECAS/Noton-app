import httpx
import json
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
        for msg in context:
            # Transférer le message et son champ images optionnel
            msg_copy = {
                "role": msg.get("role"),
                "content": msg.get("content")
            }
            if "images" in msg:
                msg_copy["images"] = msg["images"]
            messages.append(msg_copy)
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


async def chat_stream(
    message: str,
    model: str,
    context: Optional[List[Dict]] = None,
):
    """
    Appel streaming Ollama (/api/chat) renvoyant un générateur asynchrone
    de JSON au format similaire à mistral_chat_stream.
    """
    messages: List[Dict[str, Any]] = []
    if context:
        for msg in context:
            msg_copy = {
                "role": msg.get("role"),
                "content": msg.get("content")
            }
            if "images" in msg:
                msg_copy["images"] = msg["images"]
            messages.append(msg_copy)
    else:
        messages.append({"role": "user", "content": message})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    base_url = (settings.OLLAMA_BASE_URL or "http://host.docker.internal:11434").rstrip("/")
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{base_url}/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    chunk_content = (data.get("message") or {}).get("content", "")
                    done = data.get("done", False)
                    # Normalisation sous le même format attendu par le routeur
                    yield json.dumps({
                        "message": {
                            "content": chunk_content
                        },
                        "done": done
                    })
                except Exception:
                    continue
