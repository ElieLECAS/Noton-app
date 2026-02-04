import httpx
import json
from typing import List, Dict, Optional
from app.config import settings
from app.services.chat_tools import run_tool, get_web_search_system_prompt


async def get_available_models() -> List[str]:
    """Récupérer la liste des modèles Ollama disponibles"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
            
            # Extraire les noms des modèles
            models = []
            if "models" in data:
                for model in data["models"]:
                    if "name" in model:
                        models.append(model["name"])
            
            return models
    except Exception as e:
        print(f"Erreur lors de la récupération des modèles Ollama: {e}")
        return []


async def chat(
    message: str,
    model: str,
    context: Optional[List[Dict]] = None,
    tools: Optional[List[Dict]] = None,
) -> Dict:
    """Envoyer un message au chatbot Ollama, avec boucle tool_calls si des tools sont fournis."""
    messages = list(context) if context else [{"role": "user", "content": message}]
    if not context and message:
        messages = [{"role": "user", "content": message}]

    # Message système pour inciter le modèle à utiliser la recherche web quand les tools sont activés
    web_search_prompt = get_web_search_system_prompt(include_brave_search=bool(tools))
    if web_search_prompt:
        messages.insert(0, {"role": "system", "content": web_search_prompt})

    max_tool_rounds = 5
    for _ in range(max_tool_rounds):
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
            }
            if tools:
                payload["tools"] = tools

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            print(f"Erreur lors de l'appel à Ollama: {e}")
            raise

        msg = data.get("message") or {}
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            return data

        # Ajouter le message assistant (avec tool_calls) à l'historique
        messages.append(msg)

        # Exécuter chaque tool et ajouter les réponses
        for tc in tool_calls:
            fn = tc.get("function") or tc
            name = fn.get("name", "")
            args_raw = fn.get("arguments")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw) if args_raw else {}
                except json.JSONDecodeError:
                    args = {}
            else:
                args = args_raw or {}
            result = await run_tool(name, args)
            messages.append({"role": "tool", "content": result})

    return data


async def chat_stream(message: str, model: str, context: Optional[List[Dict]] = None):
    """Envoyer un message au chatbot Ollama avec streaming"""
    try:
        payload = {
            "model": model,
            "messages": context or [{"role": "user", "content": message}],
            "stream": True
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield line
    except Exception as e:
        print(f"Erreur lors du streaming Ollama: {e}")
        raise

