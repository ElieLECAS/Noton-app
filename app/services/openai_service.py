import httpx
import json
from typing import List, Dict, Optional, Any
from app.config import settings
from app.services.chat_tools import run_tool, get_web_search_system_prompt


async def get_available_models() -> List[str]:
    """Récupérer la liste des modèles OpenAI disponibles"""
    if not settings.OPENAI_API_KEY:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}"
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Filtrer les modèles GPT disponibles
            models = []
            if "data" in data:
                for model in data["data"]:
                    model_id = model.get("id", "")
                    if model_id.startswith("gpt-") and "instruct" not in model_id.lower():
                        models.append(model_id)
            
            # Ajouter les modèles depuis les settings s'ils sont configurés et ne sont pas déjà dans la liste
            if settings.OPENAI_MODEL:
                for model in settings.OPENAI_MODEL:
                    if model not in models:
                        models.append(model)
            
            # Trier et retourner les modèles
            return sorted(models)
    except Exception as e:
        print(f"Erreur lors de la récupération des modèles OpenAI: {e}")
        return []


async def chat(
    message: str,
    model: str,
    context: Optional[List[Dict]] = None,
    tools: Optional[List[Dict]] = None,
) -> Dict:
    """Envoyer un message au chatbot OpenAI, avec boucle tool_calls si des tools sont fournis."""
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY n'est pas configurée")

    messages = []
    if context:
        messages.extend(context)
    else:
        messages.append({"role": "user", "content": message})

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
                "max_tokens": settings.MAX_COMPLETION_TOKENS,
            }
            if tools:
                payload["tools"] = tools

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            print(f"Erreur lors de l'appel à OpenAI: {e}")
            raise

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            return data

        # Ajouter le message assistant (avec tool_calls) à l'historique
        messages.append(msg)

        # Exécuter chaque tool et ajouter les réponses (format OpenAI: role "tool" + tool_call_id)
        for tc in tool_calls:
            tid = tc.get("id", "")
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args_str = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}
            result = await run_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": tid, "content": result})

    return data


async def chat_stream(message: str, model: str, context: Optional[List[Dict]] = None):
    """Envoyer un message au chatbot OpenAI avec streaming"""
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY n'est pas configurée")
    
    try:
        # Construire les messages
        messages = []
        if context:
            messages.extend(context)
        else:
            messages.append({"role": "user", "content": message})
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": settings.MAX_COMPLETION_TOKENS,
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        # OpenAI renvoie des lignes au format "data: {...}" ou "data: [DONE]"
                        if line.startswith("data: "):
                            data_str = line[6:]  # Enlever "data: "
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                # Convertir au format interne attendu par le frontend
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        stream_payload = {
                                            "message": {
                                                "content": delta["content"]
                                            }
                                        }
                                        yield json.dumps(stream_payload)
                            except json.JSONDecodeError:
                                pass
    except Exception as e:
        print(f"Erreur lors du streaming OpenAI: {e}")
        raise


async def generate_image(
    prompt: str,
    model: Optional[str] = None,
    size: str = "1024x1024",
    quality: str = "standard",
) -> List[Dict]:
    """
    Générer une ou plusieurs images via l'API OpenAI (DALL-E 3).
    Retourne une liste de dicts avec 'url' (ou 'b64_json' selon response_format).
    """
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY n'est pas configurée")

    model = model or settings.OPENAI_IMAGE_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "quality": quality,
        "response_format": "url",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])

