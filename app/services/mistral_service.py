import httpx
import json
from typing import List, Dict, Optional, Any

from app.config import settings
from app.services.chat_tools import run_tool, get_web_search_system_prompt


async def chat(
    message: str,
    model: str,
    context: Optional[List[Dict]] = None,
    tools: Optional[List[Dict]] = None,
) -> Dict:
    """
    Appel au chatbot Mistral (API compatible chat completions).
    Supporte optionnellement les tools (function calling) avec la même boucle que le service OpenAI.
    """
    if not settings.MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY n'est pas configurée")

    messages: List[Dict[str, Any]] = []
    if context:
        messages.extend(context)
    else:
        messages.append({"role": "user", "content": message})

    # Message système pour la recherche web via tools (même logique qu'OpenAI)
    web_search_prompt = get_web_search_system_prompt(include_brave_search=bool(tools))
    if web_search_prompt:
        messages.insert(0, {"role": "system", "content": web_search_prompt})

    max_tool_rounds = 5
    for _ in range(max_tool_rounds):
        try:
            payload: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": False,
                "max_tokens": settings.MAX_COMPLETION_TOKENS,
            }
            if tools:
                payload["tools"] = tools

            base_url = (settings.MISTRAL_BASE_URL or "https://api.mistral.ai").rstrip("/")
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            print(f"Erreur lors de l'appel à Mistral: {e}")
            raise

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            return data

        messages.append(msg)

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


async def chat_stream(
    message: str,
    model: str,
    context: Optional[List[Dict]] = None,
):
    """
    Appel au chatbot Mistral avec streaming.
    Convertit les chunks renvoyés par Mistral vers un format compatible avec le frontend (même format que le stream OpenAI/Ollama).
    """
    if not settings.MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY n'est pas configurée")

    try:
        messages: List[Dict[str, Any]] = []
        if context:
            messages.extend(context)
        else:
            messages.append({"role": "user", "content": message})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": settings.MAX_COMPLETION_TOKENS,
        }

        base_url = (settings.MISTRAL_BASE_URL or "https://api.mistral.ai").rstrip("/")
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    # L'API renvoie des lignes "data: {...}" ou "data: [DONE]"
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    # Formater comme Ollama: {"message": {"content": "..."}}
                                    mistral_format = {
                                        "message": {
                                            "content": delta["content"],
                                        }
                                    }
                                    yield json.dumps(mistral_format)
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        print(f"Erreur lors du streaming Mistral: {e}")
        raise

