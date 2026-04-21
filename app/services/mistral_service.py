import httpx
import json
from typing import List, Dict, Optional, Any
import time
import logging

from app.config import settings
from app.services.chat_tools import run_tool, get_web_search_system_prompt

logger = logging.getLogger(__name__)


async def chat(
    message: str,
    model: str,
    context: Optional[List[Dict]] = None,
    tools: Optional[List[Dict]] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    **kwargs: Any,
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
                "max_tokens": max_tokens if max_tokens is not None else settings.MAX_COMPLETION_TOKENS,
            }
            if temperature is not None:
                payload["temperature"] = temperature
            if top_p is not None:
                payload["top_p"] = top_p
            if tools:
                payload["tools"] = tools
            
            # Ajouter les arguments supplémentaires (ex: response_format)
            payload.update(kwargs)

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
            logger.error(f"Erreur lors de l'appel à Mistral: {e}")
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


def _clean_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Nettoie la liste de messages pour Mistral:
    1. Fusionne les messages système consécutifs au début.
    2. Fusionne les messages consécutifs du même rôle (user/user, assistant/assistant).
    3. S'assure que l'ordre est respecté (system? -> user -> assistant -> user...).
    """
    if not messages:
        return []
    
    cleaned = []
    
    # 1. Gérer le système
    system_content = []
    idx = 0
    while idx < len(messages) and messages[idx].get("role") == "system":
        content = messages[idx].get("content", "")
        if content:
            system_content.append(content)
        idx += 1
    
    if system_content:
        cleaned.append({"role": "system", "content": "\n\n".join(system_content)})
    
    # 2. Gérer le reste avec fusion des doublons de rôles
    for i in range(idx, len(messages)):
        msg = messages[i]
        role = msg.get("role")
        content = msg.get("content", "")
        if not content:
            continue
            
        if cleaned and cleaned[-1]["role"] == role:
            # Même rôle que le précédent, on fusionne
            cleaned[-1]["content"] += "\n\n" + content
        else:
            cleaned.append({"role": role, "content": content})
    
    # Mistral demande que ça commence par user (si pas de system) ou que ça suive system
    # Si le premier après system est un assistant, on l'ignore ou on l'insère après un user vide
    if cleaned and cleaned[0]["role"] == "system":
        if len(cleaned) > 1 and cleaned[1]["role"] == "assistant":
            cleaned.insert(1, {"role": "user", "content": "(Suite de la conversation)"})
    elif cleaned and cleaned[0]["role"] == "assistant":
        cleaned.insert(0, {"role": "user", "content": "(Début de la conversation)"})
        
    return cleaned

async def chat_stream(
    message: str,
    model: str,
    context: Optional[List[Dict]] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    **kwargs,
):
    """
    Appel au chatbot Mistral avec streaming.
    Convertit les chunks renvoyés par Mistral vers un format compatible avec le frontend.
    """
    if not settings.MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY n'est pas configurée")

    try:
        # Sécurité anti-boucle infinie
        start_ts = time.monotonic()
        max_duration_seconds = 400
        idle_break_seconds = 90
        last_token_ts = time.monotonic()

        raw_messages: List[Dict[str, Any]] = []
        if context:
            raw_messages.extend(context)
        else:
            raw_messages.append({"role": "user", "content": message})

        # Nettoyage pour conformité API Mistral
        messages = _clean_messages(raw_messages)
        role_seq = "-".join([m["role"][0].upper() for m in messages])

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens if max_tokens is not None else settings.MAX_COMPLETION_TOKENS,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        
        payload.update(kwargs)

        # Logging pour diagnostic
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        prompt_preview = str(messages[-1].get("content", ""))[:100]
        logger.info(f"Appel Mistral: model={model}, nb_msg={len(messages)}, roles={role_seq}, chars={total_chars}, prompt='{prompt_preview}'...")

        base_url = (settings.MISTRAL_BASE_URL or "https://api.mistral.ai").rstrip("/")
        timeout = httpx.Timeout(400.0, connect=60.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                logger.info(f"Connexion Mistral en cours ({base_url})...")
                async with client.stream(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response:
                    logger.info(f"Mistral status: {response.status_code}")
                    if response.status_code != 200:
                        error_body = await response.aread()
                        logger.error(f"Mistral API Error ({response.status_code}): {error_body.decode()}")
                        response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data_str = line.split("data:", 1)[1].strip()
                            if data_str == "[DONE]":
                                break
                            if not data_str:
                                continue
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                if choices:
                                    choice0 = choices[0] or {}
                                    finish_reason = choice0.get("finish_reason")
                                    delta = choice0.get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        last_token_ts = time.monotonic()
                                        yield json.dumps({"message": {"content": content}})
                                    
                                    if finish_reason:
                                        break
                            except json.JSONDecodeError:
                                continue

                        # Timeouts de sécurité
                        if time.monotonic() - last_token_ts > idle_break_seconds:
                            logger.warning(f"Mistral stream idle timeout ({idle_break_seconds}s)")
                            break
                        if time.monotonic() - start_ts > max_duration_seconds:
                            logger.warning(f"Mistral stream max duration reached ({max_duration_seconds}s)")
                            break

            except Exception as e:
                logger.error(f"Erreur lors de la requête Mistral: {e}")
                raise

    except Exception as e:
        logger.error(f"Erreur lors du streaming Mistral: {e}")
        raise
