import asyncio
import httpx
import json
import random
from typing import List, Dict, Optional, Any
import time
import logging

from app.config import settings
from app.services.chat_tools import run_tool, get_web_search_system_prompt

logger = logging.getLogger(__name__)

MAX_RETRIES = getattr(settings, "MISTRAL_MAX_RETRIES", 5) or 5
RETRY_BACKOFF_BASE_SECONDS = getattr(settings, "MISTRAL_RETRY_BACKOFF_BASE", 2.0) or 2.0
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class MistralRateLimitError(RuntimeError):
    """Le quota ou le débit de l'API Mistral est dépassé après plusieurs tentatives."""


def _retry_wait_seconds(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), 0.5)
        except ValueError:
            pass
    return RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))


def _rate_limit_user_message() -> str:
    return (
        "L'API Mistral est temporairement saturée (limite de débit). "
        "Veuillez réessayer dans quelques instants."
    )


def _build_mistral_content(msg: Dict[str, Any]) -> Any:
    """Convertit un message (texte + images base64 optionnelles) au format Mistral/Pixtral."""
    images = msg.get("images") or []
    content = msg.get("content", "")

    if not images:
        return content

    parts: List[Dict[str, Any]] = []
    if content:
        parts.append({"type": "text", "text": content})
    for img_b64 in images:
        parts.append(
            {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{img_b64}",
            }
        )
    return parts if parts else ""


async def _post_json_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: Dict[str, str],
    payload: Dict[str, Any],
) -> httpx.Response:
    last_response: Optional[httpx.Response] = None
    for attempt in range(1, MAX_RETRIES + 1):
        response = await client.post(url, headers=headers, json=payload)
        last_response = response
        if response.status_code not in RETRYABLE_STATUS_CODES:
            if response.status_code >= 400:
                logger.error(
                    "Mistral API error (%s): %s",
                    response.status_code,
                    response.text[:2000],
                )
            response.raise_for_status()
            return response
        if attempt >= MAX_RETRIES:
            break
        wait = _retry_wait_seconds(response, attempt) + random.uniform(0, 0.5)
        logger.warning(
            "Mistral chat tentative %s/%s: HTTP %s, retry dans %.1fs",
            attempt,
            MAX_RETRIES,
            response.status_code,
            wait,
        )
        await asyncio.sleep(wait)

    assert last_response is not None
    if last_response.status_code == 429:
        raise MistralRateLimitError(_rate_limit_user_message()) from None
    last_response.raise_for_status()
    return last_response


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
    Supporte les images base64 sur les messages user (Pixtral / modèles vision).
    """
    if not settings.MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY n'est pas configurée")

    messages: List[Dict[str, Any]] = []
    if context:
        messages.extend(context)
    else:
        messages.append({"role": "user", "content": message})

    web_search_prompt = get_web_search_system_prompt(include_brave_search=bool(tools))
    if web_search_prompt:
        messages.insert(0, {"role": "system", "content": web_search_prompt})

    messages = _clean_messages(messages)

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

            payload.update(kwargs)

            base_url = (settings.MISTRAL_BASE_URL or "https://api.mistral.ai").rstrip("/")
            headers = {
                "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await _post_json_with_retry(
                    client,
                    f"{base_url}/v1/chat/completions",
                    headers=headers,
                    payload=payload,
                )
                data = response.json()
        except MistralRateLimitError:
            raise
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
    """Nettoie la liste de messages pour Mistral, en préservant les images sur les messages user."""
    if not messages:
        return []

    cleaned: List[Dict[str, Any]] = []

    system_content: List[str] = []
    idx = 0
    while idx < len(messages) and messages[idx].get("role") == "system":
        content = messages[idx].get("content", "")
        if content:
            system_content.append(str(content))
        idx += 1

    if system_content:
        cleaned.append({"role": "system", "content": "\n\n".join(system_content)})

    for i in range(idx, len(messages)):
        msg = messages[i]
        role = msg.get("role")
        content = msg.get("content", "")
        images = msg.get("images") or []

        if role == "user" and images:
            cleaned.append({"role": "user", "content": _build_mistral_content(msg)})
            continue

        if not content:
            continue

        content_str = str(content)
        if cleaned and cleaned[-1]["role"] == role and role != "user":
            cleaned[-1]["content"] += "\n\n" + content_str
        elif cleaned and cleaned[-1]["role"] == role and role == "user" and not cleaned[-1].get("_has_images"):
            cleaned[-1]["content"] += "\n\n" + content_str
        else:
            cleaned.append({"role": role, "content": content_str})

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
        start_ts = time.monotonic()
        max_duration_seconds = 400
        idle_break_seconds = 90
        last_token_ts = time.monotonic()

        raw_messages: List[Dict[str, Any]] = []
        if context:
            raw_messages.extend(context)
        else:
            raw_messages.append({"role": "user", "content": message})

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

        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        prompt_preview = str(messages[-1].get("content", ""))[:100]
        logger.info(
            "Appel Mistral: model=%s, nb_msg=%s, roles=%s, chars=%s, prompt='%s'...",
            model,
            len(messages),
            role_seq,
            total_chars,
            prompt_preview,
        )

        base_url = (settings.MISTRAL_BASE_URL or "https://api.mistral.ai").rstrip("/")
        timeout = httpx.Timeout(400.0, connect=60.0)

        headers = {
            "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        }
        completions_url = f"{base_url}/v1/chat/completions"

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                logger.info(f"Connexion Mistral en cours ({base_url})...")
                for attempt in range(1, MAX_RETRIES + 1):
                    async with client.stream(
                        "POST",
                        completions_url,
                        headers=headers,
                        json=payload,
                    ) as response:
                        logger.info(f"Mistral status: {response.status_code}")
                        if response.status_code in RETRYABLE_STATUS_CODES:
                            if attempt < MAX_RETRIES:
                                wait = _retry_wait_seconds(response, attempt) + random.uniform(
                                    0, 0.5
                                )
                                logger.warning(
                                    "Mistral stream tentative %s/%s: HTTP %s, retry dans %.1fs",
                                    attempt,
                                    MAX_RETRIES,
                                    response.status_code,
                                    wait,
                                )
                                await asyncio.sleep(wait)
                                continue
                            if response.status_code == 429:
                                raise MistralRateLimitError(_rate_limit_user_message())
                            response.raise_for_status()

                        if response.status_code != 200:
                            error_body = await response.aread()
                            logger.error(
                                "Mistral API Error (%s): %s",
                                response.status_code,
                                error_body.decode(errors="replace"),
                            )
                            if response.status_code == 429:
                                raise MistralRateLimitError(_rate_limit_user_message())
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

                            if time.monotonic() - last_token_ts > idle_break_seconds:
                                logger.warning(
                                    f"Mistral stream idle timeout ({idle_break_seconds}s)"
                                )
                                break
                            if time.monotonic() - start_ts > max_duration_seconds:
                                logger.warning(
                                    f"Mistral stream max duration reached ({max_duration_seconds}s)"
                                )
                                break
                    break

            except MistralRateLimitError:
                raise
            except Exception as e:
                logger.error(f"Erreur lors de la requête Mistral: {e}")
                raise

    except MistralRateLimitError:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du streaming Mistral: {e}")
        raise
