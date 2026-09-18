"""Client Mistral : un appel non streamé (titres) et un appel streamé (le chat).

Le flux est converti en événements JSON pour le front : ``{"thinking": ...}`` pendant le
raisonnement, ``{"message": {"content": ...}}`` pour la réponse, et ``{"usage": {...}}`` en fin
de flux — Mistral l'envoie dans le dernier chunk ; sans lui on ne sait ni si le cache de
prompt fonctionne ni où en est le budget de contexte.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = settings.MISTRAL_MAX_RETRIES or 5
RETRY_BACKOFF_BASE_SECONDS = settings.MISTRAL_RETRY_BACKOFF_BASE or 2.0
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


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }


def _completions_url() -> str:
    base_url = (settings.MISTRAL_BASE_URL or "https://api.mistral.ai").rstrip("/")
    return f"{base_url}/v1/chat/completions"


async def _post_json_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: Dict[str, str],
    payload: Dict[str, Any],
) -> httpx.Response:
    last_response: Optional[httpx.Response] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.post(url, headers=headers, json=payload)
            last_response = response
            if response.status_code not in RETRYABLE_STATUS_CODES:
                if response.status_code >= 400:
                    logger.error("Mistral API error (%s): %s", response.status_code, response.text[:2000])
                response.raise_for_status()
                return response
            wait = _retry_wait_seconds(response, attempt) + random.uniform(0, 0.5)
            logger.warning(
                "Mistral chat tentative %s/%s: HTTP %s, retry dans %.1fs",
                attempt, MAX_RETRIES, response.status_code, wait,
            )
        except httpx.RequestError as exc:
            if attempt >= MAX_RETRIES:
                logger.error("Mistral chat tentative %s/%s: erreur réseau %s, abandon.", attempt, MAX_RETRIES, exc)
                raise
            wait = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning("Mistral chat tentative %s/%s: erreur réseau %s, retry dans %.1fs", attempt, MAX_RETRIES, exc, wait)
        if attempt < MAX_RETRIES:
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
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs: Any,
) -> Dict:
    """Appel non streamé (chat completions). ``context`` remplace ``message`` s'il est fourni."""
    if not settings.MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY n'est pas configurée")

    messages = _clean_messages(list(context) if context else [{"role": "user", "content": message}])
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens if max_tokens is not None else settings.MAX_COMPLETION_TOKENS,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    payload.update(kwargs)

    logger.info("[MISTRAL] Appel sync — model=%s, nb_msg=%d", model, len(messages))
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await _post_json_with_retry(client, _completions_url(), headers=_headers(), payload=payload)
        data = response.json()
        usage = data.get("usage") or {}
        logger.info(
            "[MISTRAL] Sync OK — model=%s, prompt_tokens=%s, completion_tokens=%s",
            model, usage.get("prompt_tokens"), usage.get("completion_tokens"),
        )
        return data


def _text_of(content: Any) -> str:
    if isinstance(content, list):
        return "\n\n".join(
            part["text"] for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return "" if content is None else str(content)


def _clean_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Met la liste au format qu'attend Mistral :
    1. les messages système consécutifs du début sont fusionnés en un seul ;
    2. deux messages consécutifs du même rôle (user/user, assistant/assistant) sont fusionnés,
       les messages vides sont ignorés ;
    3. la suite commence par un utilisateur (après l'éventuel système).
    """
    if not messages:
        return []

    cleaned: List[Dict[str, Any]] = []
    system_content: List[str] = []
    idx = 0
    while idx < len(messages) and messages[idx].get("role") == "system":
        content = _text_of(messages[idx].get("content", ""))
        if content:
            system_content.append(content)
        idx += 1
    if system_content:
        cleaned.append({"role": "system", "content": "\n\n".join(system_content)})

    for msg in messages[idx:]:
        role = msg.get("role")
        content = _text_of(msg.get("content", ""))
        if not content.strip():
            continue
        prev = cleaned[-1] if cleaned else None
        if prev is not None and prev.get("role") == role and role in ("user", "assistant"):
            prev["content"] += "\n\n" + content
        else:
            cleaned.append({"role": role, "content": content})

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
    **kwargs: Any,
):
    """Appel streamé. Rend des chaînes JSON : ``thinking``, ``message``, puis ``usage``.

    ``kwargs`` part tel quel dans la charge utile (``prompt_cache_key``, ``reasoning_effort``).
    """
    if not settings.MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY n'est pas configurée")

    start_ts = time.monotonic()
    max_duration_seconds = 400
    idle_break_seconds = 90
    last_token_ts = time.monotonic()

    messages = _clean_messages(list(context) if context else [{"role": "user", "content": message}])
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
    logger.info(
        "[MISTRAL] Appel stream — model=%s, nb_msg=%d, chars=%d, cache_key=%s",
        model, len(messages), total_chars, payload.get("prompt_cache_key"),
    )

    timeout = httpx.Timeout(400.0, connect=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        has_yielded = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with client.stream("POST", _completions_url(), headers=_headers(), json=payload) as response:
                    logger.info("[MISTRAL] Stream status: %s", response.status_code)
                    if response.status_code in RETRYABLE_STATUS_CODES:
                        if attempt < MAX_RETRIES:
                            wait = _retry_wait_seconds(response, attempt) + random.uniform(0, 0.5)
                            logger.warning(
                                "Mistral stream tentative %s/%s: HTTP %s, retry dans %.1fs",
                                attempt, MAX_RETRIES, response.status_code, wait,
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
                            response.status_code, error_body.decode(errors="replace"),
                        )
                        response.raise_for_status()

                    usage: Optional[Dict[str, Any]] = None
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line.split("data:", 1)[1].strip()
                        if data_str == "[DONE]":
                            break
                        if not data_str:
                            continue
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        # L'usage n'arrive que sur le dernier chunk, parfois sans « choices » :
                        # on le lit AVANT de regarder les choix.
                        if data.get("usage"):
                            usage = data["usage"]
                        choices = data.get("choices") or []
                        if not choices:
                            continue
                        delta = (choices[0] or {}).get("delta") or {}
                        content = delta.get("content")
                        if content:
                            last_token_ts = time.monotonic()
                            if isinstance(content, list):
                                # Mode raisonnement : delta.content est une LISTE de blocs
                                # {type: "thinking"|"text"}. Le thinking part en événement
                                # distinct, le texte final en « message ».
                                for part in content:
                                    if not isinstance(part, dict):
                                        continue
                                    ptype = part.get("type")
                                    if ptype == "text" and part.get("text"):
                                        yield json.dumps({"message": {"content": part["text"]}})
                                        has_yielded = True
                                    elif ptype == "thinking":
                                        inner = part.get("thinking")
                                        if isinstance(inner, list):
                                            think_txt = "".join(
                                                tc.get("text", "") for tc in inner if isinstance(tc, dict)
                                            )
                                        else:
                                            think_txt = inner if isinstance(inner, str) else ""
                                        if think_txt:
                                            yield json.dumps({"thinking": think_txt})
                            else:
                                yield json.dumps({"message": {"content": content}})
                                has_yielded = True
                        if time.monotonic() - last_token_ts > idle_break_seconds:
                            logger.warning("Mistral stream idle timeout (%ss)", idle_break_seconds)
                            break
                        if time.monotonic() - start_ts > max_duration_seconds:
                            logger.warning("Mistral stream max duration reached (%ss)", max_duration_seconds)
                            break
                    if usage:
                        yield json.dumps({"usage": usage})
                break
            except httpx.RequestError as exc:
                if has_yielded or attempt >= MAX_RETRIES:
                    logger.error(
                        "Mistral stream tentative %s/%s: %s (has_yielded=%s), abandon.",
                        attempt, MAX_RETRIES, exc, has_yielded,
                    )
                    raise
                wait = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                logger.warning("Mistral stream tentative %s/%s: %s, retry dans %.1fs", attempt, MAX_RETRIES, exc, wait)
                await asyncio.sleep(wait)
