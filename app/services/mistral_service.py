import asyncio
import httpx
import json
import random
from typing import List, Dict, Optional, Any, Tuple
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
                    logger.error(
                        "Mistral API error (%s): %s",
                        response.status_code,
                        response.text[:2000],
                    )
                response.raise_for_status()
                return response
            
            wait = _retry_wait_seconds(response, attempt) + random.uniform(0, 0.5)
            logger.warning(
                "Mistral chat tentative %s/%s: HTTP %s, retry dans %.1fs",
                attempt,
                MAX_RETRIES,
                response.status_code,
                wait,
            )
        except httpx.RequestError as exc:
            if attempt >= MAX_RETRIES:
                logger.error(
                    "Mistral chat tentative %s/%s: Erreur réseau %s, fin des tentatives.",
                    attempt,
                    MAX_RETRIES,
                    exc,
                )
                raise
            wait = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(
                "Mistral chat tentative %s/%s: Erreur réseau %s, retry dans %.1fs",
                attempt,
                MAX_RETRIES,
                exc,
                wait,
            )
        
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

    # Nettoyage pour conformité API Mistral
    messages = _clean_messages(messages)

    role_seq = "-".join([m["role"][0].upper() for m in messages])
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    prompt_preview = str(messages[-1].get("content", ""))[:100]
    logger.info(
        "[MISTRAL] Appel sync — model=%s, nb_msg=%d, roles=%s, chars=%d, prompt='%s'...",
        model,
        len(messages),
        role_seq,
        total_chars,
        prompt_preview,
    )

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
            headers = {
                "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            }
            logger.info("[MISTRAL] Connexion sync (%s)...", base_url)
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await _post_json_with_retry(
                    client,
                    f"{base_url}/v1/chat/completions",
                    headers=headers,
                    payload=payload,
                )
                data = response.json()
                usage = data.get("usage") or {}
                logger.info(
                    "[MISTRAL] Sync OK — model=%s, status=%s, prompt_tokens=%s, completion_tokens=%s",
                    model,
                    response.status_code,
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                )
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
    """Nettoie la liste de messages pour Mistral:
    1. Fusionne les messages système consécutifs au début.
    2. Fusionne les messages consécutifs du même rôle (user/user, assistant/assistant).
    3. S'assure que l'ordre est respecté (system? -> user -> assistant -> user...).
    """
    logger.info(
        "[_clean_messages] Cleaning %d messages before sending to Mistral API...",
        len(messages),
    )
    if not messages:
        return []
    
    cleaned = []
    
    # 1. Gérer le système
    system_content = []
    idx = 0
    while idx < len(messages) and messages[idx].get("role") == "system":
        content = messages[idx].get("content", "")
        if isinstance(content, list):
            text_parts = [part["text"] for part in content if isinstance(part, dict) and part.get("type") == "text"]
            content = "\n\n".join(text_parts)
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
        
        # S'assurer que content est une string propre
        if isinstance(content, list):
            text_parts = [part["text"] for part in content if isinstance(part, dict) and part.get("type") == "text"]
            content_str = "\n\n".join(text_parts)
        else:
            content_str = str(content)
            
        images = msg.get("images") or []
        
        if not content_str.strip() and not images:
            continue
            
        if cleaned and cleaned[-1]["role"] == role:
            # Même rôle que le précédent, on fusionne
            existing_content = cleaned[-1]["content"]
            existing_is_list = isinstance(existing_content, list)
            
            if existing_is_list or images:
                # Normaliser l'existant en liste de parties
                if existing_is_list:
                    parts = list(existing_content)
                else:
                    parts = [{"type": "text", "text": str(existing_content)}]
                
                # Ajouter la nouvelle partie texte
                parts.append({"type": "text", "text": "\n\n" + content_str})
                
                # Ajouter les nouvelles images
                for img in images:
                    url = img if img.startswith("data:") else f"data:image/png;base64,{img}"
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                
                cleaned[-1]["content"] = parts
            else:
                # Fusion classique simple en string
                cleaned[-1]["content"] += "\n\n" + content_str
        else:
            # Nouveau message
            if images:
                parts = [{"type": "text", "text": content_str}]
                for img in images:
                    url = img if img.startswith("data:") else f"data:image/png;base64,{img}"
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                cleaned.append({"role": role, "content": parts})
            else:
                cleaned.append({"role": role, "content": content_str})
    
    # Mistral demande que ça commence par user (si pas de system) ou que ça suive system
    # Si le premier après system est un assistant, on l'ignore ou on l'insère après un user vide
    if cleaned and cleaned[0]["role"] == "system":
        if len(cleaned) > 1 and cleaned[1]["role"] == "assistant":
            cleaned.insert(1, {"role": "user", "content": "(Suite de la conversation)"})
    elif cleaned and cleaned[0]["role"] == "assistant":
        cleaned.insert(0, {"role": "user", "content": "(Début de la conversation)"})
        
    # Log the summary of cleaned messages
    for idx_msg, m in enumerate(cleaned):
        role = m.get("role")
        c = m.get("content")
        if isinstance(c, list):
            num_images = sum(1 for part in c if part.get("type") == "image_url")
            text_lens = sum(len(part.get("text", "")) for part in c if part.get("type") == "text")
            logger.info(
                "[_clean_messages] Msg #%d: role=%s, content=LIST containing %d images (base64) successfully prepared for LLM vision! (text length: %d)",
                idx_msg,
                role,
                num_images,
                text_lens,
            )
        else:
            logger.info(
                "[_clean_messages] Msg #%d: role=%s, content=STRING [len=%d]",
                idx_msg,
                role,
                len(str(c)),
            )
        
    return cleaned


async def chat_collect_reasoning(
    message: str = "",
    *,
    model: str,
    context: Optional[List[Dict]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    response_format: Optional[Dict] = None,
) -> Tuple[str, str]:
    """Exécute ``chat_stream`` (avec reasoning optionnel) mais COLLECTE le résultat au lieu
    de le streamer : retourne ``(texte, thinking)``. Pour les appels qui veulent le
    raisonnement ET une sortie complète en un bloc (ex. JSON structuré du routeur guidé),
    sans exposer le stream — le thinking est ensuite affiché par l'appelant."""
    extra: Dict[str, Any] = {}
    if reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort
    if response_format is not None:
        extra["response_format"] = response_format

    text_parts: List[str] = []
    think_parts: List[str] = []
    async for raw in chat_stream(
        message, model=model, context=context, temperature=temperature,
        max_tokens=max_tokens, **extra,
    ):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if data.get("thinking"):
            think_parts.append(data["thinking"])
        else:
            content = (data.get("message") or {}).get("content")
            if content:
                text_parts.append(content)
    return "".join(text_parts), "".join(think_parts)


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
        logger.info(f"[MISTRAL] Appel stream — model={model}, nb_msg={len(messages)}, roles={role_seq}, chars={total_chars}, prompt='{prompt_preview}'...")

        base_url = (settings.MISTRAL_BASE_URL or "https://api.mistral.ai").rstrip("/")
        timeout = httpx.Timeout(400.0, connect=60.0)

        headers = {
            "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        }
        completions_url = f"{base_url}/v1/chat/completions"

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                logger.info(f"[MISTRAL] Connexion stream ({base_url})...")
                has_yielded = False
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        async with client.stream(
                            "POST",
                            completions_url,
                            headers=headers,
                            json=payload,
                        ) as response:
                            logger.info(f"[MISTRAL] Stream status: {response.status_code}")
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
                                                if isinstance(content, list):
                                                    # Mode reasoning : pendant la phase de réflexion,
                                                    # delta.content est une LISTE de chunks
                                                    # {type: "thinking"|"text"}. Le thinking est émis
                                                    # comme événement DISTINCT ({"thinking": ...}) — le
                                                    # client l'affiche puis le masque à l'arrivée de la
                                                    # réponse ; le texte final passe en {"message": ...}.
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
                                                                    tc.get("text", "") for tc in inner
                                                                    if isinstance(tc, dict)
                                                                )
                                                            else:
                                                                think_txt = inner if isinstance(inner, str) else ""
                                                            if think_txt:
                                                                yield json.dumps({"thinking": think_txt})
                                                else:
                                                    yield json.dumps({"message": {"content": content}})
                                                    has_yielded = True

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
                    except httpx.RequestError as exc:
                        if has_yielded or attempt >= MAX_RETRIES:
                            logger.error(
                                "Mistral stream tentative %s/%s: Exception %s (has_yielded=%s), fin des tentatives.",
                                attempt,
                                MAX_RETRIES,
                                exc,
                                has_yielded,
                            )
                            raise
                        wait = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                        logger.warning(
                            "Mistral stream tentative %s/%s: Exception %s, retry dans %.1fs",
                            attempt,
                            MAX_RETRIES,
                            exc,
                            wait,
                        )
                        await asyncio.sleep(wait)

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
