"""Le tour de chat : le wiki entier dans le prompt, un appel, un flux, des citations vérifiées.

CAG et non RAG : aucune recherche, aucun extrait. Le prompt système (consignes + wiki) est le
même à chaque tour — c'est ce qui rend le cache Mistral efficace — et l'historique de la
conversation vient APRÈS lui, donc ne casse jamais le préfixe mis en cache.

Le seul contrôle de sortie est déterministe : les chemins de pages cités par le modèle
(``/dossier/page.md``, ce que les consignes lui demandent) sont résolus dans le wiki. Une
citation vers une page inexistante est rendue telle quelle mais marquée ``exists: false`` :
on constate et on montre, on ne réécrit pas la réponse.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from sqlmodel import Session, select

from app.config import settings
from app.models.message import Message
from app.services import wiki_service
from app.services.mistral_service import chat_stream
from app.services.wiki_service import WikiSnapshot

logger = logging.getLogger(__name__)

# Un chemin de page tel que les consignes demandent de le citer : « (/profiles/perform76-parcloses.md) ».
CITATION_RE = re.compile(r"(?<![\w/.])(/(?:[a-z0-9_-]+/)*[a-z0-9_-]+\.md)(?![\w/])")
# Les identifiants des trois registres d'anomalies.
ANOMALY_RE = re.compile(r"\b(INC|CTR|VER)-(\d{1,3})\b")
ANOMALY_PAGES = {
    "INC": "/anomalies/incoherences-internes.md",
    "CTR": "/anomalies/contradictions-entre-sources.md",
    "VER": "/anomalies/informations-a-verifier.md",
}


def sse(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def extract_citations(text: str, snapshot: WikiSnapshot) -> List[Dict[str, Any]]:
    """Les pages citées dans la réponse, dans l'ordre d'apparition, dédoublonnées, vérifiées."""
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for match in CITATION_RE.finditer(text or ""):
        path = match.group(1)
        if path in seen:
            continue
        seen.add(path)
        page = snapshot.pages.get(path)
        if page is not None and not page.missing:
            out.append(
                {
                    "path": path,
                    "title": page.title,
                    "type": page.type,
                    "status": page.status,
                    "exists": True,
                }
            )
        else:
            out.append(
                {
                    "path": path,
                    "title": path.rsplit("/", 1)[-1][:-3].replace("-", " "),
                    "type": "",
                    "status": "",
                    "exists": False,
                }
            )
    return out


def extract_anomalies(text: str) -> List[Dict[str, str]]:
    """Les identifiants d'anomalie mentionnés, avec la page registre qui les porte."""
    out: List[Dict[str, str]] = []
    seen: set = set()
    for match in ANOMALY_RE.finditer(text or ""):
        ident = f"{match.group(1)}-{match.group(2)}"
        if ident in seen:
            continue
        seen.add(ident)
        out.append({"id": ident, "path": ANOMALY_PAGES[match.group(1)]})
    return out


def load_history(session: Session, conversation_id: int) -> List[Dict[str, str]]:
    """Les derniers échanges de la conversation, bornés en nombre et en caractères, et
    commençant par un message utilisateur (Mistral l'exige après le système)."""
    rows = session.exec(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at, Message.id)
    ).all()
    items = [
        {"role": m.role, "content": m.content}
        for m in rows
        if m.role in ("user", "assistant") and (m.content or "").strip()
    ]
    items = items[-settings.CHAT_HISTORY_MAX_MESSAGES :] if settings.CHAT_HISTORY_MAX_MESSAGES > 0 else []
    while items and sum(len(i["content"]) for i in items) > settings.CHAT_HISTORY_MAX_CHARS:
        items.pop(0)
    while items and items[0]["role"] != "user":
        items.pop(0)
    return items


def _cached_tokens(usage: Optional[Dict[str, Any]]) -> Optional[int]:
    if not usage:
        return None
    for key in ("cached_tokens", "prompt_cached_tokens", "cache_read_input_tokens"):
        if isinstance(usage.get(key), int):
            return usage[key]
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
        return details["cached_tokens"]
    return None


StreamFn = Callable[..., AsyncIterator[str]]


@dataclass
class WikiAnswer:
    """Une instance par tour. ``run()`` est un générateur d'événements SSE ; l'état final
    (texte, raisonnement, sources, trace) se lit sur l'instance ensuite."""

    question: str
    history: List[Dict[str, str]]
    snapshot: WikiSnapshot
    model: str = ""
    # Résolu à l'exécution (``chat_stream`` du module) pour rester remplaçable dans les tests.
    stream_fn: Optional[StreamFn] = None

    text: str = ""
    thinking: str = ""
    usage: Optional[Dict[str, Any]] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    anomalies: List[Dict[str, str]] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            self.model = settings.MODEL_FAST

    def messages(self) -> List[Dict[str, str]]:
        return (
            [{"role": "system", "content": self.snapshot.system_prompt}]
            + list(self.history)
            + [{"role": "user", "content": self.question}]
        )

    async def run(self) -> AsyncIterator[str]:
        t0 = time.perf_counter()
        first_token_ms: Optional[int] = None
        yield sse({"stage": {"key": "wiki", "label": "Lecture du wiki"}})

        extra: Dict[str, Any] = {"prompt_cache_key": self.snapshot.cache_key}
        if settings.GENERATION_REASONING_EFFORT:
            extra["reasoning_effort"] = settings.GENERATION_REASONING_EFFORT

        text_parts: List[str] = []
        think_parts: List[str] = []
        stream_fn = self.stream_fn or chat_stream
        async for raw in stream_fn(
            "",
            model=self.model,
            context=self.messages(),
            max_tokens=settings.CHAT_MAX_TOKENS,
            temperature=settings.CHAT_TEMPERATURE,
            **extra,
        ):
            try:
                data = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            thinking = data.get("thinking")
            if thinking:
                think_parts.append(thinking)
                yield sse({"thinking": thinking})
                continue
            if data.get("usage"):
                self.usage = data["usage"]
                continue
            content = (data.get("message") or {}).get("content")
            if not content:
                continue
            if first_token_ms is None:
                first_token_ms = int((time.perf_counter() - t0) * 1000)
            text_parts.append(content)
            yield sse({"message": {"content": content}})

        self.text = "".join(text_parts).strip()
        self.thinking = "".join(think_parts)
        self.sources = extract_citations(self.text, self.snapshot)
        self.anomalies = extract_anomalies(self.text)
        usage = self.usage or {}
        self.trace = {
            "model": self.model,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": _cached_tokens(usage),
            "completion_tokens": usage.get("completion_tokens"),
            "first_token_ms": first_token_ms,
            "duration_ms": int((time.perf_counter() - t0) * 1000),
            "wiki_hash": self.snapshot.cache_key,
            "wiki_pages": len(self.snapshot.concept_pages),
            "history_messages": len(self.history),
            "cited_pages": [s["path"] for s in self.sources if s["exists"]],
            "unknown_citations": [s["path"] for s in self.sources if not s["exists"]],
            "anomalies": [a["id"] for a in self.anomalies],
        }
        wiki_service.record_call(
            {
                "at": datetime.now().isoformat(timespec="seconds"),
                "model": self.model,
                "prompt_tokens": self.trace["prompt_tokens"],
                "cached_tokens": self.trace["cached_tokens"],
                "completion_tokens": self.trace["completion_tokens"],
                "first_token_ms": first_token_ms,
                "duration_ms": self.trace["duration_ms"],
            }
        )
        if self.trace["unknown_citations"]:
            logger.info(
                "[chat] citations vers des pages inexistantes : %s",
                ", ".join(self.trace["unknown_citations"]),
            )
        yield sse({"sources": self.sources, "anomalies": self.anomalies})
