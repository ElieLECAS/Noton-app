"""Le chat : une question, le wiki entier, une réponse streamée, ses pages citées.

Événements SSE (``data: {...}``), dans l'ordre : ``stage`` → ``thinking``* → ``message``* →
``sources`` → ``done``. En cas d'échec : ``error``. Le message utilisateur est persisté avant
l'appel ; la réponse ne l'est que si elle existe.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session

from app.config import settings
from app.database import engine, get_session
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.services.mistral_service import MistralRateLimitError
from app.services.wiki_chat_service import WikiAnswer, load_history, sse
from app.services.wiki_service import WikiUnavailable, get_snapshot

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    conversation_id: int


def _persist_reply(conversation_id: int, answer: WikiAnswer) -> Optional[int]:
    """Persiste la réponse dans une session NEUVE : celle de la requête n'est pas garantie
    vivante quand le flux se termine."""
    if not answer.text:
        return None
    try:
        with Session(engine) as session:
            reply = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer.text,
                model=answer.model,
                provider="mistral",
                sources=json.dumps(answer.sources, ensure_ascii=False),
                metadata_json={"trace": answer.trace, "anomalies": answer.anomalies},
            )
            session.add(reply)
            conversation = session.get(Conversation, conversation_id)
            if conversation is not None:
                conversation.updated_at = datetime.utcnow()
                session.add(conversation)
            session.commit()
            session.refresh(reply)
            return reply.id
    except Exception:  # noqa: BLE001
        logger.exception("[chat] persistance de la réponse impossible")
        return None


@router.post("/stream")
async def stream_chat(
    request: ChatRequest,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    message = (request.message or "").strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message vide")
    conversation = session.get(Conversation, request.conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation non trouvée")
    try:
        snapshot = get_snapshot()
    except WikiUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    history = load_history(session, conversation.id)
    session.add(Message(conversation_id=conversation.id, role="user", content=message))
    conversation.updated_at = datetime.utcnow()
    session.add(conversation)
    session.commit()

    logger.info(
        "[chat] tour — conversation=%s user=%s question=%d car. historique=%d msgs wiki=%s",
        conversation.id,
        current_user.id,
        len(message),
        len(history),
        snapshot.cache_key,
    )
    answer = WikiAnswer(question=message, history=history, snapshot=snapshot, model=settings.MODEL_FAST)
    conversation_id = conversation.id

    async def generate():
        try:
            async for event in answer.run():
                yield event
            message_id = _persist_reply(conversation_id, answer)
            yield sse({"done": True, "message_id": message_id, "trace": answer.trace})
        except MistralRateLimitError as exc:
            yield sse({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("[chat] génération en échec")
            yield sse({"error": f"Génération impossible : {exc}"})

    return StreamingResponse(generate(), media_type="text/event-stream")
