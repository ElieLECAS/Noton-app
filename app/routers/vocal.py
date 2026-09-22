"""L'assistant vocal : on parle, LIA cherche dans le wiki et répond de vive voix.

``POST /api/vocal/tour?conversation_id=N`` reçoit soit l'enregistrement (corps ``audio/wav``,
16 kHz mono 16 bits, produit par le navigateur), soit une question écrite (corps JSON
``{"texte": ...}``), et rend un flux SSE : ``transcription`` → les événements du tour de chat
(``etape``, ``thinking``, ``message``, ``reset``, ``sources``) entrelacés avec ``phrase`` et
``audio`` (float32 24 kHz, base64) → ``done``. En cas d'échec : ``error``.

La question transcrite est persistée comme un message utilisateur, la réponse comme un message
assistant portant ses sources et sa trace (mesures vocales comprises) — même si l'utilisateur
a coupé la parole : le texte était final dès la fin du tour de chat. Pas de multipart : le
corps de la requête est l'audio lui-même.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.config import settings
from app.database import engine, get_session
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.services import vocal_service
from app.services.mistral_service import MistralRateLimitError
from app.services.vocal_service import AUDIO_MAX_OCTETS, TourVocal, TranscriptionImpossible
from app.services.wiki_chat_service import WikiAnswer, load_history, sse
from app.services.wiki_service import WikiUnavailable, get_snapshot

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vocal", tags=["vocal"])

TYPES_AUDIO = frozenset({
    "audio/wav", "audio/x-wav", "audio/wave", "audio/webm", "audio/ogg", "audio/mpeg", "audio/mp4",
})


def _persister(conversation_id: int, role: str, content: str, **champs: Any) -> Optional[int]:
    """Écrit un message dans une session neuve — celle de la requête n'est pas garantie vivante
    quand le flux avance — et rafraîchit la date de la conversation."""
    try:
        with Session(engine) as session:
            message = Message(conversation_id=conversation_id, role=role, content=content, **champs)
            session.add(message)
            conversation = session.get(Conversation, conversation_id)
            if conversation is not None:
                conversation.updated_at = datetime.utcnow()
                session.add(conversation)
            session.commit()
            session.refresh(message)
            return message.id
    except Exception:  # noqa: BLE001
        logger.exception("[vocal] persistance impossible (%s)", role)
        return None


@router.post("/tour")
async def tour_vocal(
    request: Request,
    conversation_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    conversation = session.get(Conversation, conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation non trouvée")
    try:
        snapshot = get_snapshot()
    except WikiUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    type_contenu = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    corps = await request.body()
    texte_ecrit: Optional[str] = None
    audio: Optional[bytes] = None
    if type_contenu == "application/json":
        try:
            texte_ecrit = str((json.loads(corps or b"{}") or {}).get("texte") or "").strip()
        except (json.JSONDecodeError, AttributeError):
            raise HTTPException(status_code=422, detail="JSON illisible")
        if not texte_ecrit:
            raise HTTPException(status_code=422, detail="Question vide")
    elif type_contenu in TYPES_AUDIO:
        if not corps:
            raise HTTPException(status_code=422, detail="Enregistrement vide")
        if len(corps) > AUDIO_MAX_OCTETS:
            raise HTTPException(status_code=413, detail="Enregistrement trop long")
        audio = corps
    else:
        raise HTTPException(
            status_code=415, detail="Corps attendu : audio/wav (l'enregistrement) ou application/json {texte}"
        )

    history = load_history(session, conversation.id)
    biais = vocal_service.biais_vocabulaire(snapshot.index)
    conv_id, user_id = conversation.id, current_user.id

    async def generate():
        answer: Optional[WikiAnswer] = None
        tour: Optional[TourVocal] = None
        transcription: Dict[str, Any] = {}
        message_id: Optional[int] = None
        trace: Dict[str, Any] = {}
        try:
            if audio is not None:
                try:
                    transcription = await vocal_service.transcrire(audio, type_contenu, biais)
                except TranscriptionImpossible as exc:
                    yield sse({"error": str(exc)})
                    return
                question = transcription["texte"]
                yield sse({"transcription": {"texte": question, "ms": transcription["ms"]}})
                if not question:
                    yield sse({"done": True, "vide": True})
                    return
            else:
                question = texte_ecrit or ""
                yield sse({"transcription": {"texte": question, "ecrite": True}})

            _persister(conv_id, "user", question)
            logger.info(
                "[vocal] tour — conversation=%s user=%s question=%d car. historique=%d msgs",
                conv_id, user_id, len(question), len(history),
            )
            answer = WikiAnswer(
                question=question,
                history=history,
                snapshot=snapshot,
                model=settings.MODEL_FAST,
                system_prompt=snapshot.vocal_prompt,
                cache_key=snapshot.vocal_cache_key,
                images=False,
            )
            tour = TourVocal(answer=answer)
            async for event in tour.run():
                yield event
        except MistralRateLimitError as exc:
            yield sse({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("[vocal] tour en échec")
            yield sse({"error": f"Génération impossible : {exc}"})
        finally:
            if answer is not None and answer.text:
                trace = dict(answer.trace)
                trace["vocal"] = {
                    **(tour.mesures if tour is not None else {}),
                    "transcription_ms": transcription.get("ms"),
                    "modele_transcription": settings.VOCAL_MODELE_TRANSCRIPTION if audio is not None else None,
                }
                message_id = _persister(
                    conv_id, "assistant", answer.text,
                    model=answer.model,
                    sources=json.dumps(answer.sources, ensure_ascii=False),
                    metadata_json={"trace": trace, "anomalies": answer.anomalies, "vocal": True},
                )
        if answer is not None and answer.text:
            yield sse({"done": True, "message_id": message_id, "trace": trace})

    return StreamingResponse(generate(), media_type="text/event-stream")
