"""Conversations, messages (lecture seule : le tour de chat les écrit) et retours utilisateurs."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select, func
from app.models.user import UserRead
from app.models.conversation import Conversation, ConversationCreate, ConversationRead, ConversationUpdate
from app.models.message import Message, MessageRead
from app.models.message_feedback import MessageFeedback, FeedbackCreate, FeedbackRead
from app.routers.auth import get_current_user
from app.database import get_session
from app.services.mistral_service import chat as mistral_chat
from app.config import settings
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _read(session: Session, conversation: Conversation) -> ConversationRead:
    count = session.exec(
        select(func.count(Message.id)).where(Message.conversation_id == conversation.id)
    ).first()
    conv_read = ConversationRead.model_validate(conversation)
    conv_read.message_count = count or 0
    return conv_read


def _owned(session: Session, conversation_id: int, user_id: int) -> Conversation:
    conversation = session.get(Conversation, conversation_id)
    if not conversation or conversation.user_id != user_id:
        raise HTTPException(status_code=404, detail="Conversation non trouvée")
    return conversation


@router.post("", response_model=ConversationRead, status_code=201)
async def create_conversation(
    conversation: ConversationCreate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    db_conversation = Conversation(
        title=conversation.title or "Nouvelle conversation",
        user_id=current_user.id,
    )
    session.add(db_conversation)
    session.commit()
    session.refresh(db_conversation)
    logger.info("Conversation créée : %s pour l'utilisateur %s", db_conversation.id, current_user.id)
    return _read(session, db_conversation)


@router.get("", response_model=List[ConversationRead])
async def list_conversations(
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    query = (
        select(Conversation, func.count(Message.id).label("message_count"))
        .outerjoin(Message)
        .where(Conversation.user_id == current_user.id)
        .group_by(Conversation.id)
        .order_by(Conversation.updated_at.desc())
    )
    conversations = []
    for conv, msg_count in session.exec(query).all():
        conv_read = ConversationRead.model_validate(conv)
        conv_read.message_count = msg_count or 0
        conversations.append(conv_read)
    return conversations


@router.get("/{conversation_id}", response_model=ConversationRead)
async def get_conversation(
    conversation_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    return _read(session, _owned(session, conversation_id, current_user.id))


@router.patch("/{conversation_id}", response_model=ConversationRead)
async def update_conversation(
    conversation_id: int,
    conversation_update: ConversationUpdate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    conversation = _owned(session, conversation_id, current_user.id)
    if conversation_update.title is not None:
        conversation.title = conversation_update.title
    conversation.updated_at = datetime.utcnow()
    session.add(conversation)
    session.commit()
    session.refresh(conversation)
    return _read(session, conversation)


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    conversation = _owned(session, conversation_id, current_user.id)
    for message in session.exec(select(Message).where(Message.conversation_id == conversation_id)).all():
        session.delete(message)
    session.delete(conversation)
    session.commit()
    logger.info("Conversation supprimée : %s", conversation_id)
    return {"ok": True, "message": "Conversation supprimée"}


@router.get("/{conversation_id}/messages", response_model=List[MessageRead])
async def list_messages(
    conversation_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _owned(session, conversation_id, current_user.id)
    return session.exec(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at, Message.id)
    ).all()


def _fallback_title(messages: List[Message]) -> str:
    first_user_msg = next((m for m in messages if m.role == "user"), None)
    if not first_user_msg:
        return "Conversation"
    title = first_user_msg.content[:30].strip()
    return title[:27] + "..." if len(title) > 30 else title


@router.post("/{conversation_id}/generate-title", response_model=ConversationRead)
async def generate_conversation_title(
    conversation_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Titre court (deux mots) généré par le modèle, seulement si le titre est encore
    automatique (« Nouvelle conversation », trop long ou tronqué)."""
    conversation = _owned(session, conversation_id, current_user.id)
    is_auto_generated = (
        conversation.title == "Nouvelle conversation"
        or len(conversation.title) > 30
        or conversation.title.endswith("...")
    )
    if not is_auto_generated:
        return _read(session, conversation)

    messages = session.exec(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
    ).all()
    if not messages:
        raise HTTPException(status_code=400, detail="La conversation n'a pas de messages")

    conversation_text = ""
    for msg in messages[:10]:
        role = "Utilisateur" if msg.role == "user" else "Assistant"
        conversation_text += f"{role}: {msg.content[:300]}\n"
        if len(conversation_text) > 2000:
            break

    prompt = f"""Analyse cette conversation et génère un titre très court (1 à 2 mots maximum) qui résume le sujet principal.

Conversation:
{conversation_text}

Réponds UNIQUEMENT avec le titre, sans explication, sans guillemets, sans ponctuation finale. Maximum 2 mots."""

    generated_title: Optional[str] = None
    try:
        if settings.MISTRAL_API_KEY:
            response = await mistral_chat(prompt, settings.MODEL_FAST, [{"role": "user", "content": prompt}])
            choices = response.get("choices") or []
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                    )
                generated_title = str(content or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Génération du titre en échec : %s", exc)
        generated_title = None

    if generated_title:
        generated_title = " ".join(generated_title.strip("\"'.,;:!?").split()[:2])
    if not generated_title or len(generated_title) < 2:
        generated_title = _fallback_title(messages)

    conversation.title = generated_title
    conversation.updated_at = datetime.utcnow()
    session.add(conversation)
    session.commit()
    session.refresh(conversation)
    logger.info("Titre généré pour la conversation %s : %s", conversation_id, generated_title)
    return _read(session, conversation)


# ---------------------------------------------------------------------------
# Retours utilisateurs
# ---------------------------------------------------------------------------


@router.post("/messages/{message_id}/feedback", response_model=FeedbackRead, status_code=201)
async def create_or_update_feedback(
    message_id: int,
    feedback_in: FeedbackCreate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Créer ou mettre à jour un retour (👍/👎) sur une réponse de LIA."""
    message = session.get(Message, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message non trouvé")
    if message.role != "assistant":
        raise HTTPException(status_code=400, detail="Seuls les messages de l'assistant peuvent recevoir du feedback")
    conversation = session.get(Conversation, message.conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Accès refusé")
    if not feedback_in.is_positive and (not feedback_in.comment or not feedback_in.comment.strip()):
        raise HTTPException(status_code=422, detail="Un commentaire est obligatoire pour un feedback négatif")

    user_msg = session.exec(
        select(Message)
        .where(
            Message.conversation_id == conversation.id,
            Message.role == "user",
            Message.created_at < message.created_at,
        )
        .order_by(Message.created_at.desc())
    ).first()
    query_text = user_msg.content if user_msg else ""

    existing = session.exec(
        select(MessageFeedback).where(
            MessageFeedback.message_id == message_id, MessageFeedback.user_id == current_user.id
        )
    ).first()
    now = datetime.utcnow()
    if existing:
        existing.is_positive = feedback_in.is_positive
        existing.comment = feedback_in.comment
        existing.category = feedback_in.category
        existing.query_text = query_text
        existing.response_text = message.content
        existing.updated_at = now
        session.add(existing)
        db_feedback = existing
    else:
        db_feedback = MessageFeedback(
            message_id=message_id,
            user_id=current_user.id,
            is_positive=feedback_in.is_positive,
            comment=feedback_in.comment,
            category=feedback_in.category,
            query_text=query_text,
            response_text=message.content,
            created_at=now,
            updated_at=now,
        )
        session.add(db_feedback)
    session.commit()
    session.refresh(db_feedback)
    return db_feedback


@router.get("/messages/{message_id}/feedback", response_model=Optional[FeedbackRead])
async def get_message_feedback(
    message_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    return session.exec(
        select(MessageFeedback).where(
            MessageFeedback.message_id == message_id, MessageFeedback.user_id == current_user.id
        )
    ).first()


@router.delete("/messages/{message_id}/feedback", status_code=204)
async def delete_message_feedback(
    message_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    feedback = session.exec(
        select(MessageFeedback).where(
            MessageFeedback.message_id == message_id, MessageFeedback.user_id == current_user.id
        )
    ).first()
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback non trouvé")
    session.delete(feedback)
    session.commit()
    return


@router.get("/feedbacks/mine")
async def get_my_feedbacks(
    page: int = 1,
    limit: int = 20,
    filter_type: Optional[str] = None,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Historique paginé des retours de l'utilisateur courant."""
    query = (
        select(
            MessageFeedback,
            Conversation.title.label("conversation_title"),
            Conversation.id.label("conversation_id"),
        )
        .outerjoin(Message, MessageFeedback.message_id == Message.id)
        .outerjoin(Conversation, Message.conversation_id == Conversation.id)
        .where(MessageFeedback.user_id == current_user.id)
    )
    total_query = select(func.count(MessageFeedback.id)).where(MessageFeedback.user_id == current_user.id)
    if filter_type == "positive":
        query = query.where(MessageFeedback.is_positive == True)  # noqa: E712
        total_query = total_query.where(MessageFeedback.is_positive == True)  # noqa: E712
    elif filter_type == "negative":
        query = query.where(MessageFeedback.is_positive == False)  # noqa: E712
        total_query = total_query.where(MessageFeedback.is_positive == False)  # noqa: E712
    query = query.order_by(MessageFeedback.created_at.desc())

    total = session.exec(total_query).first() or 0
    results = session.exec(query.offset((page - 1) * limit).limit(limit)).all()
    items = [
        {
            "id": feedback.id,
            "message_id": feedback.message_id,
            "conversation_id": conversation_id,
            "conversation_title": conversation_title,
            "is_positive": feedback.is_positive,
            "comment": feedback.comment,
            "query_text": feedback.query_text,
            "response_text": feedback.response_text,
            "category": feedback.category,
            "created_at": feedback.created_at.isoformat() if feedback.created_at else None,
            "updated_at": feedback.updated_at.isoformat() if feedback.updated_at else None,
        }
        for feedback, conversation_title, conversation_id in results
    ]
    return {"total": total, "page": page, "limit": limit, "items": items}
