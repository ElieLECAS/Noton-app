from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select, func
from app.models.user import UserRead
from app.models.conversation import Conversation, ConversationCreate, ConversationRead, ConversationUpdate
from app.models.message import Message, MessageCreate, MessageRead
from app.models.space import Space
from app.models.message_feedback import MessageFeedback, FeedbackCreate, FeedbackRead
from app.services.space_service import get_space_by_id
from app.routers.auth import get_current_user
from app.database import get_session
from app.services.mistral_service import chat as mistral_chat
from app.config import settings
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.post("", response_model=ConversationRead, status_code=201)
async def create_conversation(
    conversation: ConversationCreate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Créer une nouvelle conversation"""
    # Vérifier que l'espace est accessible (partagé ou appartenant à l'utilisateur)
    space = get_space_by_id(session, conversation.space_id, current_user.id)
    if not space:
        raise HTTPException(status_code=404, detail="Espace non trouvé")
    
    db_conversation = Conversation(
        title=conversation.title or "Nouvelle conversation",
        user_id=current_user.id,
        space_id=conversation.space_id
    )
    session.add(db_conversation)
    session.commit()
    session.refresh(db_conversation)
    
    # Ajouter le compteur de messages
    conversation_read = ConversationRead.from_orm(db_conversation)
    conversation_read.message_count = 0
    
    logger.info(f"Conversation créée: {db_conversation.id} pour l'utilisateur {current_user.id}")
    return conversation_read


@router.get("", response_model=List[ConversationRead])
async def list_conversations(
    space_id: int = None,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Lister les conversations de l'utilisateur (optionnellement filtrées par espace)."""
    query = select(
        Conversation,
        func.count(Message.id).label("message_count")
    ).outerjoin(Message).where(
        Conversation.user_id == current_user.id
    )
    
    if space_id is not None:
        query = query.where(Conversation.space_id == space_id)
    
    query = query.group_by(Conversation.id).order_by(Conversation.updated_at.desc())
    
    results = session.exec(query).all()
    
    conversations = []
    for conv, msg_count in results:
        conv_read = ConversationRead.from_orm(conv)
        conv_read.message_count = msg_count or 0
        conversations.append(conv_read)
    
    return conversations


@router.get("/{conversation_id}", response_model=ConversationRead)
async def get_conversation(
    conversation_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupérer une conversation spécifique"""
    conversation = session.get(Conversation, conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation non trouvée")
    
    # Compter les messages
    message_count = session.exec(
        select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
    ).first()
    
    conv_read = ConversationRead.from_orm(conversation)
    conv_read.message_count = message_count or 0
    
    return conv_read


@router.patch("/{conversation_id}", response_model=ConversationRead)
async def update_conversation(
    conversation_id: int,
    conversation_update: ConversationUpdate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Mettre à jour une conversation (titre, etc.)"""
    conversation = session.get(Conversation, conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation non trouvée")
    
    if conversation_update.title is not None:
        conversation.title = conversation_update.title
    
    conversation.updated_at = datetime.utcnow()
    
    session.add(conversation)
    session.commit()
    session.refresh(conversation)
    
    # Compter les messages
    message_count = session.exec(
        select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
    ).first()
    
    conv_read = ConversationRead.from_orm(conversation)
    conv_read.message_count = message_count or 0
    
    return conv_read


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Supprimer une conversation et ses messages"""
    conversation = session.get(Conversation, conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation non trouvée")
    
    # Supprimer explicitement tous les messages de la conversation
    # Cela évite les problèmes avec SQLAlchemy qui pourrait essayer de mettre conversation_id à None
    messages = session.exec(
        select(Message).where(Message.conversation_id == conversation_id)
    ).all()
    for message in messages:
        session.delete(message)
    
    # Ensuite supprimer la conversation
    session.delete(conversation)
    session.commit()
    
    logger.info(f"Conversation supprimée: {conversation_id}")
    return {"ok": True, "message": "Conversation supprimée"}


@router.get("/{conversation_id}/messages", response_model=List[MessageRead])
async def list_messages(
    conversation_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Lister les messages d'une conversation"""
    # Vérifier que la conversation appartient à l'utilisateur
    conversation = session.get(Conversation, conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation non trouvée")
    
    statement = select(Message).where(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at)
    
    messages = session.exec(statement).all()
    return messages


@router.post("/{conversation_id}/messages", response_model=MessageRead)
async def create_message(
    conversation_id: int,
    message: MessageCreate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Créer un nouveau message dans une conversation"""
    # Vérifier que la conversation appartient à l'utilisateur
    conversation = session.get(Conversation, conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation non trouvée")
    
    # S'assurer que le message est pour la bonne conversation
    if message.conversation_id != conversation_id:
        raise HTTPException(status_code=400, detail="ID de conversation incohérent")
    
    db_message = Message(**message.dict())
    session.add(db_message)
    
    # Mettre à jour la date de modification de la conversation
    conversation.updated_at = datetime.utcnow()
    
    # Générer un titre automatique si c'est le premier message utilisateur
    if conversation.title == "Nouvelle conversation":
        first_user_msg = session.exec(
            select(Message).where(
                Message.conversation_id == conversation_id,
                Message.role == "user"
            ).order_by(Message.created_at)
        ).first()
        
        if not first_user_msg:  # C'est le premier message utilisateur
            # Générer un titre à partir du contenu
            title = message.content[:50] + "..." if len(message.content) > 50 else message.content
            conversation.title = title
    
    session.add(conversation)
    session.commit()
    session.refresh(db_message)
    
    return db_message


@router.post("/{conversation_id}/generate-title", response_model=ConversationRead)
async def generate_conversation_title(
    conversation_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Générer automatiquement un titre pour une conversation basé sur ses messages"""
    # Vérifier que la conversation appartient à l'utilisateur
    conversation = session.get(Conversation, conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation non trouvée")
    
    # Ne générer un titre que si c'est encore "Nouvelle conversation" ou si c'est un titre auto-généré (trop long ou se termine par "...")
    # On considère qu'un titre auto-généré fait plus de 30 caractères ou se termine par "..."
    is_auto_generated = (
        conversation.title == "Nouvelle conversation" or
        len(conversation.title) > 30 or
        conversation.title.endswith("...")
    )
    
    if not is_auto_generated:
        # Retourner la conversation telle quelle si elle a déjà un titre personnalisé
        message_count = session.exec(
            select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
        ).first()
        conv_read = ConversationRead.from_orm(conversation)
        conv_read.message_count = message_count or 0
        return conv_read
    
    # Récupérer les messages de la conversation
    messages = session.exec(
        select(Message).where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    ).all()
    
    if not messages or len(messages) == 0:
        raise HTTPException(status_code=400, detail="La conversation n'a pas de messages")
    
    # Construire le contexte pour le modèle IA
    # Prendre les premiers messages (limiter à 2000 caractères pour éviter les prompts trop longs)
    conversation_text = ""
    for msg in messages[:10]:  # Limiter aux 10 premiers messages
        role = "Utilisateur" if msg.role == "user" else "Assistant"
        content = msg.content[:300]  # Limiter chaque message à 300 caractères
        conversation_text += f"{role}: {content}\n"
        if len(conversation_text) > 2000:
            break
    
    # Prompt pour générer un titre court
    prompt = f"""Analyse cette conversation et génère un titre très court (1 à 2 mots maximum) qui résume le sujet principal.

Conversation:
{conversation_text}

Réponds UNIQUEMENT avec le titre, sans explication, sans guillemets, sans ponctuation finale. Maximum 2 mots."""

    try:
        if not settings.MISTRAL_API_KEY:
            generated_title = None
        else:
            response = await mistral_chat(prompt, settings.MODEL_FAST, [{"role": "user", "content": prompt}])
            if "choices" in response and len(response["choices"]) > 0:
                generated_title = response["choices"][0]["message"].get("content", "").strip()
            else:
                generated_title = None
        
        # Nettoyer le titre (enlever guillemets, points, etc.)
        if generated_title:
            generated_title = generated_title.strip('"\'.,;:!?')
            # Limiter à 2 mots
            words = generated_title.split()[:2]
            generated_title = " ".join(words)
            
            # Si le titre est vide ou trop court, utiliser un titre par défaut basé sur le premier message
            if not generated_title or len(generated_title) < 2:
                first_user_msg = next((msg for msg in messages if msg.role == "user"), None)
                if first_user_msg:
                    generated_title = first_user_msg.content[:30].strip()
                    if len(generated_title) > 30:
                        generated_title = generated_title[:27] + "..."
                else:
                    generated_title = "Conversation"
        else:
            # Fallback : utiliser le début du premier message utilisateur
            first_user_msg = next((msg for msg in messages if msg.role == "user"), None)
            if first_user_msg:
                generated_title = first_user_msg.content[:30].strip()
                if len(generated_title) > 30:
                    generated_title = generated_title[:27] + "..."
            else:
                generated_title = "Conversation"
        
        # Mettre à jour le titre de la conversation
        conversation.title = generated_title
        conversation.updated_at = datetime.utcnow()
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
        
        logger.info(f"Titre généré pour la conversation {conversation_id}: {generated_title}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération du titre: {e}")
        # En cas d'erreur, utiliser un titre basé sur le premier message
        first_user_msg = next((msg for msg in messages if msg.role == "user"), None)
        if first_user_msg:
            conversation.title = first_user_msg.content[:30].strip()
            if len(conversation.title) > 30:
                conversation.title = conversation.title[:27] + "..."
        else:
            conversation.title = "Conversation"
        conversation.updated_at = datetime.utcnow()
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
    
    # Compter les messages
    message_count = session.exec(
        select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
    ).first()
    
    conv_read = ConversationRead.from_orm(conversation)
    conv_read.message_count = message_count or 0
    
    return conv_read


@router.post("/messages/{message_id}/feedback", response_model=FeedbackRead, status_code=201)
async def create_or_update_feedback(
    message_id: int,
    feedback_in: FeedbackCreate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Créer ou mettre à jour un feedback (👍/👎) sur un message de l'assistant."""
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
    
    # Récupérer la requête utilisateur précédente
    user_msg = session.exec(
        select(Message)
        .where(
            Message.conversation_id == conversation.id,
            Message.role == "user",
            Message.created_at < message.created_at
        )
        .order_by(Message.created_at.desc())
    ).first()
    query_text = user_msg.content if user_msg else ""
    
    # Extraire les chunk_ids
    chunk_ids = []
    if message.sources:
        try:
            sources_list = json.loads(message.sources)
            if isinstance(sources_list, list):
                for s in sources_list:
                    cid = s.get("source_leaf_chunk_id") or s.get("chunk_id")
                    if cid and isinstance(cid, int):
                        chunk_ids.append(cid)
        except Exception as e:
            logger.warning(f"Erreur parsing sources pour message {message.id}: {e}")
            
    existing_feedback = session.exec(
        select(MessageFeedback)
        .where(MessageFeedback.message_id == message_id, MessageFeedback.user_id == current_user.id)
    ).first()
    
    now = datetime.utcnow()
    if existing_feedback:
        # Garde-fou : ne relancer Celery que si commentaire modifié et FAQ pas encore générée
        comment_changed = existing_feedback.comment != feedback_in.comment
        should_regenerate_faq = (
            not existing_feedback.is_positive 
            and not existing_feedback.auto_faq_generated 
            and comment_changed
        )
        
        existing_feedback.is_positive = feedback_in.is_positive
        existing_feedback.comment = feedback_in.comment
        existing_feedback.query_text = query_text
        existing_feedback.response_text = message.content
        existing_feedback.chunk_ids = chunk_ids
        existing_feedback.updated_at = now
        session.add(existing_feedback)
        db_feedback = existing_feedback
    else:
        should_regenerate_faq = not feedback_in.is_positive  # Nouveau feedback négatif
        db_feedback = MessageFeedback(
            message_id=message_id,
            user_id=current_user.id,
            space_id=conversation.space_id,
            is_positive=feedback_in.is_positive,
            comment=feedback_in.comment,
            query_text=query_text,
            response_text=message.content,
            chunk_ids=chunk_ids,
            created_at=now,
            updated_at=now
        )
        session.add(db_feedback)
        
    session.commit()
    session.refresh(db_feedback)

    # Déclencher la génération de FAQ corrective si nécessaire et permission OK
    if should_regenerate_faq and "feedback.auto_faq" in current_user.permissions:
        try:
            from app.tasks.documents import generate_faq_from_feedback_task
            generate_faq_from_feedback_task.delay(db_feedback.id)
            logger.info(f"Tâche Celery de génération FAQ planifiée pour le feedback {db_feedback.id}")
        except Exception as e:
            logger.error(f"Impossible de planifier la tâche Celery de génération FAQ : {e}")

    return db_feedback


@router.get("/messages/{message_id}/feedback", response_model=Optional[FeedbackRead])
async def get_message_feedback(
    message_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupérer le feedback du user courant sur un message."""
    feedback = session.exec(
        select(MessageFeedback)
        .where(MessageFeedback.message_id == message_id, MessageFeedback.user_id == current_user.id)
    ).first()
    return feedback


@router.delete("/messages/{message_id}/feedback", status_code=204)
async def delete_message_feedback(
    message_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Supprimer le feedback du user courant sur un message."""
    feedback = session.exec(
        select(MessageFeedback)
        .where(MessageFeedback.message_id == message_id, MessageFeedback.user_id == current_user.id)
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
    session: Session = Depends(get_session)
):
    """Récupérer l'historique des feedbacks de l'utilisateur courant, paginé."""
    query = select(
        MessageFeedback,
        Space.name.label("space_name"),
        Conversation.title.label("conversation_title")
    ).outerjoin(
        Message, MessageFeedback.message_id == Message.id
    ).outerjoin(
        Conversation, Message.conversation_id == Conversation.id
    ).join(
        Space, MessageFeedback.space_id == Space.id
    ).where(
        MessageFeedback.user_id == current_user.id
    )
    
    if filter_type == "positive":
        query = query.where(MessageFeedback.is_positive == True)
    elif filter_type == "negative":
        query = query.where(MessageFeedback.is_positive == False)
        
    query = query.order_by(MessageFeedback.created_at.desc())
    
    # Total count
    total_query = select(func.count(MessageFeedback.id)).where(MessageFeedback.user_id == current_user.id)
    if filter_type == "positive":
        total_query = total_query.where(MessageFeedback.is_positive == True)
    elif filter_type == "negative":
        total_query = total_query.where(MessageFeedback.is_positive == False)
    total = session.exec(total_query).first() or 0
    
    # Paged
    offset = (page - 1) * limit
    paginated_query = query.offset(offset).limit(limit)
    results = session.exec(paginated_query).all()
    
    items = []
    for feedback, space_name, conversation_title in results:
        items.append({
            "id": feedback.id,
            "message_id": feedback.message_id,
            "space_id": feedback.space_id,
            "space_name": space_name,
            "conversation_title": conversation_title,
            "is_positive": feedback.is_positive,
            "comment": feedback.comment,
            "query_text": feedback.query_text,
            "response_text": feedback.response_text,
            "chunk_ids": feedback.chunk_ids,
            "auto_faq_generated": feedback.auto_faq_generated,
            "auto_faq_content": feedback.auto_faq_content,
            "created_at": feedback.created_at.isoformat() if feedback.created_at else None,
            "updated_at": feedback.updated_at.isoformat() if feedback.updated_at else None
        })
        
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "items": items
    }

