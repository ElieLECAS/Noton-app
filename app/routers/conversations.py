from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select, func
from app.models.user import UserRead
from app.models.conversation import Conversation, ConversationCreate, ConversationRead, ConversationUpdate
from app.models.message import Message, MessageCreate, MessageRead
from app.models.project import Project
from app.routers.auth import get_current_user
from app.database import get_session
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.post("", response_model=ConversationRead)
async def create_conversation(
    conversation: ConversationCreate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Créer une nouvelle conversation"""
    # Vérifier que le projet appartient à l'utilisateur si project_id est fourni
    if conversation.project_id:
        project = session.get(Project, conversation.project_id)
        if not project or project.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Projet non trouvé")
    
    db_conversation = Conversation(
        title=conversation.title or "Nouvelle conversation",
        user_id=current_user.id,
        project_id=conversation.project_id
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
    project_id: int = None,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Lister les conversations de l'utilisateur (optionnellement filtrées par projet)"""
    query = select(
        Conversation,
        func.count(Message.id).label("message_count")
    ).outerjoin(Message).where(
        Conversation.user_id == current_user.id
    )
    
    if project_id is not None:
        query = query.where(Conversation.project_id == project_id)
    else:
        # Si project_id n'est pas fourni, on veut les conversations sans projet (page d'accueil)
        query = query.where(Conversation.project_id.is_(None))
    
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

