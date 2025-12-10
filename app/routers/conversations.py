from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select, func
from app.models.user import UserRead
from app.models.conversation import Conversation, ConversationCreate, ConversationRead, ConversationUpdate
from app.models.message import Message, MessageCreate, MessageRead
from app.models.project import Project
from app.routers.auth import get_current_user
from app.database import get_session
from app.services.ollama_service import chat as ollama_chat
from app.services.openai_service import chat as openai_chat
from app.config import settings
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
        # Utiliser le modèle rapide (OpenAI si disponible, sinon Ollama)
        if settings.OPENAI_API_KEY and settings.OPENAI_MODEL:
            model = settings.OPENAI_MODEL[0] if isinstance(settings.OPENAI_MODEL, list) else settings.OPENAI_MODEL
            response = await openai_chat(prompt, model, [{"role": "user", "content": prompt}])
            if "choices" in response and len(response["choices"]) > 0:
                generated_title = response["choices"][0]["message"].get("content", "").strip()
            else:
                generated_title = None
        else:
            # Utiliser Ollama avec un modèle léger
            response = await ollama_chat(prompt, "llama3.2:1b", [{"role": "user", "content": prompt}])
            if "message" in response and "content" in response["message"]:
                generated_title = response["message"]["content"].strip()
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

