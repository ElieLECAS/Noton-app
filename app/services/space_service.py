from typing import Optional, List
from sqlmodel import Session, select
from app.models.space import Space, SpaceCreate, SpaceUpdate
from app.models.document_space import DocumentSpace
from app.models.conversation import Conversation
from app.models.message import Message
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def create_space(session: Session, space_create: SpaceCreate, user_id: int) -> Space:
    """Crée un nouvel espace."""
    space = Space(
        name=space_create.name,
        description=space_create.description,
        color=space_create.color,
        icon=space_create.icon,
        is_shared=space_create.is_shared,
        user_id=user_id
    )
    session.add(space)
    session.commit()
    session.refresh(space)
    logger.info(f"Espace créé: {space.name} (ID: {space.id})")
    return space


def get_spaces_by_user(session: Session, user_id: int) -> List[Space]:
    """Récupère tous les espaces accessibles (partagés ou personnels de l'utilisateur)."""
    statement = select(Space).where(
        (Space.is_shared == True) | (Space.user_id == user_id)
    ).order_by(Space.name)
    return list(session.exec(statement).all())


def get_space_by_id(session: Session, space_id: int, user_id: int) -> Optional[Space]:
    """Récupère un espace par son ID si il est partagé ou appartient à l'utilisateur."""
    space = session.get(Space, space_id)
    if not space:
        return None
    if space.is_shared or space.user_id == user_id:
        return space
    return None


def update_space(
    session: Session,
    space_id: int,
    space_update: SpaceUpdate,
    user_id: int
) -> Optional[Space]:
    """Met à jour un espace."""
    space = get_space_by_id(session, space_id, user_id)
    if not space:
        return None
    
    update_data = space_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(space, key, value)
    
    space.updated_at = datetime.utcnow()
    session.add(space)
    session.commit()
    session.refresh(space)
    
    logger.info(f"Espace mis à jour: {space.name} (ID: {space_id})")
    return space


def delete_space(session: Session, space_id: int, user_id: int) -> bool:
    """
    Supprime un espace et ses associations.
    Supprime les associations DocumentSpace et les conversations de cet espace.
    """
    space = get_space_by_id(session, space_id, user_id)
    if not space:
        return False
    
    doc_spaces = session.exec(
        select(DocumentSpace).where(DocumentSpace.space_id == space_id)
    ).all()
    for doc_space in doc_spaces:
        session.delete(doc_space)

    conversations = session.exec(
        select(Conversation).where(Conversation.space_id == space_id)
    ).all()
    conversation_ids = [conversation.id for conversation in conversations if conversation.id is not None]

    # Supprimer explicitement les messages avant les conversations pour éviter
    # que l'ORM tente un UPDATE message.conversation_id = NULL (interdit par NOT NULL).
    if conversation_ids:
        messages = session.exec(
            select(Message).where(Message.conversation_id.in_(conversation_ids))
        ).all()
        for message in messages:
            session.delete(message)

    for conversation in conversations:
        session.delete(conversation)
    
    # Supprimer explicitement les feedbacks associés à cet espace
    from app.models.message_feedback import MessageFeedback
    feedbacks = session.exec(
        select(MessageFeedback).where(MessageFeedback.space_id == space_id)
    ).all()
    for feedback in feedbacks:
        session.delete(feedback)

    session.delete(space)
    session.commit()
    
    logger.info(f"Espace supprimé: {space.name} (ID: {space_id})")
    return True
