from typing import Optional, List
from sqlmodel import Session, select
from app.models.document_space import DocumentSpace, DocumentSpaceRead
from app.models.space import Space
from app.models.document import Document
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def link_document_to_space(
    session: Session,
    document_id: int,
    space_id: int,
    user_id: int
) -> Optional[DocumentSpace]:
    """
    Lie un document à un espace.
    Vérifie que le document et l'espace existent.
    """
    document = session.exec(
        select(Document).where(
            Document.id == document_id
        )
    ).first()
    
    if not document:
        logger.error(f"Document {document_id} non trouvé")
        return None
    
    space = session.exec(
        select(Space).where(
            Space.id == space_id
        )
    ).first()
    
    if not space:
        logger.error(f"Espace {space_id} non trouvé")
        return None
    
    existing = session.exec(
        select(DocumentSpace).where(
            DocumentSpace.document_id == document_id,
            DocumentSpace.space_id == space_id
        )
    ).first()
    
    if existing:
        logger.info(f"Document {document_id} déjà lié à l'espace {space_id}")
        return existing
    
    doc_space = DocumentSpace(
        document_id=document_id,
        space_id=space_id,
        user_id=user_id
    )
    session.add(doc_space)
    session.commit()
    session.refresh(doc_space)
    
    logger.info(f"Document {document_id} lié à l'espace {space_id}")
    return doc_space


def unlink_document_from_space(
    session: Session,
    document_id: int,
    space_id: int,
    user_id: int
) -> bool:
    """
    Délie un document d'un espace.
    Supprime également les entités KAG associées pour cet espace.
    """
    from app.services.kag_graph_service import delete_entities_for_document
    
    doc_space = session.exec(
        select(DocumentSpace).where(
            DocumentSpace.document_id == document_id,
            DocumentSpace.space_id == space_id
        )
    ).first()
    
    if not doc_space:
        logger.warning(f"Association document {document_id} - espace {space_id} non trouvée")
        return False
    
    delete_entities_for_document(session, document_id, space_id)
    
    session.delete(doc_space)
    session.commit()
    
    logger.info(f"Document {document_id} délié de l'espace {space_id}")
    return True


def get_spaces_for_document(
    session: Session,
    document_id: int,
    user_id: int
) -> List[Space]:
    """Récupère tous les espaces ayant accès à un document."""
    statement = select(Space).join(DocumentSpace).where(
        DocumentSpace.document_id == document_id
    ).order_by(Space.name)
    
    return list(session.exec(statement).all())


def get_documents_for_space(
    session: Session,
    space_id: int,
    user_id: int
) -> List[Document]:
    """Récupère tous les documents accessibles dans un espace."""
    statement = select(Document).join(DocumentSpace).where(
        DocumentSpace.space_id == space_id
    ).order_by(Document.created_at.desc())
    
    return list(session.exec(statement).all())


def get_document_spaces(
    session: Session,
    document_id: int,
    user_id: int
) -> List[DocumentSpace]:
    """Récupère toutes les associations espace pour un document."""
    statement = select(DocumentSpace).where(
        DocumentSpace.document_id == document_id
    )
    return list(session.exec(statement).all())
