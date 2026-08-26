from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select, func
from app.database import get_session
from app.models.user import User, UserRead, UserCreate
from app.models.role import Role, RoleCreate, RoleRead, RoleUpdate
from app.models.permission import Permission, PermissionCreate, PermissionRead
from app.models.user_role import UserRole, UserRoleCreate, UserRoleRead
from app.models.role_permission import RolePermission, RolePermissionCreate, RolePermissionRead
from app.models.document_category import (
    DocumentCategory,
    DocumentCategoryCreate,
    DocumentCategoryRead,
    DocumentCategoryUpdate,
)
from app.models.gamme_commerciale import (
    GammeCommercialeRead,
    GammeCommercialeUpdate,
)
from app.routers.auth import get_current_user, require_permission, require_role
from app.services.auth_service import get_password_hash
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


class UserWithRoles(BaseModel):
    """Utilisateur avec ses rôles."""
    id: int
    username: str
    email: str
    created_at: str
    roles: List[str]


class RoleWithPermissions(BaseModel):
    """Rôle avec ses permissions."""
    id: int
    name: str
    description: str | None
    is_system: bool
    permissions: List[str]


class AssignRoleRequest(BaseModel):
    """Assigner un rôle à un utilisateur."""
    user_id: int
    role_id: int


class AssignPermissionRequest(BaseModel):
    """Assigner une permission à un rôle."""
    role_id: int
    permission_id: int


# ==================== CONFIG ====================

@router.get("/config")
async def get_runtime_config(
    current_user: UserRead = Depends(require_role("admin")),
):
    """Configuration effective (flags résolus) + garde-fous de cohérence.

    Aucune valeur sensible (clés, secrets, URLs de connexion) n'est exposée : uniquement
    les flags et seuils qui pilotent le pipeline RAG. Sert à vérifier en un coup d'œil ce
    qui tourne réellement (les défauts config.py sont surchargés par .env)."""
    from app.config import settings

    return {
        "feature_summary": settings.feature_summary(),
        "coherence_warnings": settings.coherence_warnings(),
        "flags": {
            "MULTIMODAL_ENABLED": settings.MULTIMODAL_ENABLED,
            "COLPALI_ENABLED": settings.COLPALI_ENABLED,
            "COLPALI_GATING_ENABLED": settings.COLPALI_GATING_ENABLED,
            "COLPALI_GATING_INTENTS": settings.colpali_gating_intents,
            "RERANKER_ENABLED": settings.RERANKER_ENABLED,
            "VISION_RERANK_ENABLED": settings.VISION_RERANK_ENABLED,
            "KAG_ENABLED": settings.KAG_ENABLED,
            "CAG_ENABLED": settings.CAG_ENABLED,
            "QUERY_UNDERSTANDING_ENABLED": settings.QUERY_UNDERSTANDING_ENABLED,
            "CONVERSATION_ANCHOR_ENABLED": settings.CONVERSATION_ANCHOR_ENABLED,
            "FICHE_TECHNIQUE_ENABLED": settings.FICHE_TECHNIQUE_ENABLED,
            "GUIDED_FLOW_ENABLED": True,  # Arbre SAV toujours actif (refonte 2026-07-30, plus de flag)
        },
        "retrieval_tuning": {
            "RAG_TOP_K": settings.RAG_TOP_K,
            "RERANK_POOL": settings.RERANK_POOL,
            "RRF_K": settings.RRF_K,
            "RETRIEVAL_CATEGORY_BOOST": settings.RETRIEVAL_CATEGORY_BOOST,
            "RETRIEVAL_CATEGORY_BOOST_MAX": settings.RETRIEVAL_CATEGORY_BOOST_MAX,
            "RETRIEVAL_AXIS_BOOST_WEIGHTS": settings.RETRIEVAL_AXIS_BOOST_WEIGHTS,
            "CONVERSATION_ANCHOR_BOOST": settings.CONVERSATION_ANCHOR_BOOST,
            "COLPALI_RELATIVE_MARGIN": settings.COLPALI_RELATIVE_MARGIN,
            "CAG_TOKEN_BUDGET": settings.CAG_TOKEN_BUDGET,
            "CAG_MAX_DOCUMENTS": settings.CAG_MAX_DOCUMENTS,
        },
    }


# ==================== USERS ====================

@router.get("/users", response_model=List[UserWithRoles])
async def list_users(
    current_user: UserRead = Depends(require_permission("config.manage_users")),
    session: Session = Depends(get_session)
):
    """Liste tous les utilisateurs avec leurs rôles."""
    users = session.exec(select(User)).all()
    result = []
    
    for user in users:
        user_roles = session.exec(
            select(UserRole).where(UserRole.user_id == user.id)
        ).all()
        role_ids = [ur.role_id for ur in user_roles]
        roles = session.exec(
            select(Role).where(Role.id.in_(role_ids))
        ).all() if role_ids else []
        
        result.append(UserWithRoles(
            id=user.id,
            username=user.username,
            email=user.email,
            created_at=user.created_at.isoformat(),
            roles=[role.name for role in roles]
        ))
    
    return result


@router.post("/users", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user_admin(
    user_create: UserCreate,
    current_user: UserRead = Depends(require_permission("config.manage_users")),
    session: Session = Depends(get_session)
):
    """Créer un nouvel utilisateur (admin)."""
    from app.services.auth_service import create_user
    try:
        user = create_user(session, user_create)
        return UserRead.model_validate(user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    current_user: UserRead = Depends(require_permission("config.manage_users")),
    session: Session = Depends(get_session)
):
    """Supprimer un utilisateur."""
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )
    
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de supprimer son propre compte"
        )
    
    # Supprimer les rôles associés
    user_roles = session.exec(select(UserRole).where(UserRole.user_id == user_id)).all()
    for ur in user_roles:
        session.delete(ur)
    
    session.delete(user)
    session.commit()


@router.post("/users/assign-role", response_model=UserRoleRead)
async def assign_role_to_user(
    request: AssignRoleRequest,
    current_user: UserRead = Depends(require_permission("config.manage_users")),
    session: Session = Depends(get_session)
):
    """Assigner un rôle unique à un utilisateur (remplace les rôles existants)."""
    user = session.get(User, request.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )
    
    role = session.get(Role, request.role_id)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rôle non trouvé"
        )
    
    existing_roles = session.exec(
        select(UserRole).where(UserRole.user_id == request.user_id)
    ).all()

    for existing_role in existing_roles:
        if existing_role.role_id == request.role_id:
            return UserRoleRead.model_validate(existing_role)
        session.delete(existing_role)

    user_role = UserRole(
        user_id=request.user_id,
        role_id=request.role_id,
        assigned_by=current_user.id
    )
    session.add(user_role)
    session.commit()
    session.refresh(user_role)
    
    return UserRoleRead.model_validate(user_role)


@router.delete("/users/{user_id}/roles/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_role_from_user(
    user_id: int,
    role_id: int,
    current_user: UserRead = Depends(require_permission("config.manage_users")),
    session: Session = Depends(get_session)
):
    """Retirer un rôle d'un utilisateur."""
    user_role = session.exec(
        select(UserRole).where(
            UserRole.user_id == user_id,
            UserRole.role_id == role_id
        )
    ).first()
    
    if not user_role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Association utilisateur-rôle non trouvée"
        )
    
    session.delete(user_role)
    session.commit()


# ==================== ROLES ====================

@router.get("/roles", response_model=List[RoleWithPermissions])
async def list_roles(
    current_user: UserRead = Depends(require_permission("config.manage_roles")),
    session: Session = Depends(get_session)
):
    """Liste tous les rôles avec leurs permissions."""
    roles = session.exec(select(Role)).all()
    result = []
    
    for role in roles:
        role_perms = session.exec(
            select(RolePermission).where(RolePermission.role_id == role.id)
        ).all()
        perm_ids = [rp.permission_id for rp in role_perms]
        permissions = session.exec(
            select(Permission).where(Permission.id.in_(perm_ids))
        ).all() if perm_ids else []
        
        result.append(RoleWithPermissions(
            id=role.id,
            name=role.name,
            description=role.description,
            is_system=role.is_system,
            permissions=[perm.code for perm in permissions]
        ))
    
    return result


@router.post("/roles", response_model=RoleRead, status_code=status.HTTP_201_CREATED)
async def create_role(
    role_create: RoleCreate,
    current_user: UserRead = Depends(require_permission("config.manage_roles")),
    session: Session = Depends(get_session)
):
    """Créer un nouveau rôle."""
    existing = session.exec(select(Role).where(Role.name == role_create.name)).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Un rôle avec ce nom existe déjà"
        )
    
    role = Role(
        name=role_create.name,
        description=role_create.description,
        is_system=False
    )
    session.add(role)
    session.commit()
    session.refresh(role)
    
    return RoleRead.model_validate(role)


@router.put("/roles/{role_id}", response_model=RoleRead)
async def update_role(
    role_id: int,
    role_update: RoleUpdate,
    current_user: UserRead = Depends(require_permission("config.manage_roles")),
    session: Session = Depends(get_session)
):
    """Mettre à jour un rôle."""
    role = session.get(Role, role_id)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rôle non trouvé"
        )
    
    if role.is_system:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de modifier un rôle système"
        )
    
    update_data = role_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(role, key, value)
    
    session.add(role)
    session.commit()
    session.refresh(role)
    
    return RoleRead.model_validate(role)


@router.delete("/roles/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_role(
    role_id: int,
    current_user: UserRead = Depends(require_permission("config.manage_roles")),
    session: Session = Depends(get_session)
):
    """Supprimer un rôle."""
    role = session.get(Role, role_id)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rôle non trouvé"
        )
    
    if role.is_system:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de supprimer un rôle système"
        )
    
    # Supprimer les associations utilisateurs
    user_roles = session.exec(select(UserRole).where(UserRole.role_id == role_id)).all()
    for ur in user_roles:
        session.delete(ur)
    
    # Supprimer les permissions associées
    role_perms = session.exec(select(RolePermission).where(RolePermission.role_id == role_id)).all()
    for rp in role_perms:
        session.delete(rp)
    
    session.delete(role)
    session.commit()


@router.post("/roles/assign-permission", response_model=RolePermissionRead)
async def assign_permission_to_role(
    request: AssignPermissionRequest,
    current_user: UserRead = Depends(require_permission("config.manage_roles")),
    session: Session = Depends(get_session)
):
    """Assigner une permission à un rôle."""
    role = session.get(Role, request.role_id)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rôle non trouvé"
        )
    
    permission = session.get(Permission, request.permission_id)
    if not permission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Permission non trouvée"
        )
    
    # Vérifier si l'association existe déjà
    existing = session.exec(
        select(RolePermission).where(
            RolePermission.role_id == request.role_id,
            RolePermission.permission_id == request.permission_id
        )
    ).first()
    
    if existing:
        return RolePermissionRead.model_validate(existing)
    
    role_perm = RolePermission(
        role_id=request.role_id,
        permission_id=request.permission_id
    )
    session.add(role_perm)
    session.commit()
    session.refresh(role_perm)
    
    return RolePermissionRead.model_validate(role_perm)


@router.delete("/roles/{role_id}/permissions/{permission_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_permission_from_role(
    role_id: int,
    permission_id: int,
    current_user: UserRead = Depends(require_permission("config.manage_roles")),
    session: Session = Depends(get_session)
):
    """Retirer une permission d'un rôle."""
    role_perm = session.exec(
        select(RolePermission).where(
            RolePermission.role_id == role_id,
            RolePermission.permission_id == permission_id
        )
    ).first()
    
    if not role_perm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Association rôle-permission non trouvée"
        )
    
    session.delete(role_perm)
    session.commit()


# ==================== PERMISSIONS ====================

@router.get("/permissions", response_model=List[PermissionRead])
async def list_permissions(
    current_user: UserRead = Depends(require_permission("config.manage_roles")),
    session: Session = Depends(get_session)
):
    """Liste toutes les permissions."""
    permissions = session.exec(select(Permission)).all()
    return [PermissionRead.model_validate(p) for p in permissions]


@router.post("/permissions", response_model=PermissionRead, status_code=status.HTTP_201_CREATED)
async def create_permission(
    permission_create: PermissionCreate,
    current_user: UserRead = Depends(require_permission("config.manage_roles")),
    session: Session = Depends(get_session)
):
    """Créer une nouvelle permission."""
    existing = session.exec(select(Permission).where(Permission.code == permission_create.code)).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Une permission avec ce code existe déjà"
        )
    
    permission = Permission(
        code=permission_create.code,
        name=permission_create.name,
        description=permission_create.description,
        category=permission_create.category,
        is_system=False
    )
    session.add(permission)
    session.commit()
    session.refresh(permission)
    
    return PermissionRead.model_validate(permission)


# ==================== QUEUES / OPS (admin rôle) ====================

@router.get("/queues/health")
async def admin_queues_health(
    current_user: UserRead = Depends(require_role("admin")),
):
    """Santé des files Celery et workers (best-effort)."""
    from app.services.celery_queue_health import get_queue_health_payload

    return get_queue_health_payload()


@router.get("/queues/workers-documents")
async def admin_workers_documents_view(
    current_user: UserRead = Depends(require_role("admin")),
):
    """Vue workers: documents en cours/en attente (worker + worker-kag)."""
    from app.services.celery_queue_health import get_workers_document_tasks_view

    return get_workers_document_tasks_view()


@router.get("/documents/stuck-processing")
async def admin_stuck_processing_documents(
    minutes: int = 30,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session)
):
    """Documents en traitement depuis plus de N minutes (heuristique ``updated_at``)."""
    from app.services.celery_queue_health import list_stuck_processing_documents

    return list_stuck_processing_documents(minutes)


@router.get("/feedbacks/stats")
async def get_admin_feedback_stats(
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session)
):
    """Statistiques globales et par espace pour les feedbacks."""
    from app.models.message_feedback import MessageFeedback
    from app.models.space import Space

    total = session.exec(select(func.count(MessageFeedback.id))).first() or 0
    positive = session.exec(select(func.count(MessageFeedback.id)).where(MessageFeedback.is_positive == True)).first() or 0
    negative = session.exec(select(func.count(MessageFeedback.id)).where(MessageFeedback.is_positive == False)).first() or 0
    ratio = round(positive / total, 2) if total > 0 else 1.0

    # Group by space and positivity in a simple DB-agnostic query
    space_stats = session.exec(
        select(
            Space.id,
            Space.name,
            MessageFeedback.is_positive,
            func.count(MessageFeedback.id)
        )
        .join(MessageFeedback, MessageFeedback.space_id == Space.id)
        .group_by(Space.id, Space.name, MessageFeedback.is_positive)
    ).all()

    by_space_dict = {}
    for space_id, space_name, is_positive, count in space_stats:
        if space_id not in by_space_dict:
            by_space_dict[space_id] = {
                "space_id": space_id,
                "space_name": space_name,
                "positive": 0,
                "negative": 0
            }
        if is_positive:
            by_space_dict[space_id]["positive"] = count
        else:
            by_space_dict[space_id]["negative"] = count

    # Répartition des classifications pour les retours négatifs
    class_stats = session.exec(
        select(
            MessageFeedback.category,
            func.count(MessageFeedback.id)
        )
        .where(MessageFeedback.is_positive == False)
        .group_by(MessageFeedback.category)
    ).all()

    classifications = {}
    for cat, count in class_stats:
        label = cat if cat else "Non classifié"
        classifications[label] = count

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "ratio": ratio,
        "by_space": list(by_space_dict.values()),
        "classifications": classifications
    }


@router.get("/feedbacks/timeline")
async def get_admin_feedback_timeline(
    period: str = "week",
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session)
):
    """Timeline des feedbacks (hebdomadaire ou mensuelle)."""
    from app.models.message_feedback import MessageFeedback
    from datetime import datetime, timedelta

    if period == "month":
        # 6 derniers mois
        start_date = datetime.utcnow() - timedelta(days=180)
    else:
        # 8 dernières semaines
        start_date = datetime.utcnow() - timedelta(days=56)

    feedbacks = session.exec(
        select(MessageFeedback.is_positive, MessageFeedback.created_at)
        .where(MessageFeedback.created_at >= start_date)
        .order_by(MessageFeedback.created_at.asc())
    ).all()

    timeline_data = {}

    if period == "month":
        for is_positive, created_at in feedbacks:
            label = created_at.strftime("%Y-%m")
            if label not in timeline_data:
                timeline_data[label] = {"positive": 0, "negative": 0}
            if is_positive:
                timeline_data[label]["positive"] += 1
            else:
                timeline_data[label]["negative"] += 1
    else:
        for is_positive, created_at in feedbacks:
            label = created_at.strftime("%Y-W%W")
            if label not in timeline_data:
                timeline_data[label] = {"positive": 0, "negative": 0}
            if is_positive:
                timeline_data[label]["positive"] += 1
            else:
                timeline_data[label]["negative"] += 1

    formatted_data = []
    for label in sorted(timeline_data.keys()):
        stats = timeline_data[label]
        if period == "month":
            try:
                dt = datetime.strptime(label, "%Y-%m")
                display_label = dt.strftime("%b %y")
            except:
                display_label = label
        else:
            parts = label.split("-W")
            if len(parts) == 2:
                display_label = f"Sem. {parts[1]}"
            else:
                display_label = label

        formatted_data.append({
            "label": display_label,
            "positive": stats["positive"],
            "negative": stats["negative"]
        })

    return {
        "period": period,
        "data": formatted_data
    }


@router.get("/feedbacks/recent")
async def get_admin_recent_feedbacks(
    page: int = 1,
    limit: int = 20,
    filter_type: Optional[str] = None,
    filter_category: Optional[str] = None,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session)
):
    """Liste paginée des retours utilisateurs récents."""
    from app.models.message_feedback import MessageFeedback
    from app.models.space import Space
    from app.models.conversation import Conversation
    from app.models.message import Message
    from app.models.user import User

    query = select(
        MessageFeedback,
        Space.name.label("space_name"),
        Conversation.title.label("conversation_title"),
        Conversation.id.label("conversation_id"),
        User.email.label("user_email")
    ).outerjoin(
        Message, MessageFeedback.message_id == Message.id
    ).outerjoin(
        Conversation, Message.conversation_id == Conversation.id
    ).join(
        Space, MessageFeedback.space_id == Space.id
    ).join(
        User, MessageFeedback.user_id == User.id
    )

    if filter_type == "positive":
        query = query.where(MessageFeedback.is_positive == True)
    elif filter_type == "negative":
        query = query.where(MessageFeedback.is_positive == False)

    if filter_category:
        if filter_category == "Non classifié":
            query = query.where((MessageFeedback.category == None) | (MessageFeedback.category == ""))
        else:
            query = query.where(MessageFeedback.category == filter_category)

    query = query.order_by(MessageFeedback.created_at.desc())

    # Total count
    total_query = select(func.count(MessageFeedback.id))
    if filter_type == "positive":
        total_query = total_query.where(MessageFeedback.is_positive == True)
    elif filter_type == "negative":
        total_query = total_query.where(MessageFeedback.is_positive == False)

    if filter_category:
        if filter_category == "Non classifié":
            total_query = total_query.where((MessageFeedback.category == None) | (MessageFeedback.category == ""))
        else:
            total_query = total_query.where(MessageFeedback.category == filter_category)
            
    total = session.exec(total_query).first() or 0

    # Paged
    offset = (page - 1) * limit
    paginated_query = query.offset(offset).limit(limit)
    results = session.exec(paginated_query).all()

    items = []
    for feedback, space_name, conversation_title, conversation_id, user_email in results:
        items.append({
            "id": feedback.id,
            "message_id": feedback.message_id,
            "space_id": feedback.space_id,
            "space_name": space_name,
            "conversation_title": conversation_title,
            "conversation_id": conversation_id,
            "user_email": user_email,
            "is_positive": feedback.is_positive,
            "comment": feedback.comment,
            "query_text": feedback.query_text,
            "response_text": feedback.response_text,
            "chunk_ids": feedback.chunk_ids,
            "auto_faq_generated": feedback.auto_faq_generated,
            "auto_faq_content": feedback.auto_faq_content,
            "category": feedback.category,
            "created_at": feedback.created_at.isoformat() if feedback.created_at else None,
            "updated_at": feedback.updated_at.isoformat() if feedback.updated_at else None
        })

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "items": items
    }


@router.delete("/feedbacks/{feedback_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_admin_feedback(
    feedback_id: int,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session)
):
    """Supprimer n'importe quel feedback (réservé aux admins)."""
    from app.models.message_feedback import MessageFeedback
    feedback = session.get(MessageFeedback, feedback_id)
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback non trouvé")
    
    session.delete(feedback)
    session.commit()
    return


# ==================== RAG EVALUATION ====================

class RetrieverEvalRequest(BaseModel):
    space_id: int
    dataset: List[dict]
    k: int = 15


class SingleRetrieverEvalRequest(BaseModel):
    space_id: int
    question: str
    type: str = "mono-document"
    pages_attendues: List[dict]
    acceptable_document_ids: List[int] = []
    intent: str = "documentation"
    k: int = 15


@router.post("/eval/retriever")
async def evaluate_retriever_api(
    request: RetrieverEvalRequest,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session)
):
    """Évaluer le retriever ColPali sur un dataset complet."""
    from app.services.retriever_evaluator import evaluate_retriever_dataset
    try:
        results = await evaluate_retriever_dataset(
            session=session,
            space_id=request.space_id,
            user_id=current_user.id,
            dataset=request.dataset,
            k=request.k
        )
        return results
    except Exception as e:
        logger.error("Error during retriever evaluation API: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'évaluation : {str(e)}"
        )


@router.post("/eval/retriever/single")
async def evaluate_retriever_single_api(
    request: SingleRetrieverEvalRequest,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session)
):
    """Évaluer le retriever ColPali sur une seule question pour la progression de l'UI."""
    from app.services.retriever_evaluator import build_question_eval_result, evaluate_cag_document_hit
    from app.services.space_search_service import search_relevant_passages
    from app.services.context_packer_service import build_cag_context

    try:
        search_res = await search_relevant_passages(
            session=session,
            space_id=request.space_id,
            query_text=request.question,
            user_id=current_user.id,
            k=request.k,
            document_filter="all",
            include_retrieval_stages=True,
        )
        passages = search_res.get("passages", [])
        stages = search_res.get("retrieval_stages") or {}

        result = build_question_eval_result(
            question=request.question,
            q_type=request.type,
            expected_pages=request.pages_attendues,
            passages=passages,
            colpali_passages=stages.get("colpali_only") or stages.get("colpali", passages),
            lexical_only_passages=stages.get("lexical_only"),
            pre_kag_passages=stages.get("pre_kag_rrf"),
            post_rrf_passages=stages.get("post_rrf"),
            kag_only_passages=stages.get("kag_only"),
            vision_rerank_enabled=stages.get("vision_rerank_enabled"),
            minilm_rerank_enabled=stages.get("minilm_rerank_enabled"),
            kag_enabled=stages.get("kag_enabled"),
        )

        # Étape CAG (niveau document) : le CAG packe des documents entiers ; on vérifie
        # que le bon document / la bonne page arrive dans le contexte final de génération.
        try:
            cag_ctx = build_cag_context(
                session, passages, system_prompt="",
                intent=request.intent, emit_sources_tag=False,
            )
            cag_docs = cag_ctx.get("cag_documents") or []
            hit = evaluate_cag_document_hit(
                cag_docs,
                acceptable_document_ids=request.acceptable_document_ids,
                expected_pages=request.pages_attendues,
            )
            packed = hit["packed_ids"]
            acc = set(request.acceptable_document_ids or [])
            result["cag"] = {
                "packed_document_ids": packed,
                "doc_hit_acceptable": hit["doc_hit_acceptable"],
                "doc_hit_strict": hit["doc_hit_strict"],
                "page_in_context": hit["page_in_context"],
                "doc_precision": round(len(set(packed) & acc) / len(packed), 3) if (packed and acc) else 0.0,
                "num_packed": len(packed),
            }
        except Exception as cag_exc:
            logger.warning("[eval single] CAG hook échoué: %s", cag_exc)
            result["cag"] = None

        return result
    except Exception as e:
        logger.error("Error during single retriever evaluation API: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'évaluation unitaire : {str(e)}"
        )


class SingleRAGEvalRequest(BaseModel):
    space_id: int
    question: str
    type: str = "mono-document"
    pages_attendues: List[dict]
    reponse_attendue: Optional[str] = None
    expected_response: Optional[str] = None
    k: int = 15
    judge_model: str = "mistral-small-latest"


@router.post("/eval/rag/single")
async def evaluate_rag_single_api(
    request: SingleRAGEvalRequest,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session)
):
    """Évaluer le retriever ColPali et la génération LLM sur une seule question."""
    from app.services.retriever_evaluator import (
        build_question_eval_result,
        generate_rag_response,
        run_llm_judge,
    )
    from app.services.space_search_service import search_relevant_passages

    try:
        search_res = await search_relevant_passages(
            session=session,
            space_id=request.space_id,
            query_text=request.question,
            user_id=current_user.id,
            k=request.k,
            document_filter="all",
            include_retrieval_stages=True,
        )
        passages = search_res.get("passages", [])
        stages = search_res.get("retrieval_stages") or {}

        eval_result = build_question_eval_result(
            question=request.question,
            q_type=request.type,
            expected_pages=request.pages_attendues,
            passages=passages,
            colpali_passages=stages.get("colpali_only") or stages.get("colpali", passages),
            lexical_only_passages=stages.get("lexical_only"),
            pre_kag_passages=stages.get("pre_kag_rrf"),
            post_rrf_passages=stages.get("post_rrf"),
            kag_only_passages=stages.get("kag_only"),
            vision_rerank_enabled=stages.get("vision_rerank_enabled"),
            minilm_rerank_enabled=stages.get("minilm_rerank_enabled"),
            kag_enabled=stages.get("kag_enabled"),
        )

        generated_response = await generate_rag_response(
            session=session,
            space_id=request.space_id,
            user_id=current_user.id,
            question=request.question,
            passages=passages,
        )

        expected = request.reponse_attendue or request.expected_response
        judge_eval = None
        if expected:
            judge_eval = await run_llm_judge(
                question=request.question,
                generated_response=generated_response,
                expected_response=expected,
                judge_model=request.judge_model,
            )

        eval_result["generated_response"] = generated_response
        eval_result["expected_response"] = expected
        eval_result["judge_evaluation"] = judge_eval
        return eval_result
    except Exception as e:
        logger.error("Error during single RAG evaluation API: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'évaluation RAG unitaire : {str(e)}"
        )


# ==================== DOCUMENT CATEGORIES ====================


class CategoryStatsResponse(BaseModel):
    category_id: int
    slug: str
    label: str
    chunk_links: int
    document_count: int


@router.get("/categories", response_model=List[DocumentCategoryRead])
async def list_categories(
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    """Liste toutes les catégories de contenu (actives et inactives)."""
    rows = session.exec(
        select(DocumentCategory).order_by(DocumentCategory.slug)
    ).all()
    return [DocumentCategoryRead.model_validate(row) for row in rows]


@router.post("/categories", response_model=DocumentCategoryRead)
async def create_category(
    payload: DocumentCategoryCreate,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    """Crée une nouvelle catégorie de contenu."""
    from datetime import datetime

    slug = payload.slug.strip().lower().replace(" ", "_")
    existing = session.exec(
        select(DocumentCategory).where(DocumentCategory.slug == slug)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Slug déjà utilisé : {slug}")

    row = DocumentCategory(
        slug=slug,
        label=payload.label.strip(),
        description=(payload.description or "").strip(),
        axis=(payload.axis or "task").strip().lower(),
        parent_slug=payload.parent_slug,
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return DocumentCategoryRead.model_validate(row)


@router.put("/categories/{category_id}", response_model=DocumentCategoryRead)
async def update_category(
    category_id: int,
    payload: DocumentCategoryUpdate,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    """Met à jour label, description ou statif actif d'une catégorie."""
    from datetime import datetime

    row = session.get(DocumentCategory, category_id)
    if not row:
        raise HTTPException(status_code=404, detail="Catégorie introuvable")

    if payload.label is not None:
        row.label = payload.label.strip()
    if payload.description is not None:
        row.description = payload.description.strip()
    if payload.axis is not None:
        row.axis = payload.axis.strip().lower()
    if payload.parent_slug is not None:
        row.parent_slug = payload.parent_slug
    if payload.is_active is not None:
        row.is_active = payload.is_active
    row.updated_at = datetime.utcnow()
    session.add(row)
    session.commit()
    session.refresh(row)
    return DocumentCategoryRead.model_validate(row)


@router.delete("/categories/{category_id}")
async def delete_category(
    category_id: int,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    """Supprime une catégorie si aucune relation chunk n'existe."""
    from app.models.chunk_category_relation import ChunkCategoryRelation

    row = session.get(DocumentCategory, category_id)
    if not row:
        raise HTTPException(status_code=404, detail="Catégorie introuvable")

    link_count = session.exec(
        select(func.count()).select_from(ChunkCategoryRelation).where(
            ChunkCategoryRelation.category_id == category_id
        )
    ).first() or 0
    if int(link_count) > 0:
        raise HTTPException(
            status_code=400,
            detail="Impossible de supprimer : des chunks sont liés à cette catégorie.",
        )

    session.delete(row)
    session.commit()
    return {"status": "deleted", "id": category_id}


# --- Suivi d'usage : conversations & messages (admin only) ---


@router.get("/conversations")
async def list_all_conversations(
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
    limit: int = 500,
):
    """Toutes les conversations (tous utilisateurs) + totaux. Admin uniquement."""
    from sqlalchemy import text

    totals = session.execute(
        text(
            "SELECT (SELECT COUNT(*) FROM conversation) AS c, "
            "(SELECT COUNT(*) FROM message) AS m"
        )
    ).first()

    rows = session.execute(
        text(
            """
            SELECT c.id, c.title, c.user_id, u.username, u.email,
                   c.space_id, s.name AS space_name,
                   c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count,
                   MAX(m.created_at) AS last_message_at
            FROM conversation c
            LEFT JOIN "user" u ON u.id = c.user_id
            LEFT JOIN space s ON s.id = c.space_id
            LEFT JOIN message m ON m.conversation_id = c.id
            GROUP BY c.id, u.username, u.email, s.name
            ORDER BY COALESCE(MAX(m.created_at), c.updated_at) DESC NULLS LAST
            LIMIT :limit
            """
        ),
        {"limit": limit},
    ).all()

    conversations = [
        {
            "id": r.id,
            "title": r.title,
            "user_id": r.user_id,
            "username": r.username,
            "email": r.email,
            "space_id": r.space_id,
            "space_name": r.space_name,
            "message_count": int(r.message_count or 0),
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            "last_message_at": r.last_message_at.isoformat() if r.last_message_at else None,
        }
        for r in rows
    ]

    return {
        "total_conversations": int(totals.c or 0) if totals else 0,
        "total_messages": int(totals.m or 0) if totals else 0,
        "count_shown": len(conversations),
        "conversations": conversations,
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation_detail(
    conversation_id: int,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    """Détail d'une conversation (messages) pour « entrer à l'intérieur ». Admin only."""
    from sqlalchemy import text

    conv = session.execute(
        text(
            """
            SELECT c.id, c.title, c.user_id, u.username, u.email,
                   c.space_id, s.name AS space_name, c.created_at, c.updated_at
            FROM conversation c
            LEFT JOIN "user" u ON u.id = c.user_id
            LEFT JOIN space s ON s.id = c.space_id
            WHERE c.id = :id
            """
        ),
        {"id": conversation_id},
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation introuvable")

    msgs = session.execute(
        text(
            """
            SELECT id, role, content, model, created_at
            FROM message
            WHERE conversation_id = :id
            ORDER BY created_at, id
            """
        ),
        {"id": conversation_id},
    ).all()

    return {
        "id": conv.id,
        "title": conv.title,
        "user_id": conv.user_id,
        "username": conv.username,
        "email": conv.email,
        "space_id": conv.space_id,
        "space_name": conv.space_name,
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "message_count": len(msgs),
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "model": m.model,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in msgs
        ],
    }


@router.get("/messages")
async def list_all_messages(
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
    role: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 1000,
):
    """Messages (tous utilisateurs) triés/groupables par conversation. Admin only.

    `role=user` → seulement les questions (voir le type de demandes). `q` filtre le contenu.
    """
    from sqlalchemy import text

    qpat = f"%{q.strip()}%" if q and q.strip() else None
    role_f = role if role in ("user", "assistant", "system") else None

    rows = session.execute(
        text(
            """
            SELECT m.id, m.conversation_id, m.role, m.content, m.created_at,
                   c.title AS conversation_title, c.user_id,
                   u.username, s.name AS space_name
            FROM message m
            INNER JOIN conversation c ON c.id = m.conversation_id
            LEFT JOIN "user" u ON u.id = c.user_id
            LEFT JOIN space s ON s.id = c.space_id
            LEFT JOIN (
                SELECT conversation_id, MAX(created_at) AS last_at
                FROM message GROUP BY conversation_id
            ) la ON la.conversation_id = m.conversation_id
            WHERE (:role IS NULL OR m.role = :role)
              AND (:qpat IS NULL OR m.content ILIKE :qpat)
            ORDER BY la.last_at DESC NULLS LAST, m.conversation_id DESC,
                     m.created_at ASC, m.id ASC
            LIMIT :limit
            """
        ),
        {"role": role_f, "qpat": qpat, "limit": limit},
    ).all()

    total_user = session.execute(
        text("SELECT COUNT(*) FROM message WHERE role = 'user'")
    ).first()

    messages = [
        {
            "id": r.id,
            "conversation_id": r.conversation_id,
            "role": r.role,
            "content": r.content,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "conversation_title": r.conversation_title,
            "username": r.username,
            "space_name": r.space_name,
        }
        for r in rows
    ]
    return {
        "count_shown": len(messages),
        "total_user_questions": int(total_user[0]) if total_user else 0,
        "messages": messages,
    }


@router.get("/categories/{category_id}/stats", response_model=CategoryStatsResponse)
async def category_stats(
    category_id: int,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    """Statistiques d'utilisation d'une catégorie."""
    from app.models.chunk_category_relation import ChunkCategoryRelation

    row = session.get(DocumentCategory, category_id)
    if not row:
        raise HTTPException(status_code=404, detail="Catégorie introuvable")

    chunk_links = session.exec(
        select(func.count()).select_from(ChunkCategoryRelation).where(
            ChunkCategoryRelation.category_id == category_id
        )
    ).first() or 0
    doc_count = session.exec(
        select(func.count(func.distinct(ChunkCategoryRelation.document_id))).where(
            ChunkCategoryRelation.category_id == category_id
        )
    ).first() or 0

    return CategoryStatsResponse(
        category_id=row.id,
        slug=row.slug,
        label=row.label,
        chunk_links=int(chunk_links),
        document_count=int(doc_count),
    )



# ---------------------------------------------------------------------------
# Connaissance métier — fiches des gammes commerciales Proferm
#
# Le vocabulaire des utilisateurs et celui des documents fournisseurs ne se recoupent
# pas : ces fiches portent ce pont, pour injection dans les prompts. Elles sont
# rédigées et validées par le métier — d'où l'édition en admin plutôt qu'un fichier.
# ---------------------------------------------------------------------------


@router.get("/gammes", response_model=List[GammeCommercialeRead])
async def list_gammes_admin(
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    """Liste les fiches de gammes, brouillons compris."""
    from app.services.gamme_knowledge_service import list_gammes

    return [GammeCommercialeRead.model_validate(g) for g in list_gammes(session)]


@router.post("/gammes/seed")
async def seed_gammes_admin(
    overwrite: bool = False,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    """Crée les fiches manquantes depuis l'amorçage rédigé à partir des documents.

    Sans `overwrite`, une fiche déjà présente n'est jamais réécrite : une fois relue par
    le métier, c'est la base qui fait foi.
    """
    from app.services.gamme_knowledge_service import seed_gammes

    return seed_gammes(session, overwrite=overwrite)


@router.put("/gammes/{gamme_id}", response_model=GammeCommercialeRead)
async def update_gamme_admin(
    gamme_id: int,
    payload: GammeCommercialeUpdate,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    """Met à jour une fiche (édition métier)."""
    from datetime import datetime

    from app.models.gamme_commerciale import STATUTS, GammeCommerciale

    row = session.get(GammeCommerciale, gamme_id)
    if not row:
        raise HTTPException(status_code=404, detail="Fiche introuvable")

    data = payload.model_dump(exclude_unset=True)
    if "statut" in data and data["statut"] not in STATUTS:
        raise HTTPException(
            status_code=400,
            detail=f"Statut invalide : {data['statut']} (attendu : {', '.join(STATUTS)})",
        )
    for key, value in data.items():
        setattr(row, key, value)
    row.updated_at = datetime.utcnow()
    row.updated_by = current_user.id
    session.add(row)
    session.commit()
    session.refresh(row)
    return GammeCommercialeRead.model_validate(row)


@router.get("/gammes/preview-prompt")
async def preview_gamme_prompt(
    only_valid: bool = False,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    """Le bloc exactement tel qu'il sera injecté dans les prompts.

    Rendu en entier, jamais filtré sur une gamme : pour écarter une référence Technal
    quand on cherche du PVC, le modèle doit connaître la règle Technal.
    """
    from app.services.gamme_knowledge_service import build_gamme_knowledge_block

    block = build_gamme_knowledge_block(session, only_valid=only_valid)
    return {"block": block, "chars": len(block), "tokens_estimes": len(block) // 4}
