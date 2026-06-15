from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select, func
from app.database import get_session
from app.models.user import User, UserRead, UserCreate
from app.models.role import Role, RoleCreate, RoleRead, RoleUpdate
from app.models.permission import Permission, PermissionCreate, PermissionRead
from app.models.user_role import UserRole, UserRoleCreate, UserRoleRead
from app.models.role_permission import RolePermission, RolePermissionCreate, RolePermissionRead
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
    from app.services.retriever_evaluator import (
        match_page, compute_context_precision, compute_context_recall, compute_mrr
    )
    from app.services.space_search_service import search_relevant_passages
    
    try:
        search_res = await search_relevant_passages(
            session=session,
            space_id=request.space_id,
            query_text=request.question,
            user_id=current_user.id,
            k=request.k,
            document_filter="all"
        )
        passages = search_res.get("passages", [])
        
        # Extraire les couples (titre, page_no) et les détails ordonnés
        retrieved_pages = []
        retrieved_details = []
        for idx, p in enumerate(passages):
            doc_title = p.get("document_title", "")
            page_no = p.get("page_no")
            if page_no is None:
                page_no = p.get("page_start", 1)
            try:
                page_no_int = int(page_no)
            except (ValueError, TypeError):
                page_no_int = 1
            retrieved_pages.append((doc_title, page_no_int))
            
            score = p.get("score", 0.0)
            is_hit = match_page(doc_title, page_no_int, request.pages_attendues)
            retrieved_details.append({
                "rank": idx + 1,
                "document_title": doc_title,
                "page": page_no_int,
                "score": round(score, 4),
                "is_hit": is_hit
            })
                
        # Calculer les métriques
        precision = compute_context_precision(retrieved_pages, request.pages_attendues)
        recall = compute_context_recall(retrieved_pages, request.pages_attendues)
        mrr = compute_mrr(retrieved_pages, request.pages_attendues)
        
        # Identifier Hits, Misses et Bruit
        hits_details = []
        misses_details = []
        noise_details = []
        
        for rd in retrieved_details:
            page_info = {
                "document_title": rd["document_title"],
                "page": rd["page"],
                "rank": rd["rank"],
                "score": rd["score"]
            }
            if rd["is_hit"]:
                if not any(h["document_title"] == rd["document_title"] and h["page"] == rd["page"] for h in hits_details):
                    hits_details.append(page_info)
            else:
                if not any(n["document_title"] == rd["document_title"] and n["page"] == rd["page"] for n in noise_details):
                    noise_details.append(page_info)
                    
        for exp in request.pages_attendues:
            exp_title = exp.get("document_title", "")
            for p in exp.get("pages", []):
                try:
                    target_p = int(p)
                except (ValueError, TypeError):
                    continue
                
                found = False
                for hit in hits_details:
                    if exp_title.lower() in hit["document_title"].lower() or hit["document_title"].lower() in exp_title.lower():
                        if hit["page"] == target_p:
                            found = True
                            break
                if not found:
                    misses_details.append({"document_title": exp_title, "page": target_p})
                    
        return {
            "question": request.question,
            "type": request.type,
            "expected_pages": request.pages_attendues,
            "retrieved_pages": [{"document_title": t, "page": p} for t, p in retrieved_pages],
            "retrieved_details": retrieved_details,
            "metrics": {
                "context_precision": round(precision, 4),
                "context_recall": round(recall, 4),
                "mrr": round(mrr, 4)
            },
            "analysis": {
                "hits": hits_details,
                "misses": misses_details,
                "noise": noise_details
            }
        }
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
        match_page, compute_context_precision, compute_context_recall, compute_mrr,
        generate_rag_response, run_llm_judge
    )
    from app.services.space_search_service import search_relevant_passages
    
    try:
        # 1. Recherche Retriever
        search_res = await search_relevant_passages(
            session=session,
            space_id=request.space_id,
            query_text=request.question,
            user_id=current_user.id,
            k=request.k,
            document_filter="all"
        )
        passages = search_res.get("passages", [])
        
        # Extraire les couples (titre, page_no) et les détails ordonnés
        retrieved_pages = []
        retrieved_details = []
        for idx, p in enumerate(passages):
            doc_title = p.get("document_title", "")
            page_no = p.get("page_no")
            if page_no is None:
                page_no = p.get("page_start", 1)
            try:
                page_no_int = int(page_no)
            except (ValueError, TypeError):
                page_no_int = 1
            retrieved_pages.append((doc_title, page_no_int))
            
            score = p.get("score", 0.0)
            is_hit = match_page(doc_title, page_no_int, request.pages_attendues)
            retrieved_details.append({
                "rank": idx + 1,
                "document_title": doc_title,
                "page": page_no_int,
                "score": round(score, 4),
                "is_hit": is_hit
            })
                
        # Calculer les métriques
        precision = compute_context_precision(retrieved_pages, request.pages_attendues)
        recall = compute_context_recall(retrieved_pages, request.pages_attendues)
        mrr = compute_mrr(retrieved_pages, request.pages_attendues)
        
        # Identifier Hits, Misses et Bruit
        hits_details = []
        misses_details = []
        noise_details = []
        
        for rd in retrieved_details:
            page_info = {
                "document_title": rd["document_title"],
                "page": rd["page"],
                "rank": rd["rank"],
                "score": rd["score"]
            }
            if rd["is_hit"]:
                if not any(h["document_title"] == rd["document_title"] and h["page"] == rd["page"] for h in hits_details):
                    hits_details.append(page_info)
            else:
                if not any(n["document_title"] == rd["document_title"] and n["page"] == rd["page"] for n in noise_details):
                    noise_details.append(page_info)
                    
        for exp in request.pages_attendues:
            exp_title = exp.get("document_title", "")
            for p in exp.get("pages", []):
                try:
                    target_p = int(p)
                except (ValueError, TypeError):
                    continue
                
                found = False
                for hit in hits_details:
                    if exp_title.lower() in hit["document_title"].lower() or hit["document_title"].lower() in exp_title.lower():
                        if hit["page"] == target_p:
                            found = True
                            break
                if not found:
                    misses_details.append({"document_title": exp_title, "page": target_p})
                    
        # 2. Génération RAG
        generated_response = await generate_rag_response(
            session=session,
            space_id=request.space_id,
            user_id=current_user.id,
            question=request.question,
            passages=passages
        )
        
        # 3. LLM Judge
        expected = request.reponse_attendue or request.expected_response
        judge_eval = None
        if expected:
            judge_eval = await run_llm_judge(
                question=request.question,
                generated_response=generated_response,
                expected_response=expected,
                judge_model=request.judge_model
            )
            
        return {
            "question": request.question,
            "type": request.type,
            "expected_pages": request.pages_attendues,
            "retrieved_pages": [{"document_title": t, "page": p} for t, p in retrieved_pages],
            "retrieved_details": retrieved_details,
            "metrics": {
                "context_precision": round(precision, 4),
                "context_recall": round(recall, 4),
                "mrr": round(mrr, 4)
            },
            "analysis": {
                "hits": hits_details,
                "misses": misses_details,
                "noise": noise_details
            },
            "generated_response": generated_response,
            "expected_response": expected,
            "judge_evaluation": judge_eval
        }
    except Exception as e:
        logger.error("Error during single RAG evaluation API: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'évaluation RAG unitaire : {str(e)}"
        )

