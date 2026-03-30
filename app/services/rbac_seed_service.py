from typing import Dict, List
from sqlmodel import Session, select
from app.models.role import Role
from app.models.permission import Permission
from app.models.role_permission import RolePermission
import logging

logger = logging.getLogger(__name__)

# Définition du catalogue de permissions
PERMISSIONS_CATALOG = [
    {
        "code": "config.manage_users",
        "name": "Gérer les utilisateurs",
        "description": "Créer, modifier, supprimer des utilisateurs",
        "category": "config"
    },
    {
        "code": "config.manage_roles",
        "name": "Gérer les rôles",
        "description": "Créer, modifier, supprimer des rôles et assigner des permissions",
        "category": "config"
    },
    {
        "code": "library.read",
        "name": "Consulter la bibliothèque",
        "description": "Voir les documents et dossiers de la bibliothèque",
        "category": "library"
    },
    {
        "code": "library.write",
        "name": "Modifier la bibliothèque",
        "description": "Ajouter, modifier, supprimer des documents et dossiers",
        "category": "library"
    },
    {
        "code": "space.create",
        "name": "Créer un espace",
        "description": "Créer de nouveaux espaces de travail",
        "category": "space"
    },
    {
        "code": "space.update",
        "name": "Modifier un espace",
        "description": "Modifier les espaces existants",
        "category": "space"
    },
    {
        "code": "space.delete",
        "name": "Supprimer un espace",
        "description": "Supprimer des espaces de travail",
        "category": "space"
    },
    {
        "code": "space.read",
        "name": "Consulter les espaces",
        "description": "Voir les espaces et leurs contenus",
        "category": "space"
    },
]

# Définition des rôles et leurs permissions
ROLES_CATALOG = {
    "admin": {
        "description": "Administrateur avec tous les droits",
        "permissions": [
            "config.manage_users",
            "config.manage_roles",
            "library.read",
            "library.write",
            "space.create",
            "space.update",
            "space.delete",
            "space.read",
        ]
    },
    "responsable": {
        "description": "Responsable avec droits d'édition",
        "permissions": [
            "library.read",
            "library.write",
            "space.create",
            "space.update",
            "space.delete",
            "space.read",
        ]
    },
    "lecteur": {
        "description": "Lecteur avec accès en lecture seule",
        "permissions": [
            "library.read",
            "space.read",
        ]
    },
}


def seed_permissions(session: Session) -> Dict[str, Permission]:
    """Crée ou met à jour toutes les permissions du catalogue."""
    permissions_map = {}
    
    for perm_data in PERMISSIONS_CATALOG:
        statement = select(Permission).where(Permission.code == perm_data["code"])
        existing_perm = session.exec(statement).first()
        
        if existing_perm:
            permissions_map[perm_data["code"]] = existing_perm
            logger.debug(f"Permission existante: {perm_data['code']}")
        else:
            new_perm = Permission(
                code=perm_data["code"],
                name=perm_data["name"],
                description=perm_data["description"],
                category=perm_data["category"],
                is_system=True
            )
            session.add(new_perm)
            session.commit()
            session.refresh(new_perm)
            permissions_map[perm_data["code"]] = new_perm
            logger.info(f"Permission créée: {perm_data['code']}")
    
    return permissions_map


def seed_roles(session: Session, permissions_map: Dict[str, Permission]) -> Dict[str, Role]:
    """Crée ou met à jour tous les rôles du catalogue."""
    roles_map = {}
    legacy_role_mapping = {
        "member": "responsable",
        "viewer": "lecteur",
    }
    
    for role_name, role_data in ROLES_CATALOG.items():
        statement = select(Role).where(Role.name == role_name)
        existing_role = session.exec(statement).first()
        legacy_name = next(
            (old_name for old_name, new_name in legacy_role_mapping.items() if new_name == role_name),
            None
        )
        legacy_role = (
            session.exec(select(Role).where(Role.name == legacy_name)).first()
            if legacy_name
            else None
        )
        
        if existing_role:
            role = existing_role
            role.description = role_data["description"]
            session.add(role)
            session.commit()
            logger.debug(f"Rôle existant: {role_name}")
        elif legacy_role:
            legacy_role.name = role_name
            legacy_role.description = role_data["description"]
            legacy_role.is_system = True
            session.add(legacy_role)
            session.commit()
            session.refresh(legacy_role)
            role = legacy_role
            logger.info(f"Rôle renommé: {legacy_name} -> {role_name}")
        else:
            role = Role(
                name=role_name,
                description=role_data["description"],
                is_system=True
            )
            session.add(role)
            session.commit()
            session.refresh(role)
            logger.info(f"Rôle créé: {role_name}")
        
        roles_map[role_name] = role
        
        # Synchroniser les permissions du rôle
        existing_perms = session.exec(
            select(RolePermission).where(RolePermission.role_id == role.id)
        ).all()
        existing_perm_ids = {rp.permission_id for rp in existing_perms}
        
        target_perm_ids = {
            permissions_map[perm_code].id
            for perm_code in role_data["permissions"]
            if perm_code in permissions_map
        }
        
        # Ajouter les permissions manquantes
        for perm_code in role_data["permissions"]:
            if perm_code in permissions_map:
                perm_id = permissions_map[perm_code].id
                if perm_id not in existing_perm_ids:
                    role_perm = RolePermission(
                        role_id=role.id,
                        permission_id=perm_id
                    )
                    session.add(role_perm)
                    logger.debug(f"Permission {perm_code} ajoutée au rôle {role_name}")
        
        # Supprimer les permissions en trop
        for rp in existing_perms:
            if rp.permission_id not in target_perm_ids:
                session.delete(rp)
                logger.debug(f"Permission retirée du rôle {role_name}")
        
        session.commit()
    
    return roles_map


def seed_rbac_system(session: Session) -> None:
    """Initialise le système RBAC complet (permissions + rôles)."""
    logger.info("Initialisation du système RBAC...")
    
    permissions_map = seed_permissions(session)
    roles_map = seed_roles(session, permissions_map)
    
    logger.info(f"Système RBAC initialisé: {len(permissions_map)} permissions, {len(roles_map)} rôles")
