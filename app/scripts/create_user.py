#!/usr/bin/env python3
"""
Script CLI pour créer un nouvel utilisateur
Usage:
    python -m app.scripts.create_user --username monuser --email mon@email.com --password monpass
    python -m app.scripts.create_user  # Mode interactif
"""

import argparse
import getpass
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlmodel import Session, select
from app.database import engine
from app.services.auth_service import create_user
from app.models.user import UserCreate
from app.models.role import Role
from app.models.user_role import UserRole


def _assign_role(session: Session, user_id: int, role_name: str) -> bool:
    """Assigne un rôle à un utilisateur (idempotent)."""
    role = session.exec(select(Role).where(Role.name == role_name)).first()
    if not role:
        return False

    existing = session.exec(
        select(UserRole).where(
            UserRole.user_id == user_id,
            UserRole.role_id == role.id,
        )
    ).first()
    if existing:
        return True

    session.add(UserRole(user_id=user_id, role_id=role.id))
    session.commit()
    return True


def create_user_cli(
    username: str = None,
    email: str = None,
    password: str = None,
    admin: bool = None,
):
    """Créer un utilisateur via CLI"""
    
    # Mode interactif si les arguments ne sont pas fournis
    if not username:
        username = input("Nom d'utilisateur: ").strip()
    if not email:
        email = input("Email: ").strip()
    if not password:
        password = getpass.getpass("Mot de passe: ")
        password_confirm = getpass.getpass("Confirmer le mot de passe: ")
        if password != password_confirm:
            print("❌ Les mots de passe ne correspondent pas!", file=sys.stderr)
            sys.exit(1)
    if admin is None:
        admin_input = input("Administrateur ? (Y/N): ").strip().lower()
        admin = admin_input in ("y", "yes", "o", "oui")
    
    # Validation basique
    if not username:
        print("❌ Le nom d'utilisateur est requis!", file=sys.stderr)
        sys.exit(1)
    if not email:
        print("❌ L'email est requis!", file=sys.stderr)
        sys.exit(1)
    if not password:
        print("❌ Le mot de passe est requis!", file=sys.stderr)
        sys.exit(1)
    if len(password) < 6:
        print("❌ Le mot de passe doit contenir au moins 6 caractères!", file=sys.stderr)
        sys.exit(1)
    
    # Créer l'utilisateur
    try:
        with Session(engine) as session:
            user_create = UserCreate(
                username=username,
                email=email,
                password=password
            )
            user = create_user(session, user_create)

            target_role = "admin" if admin else "lecteur"
            assigned = _assign_role(session, user.id, target_role)
            if not assigned:
                print(
                    f"⚠️ Utilisateur créé mais rôle '{target_role}' introuvable.",
                    file=sys.stderr,
                )

            print(f"✅ Utilisateur créé avec succès!")
            print(f"   ID: {user.id}")
            print(f"   Username: {user.username}")
            print(f"   Email: {user.email}")
            print(f"   Rôle demandé: {target_role}")
            print(f"   Créé le: {user.created_at}")
    except ValueError as e:
        print(f"❌ Erreur: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Créer un nouvel utilisateur",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Mode interactif
  python -m app.scripts.create_user
  
  # Mode avec arguments
  python -m app.scripts.create_user --username monuser --email mon@email.com --password monpass
  
  # Création d'un administrateur
  python -m app.scripts.create_user --username admin --email admin@email.com --admin
  
  # Avec mot de passe demandé de manière sécurisée
  python -m app.scripts.create_user --username monuser --email mon@email.com
        """
    )
    
    parser.add_argument(
        "--username",
        type=str,
        help="Nom d'utilisateur"
    )
    parser.add_argument(
        "--email",
        type=str,
        help="Adresse email"
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Mot de passe (non recommandé en ligne de commande pour des raisons de sécurité)"
    )
    parser.add_argument(
        "--admin",
        action="store_true",
        help="Créer l'utilisateur avec le rôle admin"
    )
    
    args = parser.parse_args()
    
    create_user_cli(
        username=args.username,
        email=args.email,
        password=args.password,
        admin=args.admin if args.admin else None
    )


if __name__ == "__main__":
    main()

