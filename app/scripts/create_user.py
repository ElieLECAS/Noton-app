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

from sqlmodel import Session
from app.database import engine
from app.services.auth_service import create_user
from app.models.user import UserCreate


def create_user_cli(username: str = None, email: str = None, password: str = None):
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
            print(f"✅ Utilisateur créé avec succès!")
            print(f"   ID: {user.id}")
            print(f"   Username: {user.username}")
            print(f"   Email: {user.email}")
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
    
    args = parser.parse_args()
    
    create_user_cli(
        username=args.username,
        email=args.email,
        password=args.password
    )


if __name__ == "__main__":
    main()

