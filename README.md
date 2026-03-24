# Noton - Application de prise de notes avec chatbot IA

Application web moderne de prise de notes développée avec FastAPI, PostgreSQL et Mistral.

## Fonctionnalités

- ✅ Authentification utilisateur (JWT)
- ✅ Gestion de projets
- ✅ Gestion de notes écrites
- ✅ Chatbot intégré (modèle `MODEL_FAST`)
- ✅ Configuration simple via `.env`

## Technologies

- **Backend**: FastAPI (Python 3.11)
- **Base de données**: PostgreSQL 15
- **ORM**: SQLModel
- **IA**: Mistral
- **Frontend**: HTML/CSS/JS (TailwindCSS)
- **Déploiement**: Docker Compose

## Installation et démarrage

### Prérequis

- Docker et Docker Compose installés
- Une clé API Mistral valide

### Démarrage

1. Cloner le projet
2. Créer un fichier `.env` à partir de `.env.example` (optionnel, les valeurs par défaut fonctionnent)
3. Lancer les services :

```bash
docker-compose up -d
```

4. Accéder à l'application : http://localhost:8000

### Créer un compte

L'application démarre sans utilisateur. Créez un compte via la page d'inscription.

## Structure du projet

```
noton-app/
├── app/                    # Application FastAPI
│   ├── main.py            # Point d'entrée
│   ├── config.py          # Configuration
│   ├── database.py        # Configuration DB
│   ├── models/            # Modèles SQLModel
│   ├── routers/           # Routes API
│   ├── services/          # Logique métier
│   └── templates/         # Templates Jinja2
├── alembic/               # Migrations DB
├── docker-compose.yaml    # Configuration Docker
└── README.md
```

## API Endpoints

### Authentification
- `POST /api/auth/register` - Inscription
- `POST /api/auth/login` - Connexion
- `GET /api/auth/me` - Utilisateur courant

### Projets
- `GET /api/projects` - Liste des projets
- `POST /api/projects` - Créer un projet
- `GET /api/projects/{id}` - Détail projet
- `PUT /api/projects/{id}` - Modifier projet
- `DELETE /api/projects/{id}` - Supprimer projet

### Notes
- `GET /api/projects/{project_id}/notes` - Liste notes
- `POST /api/projects/{project_id}/notes` - Créer une note
- `GET /api/notes/{id}` - Détail note
- `PUT /api/notes/{id}` - Modifier note
- `DELETE /api/notes/{id}` - Supprimer note

### Chatbot
- `GET /api/providers/models` - Modèle configuré
- `POST /api/chat` - Envoyer un message
- `POST /api/chat/stream` - Stream de réponse

## Développement

### Migrations

```bash
# Créer une migration
docker-compose exec web alembic revision --autogenerate -m "Description"

# Appliquer les migrations
docker-compose exec web alembic upgrade head
```

### Logs

```bash
# Voir les logs
docker-compose logs -f web
```

## Notes

- Le modèle chat est configuré via `MODEL_FAST` dans le `.env`
- L'authentification utilise JWT stocké dans localStorage côté client
- Les notes audio ne sont pas encore implémentées (prévu pour plus tard)

