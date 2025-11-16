# Résumé de l'Application Noton

## Vue d'ensemble

**Noton** est une application web de prise de notes inspirée de Notion, développée avec Django. Elle permet aux utilisateurs de créer et organiser des projets contenant des notes écrites et vocales, avec transcription automatique de l'audio en texte. L'application est conçue pour être déployée avec Docker Compose et utilise PostgreSQL comme base de données.

---

## Architecture technique

### Stack technologique

- **Backend** : Django 5.0.2 (Python 3.11)
- **Base de données** : PostgreSQL 15 (via Docker) ou SQLite3 (développement local)
- **Frontend** : Templates Django avec TailwindCSS 2.2.19 (CDN)
- **Icônes** : Tabler Icons (CDN)
- **Reconnaissance vocale** : Vosk (modèle français)
- **Traitement audio** : pydub, ffmpeg
- **Déploiement** : Docker & Docker Compose
- **Serveur WSGI** : Gunicorn (configuré mais non utilisé en développement)

### Structure du projet

```
noton/
├── docker-compose.yaml          # Configuration Docker Compose
├── noton/                       # Application Django principale
│   ├── Dockerfile              # Image Docker pour l'application
│   ├── manage.py               # Script de gestion Django
│   ├── requirements.txt        # Dépendances Python
│   ├── note/                   # Application Django "note"
│   │   ├── models.py          # Modèles de données
│   │   ├── views.py           # Vues (contrôleurs)
│   │   ├── forms.py           # Formulaires
│   │   ├── urls.py            # Routes de l'application
│   │   ├── admin.py           # Configuration admin Django
│   │   ├── speech_to_text.py  # Module de transcription audio
│   │   └── templates/         # Templates HTML
│   └── noton/                  # Configuration Django
│       ├── settings.py         # Paramètres Django
│       ├── urls.py            # Routes principales
│       └── wsgi.py            # Point d'entrée WSGI
└── models/                     # Modèles Vosk (reconnaissance vocale)
```

---

## Modèles de données

### 1. Project (Projet)
- **Champs** :
  - `title` : CharField(200) - Titre du projet
  - `description` : TextField - Description optionnelle
  - `user` : ForeignKey(User) - Propriétaire du projet
  - `created_at` : DateTimeField - Date de création
  - `updated_at` : DateTimeField - Date de dernière modification
- **Relations** : Un projet appartient à un utilisateur, contient plusieurs notes
- **Tri** : Par date de mise à jour décroissante

### 2. Note (Note)
- **Champs** :
  - `title` : CharField(200) - Titre de la note
  - `content` : TextField - Contenu texte (peut être vide pour notes vocales)
  - `original_audio` : FileField - Fichier audio (optionnel, upload vers 'audio_notes/')
  - `note_type` : CharField - Type de note : 'written' (écrite) ou 'voice' (vocale)
  - `project` : ForeignKey(Project) - Projet parent
  - `user` : ForeignKey(User) - Propriétaire de la note
  - `created_at` : DateTimeField - Date de création
  - `updated_at` : DateTimeField - Date de dernière modification
- **Relations** : Une note appartient à un projet et à un utilisateur
- **Tri** : Par date de mise à jour décroissante

### 3. ProjectSummary (Résumé de projet)
- **Champs** :
  - `project` : OneToOneField(Project) - Projet associé (relation unique)
  - `content` : TextField - Contenu du résumé généré
  - `generated_at` : DateTimeField - Date de génération
- **Relations** : Un résumé par projet (relation 1-1)
- **État actuel** : La génération de résumé est un placeholder, non implémentée avec un vrai LLM

---

## Fonctionnalités implémentées

### 1. Authentification
- Système d'authentification Django standard
- Vues de connexion/déconnexion (`accounts/login/`, `accounts/logout/`)
- Protection des vues avec `@login_required`
- Redirection après connexion vers la liste des projets

### 2. Gestion des projets
- **Liste des projets** (`/notes/`) : Affiche tous les projets de l'utilisateur
- **Création de projet** (`/notes/projects/create/`) : Formulaire pour créer un nouveau projet
- **Détail du projet** (`/notes/projects/<id>/`) : Affiche un projet et toutes ses notes
- **Suppression de projet** (`/notes/projects/<id>/delete/`) : Supprime un projet (cascade sur les notes)

### 3. Gestion des notes
- **Création de note** (`/notes/projects/<id>/notes/create/<type>/`) :
  - Type 'written' : Note texte classique
  - Type 'voice' : Note vocale avec upload de fichier audio
- **Détail de note** (`/notes/notes/<id>/`) : Affiche une note complète
- **Édition de note** (`/notes/notes/<id>/edit/`) : Modifie une note existante
- **Suppression de note** (`/notes/notes/<id>/delete/`) : Supprime une note

### 4. Transcription audio (Speech-to-Text)
- **Module** : `speech_to_text.py`
- **Technologie** : Vosk (modèle français)
- **Fonctionnalités** :
  - Détection automatique du modèle Vosk dans `/models/`
  - Conversion automatique des formats audio (mp3, m4a, etc.) en WAV mono 16kHz
  - Transcription en français
  - Gestion d'erreurs si le modèle n'est pas disponible
- **Processus** :
  1. Upload du fichier audio lors de la création d'une note vocale
  2. Sauvegarde du fichier dans `media/audio_notes/`
  3. Conversion en WAV si nécessaire (stockage temporaire dans `media/temp/`)
  4. Transcription via Vosk
  5. Remplissage automatique du champ `content` avec la transcription
  6. Nettoyage des fichiers temporaires

### 5. Résumés de projet (Partiellement implémenté)
- **Affichage** (`/notes/projects/<id>/summary/`) : Affiche le résumé d'un projet s'il existe
- **Génération** (`/notes/projects/<id>/summary/generate/`) : Génère un résumé (placeholder actuellement)
- **Régénération** (`/notes/projects/<id>/summary/regenerate/`) : Régénère un résumé
- **État actuel** : La fonctionnalité est un placeholder qui génère un texte statique. L'intégration avec un vrai LLM (OpenAI, etc.) n'est pas encore faite, bien que `openai==1.6.0` soit dans les dépendances.

---

## Interface utilisateur

### Design
- Interface inspirée de Notion avec une sidebar fixe
- Design minimaliste avec TailwindCSS
- Responsive (structure flex)

### Composants principaux
- **Sidebar** : Liste des projets avec navigation
- **Barre supérieure** : Titre du projet actuel + informations utilisateur + déconnexion
- **Zone de contenu** : Affichage dynamique selon la page

### Pages disponibles
1. Page de connexion (`note/auth/login.html`)
2. Page de déconnexion (`note/auth/logout.html`)
3. Liste des projets (`note/project_list.html`)
4. Création de projet (`note/project_create.html`)
5. Détail de projet (`note/project_detail.html`)
6. Création de note (`note/note_create.html`)
7. Détail de note (`note/note_detail.html`)
8. Édition de note (`note/note_edit.html`)
9. Résumé de projet (`note/project_summary.html`)

---

## Configuration Docker

### Services Docker Compose

1. **Service `db`** (PostgreSQL) :
   - Image : `postgres:15`
   - Port : 5432
   - Base de données : `noton`
   - Utilisateur/Mot de passe : `postgres/postgres`
   - Volume persistant : `postgres_data`

2. **Service `web`** (Django) :
   - Build depuis `./noton/Dockerfile`
   - Port : 8000
   - Dépend de `db`
   - Variables d'environnement :
     - `DEBUG=True`
     - `SECRET_KEY` (développement uniquement)
     - Configuration base de données
     - Création automatique d'un superutilisateur (admin/admin)
   - Volumes montés :
     - Code source (`./noton:/app`)
     - Médias (`./noton/media:/app/media`)
     - Statiques (`./noton/static:/app/static`)
     - Modèles Vosk (`./models:/app/models`)

### Dockerfile
- Base : `python:3.11-slim`
- Installation des dépendances système : build-essential, libpq-dev, ffmpeg, wget, unzip
- Installation des dépendances Python depuis `requirements.txt`
- Téléchargement automatique du modèle Vosk français (`vosk-model-small-fr-0.22`)
- Configuration Gunicorn (non utilisée en développement)
- Commande par défaut : migrations + collectstatic + runserver

---

## Dépendances principales

### Backend
- `Django>=5.0,<5.1` : Framework web
- `psycopg2>=2.9.5,<3.0` : Driver PostgreSQL
- `gunicorn>=20.1.0,<21.0` : Serveur WSGI
- `dj-database-url>=1.0.0,<2.0` : Configuration DB via URL
- `python-dotenv>=1.0.0,<2.0` : Gestion des variables d'environnement
- `whitenoise>=6.3.0,<7.0` : Service des fichiers statiques en production

### Audio & IA
- `vosk` : Reconnaissance vocale
- `pydub` : Traitement audio
- `openai==1.6.0` : SDK OpenAI (installé mais non utilisé actuellement)

### Autres
- `Pillow>=9.4.0,<10.0` : Traitement d'images
- `django-cors-headers==4.3.1` : Gestion CORS

---

## Points d'attention et limitations

### Fonctionnalités non complètes

1. **Génération de résumés** :
   - Placeholder dans `project_generate_summary()` (lignes 159-163 de `views.py`)
   - Le code génère un texte statique au lieu d'appeler un LLM
   - OpenAI SDK est installé mais non utilisé
   - Nécessite l'intégration d'une API LLM (OpenAI, Anthropic, etc.)

2. **Gestion des erreurs de transcription** :
   - Messages d'erreur basiques
   - Pas de retry automatique
   - Pas de gestion des formats audio non supportés

### Configuration

- **Sécurité** : `SECRET_KEY` et `DEBUG=True` sont en dur pour le développement
- **Base de données** : Configuration flexible (SQLite par défaut, PostgreSQL via env vars)
- **Modèles Vosk** : Téléchargement automatique dans Docker, mais nécessite installation manuelle en local

### Améliorations possibles

1. Intégration complète d'un LLM pour les résumés
2. Support du Markdown dans les notes (mentionné dans le README)
3. Correction automatique des transcriptions via LLM
4. Recherche full-text dans les notes
5. Partage de projets entre utilisateurs
6. Export de projets (PDF, Markdown, etc.)
7. API REST pour intégration externe
8. Tests unitaires et d'intégration

---

## État actuel du code

### Fonctionnel
✅ Authentification utilisateur  
✅ CRUD complet pour projets et notes  
✅ Upload et stockage de fichiers audio  
✅ Transcription audio → texte avec Vosk  
✅ Interface utilisateur basique  
✅ Déploiement Docker fonctionnel  

### À compléter
⚠️ Génération de résumés avec LLM (placeholder actuel)  
⚠️ Gestion avancée des erreurs  
⚠️ Tests automatisés  
⚠️ Documentation API  

---

## Points d'entrée pour continuer le développement

### 1. Intégration LLM pour les résumés
**Fichier** : `noton/note/views.py`, fonction `project_generate_summary()` (lignes 151-172)

**À faire** :
- Remplacer le placeholder par un appel à l'API OpenAI (ou autre LLM)
- Utiliser le SDK OpenAI déjà installé
- Structurer le prompt pour générer un résumé cohérent
- Gérer les erreurs API et les timeouts

### 2. Amélioration de la transcription
**Fichier** : `noton/note/speech_to_text.py`

**À faire** :
- Ajouter une correction automatique via LLM
- Améliorer la gestion des erreurs
- Ajouter un système de retry
- Support de plus de formats audio

### 3. Interface utilisateur
**Fichiers** : Templates dans `noton/note/templates/note/`

**À faire** :
- Ajouter un éditeur Markdown pour les notes
- Améliorer l'UX de la transcription (barre de progression)
- Ajouter des notifications en temps réel
- Améliorer le responsive design

### 4. Tests
**Fichier** : `noton/note/tests.py` (actuellement vide)

**À faire** :
- Tests unitaires pour les modèles
- Tests d'intégration pour les vues
- Tests pour la transcription audio
- Tests pour la génération de résumés

---

## Commandes utiles

### Développement local
```bash
# Activer l'environnement virtuel
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Migrations
python manage.py makemigrations note
python manage.py migrate

# Créer un superutilisateur
python manage.py createsuperuser

# Lancer le serveur
python manage.py runserver
```

### Docker
```bash
# Démarrer les services
docker-compose up -d

# Voir les logs
docker-compose logs -f web

# Arrêter les services
docker-compose down

# Rebuild l'image
docker-compose build --no-cache web
```

---

## Structure des URLs

- `/` → Redirige vers `/notes/`
- `/admin/` → Interface d'administration Django
- `/accounts/login/` → Page de connexion
- `/accounts/logout/` → Déconnexion
- `/notes/` → Liste des projets
- `/notes/projects/create/` → Créer un projet
- `/notes/projects/<id>/` → Détail d'un projet
- `/notes/projects/<id>/delete/` → Supprimer un projet
- `/notes/projects/<id>/notes/create/<type>/` → Créer une note (written/voice)
- `/notes/notes/<id>/` → Détail d'une note
- `/notes/notes/<id>/edit/` → Éditer une note
- `/notes/notes/<id>/delete/` → Supprimer une note
- `/notes/projects/<id>/summary/` → Voir le résumé d'un projet
- `/notes/projects/<id>/summary/generate/` → Générer un résumé
- `/notes/projects/<id>/summary/regenerate/` → Régénérer un résumé

---

## Conclusion

Noton est une application Django fonctionnelle pour la prise de notes avec support audio. L'architecture est solide, les fonctionnalités de base sont implémentées, mais l'intégration LLM pour les résumés reste à compléter. Le code est bien structuré et prêt pour l'extension avec des fonctionnalités IA avancées.

