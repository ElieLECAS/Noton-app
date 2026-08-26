# LIA — Assistant documentaire RAG multimodal (PROFERM)

Assistant de recherche sur la documentation technique de menuiserie de PROFERM
(gammes PVC, ALU, Hybride, Textural et leurs fournisseurs). L'application indexe des
PDF techniques (notices de pose, fiches produit, PV, normes) et répond aux questions
métier en s'appuyant sur un pipeline RAG multimodal (texte **et** visuel).

> Le nom d'image Docker du produit est `lia` ; le dépôt historique s'appelle `Noton-app`.

## Ce que fait l'application

- **Bibliothèque** : upload de documents, OCR/extraction (Mistral OCR + vision par page),
  chunking hiérarchique, embeddings, indexation ColPali (visuel) et pgvector (texte).
- **Espaces** : compartiments documentaires ; le chat est *scopé* à un espace.
- **Chat RAG** : récupération multi-canal → fusion → (rerank) → packing de contexte → génération.
- **KAG** : graphe d'entités/relations + classification de catégories multi-axes (facettes).
- **Admin/RBAC** : gestion utilisateurs, rôles, permissions, taxonomie de catégories, éval retrieval.

## Pipeline de retrieval (chemin chat espace)

1. **Query understanding** (`QUERY_UNDERSTANDING_ENABLED`) : un appel LLM fusionné — le seul
   avant le retrieval — produit la route, la question autonome, les `signals` (catégories
   inférées, source, matériau, intent), le `topic_shift` et l'ancrage conversationnel. Les
   requêtes retriever (`colpali` / `lexical`) sont ensuite construites sans LLM.
2. **2 retrievers en parallèle** : ColPali (visuel, *gated*) et BM25 (lexical, tsvector).
   La voie dense texte (pgvector) a été retirée le 2026-08-25 — elle lisait la même évidence
   que BM25 et votait deux fois au RRF ; le canal KAG l'avait été le 2026-07-28.
3. **Fusion RRF** puis **boost catégorie** (multiplicatif sur `rrf_score`) et **ancrage**
   conversationnel (continuité du sujet entre tours).
4. **Rerank MiniLM** cross-encoder (`RERANKER_ENABLED`).
5. **CAG** (`CAG_ENABLED`) : au lieu d'injecter des passages tronqués, packe des documents
   entiers / fenêtrés dans le contexte (fenêtre 256k), avec budgets par intent.
6. **Génération** (Mistral) en streaming, avec repli `full → eco → minimal` sur erreur 400.

Détails et audits dans `docs/`.

## Stack

- **Backend** : FastAPI (Python 3.11), SQLModel
- **Base** : PostgreSQL 15 + `pgvector` ; **LanceDB** pour les vecteurs ColPali
- **Tâches** : Celery + Redis (ou workers threads selon `TASK_BACKEND_MODE`)
- **IA** : Mistral (LLM + OCR + vision), ColQwen2 (ColPali), cross-encoder MiniLM (rerank)
- **Front** : templates Jinja2 + HTML/CSS/JS
- **Déploiement** : Docker Compose

## Installation

Prérequis : Docker + Docker Compose, une clé API Mistral, un fichier `.env` à la racine
(voir les variables consommées dans `docker-compose.yaml` et les défauts dans `app/config.py`).

```bash
docker compose up -d          # db (pgvector) + redis + web + worker
# Application : http://localhost:8001
```

Les migrations Alembic sont exécutées automatiquement par la commande du conteneur `web`
(`alembic upgrade head`) avant le démarrage d'uvicorn — point d'exécution unique.

## Configuration

**Source de vérité** : `app/config.py` porte les défauts ; `.env` les surcharge en prod.
Les `${VAR:-défaut}` de `docker-compose.yaml` ne sont que des replis, tenus alignés sur
`config.py`.

- Au démarrage, l'app journalise une **matrice de features** (`[config] retrieval: …`) et des
  **avertissements de cohérence** (ex. un flag maître désactivé qui rend des features inertes).
- Vérifier la config effective en un coup d'œil (admin) : `GET /api/admin/config`.

Flags principaux : `QUERY_UNDERSTANDING_ENABLED`, `RERANKER_ENABLED`, `VISION_RERANK_ENABLED`,
`COLPALI_ENABLED` / `COLPALI_GATING_ENABLED`, `KAG_ENABLED`, `CAG_ENABLED`,
`CONVERSATION_ANCHOR_ENABLED`, `FICHE_TECHNIQUE_ENABLED`, `MULTIMODAL_ENABLED`.

> Cookie d'auth : `AUTH_COOKIE_SECURE=true` par défaut (prod HTTPS derrière nginx). En dev
> local sur `http://`, mettre `AUTH_COOKIE_SECURE=false` dans le `.env`, sinon le navigateur
> refuse le cookie de session.

## Principales routes API

- **Auth** : `POST /api/auth/register|login|logout`, `GET /api/auth/me`
- **Bibliothèque** : `POST /api/library/upload`, `PUT /api/library/documents/{id}`,
  `POST /api/library/documents/{id}/spaces`, `GET /api/library/classification-options`
- **Espaces** : `GET/POST /api/spaces`, chat streaming scopé à l'espace
- **Conversations** : `GET/POST/DELETE /api/conversations…`
- **Admin** : utilisateurs/rôles/permissions, catégories (`/api/admin/categories`),
  config effective (`/api/admin/config`), éval retrieval (`/api/admin/eval/retriever`)

## Développement

```bash
# Tests (dépendances Docker-only)
docker compose exec web pytest

# Migrations
docker compose exec web alembic revision --autogenerate -m "Description"
docker compose exec web alembic upgrade head

# Logs
docker compose logs -f web
```

### Évaluation du retrieval

Harnais `app/services/retriever_evaluator.py` (métriques `context_precision@K`, `match_page`,
recall/MRR, LLM-judge). Datasets de vérité-terrain (golden) par fournisseur dans
`tests/fixtures/golden/` (ROTO, Profine, Kommerling), passés en corps de requête à
`POST /api/admin/eval/retriever`.
