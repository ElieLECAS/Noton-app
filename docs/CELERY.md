# Redis + Celery — traitement en arrière-plan

## Rôle

- **FastAPI (service `web`)** : API, upload des fichiers, création des enregistrements DB, envoi des jobs.
- **Redis** : broker et backend de résultats pour Celery.
- **Worker (`worker`)** : exécute Docling, chunking et embeddings sur les files `documents` / `embeddings` (un job lourd à la fois avec `-c 1`).
- **Worker KAG (`worker-kag`)** : consomme la file `kag` pour l’extraction d’entités / graphe après les embeddings, en parallèle du worker principal.

## Variables d'environnement

| Variable | Description |
|----------|-------------|
| `TASK_BACKEND_MODE` | `thread` (historique), `celery` (uniquement Celery), `hybrid` (Celery puis repli threads si broker indisponible). |
| `REDIS_URL` | URL Redis, ex. `redis://redis:6379/0` (Docker). |
| `CELERY_BROKER_URL` | Par défaut = `REDIS_URL`. |
| `CELERY_RESULT_BACKEND` | Par défaut = `REDIS_URL`. |

Dans `docker-compose.yaml`, le service `web` utilise par défaut `TASK_BACKEND_MODE=hybrid` ; le service `worker` force `TASK_BACKEND_MODE=celery` (pas de threads inutiles dans le worker).

## Démarrage (Docker)

```bash
docker compose up -d db redis
docker compose up -d web worker worker-kag
```

Sans worker actif et avec `TASK_BACKEND_MODE=celery`, les uploads restent en attente côté queue Redis jusqu’à consommation.

## Démarrage local (sans Docker)

1. Lancer Redis localement (`redis://localhost:6379/0`).
2. Terminal 1 : `uvicorn app.main:app --reload`
3. Terminal 2 : `celery -A app.celery_app worker -l INFO -Q documents,embeddings,celery -c 1 -n main@%h`
4. Terminal 3 : `celery -A app.celery_app worker -l INFO -Q kag -c 1 -n kag@%h` (phase KAG en parallèle du traitement documents/embeddings)

## Réindexation bibliothèque

- Mode thread / repli hybrid synchrone : la réponse JSON contient le résultat habituel (`document_id`, `chunks`, `status`).
- Mode Celery : réponse `{"status": "queued", "celery_task_id": "...", "document_id": ...}`.

## Rollback

Passer `TASK_BACKEND_MODE=thread` et redémarrer le `web` : les workers threads reprennent le traitement sans Redis (Redis peut rester démarré sans effet sur ce mode).

## Files Celery

- `app/celery_app.py` — application Celery (files `documents`, `embeddings`, `kag`, `celery`).
- `app/tasks/documents.py` — tâches (dont `process_library_document_kag` sur la file `kag`).
- `app/services/task_dispatch.py` — routage Celery vs threads (`dispatch_library_document_kag` pour la phase graphe post-embeddings).
