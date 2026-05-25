# Celery (Noton)

## Stack minimale

```bash
docker compose up -d --build
```

Services : `db`, `redis`, `web`, `worker`.

Optionnel (LLM local) :

```bash
docker compose --profile ollama up -d
```

## Fichiers clés

- `app/requirements.txt` — prod (Mistral OCR + LlamaIndex + mistral-embed API)
- `requirements-dev.txt` — pytest en local
- `app/Dockerfile` — image slim sans Docling ni tests
- `docker-compose.yaml` — variables factorisées (`x-app-env`)
