# Celery (Noton)

## Stack minimale

```bash
docker compose up -d --build
```

Services : `db`, `redis`, `web`, `worker`.

Concurrence worker : `CELERY_WORKER_CONCURRENCY` (défaut **1**) — un seul job document/embeddings/multimodal à la fois par worker (évite le rate limit Mistral sur plusieurs PDF).

Parallélisme intra-document : `MULTIMODAL_PAGE_CONCURRENCY` (défaut **3**) — jusqu’à 3 pages traitées en parallèle dans un même retraitement multimodal (~3× plus rapide par document).

Redémarrer le worker après changement : `docker compose up -d worker`.

Optionnel (LLM local) :

```bash
docker compose --profile ollama up -d
```

## Pipeline de traitement des documents

**Pipeline multimodal unifié** (pymupdf + Mistral Small vision) :
- Upload et retraitement utilisent le même pipeline multimodal de haute qualité
- Génère 1-5 sections + synthèse technique par page
- Remplace l'ancien pipeline PyMuPDF4LLM + Mistral OCR
- Chunks multimodaux optimisés pour le RAG (métadonnées riches, références normatives, contraintes techniques)

## Fichiers clés

- `app/requirements.txt` — prod (Mistral API + LlamaIndex + mistral-embed API)
- `requirements-dev.txt` — pytest en local
- `app/Dockerfile` — image slim sans Docling ni tests
- `docker-compose.yaml` — variables factorisées (`x-app-env`)
