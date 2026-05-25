"""Smoke test embeddings Mistral API (retrieval vectoriel sans reranker)."""
import os

from app.config import settings
from app.services.embedding_service import generate_embedding

if __name__ == "__main__":
    if not os.getenv("MISTRAL_API_KEY") and not settings.MISTRAL_API_KEY:
        raise SystemExit("MISTRAL_API_KEY requise")
    v = generate_embedding("test connexion embedding")
    if not v:
        raise SystemExit("Échec génération embedding")
    print("OK — model:", settings.EMBEDDING_MODEL, "dim:", len(v))
