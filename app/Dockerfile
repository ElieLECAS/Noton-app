FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pré-télécharger le modèle sentence-transformers pour éviter le téléchargement au runtime
# Le modèle sera dans l'image Docker, donc pas besoin de re-télécharger à chaque build
# Note: Le modèle sera téléchargé au premier usage si cette étape échoue
# Utiliser le modèle léger all-MiniLM-L6-v2 (384 dimensions) par défaut
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" || echo "Modèle sera téléchargé au runtime"

# Copier le code de l'application
COPY . .

# Exposer le port
EXPOSE 8000

# Commande par défaut
CMD ["sh", "-c", "alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8000"]

