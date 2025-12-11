FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système pour docling et PostgreSQL
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    libreoffice \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application dans /app/app/ pour correspondre aux imports "from app."
COPY app/ ./app/

# Copier alembic/ et alembic.ini à la racine /app/ pour que la commande alembic fonctionne depuis le WORKDIR
COPY app/alembic/ ./alembic/
COPY app/alembic.ini ./alembic.ini

# Exposer le port
EXPOSE 8000

# Commande par défaut (cohérente avec docker-compose.yaml)
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]

