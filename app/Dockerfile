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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . .

# Exposer le port
EXPOSE 8000

# Commande par défaut
CMD ["sh", "-c", "alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8000"]

