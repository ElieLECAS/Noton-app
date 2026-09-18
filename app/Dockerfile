FROM python:3.11-slim

WORKDIR /app

# Runtime minimal : le client PostgreSQL suffit (plus d'OCR, de PDF ni de modèles locaux).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY app/requirements.txt .

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY app/alembic/ ./alembic/
COPY app/alembic.ini ./alembic.ini

# Le wiki (wiki_llm/) n'est PAS copié dans l'image : il est monté en volume par
# docker-compose, ce qui permet de le mettre à jour par un simple git pull.

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
