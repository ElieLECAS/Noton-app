FROM python:3.11-slim

WORKDIR /app

# Runtime minimal : PostgreSQL client, LibreOffice (Office→PDF), poppler (pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libreoffice \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY app/requirements.txt .

RUN pip install --no-cache-dir -U pip "wheel" "setuptools<82" \
    && pip install --no-cache-dir -r requirements.txt

RUN python -c "\
from llama_index.core.node_parser import HierarchicalNodeParser; \
from mistralai import Mistral; \
print('Deps OK')"

COPY app/ ./app/
COPY app/alembic/ ./alembic/
COPY app/alembic.ini ./alembic.ini

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
