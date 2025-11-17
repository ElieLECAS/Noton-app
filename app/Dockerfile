FROM python:3.11-slim

WORKDIR /app

# --- OS dependencies ---
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# === STEP 1: Force stable numpy (<2.0) ===
RUN pip install --no-cache-dir "numpy<2.0.0,>=1.24.0"

# === STEP 2: Torch + Transformers (bonnes versions) ===
RUN pip install --no-cache-dir \
    "torch>=2.1.0" \
    "transformers>=4.35.0"

# === STEP 3: sentence-transformers (dépend du dessus) ===
RUN pip install --no-cache-dir "sentence-transformers>=2.7.0"

# === STEP 4: reste des deps FASTAPI & BDD ===
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    sqlmodel==0.0.14 \
    psycopg2-binary==2.9.9 \
    alembic==1.12.1 \
    python-jose[cryptography]==3.3.0 \
    passlib[bcrypt]==1.7.4 \
    bcrypt==4.0.1 \
    python-multipart==0.0.6 \
    httpx==0.25.2 \
    jinja2==3.1.2 \
    python-dotenv==1.0.0 \
    pydantic-settings>=2.3.0 \
    pgvector==0.2.4 \
    faiss-cpu==1.7.4 \
    pillow>=10.0.0 \
    openpyxl>=3.1.0 \
    aiofiles>=23.2.1 \
    pandas>=2.1.0 \
    orjson>=3.9.10

# === STEP 5: Installer Docling en dernier ===
RUN pip install --no-cache-dir "docling>=2.7.0,<3.0.0"

# === STEP 6: Pré-télécharger MiniLM ===
RUN python - << 'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EOF

# Copier le code de l'application
COPY . .

# Exposer le port
EXPOSE 8000

# Commande par défaut
CMD ["sh", "-c", "alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8000"]

