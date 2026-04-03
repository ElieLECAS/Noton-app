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

# Étape 1: PyTorch CPU (base lourde, stable) - index PyTorch pour éviter conflits
RUN pip install --no-cache-dir --upgrade pip "setuptools<82" wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Étape 2: Dépendances système critiques (numpy doit être installé avant docling/llamaindex)
RUN pip install --no-cache-dir \
    "numpy>=2.0,<3.0" \
    "pgvector==0.4.2" \
    "pydantic-settings>=2.3.0,<3.0.0"

# Étape 3: docling-core[chunking] — extra requis pour HybridChunker (semchunk, etc.) sinon import DoclingNodeParser échoue
# docling==2.8.3 : docling-core>=2.6.1,<3.0.0 | llama-index-node-parser-docling 0.4.2 : docling-core>=2.18.0,<3
RUN pip install --no-cache-dir "docling-core[chunking]>=2.18.0,<3.0.0"

# docling-core>=2.18 charge le chunker code qui importe tree_sitter (non toujours installé en transitif)
RUN pip install --no-cache-dir "tree-sitter>=0.21.0,<1.0.0"

# Étape 4: Docling (utilise docling-core déjà installé)
RUN pip install --no-cache-dir "docling==2.8.3"
# docling peut résoudre docling-core sans extras — réappliquer [chunking] pour semchunk / HybridChunker
RUN pip install --no-cache-dir "docling-core[chunking]>=2.18.0,<3.0.0"

# Étape 5: LlamaIndex core + dépendances communes
RUN pip install --no-cache-dir \
    "llama-index-core==0.14.14"

# Étape 6a: Extensions Docling pour LlamaIndex
# readers-docling : --no-deps (docling/docling-core déjà installés)
# node-parser-docling : avec deps pour que pip valide docling-core>=2.18 (déjà satisfait à l'étape 3)
RUN pip install --no-cache-dir --no-deps \
    "llama-index-readers-docling==0.4.2"
RUN pip install --no-cache-dir \
    "llama-index-node-parser-docling==0.4.2"

# Étape 6b: Autres extensions LlamaIndex (sans docling dans le lot)
RUN pip install --no-cache-dir \
    "llama-index-vector-stores-postgres==0.7.3" \
    "llama-index-embeddings-huggingface==0.6.1" \
    "llama-index-postprocessor-flag-embedding-reranker==0.4.1" \
    "FlagEmbedding==1.3.5"

# Étape 7: Reste des dépendances (sans celles déjà installées)
# Filtrer les packages déjà installés + commentaires
RUN grep -v -E '^[[:space:]]*#|^[[:space:]]*$$' requirements.txt | \
    grep -v -E '^(torch|numpy|pgvector|docling|llama-index|FlagEmbedding|pydantic-settings)([[:space:]]|-|==|>=|<|$$)' | \
    sed 's/#.*$$//' | sed '/^[[:space:]]*$$/d' > requirements-remaining.txt && \
    pip install --no-cache-dir -r requirements-remaining.txt && \
    rm requirements-remaining.txt

# Étape 7b: Forcer les versions finales compatibles (évite les upgrades transitifs).
RUN pip install --no-cache-dir \
    "pydantic>=2.0.0,<2.10" \
    "setuptools<82"

# Étape 7c: Vérifier que le chunking hiérarchique Docling est importable (sinon build échoue).
RUN python -c "from llama_index.node_parser.docling import DoclingNodeParser; print('DoclingNodeParser OK')"

# Étape 8: Les modèles seront téléchargés au runtime la première fois et persistés
# via le volume models_cache monté sur /root/.cache dans docker-compose.yaml
# Cela évite de re-télécharger à chaque rebuild de l'image Docker

# Copier le code de l'application dans /app/app/ pour correspondre aux imports "from app."
COPY app/ ./app/

# Copier alembic/ et alembic.ini à la racine /app/ pour que la commande alembic fonctionne depuis le WORKDIR
COPY app/alembic/ ./alembic/
COPY app/alembic.ini ./alembic.ini
COPY tests/ ./tests/
COPY app/pytest.ini ./pytest.ini

# Exposer le port
EXPOSE 8000

# Commande par défaut (cohérente avec docker-compose.yaml)
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]

