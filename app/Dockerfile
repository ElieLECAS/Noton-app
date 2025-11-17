FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
# Installer NumPy en premier pour éviter qu'une autre dépendance installe NumPy 2.x
COPY requirements.txt .
RUN pip install --no-cache-dir "numpy<2.0.0,>=1.24.0"

# Installer les dépendances de base (sans docling)
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
    aiofiles>=23.2.1

# Installer torch et transformers (dépendent de numpy mais doivent utiliser la version < 2.0)
RUN pip install --no-cache-dir torch>=2.0.0 transformers>=4.30.0

# Installer sentence-transformers (dépend de torch/transformers)
RUN pip install --no-cache-dir sentence-transformers>=2.7.0

# Installer docling séparément pour éviter les conflits de résolution
# Forcer numpy<2.0 pour compatibilité avec faiss-cpu
RUN pip install --no-cache-dir "docling>=2.7.0,<3.0.0" "numpy<2.0.0"

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

