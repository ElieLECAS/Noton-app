"""
Configuration centralisée pour les embeddings.
Modifier EMBEDDING_DIMENSION ici pour changer la dimension utilisée partout.
"""
import os

# Dimension des embeddings (384 pour all-MiniLM-L6-v2, 768 pour all-mpnet-base-v2)
# Peut être surchargée par variable d'environnement
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

