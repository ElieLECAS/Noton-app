"""
Configuration centralisée pour les embeddings.
Modifier EMBEDDING_DIMENSION ici pour changer la dimension utilisée partout.
"""
import os

# Dimension des embeddings (384 pour BAAI/bge-small-en-v1.5)
# Peut être surchargée par variable d'environnement
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

