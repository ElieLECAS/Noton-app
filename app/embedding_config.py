"""
Configuration centralisée pour les embeddings.
Modifier EMBEDDING_DIMENSION ici pour changer la dimension utilisée partout.
"""
import os

# Dimension des embeddings (1024 pour BAAI/bge-m3)
# Peut être surchargée par variable d'environnement
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

