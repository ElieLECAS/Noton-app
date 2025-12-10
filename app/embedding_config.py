"""
Configuration centralisée pour les embeddings.
Modifier EMBEDDING_DIMENSION ici pour changer la dimension utilisée partout.
"""
import os

# Dimension des embeddings (768 pour Nomic Embed v1)
# Peut être surchargée par variable d'environnement
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

