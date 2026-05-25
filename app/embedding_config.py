"""
Configuration centralisée pour les embeddings.
Modifier EMBEDDING_DIMENSION ici pour changer la dimension utilisée partout.
"""
import os

# Dimension des embeddings (1024 pour mistral-embed)
# Peut être surchargée par variable d'environnement
_raw_dimension = (os.getenv("EMBEDDING_DIMENSION", "1024") or "1024").strip()
try:
    # Accepte accidentellement "1024,256" et prend la première valeur numérique.
    EMBEDDING_DIMENSION = int(_raw_dimension.split(",")[0].strip())
except (TypeError, ValueError):
    EMBEDDING_DIMENSION = 1024

