from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator
from typing import Optional, Union, List
from pathlib import Path
import os


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Noton"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480
    
    # Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL")
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: Optional[List[str]] = None
    OPENAI_IMAGE_MODEL: str = "dall-e-3"  # Modèle de génération d'images (DALL-E 3)
    
    # Modèles de chat configurables
    MODEL_PRIVATE_PROVIDER: str = os.getenv("MODEL_PRIVATE_PROVIDER")
    MODEL_PRIVATE_NAME: str = os.getenv("MODEL_PRIVATE_NAME")
    MODEL_FAST_PROVIDER: str = os.getenv("MODEL_FAST_PROVIDER")
    MODEL_FAST_NAME: str = os.getenv("MODEL_FAST_NAME")
    MODEL_POWERFUL_PROVIDER: str = os.getenv("MODEL_POWERFUL_PROVIDER")
    MODEL_POWERFUL_NAME: str = os.getenv("MODEL_POWERFUL_NAME")
    
    # CPU Optimization for Docling/EasyOCR
    DOCLING_CPU_ONLY: bool = True
    DOCLING_USE_GPU: Optional[bool] = None  # None = auto-détection, True/False pour forcer
    TORCH_NUM_THREADS: Optional[int] = None  # None = utiliser tous les cœurs disponibles
    OMP_NUM_THREADS: Optional[int] = None  # None = utiliser tous les cœurs disponibles
    USE_ALL_CPU_CORES: bool = True  # Utiliser tous les cœurs par défaut (au lieu de la moitié) pour maximiser les performances
    
    # Document Processing
    MAX_CONCURRENT_DOCUMENTS: int = 2  # Nombre de documents traités en parallèle (réduit pour garder des ressources pour la navigation)
    EMBEDDING_BATCH_SIZE: int = 16  # Taille de batch embedding (CPU-only, éviter la saturation)
    HIERARCHICAL_CHUNK_SIZES: Optional[List[int]] = None  # Format attendu: "3072,1024,384"

    # Docling OCR (schémas techniques, cotes, PDF scannés)
    DOCLING_OCR_ENABLED: bool = True  # Activer l'OCR pour capturer texte dans les images/schémas
    DOCLING_OCR_LANG: Optional[str] = None  # Langues OCR, ex. "fr,en" ou "fra+eng" (None = défaut Docling)
    
    # Brave Search (recherche web pour function calling)
    BRAVE_SEARCH_API_KEY: Optional[str] = None

    # CORS
    CORS_ALLOWED_ORIGINS: Optional[List[str]] = None  # Liste des origines autorisées (None = toutes les origines)
    
    @field_validator('OPENAI_MODEL', mode='before')
    @classmethod
    def parse_openai_models(cls, v: Union[str, List[str], None]) -> Optional[List[str]]:
        """Convertit une chaîne séparée par des virgules en liste de modèles"""
        if v is None:
            return None
        if isinstance(v, str):
            # Si c'est une chaîne vide, retourner None
            if not v.strip():
                return None
            # Séparer par des virgules et nettoyer les espaces
            items = [item.strip() for item in v.split(',') if item.strip()]
            return items if items else None
        # Si c'est déjà une liste, la retourner telle quelle
        if isinstance(v, list):
            return v if v else None
        return None
    
    @field_validator('CORS_ALLOWED_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str], None]) -> Optional[List[str]]:
        """Convertit une chaîne séparée par des virgules en liste d'origines CORS, en enlevant les slashes finaux"""
        if v is None:
            return None
        if isinstance(v, str):
            # Si c'est une chaîne vide, retourner None
            if not v.strip():
                return None
            # Séparer par des virgules et nettoyer les espaces
            items = [item.strip() for item in v.split(',') if item.strip()]
            # Enlever les slashes finaux pour normaliser les URLs
            normalized = [item.rstrip('/') for item in items]
            return normalized if normalized else None
        # Si c'est déjà une liste, normaliser les URLs
        if isinstance(v, list):
            normalized = [item.rstrip('/') if isinstance(item, str) else item for item in v if item]
            return normalized if normalized else None
        return None
    
    @field_validator('DOCLING_OCR_LANG', mode='before')
    @classmethod
    def parse_ocr_lang(cls, v: Union[str, None]) -> Optional[str]:
        """Chaîne vide → None pour DOCLING_OCR_LANG."""
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return v.strip() if isinstance(v, str) else v

    @field_validator('DOCLING_USE_GPU', mode='before')
    @classmethod
    def parse_optional_bool(cls, v: Union[str, bool, None]) -> Optional[bool]:
        """Convertit les chaînes en bool pour DOCLING_USE_GPU"""
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            v_lower = v.strip().lower()
            if v_lower in ('true', '1', 'yes', 'on'):
                return True
            elif v_lower in ('false', '0', 'no', 'off'):
                return False
            elif v_lower == '':
                return None
        return None
    
    @field_validator('TORCH_NUM_THREADS', 'OMP_NUM_THREADS', mode='before')
    @classmethod
    def parse_optional_int(cls, v: Union[str, int, None]) -> Optional[int]:
        """Convertit les chaînes vides en None pour les champs int optionnels"""
        if v is None:
            return None
        if isinstance(v, str):
            # Si c'est une chaîne vide, retourner None
            if not v.strip():
                return None
            # Sinon, essayer de parser comme int
            try:
                return int(v)
            except ValueError:
                return None
        # Si c'est déjà un int, le retourner tel quel
        return int(v)

    @field_validator('HIERARCHICAL_CHUNK_SIZES', mode='before')
    @classmethod
    def parse_chunk_sizes(cls, v: Union[str, List[int], None]) -> Optional[List[int]]:
        """Convertit une chaîne CSV en liste d'entiers pour le chunking hiérarchique."""
        if v is None:
            return None
        if isinstance(v, str):
            if not v.strip():
                return None
            try:
                parsed = [int(item.strip()) for item in v.split(",") if item.strip()]
                parsed = [item for item in parsed if item > 0]
                return parsed if parsed else None
            except ValueError:
                return None
        if isinstance(v, list):
            parsed = [int(item) for item in v if int(item) > 0]
            return parsed if parsed else None
        return None
    
    model_config = ConfigDict(
        # Chercher le fichier .env à la racine du projet (pour développement local)
        # En Docker, les variables sont passées via docker-compose.yaml
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignorer les variables supplémentaires non définies
    )


settings = Settings()


def get_model_for_preset(preset: Optional[str]) -> dict:
    """Retourne provider et model pour un preset (même logique que le chat / template globals)."""
    preset = (preset or "").strip().lower()
    if preset == "powerful":
        return {"provider": settings.MODEL_POWERFUL_PROVIDER, "model": settings.MODEL_POWERFUL_NAME}
    if preset == "private":
        return {"provider": settings.MODEL_PRIVATE_PROVIDER, "model": settings.MODEL_PRIVATE_NAME}
    # "fast" ou défaut (comme le chat)
    return {"provider": settings.MODEL_FAST_PROVIDER, "model": settings.MODEL_FAST_NAME}

