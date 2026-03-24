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
    
    # Mistral
    MISTRAL_API_KEY: Optional[str] = None
    MISTRAL_BASE_URL: str = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai")
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: Optional[List[str]] = None
    # Modèle de chat unique (plus de presets private/fast/powerful)
    MODEL_FAST: str = os.getenv("MODEL_FAST", "mistral-small-latest")
    # Limite globale par défaut pour la longueur des réponses des LLM
    MAX_COMPLETION_TOKENS: int = int(os.getenv("MAX_COMPLETION_TOKENS", "1024"))
    
    # CPU Optimization for Docling/EasyOCR
    DOCLING_CPU_ONLY: bool = True
    DOCLING_USE_GPU: Optional[bool] = None  # None = auto-détection, True/False pour forcer
    TORCH_NUM_THREADS: Optional[int] = None  # None = moitié des cœurs si USE_ALL_CPU_CORES=False
    OMP_NUM_THREADS: Optional[int] = None  # Idem ; sinon suit la même logique que PyTorch
    TORCH_NUM_INTEROP_THREADS: int = 1  # Limite la parallélisation interne (moins de pic CPU, un peu plus lent)
    USE_ALL_CPU_CORES: bool = False  # False = moitié des cœurs par défaut (moins de risque de surcharge thermique)
    
    # Document Processing
    MAX_CONCURRENT_DOCUMENTS: int = 1  # Traitement strictement séquentiel: 1 document à la fois
    EMBEDDING_BATCH_SIZE: int = 8  # Batch plus petit = pics mémoire/CPU plus bas (plus lent)
    # Pauses entre jobs pour laisser respirer le CPU (secondes, 0 = désactivé)
    DOCUMENT_WORKER_COOLDOWN_SEC: float = 2.5
    DOCUMENT_WORKER_IDLE_POLL_SEC: float = 3.0  # Quand la file est vide
    EMBEDDING_WORKER_COOLDOWN_SEC: float = 2.0
    # Linux : incrémente la « nice » du thread worker (plus élevé = moins prioritaire), typ. 5–15
    DOCUMENT_PROCESS_NICE_INCREMENT: int = 10
    HIERARCHICAL_CHUNK_SIZES: Optional[List[int]] = None  # Format attendu: "3072,1024,384"

    # Docling OCR (schémas techniques, cotes, PDF scannés)
    DOCLING_OCR_ENABLED: bool = True  # Activer l'OCR pour capturer texte dans les images/schémas
    DOCLING_OCR_LANG: Optional[str] = None  # Langues OCR, ex. "fr,en" ou "fra+eng" (None = défaut Docling)
    
    # Paramètres OCR avancés
    OCR_IMAGE_SCALE: float = 2.0  # 2.0 = moins de charge mémoire/CPU que 3.0 ; monter si qualité OCR insuffisante
    OCR_PREPROCESS_ENABLED: bool = True  # Activer prétraitement adaptatif des images
    OCR_FALLBACK_ENABLED: bool = True  # Activer fallback Tesseract si Docling insuffisant
    OCR_MIN_TEXT_LENGTH: int = 50  # Seuil min caractères/page pour considérer OCR valide
    OCR_TESSERACT_CONFIG: str = "--oem 3 --psm 6 -l fra+eng"  # Config Tesseract (LSTM, bloc uniforme, fr+en)
    
    # Brave Search (recherche web pour function calling)
    BRAVE_SEARCH_API_KEY: Optional[str] = None

    # CORS
    CORS_ALLOWED_ORIGINS: Optional[List[str]] = None  # Liste des origines autorisées (None = toutes les origines)
    
    # RBAC Admin Bootstrap
    ADMIN_EMAIL: Optional[str] = None  # Email de l'utilisateur qui sera automatiquement admin
    
    # KAG - Knowledge Augmented Generation
    KAG_ENABLED: bool = True
    KAG_EXTRACTION_PROVIDER: str = "mistral"  # "openai" ou "mistral"
    KAG_EXTRACTION_MODEL: str = "mistral-large-24b"
    KAG_PARENT_ENRICHMENT_ENABLED: bool = True  # Génère résumé + 3 questions par chunk parent (section)
    
    # Multimodal - Désactivé par défaut (images extraites et stockées par Docling, pas de Vision ni chunks image)
    MULTIMODAL_ENABLED: bool = False
    VISION_MODEL: str = "gpt-4o"  # Modèle pour décrire les images si MULTIMODAL_ENABLED

    @field_validator('MULTIMODAL_ENABLED', mode='before')
    @classmethod
    def parse_multimodal_enabled(cls, v: Union[str, bool, None]) -> bool:
        """Convertit les chaînes en bool pour MULTIMODAL_ENABLED."""
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ('true', '1', 'yes', 'on')
        return False
    
    @field_validator('KAG_ENABLED', mode='before')
    @classmethod
    def parse_kag_enabled(cls, v: Union[str, bool, None]) -> bool:
        """Convertit les chaînes en bool pour KAG_ENABLED."""
        if v is None:
            return True
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ('true', '1', 'yes', 'on')
        return True
    
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

    @field_validator('TORCH_NUM_INTEROP_THREADS', mode='after')
    @classmethod
    def clamp_torch_interop_threads(cls, v: int) -> int:
        return max(1, min(16, v))

    @field_validator('DOCUMENT_PROCESS_NICE_INCREMENT', mode='after')
    @classmethod
    def clamp_document_nice_increment(cls, v: int) -> int:
        return max(0, min(19, v))

    @field_validator(
        'DOCUMENT_WORKER_COOLDOWN_SEC',
        'DOCUMENT_WORKER_IDLE_POLL_SEC',
        'EMBEDDING_WORKER_COOLDOWN_SEC',
        mode='after',
    )
    @classmethod
    def clamp_non_negative_float(cls, v: float) -> float:
        return max(0.0, float(v))

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
    """Compatibilité: retourne toujours le modèle fast unique configuré."""
    return {"provider": "mistral", "model": settings.MODEL_FAST}

