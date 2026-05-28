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
    # echo=True journalise chaque SQL (UPDATE/INSERT d'embeddings = vecteurs énormes dans les logs)
    DATABASE_ECHO: bool = False

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
    # Paramètres dédiés au chat "espaces"
    SPACE_CHAT_MAX_TOKENS: Optional[int] = None
    SPACE_CHAT_TEMPERATURE: float = 0.55
    SPACE_CHAT_TOP_P: Optional[float] = None
    # Document Processing
    MAX_CONCURRENT_DOCUMENTS: int = 1
    EMBEDDING_BATCH_SIZE: int = 16
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_MODEL: str = "mistral-embed"
    HIERARCHICAL_CHUNK_SIZES: Optional[List[int]] = None  # ex. "1024,256" pour override ingestion OCR
    USE_MARKDOWN_STRUCTURED_CHUNKING: bool = False  # Si True, utilise MarkdownNodeParser au lieu de HierarchicalNodeParser
    PDF_FORCE_OCR: bool = False  # Si True, skip pymupdf4llm et utilise Mistral OCR pour tous les PDF
    # Profondeur de titre pour regrouper les parents (1 = ex. tout "CATALOGUE TEXTURES EXTERIEURES")
    MARKDOWN_STRUCTURED_PARENT_DEPTH: int = 1

    # Mistral OCR (ingestion documents)
    MISTRAL_OCR_MODEL: str = "mistral-ocr-latest"
    MISTRAL_OCR_TIMEOUT: float = 300.0
    
    # Connaissances correctives issues des feedbacks (double recherche RAG)
    FEEDBACK_KNOWLEDGE_TITLE_PREFIX: str = "Connaissance technique"
    # Alias legacy (docs/titres existants « FAQ Corrective - … »)
    FAQ_CORRECTIVE_TITLE_PREFIX: str = "FAQ Corrective"
    FAQ_TOP_K: int = 3
    FAQ_MIN_SIMILARITY: float = 0.30
    FAQ_POST_DRAFT_ENABLED: bool = True
    
    # Brave Search (recherche web pour function calling)
    BRAVE_SEARCH_API_KEY: Optional[str] = None

    # CORS
    CORS_ALLOWED_ORIGINS: Optional[List[str]] = None  # Liste des origines autorisées (None = toutes les origines)
    
    # RBAC Admin Bootstrap
    ADMIN_EMAIL: Optional[str] = None  # Email de l'utilisateur qui sera automatiquement admin
    
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

    # Multimodal : lu depuis l’env MULTIMODAL_ENABLED (.env ou docker-compose) ;
    # False = défaut si la variable est absente (voir parse_multimodal_enabled).
    MULTIMODAL_ENABLED: bool = False
    # Pixtral via API Mistral (ex. pixtral-12b-2409) pour enrichir les chunks feuilles « picture »
    VISION_MODEL: str = "pixtral-12b-2409"
    VISION_MAX_TOKENS: int = 1500
    # Plafond d’appels vision par document (None = illimité)
    VISION_MAX_IMAGES_PER_DOCUMENT: Optional[int] = None
    # Retraitement multimodal par page (pymupdf + mistral-small vision)
    MULTIMODAL_PAGE_MODEL: str = "mistral-small-latest"
    MULTIMODAL_EXTRACT_MODEL: str = "mistral-large-latest"
    MULTIMODAL_PAGE_DPI: int = 200
    MULTIMODAL_PAGE_MAX_TOKENS: int = 8000
    # Pages traitées en parallèle au sein d’un même document multimodal
    MULTIMODAL_PAGE_CONCURRENCY: int = 3
    # v4 : seuil caractères pymupdf pour mode native vs scanned
    MULTIMODAL_NATIVE_TEXT_MIN_CHARS: int = 100
    # v4 : fenêtres dynamiques Pass 2 (rapports pro multi-pages)
    WINDOW_MAX_INPUT_TOKENS: int = 7000
    WINDOW_MAX_PAGES: int = 2
    WINDOW_PAGE_OVERLAP: int = 0
    MAX_REPORTS_PER_WINDOW: int = 2
    MISTRAL_PASS2_TIMEOUT: int = 60
    # v4 : découpe RAG-friendly (overlap entre parts consécutives)
    RAG_CHUNK_OVERLAP_TOKENS: int = 40
    # v4 : expansion retrieval locale (voisins + même window_id)
    RETRIEVAL_EXPAND_ENABLED: bool = True
    RETRIEVAL_PAGE_RADIUS: int = 1
    RERANK_GROUP_CHAR_CAP: int = 2800

    # Tâches background : thread (historique), celery (Redis), hybrid (Celery + repli threads)
    TASK_BACKEND_MODE: str = "thread"
    REDIS_URL: Optional[str] = None  # ex. redis://redis:6379/0
    CELERY_BROKER_URL: Optional[str] = None  # défaut: REDIS_URL
    CELERY_RESULT_BACKEND: Optional[str] = None  # défaut: REDIS_URL
    # Concurrence worker Celery : 1 job document lourd à la fois (parallélisme pages via MULTIMODAL_PAGE_CONCURRENCY)
    CELERY_WORKER_CONCURRENCY: int = 1

    # LangSmith — observabilité RAG
    LANGSMITH_API_KEY: Optional[str] = None
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "noton-rag"

    # Reranker cross-encoder (CPU-only)
    RERANKER_ENABLED: bool = False
    RERANKER_PROVIDER: str = os.getenv("RERANKER_PROVIDER", "local")  # local | mistral
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_POOL: int = 30  # Réduit de 50 → 30 pour MiniLM CPU (latence vs qualité)
    RERANK_CHAR_CAP: int = 1700  # ~485 tokens (ratio FR 3.5 chars/token, marge vs max_length=512)
    RERANK_BATCH_SIZE: int = 16
    EARLY_STOP_ENABLED: bool = False  # Early stop désactivé par défaut (latence CPU acceptable)
    EARLY_STOP_TOP_N: int = 5
    EARLY_STOP_MEAN_THRESHOLD: float = 0.78
    MIN_DYNAMIC_K: int = 1
    MAX_DYNAMIC_K: int = 8  # Réduit de 10 → 8 pour MiniLM (chunks plus précis)
    SOFTMAX_CUM_THRESHOLD: float = 0.80
    STUTTER_GAP: float = 0.05
    ZSCORE_FLAT_THRESHOLD: float = 0.05

    # MMR (Maximal Marginal Relevance) — diversification du contexte
    MMR_ENABLED: bool = True
    MMR_K: int = 12
    MMR_LAMBDA: float = 0.7
    MMR_MAX_PER_PARENT: int = 5

    # RRF (Reciprocal Rank Fusion) dynamique
    RRF_DYNAMIC_K_ENABLED: bool = True
    RRF_MIN_K: int = 1
    RRF_MAX_K: int = 10
    RRF_RELATIVE_THRESHOLD_FACTOR: float = 0.70

    # BM25 Lexical Search (approximation via ts_rank_cd + IDF Python)
    BM25_K1: float = 1.2
    BM25_B: float = 0.75
    BM25_MAX_QUERY_TERMS: int = 15

    @field_validator('DATABASE_ECHO', mode='before')
    @classmethod
    def parse_database_echo(cls, v: Union[str, bool, None]) -> bool:
        """SQLAlchemy echo : désactivé par défaut (évite de logger les vecteurs d'embedding)."""
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ('true', '1', 'yes', 'on')
        return False

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

    @field_validator('RERANKER_ENABLED', 'MMR_ENABLED', 'RRF_DYNAMIC_K_ENABLED', 'EARLY_STOP_ENABLED', mode='before')
    @classmethod
    def parse_bool_flags(cls, v: Union[str, bool, None]) -> bool:
        """Convertit les chaînes en bool pour les flags reranker/MMR/RRF/early_stop."""
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ('true', '1', 'yes', 'on')
        return False
    
    @field_validator('TASK_BACKEND_MODE', mode='before')
    @classmethod
    def parse_task_backend_mode(cls, v: Union[str, None]) -> str:
        """thread | celery | hybrid"""
        if v is None or (isinstance(v, str) and not v.strip()):
            return "thread"
        s = str(v).strip().lower()
        if s in ("thread", "celery", "hybrid"):
            return s
        return "thread"

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
    
    @field_validator('VISION_MAX_TOKENS', mode='before')
    @classmethod
    def parse_vision_max_tokens(cls, v: Union[str, int, None]) -> int:
        if v is None or (isinstance(v, str) and not v.strip()):
            return 1500
        try:
            return max(64, int(v))
        except (TypeError, ValueError):
            return 1500

    @field_validator('CELERY_WORKER_CONCURRENCY', mode='before')
    @classmethod
    def parse_celery_worker_concurrency(cls, v: Union[str, int, None]) -> int:
        if v is None or (isinstance(v, str) and not v.strip()):
            return 1
        try:
            return max(1, min(16, int(v)))
        except (TypeError, ValueError):
            return 1

    @field_validator('MULTIMODAL_PAGE_CONCURRENCY', mode='before')
    @classmethod
    def parse_multimodal_page_concurrency(cls, v: Union[str, int, None]) -> int:
        if v is None or (isinstance(v, str) and not v.strip()):
            return 3
        try:
            return max(1, min(8, int(v)))
        except (TypeError, ValueError):
            return 3

    @field_validator('MULTIMODAL_PAGE_DPI', mode='before')
    @classmethod
    def parse_multimodal_page_dpi(cls, v: Union[str, int, None]) -> int:
        if v is None or (isinstance(v, str) and not v.strip()):
            return 200
        try:
            return max(72, min(400, int(v)))
        except (TypeError, ValueError):
            return 200

    @field_validator('MULTIMODAL_PAGE_MAX_TOKENS', mode='before')
    @classmethod
    def parse_multimodal_page_max_tokens(cls, v: Union[str, int, None]) -> int:
        if v is None or (isinstance(v, str) and not v.strip()):
            return 8000
        try:
            return max(256, int(v))
        except (TypeError, ValueError):
            return 8000

    @field_validator(
        'SPACE_CHAT_MAX_TOKENS',
        'VISION_MAX_IMAGES_PER_DOCUMENT',
        mode='before',
    )
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

    @field_validator('SPACE_CHAT_TOP_P', mode='before')
    @classmethod
    def parse_optional_float(cls, v: Union[str, float, int, None]) -> Optional[float]:
        """Convertit les chaînes vides en None pour les champs float optionnels"""
        if v is None:
            return None
        if isinstance(v, str):
            if not v.strip():
                return None
            try:
                return float(v)
            except ValueError:
                return None
        return float(v)

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

