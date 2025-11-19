from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator
from typing import Optional, Union
from pathlib import Path


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/noton"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: Optional[str] = None
    
    # CPU Optimization for Docling/EasyOCR
    DOCLING_CPU_ONLY: bool = True
    TORCH_NUM_THREADS: Optional[int] = None  # None = utiliser tous les cœurs disponibles
    OMP_NUM_THREADS: Optional[int] = None  # None = utiliser tous les cœurs disponibles
    
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
    
    model_config = ConfigDict(
        # Chercher le fichier .env à la racine du projet (pour développement local)
        # En Docker, les variables sont passées via docker-compose.yaml
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignorer les variables supplémentaires non définies
    )


settings = Settings()

