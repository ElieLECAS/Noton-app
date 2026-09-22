"""Configuration de LIA.

Une vingtaine de réglages : base, sécurité, Mistral, le chat, le wiki. Les valeurs viennent du
``.env`` à la racine (développement) ou de l'environnement du conteneur (docker-compose).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from pydantic import ConfigDict, field_validator
from pydantic_settings import BaseSettings


def _as_bool(value: Union[str, bool, None], default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "on")


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "LIA"

    # Base de données
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    DATABASE_ECHO: bool = False

    # Sécurité
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480
    # Cookie de session : Secure par défaut (prod derrière TLS) ; passer à false en local HTTP.
    AUTH_COOKIE_SECURE: bool = True
    AUTH_COOKIE_SAMESITE: str = "lax"
    CORS_ALLOWED_ORIGINS: Optional[List[str]] = None
    # L'utilisateur dont l'email correspond reçoit le rôle admin à la connexion.
    ADMIN_EMAIL: Optional[str] = None

    # Mistral
    MISTRAL_API_KEY: Optional[str] = None
    MISTRAL_BASE_URL: str = "https://api.mistral.ai"
    MISTRAL_MAX_RETRIES: int = 5
    MISTRAL_RETRY_BACKOFF_BASE: float = 2.0
    # Le modèle du chat (Small 4 via l'alias -latest) ; aussi utilisé pour les titres.
    MODEL_FAST: str = "mistral-small-latest"
    # Défaut de max_tokens des appels non streamés (titres de conversation).
    MAX_COMPLETION_TOKENS: int = 1024

    # Le chat CAG — réglages repris du prototype wiki_llm/chat/server.py.
    # Effort de raisonnement demandé à Small ("" = aucun raisonnement).
    GENERATION_REASONING_EFFORT: str = "high"
    CHAT_TEMPERATURE: float = 0.2
    CHAT_MAX_TOKENS: int = 4096
    # Historique renvoyé au modèle : les N derniers messages, bornés en caractères. Il vient
    # APRÈS le prompt système, donc n'entame jamais le préfixe mis en cache.
    CHAT_HISTORY_MAX_MESSAGES: int = 10
    CHAT_HISTORY_MAX_CHARS: int = 24000

    # Le wiki : racine de connaissance contenant wiki/ (les .md) et raw/ (les PDF), relative
    # à la racine du projet ou absolue.
    WIKI_DIR: str = "wiki_llm"

    # La voix — Voxtral. La question est transcrite par lots (Voxtral Mini Transcribe 2 : une
    # demi-seconde pour dix secondes d'audio, avec un biais de vocabulaire tiré du wiki, ce que
    # le modèle temps réel ne permet pas) ; la réponse est dite par Voxtral TTS en flux, phrase
    # par phrase. La voix est un préréglage Mistral désigné par son slug (Marie, français).
    VOCAL_MODELE_TRANSCRIPTION: str = "voxtral-mini-latest"
    VOCAL_MODELE_SYNTHESE: str = "voxtral-mini-tts-latest"
    VOCAL_VOIX: str = "fr_marie_excited"

    @field_validator("DATABASE_ECHO", "AUTH_COOKIE_SECURE", mode="before")
    @classmethod
    def parse_bool(cls, v: Union[str, bool, None]) -> bool:
        return _as_bool(v, True if v is None else False) if v is not None else False

    @field_validator("GENERATION_REASONING_EFFORT", mode="before")
    @classmethod
    def parse_reasoning_effort(cls, v: Union[str, None]) -> str:
        return str(v or "").strip().lower()

    @field_validator("CORS_ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str], None]) -> Optional[List[str]]:
        if v is None:
            return None
        if isinstance(v, list):
            items = [str(o).strip() for o in v if str(o).strip()]
            return items or None
        text = str(v).strip()
        if not text:
            return None
        if text.startswith("["):
            import json

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = []
            items = [str(o).strip() for o in parsed if str(o).strip()]
            return items or None
        items = [o.strip() for o in text.split(",") if o.strip()]
        return items or None

    model_config = ConfigDict(
        # Le .env à la racine du projet (développement local) ; en Docker les variables
        # viennent de docker-compose.yaml.
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
