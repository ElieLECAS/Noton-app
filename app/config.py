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
    
    # LanceDB
    LANCED_DB_DIR: str = os.getenv("LANCED_DB_DIR", "./data/lancedb")

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480
    
    # Mistral
    MISTRAL_API_KEY: Optional[str] = None
    MISTRAL_BASE_URL: str = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai")
    MISTRAL_MAX_RETRIES: int = 5
    MISTRAL_RETRY_BACKOFF_BASE: float = 2.0
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: Optional[List[str]] = None
    # Modèle de chat unique (plus de presets private/fast/powerful)
    MODEL_FAST: str = os.getenv("MODEL_FAST", "mistral-small-latest")
    # Modèle dédié à la phase « compréhension de requête » (route, signaux, vagueness,
    # génération de requêtes) : tâches JSON simples → petit modèle rapide. Évite de payer
    # la latence d'un gros modèle (mistral-large) sur 4-5 appels avant le retrieval.
    MODEL_QUERY_UNDERSTANDING: str = os.getenv("MODEL_QUERY_UNDERSTANDING", "mistral-small-latest")
    # Provider pour le LLM (mistral | ollama)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mistral")
    # Limite globale par défaut pour la longueur des réponses des LLM
    MAX_COMPLETION_TOKENS: int = int(os.getenv("MAX_COMPLETION_TOKENS", "1024"))
    # Paramètres dédiés au chat "espaces"
    # Plafond de longueur de réponse transmis à l'API de génération. Vide/None → le wrapper
    # applique un plancher de 2048 (au lieu du repli global MAX_COMPLETION_TOKENS=1024) pour
    # ne pas couper les réponses procédurales longues. Fenêtre Mistral Large 256k → marge.
    SPACE_CHAT_MAX_TOKENS: Optional[int] = None
    SPACE_CHAT_TEMPERATURE: float = float(os.getenv("SPACE_CHAT_TEMPERATURE", "0.3"))
    SPACE_CHAT_TOP_P: Optional[float] = None
    # Synthèse "carte mentale" (CAG plein-contexte par nœud d'arbre thématique)
    SYNTHESIS_MAX_TOKENS: int = int(os.getenv("SYNTHESIS_MAX_TOKENS", "2500"))
    SYNTHESIS_MAX_CONTEXT_CHARS: int = int(os.getenv("SYNTHESIS_MAX_CONTEXT_CHARS", "350000"))
    SYNTHESIS_TEMPERATURE: float = float(os.getenv("SYNTHESIS_TEMPERATURE", "0.2"))
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
    FAQ_POST_DRAFT_ENABLED: bool = False
    
    # Brave Search (recherche web pour function calling)
    BRAVE_SEARCH_API_KEY: Optional[str] = None
 
    # CORS
    CORS_ALLOWED_ORIGINS: Optional[List[str]] = None  # Liste des origines autorisées (None = toutes les origines)
    
    # RBAC Admin Bootstrap
    ADMIN_EMAIL: Optional[str] = None  # Email de l'utilisateur qui sera automatiquement admin
    
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

    # Multimodal : lu depuis l’env MULTIMODAL_ENABLED (.env ou docker-compose) ;
    # False = défaut si la variable est absente (voir parse_multimodal_enabled).
    MULTIMODAL_ENABLED: bool = True
    
    # ColPali Settings
    COLPALI_ENABLED: bool = True
    COLPALI_MODEL_NAME: str = "vidore/colqwen2-v0.1"
    # Pixtral via API Mistral (ex. pixtral-12b-2409) pour enrichir les chunks feuilles « picture »
    VISION_MODEL: str = "pixtral-12b-2409"
    VISION_MAX_TOKENS: int = 1500
    # Plafond d’appels vision par document (None = illimité)
    VISION_MAX_IMAGES_PER_DOCUMENT: Optional[int] = None
    # Extraction vision par page (pipeline d'indexation documentaire)
    # mistral-small-latest : multimodal (vision), remplace ministral-8b pour une transcription plus fidèle.
    PAGE_EXTRACTION_MODEL: str = os.getenv("PAGE_EXTRACTION_MODEL", "mistral-small-latest")
    PAGE_EXTRACTION_MAX_CHUNK_TOKENS: int = int(os.getenv("PAGE_EXTRACTION_MAX_CHUNK_TOKENS", "480"))
    PAGE_EXTRACTION_MAX_CHUNKS_PER_PAGE: int = int(os.getenv("PAGE_EXTRACTION_MAX_CHUNKS_PER_PAGE", "12"))
    # 300 DPI : nécessaire pour que les cotes minuscules des dessins techniques CAO
    # (dimensions, codes profilés) soient lisibles et non hallucinées par le VLM.
    PAGE_EXTRACTION_DPI: int = int(os.getenv("PAGE_EXTRACTION_DPI", "300"))
    PAGE_EXTRACTION_CONCURRENCY: int = int(os.getenv("PAGE_EXTRACTION_CONCURRENCY", "3"))
    # mistral-small un peu plus lent que ministral-8b → marge timeout portée à 120s
    PAGE_EXTRACTION_TIMEOUT: float = float(os.getenv("PAGE_EXTRACTION_TIMEOUT", "120"))
    PAGE_EXTRACTION_MAX_TOKENS: int = int(os.getenv("PAGE_EXTRACTION_MAX_TOKENS", "4096"))

    # KAG — extraction entités/relations et retrieval graphe
    KAG_ENABLED: bool = os.getenv("KAG_ENABLED", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # mistral-small-latest (vision) par défaut, ancré sur le texte L1 déjà extrait.
    KAG_EXTRACTION_MODEL: Optional[str] = os.getenv("KAG_EXTRACTION_MODEL") or "mistral-small-latest"
    KAG_EXTRACTION_CONCURRENCY: int = int(os.getenv("KAG_EXTRACTION_CONCURRENCY", "3"))
    KAG_EXTRACTION_TIMEOUT: float = float(os.getenv("KAG_EXTRACTION_TIMEOUT", "120"))
    KAG_EXTRACTION_MAX_TOKENS: int = int(os.getenv("KAG_EXTRACTION_MAX_TOKENS", "4096"))
    KAG_MAX_ENTITIES_PER_PAGE: int = int(os.getenv("KAG_MAX_ENTITIES_PER_PAGE", "20"))
    KAG_MAX_RELATIONS_PER_PAGE: int = int(os.getenv("KAG_MAX_RELATIONS_PER_PAGE", "15"))
    # Plafonds réduits automatiquement pour ministral-3b et modèles compacts
    KAG_SMALL_MODEL_MAX_ENTITIES: int = int(os.getenv("KAG_SMALL_MODEL_MAX_ENTITIES", "8"))
    KAG_SMALL_MODEL_MAX_RELATIONS: int = int(os.getenv("KAG_SMALL_MODEL_MAX_RELATIONS", "6"))
    KAG_RETRIEVAL_HOP_LIMIT: int = int(os.getenv("KAG_RETRIEVAL_HOP_LIMIT", "1"))
    KAG_ENTITY_MATCH_MIN_SCORE: float = float(os.getenv("KAG_ENTITY_MATCH_MIN_SCORE", "0.35"))
    KAG_BATCH_SIZE: int = int(os.getenv("KAG_BATCH_SIZE", "3"))
    # Précision de classification : seuil de confiance + plafond de catégories par chunk
    # (axes task/symptom notés par le LLM). Au-dessous du seuil → catégorie ignorée.
    CATEGORY_MIN_CONFIDENCE: float = float(os.getenv("CATEGORY_MIN_CONFIDENCE", "0.55"))
    CATEGORY_MAX_PER_CHUNK: int = int(os.getenv("CATEGORY_MAX_PER_CHUNK", "3"))
    # Carte mentale : entités croisées sous chaque catégorie (top-N co-occurrentes).
    THEME_ENTITY_MAX_PER_CATEGORY: int = int(os.getenv("THEME_ENTITY_MAX_PER_CATEGORY", "6"))
    THEME_ENTITY_MIN_COOCCURRENCE: int = int(os.getenv("THEME_ENTITY_MIN_COOCCURRENCE", "2"))
    KAG_BATCH_OVERLAP: int = int(os.getenv("KAG_BATCH_OVERLAP", "1"))

    # Enrichissement contextuel inter-pages (synthèse factuelle par thème/catégorie)
    CONTEXTUAL_ENRICHMENT_ENABLED: bool = os.getenv(
        "CONTEXTUAL_ENRICHMENT_ENABLED", "true"
    ).strip().lower() in ("true", "1", "yes", "on")
    CONTEXTUAL_ENRICHMENT_MODEL: str = os.getenv(
        "CONTEXTUAL_ENRICHMENT_MODEL", "mistral-small-latest"
    )
    CONTEXTUAL_ENRICHMENT_MAX_TOKENS: int = int(
        os.getenv("CONTEXTUAL_ENRICHMENT_MAX_TOKENS", "1500")
    )
    CONTEXTUAL_ENRICHMENT_CONCURRENCY: int = int(
        os.getenv("CONTEXTUAL_ENRICHMENT_CONCURRENCY", "2")
    )
    CONTEXTUAL_ENRICHMENT_TIMEOUT: float = float(
        os.getenv("CONTEXTUAL_ENRICHMENT_TIMEOUT", "90")
    )
    CONTEXTUAL_ENRICHMENT_BATCH_SIZE: int = int(
        os.getenv("CONTEXTUAL_ENRICHMENT_BATCH_SIZE", "3")
    )
    CONTEXTUAL_ENRICHMENT_BATCH_OVERLAP: int = int(
        os.getenv("CONTEXTUAL_ENRICHMENT_BATCH_OVERLAP", "1")
    )
    # Enrichissement multimodal sélectif : redonne les PNG des pages au LLM d'enrichissement
    # uniquement pour les batches procéduraux/visuels (où la séquence du geste est portée
    # par le schéma). Texte-seul pour le reste (commercial, garantie, normes).
    CONTEXTUAL_ENRICHMENT_MULTIMODAL_ENABLED: bool = os.getenv(
        "CONTEXTUAL_ENRICHMENT_MULTIMODAL_ENABLED", "true"
    ).strip().lower() in ("true", "1", "yes", "on")
    CONTEXTUAL_ENRICHMENT_VISUAL_CATEGORIES: list = [
        c.strip().lower()
        for c in os.getenv(
            "CONTEXTUAL_ENRICHMENT_VISUAL_CATEGORIES",
            "mounting,hardware_adjustment,glazing,drilling_constraints",
        ).split(",")
        if c.strip()
    ]

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
    RAG_CHUNK_OVERLAP_TOKENS: int = 100

    # Illustration de réponse : découpe ancrée sur la référence (code profilé) présente
    # comme texte sur la page. Voir app/services/illustration_service.py.
    # Motif des codes de référence ancrables (ex. "6104", "6105A"). 3 à 5 chiffres + lettre optionnelle.
    ILLUSTRATION_REFERENCE_PATTERN: str = os.getenv(
        "ILLUSTRATION_REFERENCE_PATTERN", r"\b\d{3,5}[A-Za-z]?\b"
    )
    # Taille max de la fenêtre de découpe en fraction de la page (largeur ET hauteur).
    ILLUSTRATION_MAX_WINDOW_RATIO: float = float(
        os.getenv("ILLUSTRATION_MAX_WINDOW_RATIO", "0.55")
    )
    # Padding ajouté autour du contenu détouré (en points PDF).
    ILLUSTRATION_ANCHOR_PADDING_PTS: float = float(
        os.getenv("ILLUSTRATION_ANCHOR_PADDING_PTS", "8.0")
    )
    # Garde-fou de lecture par vision sur le crop final (confirme que le code cible est le sujet).
    ILLUSTRATION_VISION_GATE_ENABLED: bool = (
        os.getenv("ILLUSTRATION_VISION_GATE_ENABLED", "true").lower() == "true"
    )

    # Tâches background : thread (historique), celery (Redis), hybrid (Celery + repli threads)
    TASK_BACKEND_MODE: str = "thread"
    REDIS_URL: Optional[str] = None  # ex. redis://redis:6379/0
    
    # Discord Webhook Notification
    DISCORD_WEBHOOK_URL: Optional[str] = None
    CELERY_BROKER_URL: Optional[str] = None  # défaut: REDIS_URL
    CELERY_RESULT_BACKEND: Optional[str] = None  # défaut: REDIS_URL
    # Concurrence worker Celery : 1 job document lourd à la fois (parallélisme pages via MULTIMODAL_PAGE_CONCURRENCY)
    CELERY_WORKER_CONCURRENCY: int = 1

    # LangSmith — observabilité RAG
    LANGSMITH_API_KEY: Optional[str] = None
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "noton-rag"

    # Reranker cross-encoder (CPU-only)
    RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_POOL: int = int(os.getenv("RERANK_POOL", "40"))
    RERANK_CHAR_CAP: int = 1700  # ~485 tokens (ratio FR 3.5 chars/token, marge vs max_length=512)
    RERANK_BATCH_SIZE: int = 16
    EARLY_STOP_ENABLED: bool = False  # Early stop désactivé par défaut (latence CPU acceptable)
    EARLY_STOP_TOP_N: int = 5
    EARLY_STOP_MEAN_THRESHOLD: float = 0.78
    MIN_DYNAMIC_K: int = int(os.getenv("MIN_DYNAMIC_K", "0"))
    MAX_DYNAMIC_K: int = int(os.getenv("MAX_DYNAMIC_K", "12"))
    SOFTMAX_CUM_THRESHOLD: float = 0.80
    STUTTER_GAP: float = 0.05
    ZSCORE_FLAT_THRESHOLD: float = 0.05
    RERANKER_MIN_SCORE: float = -3.0
    RAG_MIN_PERTINENCE: float = 0.75
    COLPALI_PROTECTED_SLOTS: int = int(os.getenv("COLPALI_PROTECTED_SLOTS", "2"))
    COLPALI_DOMINANCE_MIN_SCORE: float = float(os.getenv("COLPALI_DOMINANCE_MIN_SCORE", "0.55"))
    COLPALI_PGVECTOR_WEAK_THRESHOLD: float = float(os.getenv("COLPALI_PGVECTOR_WEAK_THRESHOLD", "0.45"))
    COLPALI_BM25_WEAK_THRESHOLD: float = float(os.getenv("COLPALI_BM25_WEAK_THRESHOLD", "0.25"))

    BM25_MAX_QUERY_TERMS: int = 15

    # Retrieval hybride (ColPali + pgvector + BM25)
    RETRIEVAL_EXPAND_ENABLED: bool = os.getenv("RETRIEVAL_EXPAND_ENABLED", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    RETRIEVAL_PAGE_RADIUS: int = int(os.getenv("RETRIEVAL_PAGE_RADIUS", "1"))
    RETRIEVAL_EXPAND_POOL: int = int(os.getenv("RETRIEVAL_EXPAND_POOL", "20"))
    RETRIEVAL_NEIGHBOR_MIN_SCORE_RATIO: float = float(os.getenv("RETRIEVAL_NEIGHBOR_MIN_SCORE_RATIO", "0.3"))
    GENERATION_MAX_PAGE_IMAGES: int = int(os.getenv("GENERATION_MAX_PAGE_IMAGES", "3"))

    # Retrieval multimodal page-centric (refonte RRF)
    USE_MULTIMODAL_RETRIEVAL: bool = os.getenv("USE_MULTIMODAL_RETRIEVAL", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # Exécute les 4 retrievers (ColPali/pgvector/BM25/KAG) en parallèle (threads + sessions
    # DB dédiées) au lieu de séquentiellement : recouvre l'encodage ColPali CPU avec les
    # requêtes SQL. Repli sûr : false rebascule sur l'exécution séquentielle (session unique).
    RETRIEVAL_PARALLEL_ENABLED: bool = os.getenv("RETRIEVAL_PARALLEL_ENABLED", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # Query understanding (phase intention) — chaque étape = 1 appel LLM séquentiel.
    # On peut couper les étapes optionnelles pour réduire la latence avant retrieval.
    # Multi-query (groupes) : OFF par défaut (étape récente la plus coûteuse, retourne
    # « single » dans la grande majorité des cas).
    QUERY_MULTI_GROUP_ENABLED: bool = os.getenv("QUERY_MULTI_GROUP_ENABLED", "false").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # Évaluation de vagueness (clarification si demande trop floue) : ON par défaut.
    QUERY_VAGUENESS_CHECK_ENABLED: bool = os.getenv("QUERY_VAGUENESS_CHECK_ENABLED", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # Reformulation history-aware du message de suivi en question autonome avant retrieval : ON par défaut.
    QUERY_CONDENSE_ENABLED: bool = os.getenv("QUERY_CONDENSE_ENABLED", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # Compréhension FUSIONNÉE : route + signaux + condense + vagueness + détection de
    # changement de sujet (topic_shift) en UN seul appel LLM, au lieu de 4-5 appels
    # séquentiels. Divise la latence pré-retrieval. Repli sûr : mettre à false rebascule
    # sur le graphe multi-nœuds historique (mêmes prompts, comportement inchangé).
    QUERY_FUSED_UNDERSTANDING_ENABLED: bool = os.getenv("QUERY_FUSED_UNDERSTANDING_ENABLED", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "10"))
    RAG_POOL_SIZE: int = int(os.getenv("RAG_POOL_SIZE", "20"))

    # CAG post-retriever : au lieu d'injecter des passages tronqués, on packe des DOCUMENTS
    # entiers (ou des sections étendues) dans le contexte, en exploitant la fenêtre 256k de
    # Mistral Large. Le retriever devient un sélecteur de documents → meilleur rappel.
    CAG_ENABLED: bool = os.getenv("CAG_ENABLED", "true").strip().lower() in ("true", "1", "yes", "on")
    # Budget contexte documentaire en TOKENS estimés (≈ chars / CAG_CHARS_PER_TOKEN).
    CAG_TOKEN_BUDGET: int = int(os.getenv("CAG_TOKEN_BUDGET", "100000"))
    # Nombre max de documents packés (top-D agrégés depuis les passages).
    CAG_MAX_DOCUMENTS: int = int(os.getenv("CAG_MAX_DOCUMENTS", "8"))
    # Au-delà de ce volume, un document n'est PAS chargé en entier → fenêtrage par pages.
    CAG_FULL_DOC_MAX_TOKENS: int = int(os.getenv("CAG_FULL_DOC_MAX_TOKENS", "20000"))
    # Rayon de fenêtre (pages autour des pages matchées) pour les documents trop volumineux.
    CAG_PAGE_RADIUS: int = int(os.getenv("CAG_PAGE_RADIUS", "3"))
    # Estimation FR chars→tokens pour le packing sous budget.
    CAG_CHARS_PER_TOKEN: float = float(os.getenv("CAG_CHARS_PER_TOKEN", "3.5"))
    RAG_NEIGHBOR_STRATEGY: str = os.getenv("RAG_NEIGHBOR_STRATEGY", "conditional")
    # Chars max par passage injecté au LLM. Relevé (4000 → 12000) pour laisser passer des
    # PAGES ENTIÈRES (texte consolidé + enrichissement) sans troncature, en profitant de la
    # fenêtre 256k. Diminuer si le modèle de génération a une fenêtre plus courte.
    SPACE_CONTEXT_MAX_PASSAGE_CHARS: int = int(os.getenv("SPACE_CONTEXT_MAX_PASSAGE_CHARS", "12000"))
    RAG_RENDER_ALL_IMAGES: bool = os.getenv("RAG_RENDER_ALL_IMAGES", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    RAG_MAX_IMAGES: int = int(os.getenv("RAG_MAX_IMAGES", "12"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))
    COLPALI_POST_FUSION_MIN_SCORE: float = float(os.getenv("COLPALI_POST_FUSION_MIN_SCORE", "0.25"))
    COLPALI_MIN_THRESHOLD: float = float(os.getenv("COLPALI_MIN_THRESHOLD", "0.30"))
    COLPALI_RELATIVE_MARGIN: float = float(os.getenv("COLPALI_RELATIVE_MARGIN", "0.10"))
    BM25_USE_WEBSEARCH_QUERY: bool = os.getenv("BM25_USE_WEBSEARCH_QUERY", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    BM25_FILTER_SEMANTIC_LEAF: bool = os.getenv("BM25_FILTER_SEMANTIC_LEAF", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )

    # Query understanding — extraction légère LangGraph avant retrieval RAG
    QUERY_UNDERSTANDING_ENABLED: bool = os.getenv("QUERY_UNDERSTANDING_ENABLED", "false").strip().lower() in (
        "true", "1", "yes", "on"
    )

    # Boosts retrieval (signaux query understanding)
    # Catégories : appliquées avant fusion RRF ; source/matériau/entités : post-retrieval
    RETRIEVAL_CATEGORY_BOOST: float = float(os.getenv("RETRIEVAL_CATEGORY_BOOST", "0.15"))
    # Plafond du facteur multiplicatif du boost catégorie. Sans reranker cross-encoder pour
    # rattraper, le boost est le principal signal post-fusion : on borne son amplification
    # pour qu'une page mal classée (3 matches symptôme) ne puisse pas écraser un vrai signal
    # de pertinence (facteur brut ~1.9 → plafonné à 1.5 par défaut).
    RETRIEVAL_CATEGORY_BOOST_MAX: float = float(os.getenv("RETRIEVAL_CATEGORY_BOOST_MAX", "1.5"))
    # Pondération du boost catégorie par axe (un slug symptôme pèse plus qu'un doc_type).
    # JSON optionnel via env RETRIEVAL_AXIS_BOOST_WEIGHTS ; défaut sinon. Axe absent → poids 1.0.
    RETRIEVAL_AXIS_BOOST_WEIGHTS: dict = (
        __import__("json").loads(os.getenv("RETRIEVAL_AXIS_BOOST_WEIGHTS"))
        if os.getenv("RETRIEVAL_AXIS_BOOST_WEIGHTS")
        else {"symptom": 2.0, "task": 1.0, "doc_type": 0.6, "lifecycle_phase": 0.5}
    )
    RETRIEVAL_SOURCE_BOOST_MAX: float = float(os.getenv("RETRIEVAL_SOURCE_BOOST_MAX", "0.8"))
    # Ancrage documentaire conversationnel : les documents fortement matchés à un tour sont
    # mémorisés (query_context.current_documents) et leurs pages sont boostées aux tours
    # SUIVANTS — tant qu'il n'y a pas de changement de sujet (topic_shift). Empêche la
    # conversation de sauter d'un produit à l'autre (ex. KSR PVC → Lumine65 → INNOSLIDE).
    CONVERSATION_ANCHOR_ENABLED: bool = os.getenv("CONVERSATION_ANCHOR_ENABLED", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # Nombre de documents mémorisés comme ancre (le sujet courant tient en général sur 1-3 docs).
    CONVERSATION_ANCHOR_MAX_DOCS: int = int(os.getenv("CONVERSATION_ANCHOR_MAX_DOCS", "3"))
    # Boost multiplicatif du rrf_score des pages appartenant aux documents ancrés (post-fusion,
    # AVANT la coupe top_k → une page d'un doc ancré survit à la coupe).
    CONVERSATION_ANCHOR_BOOST: float = float(os.getenv("CONVERSATION_ANCHOR_BOOST", "0.5"))
    RETRIEVAL_MATERIAL_BOOST: float = float(os.getenv("RETRIEVAL_MATERIAL_BOOST", "0.3"))
    RETRIEVAL_ENTITY_BOOST: float = float(os.getenv("RETRIEVAL_ENTITY_BOOST", "0.1"))

    # Reranker vision LLM (juge de pertinence page-par-page sur les PNG ColPali)
    VISION_RERANK_ENABLED: bool = os.getenv("VISION_RERANK_ENABLED", "true").strip().lower() in ('true', '1', 'yes', 'on')
    VISION_RERANK_MODEL: str = os.getenv("VISION_RERANK_MODEL", "mistral-small-latest")
    VISION_RERANK_MIN_SCORE: float = float(os.getenv("VISION_RERANK_MIN_SCORE", "3"))  # sur une echelle 0-5
    VISION_RERANK_MAX_PAGES: int = int(os.getenv("VISION_RERANK_MAX_PAGES", "4"))
    VISION_RERANK_DPI: int = int(os.getenv("VISION_RERANK_DPI", "150"))

    # Fiche technique — lookup par référence nue ("Profil 76180", "notice seuil 76180")
    # → sortie STRUCTURÉE et sourcée (pas de génération libre / broderie).
    # Désactivé par défaut : aucun impact sur le pipeline existant tant que False.
    FICHE_TECHNIQUE_ENABLED: bool = os.getenv("FICHE_TECHNIQUE_ENABLED", "false").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # Nombre de passages récupérés pour construire la fiche (lookup, pas top-k sémantique).
    FICHE_TECHNIQUE_K: int = int(os.getenv("FICHE_TECHNIQUE_K", "12"))
    # Motif des références numériques (profilés, seuils) — codes 3 à 6 chiffres.
    FICHE_REFERENCE_PATTERN: str = os.getenv("FICHE_REFERENCE_PATTERN", r"\b\d{3,6}[A-Za-z]?\b")
    # Motif des références alphanumériques (visserie S055, gabarit T021, seuil A076).
    FICHE_REFERENCE_ALNUM_PATTERN: str = os.getenv(
        "FICHE_REFERENCE_ALNUM_PATTERN", r"\b[A-Z]{1,3}\d{2,4}\b"
    )
    # Au-delà de ce nombre de mots, une requête n'est plus considérée « référence nue »
    # (sauf si elle contient un marqueur documentaire explicite : fiche, notice, réf…).
    FICHE_MAX_WORDS: int = int(os.getenv("FICHE_MAX_WORDS", "10"))
    # Budget de tokens pour l'extraction structurée (le schéma est riche : éviter la
    # troncature → JSON invalide → fiche vide).
    FICHE_MAX_TOKENS: int = int(os.getenv("FICHE_MAX_TOKENS", "3000"))

    # Guidage procédural ("aiguillage" SAV / chantier) — moteur multi-étapes
    # Désactivé par défaut : aucun impact sur le pipeline one-shot existant tant que False.
    GUIDED_FLOW_ENABLED: bool = os.getenv("GUIDED_FLOW_ENABLED", "false").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # Nombre maximal d'étapes avant escalade automatique vers le SAV
    GUIDED_MAX_STEPS: int = int(os.getenv("GUIDED_MAX_STEPS", "8"))
    # Passages récupérés par étape (plus focalisé que RAG_TOP_K)
    GUIDED_RETRIEVAL_K: int = int(os.getenv("GUIDED_RETRIEVAL_K", "8"))
    # Arbres validés (capitalisation) prioritaires sur la génération dynamique (Phase 2)
    GUIDED_AUTHORED_TREES_ENABLED: bool = os.getenv(
        "GUIDED_AUTHORED_TREES_ENABLED", "true"
    ).strip().lower() in ("true", "1", "yes", "on")
    # Filtre catégorie strict (dur) vs boost souple (Phase 2)
    GUIDED_CATEGORY_FILTER_STRICT: bool = os.getenv(
        "GUIDED_CATEGORY_FILTER_STRICT", "false"
    ).strip().lower() in ("true", "1", "yes", "on")
    # Bornes du nombre de choix proposés à chaque aiguillage
    GUIDED_MIN_CHOICES: int = int(os.getenv("GUIDED_MIN_CHOICES", "2"))
    GUIDED_MAX_CHOICES: int = int(os.getenv("GUIDED_MAX_CHOICES", "5"))
    # Contact SAV affiché dans le récapitulatif d'escalade (fallback si absent des métadonnées)
    GUIDED_SAV_CONTACT: str = os.getenv(
        "GUIDED_SAV_CONTACT", "Service SAV PROFERM — contactez votre interlocuteur habituel."
    )

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
            return True
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ('true', '1', 'yes', 'on')
        return False

    @field_validator('COLPALI_ENABLED', mode='before')
    @classmethod
    def parse_colpali_enabled(cls, v: Union[str, bool, None]) -> bool:
        """Convertit les chaînes en bool pour COLPALI_ENABLED."""
        if v is None:
            return True
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ('true', '1', 'yes', 'on')
        return False

    @field_validator('RERANKER_ENABLED', 'EARLY_STOP_ENABLED', mode='before')
    @classmethod
    def parse_bool_flags(cls, v: Union[str, bool, None]) -> bool:
        """Convertit les chaînes en bool pour les flags Reranker/Early stop."""
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

