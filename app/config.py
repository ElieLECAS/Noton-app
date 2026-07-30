from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator
from typing import Optional, Union, List
from pathlib import Path
import json
import logging
import os

logger = logging.getLogger(__name__)

# Défauts du budget de packing CAG par intent (utilisés si CAG_BUDGET_BY_INTENT est vide).
# Au niveau module (pas attribut de Settings) pour éviter toute interférence avec la
# gestion des champs pydantic.
_CAG_BUDGET_DEFAULTS = {
    "specification": {"budget": 30000, "max_documents": 4},
    "documentation": {"budget": 30000, "max_documents": 4},
    "installation": {"budget": 60000, "max_documents": 6},
    "regulatory": {"budget": 60000, "max_documents": 6},
    "product_selection": {"budget": 100000, "max_documents": 8},
    "troubleshooting": {"budget": 100000, "max_documents": 8},
    "default": {"budget": 60000, "max_documents": 6},
}


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
    # Cookie d'auth : Secure par défaut (prod HTTPS derrière nginx). Mettre à false
    # UNIQUEMENT en dev local sur http:// (sinon le navigateur refuse le cookie).
    AUTH_COOKIE_SECURE: bool = os.getenv("AUTH_COOKIE_SECURE", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    AUTH_COOKIE_SAMESITE: str = os.getenv("AUTH_COOKIE_SAMESITE", "lax")
    
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
    # Domaine technique factuel (références, cotes, procédures de pose) : température basse
    # pour coller au contexte fourni plutôt que reformuler librement.
    SPACE_CHAT_TEMPERATURE: float = float(os.getenv("SPACE_CHAT_TEMPERATURE", "0.2"))
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
    # v1.0 : aligne sur l'index LanceDB de prod (le modele de requete DOIT etre
    # celui qui a encode les patches, sinon MaxSim incoherent). Etait v0.1.
    COLPALI_MODEL_NAME: str = "vidore/colqwen2-v1.0"
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

    # --- Extraction TEXTE NATIF (pymupdf4llm) : voie alternative à la vision ---
    # Sur une page dotée d'une couche texte, les chiffres et références sont lus dans
    # les objets texte du PDF (exacts par construction, aucune transcription par un
    # modèle) et les tableaux sortent entiers (pas de plafond de tokens de sortie LLM).
    # Sélectionnable par requête de réindexation (extractor=text) pour comparer les
    # deux voies sur un même document.
    #
    # Cap de tokens des chunks de PROSE. Ce n'est PAS le critère de découpe — la
    # SECTION l'est (un titre markdown et son corps restent dans le même chunk) :
    # ce cap n'intervient que pour scinder une section anormalement longue, en
    # réinjectant son titre dans chaque morceau. 0 = aucun cap.
    # 500 et non 900 : au-delà, les chunks deviennent illisibles dans le monitoring
    # et diluent le vecteur. Le recollage inter-pages tolère jusqu'à 1,6× ce cap.
    TEXT_EXTRACTION_MAX_CHUNK_TOKENS: int = int(
        os.getenv("TEXT_EXTRACTION_MAX_CHUNK_TOKENS", "500")
    )
    # "rows" : une ligne de tableau = un chunk autosuffisant (en-têtes de colonnes
    # réinjectés) → une question sur une référence précise matche UNE ligne au lieu
    # d'être diluée. "atomic" : le tableau entier en un seul chunk.
    TEXT_EXTRACTION_TABLE_MODE: str = os.getenv(
        "TEXT_EXTRACTION_TABLE_MODE", "rows"
    ).strip().lower()
    # En mode "rows", ajoute un chunk « tableau complet » en plus des lignes (contexte
    # d'ensemble : combien de lignes, quelles colonnes).
    TEXT_EXTRACTION_TABLE_FULL_CHUNK: bool = os.getenv(
        "TEXT_EXTRACTION_TABLE_FULL_CHUNK", "true"
    ).strip().lower() in ("true", "1", "yes", "on")
    # Plafond de chunks par page. Bien plus haut que les 12 de la voie vision : un
    # tableau de 30 lignes produit légitimement 31 chunks. Tout dépassement est
    # JOURNALISÉ en warning (la voie vision, elle, tronquait en silence). 0 = illimité.
    TEXT_EXTRACTION_MAX_CHUNKS_PER_PAGE: int = int(
        os.getenv("TEXT_EXTRACTION_MAX_CHUNKS_PER_PAGE", "120")
    )
    # Pages sans couche texte exploitable : bascule automatique sur la voie vision
    # (une page scannée ne produit rien en texte natif). false = ces pages restent vides.
    TEXT_EXTRACTION_VISION_FALLBACK: bool = os.getenv(
        "TEXT_EXTRACTION_VISION_FALLBACK", "true"
    ).strip().lower() in ("true", "1", "yes", "on")

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
    # Motif des codes de référence ancrables. Couvre :
    #   - alphanumérique fournisseur : TGY3702, TMX13, TFZ60032, T910002 ([A-Z]{1,4} + chiffres)
    #   - numérique pur ou décimal : 6104, 6105A, 259879, 9718.3
    # (l'ancien r"\b\d{3,5}[A-Za-z]?\b" ratait TOUTES les réfs alphanum + le 6 chiffres.)
    ILLUSTRATION_REFERENCE_PATTERN: str = os.getenv(
        "ILLUSTRATION_REFERENCE_PATTERN",
        r"\b(?:[A-Z]{1,4}\d{2,6}[A-Za-z]?|\d{3,6}(?:\.\d{1,2})?[A-Za-z]?)\b",
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
    # Fournisseur de reranking : "local" (cross-encoder HF) ou "mistral" (API).
    # Champ ajouté 2026-07-20 : auparavant lu via getattr → toujours "local" en silence.
    RERANKER_PROVIDER: str = os.getenv("RERANKER_PROVIDER", "local")
    # Corpus francais → cross-encoder francais par defaut (etait ms-marco anglais).
    RERANKER_MODEL: str = "antoinelouis/crossencoder-camembert-L2-mmarcoFR"
    RERANK_POOL: int = int(os.getenv("RERANK_POOL", "40"))
    RERANK_CHAR_CAP: int = 1800  # ~510 tokens FR : aligne sur max_length=512 du cross-encoder
    RERANK_BATCH_SIZE: int = 16
    EARLY_STOP_ENABLED: bool = False  # Early stop désactivé par défaut (latence CPU acceptable)
    EARLY_STOP_TOP_N: int = 5
    EARLY_STOP_MEAN_THRESHOLD: float = 0.78
    MIN_DYNAMIC_K: int = int(os.getenv("MIN_DYNAMIC_K", "0"))
    MAX_DYNAMIC_K: int = int(os.getenv("MAX_DYNAMIC_K", "12"))
    SOFTMAX_CUM_THRESHOLD: float = 0.80
    STUTTER_GAP: float = 0.05
    ZSCORE_FLAT_THRESHOLD: float = 0.05
    # Plancher de confiance absolue (échelle pertinence sigmoïde, comme RAG_MIN_PERTINENCE) :
    # au-dessus de ce score, le top-1 n'a PAS besoin de se démarquer du top-2 pour qu'on lui
    # fasse confiance. Sans ce plancher, deux passages EXCELLENTS et proches (ex. 0.91/0.90,
    # gauche/droite ou deux notices qui disent la même chose) déclenchaient le garde-fou
    # « bégaiement » comme s'il s'agissait de deux passages MÉDIOCRES et proches → abstention
    # à tort. Régression constatée le 20/07/2026 après réactivation de STUTTER_GAP=0.05.
    STUTTER_HIGH_CONFIDENCE_FLOOR: float = float(
        os.getenv("STUTTER_HIGH_CONFIDENCE_FLOOR", "0.85")
    )
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
    # Quota SOUPLE de pages par document dans le top-K final (fraction de top_k). Sans lui,
    # un catalogue de 200 pages peut occuper tous les slots et fabriquer son score
    # d'élection par le volume, évinçant la notice qui contient LA bonne page. « Souple » :
    # les slots restés vides sont rendus aux hits écartés — le quota ne mord qu'en
    # compétition. 0 = désactivé (coupe brute historique).
    RETRIEVAL_PER_DOC_QUOTA_RATIO: float = float(os.getenv("RETRIEVAL_PER_DOC_QUOTA_RATIO", "0.4"))

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
    # Évaluation de vagueness PRÉ-retrieval (bloque toute recherche documentaire si le LLM
    # juge la demande "trop vague", sur la base du seul message — sans avoir vu un document).
    # OFF par défaut : ce garde-fou se déclenchait sur des questions techniques légitimes
    # mais courtes/à sigles métier (ex. "comment transformer un OF en OB ?") et empêchait
    # tout retrieval. La clarification pertinente est désormais portée par la politique de
    # réponse du prompt système (SPACE_CHAT_SYSTEM_PROMPT, §2) : elle intervient APRÈS
    # retrieval, informée par les documents réellement trouvés — plus fiable qu'une
    # estimation à l'aveugle avant recherche. Remettre à true rétablit le blocage pré-retrieval.
    QUERY_VAGUENESS_CHECK_ENABLED: bool = os.getenv("QUERY_VAGUENESS_CHECK_ENABLED", "false").strip().lower() in (
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
    # Génération des requêtes retriever (colpali/semantic/lexical) par un appel LLM dédié
    # APRÈS la compréhension. OFF par défaut : en mode fusionné, la question autonome +
    # les entités/références extraites suffisent à construire les 3 requêtes de façon
    # DÉTERMINISTE (0 appel LLM) → on tombe à UN SEUL appel LLM avant le retrieval. Mettre
    # à true rebranche la génération LLM par groupe (utile surtout avec le multi-groupe).
    QUERY_GENERATE_QUERIES_LLM: bool = os.getenv("QUERY_GENERATE_QUERIES_LLM", "false").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # Budget temps (secondes) d'un appel LLM de COMPRÉHENSION de requête (fused, condense,
    # signaux…). Dépassé → on abandonne l'appel et on retombe sur les valeurs de repli
    # (jamais de blocage indéfini avant le retrieval). 0 = pas de plafond applicatif.
    QUERY_UNDERSTANDING_TIMEOUT_S: float = float(os.getenv("QUERY_UNDERSTANDING_TIMEOUT_S", "25"))
    # Budget temps (secondes) du RETRIEVAL complet (4 canaux + fusion + rerank). Dépassé →
    # dégradation gracieuse (0 passage, statut degraded_timeout) au lieu d'un blocage
    # indéfini si un canal freeze (ColPali CPU, MaxSim LanceDB). Généreux par défaut pour
    # ne jamais couper un ColPali légitime ; 0 = pas de plafond applicatif.
    RETRIEVAL_TIMEOUT_S: float = float(os.getenv("RETRIEVAL_TIMEOUT_S", "90"))
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "10"))
    RAG_POOL_SIZE: int = int(os.getenv("RAG_POOL_SIZE", "20"))

    # ColPali gating : ColPali est un retriever VISUEL très coûteux sur CPU (encode
    # ColQwen2 + MaxSim sur des centaines de milliers de patches → ~30s/requête). On ne
    # le lance donc QUE lorsque la requête en a besoin (marqueurs visuels : schéma, plan,
    # coupe, « où se trouve »…). Pour les requêtes texte, BM25 + pgvector suffisent. Un
    # FILET DE SÉCURITÉ relance ColPali si les retrievers texte reviennent trop faibles,
    # de sorte qu'aucun rappel n'est perdu en silence. Mettre à false rebascule sur
    # l'exécution systématique de ColPali (comportement historique).
    # Defaut false (2026-07-20) : ColPali est le meilleur retriever et indexe TOUTES
    # les pages (meme sans texte) — on le veut toujours actif. true = ancien gating par mots.
    COLPALI_GATING_ENABLED: bool = os.getenv("COLPALI_GATING_ENABLED", "false").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # Sous ce nombre de pages texte (pgvector ∪ BM25), ColPali est relancé en rattrapage
    # même si le gate l'avait écarté.
    COLPALI_GATING_FALLBACK_MIN_HITS: int = int(os.getenv("COLPALI_GATING_FALLBACK_MIN_HITS", "5"))
    # Intents qui FORCENT ColPali (au-delà des marqueurs visuels du texte). « installation »
    # par défaut : la pose/montage s'appuie fortement sur les schémas, cœur de métier — on
    # ne veut pas y perdre le visuel. Ajuster (CSV) sans redéploiement pour élargir/réduire
    # le périmètre ColPali (ex. "installation,specification" ou "" pour un gating agressif).
    # Typé str (CSV brut), PAS List : pydantic-settings tenterait sinon un json.loads() sur
    # la valeur d'env (« installation » n'est pas du JSON) et ferait échouer le démarrage.
    # La liste normalisée est exposée via la propriété colpali_gating_intents.
    COLPALI_GATING_INTENTS: str = os.getenv("COLPALI_GATING_INTENTS", "installation")

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
    # Plafond de tokens de RÉPONSE en mode CAG (réponses procédurales complètes ; le
    # plancher 2048 historique coupait les procédures longues).
    CAG_MAX_COMPLETION_TOKENS: int = int(os.getenv("CAG_MAX_COMPLETION_TOKENS", "3072"))
    # --- Reasoning natif de la génération (mistral-small : "high" | "none") ---
    # "high" = le modèle produit un ThinkChunk (jugement) avant la réponse ; "none" =
    # génération directe. Variable de bascule pour comparer les réponses (défaut high).
    # Sur Mistral, reasoning_effort est BINAIRE (pas de niveau intermédiaire).
    GENERATION_REASONING_EFFORT: str = os.getenv("GENERATION_REASONING_EFFORT", "high").strip().lower()
    # Le thinking consomme le budget de complétion : plancher relevé quand reasoning=high
    # (sinon la réponse est tronquée après le raisonnement).
    GENERATION_REASONING_MAX_TOKENS: int = int(os.getenv("GENERATION_REASONING_MAX_TOKENS", "8192"))
    # Images PNG jointes à la génération en mode CAG : UNIQUEMENT des pages réellement
    # packées dans le contexte (alignement texte/visuel), plafonnées à ce nombre.
    CAG_MAX_IMAGES: int = int(os.getenv("CAG_MAX_IMAGES", "8"))
    # Budget de packing PAR INTENT (tokens + max documents) : une question de spécification
    # ponctuelle ne paie pas le prefill d'un diagnostic SAV. JSON optionnel via env
    # CAG_BUDGET_BY_INTENT ({"installation": {"budget": 60000, "max_documents": 6}, ...}) ;
    # la clé "default" couvre les intents absents.
    # Typé str (JSON brut), PAS dict : pydantic-settings auto-parse tout champ typé dict
    # depuis l'env → un CAG_BUDGET_BY_INTENT vide (cf. docker-compose `:-`) ferait planter
    # le démarrage sur json.loads(""). La table normalisée est exposée via la propriété
    # cag_budget_by_intent (même pattern que colpali_gating_intents).
    CAG_BUDGET_BY_INTENT: str = os.getenv("CAG_BUDGET_BY_INTENT", "")
    # Cache TTL (secondes) du texte des chunks feuilles par document, pour éviter de
    # recharger/concaténer ~60 chunks SQL à chaque requête. 0 = désactivé. Un réindex
    # peut donc mettre jusqu'à TTL secondes à se refléter dans le contexte de génération.
    CAG_FULLTEXT_CACHE_TTL: int = int(os.getenv("CAG_FULLTEXT_CACHE_TTL", "300"))
    # Budget de packing du contexte de SECOURS (fallback Mistral 400) : quand la génération
    # échoue en 400 — souvent parce que le contexte est trop gros — on RE-PACKE le CAG à ce
    # petit budget au lieu de rejouer le contexte massif à l'identique (P0.2).
    CAG_ECO_TOKEN_BUDGET: int = int(os.getenv("CAG_ECO_TOKEN_BUDGET", "20000"))
    CAG_ECO_MAX_DOCUMENTS: int = int(os.getenv("CAG_ECO_MAX_DOCUMENTS", "3"))
    # --- Élection des documents à packer (aggregate_documents) ---
    # "best_passage" : le document est élu par sa MEILLEURE page, avec des bonus
    # multiplicatifs BORNÉS (familles de canaux, nombre de pages) qui départagent sans
    # jamais renverser un meilleur passage net. "legacy" : ancienne formule
    # score_max + 0.2·(somme des autres pages), où le volume écrase le max.
    CAG_ELECTION_MODE: str = os.getenv("CAG_ELECTION_MODE", "best_passage").strip().lower()
    # Bonus par FAMILLE de canaux supplémentaire. Les familles (texte = pgvector+bm25,
    # visuel = colpali, graphe = kag) évitent le double crédit lexical : pgvector et BM25
    # lisent la même évidence textuelle et ne doivent pas compter pour deux confirmations.
    CAG_ELECTION_FAMILY_BONUS: float = float(os.getenv("CAG_ELECTION_FAMILY_BONUS", "0.10"))
    # Bonus par page matchée supplémentaire, plafonné à CAG_ELECTION_PAGE_CAP pages.
    CAG_ELECTION_PAGE_BONUS: float = float(os.getenv("CAG_ELECTION_PAGE_BONUS", "0.05"))
    CAG_ELECTION_PAGE_CAP: int = int(os.getenv("CAG_ELECTION_PAGE_CAP", "2"))
    # --- Partage du budget de packing entre les documents élus ---
    # Parts par RANG d'élection, renormalisées sur le nombre de documents réellement élus
    # (2 docs → 62,5 %/37,5 %). Le budget non consommé par un document est reporté au
    # suivant. Sans ce partage, le document n°1 peut avaler tout le budget et ne laisser
    # que des miettes aux suivants. Vide = comportement historique (premier servi).
    CAG_DOC_BUDGET_SHARES: str = os.getenv("CAG_DOC_BUDGET_SHARES", "0.5,0.3,0.2")
    # --- Fenêtrage des documents trop volumineux ---
    # "greedy" : les pages matchées deviennent des SEEDS scorées, leurs voisines héritent
    # d'une valeur décroissante, et le budget du document est rempli par valeur
    # décroissante — sous budget serré ce sont les pages les plus FAIBLES qui sautent, et
    # la page qui a gagné le vote n'est jamais rognée. "radius" : ancien ±N arithmétique
    # avec rognage par distance (aveugle à la valeur).
    CAG_WINDOW_MODE: str = os.getenv("CAG_WINDOW_MODE", "greedy").strip().lower()
    # Nombre max de pages matchées promues en seeds par document : évite la « fenêtre
    # pieuvre » d'un document matché partout, qui diluerait le budget en 10 fenêtres.
    CAG_MAX_SEEDS_PER_DOC: int = int(os.getenv("CAG_MAX_SEEDS_PER_DOC", "3"))
    # Décroissance de valeur par page d'écart avec la seed la plus proche.
    CAG_NEIGHBOR_DECAY: float = float(os.getenv("CAG_NEIGHBOR_DECAY", "0.7"))
    # Marque les pages retrouvées par la recherche dans le contexte packé : sans ce
    # marquage, le classement du retriever s'évapore au packing et le modèle reçoit N pages
    # indifférenciées sans savoir où la recherche a trouvé.
    CAG_MARK_MATCHED_PAGES: bool = os.getenv("CAG_MARK_MATCHED_PAGES", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # Nombre de documents ANCRÉS (sujet courant) packés de force, même absents du retrieval
    # de ce tour. Plafonné à 1 : avec 3 ancres et CAG_MAX_DOCUMENTS=3, les documents
    # (souvent faux) d'un tour raté consommaient TOUS les slots et le bon document trouvé
    # au tour suivant n'avait plus de place. 0 = aucune garantie d'ancrage.
    CAG_ANCHOR_SLOTS: int = int(os.getenv("CAG_ANCHOR_SLOTS", "1"))
    # Garantir la PRÉSENCE d'un document ancré n'est pas lui garantir la PRIORITÉ : à true
    # l'ancre est packée (jamais tronquée par max_documents) mais classée à son score
    # d'élection réel, donc elle ne capte plus d'office la part de budget du rang 1.
    CAG_ANCHOR_RANK_BY_SCORE: bool = os.getenv("CAG_ANCHOR_RANK_BY_SCORE", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # --- Vérification post-génération (juge LLM) ---
    # Plafond de l'extrait de contexte envoyé au juge. 20 000 en dur historiquement, sur un
    # texte qui COMMENÇAIT par le prompt système : le juge ne voyait que 8 à 30 % du
    # contexte et déclarait « non documenté » ce qu'il n'avait simplement pas lu.
    # 0 = illimité (contexte complet, coût token proportionnel).
    VERIFICATION_CONTEXT_MAX_CHARS: int = int(os.getenv("VERIFICATION_CONTEXT_MAX_CHARS", "60000"))
    # Manifeste de TOUS les documents packés en tête de l'extrait, jamais tronqué : rend
    # structurellement impossible le « le contexte ne parle que de la gamme X » alors qu'un
    # document de la gamme Y était packé plus loin.
    VERIFICATION_INCLUDE_MANIFEST: bool = os.getenv(
        "VERIFICATION_INCLUDE_MANIFEST", "true"
    ).strip().lower() in ("true", "1", "yes", "on")
    # --- Boucle agentique de recherche (plan_boucle_agentique_2026-07-29, lots B0-B9) ---
    # Flag maître du juge pré-génération + relances (E4/E5). false = pipeline strictement
    # identique à l'existant, zéro appel LLM supplémentaire avant génération.
    AGENTIC_LOOP_ENABLED: bool = os.getenv("AGENTIC_LOOP_ENABLED", "false").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # "shadow" : le juge tourne et TRACE ses verdicts sans jamais agir (calibrage prod) ;
    # "active" : ses verdicts pilotent l'élection, les relances et la note de recherche.
    AGENTIC_LOOP_MODE: str = os.getenv("AGENTIC_LOOP_MODE", "shadow").strip().lower()
    # Deadline dure de la boucle (retrievals + juges cumulés) : au-delà on génère avec ce
    # qu'on a, en aveu structuré si le dernier verdict était « insuffisant ».
    LOOP_DEADLINE_S: float = float(os.getenv("LOOP_DEADLINE_S", "60"))
    # Modèle du juge (E4) et de la vérification (E7). DISTINCT du modèle de génération :
    # un modèle qui se relit se blanchit (cas TGY3710 du 29/07).
    MODEL_JUDGE: str = os.getenv("MODEL_JUDGE", "mistral-small-latest")
    # Relances de recherche max décidées par le juge (chacune = 1 retrieval + 1 juge).
    JUDGE_MAX_RETRIES: int = int(os.getenv("JUDGE_MAX_RETRIES", "2"))
    # Timeout d'un appel juge ; au-delà : verdict « unknown », on génère comme aujourd'hui.
    JUDGE_TIMEOUT_S: float = float(os.getenv("JUDGE_TIMEOUT_S", "20"))
    # Le pack-juge compare PLUS LARGE que le pack-génération : documents candidats montrés
    # au juge (le pack-génération reste borné par CAG_MAX_DOCUMENTS/l'élection).
    JUDGE_MAX_CANDIDATE_DOCS: int = int(os.getenv("JUDGE_MAX_CANDIDATE_DOCS", "5"))
    # Plafond de caractères du pack-juge (rempli par les preuves, manifeste toujours inclus).
    JUDGE_CONTEXT_MAX_CHARS: int = int(os.getenv("JUDGE_CONTEXT_MAX_CHARS", "24000"))
    # Images de pages jointes au juge : off | auto (intent visuel : installation/SAV) | always.
    JUDGE_IMAGES_MODE: str = os.getenv("JUDGE_IMAGES_MODE", "auto").strip().lower()
    JUDGE_MAX_IMAGES: int = int(os.getenv("JUDGE_MAX_IMAGES", "4"))
    # Confiance minimale pour qu'un verdict soit actionnable (sinon traité comme unknown).
    JUDGE_MIN_CONFIDENCE: float = float(os.getenv("JUDGE_MIN_CONFIDENCE", "0.5"))
    # --- Vérification post-génération : flag maître + gate (B7) ---
    # false = aucune vérification (historiquement elle tournait TOUJOURS, sans effet).
    VERIFY_ENABLED: bool = os.getenv("VERIFY_ENABLED", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # Modèle de la vérification ; vide = MODEL_JUDGE. Jamais le modèle de génération.
    VERIFY_MODEL: str = os.getenv("VERIFY_MODEL", "").strip()
    # true = la vérification BLOQUE l'émission (génération en tampon, réparation possible) ;
    # false = comportement historique (post-hoc, trace seulement).
    VERIFY_BLOCKING: bool = os.getenv("VERIFY_BLOCKING", "false").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # buffer : générer en tampon, vérifier, puis émettre (v1). stream_correct : réservé
    # (streamer puis corriger) — non implémenté, retombe sur buffer.
    VERIFY_EMIT_MODE: str = os.getenv("VERIFY_EMIT_MODE", "buffer").strip().lower()
    # Réparations max après un échec de vérification en mode bloquant.
    VERIFY_MAX_REPAIRS: int = int(os.getenv("VERIFY_MAX_REPAIRS", "1"))
    # Contrôle programmatique des CODES PRODUITS de la réponse (présence littérale dans le
    # contexte packé) : attrape les références inventées (TGY3710) sans aucun LLM.
    VERIFY_CODE_GROUNDING: bool = os.getenv("VERIFY_CODE_GROUNDING", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    # Quota par document appliqué AUSSI au chemin reranké (B2) : la coupe dynamique choisit
    # combien de pages, le quota choisit lesquelles — sans lui, 17 passages sur 19 pouvaient
    # venir du même document et le juge n'avait rien à comparer.
    RERANK_PER_DOC_QUOTA_ENABLED: bool = os.getenv(
        "RERANK_PER_DOC_QUOTA_ENABLED", "true"
    ).strip().lower() in ("true", "1", "yes", "on")
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
    # 0.18 : aligné sur docker-compose/.env (une seule valeur de vérité désormais).
    COLPALI_RELATIVE_MARGIN: float = float(os.getenv("COLPALI_RELATIVE_MARGIN", "0.18"))
    BM25_USE_WEBSEARCH_QUERY: bool = os.getenv("BM25_USE_WEBSEARCH_QUERY", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )
    BM25_FILTER_SEMANTIC_LEAF: bool = os.getenv("BM25_FILTER_SEMANTIC_LEAF", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )

    # Query understanding — extraction légère LangGraph avant retrieval RAG.
    # ON par défaut : c'est la phase qui produit signals (catégories inférées, source,
    # matériau, intent), topic_shift, l'ancrage conversationnel et le gating ColPali par
    # intent. La désactiver rend INERTES les boosts catégorie/source/matériau, l'ancrage
    # et le gating par intent (voir coherence_warnings()).
    QUERY_UNDERSTANDING_ENABLED: bool = os.getenv("QUERY_UNDERSTANDING_ENABLED", "true").strip().lower() in (
        "true", "1", "yes", "on"
    )

    # Boosts retrieval (signaux query understanding)
    # Catégories : appliquées avant fusion RRF ; source/matériau/entités : post-retrieval
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
    # Boost multiplicatif des pages du document ancré. Ramené de 0.5 à 0.15 (2026-07-27) :
    # à ×1.5 l'ancre accordait GRATUITEMENT, à chacune de ses pages, le boost catégorie
    # MAXIMAL (plafonné à 1.5 et qui, lui, exige des correspondances fortes) — elle
    # annulait donc purement et simplement le signal de pertinence. Cas mesuré : un tour
    # « comment l'installer ? » packait le catalogue conception du tour précédent au lieu
    # du catalogue fabrication qui contenait la procédure.
    CONVERSATION_ANCHOR_BOOST: float = float(os.getenv("CONVERSATION_ANCHOR_BOOST", "0.15"))
    # Nombre de pages boostées par document ancré (les mieux classées). Le but de l'ancre
    # est qu'une page pertinente SURVIVE À LA COUPE, pas qu'un document entier s'installe
    # en tête du classement. 0 = toutes les pages (comportement historique).
    CONVERSATION_ANCHOR_BOOST_MAX_PAGES: int = int(
        os.getenv("CONVERSATION_ANCHOR_BOOST_MAX_PAGES", "3")
    )
    # L'ancre perd sa garantie de packing quand l'INTENTION du tour change (ex.
    # product_selection → installation) : le sujet reste le même mais le bon TYPE de
    # document change (références → procédure de montage). Le boost léger, lui, subsiste.
    CONVERSATION_ANCHOR_INTENT_GUARD: bool = os.getenv(
        "CONVERSATION_ANCHOR_INTENT_GUARD", "true"
    ).strip().lower() in ("true", "1", "yes", "on")
    RETRIEVAL_MATERIAL_BOOST: float = float(os.getenv("RETRIEVAL_MATERIAL_BOOST", "0.3"))

    # Reranker vision LLM (juge de pertinence page-par-page sur les PNG ColPali).
    # OFF par défaut : appels LLM vision coûteux dans le chemin critique du retrieval.
    VISION_RERANK_ENABLED: bool = os.getenv("VISION_RERANK_ENABLED", "false").strip().lower() in ('true', '1', 'yes', 'on')
    VISION_RERANK_MODEL: str = os.getenv("VISION_RERANK_MODEL", "mistral-small-latest")
    VISION_RERANK_MIN_SCORE: float = float(os.getenv("VISION_RERANK_MIN_SCORE", "3"))  # sur une echelle 0-5
    VISION_RERANK_MAX_PAGES: int = int(os.getenv("VISION_RERANK_MAX_PAGES", "4"))
    VISION_RERANK_DPI: int = int(os.getenv("VISION_RERANK_DPI", "150"))

    # Fiche technique — lookup par référence nue ("Profil 76180", "notice seuil 76180")
    # → sortie STRUCTURÉE et sourcée (pas de génération libre / broderie).
    # ON par défaut (aligné prod) : court-circuite le pipeline conversationnel pour un
    # lookup par référence nue → réponse structurée sourcée.
    FICHE_TECHNIQUE_ENABLED: bool = os.getenv("FICHE_TECHNIQUE_ENABLED", "true").strip().lower() in (
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

    # Arbre SAV (refonte 2026-07-30) : plus AUCUN flag — le runtime déterministe est
    # toujours actif (reprise de session + entrées explicites bouton/chip), le RAG répond
    # toujours aux messages libres. Constantes en dur : guided_flow_service.SAV_CONTACT,
    # guided_entry_index_service.MIN_SIM, guided_attachment_context_service.CAG_MAX_CHARS.

    # Périmètre de recherche confirmé (human-in-the-loop) — filtre les documents par
    # classification (product_types/materials/proferm_gammes/source) AVANT le retrieval.
    # Master OFF par défaut : aucun impact tant que False (pipeline actuel inchangé).
    SCOPE_CONFIRMATION_ENABLED: bool = os.getenv(
        "SCOPE_CONFIRMATION_ENABLED", "false"
    ).strip().lower() in ("true", "1", "yes", "on")
    # confirm = propose une carte à valider quand un champ discriminant manque ;
    # auto = applique les champs CERTAIN sans carte (profil interne) ;
    # off = comportement actuel (aucun filtre de périmètre).
    SCOPE_MODE: str = os.getenv("SCOPE_MODE", "off").strip().lower()
    # Un champ n'est proposé À_DEMANDER que si l'espace contient AU MOINS ce nb de valeurs
    # distinctes (sinon non discriminant → inutile de demander).
    SCOPE_MIN_DISTINCT_TO_ASK: int = int(os.getenv("SCOPE_MIN_DISTINCT_TO_ASK", "2"))
    # Champs demandables "à l'aveugle" (absents ET discriminants) : l'utilisateur final
    # les connaît. Gamme/fournisseur ne sont que CONFIRMÉS s'ils sont détectés.
    SCOPE_ASKABLE_FIELDS: str = os.getenv("SCOPE_ASKABLE_FIELDS", "product_family,material")
    # TTL (s) du cache des stats de classification par espace (0 = pas de cache).
    SCOPE_STATS_CACHE_TTL: int = int(os.getenv("SCOPE_STATS_CACHE_TTL", "300"))

    @property
    def scope_askable_fields(self) -> List[str]:
        """Champs demandables à l'aveugle, normalisés depuis le CSV brut."""
        return [
            s.strip().lower()
            for s in (self.SCOPE_ASKABLE_FIELDS or "").split(",")
            if s.strip()
        ]

    @property
    def colpali_gating_intents(self) -> List[str]:
        """Liste normalisée des intents forçant ColPali (parsée depuis le CSV brut)."""
        return [
            s.strip().lower()
            for s in (self.COLPALI_GATING_INTENTS or "").split(",")
            if s.strip()
        ]

    @property
    def cag_doc_budget_shares(self) -> List[float]:
        """Parts de budget par rang d'élection, parsées depuis le CSV brut.

        Liste vide = partage désactivé (premier document servi jusqu'au budget global).
        Les valeurs non numériques ou négatives sont ignorées ; la renormalisation sur le
        nombre de documents réellement élus est faite par le packer.
        """
        shares: List[float] = []
        for raw in (self.CAG_DOC_BUDGET_SHARES or "").split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                logger.warning("CAG_DOC_BUDGET_SHARES : part illisible %r ignorée", raw)
                continue
            if value > 0:
                shares.append(value)
        return shares

    @property
    def cag_budget_by_intent(self) -> dict:
        """Table budget/max_documents par intent, parsée depuis le JSON brut env
        (repli sur les défauts si vide ou illisible)."""
        raw = (self.CAG_BUDGET_BY_INTENT or "").strip()
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, TypeError):
                logger.warning("CAG_BUDGET_BY_INTENT illisible (JSON invalide) — défauts utilisés")
        return _CAG_BUDGET_DEFAULTS

    @property
    def effective_verify_model(self) -> str:
        """Modèle réellement utilisé pour la vérification post-génération.

        VERIFY_MODEL s'il est renseigné, sinon MODEL_JUDGE — jamais le modèle de
        génération : un modèle qui se relit se blanchit (mécanisme n°1 du juge
        auto-complaisant, cas TGY3710)."""
        return self.VERIFY_MODEL or self.MODEL_JUDGE

    def coherence_warnings(self) -> List[str]:
        """Incohérences de configuration détectées au démarrage (jamais bloquantes).

        Sert de garde-fou contre le « kill switch caché » : un flag maître désactivé qui
        rend inertes des fonctionnalités par ailleurs configurées et activées.
        """
        warnings: List[str] = []

        if not self.QUERY_UNDERSTANDING_ENABLED:
            dependents = []
            if self.CONVERSATION_ANCHOR_ENABLED:
                dependents.append("ancrage conversationnel")
            if self.COLPALI_GATING_ENABLED and self.colpali_gating_intents:
                dependents.append("gating ColPali par intent")
            if self.RETRIEVAL_SOURCE_BOOST_MAX > 0 or self.RETRIEVAL_MATERIAL_BOOST > 0:
                dependents.append("boosts source/matériau")
            if dependents:
                warnings.append(
                    "QUERY_UNDERSTANDING_ENABLED=false rend INERTES : "
                    + ", ".join(dependents)
                    + " (aucun `signals` n'est produit avant le retrieval)."
                )

        if not self.RERANKER_ENABLED:
            warnings.append(
                "RERANKER_ENABLED=false : plus AUCUN étage de précision. Le classement "
                "final est de la pure fusion de rangs (RRF) — pas de seuil de pertinence, "
                "pas d'abstention, pas de reclassement par le contenu. La précision repose "
                "entièrement sur les filtres documentaires (classification + périmètre)."
            )

        if self.COLPALI_ENABLED and not self.MULTIMODAL_ENABLED:
            warnings.append(
                "COLPALI_ENABLED=true mais MULTIMODAL_ENABLED=false : l'ingestion "
                "multimodale est bloquée, aucun nouveau document ne sera indexé pour ColPali."
            )

        if not self.CORS_ALLOWED_ORIGINS:
            warnings.append(
                "CORS_ALLOWED_ORIGINS vide : CORS restreint au même origine (aucune origine "
                "cross-site autorisée). Renseigner la liste si un front séparé consomme l'API."
            )

        if self.AGENTIC_LOOP_ENABLED and self.AGENTIC_LOOP_MODE not in ("shadow", "active"):
            warnings.append(
                f"AGENTIC_LOOP_MODE={self.AGENTIC_LOOP_MODE!r} inconnu (attendu shadow|active) : "
                "la boucle sera traitée comme shadow (juge tracé, jamais actionné)."
            )
        if self.AGENTIC_LOOP_ENABLED and not self.CAG_ENABLED:
            warnings.append(
                "AGENTIC_LOOP_ENABLED=true mais CAG_ENABLED=false : le juge travaille sur les "
                "dossiers candidats du packing CAG — la boucle est INERTE sur le chemin legacy."
            )
        if self.VERIFY_BLOCKING and not self.VERIFY_ENABLED:
            warnings.append(
                "VERIFY_BLOCKING=true mais VERIFY_ENABLED=false : le gate de vérification est "
                "inerte (aucune vérification n'est exécutée)."
            )
        if self.VERIFY_ENABLED and self.effective_verify_model == self.MODEL_FAST:
            warnings.append(
                "Le modèle de vérification est identique au modèle de génération "
                f"({self.MODEL_FAST}) : le juge se relit lui-même — verdicts complaisants "
                "attendus. Renseigner VERIFY_MODEL ou MODEL_JUDGE avec un modèle distinct."
            )

        return warnings

    def feature_summary(self) -> str:
        """Matrice de features effective sur une ligne, pour vérifier la config au boot."""
        def onoff(flag: bool) -> str:
            return "on" if flag else "off"

        return (
            "[config] retrieval: "
            f"multimodal={onoff(self.MULTIMODAL_ENABLED)} "
            f"colpali={onoff(self.COLPALI_ENABLED)} "
            f"gating={onoff(self.COLPALI_GATING_ENABLED)} "
            f"reranker={onoff(self.RERANKER_ENABLED)} "
            f"vision_rerank={onoff(self.VISION_RERANK_ENABLED)} "
            f"cag={onoff(self.CAG_ENABLED)} "
            f"enrichment={onoff(self.CONTEXTUAL_ENRICHMENT_ENABLED)} "
            f"query_understanding={onoff(self.QUERY_UNDERSTANDING_ENABLED)} "
            f"anchor={onoff(self.CONVERSATION_ANCHOR_ENABLED)} "
            f"fiche={onoff(self.FICHE_TECHNIQUE_ENABLED)} "
            "guided=arbre-sav "
            f"loop={self.AGENTIC_LOOP_MODE if self.AGENTIC_LOOP_ENABLED else 'off'} "
            f"verify={('blocking' if self.VERIFY_BLOCKING else 'advisory') if self.VERIFY_ENABLED else 'off'}"
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

