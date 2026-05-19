"""
Service de recherche sémantique pour les espaces (architecture Document/Space).

Pipeline principal :
  1. Embedding de la requête via BGE-m3
  2. Détection multi-hop (heuristique : mots-clés + entités pivot)
  3a. [Standard] Candidats vectoriels (pgvector) + lexicaux (ts_rank_cd) + KAG (graphe)
      + parents enrichis (embedding section) → feuilles descendantes
  3b. [Multi-hop]  Orchestrateur max 3 sauts : hop0 hybride RRF → expansion graphe → sauts suivants
  4. Fusion RRF (vector, lexical, KAG, parent) + scoring multi-hop par RRF sur signaux + pénalité par hop
  5. Filtrage pré-reranking
  6. Early stopping si similarité vectorielle déjà élevée
  7. Reranking cross-encoder (BGE-reranker-v2-m3)
  8. Résolution des parents (clé ``node_id`` Docling + ``chunk-{id}``)
  9. Source authority (boost titre-requête)
  10. Fallback lexical ILIKE si aucun candidat hybride
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import os
import re
import threading
import unicodedata
from sqlmodel import Session, select
from sqlalchemy import or_, text
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.models.knowledge_entity import KnowledgeEntity
from app.models.chunk_entity_relation import ChunkEntityRelation
from app.models.space import Space
from app.services.space_service import get_space_by_id
from app.config import settings
from app.tracing import trace_run
import logging
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.services.query_reasoning_service import QueryIntent, reason_query_intent
from app.services.chunk_metadata_utils import (
    apply_row_metadata_defaults,
    content_type_score_multiplier,
    enrich_passage_content_for_llm,
    infer_primary_subject_key,
    merged_chunk_metadata as _merged_chunk_metadata,
    mmr_diversity_key,
    mmr_subject_key,
    table_citation_hint,
)

logger = logging.getLogger(__name__)

try:
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
    FLAG_RERANKER_AVAILABLE = True
except ImportError:
    FLAG_RERANKER_AVAILABLE = False

try:
    from llama_index.core.postprocessor import SentenceTransformersRerank
    ST_RERANKER_AVAILABLE = True
except ImportError:
    ST_RERANKER_AVAILABLE = False

RERANKER_AVAILABLE = FLAG_RERANKER_AVAILABLE or ST_RERANKER_AVAILABLE
if not RERANKER_AVAILABLE:
    logger.warning("Aucun composant de reranking (FlagEmbedding ou SentenceTransformers) disponible")

RERANKER_MODEL = settings.RERANKER_MODEL
RERANKER_CANDIDATE_MULTIPLIER = int(os.getenv("RERANKER_CANDIDATE_MULTIPLIER", "5"))
RERANKER_ENABLED = settings.RERANKER_ENABLED
MIN_VECTOR_SIMILARITY_THRESHOLD = float(os.getenv("MIN_VECTOR_SIMILARITY", "0.25"))
MAX_RERANK_CANDIDATES = int(os.getenv("MAX_RERANK_CANDIDATES", "50"))
# Deux étapes : large pool tronqué puis raffinement sur texte complet
RERANK_STAGE1_MAX = int(os.getenv("RERANK_STAGE1_MAX", "100"))
RERANK_STAGE2_POOL = int(os.getenv("RERANK_STAGE2_POOL", "25"))
RERANK_STAGE1_CHAR_CAP = int(os.getenv("RERANK_STAGE1_CHAR_CAP", "800"))
SKIP_RERANK_THRESHOLD = float(os.getenv("SKIP_RERANK_THRESHOLD", "0.85"))
_FLAG_RERANK_TOP_N = int(os.getenv("RERANKER_TOP_N", "4096"))
TRACE_VERBOSE_TEXT = os.getenv("TRACE_VERBOSE_TEXT", "false").lower() == "true"
TRACE_TEXT_MAX_CHARS = int(os.getenv("TRACE_TEXT_MAX_CHARS", "12000"))

# Singletons
_reranker_instance = None
_reranker_lock = threading.Lock() if RERANKER_AVAILABLE else None
_embed_model_instance = None


def _text_for_trace(node: TextNode) -> str:
    raw = (
        node.get_content()
        if hasattr(node, "get_content")
        else getattr(node, "text", "") or ""
    )
    if not isinstance(raw, str):
        raw = str(raw)
    if TRACE_TEXT_MAX_CHARS > 0 and len(raw) > TRACE_TEXT_MAX_CHARS:
        return raw[:TRACE_TEXT_MAX_CHARS]
    return raw


def _nodes_for_trace(candidates: List[NodeWithScore], limit: int = 80) -> List[Dict]:
    rows: List[Dict] = []
    for nws in candidates[:limit]:
        meta = dict(getattr(nws.node, "metadata", {}) or {})
        rows.append(
            {
                "score": round(float(nws.score or 0.0), 4),
                "document_title": meta.get("document_title"),
                "section": meta.get("parent_heading") or meta.get("heading"),
                "kag_entity": meta.get("kag_matched_entity"),
                "text": _text_for_trace(nws.node),
            }
        )
    return rows

_FALLBACK_STOPWORDS = {
    "the", "and", "for", "with", "dans", "avec", "pour", "une", "des", "les",
    "est", "sur", "pas", "plus", "que", "qui", "this", "that", "what", "how",
    "quoi", "comment", "quel", "quelle", "quels", "quelles", "from", "par",
    "sans", "mais", "donc", "car", "you", "your", "not", "are", "was", "were",
}

TITLE_QUERY_BOOST_PER_MATCH = float(os.getenv("TITLE_QUERY_BOOST_PER_MATCH", "0.5"))
TITLE_QUERY_BOOST_CAP = float(os.getenv("TITLE_QUERY_BOOST_CAP", "2.0"))

# Fusion hybride : RRF (Reciprocal Rank Fusion) sur vectoriel, lexical, KAG, parents enrichis
RRF_K = 60
# Poids du canal « parent enrichi » dans la somme RRF (modéré, ne domine pas le vectoriel leaf)
RRF_PARENT_LIST_WEIGHT = 0.50
# Seuil minimal de similarité parent (embedding summary+questions) pour descendre vers les feuilles
PARENT_ENRICHED_MIN_SIMILARITY = float(os.getenv("PARENT_ENRICHED_MIN_SIMILARITY", "0.65"))
# Filtrage post-fusion : scores RRF sont plus petits qu’une somme min-max sur [0,1]
RRF_MIN_SCORE = float(os.getenv("RRF_MIN_SCORE", "0.018"))
RRF_MIN_CHANNEL = float(os.getenv("RRF_MIN_CHANNEL", "0.010"))
HYBRID_MIN_SCORE = RRF_MIN_SCORE  # compat. nom interne

# Filtrage entités KAG par score de confiance calibré (KnowledgeEntity.confidence_score)
MIN_ENTITY_CONFIDENCE = float(settings.MIN_ENTITY_CONFIDENCE)


def _kag_entity_confidence_filter():
    """Entités sans score calibré exclues du retrieval KAG."""
    return KnowledgeEntity.confidence_score >= MIN_ENTITY_CONFIDENCE

# ---------------------------------------------------------------------------
# Multi-hop — constantes (budget/hops) + réglages via settings
# ---------------------------------------------------------------------------
MULTI_HOP_ENABLED = True
MULTI_HOP_MAX_HOPS = 3
MULTI_HOP_CANDIDATE_BUDGET = 80
MULTI_HOP_PER_HOP_LIMIT = 20
MULTI_HOP_PATIENCE = 1
MULTI_HOP_MIN_DELTA_NEW_CHUNKS = int(os.getenv("MULTI_HOP_MIN_DELTA_NEW_CHUNKS", "2"))
MH_RRF_PARENT_WEIGHT = RRF_PARENT_LIST_WEIGHT
MH_HOP_PENALTIES = {0: 0.00, 1: 0.05, 2: 0.10, 3: 0.15}

MMR_K = int(settings.MMR_K)
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", str(settings.MMR_LAMBDA)))
MMR_LAMBDA_COMPARATIVE = float(os.getenv("MMR_LAMBDA_COMPARATIVE", "0.55"))
MMR_LAMBDA_DEFAULT = MMR_LAMBDA

AMBIGUITY_GAP_THRESHOLD = float(os.getenv("RERANK_AMBIGUITY_GAP_THRESHOLD", "0.03"))
AMBIGUITY_STD_THRESHOLD = float(os.getenv("RERANK_AMBIGUITY_STD_THRESHOLD", "0.015"))

# Déclenchement multi-hop : triggers forts / faibles + exclusion FAQ
_MH_FAQ_PROCEDURE = re.compile(
    r"^(?:comment|how)\s+(?:installer|monter|fixer|régler|regler|choisir|commander|"
    r"mesurer|utiliser|poser|remplacer|démonter|demonter|ajuster)\b",
    re.IGNORECASE,
)
_MH_STRONG_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("comparaison", re.compile(r"\bcomparaison\b", re.IGNORECASE)),
    ("versus", re.compile(r"\b(?:versus|vs)\b", re.IGNORECASE)),
    ("difference", re.compile(r"\b(?:différence|difference)\b", re.IGNORECASE)),
    ("impact_sur", re.compile(r"\bimpact\b.+\bsur\b", re.IGNORECASE)),
    ("depend", re.compile(r"\b(?:dépend|depend)\b", re.IGNORECASE)),
    ("influence", re.compile(r"\binfluence\b", re.IGNORECASE)),
    ("implique", re.compile(r"\bimplique\b", re.IGNORECASE)),
    ("necessite", re.compile(r"\b(?:nécessite|necessite)\b", re.IGNORECASE)),
    ("incompatible", re.compile(r"\bincompatible\b", re.IGNORECASE)),
    ("remplace", re.compile(r"\bremplace\b", re.IGNORECASE)),
    ("entre_et", re.compile(r"\bentre\b.+\bet\b", re.IGNORECASE)),
    ("lien_entre", re.compile(r"\blien\s+entre\b", re.IGNORECASE)),
]
_MH_ENTRE_ET = re.compile(r"\bentre\b.+\bet\b", re.IGNORECASE)
_MH_WEAK_COMMENT = re.compile(r"\b(?:comment|pourquoi)\b", re.IGNORECASE)
_MH_WEAK_SI = re.compile(r"\bsi\b", re.IGNORECASE)
_MH_WEAK_ALORS = re.compile(r"\balors\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Structure d'état multi-hop
# ---------------------------------------------------------------------------

@dataclass
class _MultiHopState:
    """État de recherche conservé entre les sauts d'un pipeline multi-hop."""
    seen_chunk_ids: Set[int] = field(default_factory=set)
    seen_entity_ids: Set[int] = field(default_factory=set)
    seen_entity_names: Set[str] = field(default_factory=set)
    # chunk_id → {"vector": float, "lexical": float, "kag": float,
    #              "evidence": float, "hop": int, "path": str}
    chunk_signals: Dict[int, Dict] = field(default_factory=dict)
    # hop → nombre de nouveaux chunks apportés
    new_chunks_count_by_hop: Dict[int, int] = field(default_factory=dict)
    # hop_traces : chunk_id → liste des entités qui ont amené ce chunk
    hop_traces: Dict[int, List[str]] = field(default_factory=dict)


def _get_reranker():
    """Retourne le reranker (singleton)."""
    global _reranker_instance
    if not RERANKER_AVAILABLE:
        return None
    with _reranker_lock:
        if _reranker_instance is not None:
            return _reranker_instance
        use_fp16 = os.getenv("RERANKER_USE_FP16", "false").lower() == "true"
        # Utilisation de FlagEmbedding pour les modèles BGE, sinon SentenceTransformers
        is_bge = "bge-reranker" in RERANKER_MODEL.lower()
        
        logger.info(
            "Initialisation reranker %s (top_n=%s, type=%s, use_fp16=%s)...",
            RERANKER_MODEL,
            _FLAG_RERANK_TOP_N,
            "BGE" if is_bge else "SentenceTransformer",
            use_fp16,
        )
        
        try:
            if is_bge and FLAG_RERANKER_AVAILABLE:
                _reranker_instance = FlagEmbeddingReranker(
                    model=RERANKER_MODEL,
                    top_n=_FLAG_RERANK_TOP_N,
                    use_fp16=use_fp16,
                )
            elif ST_RERANKER_AVAILABLE:
                _reranker_instance = SentenceTransformersRerank(
                    model=RERANKER_MODEL,
                    top_n=_FLAG_RERANK_TOP_N,
                    device=os.getenv("EMBEDDING_DEVICE", "cpu")
                )
            elif FLAG_RERANKER_AVAILABLE:
                # Fallback sur FlagEmbeddingReranker (peut parfois charger d'autres modèles)
                _reranker_instance = FlagEmbeddingReranker(
                    model=RERANKER_MODEL,
                    top_n=_FLAG_RERANK_TOP_N,
                    use_fp16=use_fp16,
                )
            else:
                logger.error("Aucune classe de reranking disponible pour %s", RERANKER_MODEL)
                return None
        except Exception as e:
            logger.error("Erreur lors de l'instanciation du reranker %s : %s", RERANKER_MODEL, e)
            return None

        logger.info("✅ Reranker initialisé")
    return _reranker_instance


def _get_embed_model() -> HuggingFaceEmbedding:
    """Retourne le modèle d'embedding BGE-m3 (singleton)."""
    global _embed_model_instance
    if _embed_model_instance is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        device = os.getenv("EMBEDDING_DEVICE", "cpu")
        logger.info("Initialisation embedding %s sur %s...", model_name, device)
        _embed_model_instance = HuggingFaceEmbedding(
            model_name=model_name,
            device=device,
            embed_batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
        logger.info("✅ Modèle d'embedding initialisé")
    return _embed_model_instance


def _fetch_embeddings_for_chunks(
    session: Session,
    chunk_ids: List[int],
) -> Dict[int, np.ndarray]:
    """
    Récupère les embeddings bruts de la base pour un pool d'IDs.
    Utile pour la MMR sans surcharge de la requête initiale.
    """
    if not chunk_ids:
        return {}
    
    # Construction de la requête SQL directe pour les vecteurs
    stmt = text("""
        SELECT id, embedding FROM documentchunk 
        WHERE id = ANY(:ids) AND embedding IS NOT NULL
    """)
    result = session.execute(stmt, {"ids": chunk_ids})
    
    embeddings = {}
    import json
    for row in result:
        if row.embedding:
            try:
                # Si pgvector retourne une str (via raw text), on la parse
                emb_data = json.loads(row.embedding) if isinstance(row.embedding, str) else row.embedding
                embeddings[row.id] = np.array(emb_data, dtype=np.float32)
            except Exception as e:
                logger.warning("Impossible de parser l'embedding pour le chunk %s : %s", row.id, e)
            
    return embeddings


def _set_node_text_content(node, text: str) -> None:

    if hasattr(node, "set_content"):
        node.set_content(text)
    else:
        setattr(node, "text", text)


def _compute_mmr_with_parent_constraint(
    query_embedding: np.ndarray,
    candidates: List[NodeWithScore],
    candidate_embeddings: Dict[int, np.ndarray],
    target_k: int = MMR_K,
    lambda_param: float = MMR_LAMBDA,
    primary_subject_key: Optional[str] = None,
) -> List[NodeWithScore]:
    """
    Sélectionne target_k candidats parmi le pool en maximisant la MMR et la diversité de sources.

    sim_q = blend(score hybride, cosinus query/doc).
    Contrainte dure : 1 chunk par table_id ou parent_node_id.
    Pénalité souple si hors sujet primaire (gamme / pivot).
    """
    if not candidates or target_k <= 0:
        return []
        
    if len(candidates) <= 1:
        return candidates[:target_k]

    # Déterminer le score maximum pour la normalisation
    max_score = float(candidates[0].score or 1.0)
    if max_score <= 0.0:
        max_score = 1.0

    # Préparation des données
    selected_indices: List[int] = []
    candidates_pool = candidates[:]
    
    # Diversité : un chunk par tableau ou par parent section
    selected_diversity_keys: Set[str] = set()
    
    # Le premier est toujours le meilleur score (déjà trié)
    # Sauf s'il n'a pas d'embedding (fallback improbable)
    first_idx = 0
    while first_idx < len(candidates_pool):
        cid = _parse_chunk_id_from_node(candidates_pool[first_idx].node)
        if cid in candidate_embeddings:
            break
        first_idx += 1
    
    if first_idx >= len(candidates_pool):
        logger.warning("Aucun embedding trouvé pour le pool MMR, fallback tri simple")
        return candidates[:target_k]
        
    # Initialisation avec le premier
    selected_indices.append(first_idx)
    first_node = candidates_pool[first_idx].node
    div_key = mmr_diversity_key(dict(first_node.metadata or {}))
    if div_key:
        selected_diversity_keys.add(div_key)
    
    # On boucle jusqu'à avoir target_k ou épuisé le pool
    while len(selected_indices) < target_k and len(selected_indices) < len(candidates_pool):
        best_mmr = -1e9
        best_idx = -1
        
        # Embeddings des déjà sélectionnés
        sel_embeds = []
        for idx in selected_indices:
            cid = _parse_chunk_id_from_node(candidates_pool[idx].node)
            emb = candidate_embeddings[cid]
            sel_embeds.append(emb / np.linalg.norm(emb))
        sel_matrix = np.stack(sel_embeds)
        
        for i, nws in enumerate(candidates_pool):
            if i in selected_indices:
                continue
                
            cid = _parse_chunk_id_from_node(nws.node)
            if cid not in candidate_embeddings:
                continue
                
            # --- Contrainte diversité (table_id ou parent section) ---
            div_key = mmr_diversity_key(dict(nws.node.metadata or {}))
            if div_key and div_key in selected_diversity_keys:
                continue
                
            # --- Calcul MMR ---
            d_emb = candidate_embeddings[cid]
            d_emb = d_emb / np.linalg.norm(d_emb)
            
            hybrid_sim = float(nws.score or 0.0) / max_score
            embed_sim = float(np.dot(query_embedding / (np.linalg.norm(query_embedding) + 1e-9), d_emb))
            embed_sim = max(0.0, min(1.0, embed_sim))
            sim_q = (
                settings.MMR_SIMQ_HYBRID_WEIGHT * hybrid_sim
                + settings.MMR_SIMQ_EMBED_WEIGHT * embed_sim
            )

            sim_selected = float(np.max(np.dot(sel_matrix, d_emb)))
            mmr_score = lambda_param * sim_q - (1 - lambda_param) * sim_selected

            if primary_subject_key:
                cand_subject = mmr_subject_key(dict(nws.node.metadata or {}))
                if cand_subject and cand_subject != primary_subject_key:
                    mmr_score -= settings.MULTI_HOP_SUBJECT_MISMATCH_PENALTY

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i
                
        if best_idx == -1:
            logger.info(
                "MMR arrêt : plus de clés diversité disponibles (%d/%d trouvés)",
                len(selected_indices),
                target_k,
            )
            break
            
        selected_indices.append(best_idx)
        div_key = mmr_diversity_key(dict(candidates_pool[best_idx].node.metadata or {}))
        if div_key:
            selected_diversity_keys.add(div_key)
            
    final_selection = [candidates_pool[i] for i in selected_indices]
    logger.info(
        "MMR (space) : %d candidats sélectionnés sur %d (lambda=%.1f, diversité=%d)",
        len(final_selection),
        len(candidates_pool),
        lambda_param,
        len(selected_diversity_keys),
    )
    return final_selection


def _infer_adaptive_mmr_lambda(query_text: str, intent: Optional[QueryIntent]) -> float:
    """Calibre lambda MMR selon l'intention et des marqueurs comparatifs légers."""
    q = (query_text or "").lower()
    comparative_cues = (
        "comparaison", "compare", "versus", " vs ", "différence", "difference", "entre"
    )
    is_comparative = any(cue in q for cue in comparative_cues)
    if intent and isinstance(intent.intent, str):
        is_comparative = is_comparative or (intent.intent == "mixed")
    if is_comparative:
        return max(0.0, min(1.0, MMR_LAMBDA_COMPARATIVE))
    return max(0.0, min(1.0, MMR_LAMBDA_DEFAULT))


def _infer_parent_rrf_weight(query_text: str, intent: Optional[QueryIntent]) -> float:
    """Poids parent adaptatif: plus haut pour synthèse/procédure, sinon base."""
    q = (query_text or "").lower()
    explain_like = (
        "comment", "pourquoi", "étapes", "etapes", "procédure", "procedure", "résume", "resume"
    )
    if any(tok in q for tok in explain_like):
        return min(1.0, RRF_PARENT_LIST_WEIGHT + 0.15)
    if intent and getattr(intent, "intent", "") == "generic":
        return min(1.0, RRF_PARENT_LIST_WEIGHT + 0.05)
    return RRF_PARENT_LIST_WEIGHT


def _should_skip_reranking_adaptive(
    filtered_candidates: List[NodeWithScore],
    k: int,
    query_text: str,
    intent: Optional[QueryIntent],
) -> Tuple[bool, float]:
    """Skip rerank seulement si confiance élevée ET requête non ambiguë."""
    if len(filtered_candidates) < k:
        return False, 0.0
    scores = []
    for c in filtered_candidates[:k]:
        meta = dict(getattr(c.node, "metadata", {}) or {})
        vs = meta.get("vector_similarity")
        scores.append(float(vs) if vs is not None else float(c.score or 0.0))
    avg_top_k = sum(scores) / len(scores)
    if avg_top_k < SKIP_RERANK_THRESHOLD:
        return False, avg_top_k

    sorted_scores = sorted(scores, reverse=True)
    top_gap = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 1.0
    std = float(np.std(sorted_scores)) if len(sorted_scores) > 1 else 0.0
    q = (query_text or "").lower()
    ambiguity_tokens = ("ou", "ou bien", "vs", "versus", "comparaison", "différence", "difference")
    query_ambiguous = any(t in q for t in ambiguity_tokens)
    intent_ambiguous = bool(intent and getattr(intent, "intent", "") == "mixed")
    score_ambiguous = top_gap <= AMBIGUITY_GAP_THRESHOLD or std <= AMBIGUITY_STD_THRESHOLD
    ambiguous = query_ambiguous or intent_ambiguous or score_ambiguous
    return (not ambiguous), avg_top_k


def _single_stage_rerank_leaves(
    filtered_candidates: List[NodeWithScore],
    query_text: str,
    k: int,
    char_cap: int = 2000,
) -> List[NodeWithScore]:
    """
    Rerank unique (Single-Stage) sur le pool présélectionné par le MMR.
    Limite le texte envoyé au reranker à char_cap pour le ms-marco-MiniLM-L-6-v2 (512 tokens).
    """
    reranker = _get_reranker()
    if not reranker:
        return filtered_candidates[:k]

    pool = filtered_candidates
    backup: Dict[str, str] = {}
    for nws in pool:
        node = nws.node
        nid = str(getattr(node, "id_", None) or "")
        raw = (
            node.get_content()
            if hasattr(node, "get_content")
            else getattr(node, "text", "") or ""
        )
        backup[nid] = raw
        meta = dict(getattr(node, "metadata", {}) or {})
        enriched = _enrich_content_with_heading_and_figure(raw, meta)
        
        # Tronquage au niveau du cap de caractères
        short = enriched[:char_cap] if len(enriched) > char_cap else enriched
        _set_node_text_content(node, short)
    
    logger.info("Reranking (space) Single-Stage : traitement de %d candidats (cap %d chars)...", len(pool), char_cap)
    try:
        r = reranker.postprocess_nodes(
            pool,
            query_bundle=QueryBundle(query_str=query_text),
        )
        # Rétablir les textes d'origine complets pour la suite
        for nws in pool:
            nid = str(getattr(nws.node, "id_", None) or "")
            if nid in backup:
                _set_node_text_content(nws.node, backup[nid])
                
        # On s'assure de réaffecter les textes complets enrichis sur les candidats du résultat final
        for nws in r:
            nid = str(getattr(nws.node, "id_", None) or "")
            raw = backup.get(nid, "")
            meta = dict(getattr(nws.node, "metadata", {}) or {})
            enriched = _enrich_content_with_heading_and_figure(raw, meta)
            _set_node_text_content(nws.node, enriched)
            
        logger.info(
            "Reranking Single-Stage terminé : pool=%d → final=%d (top-k=%d)",
            len(pool),
            len(r),
            k,
        )
        return r[:k]
    except Exception as e:
        logger.warning("Rerank Single-Stage (space) échoué: %s", e)
        # Rétablir les textes d'origine complets en cas d'erreur
        for nws in pool:
            nid = str(getattr(nws.node, "id_", None) or "")
            if nid in backup:
                _set_node_text_content(nws.node, backup[nid])
        return filtered_candidates[:k]


def _retrieve_leaves_sql(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    candidate_k: int,
) -> List[NodeWithScore]:
    """
    Recherche vectorielle SQL directe sur les DocumentChunk LEAF via pgvector.
    """
    embed_model = _get_embed_model()
    query_embedding = embed_model.get_query_embedding(query_text)
    query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    sql_query = text(f"""
        SELECT
            dc.id,
            dc.content,
            dc.text,
            dc.chunk_index,
            dc.document_id,
            dc.metadata_json,
            dc.metadata_,
            dc.source AS chunk_source,
            d.title AS document_title,
            d.id AS document_id,
            1 - (dc.embedding <=> '{query_embedding_str}'::vector) AS similarity_score
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        INNER JOIN document_space ds ON ds.document_id = d.id
        WHERE ds.space_id = :space_id
          AND dc.embedding IS NOT NULL
          AND dc.is_leaf = true
        ORDER BY dc.embedding <=> '{query_embedding_str}'::vector
        LIMIT :limit_k
    """)

    result = session.execute(
        sql_query,
        {"space_id": space_id, "limit_k": candidate_k},
    )

    nodes_with_scores: List[NodeWithScore] = []
    for row in result:
        metadata = _merged_chunk_metadata(row.metadata_json, row.metadata_)
        apply_row_metadata_defaults(
            metadata,
            document_id=row.document_id,
            document_title=row.document_title or "Document sans titre",
            chunk_index=row.chunk_index,
            source=getattr(row, "chunk_source", None),
        )

        node = TextNode(
            id_=f"chunk-{row.id}",
            text=row.content or row.text or "",
            metadata=metadata,
        )
        nodes_with_scores.append(
            NodeWithScore(node=node, score=float(row.similarity_score))
        )

    logger.info(
        "Recherche SQL pgvector (space): %d nœuds leaf récupérés (candidate_k=%d)",
        len(nodes_with_scores),
        candidate_k,
    )
    return nodes_with_scores


def _parse_chunk_id_from_node(node: TextNode) -> Optional[int]:
    """Extrait l'ID base du chunk depuis ``chunk-{id}``."""
    nid = getattr(node, "id_", None) or ""
    if isinstance(nid, str) and nid.startswith("chunk-"):
        try:
            return int(nid.split("-", 1)[1])
        except ValueError:
            return None
    return None



def _prepare_flexible_tsquery(query_text: str) -> str:
    """
    Transforme une requête brute en tsquery flexible:
    1. Nettoyage (minuscule, retrait ponctuations simples)
    2. Filtrage des stopwords basiques (pour éviter le bruit avec le OR)
    3. Ajout de préfixes :* sur chaque terme
    4. Jointure avec OR (|) au lieu de AND (&)
    """
    if not query_text:
        return ""
    
    # Nettoyage: on garde lettres, chiffres et espaces
    cleaned = re.sub(r"[^\w\s]", " ", query_text.lower())
    # Découpage en mots
    words = [w.strip() for w in cleaned.split() if len(w.strip()) > 1]
    
    # Filtrage stopwords si plusieurs mots
    if len(words) > 1:
        words = [w for w in words if w not in _FALLBACK_STOPWORDS]
    
    if not words:
        # Fallback si tout a été filtré ou si requête courte
        words = [w.strip() for w in cleaned.split() if w.strip()]
        
    if not words:
        return ""
    
    # Construction de la chaîne pour to_tsquery : "word1:* | word2:* ..."
    # Le rank ts_rank_cd se chargera de donner plus de poids aux matches multiples.
    safe_words = [w.replace("'", "''") for w in words]
    return " | ".join(f"{w}:*" for w in safe_words)


def _retrieve_leaves_lexical_sql(
    session: Session,
    space_id: int,
    query_text: str,
    candidate_k: int,
) -> List[NodeWithScore]:
    """
    Recherche lexicale (ts_rank_cd / type BM25) sur les feuilles de l'espace.
    """
    if not query_text or not query_text.strip():
        return []

    flexible_q = _prepare_flexible_tsquery(query_text)
    if not flexible_q:
        return []

    sql_query = text(
        """
        SELECT
            dc.id,
            dc.content,
            dc.text,
            dc.chunk_index,
            dc.document_id,
            dc.metadata_json,
            dc.metadata_,
            dc.source AS chunk_source,
            d.title AS document_title,
            d.id AS document_id,
            ts_rank_cd(
                to_tsvector('simple', coalesce(dc.content, dc.text, '')),
                to_tsquery('simple', :q)
            ) AS lex_score
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        INNER JOIN document_space ds ON ds.document_id = d.id
        WHERE ds.space_id = :space_id
          AND dc.is_leaf = true
          AND coalesce(dc.content, dc.text, '') <> ''
          AND to_tsvector('simple', coalesce(dc.content, dc.text, ''))
              @@ to_tsquery('simple', :q)
        ORDER BY lex_score DESC
        LIMIT :limit_k
        """
    )

    result = session.execute(
        sql_query,
        {"space_id": space_id, "q": flexible_q, "limit_k": candidate_k},
    )

    nodes_with_scores: List[NodeWithScore] = []
    for row in result:
        metadata = _merged_chunk_metadata(row.metadata_json, row.metadata_)
        apply_row_metadata_defaults(
            metadata,
            document_id=row.document_id,
            document_title=row.document_title or "Document sans titre",
            chunk_index=row.chunk_index,
            source=getattr(row, "chunk_source", None),
        )

        node = TextNode(
            id_=f"chunk-{row.id}",
            text=row.content or row.text or "",
            metadata=metadata,
        )
        nodes_with_scores.append(
            NodeWithScore(node=node, score=float(row.lex_score or 0.0))
        )

    logger.info(
        "Recherche lexicale tsvector (space): %d feuilles (candidate_k=%d)",
        len(nodes_with_scores),
        candidate_k,
    )
    return nodes_with_scores


def _rrf_contrib(rank_zero_based: int, k: int = RRF_K) -> float:
    """Contribution RRF classique : 1 / (k + rank), rank 0 = meilleur."""
    return 1.0 / (float(k) + float(rank_zero_based))


def _retrieve_parent_enriched_sql(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    candidate_k: int,
    min_parent_sim: float = PARENT_ENRICHED_MIN_SIMILARITY,
) -> List[NodeWithScore]:
    """
    Recherche vectorielle sur les chunks parents enrichis (is_leaf=False, métadonnées LLM),
    puis propagation vers les feuilles dont ``parent_node_id`` pointe vers le parent.

    Le score des feuilles reflète la similarité requête↔embedding parent (signal assist).
    """
    embed_model = _get_embed_model()
    query_embedding = embed_model.get_query_embedding(query_text)
    query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    parent_limit = max(24, min(candidate_k, 80))
    sql_parents = text(f"""
        SELECT
            dc.id,
            dc.node_id,
            dc.content,
            dc.text,
            dc.chunk_index,
            dc.document_id,
            dc.metadata_json,
            dc.metadata_,
            d.title AS document_title,
            d.id AS document_id,
            1 - (dc.embedding <=> '{query_embedding_str}'::vector) AS similarity_score
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        INNER JOIN document_space ds ON ds.document_id = d.id
        WHERE ds.space_id = :space_id
          AND dc.embedding IS NOT NULL
          AND dc.is_leaf = false
          AND dc.metadata_json IS NOT NULL
        ORDER BY dc.embedding <=> '{query_embedding_str}'::vector
        LIMIT :limit_k
    """)

    result = session.execute(
        sql_parents,
        {"space_id": space_id, "limit_k": parent_limit},
    )
    parent_rows = list(result)
    if not parent_rows:
        logger.info("Parent enrichi (space): 0 parents candidats")
        return []

    parent_node_ids: List[str] = []
    parent_sim_by_node_id: Dict[str, float] = {}
    for row in parent_rows:
        sim = float(row.similarity_score or 0.0)
        if sim < min_parent_sim:
            continue
        nid = row.node_id
        if nid is None:
            continue
        ns = str(nid)
        parent_node_ids.append(ns)
        # Meilleur score si plusieurs parents partagent un même enfant théorique
        prev = parent_sim_by_node_id.get(ns)
        if prev is None or sim > prev:
            parent_sim_by_node_id[ns] = sim

    if not parent_node_ids:
        logger.info(
            "Parent enrichi (space): 0 parents au-dessus du seuil %.2f",
            min_parent_sim,
        )
        return []

    # Feuilles rattachées à ces sections (parent_node_id = UUID Docling du parent)
    stmt = (
        select(DocumentChunk, Document.title)
        .join(Document, Document.id == DocumentChunk.document_id)
        .join(DocumentSpace, DocumentSpace.document_id == Document.id)
        .where(
            DocumentSpace.space_id == space_id,
            DocumentChunk.is_leaf.is_(True),
            DocumentChunk.parent_node_id.in_(parent_node_ids),
        )
    )
    leaf_rows = session.exec(stmt).all()

    best_leaf: Dict[int, Tuple[float, DocumentChunk, str]] = {}
    for chunk, document_title in leaf_rows:
        pnid = chunk.parent_node_id
        if not pnid:
            continue
        pns = str(pnid)
        psim = parent_sim_by_node_id.get(pns)
        if psim is None:
            continue
        prev = best_leaf.get(chunk.id)
        if prev is None or psim > prev[0]:
            best_leaf[chunk.id] = (psim, chunk, document_title or "Document sans titre")

    nodes_with_scores: List[NodeWithScore] = []
    for _cid, (psim, chunk, doc_title) in best_leaf.items():
        metadata = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        apply_row_metadata_defaults(
            metadata,
            document_id=chunk.document_id,
            document_title=doc_title,
            chunk_index=chunk.chunk_index,
            source=chunk.source,
        )
        metadata["parent_enrichment_score"] = psim
        metadata["retrieval_signal"] = "parent_enriched_assist"

        node = TextNode(
            id_=f"chunk-{chunk.id}",
            text=chunk.content or chunk.text or "",
            metadata=metadata,
        )
        nodes_with_scores.append(NodeWithScore(node=node, score=float(psim)))

    nodes_with_scores.sort(key=lambda x: float(x.score or 0.0), reverse=True)
    nodes_with_scores = nodes_with_scores[:candidate_k]

    logger.info(
        "Parent enrichi (space): %d feuilles via %d parents (seuil sim≥%.2f)",
        len(nodes_with_scores),
        len(parent_sim_by_node_id),
        min_parent_sim,
    )
    return nodes_with_scores


def _hybrid_fuse_candidates(
    vector_candidates: List[NodeWithScore],
    lexical_candidates: List[NodeWithScore],
    graph_candidates: List[NodeWithScore],
    parent_candidates: Optional[List[NodeWithScore]] = None,
    parent_weight: float = RRF_PARENT_LIST_WEIGHT,
) -> List[NodeWithScore]:
    """
    Fusion RRF : vectoriel, lexical (BM25-like), KAG, et optionnellement parents enrichis.

    Listes ordonnées par pertinence décroissante ; chaque canal contribue 1/(RRF_K+rank),
    le canal parent est pondéré par ``parent_weight``.
    """
    parent_candidates = parent_candidates or []

    v_by_id: Dict[int, Tuple[float, TextNode]] = {}
    for nws in vector_candidates:
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is None:
            continue
        v_by_id[cid] = (float(nws.score or 0.0), nws.node)

    l_by_id: Dict[int, float] = {}
    l_nodes: Dict[int, TextNode] = {}
    for nws in lexical_candidates:
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is None:
            continue
        l_by_id[cid] = float(nws.score or 0.0)
        l_nodes[cid] = nws.node

    k_by_id: Dict[int, Tuple[float, TextNode]] = {}
    for nws in graph_candidates:
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is None:
            continue
        sc = float(nws.score or 0.0)
        if cid not in k_by_id or sc > k_by_id[cid][0]:
            k_by_id[cid] = (sc, nws.node)

    p_by_id: Dict[int, Tuple[float, TextNode]] = {}
    for nws in parent_candidates:
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is None:
            continue
        sc = float(nws.score or 0.0)
        if cid not in p_by_id or sc > p_by_id[cid][0]:
            p_by_id[cid] = (sc, nws.node)

    all_ids = set(v_by_id) | set(l_by_id) | set(k_by_id) | set(p_by_id)
    if not all_ids:
        return []

    rank_v: Dict[int, int] = {}
    for r, nws in enumerate(vector_candidates):
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is not None and cid not in rank_v:
            rank_v[cid] = r
    rank_l: Dict[int, int] = {}
    for r, nws in enumerate(lexical_candidates):
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is not None and cid not in rank_l:
            rank_l[cid] = r
    rank_k: Dict[int, int] = {}
    for r, nws in enumerate(graph_candidates):
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is not None and cid not in rank_k:
            rank_k[cid] = r
    rank_p: Dict[int, int] = {}
    for r, nws in enumerate(parent_candidates):
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is not None and cid not in rank_p:
            rank_p[cid] = r

    fused: List[NodeWithScore] = []
    for cid in all_ids:
        v_raw_s = v_by_id[cid][0] if cid in v_by_id else None

        rv = rank_v.get(cid, 10_000)
        rl = rank_l.get(cid, 10_000)
        rk = rank_k.get(cid, 10_000)
        rp = rank_p.get(cid, 10_000)

        c_v = _rrf_contrib(rv) if cid in rank_v else 0.0
        c_l = _rrf_contrib(rl) if cid in rank_l else 0.0
        c_k = _rrf_contrib(rk) if cid in rank_k else 0.0
        c_p = parent_weight * _rrf_contrib(rp) if cid in rank_p else 0.0

        hybrid = c_v + c_l + c_k + c_p

        if cid in v_by_id:
            node = v_by_id[cid][1]
        elif cid in l_nodes:
            node = l_nodes[cid]
        elif cid in k_by_id:
            node = k_by_id[cid][1]
        else:
            node = p_by_id[cid][1]

        meta = dict(getattr(node, "metadata", {}) or {})
        if v_raw_s is not None:
            meta["vector_similarity"] = v_raw_s
        meta["lexical_rrf"] = c_l
        meta["kag_rrf"] = c_k
        meta["vector_rrf"] = c_v
        meta["parent_rrf"] = c_p
        meta["lexical_norm"] = c_l  # compat. filtres / logs
        meta["kag_norm"] = c_k
        meta["hybrid_score"] = hybrid
        meta["retrieval_signal"] = "hybrid_rrf"
        meta["content_type_boost"] = content_type_score_multiplier(meta)
        hybrid *= meta["content_type_boost"]
        node.metadata = meta

        fused.append(NodeWithScore(node=node, score=hybrid))

    fused.sort(key=lambda x: float(x.score or 0.0), reverse=True)
    logger.debug(
        "Fusion RRF (space): %d candidats (vector=%d lexical=%d kag=%d parent=%d)",
        len(fused),
        len(rank_v),
        len(rank_l),
        len(rank_k),
        len(rank_p),
    )
    return fused


def _filter_hybrid_candidates(
    candidates: List[NodeWithScore],
) -> List[NodeWithScore]:
    """
    Filtre les candidats après fusion RRF : similarité vectorielle brute,
    score RRF total, ou contribution suffisante sur un canal.
    """
    out: List[NodeWithScore] = []
    for nws in candidates:
        meta = dict(getattr(nws.node, "metadata", {}) or {})
        h = float(nws.score or 0.0)
        vx = meta.get("vector_similarity")
        v_ok = vx is not None and float(vx) >= MIN_VECTOR_SIMILARITY_THRESHOLD
        ln = float(meta.get("lexical_norm", 0) or 0)
        kn = float(meta.get("kag_norm", 0) or 0)
        pr = float(meta.get("parent_rrf", 0) or 0)
        vr = float(meta.get("vector_rrf", 0) or 0)
        parent_sim = float(meta.get("parent_enrichment_score", 0) or 0)
        parent_sim_ok = parent_sim >= PARENT_ENRICHED_MIN_SIMILARITY * 0.85
        if (
            v_ok
            or h >= RRF_MIN_SCORE
            or ln >= RRF_MIN_CHANNEL
            or kn >= RRF_MIN_CHANNEL
            or pr >= RRF_MIN_CHANNEL
            or vr >= RRF_MIN_CHANNEL
            or parent_sim_ok
        ):
            out.append(nws)
    return out


def _build_parent_node_dict(
    session: Session, space_id: int, user_id: int
) -> Dict[str, TextNode]:
    """
    Charge les nœuds parents (DocumentChunk is_leaf=False) pour enrichir le contexte.

    Indexation par ``node_id`` Docling (UUID) lorsque présent : les feuilles référencent
    ``parent_node_id`` avec cet identifiant, pas ``chunk-{id}``.
    """
    statement = (
        select(DocumentChunk, Document.title)
        .join(Document, Document.id == DocumentChunk.document_id)
        .join(DocumentSpace, DocumentSpace.document_id == Document.id)
        .where(
            DocumentSpace.space_id == space_id,
            DocumentChunk.is_leaf.is_(False),
        )
    )
    rows = session.exec(statement).all()

    node_dict: Dict[str, TextNode] = {}
    for chunk, document_title in rows:
        metadata = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        apply_row_metadata_defaults(
            metadata,
            document_id=chunk.document_id,
            document_title=document_title or "Document sans titre",
            chunk_index=chunk.chunk_index,
            node_id=chunk.node_id,
            source=chunk.source,
        )
        llama_id = f"chunk-{chunk.id}"
        text = chunk.content or chunk.text or ""
        node = TextNode(
            id_=llama_id,
            text=text,
            metadata=metadata,
        )
        # Clé principale : UUID Docling / hiérarchie (aligné avec leaf.parent_node_id)
        if chunk.node_id:
            node_dict[str(chunk.node_id)] = node
        # Compatibilité anciens chemins
        node_dict[llama_id] = node

    logger.info(
        "Chargé %d nœuds parents pour space_id=%d (index node_id + chunk-id)",
        len(rows),
        space_id,
    )
    return node_dict


_PARENT_MULTIHOP_MAX = 4


def _resolve_space_parent_with_multihop(
    session: Session,
    space_id: int,
    user_id: int,
    document_id: Optional[int],
    parent_node_id: Optional[str],
    parent_node_dict: Dict[str, TextNode],
) -> Optional[TextNode]:
    """
    Si le parent direct n'est pas un chunk section (is_leaf=False), remonte la chaîne
    parent_node_id (text_full, table_full, …) jusqu'au premier parent section et fusionne
    les textes intermédiaires devant le corps du parent section.
    """
    if not parent_node_id or document_id is None:
        return None
    if parent_node_id in parent_node_dict:
        return None

    intermediates: List[str] = []
    current_pid: Optional[str] = parent_node_id
    hops = 0

    while current_pid and hops < _PARENT_MULTIHOP_MAX:
        hops += 1
        stmt = (
            select(DocumentChunk, Document.title)
            .join(Document, Document.id == DocumentChunk.document_id)
            .join(DocumentSpace, DocumentSpace.document_id == Document.id)
            .where(
                DocumentSpace.space_id == space_id,
                DocumentChunk.document_id == document_id,
                Document.user_id == user_id,
                DocumentChunk.node_id == current_pid,
            )
        )
        row = session.exec(stmt).first()
        if not row:
            break
        chunk, document_title = row
        metadata = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        apply_row_metadata_defaults(
            metadata,
            document_id=chunk.document_id,
            document_title=document_title or "Document sans titre",
            chunk_index=chunk.chunk_index,
            node_id=chunk.node_id,
            source=chunk.source,
        )
        llama_id = f"chunk-{chunk.id}"
        text = (chunk.content or chunk.text or "").strip()

        if not chunk.is_leaf:
            if intermediates:
                prefix = "\n\n---\n\n".join(reversed(intermediates))
                text = f"{prefix}\n\n---\n\n{text}"
            return TextNode(
                id_=llama_id,
                text=text,
                metadata=metadata,
            )

        if text:
            intermediates.append(text)
        current_pid = chunk.parent_node_id

    return None


def _extract_query_terms(query_text: str) -> List[str]:
    terms = [t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ0-9]+", query_text or "")]
    return [t for t in terms if len(t) >= 3 and t not in _FALLBACK_STOPWORDS][:8]


def _normalize_pivot_entities(pivot_entity_names: Optional[List[str]]) -> List[str]:
    """Normalise/déduplique les entités pivot extraites côté requête."""
    if not pivot_entity_names:
        return []
    seen: Set[str] = set()
    normalized: List[str] = []
    for name in pivot_entity_names:
        value = (name or "").strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
        if len(normalized) >= 12:
            break
    return normalized


async def _keyword_fallback_passages(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    k: int,
) -> List[Dict]:
    """Fallback lexical si aucun embedding disponible."""
    terms = _extract_query_terms(query_text)
    base_stmt = (
        select(DocumentChunk, Document.title)
        .join(Document, Document.id == DocumentChunk.document_id)
        .join(DocumentSpace, DocumentSpace.document_id == Document.id)
        .where(
            DocumentSpace.space_id == space_id,
        )
        .order_by(DocumentChunk.is_leaf.desc(), Document.updated_at.desc(), DocumentChunk.chunk_index)
    )

    rows = []
    if terms:
        stmt = base_stmt.where(
            or_(*[DocumentChunk.content.ilike(f"%{term}%") for term in terms])
        ).limit(max(k * 4, 12))
        rows = session.exec(stmt).all()

    if not rows:
        rows = session.exec(base_stmt.limit(max(k * 2, 8))).all()

    passages: List[Dict] = []
    seen_chunk_ids: set = set()
    for chunk, document_title in rows:
        if chunk.id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.id)

        content = (chunk.content or chunk.text or "").strip()
        if not content:
            continue

        lowered = content.lower()
        match_count = sum(1 for term in terms if term in lowered) if terms else 0
        score = (match_count / max(len(terms), 1)) if terms else 0.05

        node_metadata = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        apply_row_metadata_defaults(
            node_metadata,
            document_id=chunk.document_id,
            document_title=document_title or "Document sans titre",
            chunk_index=chunk.chunk_index,
            source=chunk.source,
        )
        node = TextNode(
            id_=f"chunk-{chunk.id}",
            text=content,
            metadata=node_metadata,
        )
        passages.append(_node_to_passage(node, fallback_score=score))
        if len(passages) >= k:
            break

    logger.info(
        "Fallback lexical (space): %d passages construits (terms=%s)",
        len(passages),
        terms,
    )
    return passages


def _enrich_content_with_heading_and_figure(content: str, metadata: dict) -> str:
    """Préfixe le contenu avec section, figure et résumé parent KAG."""
    return enrich_passage_content_for_llm(content, metadata)


def _merge_leaf_page_into_node_metadata(leaf_node, target_node) -> None:
    """
    Recopie page_no / plage depuis la feuille matchée vers le nœud cible (ex. parent résolu).
    Priorité à la page de la feuille pour l'ouverture PDF au bon endroit.
    """
    leaf_meta = dict(getattr(leaf_node, "metadata", {}) or {})
    m = dict(getattr(target_node, "metadata", {}) or {})
    pn = leaf_meta.get("page_no")
    if pn is not None:
        try:
            m["page_no"] = int(pn)
        except (TypeError, ValueError):
            pass
    elif m.get("page_start") is not None:
        try:
            m["page_no"] = int(m["page_start"])
        except (TypeError, ValueError):
            pass
    ps = leaf_meta.get("page_start")
    pe = leaf_meta.get("page_end")
    if ps is not None:
        try:
            m.setdefault("page_start", int(ps))
        except (TypeError, ValueError):
            pass
    if pe is not None:
        try:
            m.setdefault("page_end", int(pe))
        except (TypeError, ValueError):
            pass
    # Traçabilité: conserver l'ID du chunk feuille à l'origine de la citation.
    leaf_chunk_id = _parse_chunk_id_from_node(leaf_node)
    if leaf_chunk_id is not None:
        m["source_leaf_chunk_id"] = leaf_chunk_id
    setattr(target_node, "metadata", m)


def _node_to_passage(node, fallback_score: float = 0.0) -> Dict:
    metadata = dict(getattr(node, "metadata", {}) or {})
    document_title = metadata.get("document_title", "Document sans titre")
    document_id = metadata.get("document_id")
    chunk_index = metadata.get("chunk_index", 0)
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    raw_page = metadata.get("page_no")
    resolved_page = None
    if raw_page is not None:
        try:
            resolved_page = int(raw_page)
        except (TypeError, ValueError):
            pass
            
    if resolved_page is None:
        # Fallbacks successifs
        for key in ["page_start", "page_label", "page_idx"]:
            val = metadata.get(key)
            if val is not None:
                try:
                    resolved_page = int(val)
                    break
                except (TypeError, ValueError):
                    continue
                    
    page_no = resolved_page
    parent_heading = metadata.get("parent_heading") or metadata.get("heading")

    content = node.get_content() if hasattr(node, "get_content") else str(node)
    content_enriched = _enrich_content_with_heading_and_figure(content, metadata)
    passage_text = f"**{document_title}**\n{content_enriched}"
    chunk_id = _parse_chunk_id_from_node(node)
    source_leaf_chunk_id = metadata.get("source_leaf_chunk_id")
    table_hint = table_citation_hint(metadata)
    out = {
        "passage": passage_text,
        "passage_raw": content,
        "document_title": document_title,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "source_leaf_chunk_id": source_leaf_chunk_id,
        "chunk_index": int(chunk_index) if isinstance(chunk_index, (int, str)) else 0,
        "score": float(fallback_score or 0.0),
        "page_no": page_no,
        "section": parent_heading,
        "source": metadata.get("source"),
        "content_type": metadata.get("content_type"),
        "is_image_chunk": bool(metadata.get("is_image_chunk")),
        "image_path": metadata.get("image_path"),
        "image_filename": metadata.get("image_filename"),
        "caption": metadata.get("caption") or metadata.get("figure_title"),
    }
    if table_hint:
        out["table_citation"] = table_hint
    if settings.RAG_DEBUG_METADATA:
        out["retrieval_hop"] = metadata.get("retrieval_hop")
        out["hop_penalty"] = metadata.get("hop_penalty")
        out["retrieval_signal"] = metadata.get("retrieval_signal")
        out["kag_matched_entity"] = metadata.get("kag_matched_entity")
        out["vector_similarity"] = metadata.get("vector_similarity")
    if page_start is not None:
        try:
            out["page_start"] = int(page_start)
        except (TypeError, ValueError):
            pass
    if page_end is not None:
        try:
            out["page_end"] = int(page_end)
        except (TypeError, ValueError):
            pass
    return out


def _filter_low_similarity_candidates(
    candidates: List[NodeWithScore],
    min_threshold: float = MIN_VECTOR_SIMILARITY_THRESHOLD,
) -> List[NodeWithScore]:
    """Filtre les candidats avec faible similarité vectorielle."""
    filtered = [c for c in candidates if float(c.score or 0.0) >= min_threshold]
    if len(filtered) < len(candidates):
        logger.debug(
            "Filtrage similarité (space): %d → %d candidats (seuil=%.2f)",
            len(candidates),
            len(filtered),
            min_threshold,
        )
    return filtered


def _retrieve_via_knowledge_graph(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    limit: int = 10,
    pivot_entity_names: Optional[List[str]] = None,
) -> List[NodeWithScore]:
    """
    Récupère des chunks via le graphe de connaissances KAG de l'espace.
    
    Args:
        pivot_entity_names: Entités normalisées extraites de la requête par LLM (prioritaires)
    """
    try:
        from app.services.kag_graph_service import (
            expand_kag_query_terms_for_space,
            rank_neighbor_entities_for_query,
        )

        # Stratégie 1: utiliser les entités pivot LLM si disponibles
        normalized_pivots = _normalize_pivot_entities(pivot_entity_names)
        if normalized_pivots:
            query_terms = normalized_pivots
            logger.debug(
                "KAG retrieval (space): utilisation de %d entités pivot LLM",
                len(query_terms),
            )
        else:
            # Fallback: split naïf de la requête
            query_terms = [t.strip().lower() for t in re.findall(r"[A-Za-zÀ-ÿ0-9]+", query_text)]
            query_terms = [t for t in query_terms if len(t) >= 3 and t not in _FALLBACK_STOPWORDS]

        if not query_terms:
            return []

        query_terms = expand_kag_query_terms_for_space(session, space_id, query_terms)
        query_terms = query_terms[:120]

        # Matching exact sur name_normalized
        stmt = (
            select(
                DocumentChunk,
                Document.title,
                KnowledgeEntity.name,
                KnowledgeEntity.id,
                ChunkEntityRelation.relevance_score,
            )
            .join(ChunkEntityRelation, ChunkEntityRelation.chunk_id == DocumentChunk.id)
            .join(KnowledgeEntity, KnowledgeEntity.id == ChunkEntityRelation.entity_id)
            .join(Document, Document.id == DocumentChunk.document_id)
            .join(DocumentSpace, DocumentSpace.document_id == Document.id)
            .where(
                DocumentSpace.space_id == space_id,
                DocumentChunk.is_leaf == True,
                KnowledgeEntity.space_id == space_id,
                KnowledgeEntity.name_normalized.in_(query_terms),
                _kag_entity_confidence_filter(),
            )
            .order_by(ChunkEntityRelation.relevance_score.desc())
            .limit(limit)
        )
        results = list(session.exec(stmt).all())

        # Fallback ILIKE partiel si peu de résultats exacts
        if len(results) < limit // 2 and query_terms:
            logger.debug(
                "KAG retrieval (space): fallback ILIKE (résultats exacts=%d)",
                len(results),
            )
            ilike_terms = query_terms[:5]
            ilike_conditions = [
                KnowledgeEntity.name_normalized.ilike(f"%{term}%")
                for term in ilike_terms
            ]
            stmt_ilike = (
                select(
                    DocumentChunk,
                    Document.title,
                    KnowledgeEntity.name,
                    KnowledgeEntity.id,
                    ChunkEntityRelation.relevance_score,
                )
                .join(ChunkEntityRelation, ChunkEntityRelation.chunk_id == DocumentChunk.id)
                .join(KnowledgeEntity, KnowledgeEntity.id == ChunkEntityRelation.entity_id)
                .join(Document, Document.id == DocumentChunk.document_id)
                .join(DocumentSpace, DocumentSpace.document_id == Document.id)
                .where(
                    DocumentSpace.space_id == space_id,
                    DocumentChunk.is_leaf == True,
                    KnowledgeEntity.space_id == space_id,
                    or_(*ilike_conditions),
                    _kag_entity_confidence_filter(),
                )
                .order_by(ChunkEntityRelation.relevance_score.desc())
                .limit(limit - len(results))
            )
            ilike_results = list(session.exec(stmt_ilike).all())
            existing_chunk_ids = {row[0].id for row in results}
            for row in ilike_results:
                if row[0].id not in existing_chunk_ids:
                    results.append(row)
                    existing_chunk_ids.add(row[0].id)

        neighbor_chunk_ids: Set[int] = set()
        seed_entity_ids = {row[3] for row in results if row[3] is not None}
        ranked_neighbors = rank_neighbor_entities_for_query(
            session, space_id, seed_entity_ids, query_text=query_text
        )
        neighbor_entity_ids = {nid for nid, _ in ranked_neighbors}
        if neighbor_entity_ids and len(results) < limit:
            stmt_neigh = (
                select(
                    DocumentChunk,
                    Document.title,
                    KnowledgeEntity.name,
                    KnowledgeEntity.id,
                    ChunkEntityRelation.relevance_score,
                )
                .join(ChunkEntityRelation, ChunkEntityRelation.chunk_id == DocumentChunk.id)
                .join(KnowledgeEntity, KnowledgeEntity.id == ChunkEntityRelation.entity_id)
                .join(Document, Document.id == DocumentChunk.document_id)
                .join(DocumentSpace, DocumentSpace.document_id == Document.id)
                .where(
                    DocumentSpace.space_id == space_id,
                    DocumentChunk.is_leaf == True,
                    KnowledgeEntity.space_id == space_id,
                    KnowledgeEntity.id.in_(neighbor_entity_ids),
                    _kag_entity_confidence_filter(),
                )
                .order_by(ChunkEntityRelation.relevance_score.desc())
                .limit(max(0, limit - len(results)))
            )
            existing_chunk_ids = {row[0].id for row in results}
            for row in session.exec(stmt_neigh).all():
                if row[0].id not in existing_chunk_ids:
                    results.append(row)
                    existing_chunk_ids.add(row[0].id)
                    neighbor_chunk_ids.add(row[0].id)

        nodes_with_scores: List[NodeWithScore] = []
        for chunk, document_title, entity_name, _entity_id, relevance in results:
            metadata = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
            apply_row_metadata_defaults(
                metadata,
                document_id=chunk.document_id,
                document_title=document_title or "Document sans titre",
                chunk_index=chunk.chunk_index,
                source=chunk.source,
            )
            metadata["kag_matched_entity"] = entity_name
            if chunk.id in neighbor_chunk_ids:
                metadata["kag_neighbor_match"] = True
            rel = float(relevance)
            if chunk.id in neighbor_chunk_ids:
                rel *= 0.88

            node = TextNode(
                id_=f"chunk-{chunk.id}",
                text=chunk.content or chunk.text or "",
                metadata=metadata,
            )
            nodes_with_scores.append(NodeWithScore(node=node, score=rel))

        logger.debug(
            "KAG retrieval (space): %d chunks via graphe (query_terms=%s)",
            len(nodes_with_scores),
            query_terms[:5],
        )
        return nodes_with_scores

    except Exception as e:
        logger.warning("Erreur KAG retrieval (space): %s", e)
        return []


def _merge_with_graph_candidates(
    vector_candidates: List[NodeWithScore],
    graph_candidates: List[NodeWithScore],
    graph_boost: float = 0.2,
    pivot_entity_names: Optional[List[str]] = None,
) -> List[NodeWithScore]:
    """Fusionne les candidats vectoriels et KAG."""
    seen_node_ids = set()
    merged: List[NodeWithScore] = []
    pivot_set = set(pivot_entity_names or [])
    normalize_entity_name = None
    if pivot_set:
        from app.services.kag_extraction_service import normalize_entity_name as _norm
        normalize_entity_name = _norm

    for nws in vector_candidates:
        node_id = getattr(nws.node, "id_", None)
        if node_id:
            seen_node_ids.add(node_id)
        merged.append(nws)

    for nws in graph_candidates:
        node_id = getattr(nws.node, "id_", None)
        matched_entity = (nws.node.metadata or {}).get("kag_matched_entity", "")
        is_pivot = bool(
            pivot_set and matched_entity and normalize_entity_name
            and normalize_entity_name(matched_entity) in pivot_set
        )
        boost = 2.0 * graph_boost if is_pivot else graph_boost
        if node_id and node_id in seen_node_ids:
            for existing in merged:
                existing_id = getattr(existing.node, "id_", None)
                if existing_id == node_id:
                    existing.score = max(existing.score, nws.score + boost)
                    break
            continue

        boosted_score = min(1.0, float(nws.score or 0.0) + boost)
        nws.score = boosted_score
        merged.append(nws)
        if node_id:
            seen_node_ids.add(node_id)

    merged.sort(key=lambda x: float(x.score or 0.0), reverse=True)

    logger.debug(
        "Fusion KAG (space): %d vectoriels + %d graphe → %d total",
        len(vector_candidates),
        len(graph_candidates),
        len(merged),
    )
    return merged


def _normalize_for_gamme(s: str) -> str:
    """Lowercase + suppression des accents."""
    if not s:
        return ""
    n = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in n if unicodedata.category(c) != "Mn")


def _get_meaningful_words(text: str) -> Set[str]:
    """Extrait les mots significatifs."""
    if not text or not text.strip():
        return set()
    normalized = _normalize_for_gamme(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return {
        w for w in tokens
        if len(w) > 3 and w not in _FALLBACK_STOPWORDS
    }


def refine_with_source_authority(
    passages: List[Dict],
    query_text: str,
    reasoning_result: Optional[QueryIntent] = None,
) -> List[Dict]:
    """
    Source authority : boost les passages dont le titre correspond à la requête,
    OU qui correspondent à la source privilégiée déterminée par le raisonnement (CQR).
    """
    if not passages:
        return passages

    # 1. Boost basé sur le raisonnement (CQR)
    if reasoning_result and reasoning_result.primary_source:
        source_to_boost = reasoning_result.primary_source.lower()
        boost_value = 0.8  # Boost significatif pour la source voulue
        for p in passages:
            # On récupère la source du document (le chunk l'a via la migration/ingestion)
            doc_source = (p.get("source") or "").lower()
            if doc_source == source_to_boost:
                p["score"] = float(p.get("score") or 0.0) + boost_value

    # 2. Boost basé sur les mots du titre (Existant)
    if query_text and query_text.strip():
        query_words = _get_meaningful_words(query_text)
        if query_words:
            for p in passages:
                document_title = (p.get("document_title") or "").strip()
                if not document_title:
                    continue
                title_words = _get_meaningful_words(document_title)
                common = query_words & title_words
                if common:
                    boost = min(
                        TITLE_QUERY_BOOST_PER_MATCH * len(common),
                        TITLE_QUERY_BOOST_CAP,
                    )
                    p["score"] = float(p.get("score") or 0.0) + boost

    passages.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return passages


def _match_strong_multihop_trigger(query_text: str) -> Optional[str]:
    for name, pattern in _MH_STRONG_PATTERNS:
        if pattern.search(query_text):
            return name
    return None


def _match_weak_multihop_trigger(
    query_text: str,
    pivot_entity_names: List[str],
) -> Optional[str]:
    """Triggers faibles : nécessitent un contexte (pivot, entre X et Y, si+alors)."""
    has_pivot = len(pivot_entity_names) >= 1
    two_pivots = len(pivot_entity_names) >= 2
    strong = _match_strong_multihop_trigger(query_text)
    entre_et = bool(_MH_ENTRE_ET.search(query_text))

    if _MH_WEAK_COMMENT.search(query_text):
        if strong or entre_et or two_pivots:
            return "comment_pourquoi+context"
        if has_pivot and settings.MULTI_HOP_REQUIRE_PIVOT_FOR_WEAK:
            return "comment_pourquoi+pivot"

    if _MH_WEAK_SI.search(query_text) and _MH_WEAK_ALORS.search(query_text):
        if strong or (has_pivot and settings.MULTI_HOP_REQUIRE_PIVOT_FOR_WEAK):
            return "si_alors+context"

    if entre_et and re.search(r"\b(?:comparaison|versus|vs)\b", query_text, re.IGNORECASE):
        return "et+comparaison"

    return None


def _needs_multi_hop(query_text: str, pivot_entity_names: List[str]) -> bool:
    """
    Détecte si la requête nécessite un retrieval multi-hop.

    - ≥2 pivots CQR → oui
    - Trigger fort (comparaison, impact sur, entre X et Y, …) → oui
    - Trigger faible (comment, si/alors) → seulement avec contexte
    - FAQ procédurale (« comment installer… ») → non
    """
    if not query_text or not query_text.strip():
        return False

    q = query_text.strip()
    if _MH_FAQ_PROCEDURE.search(q):
        logger.debug("Multi-hop skipped: faq_procedure")
        return False

    if len(pivot_entity_names) >= 2:
        logger.debug("Multi-hop trigger: pivots:%d", len(pivot_entity_names))
        return True

    strong = _match_strong_multihop_trigger(q)
    if strong:
        logger.debug("Multi-hop trigger: strong:%s", strong)
        return True

    token_count = len(re.findall(r"\S+", q))
    if token_count < settings.MULTI_HOP_KEYWORD_MIN_TOKENS:
        return False

    weak = _match_weak_multihop_trigger(q, pivot_entity_names)
    if weak:
        logger.debug("Multi-hop trigger: weak:%s", weak)
        return True

    return False


def _need_multi_hop_classifier_score(
    query_text: str,
    pivot_entity_names: List[str],
    intent: Optional[QueryIntent] = None,
) -> float:
    """
    Classifieur heuristique léger [0,1] pour activer le multi-hop (Phase B).
    """
    q = (query_text or "").strip()
    if not q:
        return 0.0
    score = 0.0
    if len(pivot_entity_names) >= 2:
        score += 0.40
    if _match_strong_multihop_trigger(q):
        score += 0.35
    if _match_weak_multihop_trigger(q, pivot_entity_names):
        score += 0.20
    if intent and getattr(intent, "intent", "") == "mixed":
        score += 0.15
    if _MH_FAQ_PROCEDURE.search(q):
        score -= 0.50
    return max(0.0, min(1.0, score))


def _rank_entity_seeds_for_hop(
    entity_names: List[str],
    query_text: str,
    pivot_entity_names: List[str],
) -> List[str]:
    """Priorise les seeds d'expansion (match pivot / requête)."""
    from app.services.kag_extraction_service import normalize_entity_name

    pivot_norm = {normalize_entity_name(p) for p in pivot_entity_names if p}
    q_lower = query_text.lower()

    def _score(name: str) -> float:
        norm = normalize_entity_name(name)
        sc = 0.0
        if norm in pivot_norm:
            sc += 2.0
        if name.lower() in q_lower or norm in q_lower:
            sc += 1.0
        return sc

    ranked = sorted(set(entity_names), key=lambda n: _score(n), reverse=True)
    return ranked[: settings.MULTI_HOP_MAX_SEEDS_PER_HOP]


def _extract_top_entity_names_from_candidates(
    candidates: List[NodeWithScore],
    top_n: int = 10,
) -> List[str]:
    """
    Extrait les noms d'entités KAG dominantes des meilleurs candidats courants.
    Utilisé entre deux hops pour définir les seeds du saut suivant.
    """
    entity_counter: Dict[str, int] = {}
    for nws in candidates:
        meta = dict(getattr(nws.node, "metadata", {}) or {})
        entity = meta.get("kag_matched_entity")
        if entity and isinstance(entity, str):
            key = entity.strip().lower()
            if key:
                entity_counter[key] = entity_counter.get(key, 0) + 1
    sorted_entities = sorted(entity_counter, key=lambda e: entity_counter[e], reverse=True)
    return sorted_entities[:top_n]


def _retrieve_kag_for_entity_seeds(
    session: Session,
    space_id: int,
    entity_names: List[str],
    limit: int,
    seen_chunk_ids: Set[int],
    seen_entity_ids: Set[int],
    query_text: str = "",
) -> Tuple[List[NodeWithScore], Set[int]]:
    """
    Retrieval KAG ciblé à partir d'une liste de noms d'entités seeds.
    Exclut les chunks et entités déjà vus dans les hops précédents.
    Retourne uniquement des chunks nouveaux et l'ensemble des IDs d'entités
    touchées (seeds + voisins) pour alimenter ``seen_entity_ids``.
    """
    try:
        from app.services.kag_graph_service import (
            expand_kag_query_terms_for_space,
            rank_neighbor_entities_for_query,
        )
        from app.services.kag_extraction_service import normalize_entity_name

        normalized = [normalize_entity_name(n) for n in entity_names if n]
        normalized = [n for n in normalized if n]
        if not normalized:
            return [], set()

        expanded = expand_kag_query_terms_for_space(session, space_id, normalized)
        expanded = expanded[:80]

        stmt = (
            select(
                DocumentChunk,
                Document.title,
                KnowledgeEntity.name,
                KnowledgeEntity.id,
                ChunkEntityRelation.relevance_score,
            )
            .join(ChunkEntityRelation, ChunkEntityRelation.chunk_id == DocumentChunk.id)
            .join(KnowledgeEntity, KnowledgeEntity.id == ChunkEntityRelation.entity_id)
            .join(Document, Document.id == DocumentChunk.document_id)
            .join(DocumentSpace, DocumentSpace.document_id == Document.id)
            .where(
                DocumentSpace.space_id == space_id,
                DocumentChunk.is_leaf == True,
                KnowledgeEntity.space_id == space_id,
                KnowledgeEntity.name_normalized.in_(expanded),
                _kag_entity_confidence_filter(),
            )
            .order_by(ChunkEntityRelation.relevance_score.desc())
            .limit(limit * 2)
        )
        rows = list(session.exec(stmt).all())

        entities_touched: Set[int] = set()
        for row in rows:
            eid = row[3]
            if eid is not None:
                entities_touched.add(int(eid))

        # Expansion voisins graphe depuis les entités seeds
        neighbor_edge_scores: Dict[int, float] = {}
        seed_entity_ids = {row[3] for row in rows if row[3] is not None} - seen_entity_ids
        if seed_entity_ids:
            ranked_neighbors = rank_neighbor_entities_for_query(
                session,
                space_id,
                seed_entity_ids,
                query_text=query_text,
            )
            neighbor_edge_scores = {nid: sc for nid, sc in ranked_neighbors}
            neighbor_ids = set(neighbor_edge_scores.keys()) - seen_entity_ids
            for nid in neighbor_ids:
                entities_touched.add(int(nid))
            if neighbor_ids:
                stmt_n = (
                    select(
                        DocumentChunk,
                        Document.title,
                        KnowledgeEntity.name,
                        KnowledgeEntity.id,
                        ChunkEntityRelation.relevance_score,
                    )
                    .join(ChunkEntityRelation, ChunkEntityRelation.chunk_id == DocumentChunk.id)
                    .join(KnowledgeEntity, KnowledgeEntity.id == ChunkEntityRelation.entity_id)
                    .join(Document, Document.id == DocumentChunk.document_id)
                    .join(DocumentSpace, DocumentSpace.document_id == Document.id)
                    .where(
                        DocumentSpace.space_id == space_id,
                        DocumentChunk.is_leaf == True,
                        KnowledgeEntity.space_id == space_id,
                        KnowledgeEntity.id.in_(neighbor_ids),
                        _kag_entity_confidence_filter(),
                    )
                    .order_by(ChunkEntityRelation.relevance_score.desc())
                    .limit(limit)
                )
                existing_ids = {row[0].id for row in rows}
                for row in session.exec(stmt_n).all():
                    eid = row[3]
                    if eid is not None:
                        entities_touched.add(int(eid))
                    if row[0].id not in existing_ids:
                        rows.append(row)
                        existing_ids.add(row[0].id)

        nodes: List[NodeWithScore] = []
        seen_in_batch: Set[int] = set()
        for chunk, doc_title, entity_name, eid, relevance in rows:
            if chunk.id in seen_chunk_ids or chunk.id in seen_in_batch:
                continue
            seen_in_batch.add(chunk.id)
            metadata = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
            apply_row_metadata_defaults(
                metadata,
                document_id=chunk.document_id,
                document_title=doc_title or "Document sans titre",
                chunk_index=chunk.chunk_index,
                source=chunk.source,
            )
            metadata["kag_matched_entity"] = entity_name
            if eid is not None and int(eid) in neighbor_edge_scores:
                metadata["kag_neighbor_edge_score"] = round(
                    float(neighbor_edge_scores[int(eid)]), 4
                )
            node = TextNode(
                id_=f"chunk-{chunk.id}",
                text=chunk.content or chunk.text or "",
                metadata=metadata,
            )
            nodes.append(NodeWithScore(node=node, score=float(relevance)))
            if len(nodes) >= limit:
                break

        return nodes, entities_touched

    except Exception as exc:
        logger.warning("KAG ciblé (multi-hop) échoué: %s", exc)
        return [], set()


def _apply_multihop_depth_scoring(
    state: _MultiHopState,
    all_nodes: Dict[int, NodeWithScore],
) -> List[NodeWithScore]:
    """
    Score unifié multi-hop : RRF sur les classements par signal brut
    (vector, lexical, kag, evidence, parent) puis pénalité par profondeur.
    """
    if not state.chunk_signals:
        return list(all_nodes.values())

    cids = list(state.chunk_signals.keys())

    def _rank_by(key: str) -> Dict[int, int]:
        sorted_c = sorted(
            cids,
            key=lambda c: float(state.chunk_signals[c].get(key, 0.0) or 0.0),
            reverse=True,
        )
        return {c: i for i, c in enumerate(sorted_c)}

    rv = _rank_by("vector")
    rl = _rank_by("lexical")
    rk = _rank_by("kag")
    re_e = _rank_by("evidence")
    rp = _rank_by("parent")

    scored: List[NodeWithScore] = []
    for cid, sig in state.chunk_signals.items():
        nws = all_nodes.get(cid)
        if nws is None:
            continue
        hop = sig.get("hop", 0)
        penalty = MH_HOP_PENALTIES.get(hop, MH_HOP_PENALTIES[MULTI_HOP_MAX_HOPS])

        mh_score = (
            _rrf_contrib(rv[cid])
            + _rrf_contrib(rl[cid])
            + _rrf_contrib(rk[cid])
            + _rrf_contrib(re_e[cid])
            + MH_RRF_PARENT_WEIGHT * _rrf_contrib(rp[cid])
            - penalty
        )
        mh_score = max(0.0, mh_score)

        meta = dict(getattr(nws.node, "metadata", {}) or {})
        meta["retrieval_hop"] = hop
        meta["hop_penalty"] = penalty
        meta["evidence_score"] = sig.get("evidence", 0.0)
        meta["retrieval_path"] = sig.get("path", f"hop{hop}")
        meta["mh_score"] = mh_score
        nws.node.metadata = meta
        scored.append(NodeWithScore(node=nws.node, score=mh_score))

    scored.sort(key=lambda x: float(x.score or 0.0), reverse=True)
    return scored


def multi_hop_retrieve_space(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    pivot_entity_names: List[str],
    candidate_k: int,
) -> List[NodeWithScore]:
    """
    Orchestrateur multi-hop pour la recherche dans un espace.

    Enchaîne jusqu'à MULTI_HOP_MAX_HOPS sauts :
    - Hop 0 : retrieval hybride standard (vector + lexical + KAG pivot).
    - Hop N : expansion graphe à partir des entités dominantes des meilleurs
              chunks du saut précédent + retrieval KAG ciblé.
    Applique le scoring par profondeur et retourne les candidats fusionnés
    triés, prêts pour le filtre pré-rerank et le reranker existants.
    """
    state = _MultiHopState()
    all_nodes: Dict[int, NodeWithScore] = {}

    # --- Hop 0 : retrieval hybride complet ---
    vector_candidates = _retrieve_leaves_sql(
        session=session,
        space_id=space_id,
        user_id=user_id,
        query_text=query_text,
        candidate_k=candidate_k,
    )
    lexical_candidates = _retrieve_leaves_lexical_sql(
        session=session,
        space_id=space_id,
        query_text=query_text,
        candidate_k=candidate_k,
    )
    graph_candidates_hop0 = _retrieve_via_knowledge_graph(
        session=session,
        space_id=space_id,
        user_id=user_id,
        query_text=query_text,
        limit=candidate_k,
        pivot_entity_names=pivot_entity_names or None,
    )

    parent_candidates_hop0 = _retrieve_parent_enriched_sql(
        session=session,
        space_id=space_id,
        user_id=user_id,
        query_text=query_text,
        candidate_k=candidate_k,
    )
    p_by_id_map: Dict[int, float] = {}
    for nws in parent_candidates_hop0:
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is not None:
            sc = float(nws.score or 0.0)
            if cid not in p_by_id_map or sc > p_by_id_map[cid]:
                p_by_id_map[cid] = sc

    # Indexer les signaux bruts du hop 0
    v_by_id: Dict[int, float] = {}
    for nws in vector_candidates:
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is not None:
            v_by_id[cid] = float(nws.score or 0.0)

    l_by_id: Dict[int, float] = {}
    for nws in lexical_candidates:
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is not None:
            l_by_id[cid] = float(nws.score or 0.0)

    k_by_id: Dict[int, float] = {}
    for nws in graph_candidates_hop0:
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is not None:
            sc = float(nws.score or 0.0)
            if cid not in k_by_id or sc > k_by_id[cid]:
                k_by_id[cid] = sc

    hop0_ids = set(v_by_id) | set(l_by_id) | set(k_by_id) | set(p_by_id_map)
    new_at_hop0 = 0

    # Fusionner dans l'état global
    for cid in hop0_ids:
        state.seen_chunk_ids.add(cid)
        state.chunk_signals[cid] = {
            "vector": v_by_id.get(cid, 0.0),
            "lexical": l_by_id.get(cid, 0.0),
            "kag": k_by_id.get(cid, 0.0),
            "evidence": 0.0,
            "parent": p_by_id_map.get(cid, 0.0),
            "hop": 0,
            "path": "hop0:hybrid",
        }
        new_at_hop0 += 1

    # Conserver les nœuds en utilisant la fusion RRF pour le nœud de référence
    fused_hop0 = _hybrid_fuse_candidates(
        vector_candidates=vector_candidates,
        lexical_candidates=lexical_candidates,
        graph_candidates=graph_candidates_hop0,
        parent_candidates=parent_candidates_hop0,
    )
    for nws in fused_hop0:
        cid = _parse_chunk_id_from_node(nws.node)
        if cid is not None:
            all_nodes[cid] = nws

    state.new_chunks_count_by_hop[0] = new_at_hop0
    logger.info(
        "Multi-hop (space) hop0: %d candidats hybrides (space_id=%d)",
        len(all_nodes),
        space_id,
    )

    # --- Hop 1..N : Expansion progressive ---
    stagnation_count = 0

    for hop in range(1, MULTI_HOP_MAX_HOPS + 1):
        if len(state.seen_chunk_ids) >= MULTI_HOP_CANDIDATE_BUDGET:
            logger.info(
                "Multi-hop (space) arrêt budget (%d >= %d) au hop %d",
                len(state.seen_chunk_ids),
                MULTI_HOP_CANDIDATE_BUDGET,
                hop,
            )
            break

        # Extraire les entités seeds des meilleurs candidats actuels
        top_current = sorted(
            all_nodes.values(),
            key=lambda x: float(x.score or 0.0),
            reverse=True,
        )[:MULTI_HOP_PER_HOP_LIMIT]
        seed_entity_names = _extract_top_entity_names_from_candidates(top_current, top_n=12)
        seed_entity_names = _rank_entity_seeds_for_hop(
            seed_entity_names, query_text, pivot_entity_names
        )

        new_seeds = [n for n in seed_entity_names if n not in state.seen_entity_names]
        if not new_seeds:
            logger.info("Multi-hop (space) arrêt saturation entités au hop %d", hop)
            break

        # Marquer ces entités comme étant désormais expansionnées
        state.seen_entity_names.update(new_seeds)

        hop_candidates, entities_touched = _retrieve_kag_for_entity_seeds(
            session=session,
            space_id=space_id,
            entity_names=new_seeds,
            limit=MULTI_HOP_PER_HOP_LIMIT,
            seen_chunk_ids=state.seen_chunk_ids,
            seen_entity_ids=state.seen_entity_ids,
            query_text=query_text,
        )
        state.seen_entity_ids.update(entities_touched)

        new_count = 0
        for nws in hop_candidates:
            cid = _parse_chunk_id_from_node(nws.node)
            if cid is None or cid in state.seen_chunk_ids:
                continue
            state.seen_chunk_ids.add(cid)
            kag_score = float(nws.score or 0.0)
            meta = dict(getattr(nws.node, "metadata", {}) or {})
            entity_name = meta.get("kag_matched_entity", "")
            # L'evidence = nombre d'entités pivot déjà connues dans ce chunk
            evidence = sum(
                1 for e in pivot_entity_names
                if e and e.lower() in (meta.get("kag_matched_entity") or "").lower()
            )
            edge_sc = meta.get("kag_neighbor_edge_score")
            path_suffix = f",e={edge_sc}" if edge_sc is not None else ""
            state.chunk_signals[cid] = {
                "vector": 0.0,
                "lexical": 0.0,
                "kag": kag_score,
                "evidence": float(evidence),
                "parent": 0.0,
                "hop": hop,
                "path": f"hop{hop}:{','.join(new_seeds[:3])}{path_suffix}",
            }
            if entity_name:
                state.hop_traces.setdefault(cid, []).append(entity_name)
            all_nodes[cid] = nws
            new_count += 1

        state.new_chunks_count_by_hop[hop] = new_count

        logger.info(
            "Multi-hop (space) hop%d: +%d nouveaux chunks (seeds=%s)",
            hop,
            new_count,
            new_seeds[:3],
        )

        if new_count < MULTI_HOP_MIN_DELTA_NEW_CHUNKS:
            stagnation_count += 1
            if stagnation_count >= MULTI_HOP_PATIENCE:
                logger.info(
                    "Multi-hop (space) arrêt stagnation après hop %d (%d saut(s) vide(s))",
                    hop,
                    stagnation_count,
                )
                break
        else:
            stagnation_count = 0

    # --- Scoring final par profondeur ---
    scored_candidates = _apply_multihop_depth_scoring(state, all_nodes)

    logger.info(
        "Multi-hop (space) terminé: %d candidats finaux (hops=%s)",
        len(scored_candidates),
        dict(state.new_chunks_count_by_hop),
    )
    return scored_candidates


async def search_relevant_passages(
    session: Session,
    space_id: int,
    query_text: str,
    user_id: int,
    k: int = 15,
) -> List[Dict]:
    """
    Recherche sémantique RAG + KAG sur les documents d'un espace.
    Intègre désormais un système de raisonnement cognitif (CQR) pour la priorisation des sources.
    """
    space = get_space_by_id(session, space_id, user_id)
    if not space:
        logger.warning("Espace %d non trouvé ou non accessible pour l'utilisateur %d", space_id, user_id)
        return []

    if not query_text or not query_text.strip():
        logger.warning("Requête vide fournie")
        return []

    # --- Étape 0 : Raisonnement cognitif sur la requête ---
    reasoning_result = await reason_query_intent(query_text)
    if reasoning_result.intent != "generic":
        logger.info(
            "CQR reasoning [space]: intent=%s primary_source=%s confidence=%.2f",
            reasoning_result.intent,
            reasoning_result.primary_source,
            reasoning_result.confidence
        )

    try:
        candidate_k = (
            k * RERANKER_CANDIDATE_MULTIPLIER
            if (RERANKER_AVAILABLE and RERANKER_ENABLED)
            else k
        )

        # --- Étape 1 : extraction entités pivot (KAG) ---
        pivot_entity_names: List[str] = []
        if settings.KAG_ENABLED:
            try:
                from app.services.kag_extraction_service import extract_entities_from_query_sync

                pivot_entity_names = extract_entities_from_query_sync(query_text)
                if pivot_entity_names:
                    logger.debug("Entités pivot requête (space): %s", pivot_entity_names[:5])
            except Exception as ext_err:
                logger.debug("Extraction entités requête ignorée (space): %s", ext_err)

        # --- Étape 2 : sélection du mode retrieval ---
        mh_score = _need_multi_hop_classifier_score(
            query_text=query_text,
            pivot_entity_names=pivot_entity_names,
            intent=reasoning_result,
        )
        use_multi_hop = (
            MULTI_HOP_ENABLED
            and settings.KAG_ENABLED
            and (_needs_multi_hop(query_text, pivot_entity_names) or mh_score >= 0.55)
        )
        parent_rrf_weight = _infer_parent_rrf_weight(query_text, reasoning_result)

        if use_multi_hop:
            logger.info(
                "Multi-hop activé (space_id=%d) — pivots=%s",
                space_id,
                pivot_entity_names[:4],
            )
            with trace_run(
                "multi_hop_retrieval",
                run_type="retriever",
                inputs={
                    "query": query_text,
                    "space_id": space_id,
                    "pivot_entities": pivot_entity_names[:10],
                    "candidate_k": candidate_k,
                    "max_hops": MULTI_HOP_MAX_HOPS,
                },
                tags=["retrieval", "multi-hop", "kag", "space"],
            ) as mh_run:
                leaf_candidates = multi_hop_retrieve_space(
                    session=session,
                    space_id=space_id,
                    user_id=user_id,
                    query_text=query_text,
                    pivot_entity_names=pivot_entity_names,
                    candidate_k=candidate_k,
                )
                top3_scores = [round(float(c.score or 0), 4) for c in leaf_candidates[:3]]
                mh_outputs = {
                    "nb_candidates": len(leaf_candidates),
                    "top3_scores": top3_scores,
                }
                if TRACE_VERBOSE_TEXT:
                    mh_outputs["candidates_text"] = _nodes_for_trace(leaf_candidates)
                mh_run.end(outputs=mh_outputs)
        else:
            # Pipeline hybride standard
            with trace_run(
                "vector_retrieval",
                run_type="retriever",
                inputs={"query": query_text, "space_id": space_id, "candidate_k": candidate_k},
                tags=["retrieval", "vector", "pgvector", "space"],
            ) as vr_run:
                vector_candidates = _retrieve_leaves_sql(
                    session=session,
                    space_id=space_id,
                    user_id=user_id,
                    query_text=query_text,
                    candidate_k=candidate_k,
                )
                vr_outputs = {
                    "nb_candidates": len(vector_candidates),
                    "top3_scores": [round(float(c.score or 0), 4) for c in vector_candidates[:3]],
                }
                if TRACE_VERBOSE_TEXT:
                    vr_outputs["candidates_text"] = _nodes_for_trace(vector_candidates)
                vr_run.end(outputs=vr_outputs)

            with trace_run(
                "lexical_retrieval",
                run_type="retriever",
                inputs={"query": query_text, "space_id": space_id, "candidate_k": candidate_k},
                tags=["retrieval", "lexical", "tsvector", "space"],
            ) as lr_run:
                lexical_candidates = _retrieve_leaves_lexical_sql(
                    session=session,
                    space_id=space_id,
                    query_text=query_text,
                    candidate_k=candidate_k,
                )
                lr_outputs = {"nb_candidates": len(lexical_candidates)}
                if TRACE_VERBOSE_TEXT:
                    lr_outputs["candidates_text"] = _nodes_for_trace(lexical_candidates)
                lr_run.end(outputs=lr_outputs)

            graph_candidates: List[NodeWithScore] = []
            if settings.KAG_ENABLED:
                try:
                    with trace_run(
                        "kag_graph_retrieval",
                        run_type="retriever",
                        inputs={"query": query_text, "pivot_entities": pivot_entity_names[:10], "space_id": space_id},
                        tags=["retrieval", "kag", "graph", "space"],
                    ) as kag_run:
                        graph_candidates = _retrieve_via_knowledge_graph(
                            session=session,
                            space_id=space_id,
                            user_id=user_id,
                            query_text=query_text,
                            limit=candidate_k,
                            pivot_entity_names=pivot_entity_names or None,
                        )
                        matched = list({
                            (c.node.metadata or {}).get("kag_matched_entity", "")
                            for c in graph_candidates
                            if (c.node.metadata or {}).get("kag_matched_entity")
                        })
                        kag_outputs = {"nb_chunks": len(graph_candidates), "matched_entities": matched[:10]}
                        if TRACE_VERBOSE_TEXT:
                            kag_outputs["candidates_text"] = _nodes_for_trace(graph_candidates)
                        kag_run.end(outputs=kag_outputs)
                except Exception as kag_err:
                    logger.warning("KAG retrieval échoué (space): %s", kag_err)

            parent_candidates = _retrieve_parent_enriched_sql(
                session=session,
                space_id=space_id,
                user_id=user_id,
                query_text=query_text,
                candidate_k=candidate_k,
            )

            with trace_run(
                "hybrid_fusion",
                run_type="chain",
                inputs={
                    "nb_vector": len(vector_candidates),
                    "nb_lexical": len(lexical_candidates),
                    "nb_kag": len(graph_candidates),
                    "nb_parent": len(parent_candidates),
                    "rrf_k": RRF_K,
                    "parent_list_weight": parent_rrf_weight,
                },
                tags=["fusion", "hybrid", "rrf", "space"],
            ) as fusion_run:
                leaf_candidates = _hybrid_fuse_candidates(
                    vector_candidates=vector_candidates,
                    lexical_candidates=lexical_candidates,
                    graph_candidates=graph_candidates,
                    parent_candidates=parent_candidates,
                    parent_weight=parent_rrf_weight,
                )
                fusion_outputs = {"nb_fused": len(leaf_candidates)}
                if TRACE_VERBOSE_TEXT:
                    fusion_outputs["fused_text"] = _nodes_for_trace(leaf_candidates)
                fusion_run.end(outputs=fusion_outputs)

        if not leaf_candidates:
            logger.info(
                "Aucun candidat (space, multi_hop=%s), activation fallback lexical ILIKE",
                use_multi_hop,
            )
            return await _keyword_fallback_passages(
                session=session,
                space_id=space_id,
                user_id=user_id,
                query_text=query_text,
                k=k,
            )

        # --- Étape 3 : filtrage pré-reranking ---
        filtered_candidates = _filter_hybrid_candidates(leaf_candidates)
        if not filtered_candidates:
            logger.info(
                "Filtre hybride vide (space) — repli sur candidats fusionnés bruts"
            )
            filtered_candidates = leaf_candidates

        # Early stopping
        skip_reranking = False
        skip_reranking, avg_top_k = _should_skip_reranking_adaptive(
            filtered_candidates=filtered_candidates,
            k=k,
            query_text=query_text,
            intent=reasoning_result,
        )
        if skip_reranking:
            logger.info(
                "Similarité élevée et non ambiguë (space) (%.3f >= %.2f), skip reranking",
                avg_top_k,
                SKIP_RERANK_THRESHOLD,
            )
            top_leaves = filtered_candidates[:k]

        # --- Étape 3 : Reranking unique précédé de diversification MMR ---
        if not skip_reranking and RERANKER_AVAILABLE and RERANKER_ENABLED:
            try:
                if _get_reranker():
                    rerank_pool_size = min(MAX_RERANK_CANDIDATES, len(filtered_candidates))
                    
                    # 1. Sélection par MMR en amont pour diversifier les candidats envoyés au reranker
                    if len(filtered_candidates) > k:
                        try:
                            with trace_run(
                                "mmr_selection",
                                run_type="chain",
                                inputs={"nb_pool": len(filtered_candidates), "target_k": rerank_pool_size, "lambda": _infer_adaptive_mmr_lambda(query_text, reasoning_result)},
                                tags=["mmr", "diversity", "space"],
                            ) as mmr_run:
                                query_embedding = np.array(
                                    _get_embed_model().get_query_embedding(query_text), 
                                    dtype=np.float32
                                )
                                chunk_ids = [
                                    cid for nws in filtered_candidates
                                    if (cid := _parse_chunk_id_from_node(nws.node)) is not None
                                ]
                                candidate_embeddings = _fetch_embeddings_for_chunks(session, chunk_ids)
                                
                                primary_subject = infer_primary_subject_key(
                                    pivot_entity_names,
                                    filtered_candidates,
                                )
                                mmr_target_k = min(MMR_K, rerank_pool_size)
                                mmr_pool = _compute_mmr_with_parent_constraint(
                                    query_embedding=query_embedding,
                                    candidates=filtered_candidates,
                                    candidate_embeddings=candidate_embeddings,
                                    target_k=mmr_target_k,
                                    lambda_param=_infer_adaptive_mmr_lambda(query_text, reasoning_result),
                                    primary_subject_key=primary_subject,
                                )
                                mmr_run.end(outputs={"nb_final": len(mmr_pool)})
                        except Exception as mmr_err:
                            logger.warning("Sélection MMR pré-rerank échouée (space), fallback : %s", mmr_err)
                            mmr_pool = filtered_candidates[:rerank_pool_size]
                    else:
                        mmr_pool = filtered_candidates

                    # 2. Passage unique au reranker cross-encoder MiniLM (cap 2000 chars)
                    with trace_run(
                        "reranking",
                        run_type="chain",
                        inputs={
                            "nb_candidates": len(mmr_pool),
                            "target_k": k,
                            "multi_hop": use_multi_hop,
                        },
                        tags=["reranking", "space"],
                    ) as rerank_run:
                        top_leaves = _single_stage_rerank_leaves(
                            mmr_pool,
                            query_text,
                            k,
                            char_cap=2000,
                        )
                        rerank_run.end(outputs={"nb_pool": len(top_leaves)})
                else:
                    logger.warning("Reranker non disponible, fallback order")
                    top_leaves = filtered_candidates[:k]
            except Exception as rerank_err:
                logger.warning("Reranking échoué : %s", rerank_err)
                top_leaves = filtered_candidates[:k]
        elif not skip_reranking:
            top_leaves = filtered_candidates[:k]


        # --- Étape 4 : résolution des parents ---
        with trace_run(
            "parent_resolution",
            run_type="chain",
            inputs={"space_id": space_id, "nb_top_leaves": len(top_leaves)},
            tags=["parent", "context", "space"],
        ) as parent_run:
            parent_node_dict = _build_parent_node_dict(session, space_id, user_id)

            final_nodes: List[NodeWithScore] = []
            seen_node_ids: set = set()
            parents_resolved = 0
            parents_not_found = 0

            for nws in top_leaves:
                score = float(getattr(nws, "score", 0.0) or 0.0)
                leaf_meta = dict(getattr(nws.node, "metadata", {}) or {})
                parent_node_id = leaf_meta.get("parent_node_id")

                target_node = None
                if parent_node_id:
                    target_node = parent_node_dict.get(parent_node_id)
                    if target_node is None:
                        doc_id = leaf_meta.get("document_id")
                        try:
                            doc_id_int = int(doc_id) if doc_id is not None else None
                        except (TypeError, ValueError):
                            doc_id_int = None
                        target_node = _resolve_space_parent_with_multihop(
                            session,
                            space_id,
                            user_id,
                            doc_id_int,
                            parent_node_id,
                            parent_node_dict,
                        )
                if target_node is None:
                    target_node = nws.node
                    if parent_node_id:
                        parents_not_found += 1
                else:
                    parents_resolved += 1
                    _merge_leaf_page_into_node_metadata(nws.node, target_node)

                node_id = getattr(target_node, "id_", None)
                if node_id and node_id in seen_node_ids:
                    continue
                if node_id:
                    seen_node_ids.add(node_id)

                final_nodes.append(NodeWithScore(node=target_node, score=score))

            parent_outputs = {
                "nb_final_passages": len(final_nodes),
                "parents_resolved": parents_resolved,
                "parents_not_found": parents_not_found,
            }
            if TRACE_VERBOSE_TEXT:
                parent_outputs["top_leaves_text"] = _nodes_for_trace(top_leaves)
                parent_outputs["final_nodes_text"] = _nodes_for_trace(final_nodes)
            parent_run.end(outputs=parent_outputs)

        logger.info(
            "Résolution parents (space): %d passages finaux (%d parents résolus, %d non trouvés)",
            len(final_nodes),
            parents_resolved,
            parents_not_found,
        )

        passages = [
            _node_to_passage(nws.node, fallback_score=float(nws.score or 0.0))
            for nws in final_nodes
        ]

        score_strs = [f"{p['score']:.3f}" for p in passages[:3]]
        logger.info(
            "Trouvé %d passages (space)%s (scores: %s...)",
            len(passages),
            " [reranked]" if (RERANKER_AVAILABLE and RERANKER_ENABLED) else "",
            score_strs,
        )

        passages = refine_with_source_authority(passages, query_text, reasoning_result=reasoning_result)

        if not passages:
            return await _keyword_fallback_passages(
                session=session,
                space_id=space_id,
                user_id=user_id,
                query_text=query_text,
                k=k,
            )

        return passages

    except Exception as e:
        logger.error("Erreur recherche passages (space): %s", e, exc_info=True)
        return []
