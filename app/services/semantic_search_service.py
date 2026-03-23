"""
Service de recherche sémantique RAG simplifié et optimisé.

Pipeline :
  1. Embedding de la requête via BGE-m3 (singleton)
  2. Recherche vectorielle SQL directe (pgvector <=>) sur les chunks LEAF uniquement
  3. Optimisations pré-reranking :
     - Filtrage des candidats avec faible similarité vectorielle (< MIN_VECTOR_SIMILARITY_THRESHOLD)
     - Early stopping si similarité déjà élevée (>= SKIP_RERANK_THRESHOLD)
  4. Reranking des candidats avec BGE-reranker-v2-m3 (cross-encoder, sur les leaves,
     courts → rapide) — remplace le reranking sur les parents qui prenait ~56s
  5. Résolution des parents pour fournir un contexte enrichi au LLM
  6. Fallback lexical si aucun embedding disponible

Optimisations :
  - Filtrage pré-reranking : évite de reranker des candidats peu pertinents (-20 à -40% de temps)
  - Early stopping : skip le reranking si similarité vectorielle déjà élevée (~10-20% des cas)
  - Limite dynamique : ajuste le nombre de candidats selon les besoins (max MAX_RERANK_CANDIDATES)
  - Device configurable : possibilité d'utiliser GPU via RERANKER_DEVICE

Les anciens composants LlamaIndex (PGVectorStore, VectorStoreIndex, RecursiveRetriever,
MetadataFilter) ont été supprimés : la recherche SQL directe est plus fiable et cohérente
avec l'architecture de la table notechunk (insertions via SQLModel ORM).
"""

from typing import Dict, List, Optional, Set
import os
import re
import threading
import unicodedata
from sqlmodel import Session, select
from sqlalchemy import or_
from app.models.note import Note
from app.models.document_chunk import DocumentChunk
from app.services.embedding_service import generate_embedding
from app.services.project_service import get_project_by_id
import logging
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.config import settings

logger = logging.getLogger(__name__)

try:
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("FlagEmbeddingReranker non disponible, reranking désactivé")

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_CANDIDATE_MULTIPLIER = 3
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
# Optimisations du reranking
MIN_VECTOR_SIMILARITY_THRESHOLD = float(os.getenv("MIN_VECTOR_SIMILARITY", "0.25"))
MAX_RERANK_CANDIDATES = int(os.getenv("MAX_RERANK_CANDIDATES", "50"))
SKIP_RERANK_THRESHOLD = float(os.getenv("SKIP_RERANK_THRESHOLD", "0.85"))

# ---------------------------------------------------------------------------
# Singletons — chargement unique des modèles lourds
# ---------------------------------------------------------------------------

_reranker_instance = None
_reranker_lock = threading.Lock() if RERANKER_AVAILABLE else None

_embed_model_instance = None
_embed_model_lock = threading.Lock()


def _get_reranker():
    """
    Retourne le reranker (singleton) avec configuration optimisée.
    
    Configuration :
    - top_n=None : pas de limite fixe, ajusté dynamiquement selon les besoins
    - device : configurable via RERANKER_DEVICE (défaut: cpu)
    """
    global _reranker_instance
    if not RERANKER_AVAILABLE:
        return None
    with _reranker_lock:
        if _reranker_instance is None:
            device = os.getenv("RERANKER_DEVICE", "cpu")
            logger.info(
                "Initialisation du reranker %s sur %s (une seule fois)...",
                RERANKER_MODEL,
                device,
            )
            _reranker_instance = FlagEmbeddingReranker(
                model=RERANKER_MODEL,
                top_n=None,  # Pas de limite fixe, ajusté dynamiquement
                device=device,
            )
            logger.info("✅ Reranker initialisé et prêt (device=%s)", device)
        return _reranker_instance


def _get_embed_model() -> HuggingFaceEmbedding:
    """Retourne le modèle d'embedding BGE-m3 (singleton)."""
    global _embed_model_instance
    with _embed_model_lock:
        if _embed_model_instance is None:
            model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
            device = os.getenv("EMBEDDING_DEVICE", "cpu")
            logger.info(
                "Initialisation du modèle d'embedding %s sur %s (une seule fois)...",
                model_name,
                device,
            )
            _embed_model_instance = HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
                embed_batch_size=settings.EMBEDDING_BATCH_SIZE,
            )
            logger.info("✅ Modèle d'embedding initialisé et prêt")
        return _embed_model_instance


# ---------------------------------------------------------------------------
# Recherche vectorielle SQL directe (pgvector)
# ---------------------------------------------------------------------------

def _retrieve_leaves_sql(
    session: Session,
    project_id: int,
    user_id: int,
    query_text: str,
    candidate_k: int,
) -> List[NodeWithScore]:
    """
    Recherche vectorielle SQL directe sur les chunks LEAF via l'opérateur pgvector <=>.

    Avantages par rapport à LlamaIndex PGVectorStore :
    - Filtre directement sur les colonnes SQL (is_leaf, project_id, user_id) au lieu des
      métadonnées JSON → résultats toujours cohérents, sans friction de typage
    - Pas de dépendance à la structure de table LlamaIndex
    - Une seule requête SQL, pas de boucles de fallback
    """
    from sqlalchemy import text

    embed_model = _get_embed_model()
    query_embedding = embed_model.get_query_embedding(query_text)
    query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    sql_query = text(f"""
        SELECT
            nc.id,
            nc.content,
            nc.text,
            nc.node_id,
            nc.parent_node_id,
            nc.metadata_json,
            nc.metadata_,
            n.title  AS note_title,
            n.id     AS note_id,
            1 - (nc.embedding <=> '{query_embedding_str}'::vector) AS similarity_score
        FROM notechunk nc
        INNER JOIN note n ON nc.note_id = n.id
        WHERE n.project_id = :project_id
          AND n.user_id    = :user_id
          AND nc.embedding IS NOT NULL
          AND nc.is_leaf   = true
        ORDER BY nc.embedding <=> '{query_embedding_str}'::vector
        LIMIT :limit_k
    """)

    result = session.execute(
        sql_query,
        {"project_id": project_id, "user_id": user_id, "limit_k": candidate_k},
    )

    nodes_with_scores: List[NodeWithScore] = []
    for row in result:
        metadata = dict(row.metadata_json or row.metadata_ or {})
        metadata.setdefault("note_id", row.note_id)
        metadata.setdefault("note_title", row.note_title or "Note sans titre")
        metadata.setdefault("node_id", row.node_id)
        metadata.setdefault("parent_node_id", row.parent_node_id)

        node = TextNode(
            id_=row.node_id or f"chunk-{row.id}",
            text=row.content or row.text or "",
            metadata=metadata,
        )
        nodes_with_scores.append(
            NodeWithScore(node=node, score=float(row.similarity_score))
        )

    logger.info(
        "Recherche SQL pgvector : %d nœuds leaf récupérés (candidate_k=%d)",
        len(nodes_with_scores),
        candidate_k,
    )
    return nodes_with_scores


# ---------------------------------------------------------------------------
# Résolution des parents (contexte enrichi pour le LLM)
# ---------------------------------------------------------------------------

def _build_parent_node_dict(
    session: Session, project_id: int, user_id: int
) -> Dict[str, TextNode]:
    """
    Charge les nœuds parents depuis la base pour enrichir le contexte envoyé au LLM.
    Appelé UNE SEULE FOIS par requête, après le reranking sur les leaves.
    """
    statement = (
        select(NoteChunk, Note.title)
        .join(Note, Note.id == NoteChunk.note_id)
        .where(
            Note.project_id == project_id,
            Note.user_id == user_id,
            NoteChunk.is_leaf.is_(False),
            NoteChunk.node_id.is_not(None),
        )
    )
    rows = session.exec(statement).all()

    node_dict: Dict[str, TextNode] = {}
    for chunk, note_title in rows:
        metadata = dict(chunk.metadata_json or {})
        metadata.setdefault("note_id", chunk.note_id)
        metadata.setdefault("note_title", note_title or "Note sans titre")
        metadata.setdefault("node_id", chunk.node_id)
        metadata.setdefault("parent_node_id", chunk.parent_node_id)
        node_dict[chunk.node_id] = TextNode(
            id_=chunk.node_id,
            text=chunk.content or chunk.text or "",
            metadata=metadata,
        )

    logger.info(
        "Chargé %d nœuds parents pour project_id=%d, user_id=%d",
        len(node_dict),
        project_id,
        user_id,
    )
    return node_dict


# ---------------------------------------------------------------------------
# Fallback lexical
# ---------------------------------------------------------------------------

_FALLBACK_STOPWORDS = {
    "the", "and", "for", "with", "dans", "avec", "pour", "une", "des", "les",
    "est", "sur", "pas", "plus", "que", "qui", "this", "that", "what", "how",
    "quoi", "comment", "quel", "quelle", "quels", "quelles", "from", "par",
    "sans", "mais", "donc", "car", "you", "your", "not", "are", "was", "were",
}


def _extract_query_terms(query_text: str) -> List[str]:
    terms = [t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ0-9]+", query_text or "")]
    return [t for t in terms if len(t) >= 3 and t not in _FALLBACK_STOPWORDS][:8]


def _keyword_fallback_passages(
    session: Session,
    project_id: int,
    user_id: int,
    query_text: str,
    k: int,
) -> List[Dict]:
    """
    Fallback lexical si aucun embedding n'est disponible ou si la recherche vectorielle
    ne retourne rien. Évite d'envoyer un contexte vide au LLM.
    """
    terms = _extract_query_terms(query_text)
    base_stmt = (
        select(NoteChunk, Note.title)
        .join(Note, Note.id == NoteChunk.note_id)
        .where(Note.project_id == project_id, Note.user_id == user_id)
        .order_by(NoteChunk.is_leaf.desc(), Note.updated_at.desc(), NoteChunk.chunk_index)
    )

    rows = []
    if terms:
        stmt = base_stmt.where(
            or_(*[NoteChunk.content.ilike(f"%{term}%") for term in terms])
        ).limit(max(k * 4, 12))
        rows = session.exec(stmt).all()

    if not rows:
        rows = session.exec(base_stmt.limit(max(k * 2, 8))).all()

    passages: List[Dict] = []
    seen_chunk_ids: set = set()
    for chunk, note_title in rows:
        if chunk.id in seen_chunk_ids:
            continue
        if (chunk.metadata_json or {}).get("is_image_chunk"):
            continue
        seen_chunk_ids.add(chunk.id)

        content = (chunk.content or chunk.text or "").strip()
        if not content:
            continue

        lowered = content.lower()
        match_count = sum(1 for term in terms if term in lowered) if terms else 0
        score = (match_count / max(len(terms), 1)) if terms else 0.05

        node_metadata = dict(chunk.metadata_json or {})
        node_metadata.setdefault("note_id", chunk.note_id)
        node_metadata.setdefault("note_title", note_title or "Note sans titre")
        node_metadata.setdefault("node_id", chunk.node_id or f"chunk-{chunk.id}")
        node_metadata.setdefault("chunk_index", chunk.chunk_index)
        node = TextNode(
            id_=chunk.node_id or f"chunk-{chunk.id}",
            text=content,
            metadata=node_metadata,
        )
        passages.append(_node_to_passage(node, fallback_score=score))
        if len(passages) >= k:
            break

    logger.info(
        "Fallback lexical activé: %d passages construits (terms=%s)",
        len(passages),
        terms,
    )
    return passages


# ---------------------------------------------------------------------------
# Conversion nœud → passage
# ---------------------------------------------------------------------------

def _enrich_content_with_heading_and_figure(content: str, metadata: dict) -> str:
    """
    Préfixe le contenu avec parent_heading et figure_title pour le reranker et le LLM.

    Les chunks avec titres de section et légendes descriptifs sont ainsi mieux
    priorisés par le reranker et le contexte est plus explicite pour le LLM.
    """
    parent_heading = metadata.get("parent_heading") or metadata.get("heading")
    figure_title = metadata.get("figure_title") or metadata.get("image_anchor")
    parts = []
    if parent_heading and str(parent_heading).strip():
        parts.append(f"[Section: {parent_heading.strip()}]")
    if figure_title and str(figure_title).strip():
        parts.append(str(figure_title).strip())
    if not parts:
        return content
    prefix = " ".join(parts) + "\n\n"
    return prefix + content if content else prefix.strip()


def _node_to_passage(node, fallback_score: float = 0.0) -> Dict:
    metadata = dict(getattr(node, "metadata", {}) or {})
    note_title = metadata.get("note_title", "Note sans titre")
    note_id = metadata.get("note_id")
    node_id = metadata.get("node_id")
    chunk_index = metadata.get("chunk_index", 0)
    page_no = metadata.get("page_no")
    parent_heading = metadata.get("parent_heading")
    # Multimodal : chemin image si chunk image
    image_path = metadata.get("image_path")
    image_filename = metadata.get("image_filename")
    is_image_chunk = metadata.get("is_image_chunk", False)
    caption = metadata.get("caption", "")
    
    content = node.get_content() if hasattr(node, "get_content") else str(node)
    # Enrichir avec parent_heading et figure_title pour le LLM
    content_enriched = _enrich_content_with_heading_and_figure(content, metadata)
    passage_text = f"**{note_title}**\n{content_enriched}"
    return {
        "passage": passage_text,
        "passage_raw": content,
        "note_title": note_title,
        "note_id": note_id,
        "chunk_id": node_id,
        "chunk_index": int(chunk_index) if isinstance(chunk_index, (int, str)) else 0,
        "score": float(fallback_score or 0.0),
        "page_no": int(page_no) if page_no is not None else None,
        "section": parent_heading,
        "image_path": image_path,
        "image_filename": image_filename,
        "is_image_chunk": is_image_chunk,
        "caption": caption,
    }


# ---------------------------------------------------------------------------
# Optimisations du reranking
# ---------------------------------------------------------------------------

def _filter_low_similarity_candidates(
    candidates: List[NodeWithScore],
    min_threshold: float = MIN_VECTOR_SIMILARITY_THRESHOLD,
) -> List[NodeWithScore]:
    """
    Filtre les candidats avec une similarité vectorielle trop faible avant le reranking.
    
    Évite de reranker des candidats qui ont déjà une très faible pertinence,
    ce qui réduit le temps de traitement du reranker.
    
    Args:
        candidates: Liste de NodeWithScore avec scores de similarité vectorielle
        min_threshold: Seuil minimum de similarité (défaut: 0.25)
    
    Returns:
        Liste filtrée de candidats avec similarité >= min_threshold
    """
    filtered = [c for c in candidates if float(c.score or 0.0) >= min_threshold]
    if len(filtered) < len(candidates):
        logger.debug(
            "Filtrage similarité vectorielle: %d → %d candidats (seuil=%.2f)",
            len(candidates),
            len(filtered),
            min_threshold,
        )
    return filtered


# ---------------------------------------------------------------------------
# KAG - Knowledge Graph Retrieval
# ---------------------------------------------------------------------------

def _retrieve_via_knowledge_graph(
    session: Session,
    project_id: int,
    user_id: int,
    query_text: str,
    limit: int = 10,
) -> List[NodeWithScore]:
    """
    Récupère des chunks via le graphe de connaissances KAG.
    
    1. Extrait les entités mentionnées dans la query
    2. Trouve les chunks liés à ces entités via le graphe
    3. Retourne les chunks sous forme de NodeWithScore
    """
    try:
        from app.services.kag_extraction_service import normalize_entity_name
        from app.services.kag_graph_service import get_chunks_by_entity_names
        
        query_terms = [t.strip().lower() for t in re.findall(r"[A-Za-zÀ-ÿ0-9]+", query_text)]
        query_terms = [t for t in query_terms if len(t) >= 3 and t not in _FALLBACK_STOPWORDS]
        
        if not query_terms:
            return []
        
        results = get_chunks_by_entity_names(
            session=session,
            entity_names=query_terms,
            project_id=project_id,
            user_id=user_id,
            limit=limit,
        )
        
        nodes_with_scores: List[NodeWithScore] = []
        for result in results:
            chunk = result["chunk"]
            relevance = result.get("relevance_score", 0.5)
            
            metadata = dict(chunk.metadata_json or {})
            metadata.setdefault("note_id", chunk.note_id)
            metadata.setdefault("node_id", chunk.node_id)
            metadata.setdefault("parent_node_id", chunk.parent_node_id)
            metadata["kag_matched_entity"] = result.get("entity_name", "")
            
            node = TextNode(
                id_=chunk.node_id or f"chunk-{chunk.id}",
                text=chunk.content or chunk.text or "",
                metadata=metadata,
            )
            nodes_with_scores.append(NodeWithScore(node=node, score=float(relevance)))
        
        logger.debug(
            "KAG retrieval: %d chunks via graphe (query_terms=%s)",
            len(nodes_with_scores),
            query_terms[:5],
        )
        return nodes_with_scores
        
    except Exception as e:
        logger.warning("Erreur KAG retrieval: %s", e)
        return []


def search_relevant_chunks(
    session: Session,
    project_id: int,
    query_text: str,
    user_id: int,
    k: int = 15
) -> List[Dict]:
    """
    Recherche sémantique dans les DocumentChunk du projet.
    Utilise le nouveau système de chunking intelligent basé sur Docling.
    
    Cette fonction remplace search_relevant_passages() en utilisant
    directement les chunks précalculés avec leurs embeddings.
    
    Args:
        session: Session SQLModel
        project_id: ID du projet
        query_text: Texte de la requête
        user_id: ID de l'utilisateur (pour vérification de sécurité)
        k: Nombre de chunks à retourner
        
    Returns:
        Liste de dictionnaires contenant le chunk, la note source et le score
    """
    project = get_project_by_id(session, project_id, user_id)
    if not project:
        logger.warning(
            "Projet %d non trouvé ou n'appartient pas à l'utilisateur %d",
            project_id,
            user_id,
        )
        return []

    if not query_text or not query_text.strip():
        logger.warning("Requête vide fournie")
        return []

    try:
        # Générer l'embedding de la requête
        query_embedding = generate_embedding(query_text)
        if query_embedding is None:
            logger.warning("Impossible de générer l'embedding de la requête")
            return []
        
        from sqlalchemy import text
        
        # Convertir l'embedding en format PostgreSQL array
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Rechercher dans les DocumentChunk via join avec Note pour filtrer par projet
        connection = session.connection()
        
        sql_query = f"""
            SELECT 
                dc.id,
                dc.note_id,
                dc.chunk_index,
                dc.content,
                dc.chunk_type,
                dc.page_number,
                dc.section_title,
                dc.start_char,
                dc.end_char,
                dc.created_at,
                n.title as note_title,
                n.note_type as note_type,
                n.source_file_type as source_file_type,
                1 - (dc.embedding <=> '{embedding_str}'::vector) as similarity_score
            FROM document_chunk dc
            INNER JOIN note n ON dc.note_id = n.id
            WHERE n.project_id = {project_id}
                AND n.user_id = {user_id}
                AND dc.embedding IS NOT NULL
            ORDER BY dc.embedding <=> '{embedding_str}'::vector
            LIMIT {k}
        """
        
        result = connection.execute(text(sql_query))
        
        # Convertir les résultats
        chunks_results = []
        for row in result:
            chunk_dict = {
                'chunk_id': row.id,
                'note_id': row.note_id,
                'note_title': row.note_title,
                'note_type': row.note_type,
                'source_file_type': row.source_file_type,
                'content': row.content,
                'chunk_type': row.chunk_type,
                'page_number': row.page_number,
                'section_title': row.section_title,
                'score': float(row.similarity_score)
            }
            chunks_results.append(chunk_dict)
        
        top_scores = [f"{c['score']:.3f}" for c in chunks_results[:3]]
        logger.info(f"✅ Trouvé {len(chunks_results)} chunks pertinents (scores: {top_scores})")
        return chunks_results
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche dans les chunks: {e}", exc_info=True)
        return []


def search_relevant_passages(
    session: Session,
    project_id: int,
    query_text: str,
    user_id: int,
    k: int = 15,
    passage_size: int = 500
) -> List[Dict]:
    """
    DEPRECATED: Utiliser search_relevant_chunks() à la place.
    
    Recherche sémantique avancée qui cherche dans TOUTES les notes du projet
    et retourne les PASSAGES les plus pertinents (pas les notes complètes).
    
    Cette fonction est conservée pour compatibilité mais redirige vers
    search_relevant_chunks() qui utilise le nouveau système.
    
    Args:
        session: Session SQLModel
        project_id: ID du projet
        query_text: Texte de la requête
        user_id: ID de l'utilisateur (pour vérification de sécurité)
        k: Nombre de passages à retourner
        passage_size: Taille approximative des passages en caractères (ignoré)
        
    Returns:
        Liste de dictionnaires contenant le passage, la note source et le score
    """
    # Rediriger vers la nouvelle fonction
    chunks = search_relevant_chunks(session, project_id, query_text, user_id, k)
    
    # Convertir au format attendu par l'ancien système
    passages = []
    for chunk in chunks:
        # Formater le contenu en markdown si nécessaire
        passage_content = chunk['content']
        
        # Ajouter les métadonnées au passage
        meta_parts = []
        if chunk.get('page_number'):
            meta_parts.append(f"Page {chunk['page_number']}")
        if chunk.get('section_title'):
            meta_parts.append(f"Section: {chunk['section_title']}")
        if chunk.get('chunk_type') and chunk['chunk_type'] != 'text':
            meta_parts.append(f"Type: {chunk['chunk_type']}")
        
        if meta_parts:
            passage_content = f"*[{', '.join(meta_parts)}]*\n\n{passage_content}"
        
        passages.append({
            'passage': f"**{chunk['note_title']}**\n{passage_content}",
            'note_title': chunk['note_title'],
            'note_id': chunk['note_id'],
            'score': chunk['score']
        })
    
    return passages

    if not query_text or not query_text.strip():
        logger.warning("Requête vide fournie")
        return []

    try:
        passages = search_relevant_passages(
            session=session,
            project_id=project_id,
            query_text=query_text,
            user_id=user_id,
            k=k,
        )
        note_ids = [p["note_id"] for p in passages if p.get("note_id")]
        if not note_ids:
            return []

        note_stmt = select(Note).where(Note.id.in_(note_ids))
        notes = {note.id: note for note in session.exec(note_stmt).all()}

        results = []
        for passage in passages:
            note_id = passage.get("note_id")
            note = notes.get(note_id)
            if note:
                results.append(
                    {"note": note, "score": float(passage.get("score", 0.0))}
                )
        return results

    except Exception as e:
        logger.error("Erreur lors de la recherche sémantique: %s", e, exc_info=True)
        return []
