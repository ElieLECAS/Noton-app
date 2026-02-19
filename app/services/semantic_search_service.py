from typing import Dict, List
import os
import re
from sqlmodel import Session, select
from sqlalchemy import or_
from app.models.note import Note
from app.models.note_chunk import NoteChunk
from app.services.project_service import get_project_by_id
import logging
from sqlalchemy.engine import make_url
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings as LlamaSettings
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import QueryBundle, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from app.embedding_config import EMBEDDING_DIMENSION
from app.config import settings

logger = logging.getLogger(__name__)

try:
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
except ImportError:
    from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

try:
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("FlagEmbeddingReranker non disponible, reranking désactivé")

# Modèle reranker : Cross-Encoder v2-m3 (~568M), meilleure précision sur sujets techniques
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
# Nombre de candidats récupérés avant reranking (>k pour que le reranker ait de la matière)
RERANKER_CANDIDATE_MULTIPLIER = 3
_FALLBACK_STOPWORDS = {
    "the", "and", "for", "with", "dans", "avec", "pour", "une", "des", "les",
    "est", "sur", "pas", "plus", "que", "qui", "this", "that", "what", "how",
    "quoi", "comment", "quel", "quelle", "quels", "quelles", "from", "par",
    "sans", "mais", "donc", "car", "you", "your", "not", "are", "was", "were",
}


def _get_embed_model() -> HuggingFaceEmbedding:
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    return HuggingFaceEmbedding(
        model_name=model_name,
        device=device,
        embed_batch_size=settings.EMBEDDING_BATCH_SIZE,
    )


def _build_pgvector_store() -> PGVectorStore:
    url = make_url(settings.DATABASE_URL)
    base_kwargs = {
        "database": url.database,
        "host": url.host or "localhost",
        "port": int(url.port or 5432),
        "user": url.username or "",
        "password": url.password or "",
        "table_name": "notechunk",
        "embed_dim": EMBEDDING_DIMENSION,
    }

    # Compatibilité multi-versions de llama-index-vector-stores-postgres.
    candidate_kwargs = [
        {"text_column": "text", "metadata_column": "metadata_"},
        {"text_column": "content", "metadata_column": "metadata_json"},
        {},
    ]
    last_error: Exception | None = None
    for extra_kwargs in candidate_kwargs:
        try:
            return PGVectorStore.from_params(**base_kwargs, **extra_kwargs)
        except TypeError as exc:
            last_error = exc
            continue

    if last_error:
        raise last_error
    return PGVectorStore.from_params(**base_kwargs)


def _build_parent_node_dict(session: Session, project_id: int, user_id: int) -> Dict[str, TextNode]:
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
    return node_dict


def _get_recursive_retriever(
    session: Session,
    project_id: int,
    user_id: int,
    k: int,
    leaf_filter_value: str | bool | None = "true",
) -> RecursiveRetriever:
    vector_store = _build_pgvector_store()
    embed_model = _get_embed_model()
    LlamaSettings.embed_model = embed_model

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    filters_list = [
        MetadataFilter(key="project_id", value=project_id),
        MetadataFilter(key="user_id", value=user_id),
    ]
    if leaf_filter_value is not None:
        filters_list.append(MetadataFilter(key="is_leaf", value=leaf_filter_value))
    filters = MetadataFilters(filters=filters_list)
    vector_retriever = index.as_retriever(similarity_top_k=k, filters=filters)
    parent_node_dict = _build_parent_node_dict(session, project_id, user_id)

    return RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        node_dict=parent_node_dict,
        verbose=False,
    )


def _retrieve_with_filter_fallbacks(
    session: Session,
    project_id: int,
    user_id: int,
    query_text: str,
    candidate_k: int,
):
    """
    Tente plusieurs variantes de filtre metadata pour éviter les 0 résultats
    dus aux différences de typage (`"true"` vs `True`) selon versions/librairies.
    """
    for leaf_filter_value in ("true", True, None):
        recursive_retriever = _get_recursive_retriever(
            session=session,
            project_id=project_id,
            user_id=user_id,
            k=candidate_k,
            leaf_filter_value=leaf_filter_value,
        )
        retrieved = recursive_retriever.retrieve(query_text)
        logger.info(
            "Base retriever (pgvector) a retourné %s nœuds pour k=%s (is_leaf=%r)",
            len(retrieved),
            candidate_k,
            leaf_filter_value,
        )
        if retrieved:
            return retrieved
    return []


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
    Fallback robuste si le retriever vectoriel ne retourne rien:
    on fait une recherche lexicale simple pour éviter d'envoyer un contexte vide au LLM.
    """
    terms = _extract_query_terms(query_text)
    base_stmt = (
        select(NoteChunk, Note.title)
        .join(Note, Note.id == NoteChunk.note_id)
        .where(
            Note.project_id == project_id,
            Note.user_id == user_id,
        )
        .order_by(NoteChunk.is_leaf.desc(), Note.updated_at.desc(), NoteChunk.chunk_index)
    )

    rows = []
    if terms:
        stmt = base_stmt.where(or_(*[NoteChunk.content.ilike(f"%{term}%") for term in terms])).limit(max(k * 4, 12))
        rows = session.exec(stmt).all()

    if not rows:
        rows = session.exec(base_stmt.limit(max(k * 2, 8))).all()

    passages: List[Dict] = []
    seen_chunk_ids: set[int] = set()
    for chunk, note_title in rows:
        if chunk.id in seen_chunk_ids:
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
        "Fallback lexical activé: %s passages construits (terms=%s)",
        len(passages),
        terms,
    )
    return passages


def _node_to_passage(node, fallback_score: float = 0.0) -> Dict:
    metadata = dict(getattr(node, "metadata", {}) or {})
    note_title = metadata.get("note_title", "Note sans titre")
    note_id = metadata.get("note_id")
    node_id = metadata.get("node_id")
    chunk_index = metadata.get("chunk_index", 0)
    content = node.get_content() if hasattr(node, "get_content") else str(node)
    passage_text = f"**{note_title}**\n{content}"
    return {
        "passage": passage_text,
        "note_title": note_title,
        "note_id": note_id,
        "chunk_id": node_id,
        "chunk_index": int(chunk_index) if isinstance(chunk_index, int | str) else 0,
        "score": float(fallback_score or 0.0),
    }


def search_relevant_notes(
    session: Session,
    project_id: int,
    query_text: str,
    user_id: int,
    k: int = 10
) -> List[Dict]:
    """
    Recherche sémantique RAG sur les chunks de notes.
    Retourne les notes les plus pertinentes par rapport à la requête, basées sur leurs chunks.
    
    Args:
        session: Session SQLModel
        project_id: ID du projet
        query_text: Texte de la requête
        user_id: ID de l'utilisateur (pour vérification de sécurité)
        k: Nombre de notes à retourner
        
    Returns:
        Liste de dictionnaires contenant la note et son score de similarité
    """
    # Vérifier que le projet appartient à l'utilisateur
    project = get_project_by_id(session, project_id, user_id)
    if not project:
        logger.warning(f"Projet {project_id} non trouvé ou n'appartient pas à l'utilisateur {user_id}")
        return []
    
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
                results.append({"note": note, "score": float(passage.get("score", 0.0))})
        return results
    except Exception as e:
        logger.error(f"Erreur lors de la recherche sémantique: {e}", exc_info=True)
        return []


def search_relevant_passages(
    session: Session,
    project_id: int,
    query_text: str,
    user_id: int,
    k: int = 8,
    passage_size: int = 500
) -> List[Dict]:
    """
    Recherche sémantique avancée qui cherche directement dans les CHUNKS des notes
    et retourne les chunks les plus pertinents (passages).
    
    Cette approche utilise le nouveau système RAG basé sur les chunks :
    - Recherche directement dans les chunks avec embeddings
    - Plus précis car chaque chunk a son propre embedding
    - Respecte la structure markdown (paragraphes, tableaux)
    - Plus efficace que l'ancien système
    
    Args:
        session: Session SQLModel
        project_id: ID du projet
        query_text: Texte de la requête
        user_id: ID de l'utilisateur (pour vérification de sécurité)
        k: Nombre de passages (chunks) à retourner
        passage_size: Ignoré (les chunks ont déjà une taille optimale)
    
    Returns:
        Liste de dictionnaires contenant le passage (chunk), la note source et le score
    """
    # Vérifier que le projet appartient à l'utilisateur
    project = get_project_by_id(session, project_id, user_id)
    if not project:
        logger.warning(f"Projet {project_id} non trouvé ou n'appartient pas à l'utilisateur {user_id}")
        return []
    
    if not query_text or not query_text.strip():
        logger.warning("Requête vide fournie")
        return []
    
    try:
        # Récupérer plus de candidats si le reranker est actif
        candidate_k = k * RERANKER_CANDIDATE_MULTIPLIER if RERANKER_AVAILABLE else k
        retrieved = _retrieve_with_filter_fallbacks(
            session=session,
            project_id=project_id,
            user_id=user_id,
            query_text=query_text,
            candidate_k=candidate_k,
        )

        parent_node_dict = _build_parent_node_dict(session, project_id, user_id)

        # Remonter au nœud parent pour chaque résultat (contexte enrichi)
        candidate_nodes = []
        seen_node_ids: set = set()
        for node_with_score in retrieved:
            score = float(getattr(node_with_score, "score", 0.0) or 0.0)
            node = node_with_score.node
            metadata = dict(getattr(node, "metadata", {}) or {})
            parent_node_id = metadata.get("parent_node_id")

            target_node = parent_node_dict.get(parent_node_id) if parent_node_id else None
            if target_node is None:
                target_node = node

            node_identifier = getattr(target_node, "node_id", None) or metadata.get("node_id")
            if node_identifier and node_identifier in seen_node_ids:
                continue
            if node_identifier:
                seen_node_ids.add(node_identifier)

            # Conserver le score pour pouvoir reranker
            from llama_index.core.schema import NodeWithScore
            candidate_nodes.append(NodeWithScore(node=target_node, score=score))

        logger.info(
            "Après résolution parent: %s candidats uniques",
            len(candidate_nodes),
        )

        # Appliquer le reranker sur les nœuds parents (contexte riche)
        if RERANKER_AVAILABLE and candidate_nodes:
            try:
                reranker = FlagEmbeddingReranker(model=RERANKER_MODEL, top_n=k)
                reranked = reranker.postprocess_nodes(
                    candidate_nodes,
                    query_bundle=QueryBundle(query_str=query_text),
                )
                logger.info("Reranking appliqué: %s → %s candidats", len(candidate_nodes), len(reranked))
                final_nodes = reranked
            except Exception as rerank_err:
                logger.warning("Reranking échoué, fallback sur ordre vectoriel: %s", rerank_err)
                final_nodes = candidate_nodes[:k]
        else:
            final_nodes = candidate_nodes[:k]

        passages: List[Dict] = []
        for node_with_score in final_nodes:
            score = float(getattr(node_with_score, "score", 0.0) or 0.0)
            passages.append(_node_to_passage(node_with_score.node, fallback_score=score))

        score_strs = [f"{p['score']:.3f}" for p in passages[:3]]
        logger.info(
            "Trouvé %s passages pertinents%s (scores: %s...)",
            len(passages),
            " [reranked]" if RERANKER_AVAILABLE else "",
            score_strs,
        )
        if not passages:
            return _keyword_fallback_passages(
                session=session,
                project_id=project_id,
                user_id=user_id,
                query_text=query_text,
                k=k,
            )
        return passages
    except Exception as e:
        logger.error(f"Erreur lors de la recherche de passages: {e}", exc_info=True)
        return []

