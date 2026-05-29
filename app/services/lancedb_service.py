import os
import logging
from typing import List, Optional, Dict, Any
import lancedb
from lancedb.pydantic import LanceModel, Vector
from app.config import settings

logger = logging.getLogger(__name__)

# Define schema for ColPali page patches
class LanceColPaliPatch(LanceModel):
    id: str  # Format: "chunkid_patchindex"
    chunk_id: int
    document_id: int
    patch_index: int
    vector: Vector(128)  # ColPali embeddings are 128-dimensional

_db = None
_colpali_table = None

def get_lancedb_client():
    global _db, _colpali_table
    if _db is None:
        db_dir = os.path.abspath(getattr(settings, "LANCED_DB_DIR", "./data/lancedb"))
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Connecting to LanceDB at: {db_dir}")
        _db = lancedb.connect(db_dir)
        
        # Initialize ColPali patches table
        colpali_table_name = "colpali_patches"
        if colpali_table_name in _db.table_names():
            _colpali_table = _db.open_table(colpali_table_name)
        else:
            _colpali_table = _db.create_table(colpali_table_name, schema=LanceColPaliPatch)
            logger.info("Created LanceDB table 'colpali_patches'")
            
    return _db

def get_colpali_table():
    get_lancedb_client()
    global _colpali_table
    if _colpali_table is None and _db is not None:
        _colpali_table = _db.open_table("colpali_patches")
    return _colpali_table

def insert_colpali_patches_lancedb(document_id: int, chunk_id: int, patch_vectors: List[List[float]]):
    """
    Inserts ColPali patches for a specific chunk.
    patch_vectors: List of 128-dimensional patch embeddings for a single page.
    """
    try:
        table = get_colpali_table()
        # Delete existing patches for this chunk to prevent duplicates
        table.delete(f"chunk_id = {chunk_id}")
        
        patches_data = [
            {
                "id": f"{chunk_id}_{idx}",
                "chunk_id": chunk_id,
                "document_id": document_id,
                "patch_index": idx,
                "vector": vec
            }
            for idx, vec in enumerate(patch_vectors)
        ]
        table.add(patches_data)
        logger.info(f"Added {len(patches_data)} ColPali patches for chunk_id={chunk_id}")
    except Exception as e:
        logger.error(f"Error writing ColPali patches to LanceDB: {e}", exc_info=True)

def delete_chunks_lancedb(document_id: int):
    """Deletes all ColPali patches for a document from LanceDB."""
    try:
        colpali_table = get_colpali_table()
        colpali_table.delete(f"document_id = {document_id}")
        logger.info(f"Deleted all ColPali vector data for document_id={document_id} from LanceDB")
    except Exception as e:
        logger.error(f"Error deleting from LanceDB for document_id={document_id}: {e}", exc_info=True)

def delete_single_chunk_lancedb(chunk_id: int):
    """Deletes ColPali patches for a specific chunk ID."""
    try:
        colpali_table = get_colpali_table()
        colpali_table.delete(f"chunk_id = {chunk_id}")
        logger.info(f"Deleted ColPali patches for chunk_id={chunk_id} from LanceDB")
    except Exception as e:
        logger.error(f"Error deleting chunk_id={chunk_id} from LanceDB: {e}", exc_info=True)

def search_colpali_lancedb(query_token_embeddings: List[List[float]], document_ids: List[int], limit: int) -> List[Dict[str, Any]]:
    """
    Performs late interaction (MaxSim) search on ColPali patches table.
    query_token_embeddings: List of token embeddings [num_tokens, 128]
    """
    if not document_ids or not query_token_embeddings:
        return []
    try:
        table = get_colpali_table()
        doc_ids_str = ",".join(map(str, document_ids))
        filter_str = f"document_id in ({doc_ids_str})"
        
        # 1. Search the top matching patches for each query token vector
        candidate_patches = []
        for token_idx, token_vec in enumerate(query_token_embeddings):
            res = table.search(token_vec).metric("cosine").where(filter_str).limit(100).to_list()
            for r in res:
                r["query_token_index"] = token_idx
                candidate_patches.append(r)
                
        if not candidate_patches:
            return []
            
        # 2. Compute MaxSim per chunk: sum_{query_token} max_{patch} similarity(query_token, patch)
        # Cosine similarity = 1.0 - _distance
        chunk_scores: Dict[int, Dict[int, float]] = {}
        for patch in candidate_patches:
            chunk_id = int(patch["chunk_id"])
            token_idx = patch["query_token_index"]
            sim = 1.0 - float(patch["_distance"])
            
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {}
            if token_idx not in chunk_scores[chunk_id] or sim > chunk_scores[chunk_id][token_idx]:
                chunk_scores[chunk_id][token_idx] = sim
                
        # 3. Sum up the maximum similarities and format as list of dicts with calculated distance
        final_results = []
        num_query_tokens = len(query_token_embeddings)
        for chunk_id, token_sims in chunk_scores.items():
            maxsim_sum = sum(token_sims.values())
            # Normalize to [0, 1] similarity
            avg_similarity = maxsim_sum / num_query_tokens
            # Convert back to a distance format for compatibility
            distance = 1.0 - avg_similarity
            final_results.append({
                "id": chunk_id,
                "document_id": next((p["document_id"] for p in candidate_patches if p["chunk_id"] == chunk_id), None),
                "_distance": distance
            })
            
        final_results.sort(key=lambda x: x["_distance"])
        return final_results[:limit]
    except Exception as e:
        logger.error(f"Error executing ColPali MaxSim search in LanceDB: {e}", exc_info=True)
        return []
