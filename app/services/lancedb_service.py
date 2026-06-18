import os
import logging
from typing import List, Optional, Dict, Any
import lancedb
import numpy as np
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
    global _db
    if _db is not None:
        return _db.open_table("colpali_patches")
    return None

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
        
        # Try to build/update IVF_SQ index to optimize storage/search (Option A)
        try:
            table.create_index(
                vector_column_name="vector",
                index_type="IVF_SQ",
                metric="cosine"
            )
            logger.info("Successfully updated IVF_SQ index on 'colpali_patches'")
        except Exception as idx_err:
            # Silence this error because LanceDB requires a minimum number of vectors
            # to train the IVF partitions (e.g. at least 1,000 or 10,000 vectors).
            logger.debug(f"Could not build IVF_SQ index yet (normal if database is small): {idx_err}")
    except Exception as e:
        logger.error(f"Error writing ColPali patches to LanceDB: {e}", exc_info=True)

def insert_colpali_patches_batch_lancedb(document_id: int, chunk_patches_list: List[tuple[int, List[List[float]]]]):
    """
    Inserts ColPali patches in batch for all chunks of a document.
    chunk_patches_list: List of tuples (chunk_id, patch_vectors)
    """
    try:
        table = get_colpali_table()
        # Delete existing patches for this document to prevent duplicates
        table.delete(f"document_id = {document_id}")
        
        patches_data = []
        for chunk_id, patch_vectors in chunk_patches_list:
            for idx, vec in enumerate(patch_vectors):
                patches_data.append({
                    "id": f"{chunk_id}_{idx}",
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "patch_index": idx,
                    "vector": vec
                })
        
        if patches_data:
            table.add(patches_data)
            logger.info(f"Added {len(patches_data)} ColPali patches in batch for document_id={document_id}")
            
            # Try to build/update IVF_SQ index once for all patches
            try:
                table.create_index(
                    vector_column_name="vector",
                    index_type="IVF_SQ",
                    metric="cosine"
                )
                logger.info("Successfully updated IVF_SQ index on 'colpali_patches'")
            except Exception as idx_err:
                logger.debug(f"Could not build IVF_SQ index yet (normal if database is small): {idx_err}")
    except Exception as e:
        logger.error(f"Error writing ColPali patches batch to LanceDB: {e}", exc_info=True)

def delete_chunks_lancedb(document_id: int):
    """Deletes all ColPali patches for a document from LanceDB."""
    try:
        colpali_table = get_colpali_table()
        colpali_table.delete(f"document_id = {document_id}")
        logger.info(f"Deleted all ColPali vector data for document_id={document_id} from LanceDB")
    except Exception as e:
        logger.error(f"Error deleting from LanceDB for document_id={document_id}: {e}", exc_info=True)


def delete_colpali_patches_for_document(document_id: int) -> None:
    """Alias explicite pour le pipeline document_indexing_service (mode full)."""
    delete_chunks_lancedb(document_id)

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
    logger.info(
        "[search_colpali_lancedb] Starting MaxSim search for %d query tokens, limiting to %d documents (ids: %s)",
        len(query_token_embeddings) if query_token_embeddings else 0,
        len(document_ids) if document_ids else 0,
        document_ids,
    )
    if not document_ids or not query_token_embeddings:
        logger.warning("[search_colpali_lancedb] Missing document_ids or query_token_embeddings, aborting.")
        return []
    try:
        from concurrent.futures import ThreadPoolExecutor
        
        table = get_colpali_table()
        doc_ids_str = ",".join(map(str, document_ids))
        filter_str = f"document_id in ({doc_ids_str})"
        
        # 1. Quick check of total patch count by selecting only chunk_id (no vector data loaded)
        quick_res = table.search().where(filter_str).select(["chunk_id"]).to_list()
        total_patches = len(quick_res)
        logger.info("[search_colpali_lancedb] Total patches in space for these documents: %d", total_patches)
        
        # 2. Retrieve patches based on scale
        # 150000 patches is approximately 150 pages.
        tbl = None
        if total_patches <= 150000:
            logger.info("[search_colpali_lancedb] Small/medium scale space (<= 150 pages). Performing exact MaxSim on all pages.")
            tbl = table.search().where(filter_str).select(["chunk_id", "document_id", "vector"]).to_arrow()
        else:
            logger.info("[search_colpali_lancedb] Large scale space (> 150 pages). Performing token-level candidate retrieval (limit=250 per token).")
            candidate_chunk_ids = set()
            
            def search_token(token_vec):
                try:
                    res = table.search(token_vec).metric("cosine").where(filter_str).select(["chunk_id", "_distance"]).limit(250).to_list()
                    return [int(r["chunk_id"]) for r in res]
                except Exception as ex:
                    logger.warning("Error searching token in LanceDB: %s", ex)
                    return []
            
            max_workers = min(16, len(query_token_embeddings))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                search_results = executor.map(search_token, query_token_embeddings)
                for chunk_ids in search_results:
                    candidate_chunk_ids.update(chunk_ids)
            
            logger.info("[search_colpali_lancedb] Found %d unique candidate pages. Fetching patch vectors.", len(candidate_chunk_ids))
            if candidate_chunk_ids:
                candidate_ids_str = ",".join(map(str, candidate_chunk_ids))
                patch_filter = f"chunk_id in ({candidate_ids_str})"
                tbl = table.search().where(patch_filter).select(["chunk_id", "document_id", "vector"]).to_arrow()
                
        if tbl is None or len(tbl) == 0:
            logger.info("[search_colpali_lancedb] No patches found.")
            return []
            
        logger.info("[search_colpali_lancedb] Loaded %d total patch vectors for MaxSim calculation.", len(tbl))
        
        # Extract columns to numpy arrays using zero-copy (or direct copies) to bypass Python list/dict conversion
        chunk_ids = tbl["chunk_id"].to_numpy(zero_copy_only=False)
        doc_ids = tbl["document_id"].to_numpy(zero_copy_only=False)
        
        try:
            # Flatten/reshape the fixed-size list array of shape (N, 128)
            combined_vectors = tbl["vector"].combine_chunks()
            flat_values = combined_vectors.values.to_numpy(zero_copy_only=False)
            vectors_numpy = flat_values.reshape(-1, 128)
        except Exception as pyarrow_err:
            logger.warning("Error converting Arrow vectors directly: %s. Falling back to slow list conversion.", pyarrow_err)
            vectors_numpy = np.array(tbl["vector"].to_pylist(), dtype=np.float32)
            
        # 3. Group patches by chunk_id
        from collections import defaultdict
        chunk_to_indices = defaultdict(list)
        chunk_to_doc = {}
        for idx, (c_id, d_id) in enumerate(zip(chunk_ids, doc_ids)):
            c_id = int(c_id)
            chunk_to_indices[c_id].append(idx)
            if c_id not in chunk_to_doc:
                chunk_to_doc[c_id] = int(d_id)
            
        # 4. Compute MaxSim per page using NumPy
        Q = np.array(query_token_embeddings, dtype=np.float32)  # (T, 128)
        Q_norms = np.linalg.norm(Q, axis=1, keepdims=True)
        Q_norms = np.where(Q_norms == 0, 1.0, Q_norms)
        Q = Q / Q_norms
        
        final_results = []
        num_tokens = len(query_token_embeddings)
        
        for chunk_id, indices in chunk_to_indices.items():
            if not indices:
                continue
            P = vectors_numpy[indices]  # shape (P, 128)
            P_norms = np.linalg.norm(P, axis=1, keepdims=True)
            P_norms = np.where(P_norms == 0, 1.0, P_norms)
            P = P / P_norms
            
            # Cosine similarity matrix: shape (T, P)
            S = np.dot(Q, P.T)
            # Max similarity for each query token: shape (T,)
            max_sims = np.max(S, axis=1)
            # Sum of max similarities
            maxsim_sum = float(np.sum(max_sims))
            
            # Convert score to distance for backward compatibility.
            # MaxSim sum range: [0, T]
            # Average similarity: MaxSim_sum / T
            # Distance = 1.0 - (MaxSim_sum / T)
            avg_similarity = maxsim_sum / max(num_tokens, 1)
            distance = 1.0 - avg_similarity
            
            final_results.append({
                "id": chunk_id,
                "document_id": chunk_to_doc[chunk_id],
                "_distance": distance,
                "maxsim_score": maxsim_sum
            })
            
        final_results.sort(key=lambda x: x["_distance"])
        logger.info(
            "[search_colpali_lancedb] Sorted results. Top 5 match distances: %s",
            [round(r["_distance"], 4) for r in final_results[:5]],
        )
        return final_results[:limit]
    except Exception as e:
        logger.error(f"Error executing ColPali MaxSim search in LanceDB: {e}", exc_info=True)
        return []
