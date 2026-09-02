import os
import logging
import time
from typing import Any, Dict, List, Optional, Set
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

def _ensure_vector_index(table) -> None:
    """Crée l'index IVF_SQ UNE SEULE FOIS, jamais à chaque insertion.

    ``create_index`` reconstruit l'index sur la table ENTIÈRE : mesuré à 87 s pour
    1,6 M de vecteurs. L'appeler à chaque document rendait le coût d'une insertion
    proportionnel à tout le corpus (et une passe de réparation sur 15 documents
    passait 20 min à ne rebâtir que des index), en plus de provoquer des conflits de
    commit entre transactions CreateIndex concurrentes.

    Les lignes ajoutées après coup restent interrogeables (Lance balaie le fragment
    non indexé) ; c'est ``optimize_colpali_index`` qui les fait rejoindre l'index,
    à appeler une fois en fin de passe et non par document.
    """
    try:
        if table.list_indices():
            return
    except Exception as exc:  # API absente selon la version : on tente la création
        logger.debug(f"list_indices indisponible ({exc}) — tentative de création directe")
    try:
        table.create_index(
            vector_column_name="vector",
            index_type="IVF_SQ",
            metric="cosine",
        )
        logger.info("Index IVF_SQ créé sur 'colpali_patches'")
    except Exception as idx_err:
        # LanceDB exige un minimum de vecteurs pour entraîner les partitions IVF.
        logger.debug(f"Index IVF_SQ pas encore constructible (normal si la base est petite): {idx_err}")


def optimize_colpali_index() -> dict:
    """Intègre à l'index les vecteurs écrits depuis sa création (maintenance).

    À lancer UNE fois après une passe d'écriture en masse (réparation globale,
    réindexation de corpus), jamais par document.
    """
    try:
        table = get_colpali_table()
        before = 0
        try:
            for idx in table.list_indices():
                before += int(getattr(idx, "num_unindexed_rows", 0) or 0)
        except Exception:
            pass
        t0 = time.perf_counter()
        table.optimize()
        elapsed = time.perf_counter() - t0
        logger.info(
            "[LanceDB] optimize() terminé en %.1fs (%d ligne(s) non indexée(s) avant)",
            elapsed,
            before,
        )
        return {"status": "ok", "seconds": round(elapsed, 1), "unindexed_before": before}
    except Exception as exc:
        logger.error(f"Error optimizing ColPali index: {exc}", exc_info=True)
        return {"status": "error", "reason": str(exc)}


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
            # Invariant posé À L'ÉCRITURE : les vecteurs stockés sont L2-normalisés.
            # ColQwen2 les sort déjà ainsi (vérifié : 400 000 vecteurs, norme 1,000000,
            # écart-type 4e-8), mais l'affirmer ici coûte une passe vectorisée par
            # document et permet à la recherche de ne PLUS renormaliser à chaque requête.
            arr = np.asarray(patch_vectors, dtype=np.float32)
            if arr.ndim != 2:
                logger.warning(
                    "Patches de forme inattendue pour chunk_id=%s (%s) — ignorés",
                    chunk_id,
                    arr.shape,
                )
                continue
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.where(norms == 0, 1.0, norms)
            for idx, vec in enumerate(arr):
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
            _ensure_vector_index(table)
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

def get_colpali_document_ids() -> Optional[List[int]]:
    """Ids de documents distincts présents dans la table colpali_patches.

    Sert à la réparation de topologie en masse : on ne visite que les documents
    qui ont réellement des patches. None si le scan LanceDB échoue.
    """
    try:
        table = get_colpali_table()
        tbl = table.search().select(["document_id"]).to_arrow()
        if not tbl.num_rows:
            return []
        ids = np.unique(tbl["document_id"].to_numpy(zero_copy_only=False))
        return [int(d) for d in ids.tolist()]
    except Exception as e:
        logger.error(f"Error scanning ColPali document ids from LanceDB: {e}", exc_info=True)
        return None


def fetch_colpali_patch_vectors_for_chunks(
    chunk_ids: List[int],
) -> Dict[int, "np.ndarray"]:
    """Vecteurs de patches par chunk_id (tableau ``(n_patches, 128)``), triés par patch_index.

    Utilisé par la réparation de topologie : on relit le jeu de patches d'UN chunk
    représentatif par page pour le ré-attacher à l'anchor de la page, sans repasser
    par le modèle. Les vecteurs restent en NumPy de bout en bout — les convertir en
    listes Python coûtait 15 s sur un gros document (des millions d'objets flottants)
    pour être aussitôt reconverties en Arrow à l'écriture.
    """
    if not chunk_ids:
        return {}
    try:
        table = get_colpali_table()
        ids_str = ",".join(map(str, chunk_ids))
        tbl = (
            table.search()
            .where(f"chunk_id in ({ids_str})")
            .select(["chunk_id", "patch_index", "vector"])
            .to_arrow()
        )
        if not tbl.num_rows:
            return {}

        c_ids = tbl["chunk_id"].to_numpy(zero_copy_only=False)
        p_idx = tbl["patch_index"].to_numpy(zero_copy_only=False)
        try:
            flat = tbl["vector"].combine_chunks().values.to_numpy(zero_copy_only=False)
            vectors = flat.reshape(-1, 128)
        except Exception as arrow_err:
            logger.warning(
                "fetch_colpali_patch_vectors_for_chunks: conversion Arrow directe "
                "impossible (%s), repli lent to_pylist.",
                arrow_err,
            )
            vectors = np.array(tbl["vector"].to_pylist(), dtype=np.float32)

        by_chunk: Dict[int, List[tuple]] = {}
        for row_i, (c_id, patch_i) in enumerate(zip(c_ids, p_idx)):
            by_chunk.setdefault(int(c_id), []).append((int(patch_i), row_i))

        result: Dict[int, np.ndarray] = {}
        for c_id, entries in by_chunk.items():
            entries.sort(key=lambda t: t[0])
            order = np.fromiter((row_i for _, row_i in entries), dtype=np.int64, count=len(entries))
            result[c_id] = vectors[order]
        return result
    except Exception as e:
        logger.error(
            f"Error fetching ColPali patch vectors for chunks: {e}", exc_info=True
        )
        return {}


def get_colpali_chunk_ids_by_document(document_ids: List[int]) -> Dict[int, Optional[set]]:
    """Ids de chunks distincts présents dans LanceDB, par document (audit de sync).

    Retourne {document_id: set(chunk_ids)} — set vide si aucun patch. En cas
    d'échec du scan LanceDB, la valeur est None (état inconnu, à distinguer
    de « aucun patch »).
    """
    result: Dict[int, Optional[set]] = {int(d): set() for d in document_ids}
    if not document_ids:
        return result
    try:
        table = get_colpali_table()
        doc_ids_str = ",".join(map(str, document_ids))
        tbl = (
            table.search()
            .where(f"document_id in ({doc_ids_str})")
            .select(["chunk_id", "document_id"])
            .to_arrow()
        )
        if tbl.num_rows:
            # Dédoublonnage sur une clé composite 1D plutôt qu'un np.unique lexicographique
            # sur une matrice (N, 2) : même résultat, une seule passe de tri sur un
            # tableau contigu — le scan portant sur des millions de patches, l'écart
            # se compte en secondes par appel du tableau de bord.
            doc_col = tbl["document_id"].to_numpy(zero_copy_only=False).astype(np.int64)
            chunk_col = tbl["chunk_id"].to_numpy(zero_copy_only=False).astype(np.int64)
            keys = np.unique((doc_col << 32) | (chunk_col & 0xFFFFFFFF))
            for key in keys.tolist():
                result.setdefault(int(key >> 32), set()).add(int(key & 0xFFFFFFFF))
        return result
    except Exception as e:
        logger.error(f"Error scanning ColPali chunk ids from LanceDB: {e}", exc_info=True)
        return {int(d): None for d in document_ids}


# Au-delà de ce nombre de patches dans le périmètre, on passe par une présélection
# ANN au lieu du MaxSim exact sur tout l'espace (~150 pages à 747 patches).
EXACT_MAXSIM_MAX_PATCHES = 150_000
# Patches ramenés par token lors de la présélection ANN.
#
# Volontairement AUCUN plafond sur le nombre de pages candidates. L'idée paraissait
# évidente — sur 776 pages retenues, 36 % ne sont effleurées que par un seul patch — mais
# la mesure l'a réfutée : le filtre dynamique appliqué juste après (seuil absolu + marge
# relative) conserve LES 40 PAGES du pool dans 7 requêtes sur 8, donc les rangs 20 à 40
# pèsent autant que la tête. Face à un MaxSim exact de référence, plafonner à 400 pages ne
# reproduisait l'ensemble retenu que dans 38 % des cas (25 % à 200), là où l'absence de
# plafond atteint 99,7 % de rappel. Réduire le volume chargé passe donc par le pooling de
# tokens ou un dtype plus court, pas par une coupe de pages.
ANN_LIMIT_PER_TOKEN = 250
# Contrôle échantillonné de l'invariant de normalisation (cf. insert).
_NORM_SAMPLE_SIZE = 512
_NORM_TOLERANCE = 1e-3


def _patches_are_normalized(vectors: "np.ndarray") -> bool:
    """Vérifie sur un échantillon que les vecteurs stockés sont bien L2-normalisés.

    Coûte quelques microsecondes et permet de sauter la renormalisation complète
    (mesurée à 6,9 s sur 1,38 M de vecteurs) tout en rendant visible une régression
    d'écriture au lieu de fausser silencieusement les scores.
    """
    n = len(vectors)
    if n == 0:
        return True
    sample = vectors if n <= _NORM_SAMPLE_SIZE else vectors[:: max(1, n // _NORM_SAMPLE_SIZE)][:_NORM_SAMPLE_SIZE]
    norms = np.linalg.norm(sample, axis=1)
    return bool(np.all(np.abs(norms - 1.0) <= _NORM_TOLERANCE))


def _collect_candidate_pages(
    table,
    query_token_embeddings: List[List[float]],
    filter_str: str,
    max_workers: int,
) -> List[int]:
    """Pages candidates : union des meilleurs patches de chaque token (recherche ANN).

    Les identifiants sont lus en Arrow puis en NumPy ; l'ancien ``to_list()`` créait un
    dictionnaire Python par patch (jusqu'à ``tokens × ANN_LIMIT_PER_TOKEN``) pour n'en
    extraire qu'un entier.
    """
    from concurrent.futures import ThreadPoolExecutor

    def search_token(token_vec):
        try:
            res = (
                table.search(token_vec)
                .metric("cosine")
                .where(filter_str)
                .select(["chunk_id"])
                .limit(ANN_LIMIT_PER_TOKEN)
                .to_arrow()
            )
            if not res.num_rows:
                return None
            return res["chunk_id"].to_numpy(zero_copy_only=False)
        except Exception as ex:
            logger.warning("Error searching token in LanceDB: %s", ex)
            return None

    candidates: Set[int] = set()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chunk_ids in executor.map(search_token, query_token_embeddings):
            if chunk_ids is None:
                continue
            candidates.update(int(c) for c in chunk_ids)
    return sorted(candidates)


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
        _t_start = time.perf_counter()
        _t_candidate = _t_start  # borne de fin de la phase « récupération candidats »
        table = get_colpali_table()
        doc_ids_str = ",".join(map(str, document_ids))
        filter_str = f"document_id in ({doc_ids_str})"

        # 1. Volume du périmètre. count_rows compte côté moteur ; l'ancien
        # `select(["chunk_id"]).to_list()` matérialisait un dict Python par patch
        # (mesuré : 2 388 ms contre 51 ms) uniquement pour en prendre la longueur.
        total_patches = table.count_rows(filter=filter_str)
        logger.info("[search_colpali_lancedb] Total patches in space for these documents: %d", total_patches)

        # 2. Retrieve patches based on scale
        tbl = None
        if total_patches <= EXACT_MAXSIM_MAX_PATCHES:
            logger.info("[search_colpali_lancedb] Small/medium scale space (<= 150 pages). Performing exact MaxSim on all pages.")
            tbl = table.search().where(filter_str).select(["chunk_id", "document_id", "vector"]).to_arrow()
        else:
            logger.info(
                "[search_colpali_lancedb] Large scale space (> 150 pages). Recherche ANN "
                "par token (%d patches/token).",
                ANN_LIMIT_PER_TOKEN,
            )
            candidate_chunk_ids = _collect_candidate_pages(
                table,
                query_token_embeddings,
                filter_str,
                max_workers=min(16, len(query_token_embeddings)),
            )
            logger.info(
                "[search_colpali_lancedb] %d page(s) candidate(s).", len(candidate_chunk_ids)
            )
            _t_candidate = time.perf_counter()
            if candidate_chunk_ids:
                candidate_ids_str = ",".join(map(str, candidate_chunk_ids))
                patch_filter = f"chunk_id in ({candidate_ids_str})"
                tbl = table.search().where(patch_filter).select(["chunk_id", "document_id", "vector"]).to_arrow()

        if tbl is None or len(tbl) == 0:
            logger.info("[search_colpali_lancedb] No patches found.")
            return []

        _t_loaded = time.perf_counter()
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
            
        # 4. Compute MaxSim per page using NumPy — version vectorisée.
        # Au lieu d'un produit matriciel + une normalisation PAR page (des centaines de
        # petites opérations dispatched depuis Python), on normalise TOUS les patches en
        # une passe et on calcule UNE seule grande matrice de similarité (T × N_total) via
        # un unique appel BLAS multithreadé. Chaque page ne fait plus qu'un max+somme sur
        # sa tranche de colonnes. Résultat numériquement identique, nettement plus rapide.
        Q = np.array(query_token_embeddings, dtype=np.float32)  # (T, 128)
        Q_norms = np.linalg.norm(Q, axis=1, keepdims=True)
        Q_norms = np.where(Q_norms == 0, 1.0, Q_norms)
        Q = Q / Q_norms

        num_tokens = len(query_token_embeddings)

        # Les patches sont normalisés À L'ÉCRITURE (cf. insert_colpali_patches_batch_lancedb),
        # donc pas de renormalisation ici : elle coûtait 6,9 s sur 1,38 M de vecteurs pour
        # rediviser par 1,0. Contrôle échantillonné pour qu'une régression se voie.
        P_all = vectors_numpy.astype(np.float32, copy=False)
        if not _patches_are_normalized(P_all):
            logger.warning(
                "[search_colpali_lancedb] Patches non normalisés détectés — renormalisation "
                "de secours (ré-indexer ces documents pour rétablir l'invariant)."
            )
            P_all_norms = np.linalg.norm(P_all, axis=1, keepdims=True)
            P_all = P_all / np.where(P_all_norms == 0, 1.0, P_all_norms)

        # Matrice de similarité cosinus complète (T × N_total) — un seul GEMM.
        sims_all = Q @ P_all.T

        final_results = []
        for chunk_id, indices in chunk_to_indices.items():
            if not indices:
                continue
            # Max par token sur les patches de CETTE page, puis somme (= MaxSim).
            maxsim_sum = float(np.sum(np.max(sims_all[:, indices], axis=1)))

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
        # [PERF] Décompose le coût MaxSim : recherche candidats (ANN par token) vs
        # chargement des vecteurs (I/O disque + Arrow) vs calcul NumPy. Guide le réglage
        # (ex. baisser limit/candidats, ou gater ColPali) sans deviner.
        _t_maxsim = time.perf_counter()
        logger.info(
            "[PERF][colpali] MaxSim — total %.2fs = candidats %.2fs + chargement %.2fs (%d vecteurs) + calcul %.2fs (%d pages)",
            _t_maxsim - _t_start,
            _t_candidate - _t_start,
            _t_loaded - _t_candidate,
            len(tbl),
            _t_maxsim - _t_loaded,
            len(chunk_to_indices),
        )
        logger.info(
            "[search_colpali_lancedb] Sorted results. Top 5 match distances: %s",
            [round(r["_distance"], 4) for r in final_results[:5]],
        )
        return final_results[:limit]
    except Exception as e:
        logger.error(f"Error executing ColPali MaxSim search in LanceDB: {e}", exc_info=True)
        return []
