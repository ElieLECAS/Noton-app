import logging
import time
from typing import List, Dict, Any, Tuple
from sqlmodel import Session
from app.services.space_search_service import search_relevant_passages

logger = logging.getLogger(__name__)

def match_page(retrieved_doc_title: str, retrieved_page_no: int, expected_pages: List[Dict[str, Any]]) -> bool:
    """
    Détermine si une page récupérée correspond à l'une des pages attendues.
    Matching flexible sur le titre (insensible à la casse, sous-chaîne) et exact sur le numéro de page.
    """
    if not retrieved_doc_title:
        return False
    
    ret_title_lower = retrieved_doc_title.lower()
    
    for exp in expected_pages:
        exp_title = exp.get("document_title")
        if not exp_title:
            continue
        
        exp_title_lower = exp_title.lower()
        # Vérifier si l'un est sous-chaîne de l'autre
        if exp_title_lower in ret_title_lower or ret_title_lower in exp_title_lower:
            # Récupérer les pages attendues (converties en int pour la comparaison)
            exp_pages = exp.get("pages", [])
            parsed_pages = []
            for p in exp_pages:
                try:
                    parsed_pages.append(int(p))
                except (ValueError, TypeError):
                    pass
            
            if retrieved_page_no in parsed_pages:
                return True
                
    return False

def compute_context_precision(retrieved_pages: List[Tuple[str, int]], expected_pages: List[Dict[str, Any]]) -> float:
    """
    Context Precision@K = (Sum_{i=1}^{K} (Precision@i * Relevance(i))) / Total Expected Pages Retrieved
    """
    hits = []
    num_expected_retrieved = 0
    
    for idx, (doc_title, page_no) in enumerate(retrieved_pages):
        is_relevant = match_page(doc_title, page_no, expected_pages)
        hits.append(1 if is_relevant else 0)
        if is_relevant:
            num_expected_retrieved += 1
            
    if num_expected_retrieved == 0:
        return 0.0
        
    precision_sum = 0.0
    relevant_so_far = 0
    for idx, hit in enumerate(hits):
        if hit == 1:
            relevant_so_far += 1
            precision_at_i = relevant_so_far / (idx + 1)
            precision_sum += precision_at_i
            
    return precision_sum / num_expected_retrieved

def compute_context_recall(retrieved_pages: List[Tuple[str, int]], expected_pages: List[Dict[str, Any]]) -> float:
    """
    Context Recall = Total Expected Pages Retrieved / Total Expected Pages
    """
    total_expected = 0
    for exp in expected_pages:
        total_expected += len(exp.get("pages", []))
        
    if total_expected == 0:
        return 1.0
        
    retrieved_expected_count = 0
    # Parcourir chaque page attendue unique et vérifier si elle a été récupérée
    for exp in expected_pages:
        exp_title = exp.get("document_title", "")
        for p in exp.get("pages", []):
            try:
                target_p = int(p)
            except (ValueError, TypeError):
                continue
            
            # Vérifier si ce couple (titre, page) est dans les résultats récupérés
            found = False
            for doc_title, page_no in retrieved_pages:
                if not doc_title:
                    continue
                if exp_title.lower() in doc_title.lower() or doc_title.lower() in exp_title.lower():
                    if page_no == target_p:
                        found = True
                        break
            if found:
                retrieved_expected_count += 1
                
    return retrieved_expected_count / total_expected

def compute_mrr(retrieved_pages: List[Tuple[str, int]], expected_pages: List[Dict[str, Any]]) -> float:
    """
    MRR = 1 / rang de la première page pertinente trouvée
    """
    for idx, (doc_title, page_no) in enumerate(retrieved_pages):
        if match_page(doc_title, page_no, expected_pages):
            return 1.0 / (idx + 1)
    return 0.0

async def evaluate_retriever_dataset(
    session: Session,
    space_id: int,
    user_id: int,
    dataset: List[Dict[str, Any]],
    k: int = 15,
) -> Dict[str, Any]:
    """
    Évalue le retriever ColPali sur un dataset de Q&A.
    """
    start_time = time.time()
    results = []
    
    total_precision = 0.0
    total_recall = 0.0
    total_mrr = 0.0
    
    # Séparer les scores par type de question (mono vs cross) si défini
    type_stats = {}
    
    for item in dataset:
        question = item.get("question", "").strip()
        q_type = item.get("type", "mono-document").strip()
        expected_pages = item.get("pages_attendues", [])
        
        if not question:
            continue
            
        # Exécuter la recherche dans l'espace
        search_res = await search_relevant_passages(
            session=session,
            space_id=space_id,
            query_text=question,
            user_id=user_id,
            k=k,
            document_filter="all"
        )
        
        passages = search_res.get("passages", [])
        
        # Extraire les couples (titre, page_no)
        retrieved_pages = []
        for p in passages:
            doc_title = p.get("document_title", "")
            page_no = p.get("page_no")
            # Fallback page_no
            if page_no is None:
                page_no = p.get("page_start", 1)
            try:
                retrieved_pages.append((doc_title, int(page_no)))
            except (ValueError, TypeError):
                retrieved_pages.append((doc_title, 1))
                
        # Calculer les métriques
        precision = compute_context_precision(retrieved_pages, expected_pages)
        recall = compute_context_recall(retrieved_pages, expected_pages)
        mrr = compute_mrr(retrieved_pages, expected_pages)
        
        total_precision += precision
        total_recall += recall
        total_mrr += mrr
        
        # Classifier les pages récupérées pour l'affichage visuel
        hits_details = []
        misses_details = []
        noise_details = []
        
        # 1. Identifier les Hits et le Bruit (Noise) parmi les pages récupérées
        for doc_title, page_no in retrieved_pages:
            is_hit = match_page(doc_title, page_no, expected_pages)
            page_info = {"document_title": doc_title, "page": page_no}
            if is_hit:
                if page_info not in hits_details:
                    hits_details.append(page_info)
            else:
                if page_info not in noise_details:
                    noise_details.append(page_info)
                    
        # 2. Identifier les Manqués (Misses) parmi les pages attendues
        for exp in expected_pages:
            exp_title = exp.get("document_title", "")
            for p in exp.get("pages", []):
                try:
                    target_p = int(p)
                except (ValueError, TypeError):
                    continue
                
                # Vérifier si cette page attendue a été trouvée
                found = False
                for hit in hits_details:
                    if exp_title.lower() in hit["document_title"].lower() or hit["document_title"].lower() in exp_title.lower():
                        if hit["page"] == target_p:
                            found = True
                            break
                if not found:
                    misses_details.append({"document_title": exp_title, "page": target_p})
        
        # Enregistrer les statistiques par type
        if q_type not in type_stats:
            type_stats[q_type] = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "count": 0}
        type_stats[q_type]["precision"] += precision
        type_stats[q_type]["recall"] += recall
        type_stats[q_type]["mrr"] += mrr
        type_stats[q_type]["count"] += 1
        
        results.append({
            "question": question,
            "type": q_type,
            "expected_pages": expected_pages,
            "retrieved_pages": [{"document_title": t, "page": p} for t, p in retrieved_pages],
            "metrics": {
                "context_precision": round(precision, 4),
                "context_recall": round(recall, 4),
                "mrr": round(mrr, 4)
            },
            "analysis": {
                "hits": hits_details,
                "misses": misses_details,
                "noise": noise_details
            }
        })
        
    num_queries = len(results)
    global_precision = total_precision / num_queries if num_queries > 0 else 0.0
    global_recall = total_recall / num_queries if num_queries > 0 else 0.0
    global_mrr = total_mrr / num_queries if num_queries > 0 else 0.0
    
    # Formater les stats par type
    formatted_type_stats = {}
    for t, stats in type_stats.items():
        count = stats["count"]
        formatted_type_stats[t] = {
            "count": count,
            "context_precision": round(stats["precision"] / count, 4) if count > 0 else 0.0,
            "context_recall": round(stats["recall"] / count, 4) if count > 0 else 0.0,
            "mrr": round(stats["mrr"] / count, 4) if count > 0 else 0.0
        }
        
    elapsed = time.time() - start_time
    
    return {
        "global_metrics": {
            "context_precision": round(global_precision, 4),
            "context_recall": round(global_recall, 4),
            "mrr": round(global_mrr, 4),
            "total_questions": num_queries,
            "execution_time_seconds": round(elapsed, 2)
        },
        "type_stats": formatted_type_stats,
        "details": results
    }
