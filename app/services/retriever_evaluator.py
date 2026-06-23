import logging
import time
from typing import List, Dict, Any, Tuple, Optional
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
    ret_tokens = set(ret_title_lower.split())

    for exp in expected_pages:
        exp_title = exp.get("document_title")
        if not exp_title:
            continue

        exp_title_lower = exp_title.lower()
        exp_tokens = set(exp_title_lower.split())
        # Matching flexible : sous-chaîne contiguë OU sous-ensemble de mots
        # (ex. "Notice Perform 70" ⊆ "Notice de pose Perform 70 - Rev3")
        if (
            exp_title_lower in ret_title_lower
            or ret_title_lower in exp_title_lower
            or (exp_tokens and exp_tokens.issubset(ret_tokens))
            or (ret_tokens and ret_tokens.issubset(exp_tokens))
        ):
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
    for exp in expected_pages:
        exp_title = exp.get("document_title", "")
        for p in exp.get("pages", []):
            try:
                target_p = int(p)
            except (ValueError, TypeError):
                continue

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


def _passages_to_retrieved_pages(passages: List[Dict[str, Any]]) -> Tuple[List[Tuple[str, int]], List[Dict[str, Any]]]:
    """Extrait couples (titre, page) et détails ordonnés depuis une liste de passages."""
    retrieved_pages: List[Tuple[str, int]] = []
    retrieved_details: List[Dict[str, Any]] = []
    for idx, p in enumerate(passages):
        doc_title = p.get("document_title", "")
        page_no = p.get("page_no")
        if page_no is None:
            page_no = p.get("page_start", 1)
        try:
            page_no_int = int(page_no)
        except (ValueError, TypeError):
            page_no_int = 1
        retrieved_pages.append((doc_title, page_no_int))
        detail: Dict[str, Any] = {
            "rank": idx + 1,
            "document_title": doc_title,
            "page": page_no_int,
            "score": round(float(p.get("score", 0.0)), 4),
        }
        if p.get("rerank_score") is not None:
            detail["rerank_score"] = round(float(p["rerank_score"]), 4)
        retrieved_details.append(detail)
    return retrieved_pages, retrieved_details


def _build_analysis(
    retrieved_details: List[Dict[str, Any]],
    expected_pages: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Construit hits / misses / noise à partir des détails récupérés."""
    hits_details: List[Dict[str, Any]] = []
    noise_details: List[Dict[str, Any]] = []

    for rd in retrieved_details:
        is_hit = match_page(rd["document_title"], rd["page"], expected_pages)
        page_info = {
            "document_title": rd["document_title"],
            "page": rd["page"],
            "rank": rd["rank"],
            "score": rd["score"],
        }
        if rd.get("rerank_score") is not None:
            page_info["rerank_score"] = rd["rerank_score"]
        if is_hit:
            if not any(
                h["document_title"] == rd["document_title"] and h["page"] == rd["page"]
                for h in hits_details
            ):
                hits_details.append(page_info)
        else:
            if not any(
                n["document_title"] == rd["document_title"] and n["page"] == rd["page"]
                for n in noise_details
            ):
                noise_details.append(page_info)

    misses_details: List[Dict[str, Any]] = []
    for exp in expected_pages:
        exp_title = exp.get("document_title", "")
        for p in exp.get("pages", []):
            try:
                target_p = int(p)
            except (ValueError, TypeError):
                continue
            found = False
            for hit in hits_details:
                if exp_title.lower() in hit["document_title"].lower() or hit["document_title"].lower() in exp_title.lower():
                    if hit["page"] == target_p:
                        found = True
                        break
            if not found:
                misses_details.append({"document_title": exp_title, "page": target_p})

    return {"hits": hits_details, "misses": misses_details, "noise": noise_details}


def compute_stage_metrics(
    passages: List[Dict[str, Any]],
    expected_pages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calcule métriques + analysis pour une étape de retrieval."""
    retrieved_pages, retrieved_details = _passages_to_retrieved_pages(passages)
    for rd in retrieved_details:
        rd["is_hit"] = match_page(rd["document_title"], rd["page"], expected_pages)

    precision = compute_context_precision(retrieved_pages, expected_pages)
    recall = compute_context_recall(retrieved_pages, expected_pages)
    mrr = compute_mrr(retrieved_pages, expected_pages)
    analysis = _build_analysis(retrieved_details, expected_pages)

    return {
        "retrieved_pages": [{"document_title": t, "page": p} for t, p in retrieved_pages],
        "retrieved_details": retrieved_details,
        "metrics": {
            "context_precision": round(precision, 4),
            "context_recall": round(recall, 4),
            "mrr": round(mrr, 4),
        },
        "analysis": analysis,
    }


def build_question_eval_result(
    question: str,
    q_type: str,
    expected_pages: List[Dict[str, Any]],
    passages: List[Dict[str, Any]],
    colpali_passages: Optional[List[Dict[str, Any]]] = None,
    pgvector_only_passages: Optional[List[Dict[str, Any]]] = None,
    lexical_only_passages: Optional[List[Dict[str, Any]]] = None,
    pre_kag_passages: Optional[List[Dict[str, Any]]] = None,
    post_rrf_passages: Optional[List[Dict[str, Any]]] = None,
    kag_only_passages: Optional[List[Dict[str, Any]]] = None,
    vision_rerank_enabled: Optional[bool] = None,
    minilm_rerank_enabled: Optional[bool] = None,
    kag_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Construit le résultat d'évaluation pour une question.
    passages = étape finale (post-MiniLM ou post-vision rerank).
    colpali_passages = ColPali seul (LanceDB + seuil dynamique).
    pgvector_only_passages = pgvector seul (vecteur texte mistral-embed).
    lexical_only_passages = lexical seul (BM25 / tsvector).
    pre_kag_passages = fusion RRF triple retriever (sans graphe KAG).
    post_rrf_passages = fusion RRF avec KAG (avant rerank MiniLM).
    kag_only_passages = canal KAG seul (graphe entités).
    """
    post = compute_stage_metrics(passages, expected_pages)
    result: Dict[str, Any] = {
        "question": question,
        "type": q_type,
        "expected_pages": expected_pages,
        "retrieved_pages": post["retrieved_pages"],
        "retrieved_details": post["retrieved_details"],
        "metrics": post["metrics"],
        "analysis": post["analysis"],
    }

    if colpali_passages is not None:
        colpali = compute_stage_metrics(colpali_passages, expected_pages)
        post_metrics = post["metrics"]
        colpali_metrics = colpali["metrics"]
        result["metrics_colpali"] = colpali_metrics
        result["analysis_colpali"] = colpali["analysis"]
        result["retrieved_pages_colpali"] = colpali["retrieved_pages"]
        result["retrieved_details_colpali"] = colpali["retrieved_details"]
        result["rerank_delta"] = {
            "precision": round(post_metrics["context_precision"] - colpali_metrics["context_precision"], 4),
            "recall": round(post_metrics["context_recall"] - colpali_metrics["context_recall"], 4),
            "mrr": round(post_metrics["mrr"] - colpali_metrics["mrr"], 4),
        }
        if vision_rerank_enabled is not None:
            result["vision_rerank_enabled"] = vision_rerank_enabled
        if minilm_rerank_enabled is not None:
            result["minilm_rerank_enabled"] = minilm_rerank_enabled

    if post_rrf_passages is not None:
        post_rrf = compute_stage_metrics(post_rrf_passages, expected_pages)
        result["metrics_post_rrf"] = post_rrf["metrics"]
        result["analysis_post_rrf"] = post_rrf["analysis"]
        result["retrieved_pages_post_rrf"] = post_rrf["retrieved_pages"]
        result["retrieved_details_post_rrf"] = post_rrf["retrieved_details"]
        if colpali_passages is not None:
            result["rrf_delta"] = {
                "precision": round(
                    post_rrf["metrics"]["context_precision"] - result["metrics_colpali"]["context_precision"],
                    4,
                ),
                "recall": round(
                    post_rrf["metrics"]["context_recall"] - result["metrics_colpali"]["context_recall"],
                    4,
                ),
                "mrr": round(
                    post_rrf["metrics"]["mrr"] - result["metrics_colpali"]["mrr"],
                    4,
                ),
            }
            result["minilm_delta"] = {
                "precision": round(
                    post["metrics"]["context_precision"] - post_rrf["metrics"]["context_precision"],
                    4,
                ),
                "recall": round(
                    post["metrics"]["context_recall"] - post_rrf["metrics"]["context_recall"],
                    4,
                ),
                "mrr": round(
                    post["metrics"]["mrr"] - post_rrf["metrics"]["mrr"],
                    4,
                ),
            }

    if pre_kag_passages is not None:
        pre_kag = compute_stage_metrics(pre_kag_passages, expected_pages)
        result["metrics_pre_kag"] = pre_kag["metrics"]
        result["analysis_pre_kag"] = pre_kag["analysis"]
        result["retrieved_pages_pre_kag"] = pre_kag["retrieved_pages"]
        result["retrieved_details_pre_kag"] = pre_kag["retrieved_details"]
        if post_rrf_passages is not None:
            result["kag_delta"] = {
                "precision": round(
                    result["metrics_post_rrf"]["context_precision"] - pre_kag["metrics"]["context_precision"],
                    4,
                ),
                "recall": round(
                    result["metrics_post_rrf"]["context_recall"] - pre_kag["metrics"]["context_recall"],
                    4,
                ),
                "mrr": round(
                    result["metrics_post_rrf"]["mrr"] - pre_kag["metrics"]["mrr"],
                    4,
                ),
            }

    if kag_only_passages is not None:
        kag_only = compute_stage_metrics(kag_only_passages, expected_pages)
        result["metrics_kag_only"] = kag_only["metrics"]
        result["analysis_kag_only"] = kag_only["analysis"]
        result["retrieved_pages_kag_only"] = kag_only["retrieved_pages"]
        result["retrieved_details_kag_only"] = kag_only["retrieved_details"]

    if pgvector_only_passages is not None:
        pgvector_only = compute_stage_metrics(pgvector_only_passages, expected_pages)
        result["metrics_pgvector_only"] = pgvector_only["metrics"]
        result["analysis_pgvector_only"] = pgvector_only["analysis"]
        result["retrieved_pages_pgvector_only"] = pgvector_only["retrieved_pages"]
        result["retrieved_details_pgvector_only"] = pgvector_only["retrieved_details"]

    if lexical_only_passages is not None:
        lexical_only = compute_stage_metrics(lexical_only_passages, expected_pages)
        result["metrics_lexical_only"] = lexical_only["metrics"]
        result["analysis_lexical_only"] = lexical_only["analysis"]
        result["retrieved_pages_lexical_only"] = lexical_only["retrieved_pages"]
        result["retrieved_details_lexical_only"] = lexical_only["retrieved_details"]

    if kag_enabled is not None:
        result["kag_enabled"] = kag_enabled

    return result


def _aggregate_global_metrics(details: List[Dict[str, Any]], prefix: str = "") -> Dict[str, float]:
    """Agrège precision/recall/mrr sur une liste de résultats par question."""
    metrics_key = "metrics"
    if prefix == "colpali":
        metrics_key = "metrics_colpali"
    elif prefix == "post_rrf":
        metrics_key = "metrics_post_rrf"
    elif prefix == "pre_kag":
        metrics_key = "metrics_pre_kag"
    elif prefix == "kag_only":
        metrics_key = "metrics_kag_only"
    elif prefix == "pgvector_only":
        metrics_key = "metrics_pgvector_only"
    elif prefix == "lexical_only":
        metrics_key = "metrics_lexical_only"

    n = len(details)
    if n == 0:
        return {"context_precision": 0.0, "context_recall": 0.0, "mrr": 0.0}

    total_p = sum(d[metrics_key]["context_precision"] for d in details)
    total_r = sum(d[metrics_key]["context_recall"] for d in details)
    total_m = sum(d[metrics_key]["mrr"] for d in details)
    return {
        "context_precision": round(total_p / n, 4),
        "context_recall": round(total_r / n, 4),
        "mrr": round(total_m / n, 4),
    }


def _aggregate_type_stats(details: List[Dict[str, Any]], metrics_key: str = "metrics") -> Dict[str, Any]:
    type_stats: Dict[str, Dict[str, float]] = {}
    for d in details:
        q_type = d.get("type", "mono-document")
        if q_type not in type_stats:
            type_stats[q_type] = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "count": 0}
        m = d[metrics_key]
        type_stats[q_type]["precision"] += m["context_precision"]
        type_stats[q_type]["recall"] += m["context_recall"]
        type_stats[q_type]["mrr"] += m["mrr"]
        type_stats[q_type]["count"] += 1

    formatted: Dict[str, Any] = {}
    for t, stats in type_stats.items():
        count = stats["count"]
        formatted[t] = {
            "count": count,
            "context_precision": round(stats["precision"] / count, 4) if count > 0 else 0.0,
            "context_recall": round(stats["recall"] / count, 4) if count > 0 else 0.0,
            "mrr": round(stats["mrr"] / count, 4) if count > 0 else 0.0,
        }
    return formatted


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

    for item in dataset:
        question = item.get("question", "").strip()
        q_type = item.get("type", "mono-document").strip()
        expected_pages = item.get("pages_attendues", [])

        if not question:
            continue

        search_res = await search_relevant_passages(
            session=session,
            space_id=space_id,
            query_text=question,
            user_id=user_id,
            k=k,
            document_filter="all",
            include_retrieval_stages=True,
        )

        passages = search_res.get("passages", [])
        stages = search_res.get("retrieval_stages") or {}
        colpali_passages = stages.get("colpali_only") or stages.get("colpali")
        pgvector_only_passages = stages.get("pgvector_only")
        lexical_only_passages = stages.get("lexical_only")
        post_rrf_passages = stages.get("post_rrf")
        pre_kag_passages = stages.get("pre_kag_rrf")
        kag_only_passages = stages.get("kag_only")

        results.append(
            build_question_eval_result(
                question=question,
                q_type=q_type,
                expected_pages=expected_pages,
                passages=passages,
                colpali_passages=colpali_passages,
                pgvector_only_passages=pgvector_only_passages,
                lexical_only_passages=lexical_only_passages,
                pre_kag_passages=pre_kag_passages,
                post_rrf_passages=post_rrf_passages,
                kag_only_passages=kag_only_passages,
                vision_rerank_enabled=stages.get("vision_rerank_enabled"),
                minilm_rerank_enabled=stages.get("minilm_rerank_enabled"),
                kag_enabled=stages.get("kag_enabled"),
            )
        )

    num_queries = len(results)
    global_metrics = _aggregate_global_metrics(results)
    global_metrics["total_questions"] = num_queries
    global_metrics["execution_time_seconds"] = round(time.time() - start_time, 2)

    global_metrics_colpali = _aggregate_global_metrics(results, prefix="colpali")
    global_metrics_pgvector_only = (
        _aggregate_global_metrics(results, prefix="pgvector_only")
        if results and results[0].get("metrics_pgvector_only")
        else None
    )
    global_metrics_lexical_only = (
        _aggregate_global_metrics(results, prefix="lexical_only")
        if results and results[0].get("metrics_lexical_only")
        else None
    )
    global_metrics_post_rrf = _aggregate_global_metrics(results, prefix="post_rrf") if results and results[0].get("metrics_post_rrf") else None
    global_metrics_pre_kag = _aggregate_global_metrics(results, prefix="pre_kag") if results and results[0].get("metrics_pre_kag") else None

    rerank_impact = {"precision_delta": 0.0, "recall_delta": 0.0, "mrr_delta": 0.0}
    kag_impact = {"precision_delta": 0.0, "recall_delta": 0.0, "mrr_delta": 0.0}
    if num_queries > 0 and results[0].get("rerank_delta"):
        rerank_impact = {
            "precision_delta": round(
                sum(r["rerank_delta"]["precision"] for r in results) / num_queries, 4
            ),
            "recall_delta": round(
                sum(r["rerank_delta"]["recall"] for r in results) / num_queries, 4
            ),
            "mrr_delta": round(
                sum(r["rerank_delta"]["mrr"] for r in results) / num_queries, 4
            ),
        }
    kag_delta_count = sum(1 for r in results if r.get("kag_delta"))
    if kag_delta_count > 0:
        kag_impact = {
            "precision_delta": round(
                sum(r["kag_delta"]["precision"] for r in results if r.get("kag_delta")) / kag_delta_count,
                4,
            ),
            "recall_delta": round(
                sum(r["kag_delta"]["recall"] for r in results if r.get("kag_delta")) / kag_delta_count,
                4,
            ),
            "mrr_delta": round(
                sum(r["kag_delta"]["mrr"] for r in results if r.get("kag_delta")) / kag_delta_count,
                4,
            ),
        }

    vision_rerank_enabled = any(r.get("vision_rerank_enabled") for r in results)
    minilm_rerank_enabled = any(r.get("minilm_rerank_enabled") for r in results)
    kag_enabled = any(r.get("kag_enabled") for r in results)

    eval_result: Dict[str, Any] = {
        "global_metrics": global_metrics,
        "global_metrics_colpali": global_metrics_colpali,
        "rerank_impact": rerank_impact,
        "kag_impact": kag_impact,
        "vision_rerank_enabled": vision_rerank_enabled,
        "minilm_rerank_enabled": minilm_rerank_enabled,
        "kag_enabled": kag_enabled,
        "type_stats": _aggregate_type_stats(results),
        "type_stats_colpali": _aggregate_type_stats(results, metrics_key="metrics_colpali"),
        "details": results,
    }
    if global_metrics_pgvector_only:
        eval_result["global_metrics_pgvector_only"] = global_metrics_pgvector_only
        eval_result["type_stats_pgvector_only"] = _aggregate_type_stats(results, metrics_key="metrics_pgvector_only")
    if global_metrics_lexical_only:
        eval_result["global_metrics_lexical_only"] = global_metrics_lexical_only
        eval_result["type_stats_lexical_only"] = _aggregate_type_stats(results, metrics_key="metrics_lexical_only")
    if global_metrics_post_rrf:
        eval_result["global_metrics_post_rrf"] = global_metrics_post_rrf
        eval_result["type_stats_post_rrf"] = _aggregate_type_stats(results, metrics_key="metrics_post_rrf")
    if global_metrics_pre_kag:
        eval_result["global_metrics_pre_kag"] = global_metrics_pre_kag
        eval_result["type_stats_pre_kag"] = _aggregate_type_stats(results, metrics_key="metrics_pre_kag")
    return eval_result


async def generate_rag_response(
    session: Session,
    space_id: int,
    user_id: int,
    question: str,
    passages: List[Dict[str, Any]],
) -> str:
    """
    Génère la réponse de l'assistant à partir des passages RAG en mimant le chatbot.
    """
    from app.config import settings
    from app.services.rag_generation_service import build_rag_generation_messages

    if not passages:
        return "Je ne trouve pas de réponse à votre question dans les documents disponibles dans cet espace car aucune source n'est jugée suffisamment pertinente (seuil minimum de 75%)."

    try:
        messages = await build_rag_generation_messages(
            session,
            passages,
            question,
            model=settings.MODEL_FAST,
        )
    except ImportError:
        logger.error("Impossible d'importer le formateur de contexte RAG")
        return "Erreur d'importation du formateur de contexte."

    try:
        if settings.LLM_PROVIDER == "ollama":
            from app.services.ollama_service import chat as ollama_chat
            response = await ollama_chat(
                message="",
                model=settings.MODEL_FAST,
                context=messages
            )
        else:
            from app.services.mistral_service import chat as mistral_chat
            response = await mistral_chat(
                message="",
                model=settings.MODEL_FAST,
                context=messages
            )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error("Error during evaluation RAG response generation: %s", e)
        return f"Erreur lors de la génération : {str(e)}"


async def run_llm_judge(
    question: str,
    generated_response: str,
    expected_response: str,
    judge_model: str = "mistral-small-latest"
) -> Dict[str, Any]:
    """
    Évalue la réponse générée par rapport à la réponse attendue en utilisant un LLM juge.
    """
    from app.config import settings
    import json

    JUDGE_SYSTEM_PROMPT = """Tu es un expert en évaluation de systèmes de Questions-Réponses RAG.
Ton rôle est d'évaluer la pertinence et l'exactitude de la réponse générée par rapport à une réponse attendue de référence (ground truth).

Tu dois attribuer une note de 1 à 5 selon les critères suivants :
1 : Complètement fausse, hors-sujet ou contenant des hallucinations graves.
2 : Contient de nombreuses inexactitudes ou omet la quasi-totalité des informations importantes.
3 : Partiellement correcte, mais manque de précision ou omet des détails clés.
4 : Presque parfaite, exacte et compréhensible, avec de légères omissions non critiques.
5 : Excellente, totalement fidèle à la réponse attendue, complète et précise.

Renvoie STRICTEMENT un objet JSON avec cette structure :
{
  "score": <int entre 1 et 5>,
  "justification": "<Explication détaillée en français>"
}"""

    prompt = f"""Question posée : {question}
Réponse attendue : {expected_response}
Réponse générée : {generated_response}"""

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    try:
        if settings.MISTRAL_API_KEY:
            from app.services.mistral_service import chat as mistral_chat
            response = await mistral_chat(
                message="",
                model=judge_model,
                context=messages,
                response_format={"type": "json_object"}
            )
        elif settings.LLM_PROVIDER == "ollama":
            from app.services.ollama_service import chat as ollama_chat
            response = await ollama_chat(
                message="",
                model=settings.MODEL_FAST,
                context=messages
            )
        else:
            raise ValueError("Aucun fournisseur de LLM configuré.")

        content = response["choices"][0]["message"]["content"]
        eval_result = json.loads(content)

        score = eval_result.get("score")
        justification = eval_result.get("justification", "")
        if isinstance(score, (int, float)):
            score = int(score)
        else:
            score = 3

        return {"score": score, "justification": justification}
    except Exception as e:
        logger.error("Error during LLM evaluation judge: %s", e)
        return {"score": 1, "justification": f"Erreur d'évaluation par le juge : {str(e)}"}
