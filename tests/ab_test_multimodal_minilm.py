"""
Script d'évaluation A/B pour comparer baseline vs config optimisée MiniLM.

Usage:
    python tests/ab_test_multimodal_minilm.py --space-id 1 --user-id 1

Prerequisites:
    - Avoir des documents indexés dans un espace
    - RERANKER_ENABLED=True dans .env
    - Base de données accessible
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from typing import List, Dict, Optional

from sqlmodel import Session

from app.config import settings
from app.database import engine
from app.services import space_search_service, reranker_service


@dataclass
class QueryTestCase:
    """Cas de test pour l'évaluation A/B."""
    query: str
    expected_keywords: List[str]  # Mots-clés attendus dans les résultats pertinents
    category: str  # "norm", "table", "instruction", "mixed"


# Dataset de test (à adapter selon votre domaine métier)
TEST_QUERIES = [
    QueryTestCase(
        query="Quelle est la norme pour les menuiseries extérieures ?",
        expected_keywords=["NF EN 14351", "menuiserie", "certification"],
        category="norm"
    ),
    QueryTestCase(
        query="Dimensions maximales pour un ouvrant de fenêtre",
        expected_keywords=["hauteur", "largeur", "ouvrant", "dimension"],
        category="table"
    ),
    QueryTestCase(
        query="Procédure de montage des profilés Perform 70",
        expected_keywords=["montage", "perform", "assemblage", "étape"],
        category="instruction"
    ),
    QueryTestCase(
        query="Compatibilité vitrage soleal avec profilé PF 76",
        expected_keywords=["vitrage", "soleal", "PF 76", "compatible"],
        category="mixed"
    ),
    QueryTestCase(
        query="Épaisseur minimale pour isolation thermique",
        expected_keywords=["épaisseur", "isolation", "thermique", "mm"],
        category="table"
    ),
    # Ajouter 10-25 autres questions représentatives de votre domaine
]


@dataclass
class EvalResult:
    """Résultat d'évaluation pour une configuration."""
    config_name: str
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_precision_top3: float
    avg_recall_keywords: float
    truncation_ratio: float
    avg_chunks_returned: float
    status_distribution: Dict[str, int]


async def run_query_with_config(
    session: Session,
    space_id: int,
    user_id: int,
    query: str,
    config_override: Dict,
) -> tuple[List[Dict], float]:
    """
    Exécute une requête avec une configuration donnée.
    
    Returns:
        (passages, latency_ms)
    """
    # Sauvegarder config originale
    original_config = {
        "RERANK_POOL": settings.RERANK_POOL,
        "RERANK_CHAR_CAP": settings.RERANK_CHAR_CAP,
        "MAX_DYNAMIC_K": settings.MAX_DYNAMIC_K,
    }
    
    # Appliquer override
    for key, value in config_override.items():
        setattr(settings, key, value)
    
    t_start = time.perf_counter()
    result = await space_search_service.search_relevant_passages(
        session=session,
        space_id=space_id,
        query_text=query,
        user_id=user_id,
        k=15,
    )
    latency_ms = (time.perf_counter() - t_start) * 1000
    
    # Restaurer config originale
    for key, value in original_config.items():
        setattr(settings, key, value)
    
    return result.get("passages", []), latency_ms


def compute_precision_top3(passages: List[Dict], expected_keywords: List[str]) -> float:
    """
    Calcule la précision top-3 : ratio de passages pertinents dans les 3 premiers.
    Un passage est pertinent s'il contient au moins 1 mot-clé attendu.
    """
    if not passages:
        return 0.0
    
    top3 = passages[:3]
    relevant = 0
    
    for p in top3:
        text = (p.get("passage_raw", "") or "").lower()
        if any(kw.lower() in text for kw in expected_keywords):
            relevant += 1
    
    return relevant / min(3, len(passages))


def compute_recall_keywords(passages: List[Dict], expected_keywords: List[str]) -> float:
    """
    Calcule le rappel de mots-clés : ratio de mots-clés trouvés dans tous les passages.
    """
    if not expected_keywords:
        return 1.0
    
    found_keywords = set()
    
    for p in passages:
        text = (p.get("passage_raw", "") or "").lower()
        for kw in expected_keywords:
            if kw.lower() in text:
                found_keywords.add(kw.lower())
    
    return len(found_keywords) / len(expected_keywords)


async def evaluate_config(
    session: Session,
    space_id: int,
    user_id: int,
    test_cases: List[QueryTestCase],
    config_name: str,
    config_override: Dict,
) -> EvalResult:
    """Évalue une configuration sur l'ensemble des cas de test."""
    latencies = []
    precisions_top3 = []
    recalls_keywords = []
    chunks_counts = []
    statuses = {}
    
    # Reset truncation stats avant l'éval
    reranker_service.reset_truncation_stats()
    
    for test_case in test_cases:
        passages, latency = await run_query_with_config(
            session, space_id, user_id, test_case.query, config_override
        )
        
        latencies.append(latency)
        precision = compute_precision_top3(passages, test_case.expected_keywords)
        recall = compute_recall_keywords(passages, test_case.expected_keywords)
        
        precisions_top3.append(precision)
        recalls_keywords.append(recall)
        chunks_counts.append(len(passages))
        
        # Track status distribution
        # Note: status n'est pas retourné dans passages, à extraire du result dict si besoin
    
    # Récupérer statistiques de troncature
    trunc_stats = reranker_service.get_truncation_stats()
    
    return EvalResult(
        config_name=config_name,
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0],
        avg_precision_top3=statistics.mean(precisions_top3),
        avg_recall_keywords=statistics.mean(recalls_keywords),
        truncation_ratio=trunc_stats.get("truncation_ratio", 0.0),
        avg_chunks_returned=statistics.mean(chunks_counts),
        status_distribution=statuses,
    )


def print_comparison(baseline: EvalResult, optimized: EvalResult):
    """Affiche la comparaison entre baseline et config optimisée."""
    print("\n" + "=" * 80)
    print("RÉSULTATS A/B TEST : Baseline vs Optimisé MiniLM 512")
    print("=" * 80)
    
    print(f"\n{'Métrique':<40} {'Baseline':<15} {'Optimisé':<15} {'Δ%':<10}")
    print("-" * 80)
    
    # Latence
    delta_latency = ((optimized.avg_latency_ms - baseline.avg_latency_ms) / baseline.avg_latency_ms) * 100
    print(f"{'Latence moyenne (ms)':<40} {baseline.avg_latency_ms:>14.1f} {optimized.avg_latency_ms:>14.1f} {delta_latency:>9.1f}%")
    
    delta_p50 = ((optimized.p50_latency_ms - baseline.p50_latency_ms) / baseline.p50_latency_ms) * 100
    print(f"{'Latence P50 (ms)':<40} {baseline.p50_latency_ms:>14.1f} {optimized.p50_latency_ms:>14.1f} {delta_p50:>9.1f}%")
    
    delta_p95 = ((optimized.p95_latency_ms - baseline.p95_latency_ms) / baseline.p95_latency_ms) * 100
    print(f"{'Latence P95 (ms)':<40} {baseline.p95_latency_ms:>14.1f} {optimized.p95_latency_ms:>14.1f} {delta_p95:>9.1f}%")
    
    # Qualité
    delta_prec = ((optimized.avg_precision_top3 - baseline.avg_precision_top3) / baseline.avg_precision_top3) * 100 if baseline.avg_precision_top3 > 0 else 0
    print(f"{'Précision top-3':<40} {baseline.avg_precision_top3:>14.2f} {optimized.avg_precision_top3:>14.2f} {delta_prec:>9.1f}%")
    
    delta_recall = ((optimized.avg_recall_keywords - baseline.avg_recall_keywords) / baseline.avg_recall_keywords) * 100 if baseline.avg_recall_keywords > 0 else 0
    print(f"{'Rappel mots-clés':<40} {baseline.avg_recall_keywords:>14.2f} {optimized.avg_recall_keywords:>14.2f} {delta_recall:>9.1f}%")
    
    # Troncature
    delta_trunc = ((optimized.truncation_ratio - baseline.truncation_ratio) / baseline.truncation_ratio) * 100 if baseline.truncation_ratio > 0 else 0
    print(f"{'Ratio troncature':<40} {baseline.truncation_ratio:>14.2%} {optimized.truncation_ratio:>14.2%} {delta_trunc:>9.1f}%")
    
    # Chunks
    delta_chunks = ((optimized.avg_chunks_returned - baseline.avg_chunks_returned) / baseline.avg_chunks_returned) * 100 if baseline.avg_chunks_returned > 0 else 0
    print(f"{'Chunks retournés (avg)':<40} {baseline.avg_chunks_returned:>14.1f} {optimized.avg_chunks_returned:>14.1f} {delta_chunks:>9.1f}%")
    
    print("\n" + "=" * 80)
    print("RECOMMANDATION :")
    
    # Décision basée sur métriques
    if delta_latency < -20 and delta_prec >= -5:  # Gain latence significatif, qualité stable
        print("✅ Config OPTIMISÉE recommandée : gain latence net, qualité préservée")
    elif delta_prec > 5 and delta_latency < 10:  # Gain qualité, latence acceptable
        print("✅ Config OPTIMISÉE recommandée : amélioration qualité, latence acceptable")
    elif delta_latency < -10 and -10 < delta_prec < 0:  # Compromis acceptable
        print("⚠️  Config OPTIMISÉE acceptable : compromis latence/qualité à valider manuellement")
    else:
        print("❌ Config BASELINE recommandée : conserver paramètres actuels")
    
    print("=" * 80 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="A/B test MiniLM 512 optimization")
    parser.add_argument("--space-id", type=int, required=True, help="ID de l'espace à tester")
    parser.add_argument("--user-id", type=int, required=True, help="ID de l'utilisateur")
    parser.add_argument("--queries-file", type=str, help="Fichier JSON de questions custom (optionnel)")
    args = parser.parse_args()
    
    test_cases = TEST_QUERIES
    
    # Charger questions custom si fourni
    if args.queries_file:
        with open(args.queries_file, 'r', encoding='utf-8') as f:
            custom_queries = json.load(f)
            test_cases = [
                QueryTestCase(**q) for q in custom_queries
            ]
    
    print(f"\n🧪 Démarrage A/B test avec {len(test_cases)} questions")
    print(f"   Espace ID: {args.space_id}, User ID: {args.user_id}\n")
    
    with Session(engine) as session:
        # Configuration baseline
        print("⏳ Évaluation BASELINE (RERANK_POOL=50, CHAR_CAP=8000)...")
        baseline_result = await evaluate_config(
            session=session,
            space_id=args.space_id,
            user_id=args.user_id,
            test_cases=test_cases,
            config_name="Baseline",
            config_override={
                "RERANK_POOL": 50,
                "RERANK_CHAR_CAP": 8000,
                "MAX_DYNAMIC_K": 10,
            }
        )
        
        # Configuration optimisée
        print("⏳ Évaluation OPTIMISÉE (RERANK_POOL=30, CHAR_CAP=1700)...")
        optimized_result = await evaluate_config(
            session=session,
            space_id=args.space_id,
            user_id=args.user_id,
            test_cases=test_cases,
            config_name="Optimized",
            config_override={
                "RERANK_POOL": 30,
                "RERANK_CHAR_CAP": 1700,
                "MAX_DYNAMIC_K": 8,
            }
        )
    
    # Afficher comparaison
    print_comparison(baseline_result, optimized_result)
    
    # Sauvegarder résultats
    results = {
        "baseline": baseline_result.__dict__,
        "optimized": optimized_result.__dict__,
        "test_cases_count": len(test_cases),
        "space_id": args.space_id,
    }
    
    output_file = "ab_test_results_minilm512.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Résultats sauvegardés dans {output_file}\n")


if __name__ == "__main__":
    asyncio.run(main())
