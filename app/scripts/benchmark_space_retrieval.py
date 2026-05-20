"""
Benchmark offline simple du retrieval espace (B13/C9).

Usage:
  python -m app.scripts.benchmark_space_retrieval --space-id 1 --user-id 1 --input qa_gold.json

Format qa_gold.json:
[
  {"question": "...", "gold_chunk_ids": [12, 18]},
  {"question": "...", "gold_chunk_ids": [44]}
]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

from sqlmodel import Session

from app.database import engine
from app.services.space_search_service import search_relevant_passages


def _precision_at_k(pred_chunk_ids: List[int], gold_chunk_ids: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    pred_top = pred_chunk_ids[:k]
    if not pred_top:
        return 0.0
    gold = set(gold_chunk_ids or [])
    hit = sum(1 for cid in pred_top if cid in gold)
    return hit / float(len(pred_top))


def _recall_at_k(pred_chunk_ids: List[int], gold_chunk_ids: List[int], k: int) -> float:
    gold = set(gold_chunk_ids or [])
    if not gold:
        return 0.0
    pred_top = set(pred_chunk_ids[:k])
    return len(pred_top & gold) / float(len(gold))


async def _run(space_id: int, user_id: int, qa_rows: List[Dict], k: int) -> Dict[str, float]:
    p3 = []
    r5 = []
    with Session(engine) as session:
        for row in qa_rows:
            q = (row.get("question") or "").strip()
            if not q:
                continue
            passages = await search_relevant_passages(
                session=session,
                space_id=space_id,
                query_text=q,
                user_id=user_id,
                k=max(8, k),
            )
            pred = [int(p.get("source_leaf_chunk_id") or p.get("chunk_id") or -1) for p in passages if (p.get("source_leaf_chunk_id") or p.get("chunk_id"))]
            gold = [int(x) for x in (row.get("gold_chunk_ids") or [])]
            p3.append(_precision_at_k(pred, gold, 3))
            r5.append(_recall_at_k(pred, gold, 5))
    return {
        "count": len(p3),
        "precision_at_3": (sum(p3) / len(p3)) if p3 else 0.0,
        "recall_at_5": (sum(r5) / len(r5)) if r5 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-id", type=int, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    qa_rows = json.loads(args.input.read_text(encoding="utf-8"))
    import asyncio

    metrics = asyncio.run(_run(args.space_id, args.user_id, qa_rows, args.k))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
