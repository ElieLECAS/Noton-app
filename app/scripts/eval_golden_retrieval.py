"""Runner d'évaluation retrieval CAG-aware + comparatif reranker ON/OFF.

Pour chaque question du golden, on fait UN passage de retrieval (avec stages) puis
on construit le contexte CAG dans deux configurations, à partir du MÊME retrieval :
  - reranker ON  : CAG sur les passages post_rerank (MiniLM/camembert),
  - reranker OFF : CAG sur les passages post_rrf (fusion RRF brute, pré-reranker).

On mesure au NIVEAU DOCUMENT (aligné CAG, pas page-stricte) :
  - doc-recall (acceptable) : un des acceptable_document_ids est-il packé,
  - doc-recall (source)     : le document source exact est-il packé,
  - page-in-context         : la page attendue est-elle dans le set packé,
  - doc-precision           : part des docs packés qui sont dans l'acceptable set.

Usage : python -m app.scripts.eval_golden_retrieval [golden.json] [user_id] [intent]
"""
from __future__ import annotations

import asyncio
import json
import sys
from statistics import mean

from sqlmodel import Session

from app.database import engine
from app.services.space_search_service import search_relevant_passages
from app.services.context_packer_service import build_cag_context
from app.services.retriever_evaluator import evaluate_cag_document_hit

DEFAULT_GOLDEN = "tests/fixtures/golden/space28_generale_retrieval.json"


def _cag_doc_metrics(session, hits, intent, acceptable, source_id, expected):
    """Construit le CAG sur `hits` et retourne (doc_acc, doc_src, page_ctx, precision, packed)."""
    cag = build_cag_context(session, hits or [], system_prompt="", intent=intent, emit_sources_tag=False)
    cagdocs = cag.get("cag_documents") or []
    hit = evaluate_cag_document_hit(cagdocs, acceptable_document_ids=list(acceptable), expected_pages=expected)
    packed = hit["packed_ids"]
    doc_acc = hit["doc_hit_acceptable"]
    doc_src = source_id in packed
    page_ctx = hit["page_in_context"]
    precision = (len(set(packed) & acceptable) / len(packed)) if packed else 0.0
    return doc_acc, doc_src, page_ctx, precision, packed


async def main():
    golden_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GOLDEN
    user_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    intent = sys.argv[3] if len(sys.argv) > 3 else "documentation"

    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)
    entries = golden if isinstance(golden, list) else golden["entries"]
    space_id = 28

    agg = {
        "on":  {"doc_acc": 0, "doc_src": 0, "page": 0, "prec": []},
        "off": {"doc_acc": 0, "doc_src": 0, "page": 0, "prec": []},
    }
    diffs = []

    with Session(engine) as session:
        for e in entries:
            q = e["question"]
            acceptable = set(e.get("acceptable_document_ids") or [])
            source_id = (e.get("acceptable_document_ids") or [None])[0]
            expected = e.get("pages_attendues") or []

            r = await search_relevant_passages(
                session=session, space_id=space_id, query_text=q, user_id=user_id,
                k=15, include_retrieval_stages=True,
            )
            stages = r.get("retrieval_stages") or {}
            post_rerank = r.get("passages") or []          # reranker ON (état réel)
            post_rrf = stages.get("post_rrf") or []         # reranker OFF (fusion brute)

            on = _cag_doc_metrics(session, post_rerank, intent, acceptable, source_id, expected)
            off = _cag_doc_metrics(session, post_rrf, intent, acceptable, source_id, expected)

            for tag, res in (("on", on), ("off", off)):
                agg[tag]["doc_acc"] += int(res[0])
                agg[tag]["doc_src"] += int(res[1])
                agg[tag]["page"] += int(res[2])
                agg[tag]["prec"].append(res[3])

            # signaler les questions où ON et OFF divergent sur le doc source
            if on[1] != off[1] or on[2] != off[2]:
                diffs.append((e["id"], source_id, on, off))
            print(f"  {e['id']:8} src={source_id}  ON src={on[1]} page={on[2]} packed={on[4]}  |  OFF src={off[1]} page={off[2]} packed={off[4]}")

    n = len(entries)
    def pct(x): return f"{x}/{n} ({round(100*x/n)}%)"
    print("\n" + "=" * 66)
    print(f"COMPARATIF CAG — reranker ON vs OFF  (intent={intent}, {n} questions)")
    print("=" * 66)
    print(f"{'métrique (niveau document)':32} {'ON (rerank)':>14} {'OFF (RRF)':>14}")
    print(f"{'doc packé (acceptable)':32} {pct(agg['on']['doc_acc']):>14} {pct(agg['off']['doc_acc']):>14}")
    print(f"{'doc packé (source exact)':32} {pct(agg['on']['doc_src']):>14} {pct(agg['off']['doc_src']):>14}")
    print(f"{'page attendue dans le contexte':32} {pct(agg['on']['page']):>14} {pct(agg['off']['page']):>14}")
    print(f"{'doc-precision (moyenne)':32} {round(mean(agg['on']['prec']),3):>14} {round(mean(agg['off']['prec']),3):>14}")

    if diffs:
        print("\n[Questions où ON et OFF divergent sur le doc/page source]")
        for qid, src, on, off in diffs:
            print(f"  {qid} (src={src}): ON src={on[1]}/page={on[2]} packed={on[4]}  vs  OFF src={off[1]}/page={off[2]} packed={off[4]}")


if __name__ == "__main__":
    asyncio.run(main())
