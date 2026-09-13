"""Rejoue l'ÉLECTION RÉELLE (``elect_documents`` tel qu'il est dans le code) sur de vrais
pools de retrieval — à lancer AVANT puis APRÈS une modification de l'élection, et comparer.

Contrairement à ``scripts/sim_election.py`` (qui ré-implémentait des heuristiques
candidates), ce script n'a aucune logique propre : il appelle les mêmes fonctions que le
routeur (retrievers, fusion, élection) et imprime ce que l'élection décide, plus les
douze premiers passages fusionnés pour voir où se trouvait le bon document.

Usage (dans le conteneur) :
    docker compose exec web python -m app.scripts.sim_election_before_after
    docker compose exec web python -m app.scripts.sim_election_before_after --only lumine,perform
"""
from __future__ import annotations

import argparse
import asyncio
import time
from typing import Dict, List, Optional, Sequence, Tuple

from sqlmodel import Session, text

from app.config import settings
from app.database import engine
from app.services.document_election_service import elect_documents, format_election_log
from app.services.page_retrieval_service import fuse_multimodal_hits
from app.services.space_search_service import _run_retrievers

SPACE = 28
POOL = max(settings.RERANK_POOL, settings.RAG_POOL_SIZE, settings.RAG_TOP_K)

# Périmètre LUMINE utilisé par la simulation du 01/09 (cas témoin).
PERIM_LUMINE = [383, 389, 390, 391, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403,
                404, 405, 406, 407, 409, 411, 412, 413, 415, 421, 423, 424, 425, 429, 432, 439]

# (clé, libellé, question, périmètre, documents attendus — d'après les mesures consignées)
CASES: List[Tuple[str, str, str, Optional[List[int]], List[int]]] = [
    ("lumine", "LUMINE 55 — acoustique (ambigu : Lumine55 #439 porte les 40 dB)",
     "Quelle est la performance acoustique maximale de la gamme LUMINE 55 en dB ?",
     PERIM_LUMINE, [439]),
    ("perform", "Perform — couleurs (le dépliant #425 et le catalogue #424 portent les couleurs)",
     "donne moi les couleurs dispo pour la gamme perform",
     None, [425, 424]),
    ("tgy", "TGY3704 — rallonge (référence visible qu'en image, catalogue SOLEAL #397)",
     "rallonge TGY3704",
     None, [397]),
    ("roto", "Roto NX — OF → OB (cas sain, doit rester net sur #383)",
     "Comment transformer un ouvrant a la francaise en oscillo-battant sur ferrure Roto NX ?",
     None, [383]),
    ("gammes", "Gammes / matériaux (question large : le catalogue a raison)",
     "Quelles gammes de menuiseries proposez-vous et pour quels materiaux ?",
     None, [424]),
    ("parclose", "Perform76 — parclose 2452 (planche p.8 du dossier technique #438)",
     "épaisseur de vitrage pour une parclose 2452",
     None, [438]),
]


def _space_docs(session: Session) -> List[int]:
    rows = session.execute(
        text("SELECT d.id FROM document d JOIN document_space ds ON ds.document_id = d.id "
             "WHERE ds.space_id = :sp"),
        {"sp": SPACE},
    ).all()
    return [int(r[0]) for r in rows]


def _titles(session: Session, ids: Sequence[int]) -> Dict[int, str]:
    if not ids:
        return {}
    rows = session.execute(
        text("SELECT id, title FROM document WHERE id = ANY(:ids)"), {"ids": list(ids)}
    ).all()
    return {int(r[0]): (r[1] or "") for r in rows}


async def run_case(session: Session, key: str, label: str, query: str,
                   scope: Optional[List[int]], expected: List[int]) -> Dict[str, object]:
    print("\n" + "=" * 100)
    print(f"[{key}] {label}\n  question : « {query} »")
    doc_ids = scope or _space_docs(session)
    t0 = time.perf_counter()
    cache: Dict[str, object] = {}
    cp, bm = await _run_retrievers(
        session, SPACE, doc_ids, query, query, POOL, use_colpali=True, embed_cache=cache
    )
    fused = fuse_multimodal_hits(cp, bm, rrf_k=settings.RRF_K, top_k=POOL)
    dt = time.perf_counter() - t0
    titles = _titles(session, sorted({int(h.document_id) for h in fused}))

    print(f"  pool fusionné : {len(fused)} pages ({len(cp)} ColPali, {len(bm)} BM25) en {dt:.1f}s")
    print("  douze premiers passages (rang · doc · page · rrf · colpali · bm25 · canaux) :")
    for rank, h in enumerate(fused[:12], start=1):
        mark = " ◄ attendu" if int(h.document_id) in expected else ""
        print(
            f"   {rank:2d}. #{int(h.document_id):<4} p.{int(h.page_no):<3} rrf={float(h.rrf_score or 0):.4f} "
            f"cp={float(h.colpali_score or 0):.3f} bm={float(h.bm25_score or 0):.2f} "
            f"[{'+'.join(h.retrieval_sources or [])}] {titles.get(int(h.document_id), '')[:38]}{mark}"
        )

    result = elect_documents(session, fused, query_text=query, signals=None,
                             max_docs=settings.CAG_MAX_DOCUMENTS)
    print("  " + format_election_log(result))
    elected = result.elected_ids
    found = [d for d in expected if d in elected]
    cand_only = [c.document_id for c in result.candidates
                 if c.document_id in expected and c.document_id not in elected]
    verdict = "OK" if found else ("CANDIDAT NON ÉLU" if cand_only else "ABSENT DES CANDIDATS")
    print(f"  élus={elected} attendus={expected} → {verdict}"
          + (f" (candidat(s) non élu(s) : {cand_only})" if cand_only else ""))
    return {"key": key, "elected": elected, "expected": expected, "verdict": verdict,
            "decision": result.decision, "seconds": round(dt, 1)}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="clés séparées par des virgules")
    args = parser.parse_args()
    only = {k.strip() for k in args.only.split(",") if k.strip()}

    summary: List[Dict[str, object]] = []
    with Session(engine) as session:
        for key, label, query, scope, expected in CASES:
            if only and key not in only:
                continue
            try:
                summary.append(await run_case(session, key, label, query, scope, expected))
            except Exception as exc:  # noqa: BLE001
                print(f"  ÉCHEC : {exc!r}")
                summary.append({"key": key, "verdict": f"ERREUR {exc!r}"})

    print("\n" + "=" * 100)
    print("RÉSUMÉ")
    for s in summary:
        print(f"  {s['key']:<9} {s.get('verdict', ''):<22} élus={s.get('elected')} "
              f"décision={s.get('decision')} ({s.get('seconds')}s)")


if __name__ == "__main__":
    asyncio.run(main())
