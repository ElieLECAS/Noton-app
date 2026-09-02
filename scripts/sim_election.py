"""Simulation AVANT/APRES de l'election sur de VRAIS pools de retrieval (espace 28).

AVANT = formule actuelle (document_election_service._compute_election_score + mono-doc)
APRES = L1 cercle de contention + L2 pages qualifiees + L4 largeur adaptative + L5 fusion union

Aucune modification du code de prod : tout est recalcule ici a partir des hits fusionnes.
"""
import asyncio, sys
import numpy as np
from sqlmodel import Session
from app.database import engine
from app.config import settings
from app.services.space_search_service import _run_retrievers
from app.services.page_retrieval_service import fuse_multimodal_hits
from app.services.document_election_service import score_documents, ELECTION_PAGE_CAP

SPACE = 28
POOL, TOPK, IMAGES = 40, 20, 8
DELTA = 0.05        # L1 : cercle de contention (~1 rang RRF)
PEAK_GAP = 0.10     # L4 : palier ColPali (largeur du filtre dynamique)
Q_BM25 = 0.4        # L2 : barre de qualification bm25 (relative au pool)
Q_RRF = 0.5
RRF_GAP = 0.15     # L4 : ecart relatif de pic RRF dominant/dauphin (~3 rangs)         # L2 : barre de qualification rrf (relative au doc)

PERIM_LUMINE = [383,389,390,391,393,394,395,396,397,398,399,400,401,402,403,404,
                405,406,407,409,411,412,413,415,421,423,424,425,429,432,439]

CASES = [
    ("A. Cas temoin (ambigu)", "Quelle est la performance acoustique maximale de la gamme LUMINE 55 en dB ?", PERIM_LUMINE),
    ("B. Notice specifique (doit rester mono)", "Comment transformer un ouvrant a la francaise en oscillo-battant sur ferrure Roto NX ?", None),
    ("C. Question generaliste (le catalogue a raison)", "Quelles gammes de menuiseries proposez-vous et pour quels materiaux ?", None),
]


def space_docs(s):
    from sqlmodel import text
    return [int(r[0]) for r in s.execute(text(
        "SELECT d.id FROM document d JOIN document_space ds ON ds.document_id=d.id WHERE ds.space_id=:sp"
    ), {"sp": SPACE}).all()]


def titles(s, ids):
    from sqlmodel import text
    if not ids: return {}
    rows = s.execute(text("SELECT id, title FROM document WHERE id IN :ids"), {"ids": tuple(ids)}).all()
    return {int(r[0]): r[1] for r in rows}


def agg(fused):
    """Agrege par document : pic rrf, pics bruts, pages brutes ET qualifiees."""
    pool_bm25 = max((float(h.bm25_score or 0) for h in fused), default=0.0)
    docs = {}
    for h in fused:
        d = docs.setdefault(int(h.document_id), {
            "score_max": 0.0, "colpali_best": 0.0, "bm25_best": 0.0,
            "families": set(), "pages_raw": {}, "pages_qual": {}})
        rrf = float(h.rrf_score or 0)
        d["score_max"] = max(d["score_max"], rrf)
        d["colpali_best"] = max(d["colpali_best"], float(h.colpali_score or 0))
        d["bm25_best"] = max(d["bm25_best"], float(h.bm25_score or 0))
        for src in (h.retrieval_sources or []):
            d["families"].add({"bm25": "texte", "colpali": "visuel"}.get(src, src))
        p = int(h.page_no)
        d["pages_raw"][p] = max(d["pages_raw"].get(p, 0.0), rrf)
    for did, d in docs.items():
        for h in fused:
            if int(h.document_id) != did: continue
            p, rrf = int(h.page_no), float(h.rrf_score or 0)
            seen_colpali = "colpali" in (h.retrieval_sources or [])
            strong_bm25 = float(h.bm25_score or 0) >= Q_BM25 * pool_bm25 if pool_bm25 else False
            strong_rrf = rrf >= Q_RRF * d["score_max"] if d["score_max"] else False
            if seen_colpali or strong_bm25 or strong_rrf:
                d["pages_qual"][p] = max(d["pages_qual"].get(p, 0.0), rrf)
    return docs


def elect_score(base, n_fam, n_pages, cap):
    if base <= 0: return base
    bonus = (settings.CAG_ELECTION_FAMILY_BONUS * (max(1, n_fam) - 1)
             + settings.CAG_ELECTION_PAGE_BONUS * min(max(1, n_pages) - 1, cap))
    return base * (1.0 + bonus)


def before(docs):
    out = []
    for did, d in docs.items():
        out.append((did, elect_score(d["score_max"], len(d["families"]), len(d["pages_raw"]), ELECTION_PAGE_CAP), d))
    out.sort(key=lambda t: (-t[1], -t[2]["score_max"], t[0]))
    return out


def after(docs):
    scored = []
    for did, d in docs.items():
        scored.append((did, elect_score(d["score_max"], len(d["families"]), len(d["pages_qual"]), ELECTION_PAGE_CAP), d))
    best_peak = max((d["score_max"] for _, _, d in scored), default=0.0)
    circle = [t for t in scored if t[2]["score_max"] >= (1 - DELTA) * best_peak]
    outside = [t for t in scored if t not in circle]
    circle.sort(key=lambda t: (-t[1], -t[2]["score_max"], t[0]))
    outside.sort(key=lambda t: (-t[1], -t[2]["score_max"], t[0]))
    ranked = circle + outside
    dom = ranked[0]
    peak_holder = max(scored, key=lambda t: t[2]["score_max"])[0]
    # CONFIANCE : ecart entre le dominant et son DAUPHIN (pas le max du pool).
    # Deux espaces bruts, l'un OU l'autre suffit :
    #   - ecart relatif de pic RRF >= 15 % (~3 rangs pleins : le retrieval est d'accord)
    #   - ecart de pic ColPali >= 0.10 (largeur de palier du filtre dynamique)
    if len(ranked) < 2:
        return ranked, [dom], "mono (candidat unique)", circle, {dom[0]: "dominant"}
    run = ranked[1]
    rrf_gap = ((dom[2]["score_max"] - run[2]["score_max"]) / dom[2]["score_max"]) if dom[2]["score_max"] else 0.0
    cp_gap = dom[2]["colpali_best"] - run[2]["colpali_best"]
    if rrf_gap >= RRF_GAP:
        return ranked, [dom], f"mono (ecart rrf dominant/dauphin {rrf_gap:.1%} >= {RRF_GAP:.0%})", circle, {dom[0]: "dominant"}
    if cp_gap >= PEAK_GAP:
        return ranked, [dom], f"mono (ecart pic ColPali {cp_gap:.3f} >= {PEAK_GAP})", circle, {dom[0]: "dominant"}
    elected, reasons = [dom], {dom[0]: "dominant"}
    for t in ranked[1:]:
        if len(elected) >= 3: break
        if t[1] < 0.70 * dom[1]: break
        if t[0] in {c[0] for c in circle}: reasons[t[0]] = "contention_tie"
        elif t[0] == peak_holder: reasons[t[0]] = "peak_holder"
        else: reasons[t[0]] = "peak_proximity"
        elected.append(t)
    return ranked, elected, f"multi x{len(elected)} (ambigu : rrf {rrf_gap:.1%} < {RRF_GAP:.0%} ET pic ColPali {cp_gap:+.3f} < {PEAK_GAP})", circle, reasons


async def scoped(s, doc_ids, cq, lq, cache):
    cp, bm = await _run_retrievers(s, SPACE, doc_ids, cq, lq, POOL, use_colpali=True, embed_cache=cache)
    return cp, bm


async def run_case(s, label, query, scope):
    print("\n" + "=" * 100)
    print(f"{label}\n  \"{query}\"")
    doc_ids = scope or space_docs(s)
    cache = {}
    cp, bm = await _run_retrievers(s, SPACE, doc_ids, query, query, POOL, use_colpali=True, embed_cache=cache)
    fused = fuse_multimodal_hits(cp, bm, rrf_k=settings.RRF_K, top_k=POOL)
    docs = agg(fused)
    T = titles(s, list(docs))

    b = before(docs)
    print(f"\n  Pool fusionne : {len(fused)} pages / {len(docs)} documents  (perimetre {len(doc_ids)} docs)")
    print("\n  --- AVANT (formule actuelle) ---")
    print(f"  {'doc':>5} {'pic rrf':>9} {'pgs':>4} {'fam':>4} {'score elu':>10}  titre")
    for did, sc, d in b[:5]:
        star = " *ELU*" if did == b[0][0] else "      "
        print(f"  {did:>5} {d['score_max']:>9.4f} {len(d['pages_raw']):>4} {len(d['families']):>4} {sc:>10.4f}{star} {T.get(did,'?')[:38]}")
    print(f"  -> elu : 1 document (mono force) = #{b[0][0]} {T.get(b[0][0],'?')[:40]}")

    ranked, elected, decision, circle, reasons = after(docs)
    circle_ids = {t[0] for t in circle}
    print("\n  --- APRES (L1 cercle + L2 pages qualifiees + L4 largeur adaptative) ---")
    print(f"  seuil du cercle : pic rrf >= {(1-DELTA):.2f} x {max(d['score_max'] for d in docs.values()):.4f} = {(1-DELTA)*max(d['score_max'] for d in docs.values()):.4f}")
    print(f"  {'doc':>5} {'pic rrf':>9} {'cercle':>7} {'pgs br':>7} {'pgs qu':>7} {'pic cp':>7} {'score elu':>10}  titre")
    for did, sc, d in ranked[:5]:
        inc = "OUI" if did in circle_ids else "non"
        print(f"  {did:>5} {d['score_max']:>9.4f} {inc:>7} {len(d['pages_raw']):>7} {len(d['pages_qual']):>7} {d['colpali_best']:>7.3f} {sc:>10.4f}  {T.get(did,'?')[:34]}")
    print(f"  -> DECISION : {decision}")
    for did, sc, d in elected:
        print(f"     elu #{did} [{reasons.get(did,'?')}] {T.get(did,'?')[:40]}")

    # PHASE B : MaxSim exact par elu, puis fusion UNION (L5)
    all_cp, all_bm = [], []
    for did, _, _ in elected:
        c, b2 = await scoped(s, [did], query, query, cache)
        all_cp.extend(c); all_bm.extend(b2)
    union = fuse_multimodal_hits(all_cp, all_bm, rrf_k=settings.RRF_K, top_k=TOPK)
    print(f"\n  --- {IMAGES} PAGES ENVOYEES AU LLM ---")
    cb, _ = await scoped(s, [b[0][0]], query, query, cache)
    ub = fuse_multimodal_hits(cb, [h for h in bm if int(h.document_id) == b[0][0]], rrf_k=settings.RRF_K, top_k=TOPK)
    print("  AVANT :", ", ".join(f"#{h.document_id}p{h.page_no}" for h in ub[:IMAGES]) or "(vide)")
    print("  APRES :", ", ".join(f"#{h.document_id}p{h.page_no}" for h in union[:IMAGES]) or "(vide)")
    docs_ap = {}
    for h in union[:IMAGES]: docs_ap[int(h.document_id)] = docs_ap.get(int(h.document_id), 0) + 1
    print("  repartition APRES :", ", ".join(f"{T.get(d,'?')[:24]}={n}p" for d, n in sorted(docs_ap.items(), key=lambda kv: -kv[1])))


async def main():
    with Session(engine) as s:
        for label, q, scope in CASES:
            try:
                await run_case(s, label, q, scope)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  ECHEC {label}: {e}")

asyncio.run(main())
