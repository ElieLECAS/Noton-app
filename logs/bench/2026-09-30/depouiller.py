"""Dépouillement du banc « Claude imitant LIA » contre « Claude normal » (30/09/2026).

Dans le conteneur : python logs/bench/2026-09-30/depouiller.py
Écrit trouvabilite.json et navigation.json à côté, et imprime les tableaux.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/app")
from app.services.wiki_index import trier_resultats  # noqa: E402
from app.services.wiki_service import load_snapshot, wiki_root  # noqa: E402

B = Path(__file__).resolve().parent
idx = load_snapshot(wiki_root()).index
questions = json.loads((B / "questions_reelles.json").read_text(encoding="utf-8"))
gen = B / "questions_generees.json"
if gen.exists():
    questions += json.loads(gen.read_text(encoding="utf-8"))


def rang(requete, attendues):
    res = idx.search(mots_cles=requete, limite=15)
    for i, p in enumerate(res, 1):
        if p["chemin"] in attendues:
            return i
    return None


trouv, nav = [], []
for q in questions:
    qid = q["id"]
    normal = B / "normal" / f"{qid}.json"
    attendues = set()
    formulations = []
    if normal.exists():
        n = json.loads(normal.read_text(encoding="utf-8"))
        attendues = {p["page"] for p in n.get("preuves", []) if p.get("page")}
        formulations = n.get("formulations", [])
    if q.get("page"):
        attendues.add(q["page"])
    if attendues:
        rangs = {"question": rang(q["question"], attendues)}
        for k, f in enumerate(formulations[:3], 1):
            rangs[f"f{k}"] = rang(f, attendues)
        trouv.append({"id": qid, "attendues": sorted(attendues), "rangs": rangs})

    session = B / "lia" / f"{qid}.json"
    reponse = B / "lia" / f"{qid}.reponse.json"
    if session.exists():
        s = json.loads(session.read_text(encoding="utf-8"))
        r = json.loads(reponse.read_text(encoding="utf-8")) if reponse.exists() else {}
        lues = set(s["pages_lues"])
        nav.append({
            "id": qid,
            "appels": len(s["appels"]),
            "caracteres": sum(a["caracteres"] for a in s["appels"]),
            "requetes": [a["args"].get("mots_cles") or a["args"].get("chemin") or a["args"].get("identifiant")
                         for a in s["appels"]],
            "page_attendue_lue": bool(attendues & lues) if attendues else None,
            "trouve": r.get("trouve"),
        })

(B / "trouvabilite.json").write_text(json.dumps(trouv, ensure_ascii=False, indent=2), encoding="utf-8")
(B / "navigation.json").write_text(json.dumps(nav, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- tableaux
def top3(r):
    return r is not None and r <= 3

tous = [r for t in trouv for r in t["rangs"].values()]
print(f"TROUVABILITÉ — {len(trouv)} questions avec page attendue, {len(tous)} formulations")
print(f"  question brute : top 3 {sum(top3(t['rangs']['question']) for t in trouv)}/{len(trouv)}")
print(f"  toutes formulations : top 3 {sum(top3(r) for r in tous)}/{len(tous)} "
      f"({100 * sum(top3(r) for r in tous) / max(1, len(tous)):.0f} %), absentes du top 15 : {sum(r is None for r in tous)}")
for t in trouv:
    print(f"  {t['id']:<4} {str(t['rangs']):<60} {', '.join(t['attendues'])[:90]}")

print(f"\nNAVIGATION CLAUDE-LIA — {len(nav)} questions")
if nav:
    print(f"  appels : médiane {statistics.median(n['appels'] for n in nav)}, max {max(n['appels'] for n in nav)}")
    print(f"  caractères lus : médiane {statistics.median(n['caracteres'] for n in nav):,.0f}, "
          f"max {max(n['caracteres'] for n in nav):,}")
    ok = [n for n in nav if n["page_attendue_lue"] is not None]
    print(f"  page attendue lue : {sum(n['page_attendue_lue'] for n in ok)}/{len(ok)}")
    for n in nav:
        print(f"  {n['id']:<4} {n['appels']} app. {n['caracteres']:>8,} car. lue={n['page_attendue_lue']!s:<5} "
              f"trouve={n['trouve']!s:<8} {' | '.join(str(x) for x in n['requetes'])[:110]}")
