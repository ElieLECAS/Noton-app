"""Propose la découpe des pages trop longues du wiki, sans en modifier aucune (lot L2).

Chaque page assertive au-dessus du budget est coupée **le long de ses propres titres** : les
sections ``# …`` sont rangées dans l'ordre de la source en parties qui tiennent dans le budget ;
une section qui dépasse seule le budget est coupée à ses ``## …`` ; ce qui ne se coupe plus est
signalé. Les trois sections de queue (« Ce que la source ne donne pas », « Citations », « Voir
aussi ») ne comptent pas : chaque partie aura les siennes.

Le résultat est une proposition de travail, pas un découpage : titres, descriptions, tags et
phrases d'ouverture des parties s'écrivent à la main (protocole, *Writing to be found*).

Usage (dans le conteneur) ::

    python -m app.scripts.plan_decoupe [--budget 20000] [--cible 16000] [--json logs/l2/plan_decoupe.json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from app.services.wiki_service import load_snapshot, wiki_root

QUEUE = ("ce que la source ne donne pas", "ce que le document ne dit pas", "citations", "voir aussi")
EXEMPTES = ("/sources/", "/anomalies/")


def sections(corps: str, niveau: str) -> List[Dict[str, Any]]:
    """Coupe ``corps`` aux titres de ce niveau ; le texte avant le premier titre est le préambule."""
    motif = re.compile(rf"^{niveau} (.+)$", re.MULTILINE)
    bornes = [(m.start(), m.group(1).strip()) for m in motif.finditer(corps)]
    if not bornes:
        return [{"titre": "(préambule)", "taille": len(corps), "texte": corps}]
    out = []
    if bornes[0][0] > 0:
        out.append({"titre": "(préambule)", "taille": bornes[0][0], "texte": corps[: bornes[0][0]]})
    for k, (debut, titre) in enumerate(bornes):
        fin = bornes[k + 1][0] if k + 1 < len(bornes) else len(corps)
        out.append({"titre": titre, "taille": fin - debut, "texte": corps[debut:fin]})
    return out


def eclater(sec: Dict[str, Any], cible: int) -> List[Dict[str, Any]]:
    """Une section plus grosse que la cible : on la coupe à ses ``##``, sinon elle reste seule."""
    if sec["taille"] <= cible:
        return [sec]
    sous = sections(sec["texte"], "##")
    if len(sous) <= 1:
        return [dict(sec, indivisible=True)]
    return [dict(s, titre=f"{sec['titre']} › {s['titre']}", parent=sec["titre"]) for s in sous]


def plan_page(corps: str, cible: int) -> Dict[str, Any]:
    secs = [s for s in sections(corps, "#") if s["titre"].lower() not in QUEUE]
    plat: List[Dict[str, Any]] = []
    for s in secs:
        plat.extend(eclater(s, cible))
    parties: List[Dict[str, Any]] = []
    courante: Dict[str, Any] = {"sections": [], "taille": 0}
    for s in plat:
        if courante["sections"] and courante["taille"] + s["taille"] > cible:
            parties.append(courante)
            courante = {"sections": [], "taille": 0}
        courante["sections"].append({k: s[k] for k in ("titre", "taille") if k in s} | (
            {"indivisible": True} if s.get("indivisible") else {}))
        courante["taille"] += s["taille"]
    if courante["sections"]:
        parties.append(courante)
    return {"parties": parties}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--budget", type=int, default=20000, help="taille au-dessus de laquelle on coupe")
    parser.add_argument("--cible", type=int, default=16000, help="taille visée par partie (marge pour l'ouverture et la queue)")
    parser.add_argument("--json", default="logs/l2/plan_decoupe.json")
    ns = parser.parse_args()

    snap = load_snapshot(wiki_root())
    plans = []
    for pid, page in sorted(snap.pages.items()):
        if not page.is_concept or page.missing or pid.startswith(EXEMPTES) or len(page.body) <= ns.budget:
            continue
        plan = plan_page(page.body, ns.cible)
        plan.update(page=pid, taille=len(page.body), type=page.type or "-", parties_n=len(plan["parties"]),
                    indivisibles=[s["titre"] for p in plan["parties"] for s in p["sections"] if s.get("indivisible")])
        plans.append(plan)
    plans.sort(key=lambda p: -p["taille"])

    chemin = Path(ns.json)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(plans, ensure_ascii=False, indent=2), encoding="utf-8")

    n = sum(p["parties_n"] for p in plans)
    print(f"{len(plans)} pages au-dessus de {ns.budget:,} car. → {n} parties (cible {ns.cible:,}) : "
          f"+{n - len(plans)} pages, wiki de {len(snap.pages)} à ~{len(snap.pages) + n - len(plans)}")
    print(f"pages avec une section qui ne se coupe plus : {sum(1 for p in plans if p['indivisibles'])}")
    for p in plans:
        tailles = " ".join(f"{q['taille'] // 1000}k" for q in p["parties"])
        drapeau = "  ⚠ " + "; ".join(t[:38] for t in p["indivisibles"][:2]) if p["indivisibles"] else ""
        print(f"{p['taille']:>8,}  {p['parties_n']:>2} p.  [{tailles}]  {p['page']}{drapeau}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
