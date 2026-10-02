"""Mesure hors ligne de la récupération — sans modèle, en quelques dizaines de secondes.

Rejoue la carte de LIA (``WikiIndex.classer_multi`` puis ``WikiIndex.carte``, le code du chat
lui-même) sur deux jeux : le golden de 40 questions (``tests/fixtures/golden/golden_40_questions.json``)
et le banc du 30/09 (``logs/bench/2026-09-30`` : questions, pages attendues, extraits de preuve,
trois reformulations par question). Une carte est ce que lit un modèle qui navigue : douze
résultats, leurs sections et les lignes qui répondent. La mesure se fait sur la question seule,
puis sur la question et ses reformulations fusionnées. Trois mesures :

  * **section désignée** : la section qui porte la preuve figure-t-elle parmi celles que la carte
    liste pour ses pages ? (le modèle n'a plus qu'à la lire) ;
  * **dans les lignes**  : l'extrait de preuve est-il déjà sous les yeux, dans les lignes de la carte ?
  * **fiche**            : pour une référence rare, la fiche de la question désigne-t-elle la section ?

La livraison de six pages par recherche, qu'elle remplace, a été mesurée avec ce même outil le
02/10 : preuve livrée 95 % (golden) et 91,1 % (banc) pour ~52 000 caractères ; trois pages
entières, 95 % et 82,1 % pour ~69 000 (``docs/plan_glm_first_2026-10-02.md``).

Usage ::

    python -m app.scripts.mesurer_recuperation
    python -m app.scripts.mesurer_recuperation --grille
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from app.services.wiki_index import (
    CARTE_SECTIONS,
    Livraison,
    WikiIndex,
    cotes_de,
    deaccent,
)
from app.services.wiki_service import load_snapshot, wiki_root

RACINE = Path(__file__).resolve().parents[2]
GOLDEN = RACINE / "tests" / "fixtures" / "golden" / "golden_40_questions.json"
BANC = RACINE / "logs" / "bench" / "2026-09-30"


def plat(texte: str) -> str:
    """Le texte sans mise en forme : un extrait se retrouve quelle que soit la découpe."""
    return re.sub(r"\s+", " ", re.sub(r"[*|`\\#>_\[\]()]|<[^>]+>", " ", deaccent(texte.lower()))).strip()


def jeux() -> Dict[str, List[Dict[str, Any]]]:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["entries"]
    banc: List[Dict[str, Any]] = []
    if BANC.exists():
        questions = {
            q["id"]: q
            for f in ("questions_reelles.json", "questions_generees.json")
            for q in json.loads((BANC / f).read_text(encoding="utf-8"))
        }
        for t in json.loads((BANC / "trouvabilite.json").read_text(encoding="utf-8")):
            fiche = BANC / "normal" / f"{t['id']}.json"
            normal = json.loads(fiche.read_text(encoding="utf-8")) if fiche.exists() else {}
            banc.append({
                "id": t["id"],
                "question": questions[t["id"]]["question"],
                "pages": [p for p in t["attendues"] if not p.startswith("/anomalies/")],
                "preuves": [p for p in normal.get("preuves") or [] if p.get("page") and p.get("extrait")],
                "reformulations": (normal.get("formulations") or [])[:3],
            })
    return {"golden": golden, "banc": banc}


def sections_de_preuve(index: WikiIndex, preuves: Sequence[Dict[str, Any]]) -> Dict[str, Set[int]]:
    """Page de preuve → sections qui portent l'un de ses extraits."""
    par_page: Dict[str, Set[int]] = {}
    for p in preuves:
        cible = plat(p["extrait"])[:60]
        trouvees = {sid for sid in index.sections_par_page.get(p["page"], []) if cible in plat(index.sections[sid]["texte"])}
        par_page.setdefault(p["page"], set()).update(trouvees)
    return par_page


def carte_actuelle(index: WikiIndex, question: str, formulations: Sequence[str] = (), **reglages: Any) -> Tuple[str, List[str], Set[int]]:
    """La carte de la question (et de ses formulations, fusionnées), plus la page dominante si
    elle tient entière : ``(texte lu par le modèle, pages listées, sections désignées)``."""
    requetes = [question] + [f for f in formulations if f]
    classement = index.classer_multi(requetes, cotes=cotes_de(question))
    livraison = Livraison()
    texte, chemins = index.carte(classement, index.jetons_requete(requetes), livraison, **reglages)
    dominante = index.page_dominante(classement, livraison)
    if dominante is not None:
        livre, _ = index.page_entiere(dominante["entree"]["chemin"], livraison)
        texte += "\n\n" + livre
    designees = {sid for c in classement if c["entree"]["chemin"] in chemins for sid, _ in c["sections"][:CARTE_SECTIONS]}
    return texte, chemins, designees


def evaluer_carte(index: WikiIndex, items: Sequence[Dict[str, Any]], formulations: bool = False, **reglages: Any) -> Dict[str, float]:
    n = une = toutes = lignes = pages = volume = dominantes = fiche_n = fiche_ok = 0
    for item in items:
        if not item["preuves"]:
            continue
        n += 1
        texte, chemins, designees = carte_actuelle(
            index, item["question"], (item.get("reformulations") or [])[:3] if formulations else (), **reglages
        )
        volume += len(texte)
        dominantes += "page entière" in texte
        par_page = sections_de_preuve(index, item["preuves"])
        atteintes = {page: bool(sids & designees) for page, sids in par_page.items()}
        une += any(atteintes.values())
        toutes += all(atteintes.values())
        lignes += any(plat(p["extrait"])[:60] in plat(texte) for p in item["preuves"])
        attendues = [p for p in item["pages"] if not p.startswith("/anomalies/")]
        pages += all(p in chemins for p in attendues) if attendues else 0
        fiche = index.fiche_question(item["question"])
        if fiche["rares"]:
            fiche_n += 1
            touchees = {(e["chemin"], e["section"]) for liste in fiche["emplacements"].values() for e in liste}
            visees = {(index.sections[sid]["chemin"], index.sections[sid]["numero"]) for sids in par_page.values() for sid in sids}
            fiche_ok += bool(touchees & visees)
    pct = lambda a, b: 100 * a / b if b else 0.0  # noqa: E731
    return {
        "questions": n, "une": pct(une, n), "toutes": pct(toutes, n), "lignes": pct(lignes, n),
        "pages": pct(pages, n), "volume": volume / n if n else 0, "dominantes": pct(dominantes, n),
        "fiche": pct(fiche_ok, fiche_n), "fiche_n": fiche_n,
    }


def ligne_carte(nom: str, r: Dict[str, float]) -> str:
    return (f"  {nom:38} section désignée {r['une']:5.1f} % | toutes {r['toutes']:5.1f} % | dans les lignes "
            f"{r['lignes']:5.1f} % | {r['volume'] / 1000:5.1f} k car. | page dominante {r['dominantes']:4.1f} % "
            f"| fiche {r['fiche']:5.1f} % ({r['fiche_n']} q.)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grille", action="store_true", help="Balayer le nombre de pages de la carte.")
    ns = parser.parse_args()
    index = load_snapshot(wiki_root()).index
    for nom, items in jeux().items():
        if not items:
            continue
        print(f"\n=== {nom} ({len(items)} questions)")
        print(ligne_carte("carte, question seule", evaluer_carte(index, items)))
        if nom == "banc":
            print(ligne_carte("carte, 4 formulations fusionnées", evaluer_carte(index, items, formulations=True)))
        if ns.grille:
            for pages in (8, 12, 15):
                print(ligne_carte(f"carte {pages} pages", evaluer_carte(index, items, pages=pages)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
