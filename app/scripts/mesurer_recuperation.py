"""Mesure hors ligne de la récupération — sans modèle, en quelques dizaines de secondes.

Rejoue la première recherche de LIA (``WikiIndex.classer`` puis ``WikiIndex.livrer``, le code du
chat lui-même) avec le texte de la question comme requête, et vérifie ce qui arrive au modèle :

  * **preuve livrée**     : au moins un extrait de preuve figure dans ce qui est livré ;
  * **toutes les preuves** : chaque page de preuve a au moins un de ses extraits livré ;
  * **pages utiles**      : les pages attendues sont livrées, entières ou en sections ;
  * **volume**            : caractères livrés, nombre de pages.

Deux jeux : le golden 40 questions (``tests/fixtures/golden/golden_40_questions.json``) et le banc
du 30/09 (``logs/bench/2026-09-30`` : questions, pages attendues, extraits de preuve, trois
reformulations par question). Comparé à la livraison historique (« 3 pages entières »,
reconstituée ici) et sur une grille de réglages.

Depuis le 02/10 il mesure aussi **la carte** (``WikiIndex.carte``), ce que lit un modèle qui
navigue : la question seule, puis la question et ses reformulations fusionnées (``classer_multi``).
Trois mesures y répondent à une autre question que « la preuve est-elle livrée ? » :

  * **section désignée** : la section qui porte la preuve figure-t-elle parmi celles que la carte
    liste pour ses pages ? (le modèle n'a plus qu'à la lire) ;
  * **dans les lignes**  : l'extrait de preuve est-il déjà sous les yeux, dans les lignes de la carte ?
  * **fiche**            : pour une référence rare, la fiche de la question désigne-t-elle la section ?

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
    BUDGET_RECHERCHE,
    CARTE_SECTIONS,
    PAGE_ENTIERE_MAX,
    PAGES_PAR_RECHERCHE,
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


def livraison_historique(index: WikiIndex, question: str) -> Tuple[str, List[str]]:
    """Ce que LIA recevait avant le 01/10 : les trois premières pages entières, concepts d'abord."""
    pages = index.search(mots_cles=question, limite=15, cotes=cotes_de(question))
    retenues = [p for p in pages if not p["chemin"].startswith("/anomalies/")]
    entieres = ([p for p in retenues if not p["chemin"].startswith("/sources/")]
                + [p for p in retenues if p["chemin"].startswith("/sources/")])[:3]
    return "\n\n".join(p["corps"] for p in entieres), [p["chemin"] for p in entieres]


def livraison_actuelle(index: WikiIndex, question: str, **reglages: Any) -> Tuple[str, List[str]]:
    texte, livrees = index.livrer(index.classer(question, cotes=cotes_de(question)), Livraison(), **reglages)
    principal = texte.split("===== AUTRES RÉSULTATS =====")[0]
    return principal, [l["chemin"] for l in livrees]


def evaluer(items: Sequence[Dict[str, Any]], livrer, reformulations: bool = False) -> Dict[str, float]:
    n = preuve = toutes = pages = volume = nb_pages = avec_preuves = avec_pages = 0
    for item in items:
        questions = [item["question"]] + (list(item.get("reformulations") or []) if reformulations else [])
        for question in questions:
            if not question:
                continue
            texte, livrees = livrer(question)
            plat_texte = plat(texte)
            n += 1
            volume += len(texte)
            nb_pages += len(livrees)
            if item["pages"]:
                avec_pages += 1
                pages += all(p in livrees for p in item["pages"])
            if item["preuves"]:
                avec_preuves += 1
                trouves = {p["page"] for p in item["preuves"] if plat(p["extrait"])[:60] in plat_texte}
                preuve += bool(trouves)
                toutes += trouves >= {p["page"] for p in item["preuves"]}
    pct = lambda a, b: 100 * a / b if b else 0.0  # noqa: E731
    return {
        "questions": n,
        "preuve": pct(preuve, avec_preuves),
        "toutes": pct(toutes, avec_preuves),
        "pages": pct(pages, avec_pages),
        "volume": volume / n if n else 0,
        "nb_pages": nb_pages / n if n else 0,
    }


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
        livre, _ = index.livrer([dominante], livraison, pages_max=1, autres_max=0)
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


def ligne(nom: str, r: Dict[str, float]) -> str:
    return (f"  {nom:38} preuve {r['preuve']:5.1f} % | toutes {r['toutes']:5.1f} % | pages utiles "
            f"{r['pages']:5.1f} % | {r['volume'] / 1000:5.1f} k car. | {r['nb_pages']:.1f} pages")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grille", action="store_true", help="Balayer seuil de page entière et nombre de pages.")
    ns = parser.parse_args()
    index = load_snapshot(wiki_root()).index
    for nom, items in jeux().items():
        if not items:
            continue
        print(f"\n=== {nom} ({len(items)} questions)")
        print(ligne("historique : 3 pages entières", evaluer(items, lambda q: livraison_historique(index, q))))
        print(ligne(f"actuel ({PAGE_ENTIERE_MAX // 1000} k, {PAGES_PAR_RECHERCHE} pages, {BUDGET_RECHERCHE // 1000} k)",
                    evaluer(items, lambda q: livraison_actuelle(index, q))))
        if nom == "banc":
            print(ligne("actuel, avec les 3 reformulations", evaluer(items, lambda q: livraison_actuelle(index, q), True)))
        print(ligne_carte("carte, question seule", evaluer_carte(index, items)))
        if nom == "banc":
            print(ligne_carte("carte, 4 formulations fusionnées", evaluer_carte(index, items, formulations=True)))
        if ns.grille:
            for pages in (8, 12, 15):
                print(ligne_carte(f"carte {pages} pages", evaluer_carte(index, items, pages=pages)))
            for seuil in (15_000, 20_000, 30_000):
                for pages in (4, 6, 8):
                    for budget in (40_000, 60_000, 100_000):
                        r = evaluer(items, lambda q: livraison_actuelle(index, q, seuil_page=seuil, pages_max=pages, budget=budget))
                        print(ligne(f"seuil {seuil // 1000} k, {pages} pages, budget {budget // 1000} k", r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
