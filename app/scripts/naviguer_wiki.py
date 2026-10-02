"""Naviguer le wiki avec les outils de LIA, et rien d'autre — pour un banc où Claude joue LIA.

Chaque commande exécute le code du chat lui-même (``WikiAnswer._executer``) : même index, même
carte, même lecture (sections avec sommaire), même rapprochement des anomalies après une lecture.
Ce qui sort ici est ce que LIA reçoit, au caractère près, y compris le message de l'utilisateur et
sa fiche au premier appel. Une session par question garde l'état du tour (pages lues, ce qui a
déjà été livré, anomalies déjà poussées, appels) dans un fichier JSON et journalise chaque appel ;
le résultat de chaque appel est écrit à côté (``<session>.appelN.txt``), en entier.

Usage (dans le conteneur) ::

    python -m app.scripts.naviguer_wiki prompt
    python -m app.scripts.naviguer_wiki --session logs/bench/x/q01.json --question "…" \\
        chercher '{"requetes": ["crémone TGY3702"]}'
    python -m app.scripts.naviguer_wiki --session logs/bench/x/q01.json lire '{"lectures": [{"chemin": "/…"}]}'
    python -m app.scripts.naviguer_wiki --session logs/bench/x/q01.json lire \\
        '{"lectures": [{"chemin": "/…", "sections": ["§13"]}]}'

Budget : cinq appels d'outil par question (LIA : six appels du modèle, le dernier répond).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.services.wiki_chat_service import WikiAnswer
from app.services.wiki_index import Livraison
from app.services.wiki_service import load_snapshot, wiki_root

BUDGET = 5
OUTILS = ("chercher", "lire", "lire_anomalie")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", help="Fichier d'état de la question (JSON).")
    parser.add_argument("--question", help="La question, au premier appel de la session.")
    parser.add_argument("outil", choices=("prompt",) + OUTILS)
    parser.add_argument("args", nargs="?", default="{}", help="Arguments de l'outil, en JSON.")
    ns = parser.parse_args()

    snapshot = load_snapshot(wiki_root())
    if ns.outil == "prompt":
        print(snapshot.system_prompt)
        return 0
    if not ns.session:
        parser.error("--session est requis pour un outil")

    chemin = Path(ns.session)
    etat = json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else {
        "question": ns.question or "", "pages_lues": [], "anomalies": [], "appels": [],
    }
    if not etat["question"]:
        parser.error("--question est requis au premier appel")
    if len(etat["appels"]) >= BUDGET:
        print(f"BUDGET ÉPUISÉ ({BUDGET} appels) : réponds maintenant avec ce que tu as lu.")
        return 0

    # « - » : les arguments arrivent sur l'entrée standard (une apostrophe dans les mots-clés
    # casse la ligne de commande).
    args = json.loads(sys.stdin.read() if ns.args == "-" else ns.args)
    answer = WikiAnswer(question=etat["question"], history=[], snapshot=snapshot,
                        livraison=Livraison.depuis_dict(etat.get("livraison")))
    par_chemin = {e["chemin"]: e for e in answer.index.entries}
    pages_lues = [par_chemin[c] for c in etat["pages_lues"] if c in par_chemin]
    deja = set(etat["anomalies"])
    avant = len(pages_lues)

    resultat = answer._executer(ns.outil, args, pages_lues)
    resultat += answer._pied(len(etat["appels"]) + 1)
    injecte: list = []
    # Les anomalies suivent une lecture, jamais une simple carte.
    if pages_lues:
        answer._injecter_anomalies(injecte, pages_lues, deja)

    sortie = resultat
    if not etat["appels"]:
        sortie = "[message de l'utilisateur]\n" + answer.message_utilisateur() + "\n\n[résultat de l'outil]\n" + sortie
    if injecte:
        sortie += "\n\n[message système]\n" + injecte[0]["content"]
    # Une recherche livre souvent plus de 60 000 caractères : un terminal les tronquerait, alors
    # que LIA les reçoit en entier. Le résultat va dans un fichier, lu en entier par qui joue LIA.
    n = len(etat["appels"]) + 1
    fichier = chemin.with_name(f"{chemin.stem}.appel{n}.txt")
    fichier.parent.mkdir(parents=True, exist_ok=True)
    fichier.write_text(sortie, encoding="utf-8")
    print(f"Résultat de {ns.outil} : {fichier} ({len(sortie):,} car., {sortie.count(chr(10)) + 1} lignes)"
          " — lis-le EN ENTIER, c'est ce que LIA reçoit.")

    etat["pages_lues"] = [p["chemin"] for p in pages_lues]
    etat["anomalies"] = sorted(deja)
    etat["livraison"] = answer.livraison.vers_dict()
    etat["appels"].append({
        "outil": ns.outil,
        "args": args,
        "pages_nouvelles": etat["pages_lues"][avant:],
        "caracteres": len(sortie),
    })
    chemin.write_text(json.dumps(etat, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[appel {len(etat['appels'])}/{BUDGET}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
