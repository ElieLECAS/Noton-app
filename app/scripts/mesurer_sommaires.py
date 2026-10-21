"""Mesure hors ligne des sommaires (N0 / N1) — sans modèle, en quelques secondes.

Les sommaires (``app/services/wiki_sommaires.py``) remplaceraient ``index.md`` dans le prompt
permanent : la racine (N0) y resterait, un sommaire par produit (N1) se lirait à la demande. Avant
de les brancher, trois questions, sur le golden de 40 questions et le banc du 30/09 (les jeux de
``mesurer_recuperation``) :

  * **volume** : ce que pèsent la racine et chaque sommaire, contre l'index plat ;
  * **couverture** : combien de sommaires faut-il ouvrir pour avoir sous les yeux la ligne de
    chaque page attendue, et combien de caractères cela coûte ;
  * **pointage** : rang de la page attendue quand on classe (BM25 sur les lignes) tout l'index
    plat, puis seulement les sommaires utiles (oracle), puis ceux que la question nomme
    (« routage » lexical par la racine : un nom de produit ou de gamme dans la question).

Le pointage est un indicateur lexical : un modèle lit les lignes, il ne les classe pas par BM25.
Il dit si un sommaire rend la bonne ligne plus facile à trouver, pas si le modèle la trouvera.

Usage ::

    python -m app.scripts.mesurer_sommaires
"""
from __future__ import annotations

import math
import re
from collections import Counter
from statistics import median
from typing import Dict, List, Sequence, Set

from app.scripts.mesurer_recuperation import jeux
from app.services.wiki_index import tokenise
from app.services.wiki_service import CHARS_PER_TOKEN, load_snapshot, wiki_root
from app.services.wiki_sommaires import Sommaires, sommaires, systemes_des_gammes

LIGNE_RE = re.compile(r"^\* \[[^\]]*\]\((/[^)]+\.md)\)", re.M)


def k_tokens(caracteres: float) -> str:
    return f"{caracteres / CHARS_PER_TOKEN / 1000:5.1f} k tokens"


def lignes_de(texte: str) -> Dict[str, str]:
    """Chemin → ligne, pour chaque ligne de page d'un index ou d'un sommaire."""
    return {m.group(1): texte[m.start():texte.find("\n", m.start())] for m in LIGNE_RE.finditer(texte)}


class Bm25:
    """BM25 sur des lignes, IDF calculé une fois sur l'index plat : les rangs restent comparables
    d'un sous-ensemble à l'autre."""

    def __init__(self, lignes: Dict[str, str]):
        self.docs = {c: tokenise(l) for c, l in lignes.items()}
        self.moyenne = sum(map(len, self.docs.values())) / len(self.docs)
        df = Counter(t for d in self.docs.values() for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - k + 0.5) / (k + 0.5)) for t, k in df.items()}

    def rang(self, question: str, candidats: Sequence[str], attendues: Set[str]) -> int:
        q = tokenise(question)
        scores = []
        for c in candidats:
            d = self.docs.get(c)
            if d is None:
                continue
            tf, long_ = Counter(d), len(d)
            s = sum(self.idf.get(t, 0) * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * long_ / self.moyenne))
                    for t in q if t in tf)
            scores.append((s, c))
        classes = [c for s, c in sorted(scores, reverse=True) if s > 0]
        return min((i for i, c in enumerate(classes) if c in attendues), default=10_000)


def couverture_minimale(attendues: Set[str], s: Sommaires) -> List[str]:
    """Le plus petit jeu (glouton) de sommaires dont les lignes couvrent toutes les pages attendues."""
    reste, choisis = set(attendues), []
    while reste:
        meilleur = max(s.par_chemin.values(), key=lambda x: (len(reste & {p.id for p in x.pages}), -len(x.texte)))
        gagnees = reste & {p.id for p in meilleur.pages}
        if not gagnees:
            break
        choisis.append(meilleur.chemin)
        reste -= gagnees
    return choisis


def routage(question: str, s: Sommaires, gammes: Dict[str, List[str]]) -> List[str]:
    """Les sommaires que la question désigne par un nom de la racine : le produit lui-même (« 76 »,
    « SOLEAL FY »), ou une gamme et ses systèmes (« PERFORM » → 76, 70)."""
    mots = set(tokenise(question))
    choisis: List[str] = []
    for x in s.par_chemin.values():
        nom = set(tokenise(x.produit))
        if nom and nom <= mots:
            choisis.append(x.chemin)
            for systeme in gammes.get(x.produit, []):
                y = s.du_produit(systeme)
                if y is not None and y.chemin not in choisis:
                    choisis.append(y.chemin)
    for g, systemes in gammes.items():
        if set(tokenise(g)) <= mots:
            for systeme in systemes:
                y = s.du_produit(systeme)
                if y is not None and y.chemin not in choisis:
                    choisis.append(y.chemin)
    return choisis


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:5.1f} %" if b else "  -  "


def main() -> None:
    snap = load_snapshot(wiki_root())
    s = sommaires(snap.pages.values())
    gammes = systemes_des_gammes(snap.pages.values())
    index = snap.pages["/index.md"].raw_text
    plat = lignes_de(index)
    bm25 = Bm25(plat)

    tailles = sorted(len(x.texte) for x in s.par_chemin.values())
    print("=== volume")
    print(f"  index plat (prompt actuel)   {len(index):8,} car. {k_tokens(len(index))}  ({len(plat)} lignes)".replace(",", " "))
    print(f"  racine N0                    {len(s.racine):8,} car. {k_tokens(len(s.racine))}".replace(",", " "))
    print(f"  sommaires N1 ({len(tailles)})            médiane {k_tokens(median(tailles))}, max {k_tokens(tailles[-1])}, total {k_tokens(sum(tailles))}")
    for x in sorted(s.par_chemin.values(), key=lambda x: -len(x.texte))[:5]:
        print(f"    {x.chemin:42} {len(x.pages):4} pages {k_tokens(len(x.texte))}")
    if s.alertes:
        print("  alertes :", *s.alertes, sep="\n    ")

    for nom, items in jeux().items():
        n = 0
        nb_sommaires: Counter = Counter()
        volumes: List[int] = []
        plat5 = plat20 = oracle5 = oracle20 = route_une = route_toutes = route_aucune = route5 = 0
        for item in items:
            attendues = {p for p in item["pages"] if not p.startswith("/anomalies/") and p in plat}
            if not attendues:
                continue
            n += 1
            utiles = couverture_minimale(attendues, s)
            nb_sommaires[min(len(utiles), 3)] += 1
            volumes.append(sum(len(s.par_chemin[c].texte) for c in utiles))
            r = bm25.rang(item["question"], list(plat), attendues)
            plat5 += r < 5
            plat20 += r < 20
            lignes_utiles = [p.id for c in utiles for p in s.par_chemin[c].pages]
            r = bm25.rang(item["question"], lignes_utiles, attendues)
            oracle5 += r < 5
            oracle20 += r < 20
            routes = routage(item["question"], s, gammes)
            if not routes:
                route_aucune += 1
                continue
            vues = {p.id for c in routes for p in s.par_chemin[c].pages}
            route_une += bool(attendues & vues)
            route_toutes += attendues <= vues
            r = bm25.rang(item["question"], [p.id for c in routes for p in s.par_chemin[c].pages], attendues)
            route5 += r < 5
        volumes.sort()
        nommees = n - route_aucune
        print(f"\n=== {nom} ({n} questions avec pages attendues)")
        print(f"  sommaires à ouvrir pour voir toutes les pages attendues : 1 → {pct(nb_sommaires[1], n)} | "
              f"2 → {pct(nb_sommaires[2], n)} | 3 et plus → {pct(nb_sommaires[3], n)}")
        print(f"  volume de ces sommaires : médiane {k_tokens(volumes[len(volumes) // 2])}, "
              f"p90 {k_tokens(volumes[int(len(volumes) * 0.9)])} (index plat : {k_tokens(len(index))})")
        print(f"  pointage, page attendue dans le top 5 / top 20 :")
        print(f"    index plat                  {pct(plat5, n)} / {pct(plat20, n)}")
        print(f"    sommaires utiles (oracle)   {pct(oracle5, n)} / {pct(oracle20, n)}")
        print(f"  routage lexical par la racine : produit ou gamme nommé dans {pct(nommees, n)} des questions ; parmi elles,")
        print(f"    une page attendue dans les sommaires désignés {pct(route_une, nommees)} | toutes {pct(route_toutes, nommees)} | top 5 {pct(route5, nommees)}")


if __name__ == "__main__":
    main()
