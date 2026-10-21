"""Les tableaux HTML du wiki, lus comme les tableaux markdown.

Le protocole (``wiki_llm/CLAUDE.md``, « Schémas et cotes ») écrit en HTML un tableau à cellules fusionnées
ou à en-tête sur deux niveaux : les cotes de débit du système 70, les courbes d'utilisation des coulissants,
les inerties des meneaux. ``faisabilite.tables`` ne lit que le markdown ; ``tables_toutes`` rend les deux
espèces, dans l'ordre du document, au même format (``Table``) :

- l'en-tête à plusieurs niveaux est aplati colonne par colonne, les niveaux séparés par `` / `` :
  ``Battement 0140 (p. 109) / DEO (mm)`` ; une cellule d'en-tête fusionnée sur plusieurs colonnes
  s'applique à chacune ;
- une cellule fusionnée en hauteur (``rowspan``) est répétée sur chaque ligne qu'elle couvre : une ligne
  porte toujours son contexte (la règle « une ligne, un contexte » du protocole) ;
- les images sont retirées du texte des cellules (``brut`` les garde) ; ``<br>`` devient un espace.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

from app.services.faisabilite import Table, tables as tables_markdown

TABLE_RE = re.compile(r"<table\b.*?</table>", re.S | re.I)
TITRE_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


class _Grille(HTMLParser):
    """Collecte les lignes d'un seul ``<table>`` : (cellules, ligne d'en-tête ?)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lignes: List[List[Dict[str, Any]]] = []
        self._ligne: Optional[List[Dict[str, Any]]] = None
        self._cellule: Optional[Dict[str, Any]] = None
        self.en_thead = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        a = dict(attrs)
        if tag == "thead":
            self.en_thead = True
        elif tag == "tr":
            self._ligne = []
        elif tag in ("td", "th") and self._ligne is not None:
            self._cellule = {"th": tag == "th", "texte": "", "brut": "", "rowspan": int(a.get("rowspan") or 1),
                             "colspan": int(a.get("colspan") or 1), "thead": self.en_thead}
        elif tag == "br" and self._cellule is not None:
            self._cellule["texte"] += " "
            self._cellule["brut"] += " "
        elif tag == "img" and self._cellule is not None:
            self._cellule["brut"] += f"![{a.get('alt') or ''}]({a.get('src') or ''})"

    def handle_endtag(self, tag: str) -> None:
        if tag == "thead":
            self.en_thead = False
        elif tag in ("td", "th") and self._cellule is not None and self._ligne is not None:
            self._cellule["texte"] = re.sub(r"\s+", " ", self._cellule["texte"]).strip()
            self._cellule["brut"] = re.sub(r"\s+", " ", self._cellule["brut"]).strip()
            self._ligne.append(self._cellule)
            self._cellule = None
        elif tag == "tr" and self._ligne is not None:
            self.lignes.append(self._ligne)
            self._ligne = None

    def handle_data(self, data: str) -> None:
        if self._cellule is not None:
            self._cellule["texte"] += data
            self._cellule["brut"] += data


def _etendre(lignes: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    """La grille pleine : colspan et rowspan déroulés, une cellule par colonne et par ligne."""
    grille: List[List[Dict[str, Any]]] = []
    report: Dict[int, Tuple[int, Dict[str, Any]]] = {}  # colonne → (lignes restantes, cellule)
    for ligne in lignes:
        sortie: List[Dict[str, Any]] = []
        col, cellules = 0, iter(ligne)
        while True:
            while col in report:
                reste, cel = report[col]
                sortie.append(cel)
                report[col] = (reste - 1, cel) if reste > 1 else None  # type: ignore[assignment]
                if report[col] is None:
                    del report[col]
                col += 1
            cel = next(cellules, None)
            if cel is None:
                break
            for k in range(cel["colspan"]):
                sortie.append(cel)
                if cel["rowspan"] > 1:
                    report[col + k] = (cel["rowspan"] - 1, cel)
            col += cel["colspan"]
        # Des colonnes reportées au-delà de la dernière cellule de la ligne.
        while col in report:
            reste, cel = report[col]
            sortie.append(cel)
            if reste > 1:
                report[col] = (reste - 1, cel)
            else:
                del report[col]
            col += 1
        grille.append(sortie)
    return grille


def table_html(bloc: str, chemin: List[str], localisation: str = "") -> Optional[Table]:
    p = _Grille()
    p.feed(bloc)
    grille = _etendre(p.lignes)
    if not grille:
        return None
    # Les lignes d'en-tête : celles du <thead>, sinon les premières lignes faites de <th> seulement.
    n_tete = sum(1 for l in grille if l and all(c["thead"] for c in l)) or \
        next((i for i, l in enumerate(grille) if not all(c["th"] for c in l)), 0)
    n_tete = max(n_tete, 1)
    largeur = max(len(l) for l in grille)
    entete: List[str] = []
    for j in range(largeur):
        niveaux: List[str] = []
        for l in grille[:n_tete]:
            if j < len(l) and l[j]["texte"] and l[j]["texte"] not in niveaux:
                niveaux.append(l[j]["texte"])
        entete.append(" / ".join(niveaux))
    corps = [l + [{"texte": "", "brut": ""}] * (largeur - len(l)) for l in grille[n_tete:]]
    return Table(chemin, entete, [[c["texte"] for c in l] for l in corps], [[c["brut"] for c in l] for l in corps], localisation)


def tables_toutes(body: str) -> List[Table]:
    """Les tableaux markdown et HTML du corps, dans l'ordre du document."""
    trouves: List[Tuple[int, Table]] = []
    # Chaque tableau HTML avec les titres qui le précèdent ; la position garde l'ordre du document.
    titres = [(m.start(), len(m.group(1)), m.group(2)) for m in re.finditer(r"(?m)^(#{1,6})\s+(.*?)\s*$", body)]
    for m in TABLE_RE.finditer(body):
        chemin: List[Tuple[int, str]] = []
        for p, niveau, titre in titres:
            if p >= m.start():
                break
            chemin = [(n, t) for n, t in chemin if n < niveau] + [(niveau, titre)]
        suite = body[m.end():m.end() + 400]
        loc = next((l.strip()[1:-1] for l in suite.split("\n")[:5] if l.strip().startswith("(schéma:")), "")
        t = table_html(m.group(0), [t for _, t in chemin], loc)
        if t is not None:
            trouves.append((m.start(), t))
    # Les tableaux markdown, hors des blocs HTML.
    sans_html = TABLE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), body)
    pos_md = _positions_markdown(sans_html)
    for (p, t) in zip(pos_md, tables_markdown(sans_html)):
        trouves.append((p, t))
    return [t for _, t in sorted(trouves, key=lambda pt: pt[0])]


def _positions_markdown(body: str) -> List[int]:
    """La position de début de chaque tableau markdown, dans l'ordre où ``tables`` les rend."""
    lignes = body.split("\n")
    out: List[int] = []
    pos = 0
    for i, l in enumerate(lignes):
        if l.strip().startswith("|") and i + 1 < len(lignes) and re.match(r"^\s*\|\s*:?-{2,}", lignes[i + 1]) \
                and not (i > 0 and lignes[i - 1].strip().startswith("|")):
            out.append(pos)
        pos += len(l) + 1
    return out
