"""La carte mentale du wiki : l'arbre que la vue « Carte » déplie de gauche à droite.

Les nœuds ne sont PAS les liens du graphe : c'est la taxonomie de la frontmatter, celle de
l'accueil et des tableaux de bord du wiki (gamme › système › type › page). Le navigateur ne
fait que la mise en page ; le CHOIX des nœuds se décide ici, où il se teste.

Deux règles de propreté :

- un niveau qui ne départage rien est sauté — une gamme d'un seul système va droit aux
  types, une branche d'un seul type ou de six pages au plus va droit aux pages, un type
  d'une seule page cède la place à la page ;
- aucune branche vide : un regroupement sans page disparaît, index/log et pages à écrire
  n'y figurent jamais.

Les règles d'appartenance reprennent celles de l'interface (``hubMembers``, ``systemsIn``,
``productOf`` dans ``wiki.html``) : une gamme reçoit aussi les pages sans gamme des systèmes
que déclare sa page, mais une page commune n'ouvre dans la gamme que les systèmes déclarés.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from app.services.wiki_service import USAGES, WikiPage, WikiSnapshot

ORDRE_TYPES = ("Fournisseur", "Gamme", "Profilé", "Quincaillerie", "Vitrage", "Équipement",
               "Porte d'entrée", "Procédure", "Certification", "Garantie", "Anomalie",
               "Document source", "Réservé", "À écrire", "Sans type")
TYPE_PLURIEL = {
    "Gamme": "Gammes", "Profilé": "Profilés", "Quincaillerie": "Quincaillerie", "Vitrage": "Vitrages",
    "Équipement": "Équipements", "Porte d'entrée": "Portes d'entrée", "Procédure": "Procédures",
    "Certification": "Certifications", "Garantie": "Garanties", "Référence": "Référence",
    "Fournisseur": "Fournisseurs", "Anomalie": "Anomalies", "Document source": "Documents sources",
}
USAGE_LABEL = {"atelier": "Atelier et fabrication", "pose": "Pose et chantier",
               "chiffrage": "Chiffrage et prescription", "sav": "SAV et entretien"}
MATERIAUX = (("PVC", "PVC"), ("Mixte", "Mixte aluminium et PVC"),
             ("Aluminium", "Aluminium"), ("Autre", "Autres gammes"))
RANG_PRODUIT = {"gamme": 0, "systeme": 1, "fournisseur": 2}
# Au-delà, une liste de pages se range par type.
PAGES_SANS_TYPE = 6

SOURCE = "Document source"
ANOMALIE = "Anomalie"


def _cle(s: str) -> Tuple:
    """Tri « à la française » : sans casse ni accents, les nombres dans l'ordre (70 < 76 < 120)."""
    plat = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().casefold()
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", plat))


def _rang_type(t: str) -> Tuple:
    return (ORDRE_TYPES.index(t) if t in ORDRE_TYPES else 99, _cle(t))


def sys_label(s: str) -> str:
    return f"Système {s}" if s.isdigit() else s


def hub_label(kind: str, key: str) -> str:
    if kind == "systeme":
        return sys_label(key)
    if kind == "usage":
        return USAGE_LABEL.get(key, key)
    return key


def _materiau(page: WikiPage) -> str:
    tags = set(page.tags)
    if "pvc" in tags and "aluminium" in tags:
        return "Mixte"
    if "pvc" in tags:
        return "PVC"
    if "aluminium" in tags:
        return "Aluminium"
    return "Autre"


class _Taxonomie:
    """Les entrées de navigation calculées depuis la frontmatter."""

    def __init__(self, snap: WikiSnapshot):
        self.concept = sorted(snap.concept_pages, key=lambda p: p.id)
        self.gamme_page: Dict[str, WikiPage] = {}
        self.fournisseur_page: Dict[str, WikiPage] = {}
        for p in self.concept:
            if p.type == "Gamme":
                # La page d'une gamme : celle dont le titre EST la gamme, sinon la première.
                for g in p.gamme:
                    cur = self.gamme_page.get(g)
                    if cur is None or (p.title == g and cur.title != g):
                        self.gamme_page[g] = p
            if p.type == "Fournisseur":
                for f in p.fournisseur:
                    self.fournisseur_page.setdefault(f, p)
        self.gammes = sorted(self.gamme_page, key=_cle)
        self.fournisseurs = sorted({f for p in self.concept for f in p.fournisseur}, key=_cle)

    def hub_page(self, kind: str, key: str) -> Optional[WikiPage]:
        if kind == "gamme":
            return self.gamme_page.get(key)
        if kind == "fournisseur":
            return self.fournisseur_page.get(key)
        return None

    def members(self, kind: str, key: str) -> List[WikiPage]:
        me = self.hub_page(kind, key)
        if kind == "gamme":
            declares = set(me.systeme) if me else set()
            out = [p for p in self.concept if key in p.gamme
                   or (not p.gamme and any(s in declares for s in p.systeme))]
        else:
            out = [p for p in self.concept if key in getattr(p, kind)]
        return [p for p in out if p is not me and p.type != ANOMALIE and p.type != SOURCE]

    def systems_in(self, kind: str, key: str, p: WikiPage) -> List[str]:
        if kind != "gamme" or key in p.gamme:
            return list(p.systeme)
        me = self.hub_page(kind, key)
        declares = me.systeme if me else []
        return [s for s in p.systeme if s in declares]

    def product_of(self, p: WikiPage) -> Optional[Tuple[str, str]]:
        if p.gamme and p.gamme[0] in self.gamme_page:
            return ("gamme", p.gamme[0])
        if p.systeme:
            return ("systeme", p.systeme[0])
        if p.fournisseur:
            return ("fournisseur", p.fournisseur[0])
        return None


def _feuille(p: WikiPage) -> Dict[str, Any]:
    return {"label": p.title, "page": p.id, "type": p.type, "description": p.description}


def _entree(tx: _Taxonomie, kind: str, key: str, children: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Une entrée de navigation. Gamme et fournisseur ont leur page : le nœud l'ouvre."""
    n: Dict[str, Any] = {"label": hub_label(kind, key), "hub": [kind, key], "children": children}
    page = tx.hub_page(kind, key)
    if page is not None:
        n.update(page=page.id, type=page.type)
    return n


def _par_titre(pages: List[WikiPage]) -> List[WikiPage]:
    return sorted(pages, key=lambda p: (_cle(p.title), p.id))


def _par_type(pages: List[WikiPage]) -> List[Dict[str, Any]]:
    pages = _par_titre(pages)
    groupes: Dict[str, List[WikiPage]] = {}
    for p in pages:
        groupes.setdefault(p.type, []).append(p)
    if len(groupes) <= 1 or len(pages) <= PAGES_SANS_TYPE:
        return [_feuille(p) for p in pages]
    out = []
    for t in sorted(groupes, key=_rang_type):
        ps = groupes[t]
        out.append(_feuille(ps[0]) if len(ps) == 1
                   else {"label": TYPE_PLURIEL.get(t, t), "children": [_feuille(p) for p in ps]})
    return out


def _produit(tx: _Taxonomie, kind: str, key: str) -> List[Dict[str, Any]]:
    """Une gamme ou un fournisseur : par système, dans l'ordre que déclare sa page."""
    pages = tx.members(kind, key)
    me = tx.hub_page(kind, key)
    ordre = list(me.systeme) if me else []
    # Une page de gamme couvre plusieurs systèmes : elle passe devant eux, pas dans chacun.
    gammes = [p for p in pages if p.type == "Gamme"]
    par_sys: Dict[str, List[WikiPage]] = {}
    for p in pages:
        if p.type == "Gamme":
            continue
        for s in tx.systems_in(kind, key, p) or [""]:
            par_sys.setdefault(s, []).append(p)
    cles = sorted(par_sys, key=lambda s: (1000 if s == "" else ordre.index(s) if s in ordre else 100, _cle(s)))
    if len(cles) <= 1:
        return _par_type(pages)
    return [_feuille(p) for p in _par_titre(gammes)] + [_entree(tx, "systeme", s, _par_type(par_sys[s])) if s
            else {"label": "Autres pages", "children": _par_type(par_sys[s])} for s in cles]


def _metier(tx: _Taxonomie, usage: str) -> List[Dict[str, Any]]:
    """Un métier : par produit (gamme, sinon système, sinon fournisseur)."""
    groupes: Dict[Optional[Tuple[str, str]], List[WikiPage]] = {}
    for p in tx.members("usage", usage):
        groupes.setdefault(tx.product_of(p), []).append(p)
    rang = lambda prod: (RANG_PRODUIT[prod[0]], _cle(hub_label(*prod))) if prod else (9, ())
    return [_entree(tx, prod[0], prod[1], _par_type(groupes[prod])) if prod
            else {"label": "Hors gamme", "children": _par_type(groupes[prod])}
            for prod in sorted(groupes, key=rang)]


def _elaguer(n: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Retire les regroupements vides : un nœud reste s'il ouvre une page ou mène à une page.
    Un métier ou un système sans page n'ouvrirait qu'une entrée vide : il part."""
    if "children" in n:
        kids = [k for k in map(_elaguer, n["children"]) if k]
        if kids:
            n["children"] = kids
        else:
            n.pop("children")
    return n if n.get("page") or n.get("children") else None


def _tous(n: Dict[str, Any]):
    yield n
    for c in n.get("children", []):
        yield from _tous(c)


def _numeroter(n: Dict[str, Any], key: str = "r") -> None:
    """La clé d'un nœud est son chemin d'indices : stable tant que le wiki ne change pas."""
    n["key"] = key
    for i, c in enumerate(n.get("children", [])):
        _numeroter(c, f"{key}.{i}")


def carte(snap: WikiSnapshot) -> Dict[str, Any]:
    tx = _Taxonomie(snap)
    par_mat: Dict[str, List[str]] = {k: [] for k, _ in MATERIAUX}
    for g in tx.gammes:
        par_mat[_materiau(tx.gamme_page[g])].append(g)
    mats = [{"label": lab, "children": [_entree(tx, "gamme", g, _produit(tx, "gamme", g)) for g in par_mat[k]]}
            for k, lab in MATERIAUX if par_mat[k]]
    fournisseurs = [_entree(tx, "fournisseur", f, _produit(tx, "fournisseur", f)) for f in tx.fournisseurs]
    # Hors gamme : tout ce que ni une gamme ni un fournisseur ne range — aucune page ne manque
    # à la carte, pas même celle d'un système qu'aucune gamme ne déclare.
    ranges = {n.get("page") for b in mats + fournisseurs for n in _tous(b)}
    hors = [p for p in tx.concept if p.type not in (SOURCE, ANOMALIE) and p.id not in ranges]
    du_type = lambda t: [_feuille(p) for p in _par_titre([p for p in tx.concept if p.type == t])]
    branches = [
        {"label": "Gammes", "hue": "Gamme", "children": mats[0]["children"] if len(mats) == 1 else mats},
        # Vue transversale : une page s'y retrouve, mais n'y est pas « rangée ».
        {"label": "Métiers", "hue": "Procédure", "cross": True,
         "children": [_entree(tx, "usage", u, _metier(tx, u)) for u in USAGES]},
        {"label": "Fournisseurs", "hue": "Fournisseur", "children": fournisseurs},
        {"label": "Hors gamme", "hue": "Équipement", "children": _par_type(hors)},
        {"label": "Registres d’anomalies", "hue": ANOMALIE, "children": du_type(ANOMALIE)},
        {"label": "Documents sources", "hue": SOURCE, "children": du_type(SOURCE)},
    ]
    racine = {"label": "Wiki PROFERM", "children": branches}
    _elaguer(racine)
    _numeroter(racine)
    return racine
