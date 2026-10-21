"""Le calculateur du système 70 profine (PERFORM, HYBRIDE et TEXTURAL en 70), 21/10/2026.

Deux sorties, comme pour la PERFORM76 : la **faisabilité** (``verifier``) et la **fiche de débit**
(``fiche_debit``). Toutes les valeurs sont lues dans les tableaux du wiki à chaque rechargement de
l'instantané ; seuls des chemins de pages, des titres de section et des en-têtes de colonnes sont écrits ici.

**Le rattachement n'est écrit nulle part (VER-28).** Aucun document PROFERM ne dit que PERFORM70, HYBRIDE70 ou
TEXTURAL70 sont le système profine e.VOLUTION (planches 6xxx) dont le wiki a le manuel de mise en œuvre : la gamme
n'est qu'un libellé. Chaque verdict porte donc, en tête, un contrôle « info » qui affiche VER-28 : ce que l'on
calcule ici vaut pour le système profine, pas pour une fabrication PROFERM tant que la question reste ouverte.

**Deux générations de documents, jamais fusionnées** : le manuel *Plateforme* de septembre 2023 (par défaut) et le
classeur *e.VOLUTION* d'août 2008 (pour le SAV). Les pages de l'un et de l'autre sont lues une par une, par leur
chemin — le wiki range les deux dans une même « famille » de pages, qu'il ne faut pas lire d'un seul tenant. Quand
l'utilisateur choisit 2008, l'écart avec 2023 est affiché sur les valeurs qu'il touche (vitrage à déduire 55,5
contre 56, battement intérieur à DEO − 6 contre DEO − 12, vitrage de 12 mm contre 14, plages de parcloses).

Ce que le wiki permet, et ce que ce module en fait :

- **le débit** : de la dimension hors tout à la dimension extérieure d'ouvrant (DEO), à la feuillure (DFO) et au
  vitrage, pour un fixe, un ouvrant, deux ouvrants sur meneau et deux ouvrants à battement (dont le débit des
  battements, lu comme formule) ; le seuil 9F67 / 9F68 en montage coupe 90° (variante 1) ;
- **le DTD 6/16-2335** borne la baie selon le type de fenêtre ; le tableau de la porte d'entrée borne l'ouvrant ;
- **l'abaque d'ouvrant** : le repère de la case (largeur × hauteur d'ouvrant, bandes de 100 mm, jamais d'interpolation),
  les limites tracées (réalisation, oblique basse, précision de lecture ±10 mm), la règle des 25 % et le poids de
  vitrage pour lequel les abaques sont établis. Il n'y a ni couleur ni courbe de verre, contrairement au 76 ;
- **l'inertie d'un meneau** : le moment d'inertie requis lu dans la table de la classe de vent (portée × largeur de
  charge, additionnés pour les deux côtés) contre l'inertie du renfort du meneau ;
- **le vitrage** : la parclose à l'épaisseur du vitrage, dans la plage que le tableau écrit.

Ce qui reste « sur étude » : la **ferrure** et le pivot (le wiki n'a aucune borne pour le 70 : FERCO « sur
justifications »), l'ouvrant sur seuil (aucune cote de débit), la couleur sombre (L* sous le seuil du DTD : renfort systématique),
les ouvrants hors abaque, tout ce qui tombe entre deux graduations ou à moins de la précision de lecture.

Hypothèses, affichées sur les contrôles qu'elles touchent : les axes d'un abaque sont les dimensions extérieures
d'ouvrant (la planche cote « Dimensions ouvrant » ; recoupé par le V258 : DHT 1 400 × 2 150 pour 650 × 2 070) ; chaque
vantail d'une fenêtre à meneau se lit sur l'abaque d'un vantail ; le débit d'un battement part de la hauteur
d'ouvrant (le battement est vertical, « sur toute la hauteur de l'ouvrant ») ; le poids est celui du verre seul.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services import wiki_famille
from app.services.faisabilite import (DonneeIntrouvable, ETUDE, HORS, INFO, OK, Table, _borne, _controle, _fmt,
                                      _nombre, _source, lire_vitrage)
from app.services.faisabilite_systemes import coercer, synthese
from app.services.wiki_service import WikiSnapshot
from app.services.wiki_tables import tables_toutes

ID = "systeme70"
LIBELLE = "Système 70 (PERFORM, HYBRIDE, TEXTURAL en 70)"
GAMMES = ["PERFORM", "HYBRIDE", "TEXTURAL"]
PORTEE = ("Débit complet de la Plateforme 2023 (dormant, meneau, ouvrant, battement, seuil), limites du DTD et de la "
          "porte, repère d'abaque d'ouvrant, inertie des meneaux, parclose à l'épaisseur du vitrage ; e.VOLUTION 2008 "
          "en choix, avec l'écart affiché. Ferrure, pivot et ouvrant sur seuil : « sur étude ». Le rattachement des "
          "gammes PERFORM, HYBRIDE et TEXTURAL à ce système n'est écrit nulle part (VER-28).")

P = "/profiles/systeme-70-"
PAGE_DTD = "/certifications/dtd-6-16-2335-meneaux-renforts-capotage-dimensions.md"
PAGE_SEUIL = "/procedures/mise-en-oeuvre-seuil-systeme-70.md"
PAGE_PORTE = "/procedures/porte-d-entree-systeme-70.md"
PAGE_MENEAUX = P + "meneaux-et-traverses.md"
PAGE_ACCESSOIRES = P + "accessoires-par-profile.md"
PAGES_VER28 = ("/gammes/perform.md", "/gammes/hybride.md", "/gammes/textural.md", "/quincaillerie/ferco.md")

GENERATIONS = OrderedDict([
    ("2023", "Plateforme, manuel de septembre 2023"),
    ("2008", "e.VOLUTION, classeur d'août 2008 (SAV)"),
])
PAGES_DEBIT = {
    "2023": [P + "cotes-de-debit.md", P + "cotes-de-debit-battements.md"],
    "2008": [P + "cotes-de-debit-dormants-meneaux-evo2008.md", P + "cotes-de-debit-ouvrants-battements-evo2008.md"],
}
PAGE_VITRAGE = {"2023": P + "tableau-de-vitrage.md", "2008": P + "tableau-de-vitrage-tableaux-de-vitrage-evo2008.md"}
PAGE_EPAISSEURS = {"2023": P + "tableau-de-vitrage.md", "2008": P + "tableau-de-vitrage-directives-de-vitrage-evo2008.md"}
PAGE_REGLES = {"2023": P + "abaques-dimensionnels.md", "2008": P + "abaques-dimensionnels-liste-et-remarques-evo2008.md"}
PAGES_ABAQUES = {
    "2023": [P + n + ".md" for n in ("abaques-va2", "abaques-va2-2-vantaux-of-ob", "abaques-va2-2-vantaux-of",
                                     "abaques-va3", "abaques-va3-2-vantaux-of-ob", "abaques-va3-2-vantaux-of")],
    "2008": [P + "abaques-evo2008" + n + ".md" for n in ("", "-va2-ob-1-vantail", "-va2-2-vantaux-of-ob",
                                                         "-va2-2-vantaux-of", "-va3-of-1-vantail",
                                                         "-va3-ob-1-vantail", "-va3-2-vantaux-of")],
}
PAGE_LEGENDES_2023 = P + "abaques-dimensionnels.md"

CONFIGURATIONS = OrderedDict([
    ("fixe", "Châssis fixe"),
    ("1v_of", "1 vantail OF"),
    ("1v_ob", "1 vantail OB"),
    ("2v_meneau", "2 vantaux sur meneau"),
    ("2v_battement", "2 vantaux OF à battement"),
    ("2v_battement_ob", "2 vantaux OF/OB à battement"),
    ("porte_1v", "Porte d'entrée 1 vantail"),
    ("porte_2v", "Porte d'entrée 2 vantaux"),
    ("soufflet", "Soufflet"),
    ("coulissant", "Oscillo-coulissant 1 vantail + fixe"),
])
CFG_OUVRANT = ("1v_of", "1v_ob", "2v_meneau", "2v_battement", "2v_battement_ob")
CFG_BATTEMENT = ("2v_battement", "2v_battement_ob")
CFG_BAS = CFG_OUVRANT + ("porte_1v", "porte_2v")
# Quelle planche d'abaque lit chaque configuration, et quelle(s) ligne(s) du DTD la borne.
TYPE_ABAQUE = {"1v_of": "of1", "1v_ob": "ob1", "2v_meneau": "of1", "2v_battement": "of2", "2v_battement_ob": "ofob2"}
LIGNES_DTD = {
    "1v_of": (["Française 1 vantail"], "une"),
    "1v_ob": (["Oscillo battante 1 vantail"], "une"),
    # Deux vantaux séparés par un meneau : le tableau n'a pas de ligne qui les nomme (ni « fixe »). Les deux lignes
    # à deux vantaux sont donc toutes deux lues, et une baie qui passe l'une sans passer l'autre reste « sur étude ».
    "2v_meneau": (["Oscillo battante 2 vantaux", "Française 2 vantaux + fixe"], "toutes"),
    "2v_battement": (["Oscillo battante 2 vantaux"], "une"),
    "2v_battement_ob": (["Oscillo battante 2 vantaux"], "une"),
    "soufflet": (["Soufflet"], "une"),
    "coulissant": (["Oscillo-coulissante 1 vantail + fixe"], "une"),
}
# Des libellés du manuel qui ne concordent pas entre le tableau et le dessin d'une même planche (valeurs identiques), et
# l'exemple du classeur de 2008 : inscrits au registre, affichés sur le contrôle de débit qui lit ces lignes.
ANOMALIES_BATTEMENT = {"2023": {"6128": ["INC-126"], "1578": ["INC-127"], "6132": ["INC-128", "INC-129"]},
                       "2008": {"0140": ["INC-218"]}}
FINITIONS = OrderedDict([
    ("blanc", "Blanc (PVC non revêtu)"),
    ("filme", "Filmé, plaxé ou laqué"),
])
ASSEMBLAGES = OrderedDict([
    ("set", "Set d'assemblage mécanique"),
    ("equerre", "Équerre d'assemblage"),
])
# Des compositions courantes verre-lame-verre : une saisie, pas une donnée du wiki.
VITRAGES = ([f"{v}-{lame}-{v}" for v in ("4", "6", "8", "10", "44.2") for lame in (6, 8, 10, 12, 14, 16, 18, 20, 24)]
            + ["6-16-4", "8-16-4", "10-16-4", "6-16-8", "44.2-16-4", "4-16-44.2"]
            + ["4-12-4-12-4", "4-14-4-14-4", "4-16-4-16-4", "4-20-4-20-4", "6-12-6-12-6", "10-20-10-20-10"])

# Les tables d'inertie sont imprimées à une décimale : chaque case est connue à une demi-décimale près.
PRECISION_CASE_INERTIE = 0.05
HYPOTHESE_AXES = "hypothèse : les axes de l'abaque sont les dimensions extérieures d'ouvrant (DEO)"
HYPOTHESE_MENEAU = "hypothèse : chaque vantail d'une fenêtre à meneau se lit sur l'abaque d'un vantail"
HYPOTHESE_BATTEMENT = "hypothèse : le battement, vertical, se débite à partir de la hauteur d'ouvrant"
HYPOTHESE_VERRE = "hypothèse : poids du verre seul (le wiki n'a pas le poids des profilés)"
VER28 = ("Aucun document PROFERM ne rattache PERFORM, HYBRIDE ou TEXTURAL au système 70 profine (e.VOLUTION, "
         "planches 6xxx) : les gammes ne sont qu'un libellé, et la ferrure montée sur le 70 n'est écrite nulle part. "
         "Chaque valeur ci-dessous vient du manuel profine ; le verdict vaut pour ce système, pas pour une "
         "fabrication PROFERM tant que VER-28 est ouvert.")


# ---- l'évaluateur de formules de débit --------------------------------------------------------


class FormuleIllisible(ValueError):
    """Une formule que la grammaire ne comprend pas : la ligne n'est pas calculée."""


_JETON = re.compile(r"(?P<nb>\d+(?:[.,]\d+)?)|(?P<id>[A-Za-zÀ-ÿ_][A-Za-zÀ-ÿ_0-9]*)|(?P<op>[-+−–×*÷/()])")


def _jetons(texte: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(texte):
        if texte[pos].isspace():
            pos += 1
            continue
        m = _JETON.match(texte, pos)
        if not m:
            raise FormuleIllisible(f"caractère « {texte[pos]} » inattendu")
        genre, brut = m.lastgroup or "", m.group(0)
        if genre == "id" and brut == "x":      # « 2 x (56) » : le x de multiplication
            genre, brut = "op", "×"
        out.append((genre, brut))
        pos = m.end()
    return out


class _Analyseur:
    """nombres, variables nommées (en majuscules : DEO, DHT, X), + − × ÷ et parenthèses ; jamais ``eval``."""

    def __init__(self, jetons: List[Tuple[str, str]], variables: Dict[str, float]):
        self.j, self.v, self.i = jetons, variables, 0

    def _suivant(self) -> Optional[Tuple[str, str]]:
        return self.j[self.i] if self.i < len(self.j) else None

    def analyser(self) -> float:
        valeur = self._expression()
        if self._suivant() is not None:
            raise FormuleIllisible(f"« {self._suivant()[1]} » n'est pas attendu ici")
        return valeur

    def _expression(self) -> float:
        valeur = self._terme()
        while (j := self._suivant()) and j[0] == "op" and j[1] in "+-−–":
            self.i += 1
            droite = self._terme()
            valeur = valeur + droite if j[1] == "+" else valeur - droite
        return valeur

    def _terme(self) -> float:
        valeur = self._facteur()
        while (j := self._suivant()) and j[0] == "op" and j[1] in "*×÷/":
            self.i += 1
            droite = self._facteur()
            if j[1] in "÷/":
                if droite == 0:
                    raise FormuleIllisible("division par zéro")
                valeur = valeur / droite
            else:
                valeur = valeur * droite
        return valeur

    def _facteur(self) -> float:
        j = self._suivant()
        if j is None:
            raise FormuleIllisible("formule incomplète")
        self.i += 1
        genre, brut = j
        if genre == "nb":
            return float(brut.replace(",", "."))
        if genre == "id":
            if brut not in self.v:
                raise FormuleIllisible(f"« {brut} » n'est pas une grandeur connue")
            return float(self.v[brut])
        if brut in "-−–":
            return -self._facteur()
        if brut == "(":
            valeur = self._expression()
            fermant = self._suivant()
            if fermant is None or fermant[1] != ")":
                raise FormuleIllisible("parenthèse non fermée")
            self.i += 1
            return valeur
        raise FormuleIllisible(f"« {brut} » n'est pas attendu ici")


def evaluer(formule: str, variables: Dict[str, float]) -> float:
    """« X − 22 », « DEO − 70 mm », « « Débit battement (0141) = DEO – 72 mm » » : la valeur, ou
    ``FormuleIllisible``. Seul ce qui suit le dernier « = » se calcule ; l'unité « mm » est ignorée."""
    texte = formule.replace("«", " ").replace("»", " ")
    if "=" in texte:
        texte = texte.rsplit("=", 1)[1]
    texte = re.sub(r"\bmm\b", " ", texte).strip().rstrip(".")
    if not texte:
        raise FormuleIllisible("formule vide")
    return _Analyseur(_jetons(texte), variables).analyser()


# ---- lecture des tableaux --------------------------------------------------------------------


def _l(x: Optional[float]) -> str:
    return "—" if x is None else _fmt(x, 1 if round(x, 4) % 1 else 0)


def _num(c: str) -> Optional[float]:
    """Une cote : « 37,5 » → 37,5 ; « — », « - », vide → None."""
    return _nombre(c.replace("−", "-")) if c.strip() not in ("", "—", "-", "–") else None


def _sans_espaces(c: str) -> str:
    return re.sub(r"[\s  ]", "", c)


def _col(entete: List[str], debut: str, page: str) -> int:
    for k, h in enumerate(entete):
        if h.startswith(debut):
            return k
    raise DonneeIntrouvable(f"{page} : colonne « {debut}… » introuvable")


def _col_option(entete: List[str], debut: str) -> Optional[int]:
    return next((k for k, h in enumerate(entete) if h.startswith(debut)), None)


def _section(t: Table) -> str:
    """Le titre de section qui existe dans la page : « Tableau » seul ne dit rien, on remonte d'un cran."""
    if len(t.chemin) > 1 and t.chemin[-1] == "Tableau":
        return t.chemin[-2]
    return t.chemin[-1] if t.chemin else ""


def _tables_des(snap: WikiSnapshot, pages: List[str]) -> List[Tuple[str, Table]]:
    """Les tableaux (markdown et HTML) de chaque page, lue seule : jamais la famille d'un seul tenant."""
    sortie: List[Tuple[str, Table]] = []
    for page in pages:
        p = snap.pages.get(page)
        if p is None or p.missing:
            raise DonneeIntrouvable(f"page absente du wiki : {page}")
        sortie += [(page, t) for t in tables_toutes(p.body)]
    return sortie


def _trouver(tt: List[Tuple[str, Table]], predicat, quoi: str) -> Tuple[str, Table]:
    for page, t in tt:
        if predicat(t):
            return page, t
    raise DonneeIntrouvable(f"tableau « {quoi} » introuvable dans le wiki")


def _image(brut: List[str]) -> Optional[str]:
    for c in reversed(brut):
        m = re.search(r"\((/assets/[^)]+)\)", c)
        if m:
            return m.group(1)
    return None


def _refs4(texte: str) -> List[str]:
    return re.findall(r"\b\d{4}\b", texte)


def _corps_page(snap: WikiSnapshot, page: str) -> str:
    p = snap.pages.get(page)
    if p is None or p.missing:
        raise DonneeIntrouvable(f"page absente du wiki : {page}")
    return p.body


# ---- les données, telles que le wiki les écrit ------------------------------------------------


@dataclass
class Debit:
    pages: List[str]
    dormants: Dict[str, Dict[str, Any]]
    meneaux: Dict[str, Dict[str, Any]]
    ouvrants: Dict[str, Dict[str, Any]]
    battement_dormant: Dict[Tuple[str, str], Dict[str, Optional[str]]]   # (dormant, groupe) -> formules DEO, DFO
    groupes: Dict[str, str]                                              # battement -> groupe de colonnes
    battements: Dict[str, Dict[str, Any]]                                # battement extérieur -> formules
    planches: Optional[List[Dict[str, Any]]]                             # les planches de battement (2023)


@dataclass
class Vitrage:
    pages: List[str]
    feuillures: Dict[int, Dict[str, Any]]    # largeur de feuillure -> {rows, section, localisation}
    plage: Tuple[float, float]               # épaisseurs admises sans l'élargisseur
    plage_elargisseur: Tuple[float, float]


@dataclass
class Planche:
    page: str
    titre: str
    variante: str                            # « principal » ou « autre »
    classe: str                              # « VA2 » ou « VA3 »
    genre: str                               # of1, ob1, of2, ofob2
    ouvrants: List[List[str]]                # les groupes de la planche, chacun une liste de références (2023)
    renforts: List[str]
    pdf: Optional[int]
    colonnes: List[Tuple[int, int]]
    bandes: List[Tuple[int, int, Optional[int], List[str]]]  # (haut mini, haut maxi, paumelles, cellules)
    limites: Dict[str, Dict[int, Optional[List[float]]]]
    localisation: str
    section: str
    anomalies: List[str] = field(default_factory=list)
    avec_g: bool = False

    @property
    def refs(self) -> List[str]:
        return [r for g in self.ouvrants for r in g]


@dataclass
class Abaques:
    planches: List[Planche]
    legendes: Dict[str, Dict[str, str]]      # genre -> repère -> texte
    precision_mm: float
    rapport_max: float                       # la largeur ne dépasse pas ce rapport de la hauteur (1,25)
    verre_kg_m2: float
    verre_mm: float
    densite: float
    entraxe_mm: Optional[float]
    pages: List[str]
    page_regles: str


@dataclass
class Generation:
    id: str
    debit: Debit
    vitrage: Vitrage
    abaques: Abaques
    meneaux_renfort: Dict[str, Dict[str, Any]]   # meneau -> {renfort, iw, section, page, localisation}


@dataclass
class Donnees:
    generations: Dict[str, Generation]
    dtd: List[Tuple[str, float, float]]
    dtd_section: str
    dtd_localisation: str
    fiches: List[Tuple[float, int]]              # (hauteur d'ouvrant maxi en m, nombre de fiches), du plus bas au plus haut
    verrouillage_mm: Optional[float]
    vitrage_dtd_mm: Optional[float]              # « isolant double ou triple jusqu'à 42 mm » (2.2.3.7)
    traverse_renfort_mm: float                   # « les traverses d'ouvrant au-delà de 750 mm de large » (2.2.3.4)
    lstar_mini: float                            # « L* inférieure à 82 » : renfort systématique (2.2.3.4)
    porte: Dict[str, Dict[str, Tuple[float, float]]]   # porte -> colonne -> (L, H)
    porte_localisation: str
    seuil: Dict[str, Dict[str, Any]]             # dormant -> formules de débit du seuil
    seuils: List[str]
    accessoires: Dict[str, List[Dict[str, Any]]]
    statuts: Dict[str, str]
    classes: "OrderedDict[str, Dict[str, Any]]"  # classe de vent -> {page, libelle}


_CACHE: Dict[str, Any] = {"cle": None, "donnees": None}
_CACHE_INERTIES: Dict[Any, Dict[str, Any]] = {}


def donnees(snap: WikiSnapshot) -> Donnees:
    cle = (id(snap), snap.loaded_at)
    if _CACHE["cle"] != cle:
        _CACHE["donnees"] = _charge(snap)
        _CACHE["cle"] = cle
        _CACHE_INERTIES.clear()
    return _CACHE["donnees"]


# -- débit

def _charge_debit(snap: WikiSnapshot, gen: str) -> Debit:
    pages = PAGES_DEBIT[gen]
    tt = _tables_des(snap, pages)

    def fiche(t: Table, page: str, **plus: Any) -> Dict[str, Any]:
        return {**plus, "section": _section(t), "localisation": t.localisation, "page": page}

    page, t = _trouver(tt, lambda t: t.entete[0] == "Dormant" and "① DEO (mm)" in t.entete
                       and not any("Battement" in h for h in t.entete), "dormants, ① DEO (mm)")
    c = {k: _col(t.entete, k, page) for k in ("① DEO (mm)", "③ Vitrage", "Renfort de dormant (mm)", "④ Meneau / traverse (mm)")}
    c_set = _col(t.entete, "Renfort de meneau / traverse (mm) / set", page)
    c_eq = _col(t.entete, "Renfort de meneau / traverse (mm) / équerre", page)
    dormants = {r[0]: fiche(t, page, deo=_num(r[c["① DEO (mm)"]]), vitrage=_num(r[c["③ Vitrage"]]),
                            renfort=_num(r[c["Renfort de dormant (mm)"]]), meneau=_num(r[c["④ Meneau / traverse (mm)"]]),
                            renfort_meneau={"set": _num(r[c_set]), "equerre": _num(r[c_eq])}) for r in t.lignes}

    page, t = _trouver(tt, lambda t: t.entete[0] == "Meneau / traverse" and "① DEO (mm)" in t.entete, "meneaux, ① DEO (mm)")
    c = {k: _col(t.entete, k, page) for k in ("① DEO (mm)", "③ Vitrage")}
    meneaux = {r[0]: fiche(t, page, deo=_num(r[c["① DEO (mm)"]]), vitrage=_num(r[c["③ Vitrage"]])) for r in t.lignes}

    page, t = _trouver(tt, lambda t: t.entete[0] == "Ouvrant" and "② DFO (mm)" in t.entete
                       and any(h.startswith("③ Vitrage") for h in t.entete), "ouvrants, ② DFO")
    c = {k: _col(t.entete, k, page) for k in ("② DFO (mm)", "③ Vitrage", "Renfort d'ouvrant (mm)")}
    ouvrants = {r[0]: fiche(t, page, dfo=_num(r[c["② DFO (mm)"]]), vitrage=_num(r[c["③ Vitrage"]]),
                            renfort=_num(r[c["Renfort d'ouvrant (mm)"]])) for r in t.lignes}

    # un dormant qui reçoit un battement : une colonne DEO et une DFO par groupe de battements
    page_b, t = _trouver(tt, lambda t: t.entete[0] == "Dormant" and any("Battement" in h and h.endswith("/ DEO (mm)")
                                                                      for h in t.entete), "dormant avec battement")
    groupes: Dict[str, str] = {}
    cols: Dict[str, Dict[str, int]] = {}
    for k, h in enumerate(t.entete[1:], 1):
        tete, _, mesure = h.rpartition(" / ")
        refs = _refs4(re.sub(r"\(.*?\)", "", tete))
        cle = ",".join(refs)
        for ref in refs:
            groupes[ref] = cle
        cols.setdefault(cle, {})["deo" if mesure.startswith("DEO") else "dfo"] = k
    battement_dormant = {
        (r[0], cle): fiche(t, page_b, deo=r[k["deo"]] if "X" in r[k["deo"]] else None, dfo=r[k["dfo"]] if "X" in r[k["dfo"]] else None)
        for r in t.lignes for cle, k in cols.items()}

    # le débit des battements : une formule lue comme texte. 2023 : une ligne par battement extérieur, avec son intérieur ;
    # 2008 : une ligne par battement (« 6130 (extérieur) », « 6133 (intérieur) »), page par page.
    battements: Dict[str, Dict[str, Any]] = {}
    planches: Optional[List[Dict[str, Any]]] = None
    if gen == "2023":
        page, t = _trouver(tt, lambda t: t.entete[:1] == ["Battement extérieur"], "débit des battements (extérieur, intérieur)")
        ce, ci = _col(t.entete, "Débit du battement ext", page), _col(t.entete, "Débit du battement int", page)
        for r in t.lignes:
            for ext in _refs4(r[0]):
                battements[ext] = fiche(t, page, exterieur=r[ce], interieur=r[ci] if "DEO" in r[ci] or "DFO" in r[ci] else None,
                                        interieurs=_refs4(r[1]))
        # les planches de battement : ni tous les dormants ni tous les ouvrants n'y sont dessinés (VER-85)
        _, t = _trouver(tt, lambda t: t.entete[:2] == ["Page PDF", "Battement extérieur"], "planches de battement")
        planches = [{"pdf": int(_num(r[0]) or 0), "exterieurs": _refs4(r[1]), "interieurs": _refs4(r[2]),
                     "ouvrants": _refs4(r[3]) or None, "absents": [] if "aucun" in r[4] else _refs4(r[4])} for r in t.lignes]
    else:
        page, t = _trouver(tt, lambda t: t.entete[:2] == ["Battement", "Débit imprimé"], "débit des battements")
        for k, r in enumerate(t.lignes):
            m = re.match(r"(\d{4})(?:\s*\(extérieur\))?$", r[0].strip())
            if not m:
                continue
            suite = t.lignes[k + 1] if k + 1 < len(t.lignes) else None
            interieur = suite if suite and "(intérieur)" in suite[0] and suite[2] == r[2] else None
            battements.setdefault(m.group(1), fiche(t, page, exterieur=r[1], interieur=interieur[1] if interieur else None,
                                                    interieurs=_refs4(interieur[0]) if interieur else []))

    return Debit(pages, dormants, meneaux, ouvrants, battement_dormant, groupes, battements, planches)


# -- vitrage

def _plage(texte: str) -> Optional[Tuple[float, float]]:
    n = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:,\d+)?", texte)]
    return (n[0], n[1]) if len(n) >= 2 else None


def _lignes_parclose(t: Table) -> List[Dict[str, Any]]:
    c_ref = _col(t.entete, "Parclose", "tableau de vitrage")
    c_nom = _col_option(t.entete, "Épaisseur de vitrage nominale") or _col(t.entete, "Épaisseur du vitrage", "tableau de vitrage")
    c_min, c_max = _col_option(t.entete, "Épaisseur mini admise"), _col_option(t.entete, "Épaisseur maxi admise")
    c_plage = _col_option(t.entete, "Plage admise")
    c_ecr = _col_option(t.entete, "Écriture")
    rows = []
    for r, b in zip(t.lignes, t.brut):
        if c_min is not None and c_max is not None:
            mini, maxi = _num(r[c_min]), _num(r[c_max])
        else:
            pl = _plage(r[c_plage]) if c_plage is not None else None
            mini, maxi = pl if pl else (None, None)
        rows.append({"refs": re.findall(r"[0-9][0-9A-Z]{3,4}", r[c_ref]) or [r[c_ref]], "nominal": _num(r[c_nom]),
                     "mini": mini, "maxi": maxi, "ecriture": r[c_ecr] if c_ecr is not None else "",
                     "image": _image(b)})
    return rows


def _charge_vitrage(snap: WikiSnapshot, gen: str) -> Vitrage:
    tt = _tables_des(snap, [PAGE_VITRAGE[gen]])
    feuillures: Dict[int, Dict[str, Any]] = {}
    if gen == "2023":
        for page, t in tt:
            m = re.search(r"feuillure de (\d+) mm", t.chemin[-1] if t.chemin else "")
            if t.entete[:1] == ["Parclose"] and m:
                feuillures[int(m.group(1))] = {"rows": _lignes_parclose(t), "section": t.chemin[-1],
                                               "localisation": t.localisation, "page": page}
    else:
        joint = [(p, t) for p, t in tt if t.entete[:1] == ["Parclose"] and t.chemin
                 and t.chemin[-1].startswith("Joint PCE 9C32") and "feuillure 54 mm" in t.chemin[-1]]
        if len(joint) < 2:
            raise DonneeIntrouvable(f"{PAGE_VITRAGE[gen]} : les deux tableaux « joint PCE 9C32, feuillure 54 mm » manquent")
        for (page, t), largeur in zip(joint[:2], (54, 70)):
            feuillures[largeur] = {"rows": _lignes_parclose(t), "section": t.chemin[-1],
                                   "localisation": t.localisation, "page": page}
    if len(feuillures) < 2:
        raise DonneeIntrouvable(f"{PAGE_VITRAGE[gen]} : tableaux de parcloses des deux feuillures introuvables")
    _, t = _trouver(_tables_des(snap, [PAGE_EPAISSEURS[gen]]),
                    lambda t: t.entete[0].startswith("Élément") and any(r[0].startswith("Ouvrant à déport") for r in t.lignes),
                    "épaisseurs de vitrage admises")
    ligne = next(r for r in t.lignes if r[0].startswith("Ouvrant à déport"))
    seul, large = _plage(ligne[1]), _plage(ligne[2])
    if not seul or not large:
        raise DonneeIntrouvable(f"{PAGE_EPAISSEURS[gen]} : épaisseurs de vitrage admises illisibles")
    return Vitrage([PAGE_VITRAGE[gen], PAGE_EPAISSEURS[gen]], feuillures, seul, large)


# -- abaques

_TITRE_PLANCHE = re.compile(
    r"^(?P<prefixe>Planche titrée (?P<titre_classe>VA\d) \(PDF p\. (?P<pdf>\d+)\) — )?(?:(?P<classe>VA\d) — )?"
    r"(?P<genre>OF 1 vantail|OB 1 vantail|2 vantaux OF/OB|2 vantaux OF(?: avec repère G)?)"
    r"(?:, ouvrants (?P<ouv>.+)|, renforts? (?P<ren>.+?) \(p\. (?P<page>\d+)\))$")


def _bornes(texte: str) -> Optional[Tuple[int, int]]:
    n = [int(x) for x in re.findall(r"\d+", _sans_espaces(re.sub(r"\(mm\)", "", texte)))]
    return (n[0], n[1]) if len(n) >= 2 else None


def _valeurs_limite(c: str) -> Optional[List[float]]:
    """« 1 880 » → [1880] ; « 2 300 à 2 000 (trait vertical) » → [2300, 2000] ; « - » → None."""
    if c.strip() in ("", "-", "—"):
        return None
    n = [float(_sans_espaces(x)) for x in re.findall(r"\d[\d  ]*", c)]
    return n or None


def _intro(corps: str, titre: str, rang: int) -> str:
    """Le texte sous le ``rang``-ième titre de ce nom (un titre de planche peut se répéter dans une page)."""
    debuts = [m for m in re.finditer(r"^(#{1,6})\s+" + re.escape(titre) + r"\s*$", corps, re.M)]
    if not debuts:
        return ""
    m = debuts[min(rang, len(debuts) - 1)]
    suite = corps[m.end():]
    fin = re.search(r"^#{1," + str(len(m.group(1))) + r"}\s", suite, re.M)
    return suite[:fin.start()] if fin else suite


def _charge_abaques(snap: WikiSnapshot, gen: str) -> Abaques:
    pages = PAGES_ABAQUES[gen]
    planches: List[Planche] = []
    legendes: Dict[str, Dict[str, str]] = {}
    precision = None

    def genre_legende(titre: str) -> Optional[str]:
        t = titre.lower()
        if "2 vantaux of/ob" in t:
            return "ofob2"
        if "2 vantaux of" in t:
            return "of2"
        if "ob 1 vantail" in t:
            return "ob1"
        if "of 1 vantail" in t:
            return "of1"
        return None

    def lire_legende(t: Table) -> None:
        genre = genre_legende(t.chemin[-1] if t.chemin else "")
        if genre:
            legendes.setdefault(genre, {r[0]: r[1] for r in t.lignes})

    for page in pages:
        corps = _corps_page(snap, page)
        m = re.search(r"précision de lecture ±(\d+) mm", corps)
        if m and precision is None:
            precision = float(m.group(1))
        # Un titre peut revenir (une planche titrée « OF/OB » est en réalité une planche OF, INC-138) : chaque tableau de
        # repères ouvre sa propre planche, les tableaux de limites se rangent dans la dernière du même titre.
        groupes: List[Dict[str, Any]] = []
        for t in tables_toutes(corps):
            titre = next((e for e in t.chemin if _TITRE_PLANCHE.match(e)), None)
            if t.entete[:2] == ["Repère", "Dessin du pictogramme"]:
                if gen == "2008":      # 2008 : la légende est sur chaque page de planches
                    lire_legende(t)
                continue
            if titre is None:
                continue
            reste = " ".join(t.chemin[t.chemin.index(titre) + 1:])
            variante = "autre" if "Autre possibilité" in reste else "principal"
            memes = [g for g in groupes if g["titre"] == titre and g["variante"] == variante]
            if t.entete[0].startswith("Hauteur d'ouvrant"):
                rang = len([g for g in groupes if g["titre"] == titre and g["variante"] == "principal"])
                groupes.append({"titre": titre, "variante": variante, "reperes": t, "limites": None,
                                "localisation": t.localisation, "rang": rang - (1 if variante == "autre" else 0)})
            elif memes and t.entete[0].startswith("Largeur d'ouvrant") and any(h.startswith("Limite") for h in t.entete)                     and memes[-1]["limites"] is None:
                memes[-1]["limites"] = t
                # « Limite de réalisation, abaque principal » : le tableau de l'« autre possibilité » borne aussi l'abaque
                # principal de la même planche, qui n'a pas de tableau de limites propre.
                if variante == "autre":
                    precedent = [g for g in groupes if g["titre"] == titre and g["variante"] == "principal"]
                    if precedent and precedent[-1]["limites"] is None:
                        precedent[-1]["limites"] = t
        for g in groupes:
            titre, variante, tr, tl = g["titre"], g["variante"], g["reperes"], g["limites"]
            m = _TITRE_PLANCHE.match(titre)
            classe = m.group("classe") or m.group("titre_classe")
            mp = re.search(r"abaques-(va\d)", page)
            if mp:
                classe = mp.group(1).upper()
            avec_paum = any(h.startswith("Paumelles") for h in tr.entete)
            genre = m.group("genre")
            if genre.startswith("2 vantaux"):
                genre = "of2" if avec_paum else "ofob2"
            else:
                genre = "of1" if avec_paum else "ob1"
            c0 = 2 if avec_paum else 1
            colonnes = [_bornes(h) for h in tr.entete[c0:]]
            if None in colonnes:
                raise DonneeIntrouvable(f"{page} : en-têtes de largeur illisibles dans « {titre} »")
            bandes = []
            for r in tr.lignes:
                b = _bornes(r[0])
                if b is None:
                    raise DonneeIntrouvable(f"{page} : bande de hauteur illisible « {r[0]} »")
                bandes.append((b[0], b[1], int(_num(r[1])) if avec_paum and _num(r[1]) is not None else None, r[c0:]))
            limites: Dict[str, Dict[int, Optional[List[float]]]] = {}
            if tl is not None:
                for k, h in enumerate(tl.entete[1:], 1):
                    nom = ("realisation" if h.startswith("Limite de réalisation") else
                           "oblique" if h.startswith("Limite oblique basse") else None)
                    if nom:
                        limites[nom] = {int(_sans_espaces(r[0])): _valeurs_limite(r[k]) for r in tl.lignes}
            ouvrants: List[List[str]] = []
            renforts: List[str] = []
            pdf = int(m.group("pdf") or m.group("page") or 0) or None
            if m.group("ouv"):
                ouvrants = [re.findall(r"\b[0-9][0-9A-Z]{3}\b", item) for item in m.group("ouv").split("/")]
                intro = _intro(corps, titre, max(g["rang"], 0))
                mr = re.search(r"renforts? dessinés? ([V\d, et]+?)\s*\[\d+ p\. (\d+)\]", intro)
                if mr:
                    renforts = re.findall(r"V\d{3}", mr.group(1))
                    pdf = pdf or int(mr.group(2))
            else:
                renforts = re.findall(r"V\d{3}", m.group("ren"))
            anomalies: List[str] = []
            if gen == "2023":
                if m.group("prefixe"):
                    anomalies.append("INC-135")
                if classe == "VA3" and genre == "ob1":
                    anomalies.append("INC-137")
                if classe == "VA3" and genre == "of2":
                    anomalies += ["INC-136", "INC-138"]
            else:
                if classe == "VA3" and genre == "ob1":
                    anomalies.append("INC-219")
                if classe == "VA3" and genre == "of2":
                    anomalies.append("INC-220")
            planches.append(Planche(page, titre, variante, classe, genre, ouvrants, renforts, pdf, colonnes, bandes,
                                    limites, g["localisation"], titre.split(" — ", 1)[-1] if m.group("prefixe") else titre,
                                    anomalies, "avec repère G" in titre))
    if not planches:
        raise DonneeIntrouvable(f"abaques {gen} : aucune planche lue")

    # les légendes : 2023 sur la page des règles, 2008 sur chaque page de planches (lues ci-dessus)
    regles = _corps_page(snap, PAGE_REGLES[gen])
    if gen == "2023":
        for t in tables_toutes(_corps_page(snap, PAGE_LEGENDES_2023)):
            if t.entete[:2] == ["Repère", "Dessin du pictogramme"]:
                lire_legende(t)
    plat = re.sub(r"\s+", " ", regles.replace("*", ""))
    mv = re.search(r"ne dépassant pas (\d+) ?k[gG]/m², soit (\d+) mm de verre", plat)
    mr = re.search(r"dépasser de (\d+) % la hauteur", plat)
    md = re.search(r"masse de (\d+(?:,\d+)?) kg/m² par mm", plat)
    me = re.search(r"entraxe maxi de verrouillage est de (\d+) mm", plat)
    if not (mv and mr and md) or precision is None:
        raise DonneeIntrouvable(f"{PAGE_REGLES[gen]} : poids de vitrage, règle des 25 % ou précision de lecture introuvables")
    return Abaques(planches, legendes, precision, 1 + int(mr.group(1)) / 100, float(mv.group(1)), float(mv.group(2)),
                   float(md.group(1).replace(",", ".")), float(me.group(1)) if me else None, pages, PAGE_REGLES[gen])


# -- renforts de meneau

def _charge_renforts(snap: WikiSnapshot, gen: str) -> Dict[str, Dict[str, Any]]:
    tt = _tables_des(snap, [PAGE_MENEAUX])
    sortie: Dict[str, Dict[str, Any]] = {}
    if gen == "2023":
        page, t = _trouver(tt, lambda t: t.entete[:2] == ["Profilé", "Renfort"] and any(h.startswith("IW") for h in t.entete),
                           "renforts des meneaux et traverses (IW)")
        ci, cr = _col(t.entete, "IW", page), _col(t.entete, "Renfort", page)
        for r in t.lignes:
            sortie[r[0]] = {"renfort": r[cr], "iw": _num(r[ci]), "section": _section(t), "localisation": t.localisation,
                            "page": page}
    else:
        page, t = _trouver(tt, lambda t: t.entete[:1] == ["Profilé"] and any(h.startswith("Renfort du tableau") for h in t.entete),
                           "meneaux et traverses de 2008 (Iz)")
        ci, cr = _col(t.entete, "Iz", page), _col(t.entete, "Renfort du tableau", page)
        for r in t.lignes:
            sortie[r[0]] = {"renfort": r[cr], "iw": _num(r[ci]), "section": _section(t), "localisation": t.localisation,
                            "page": page}
    return sortie


def _charge(snap: WikiSnapshot) -> Donnees:
    generations = {g: Generation(g, _charge_debit(snap, g), _charge_vitrage(snap, g), _charge_abaques(snap, g),
                                 _charge_renforts(snap, g)) for g in GENERATIONS}

    # -- le DTD : limites de baie, fiches par vantail, verrouillage
    tt = _tables_des(snap, [PAGE_DTD])
    page, t = _trouver(tt, lambda t: t.entete[:3] == ["Type de fenêtre", "H (m)", "L (m)"], "dimensions maximales du DTD")
    dtd = [(r[0], _num(r[1]), _num(r[2])) for r in t.lignes]
    dtd_section, dtd_loc = _section(t), t.localisation
    _, tf = _trouver(tt, lambda t: t.entete[:2] == ["Hauteur d'ouvrant (m)", "Nombre de fiches par vantail"], "fiches par vantail")
    fiches: List[Tuple[float, int]] = []
    for r in tf.lignes:
        # « jusqu'à 0,90 » : jusqu'à cette hauteur ; « au-delà de 2,10 » : sans borne haute
        n = _nombre(re.search(r"\d+(?:,\d+)?", r[0]).group(0)) if "jusqu'à" in r[0] else float("inf")
        fiches.append((n, int(_num(r[1]))))
    corps_dtd = re.sub(r"\s+", " ", _corps_page(snap, PAGE_DTD))
    mv = re.search(r"verrouillages complémentaires haut et bas centrés à partir de (\d+) mm", corps_dtd)
    mdt = re.search(r"Isolant double ou triple jusqu'à (\d+) mm", corps_dtd)
    mtr = re.search(r"les traverses d'ouvrant au-delà de (\d+) mm de large", corps_dtd)
    mls = re.search(r"L\* inférieure à (\d+)", corps_dtd)
    if not (mtr and mls):
        raise DonneeIntrouvable(f"{PAGE_DTD} : les cas de renforcement systématique (2.2.3.4) sont introuvables")

    # -- la porte d'entrée
    _, tp = _trouver(_tables_des(snap, [PAGE_PORTE]), lambda t: t.entete[:1] == ["Porte"] and "Dimensions maximales d'ouvrant" in t.chemin,
                     "dimensions maximales de la porte")
    porte: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for r in tp.lignes:
        porte[r[0]] = {}
        for h, c in zip(tp.entete[1:], r[1:]):
            n = [float(_sans_espaces(x)) for x in re.findall(r"\d[\d  ]*", c)]
            if len(n) == 2:
                porte[r[0]][h] = (n[0], n[1])
    porte_loc = tp.localisation

    # -- le seuil, variante 1 : coupe 90° sur dormant
    page_s, ts = _trouver(_tables_des(snap, [PAGE_SEUIL]),
                          lambda t: t.entete[:1] == ["Dormant"] and any(h.startswith("Débit seuil, largeur") for h in t.entete)
                          and t.chemin and t.chemin[-1].startswith("Variante 1"), "débit des seuils, variante 1")
    cs, cm = _col(ts.entete, "Débit seuil, largeur", page_s), _col(ts.entete, "Débit montant dormant", page_s)
    seuils = [re.search(r"Embout pour (\S+)", h).group(1) for h in ts.entete if h.startswith("Embout pour ")]
    seuil = {r[0]: {"largeur": r[cs], "montant": r[cm],
                    "embouts": {re.search(r"Embout pour (\S+)", h).group(1): r[k]
                                for k, h in enumerate(ts.entete) if h.startswith("Embout pour ")},
                    "section": _section(ts), "localisation": ts.localisation, "page": page_s} for r in ts.lignes}

    # -- les accessoires de chaque profilé, avec leur dessin
    accessoires: Dict[str, List[Dict[str, Any]]] = {}
    for page, t in _tables_des_famille(snap, PAGE_ACCESSOIRES):
        if t.entete[:2] != ["Référence", "Désignation"] or not t.chemin:
            continue
        m = re.match(r"(?:Dormant|Ouvrant|Meneau|Traverse|Battement\S*|Seuil)\s+([0-9][0-9A-Z]{3})", t.chemin[-1])
        profils = [m.group(1)] if m else ([t.chemin[-1]] if t.chemin[-1] == "Seuil" else [])
        for r, b in zip(t.lignes, t.brut):
            for p in profils:
                accessoires.setdefault(p, []).append({"ref": r[0], "designation": r[1], "image": _image(b),
                                                      "section": t.chemin[-1], "page": page})
    # -- les classes de vent des tables d'inertie
    classes: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for chemin in sorted(snap.pages):
        m = re.fullmatch(r"/profiles/systeme-70-inerties-(v[abc][1-5])\.md", chemin)
        if not m:
            continue
        mt = re.search(r"classement V\\?\*([ABC]\d) \((\d+) Pa, flèche (1/\d+)\)", snap.pages[chemin].body[:1500])
        if mt:
            classes[m.group(1)] = {"page": chemin, "libelle": f"V*{mt.group(1)} ({mt.group(2)} Pa, flèche {mt.group(3)})"}
    if not classes:
        raise DonneeIntrouvable("tables d'inertie du système 70 : aucune page lue")

    pages = {PAGE_DTD, PAGE_PORTE, PAGE_SEUIL, PAGE_MENEAUX, PAGE_ACCESSOIRES}
    for g in generations.values():
        pages |= set(g.debit.pages) | set(g.vitrage.pages) | set(g.abaques.pages) | {g.abaques.page_regles}
    pages |= {c["page"] for c in classes.values()}
    d = Donnees(generations, dtd, dtd_section, dtd_loc, sorted(fiches), float(mv.group(1)) if mv else None,
                float(mdt.group(1)) if mdt else None, float(mtr.group(1)), float(mls.group(1)), porte, porte_loc, seuil,
                seuils, accessoires,
                {p: (snap.pages[p].status or "") for p in pages if p in snap.pages}, classes)
    return d


def _tables_des_famille(snap: WikiSnapshot, tete: str) -> List[Tuple[str, Table]]:
    """Les tableaux d'une page et de ses suites (les accessoires par profilé sont coupés en six pages)."""
    return _tables_des(snap, [tete] + wiki_famille.suites(snap, tete))


# ---- les tables d'inertie, lues à la demande --------------------------------------------------


def _inertie(snap: WikiSnapshot, d: Donnees, classe: str) -> Dict[str, Any]:
    cle = (id(snap), classe)
    if cle not in _CACHE_INERTIES:
        page = d.classes[classe]["page"]
        _, t = _trouver(_tables_des(snap, [page]), lambda t: t.entete[:1] == ["L - portée (cm)"], f"inerties {classe}")
        largeurs = [int(_num(h.rsplit("/", 1)[1])) for h in t.entete[1:-1]]
        fleche = _col(t.entete, "Flèche calculée", page)
        grille = {int(_num(r[0])): {a: _num(c) for a, c in zip(largeurs, r[1:fleche])} for r in t.lignes}
        fleches = {int(_num(r[0])): _num(r[fleche]) for r in t.lignes}
        _CACHE_INERTIES[cle] = {"grille": grille, "largeurs": largeurs, "fleches": fleches, "page": page,
                                "section": t.chemin[-1] if t.chemin else "", "localisation": t.localisation,
                                "libelle": d.classes[classe]["libelle"]}
    return _CACHE_INERTIES[cle]


# ---- le formulaire ---------------------------------------------------------------------------


def champs(snap: WikiSnapshot) -> List[Dict[str, Any]]:
    d = donnees(snap)
    g23, g08 = d.generations["2023"], d.generations["2008"]

    def union(un: Dict[str, Any], deux: Dict[str, Any]) -> Dict[str, str]:
        out = {}
        for ref in list(un) + [r for r in deux if r not in un]:
            reste = "" if ref in un and ref in deux else (" (2023 seulement)" if ref in un else " (2008 seulement)")
            out[ref] = ref + reste
        return out

    battements = {}
    for ref in list(g23.debit.groupes) + [r for r in g08.debit.groupes if r not in g23.debit.groupes]:
        battements[ref] = ref
    renforts_2008 = OrderedDict()
    for p in g08.abaques.planches:
        if p.renforts and not p.avec_g and p.variante == "principal":
            renforts_2008.setdefault(p.renforts[0], " / ".join(p.renforts))
    defaut_classe = "va3" if "va3" in d.classes else next(iter(d.classes))
    return [
        {"id": "generation", "libelle": "Génération des documents", "type": "select", "options": dict(GENERATIONS),
         "defaut": "2023", "aide": "2023 par défaut ; 2008 pour le SAV d'une menuiserie du classeur e.VOLUTION. "
                                   "Les deux ne sont jamais mêlées : chaque écart est affiché."},
        {"id": "configuration", "libelle": "Configuration", "type": "select", "options": dict(CONFIGURATIONS), "defaut": "1v_of"},
        {"id": "largeur_mm", "libelle": "Largeur hors tout (DHT)", "type": "nombre", "min": 300, "max": 6000, "defaut": 1200,
         "unite": "mm"},
        {"id": "hauteur_mm", "libelle": "Hauteur hors tout (DHT)", "type": "nombre", "min": 300, "max": 4000, "defaut": 1400,
         "unite": "mm"},
        {"id": "dormant", "libelle": "Dormant", "type": "select", "options": union(g23.debit.dormants, g08.debit.dormants),
         "defaut": "6101"},
        {"id": "ouvrant", "libelle": "Ouvrant", "type": "select", "options": union(g23.debit.ouvrants, g08.debit.ouvrants),
         "defaut": "6112", "si": {"configuration": list(CFG_OUVRANT) + ["porte_1v", "porte_2v"]}},
        {"id": "meneau", "libelle": "Meneau", "type": "select", "options": union(g23.debit.meneaux, g08.debit.meneaux),
         "defaut": "6127", "si": {"configuration": ["2v_meneau"]}},
        {"id": "battement", "libelle": "Battement extérieur", "type": "select", "options": battements,
         "defaut": next(iter(g23.debit.groupes), None), "si": {"configuration": list(CFG_BATTEMENT)}},
        {"id": "bas", "libelle": "Bas du châssis", "type": "select",
         "options": {"dormant": "Traverse basse de dormant", "seuil": "Seuil (porte-fenêtre)"}, "defaut": "dormant",
         "si": {"configuration": list(CFG_BAS)}},
        {"id": "seuil", "libelle": "Seuil", "type": "select", "options": {s: s for s in d.seuils},
         "defaut": d.seuils[0] if d.seuils else None, "si": {"bas": ["seuil"]}},
        {"id": "finition", "libelle": "Finition des profilés", "type": "select", "options": dict(FINITIONS), "defaut": "blanc"},
        {"id": "vitrage", "libelle": "Vitrage", "type": "select", "options": {v: v for v in VITRAGES}, "defaut": "4-16-4",
         "aide": "verre-lame-verre ; « 44.2 » est un feuilleté de deux verres de 4 mm."},
        {"id": "classe_vent", "libelle": "Classement au vent", "type": "select",
         "options": {k: v["libelle"] for k, v in d.classes.items()}, "defaut": defaut_classe,
         "aide": "Les abaques d'ouvrant n'existent que pour V*A2 et V*A3 ; V*A3 par défaut (catalogue)."},
        {"id": "renfort_2008", "libelle": "Renfort d'ouvrant (abaques 2008)", "type": "select",
         "options": dict(renforts_2008), "defaut": next(iter(renforts_2008), None), "si": {"generation": ["2008"]},
         "aide": "Les planches de 2008 ne nomment aucun ouvrant : on les choisit par le renfort dessiné."},
        {"id": "assemblage", "libelle": "Assemblage du meneau", "type": "select", "options": dict(ASSEMBLAGES),
         "defaut": "set", "si": {"configuration": ["2v_meneau"]}},
        {"id": "portee_cm", "libelle": "Portée du meneau", "type": "nombre", "min": 50, "max": 650, "unite": "cm",
         "requis": False, "si": {"configuration": ["2v_meneau"]},
         "aide": "Vide : la hauteur hors tout, qui majore la portée."},
        {"id": "a_cm", "libelle": "Largeur de charge a", "type": "nombre", "min": 10, "max": 200, "unite": "cm",
         "requis": False, "si": {"configuration": ["2v_meneau"]},
         "aide": "Lue sur le plan de charge (une image) : à saisir."},
        {"id": "b_cm", "libelle": "Largeur de charge b", "type": "nombre", "min": 10, "max": 200, "unite": "cm",
         "requis": False, "si": {"configuration": ["2v_meneau"]}},
    ]


def options(snap: WikiSnapshot) -> Dict[str, Any]:
    d = donnees(snap)
    return {
        "champs": champs(snap),
        "configurations": dict(CONFIGURATIONS),
        "pages": sorted(d.statuts),
        "brouillons": sorted(p for p, st in d.statuts.items() if st == "draft"),
        "avertissements": [
            "Rattachement des gammes PERFORM, HYBRIDE et TEXTURAL au système 70 : écrit dans aucun document (VER-28).",
            "Ferrure et pivot du 70 : aucune borne dans le wiki, « sur étude ». Ouvrant sur seuil : aucune cote de débit.",
        ],
    }


# ---- le calcul du débit ----------------------------------------------------------------------


def _src(page: str, section: str = "", localisation: str = "") -> Dict[str, str]:
    return _source(page, section, localisation)


def _src_ligne(ligne: Dict[str, Any]) -> Dict[str, str]:
    return _src(ligne["page"], ligne["section"], ligne["localisation"])


def calcul_debit(g: Generation, d: Donnees, s: Dict[str, Any]) -> Dict[str, Any]:
    """Dimension hors tout → DEO, DFO et vitrage, étapes écrites comme le manuel. Ce que le wiki ne donne pas n'est
    pas calculé : ``manque`` dit pourquoi."""
    deb = g.debit
    dor = deb.dormants.get(s["dormant"])
    res: Dict[str, Any] = {"etapes": [], "vantaux": 0, "manque": None, "sources": [], "anomalies": [], "notes": []}
    if dor is None or dor["deo"] is None:
        res["manque"] = f"le dormant {s['dormant']} n'a pas de cote de débit dans « {GENERATIONS[g.id]} »"
        if s["dormant"] == "6102" and g.id == "2023":
            res["anomalies"].append("VER-85")
        return res
    a = dor["deo"]
    L, H = s["largeur_mm"], s["hauteur_mm"]
    etapes: List[Dict[str, str]] = res["etapes"]
    res["sources"].append(_src_ligne(dor))
    cfg = s["configuration"]
    if cfg in ("soufflet", "coulissant", "porte_1v", "porte_2v"):
        return res
    if cfg == "fixe":
        v = dor["vitrage"]
        if v is None:
            res["manque"] = f"le dormant {s['dormant']} n'a pas de cote de vitrage fixe"
            return res
        res["vitrage"] = (L - 2 * v, H - 2 * v)
        etapes.append({"etape": "Vitrage du châssis fixe", "calcul": f"DHT − 2 × {_l(v)} = {_l(L - 2 * v)} × {_l(H - 2 * v)}",
                       "resultat": f"{_l(L - 2 * v)} × {_l(H - 2 * v)}"})
        return res
    ouv = deb.ouvrants.get(s["ouvrant"] or "")
    if ouv is None or ouv["vitrage"] is None or ouv["dfo"] is None:
        res["manque"] = f"l'ouvrant {s['ouvrant']} n'a pas de cote de débit dans « {GENERATIONS[g.id]} »"
        return res
    res["sources"].append(_src_ligne(ouv))
    if cfg in ("1v_of", "1v_ob"):
        deo_l = L - 2 * a
        etapes.append({"etape": "Largeur extérieure d'ouvrant", "calcul": f"DEO = {_l(L)} − 2 × {_l(a)}", "resultat": _l(deo_l)})
        res["vantaux"] = 1
    elif cfg == "2v_meneau":
        men = deb.meneaux.get(s["meneau"] or "")
        if men is None or men["deo"] is None:
            res["manque"] = (f"le meneau {s['meneau']} n'a pas de cote ① DEO dans « {GENERATIONS[g.id]} »"
                             if men is None else f"le meneau {s['meneau']} n'a pas de cote ① DEO : le tableau écrit « — »")
            return res
        res["sources"].append(_src_ligne(men))
        x = L / 2
        deo_l = x - (a + men["deo"])
        etapes.append({"etape": "Demi-largeur au meneau", "calcul": f"X = {_l(L)} / 2", "resultat": _l(x)})
        etapes.append({"etape": "Largeur extérieure d'ouvrant", "calcul": f"DEO = X − (a + b) = {_l(x)} − ({_l(a)} + {_l(men['deo'])})",
                       "resultat": _l(deo_l)})
        res["vantaux"] = 2
    else:  # deux vantaux à battement
        groupe = deb.groupes.get(s["battement"] or "")
        ligne = deb.battement_dormant.get((s["dormant"], groupe or ""))
        if groupe is None or ligne is None or not ligne["deo"]:
            res["manque"] = (f"aucune cote de débit pour le dormant {s['dormant']} avec le battement {s['battement']} "
                             f"dans « {GENERATIONS[g.id]} »")
            if s["dormant"] in ("6102", "6155", "6156", "6158"):
                res["anomalies"].append("VER-85")
            return res
        x = L / 2
        try:
            deo_l = evaluer(ligne["deo"], {"X": x})
        except FormuleIllisible as exc:
            res["manque"] = f"formule « {ligne['deo']} » non calculée : {exc}"
            return res
        res["sources"].append(_src(ligne["page"], ligne["section"], ligne["localisation"]))
        etapes.append({"etape": "Axe du châssis au bord du dormant", "calcul": f"X = {_l(L)} / 2", "resultat": _l(x)})
        etapes.append({"etape": "Largeur extérieure d'ouvrant", "calcul": f"DEO = {ligne['deo']} avec X = {_l(x)}", "resultat": _l(deo_l)})
        res["vantaux"] = 2
        # la DFO du tableau doit retomber sur la DEO moins deux fois la feuillure de l'ouvrant : sinon le dire
        if ligne["dfo"]:
            try:
                dfo_tableau = evaluer(ligne["dfo"], {"X": x})
                if abs(dfo_tableau - (deo_l - 2 * ouv["dfo"])) > 1e-6:
                    res["notes"].append(f"la DFO du tableau ({_l(dfo_tableau)} mm) ne retombe pas sur DEO − 2 × {_l(ouv['dfo'])} "
                                        f"({_l(deo_l - 2 * ouv['dfo'])} mm)")
            except FormuleIllisible:
                pass
    # la hauteur
    if s.get("bas") == "seuil":
        res["hauteur_manque"] = "la hauteur d'un ouvrant sur seuil n'a aucune cote de débit dans le wiki"
        deo_h = None
        etapes.append({"etape": "Hauteur extérieure d'ouvrant", "calcul": "ouvrant sur seuil", "resultat": "non donnée"})
    else:
        deo_h = H - 2 * a
        etapes.append({"etape": "Hauteur extérieure d'ouvrant", "calcul": f"DEO = {_l(H)} − 2 × {_l(a)}", "resultat": _l(deo_h)})
    c = ouv["vitrage"]
    vl = deo_l - 2 * c
    vh = deo_h - 2 * c if deo_h is not None else None
    if vl <= 0 or (vh is not None and vh <= 0):
        res["manque"] = f"l'ouvrant ne laisse aucun vitrage ({_l(deo_l)} × {_l(deo_h) if deo_h is not None else '—'} mm de DEO)"
        return res
    etapes.append({"etape": "Vitrage de l'ouvrant",
                   "calcul": f"DEO − 2 × {_l(c)} = {_l(vl)} × {_l(vh) if vh is not None else '—'}",
                   "resultat": f"{_l(vl)} × {_l(vh) if vh is not None else '—'}"})
    res.update(deo=(deo_l, deo_h), vitrage=(vl, vh),
               dfo=(deo_l - 2 * ouv["dfo"], deo_h - 2 * ouv["dfo"] if deo_h is not None else None))
    # le battement : débit à partir de la hauteur d'ouvrant
    if cfg in CFG_BATTEMENT:
        b = deb.battements.get(s["battement"])
        if b is not None and deo_h is not None:
            res["battement"] = {}
            for role, cle in (("exterieur", "exterieur"), ("interieur", "interieur")):
                formule = b.get(cle)
                if formule:
                    try:
                        res["battement"][role] = evaluer(formule, {"DEO": deo_h, "DFO": deo_h - 2 * ouv["dfo"]})
                    except FormuleIllisible as exc:
                        res["notes"].append(f"débit du battement {role} « {formule} » non calculé : {exc}")
            res["sources"].append(_src(b["page"], b["section"], b["localisation"]))
    return res


def _seuil_debit(g: Generation, d: Donnees, s: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Le seuil et les montants sur seuil (variante 1, coupe 90°). Le tableau du seuil est celui de 2023."""
    ligne = d.seuil.get(s["dormant"])
    if ligne is None:
        return None
    out: Dict[str, Any] = {"ligne": ligne, "seuil": s.get("seuil") or (d.seuils[0] if d.seuils else None)}
    try:
        out["largeur"] = evaluer(ligne["largeur"], {"DHT": s["largeur_mm"]})
        out["montant"] = evaluer(ligne["montant"], {"DHT": s["hauteur_mm"]})
    except FormuleIllisible as exc:
        out["manque"] = f"formule du seuil non calculée : {exc}"
    return out


def _ecarts(d: Donnees, s: Dict[str, Any]) -> List[str]:
    """Ce que 2008 change, pour les profilés et le vitrage choisis, par rapport au manuel de 2023."""
    g8, g3 = d.generations["2008"], d.generations["2023"]
    out: List[str] = []

    def cmp(libelle: str, a: Optional[float], b: Optional[float]) -> None:
        if a is not None and b is not None and abs(a - b) > 1e-9:
            out.append(f"{libelle} : {_l(a)} (2008) contre {_l(b)} (2023)")

    dor = s["dormant"]
    if dor in g8.debit.dormants and dor in g3.debit.dormants:
        cmp(f"vitrage fixe du dormant {dor}", g8.debit.dormants[dor]["vitrage"], g3.debit.dormants[dor]["vitrage"])
    if s.get("ouvrant") in g8.debit.ouvrants and s["ouvrant"] in g3.debit.ouvrants:
        cmp(f"vitrage de l'ouvrant {s['ouvrant']}", g8.debit.ouvrants[s["ouvrant"]]["vitrage"], g3.debit.ouvrants[s["ouvrant"]]["vitrage"])
    if s.get("meneau") in g8.debit.meneaux and s["meneau"] in g3.debit.meneaux:
        cmp(f"vitrage fixe du meneau {s['meneau']}", g8.debit.meneaux[s["meneau"]]["vitrage"], g3.debit.meneaux[s["meneau"]]["vitrage"])
    if s.get("battement") in g8.debit.battements and s["battement"] in g3.debit.battements:
        for role in ("exterieur", "interieur"):
            f8, f3 = g8.debit.battements[s["battement"]].get(role), g3.debit.battements[s["battement"]].get(role)
            if f8 and f3 and _formule_lue(f8) != _formule_lue(f3):
                out.append(f"débit du battement {role} {s['battement']} : {_formule_lue(f8)} (2008) contre {_formule_lue(f3)} (2023)")
    cmp("poids de vitrage maxi des abaques (kg/m²)", g8.abaques.verre_kg_m2, g3.abaques.verre_kg_m2)
    cmp("épaisseur de verre maxi des abaques (mm)", g8.abaques.verre_mm, g3.abaques.verre_mm)
    cmp("épaisseur de vitrage maxi admise (mm)", g8.vitrage.plage[1], g3.vitrage.plage[1])
    return out


# ---- l'abaque d'ouvrant ----------------------------------------------------------------------


def _bande(bornes: List[Tuple[int, int]], v: float) -> Optional[int]:
    """L'indice de la bande qui porte v ; sur une graduation, la bande des plus grandes dimensions (la plus exigeante)."""
    trouvees = [k for k, (lo, hi) in enumerate(bornes) if lo <= v <= hi]
    return max(trouvees, key=lambda k: bornes[k][0]) if trouvees else None


def _limite(valeurs: Dict[int, Optional[List[float]]], w: float, haute: bool) -> Tuple[Optional[float], str]:
    """Une limite relevée à chaque graduation de largeur, SANS interpolation : la plus restrictive des deux
    graduations voisines (la plus basse pour une limite haute, la plus haute pour une limite basse)."""
    grad = sorted(valeurs)
    if not grad or w < grad[0] or w > grad[-1]:
        return None, "hors des graduations de largeur"
    bas, haut = max(g for g in grad if g <= w), min(g for g in grad if g >= w)
    lus: List[float] = []
    for gr in {bas, haut}:
        v = valeurs[gr]
        if v:
            lus.append(min(v) if haute else max(v))
    if not lus:
        return None, "la limite ne coupe pas cette largeur"
    return (min(lus) if haute else max(lus)), (f"à {bas} mm" if bas == haut else f"entre {bas} et {haut} mm, la plus restrictive")


def _planche_pour(g: Generation, s: Dict[str, Any], genre: str, classe: str) -> Tuple[Optional[Planche], str]:
    cand = [p for p in g.abaques.planches if p.classe == classe and p.genre == genre and p.variante == "principal"
            and not p.avec_g]
    if g.id == "2008":
        ren = s.get("renfort_2008")
        cand = [p for p in cand if p.renforts and p.renforts[0] == ren]
        return (cand[0], "") if len(cand) == 1 else (None, f"{len(cand)} planches de 2008 pour le renfort {ren}")
    ouvrant = s["ouvrant"]
    cand = [p for p in cand if ouvrant in p.refs]
    if genre in ("of2", "ofob2"):
        zero140 = s.get("battement") == "0140"
        cand = [p for p in cand if any(any(r == "0140" for r in grp) for grp in p.ouvrants) == zero140]
    if len(cand) == 1:
        return cand[0], ""
    return None, ("aucune planche ne nomme cet ouvrant" if not cand else f"{len(cand)} planches possibles")


def _lire(p: Planche, w: float, h: float) -> Dict[str, Any]:
    """La case de l'abaque où tombent la largeur et la hauteur d'ouvrant (jamais d'interpolation)."""
    kb = _bande([(lo, hi) for lo, hi, _, _ in p.bandes], h)
    kc = _bande(p.colonnes, w)
    if kb is None or kc is None:
        return {"etat": "hors", "cellule": None}
    lo, hi, paum, cellules = p.bandes[kb]
    cel = cellules[kc].strip()
    reperes = re.findall(r"[A-G]\d", cel)
    if not reperes:
        return {"etat": "vide", "cellule": cel, "bande": (lo, hi), "colonne": p.colonnes[kc], "paumelles": paum}
    return {"etat": "composite" if len(reperes) > 1 else "ok", "cellule": cel, "reperes": reperes,
            "renvoi": "renvoi" in cel, "bande": (lo, hi), "colonne": p.colonnes[kc], "paumelles": paum}


def _pieces(texte: str) -> Optional[Tuple[int, int]]:
    """(traverses, montants) renforcés d'après la légende d'un repère d'un seul vantail ; None si elle est ambiguë."""
    t = texte.lower()
    if "aucun" in t:
        return 0, 0
    if "vantail" in t:
        return None
    trav = mont = 0
    if "quatre côtés" in t:
        return 2, 2
    if re.search(r"traverse haute et .*traverse basse|deux traverses|traverses haute et basse", t):
        trav = 2
    elif "traverse basse" in t or "traverse haute" in t:
        trav = 1
    if "deux montants" in t:
        mont = 2
    elif "montant" in t:
        mont = 1
    return (trav, mont) if (trav or mont) else None


GENRES = {"of1": "OF 1 vantail", "ob1": "OB 1 vantail", "of2": "2 vantaux OF", "ofob2": "2 vantaux OF/OB"}


def _controle_ouvrant(g: Generation, d: Donnees, s: Dict[str, Any], deb: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Le repère de la case, les limites tracées, la règle des 25 %, la couleur ; puis ce que le repère fait renforcer."""
    cfg = s["configuration"]
    genre = TYPE_ABAQUE[cfg]
    classe = (s["classe_vent"] or "").upper()
    nom_classe = "V*" + classe[1:]
    ab = g.abaques
    titre = "Abaque d'ouvrant"
    src_regles = [_src(ab.page_regles, "Remarques importantes concernant les abaques de renforcement" if g.id == "2023" else "", "")]
    if deb.get("manque") or not deb["vantaux"]:
        return []
    wl, wh = deb["deo"]
    if wh is None:
        return [_controle("ouvrant", titre, ETUDE,
                          "La hauteur d'un ouvrant sur seuil n'a aucune cote de débit dans le wiki : l'abaque ne peut pas être lu.",
                          src_regles)]
    if classe not in ("VA2", "VA3"):
        return [_controle("ouvrant", titre, ETUDE,
                          f"Aucun abaque d'ouvrant n'est dessiné pour le classement {nom_classe} : seuls V*A2 et V*A3 le sont.", src_regles)]
    p, raison = _planche_pour(g, s, genre, classe)
    if p is None:
        quoi = "les ouvrants appariés au battement" if genre in ("of2", "ofob2") else "l'ouvrant"
        return [_controle("ouvrant", titre, ETUDE,
                          f"Pas d'abaque {classe} {GENRES[genre]} pour {quoi} {s['ouvrant']} ({raison}) : hors abaques, sur étude.",
                          [_src(ab.pages[0], "", "")])]
    src = [_src(p.page, p.section, p.localisation)]
    hyp = [HYPOTHESE_AXES] + ([HYPOTHESE_MENEAU] if cfg == "2v_meneau" else [])
    lec = _lire(p, wl, wh)
    statuts: List[str] = []
    phrases: List[str] = []
    # la case
    if lec["etat"] == "hors":
        statuts.append(ETUDE)
        phrases.append(f"{_l(wl)} × {_l(wh)} mm est hors de la grille de l'abaque : le manuel ne donne pas de renfort")
    elif lec["etat"] == "vide":
        statuts.append(ETUDE)
        phrases.append(f"la case {_l(lec['bande'][0])}–{_l(lec['bande'][1])} mm de hauteur × {lec['colonne'][0]}–{lec['colonne'][1]} mm de "
                       "largeur n'a pas de repère (« - » : hors de la surface réalisable)")
    elif lec["etat"] == "composite":
        statuts.append(ETUDE)
        phrases.append("la case porte deux repères (" + " / ".join(lec["reperes"]) + ") : lequel s'applique dépend de la zone, que le "
                       "tableau ne dit pas")
    else:
        phrases.append(f"repère {lec['reperes'][0]} (case {_l(lec['bande'][0])}–{_l(lec['bande'][1])} mm de hauteur × "
                       f"{lec['colonne'][0]}–{lec['colonne'][1]} mm de largeur)"
                       + (", case coupée par une limite (renvoi)" if lec["renvoi"] else ""))
        if lec["renvoi"]:
            statuts.append(ETUDE)
    # les limites tracées, à la précision de lecture écrite par la page
    prec = ab.precision_mm
    lim_r, com_r = _limite(p.limites["realisation"], wl, True) if p.limites.get("realisation") else (None, "")
    lim_o, com_o = _limite(p.limites["oblique"], wl, False) if p.limites.get("oblique") else (None, "")
    proche = f"à moins de la précision de lecture (±{_l(prec)} mm) de"
    if lim_r is not None:
        v = _borne(wh, lim_r, prec)
        statuts.append(v)
        if v != OK:
            phrases.append(f"{proche if v == ETUDE else 'au-delà de'} la limite de réalisation : {_l(wh)} mm pour {_l(lim_r)} mm admis ({com_r})")
    if lim_o is not None:
        v = OK if wh >= lim_o + prec else (ETUDE if wh > lim_o - prec else HORS)
        statuts.append(v)
        if v != OK:
            phrases.append(f"{proche if v == ETUDE else 'sous'} la limite oblique basse : {_l(wh)} mm pour {_l(lim_o)} mm mini ({com_o})")
    # la largeur ne dépasse pas 125 % de la hauteur
    if wl > ab.rapport_max * wh:
        statuts.append(HORS)
        phrases.append(f"la largeur {_l(wl)} mm dépasse de plus de {_l((ab.rapport_max - 1) * 100)} % la hauteur {_l(wh)} mm")
    # le film ou la laque peut être sombre (L* sous le seuil du DTD) : renfort systématique, V258 exclu
    if s["finition"] == "filme":
        statuts.append(ETUDE)
        phrases.append(f"profilés filmés, plaxés ou laqués : si le L* du film est inférieur à {_l(d.lstar_mini)}, le renfort est "
                       "systématique (DTD 2.2.3.4) et le renfort V258 est exclu ; le wiki n'a pas le L* des films, le repère de la case "
                       "ne suffit donc pas")
    pire = HORS if HORS in statuts else (ETUDE if ETUDE in statuts else OK)
    texte = f"Vantail {_l(wl)} × {_l(wh)} mm (L × H, DEO), abaque {classe} {GENRES[p.genre]} : " + " ; ".join(phrases) + "."
    # l'« autre possibilité » du V258, quand la planche la porte : donnée à titre d'information
    autre = next((q for q in ab.planches if q.variante == "autre" and q.page == p.page and q.titre == p.titre and q.pdf == p.pdf), None)
    if autre is not None and lec["etat"] == "ok":
        la = _lire(autre, wl, wh)
        if la["etat"] in ("ok", "composite"):
            texte += f" « Autre possibilité d'utilisation du V258 » de la même planche : repère {' / '.join(la['reperes'])} (non retenue)."
    texte += " (" + " ; ".join(hyp) + ".)"
    ctl = [_controle("ouvrant", titre, pire, texte, src, list(p.anomalies),
                     {"repere": (lec.get("reperes") or [None])[0], "paumelles": lec.get("paumelles"), "planche": p.titre,
                      "pdf": p.pdf, "renforts": p.renforts, "genre": p.genre, "hypotheses": hyp})]
    if lec["etat"] == "ok":
        rep = lec["reperes"][0]
        leg = ab.legendes.get(p.genre, {})
        lettre, chiffre = rep[0], rep[1]
        det = f"Repère {rep} : {lettre} = {leg.get(lettre, 'légende non lue')} ; {chiffre} = {leg.get(chiffre, 'légende non lue')}."
        if p.renforts:
            det += (f" Renfort dessiné sur la planche : {' et '.join(p.renforts)}"
                    + (" (V258 : zone grisée, non lue ici)" if "V258" in p.renforts else "") + ".")
        if lec.get("paumelles"):
            det += f" Bande de paumelles de l'abaque : {lec['paumelles']}."
        if ab.entraxe_mm:
            det += f" Entraxe maxi de verrouillage : {_l(ab.entraxe_mm)} mm."
        if wl > d.traverse_renfort_mm:
            det += f" Les traverses d'ouvrant de plus de {_l(d.traverse_renfort_mm)} mm de large sont renforcées systématiquement (DTD 2.2.3.4)."
        ctl.append(_controle("renfort", "Renforcement et verrouillage", INFO, det, src, ["CTR-82"] if g.id == "2008" else [],
                             valeurs={"repere": rep, "lettre": lettre, "chiffre": chiffre, "renforts": p.renforts,
                                      "pieces": _pieces(leg.get(lettre, "")) if p.genre in ("of1", "ob1") else None,
                                      "legende": leg.get(lettre), "paumelles": lec.get("paumelles")}))
    return ctl


# ---- le vitrage et la parclose -----------------------------------------------------------------

REFS_CTR92 = ("2454", "2435", "0136")


def _controle_parclose(g: Generation, d: Donnees, s: Dict[str, Any], vit) -> Dict[str, Any]:
    """La parclose à l'épaisseur totale du vitrage, dans la plage que le tableau écrit pour chaque ligne."""
    v = g.vitrage
    t = vit.total_mm
    titre = "Vitrage et parclose"
    src_epaisseur = _src(v.pages[1], "Épaisseurs de vitrage admises" if g.id == "2023" else "Épaisseurs du vitrage (page imprimée 2)")
    anomalies = ["CTR-91"] if (g.id == "2008" or t > d.generations["2008"].vitrage.plage[1]) else []
    mm = f"{_l(t)} mm"
    if t < v.plage[0]:
        return _controle("parclose", titre, HORS, f"Vitrage de {mm} : en dessous des {_l(v.plage[0])} mm admis.", [src_epaisseur], anomalies)
    if t > v.plage_elargisseur[1]:
        return _controle("parclose", titre, HORS, f"Vitrage de {mm} : au-delà des {_l(v.plage_elargisseur[1])} mm admis avec l'élargisseur "
                         "de parclose.", [src_epaisseur], anomalies)
    largeur = 54 if t <= v.plage[1] else 70
    f = v.feuillures[largeur]
    src = [_src(f["page"], f["section"], f["localisation"]), src_epaisseur]
    avec = f" (feuillure de {largeur} mm" + (", avec l'élargisseur de feuillure)" if largeur == 70 else ")")
    dans = [r for r in f["rows"] if r["mini"] is not None and r["maxi"] is not None and r["mini"] <= t <= r["maxi"]]
    # Deux parcloses sans plage écrite (76509, 76515) ne se lisent qu'à leur épaisseur nominale.
    nominales = [r for r in f["rows"] if r["mini"] is None and r["maxi"] is None and r["nominal"] is not None
                 and abs(r["nominal"] - t) < 0.005]
    choisies = sorted(dans or nominales, key=lambda r: (abs(r["nominal"] - t), r["ecriture"].startswith("sous")))
    if not choisies:
        def haut(r):
            return r["maxi"] if r["maxi"] is not None else r["nominal"]

        def bas(r):
            return r["mini"] if r["mini"] is not None else r["nominal"]

        dessous = max((r for r in f["rows"] if haut(r) < t), key=haut, default=None)
        dessus = min((r for r in f["rows"] if bas(r) > t), key=bas, default=None)
        voisins = " et ".join(x for x in (
            f"{' / '.join(dessous['refs'])} (jusqu'à {_l(haut(dessous))} mm)" if dessous else "",
            f"{' / '.join(dessus['refs'])} (dès {_l(bas(dessus))} mm)" if dessus else "") if x)
        return _controle("parclose", titre, ETUDE,
                         f"Vitrage de {mm}{avec} : aucune parclose du tableau ne couvre cette épaisseur"
                         + (f" (les plus proches : {voisins})" if voisins else "") + ". Le tableau a des trous que le manuel ne comble pas.",
                         src, anomalies + (["VER-86"] if g.id == "2023" else []), {"parcloses": []})
    premiere = choisies[0]
    alternatives = [r for r in choisies[1:] if r["ecriture"].startswith("sous") and r["nominal"] == premiere["nominal"]]
    autres = [r for r in choisies[1:] if r not in alternatives]
    det = (f"Vitrage de {mm} ({_l(vit.verre_mm)} mm de verre){avec} : parclose {' / '.join(premiere['refs'])}, épaisseur nominale "
           f"{_l(premiere['nominal'])} mm"
           + (f", plage écrite {_l(premiere['mini'])} à {_l(premiere['maxi'])} mm" if premiere["mini"] is not None else ", sans plage écrite"))
    if alternatives:
        det += " ; autres parcloses données pour la même épaisseur : " + ", ".join(" / ".join(r["refs"]) for r in alternatives)
    if autres:
        det += (" ; la plage de " + ", ".join(" / ".join(r["refs"]) for r in autres)
                + " couvre aussi cette épaisseur, le critère entre les deux n'est pas donné")
    statut = OK
    if largeur == 70:
        det += f". Au-delà de {_l(v.plage[1])} mm le document exige l'élargisseur de parclose"
    if d.vitrage_dtd_mm is not None and t > d.vitrage_dtd_mm:
        # le DTD (2.2.3.7) s'arrête à cette épaisseur : le manuel va plus loin, pas le DTD
        statut = ETUDE
        det += f" ; le DTD s'arrête à {_l(d.vitrage_dtd_mm)} mm de vitrage (2.2.3.7)"
    det += "."
    anom = anomalies + (["CTR-92"] if any(set(r["refs"]) & set(REFS_CTR92) for r in choisies) else [])
    return _controle("parclose", titre, statut, det, src, anom,
                     {"parcloses": [{"refs": r["refs"], "nominal": r["nominal"], "mini": r["mini"], "maxi": r["maxi"], "image": r["image"]}
                                    for r in choisies], "feuillure": largeur})


# ---- les limites de baie ----------------------------------------------------------------------


def _controle_dtd(d: Donnees, s: Dict[str, Any]) -> Dict[str, Any]:
    cfg = s["configuration"]
    L, H = s["largeur_mm"] / 1000, s["hauteur_mm"] / 1000
    src = [_src(PAGE_DTD, d.dtd_section, d.dtd_localisation)]
    titre = "Dimensions maximales de baie (DTD)"
    if cfg == "fixe":
        return _controle("dtd", titre, INFO, "Le DTD ne donne pas de dimension maximale pour un châssis fixe.", src)
    libelles, mode = LIGNES_DTD[cfg]
    lignes = [(c, h, l) for c, h, l in d.dtd if c in libelles]
    if not lignes:
        return _controle("dtd", titre, ETUDE, f"Aucune ligne du DTD pour « {' » ou « '.join(libelles)} ».", src)
    verdicts = [(OK if (H <= h and L <= l) else HORS, c, h, l) for c, h, l in lignes]
    couples = " ou ".join(f"{_fmt(h, 2)} × {_fmt(l, 2)} m ({c})" for _, c, h, l in verdicts)
    if mode == "une":
        v = OK if any(x[0] == OK for x in verdicts) else HORS
    else:
        v = OK if all(x[0] == OK for x in verdicts) else (HORS if all(x[0] == HORS for x in verdicts) else ETUDE)
    anomalies = ["VER-30"] if any("Oscillo battante 2 vantaux" in c for _, c, _, _ in verdicts) else []
    gauche = f"{_fmt(H, 2)} × {_fmt(L, 2)} m (H × L)"
    if v == OK:
        det = f"{gauche} dans la limite {couples}."
    elif v == ETUDE:
        det = (f"{gauche} passe l'une des deux lignes à deux vantaux sans passer l'autre ({couples}) : le tableau ne dit pas "
               "laquelle vaut pour deux vantaux séparés par un meneau.")
    else:
        det = (f"{gauche} dépasse {couples}. Pour une fabrication certifiée, des dimensions supérieures peuvent être "
               "admises par le Certificat de Qualification du menuisier ; elles ne sont pas dans le wiki.")
    return _controle("dtd", titre, v, det, src, anomalies, {"limites": [(h, l) for _, _, h, l in verdicts]})


def _controle_porte(g: Generation, d: Donnees, s: Dict[str, Any]) -> Dict[str, Any]:
    titre = "Dimensions maximales de la porte d'entrée"
    src = [_src(PAGE_PORTE, "Dimensions maximales d'ouvrant", d.porte_localisation)]
    if g.id != "2023":
        return _controle("porte", titre, ETUDE, "Seul le manuel de 2023 donne le tableau des dimensions maximales d'ouvrant "
                         "lu ici ; le classeur de 2008 n'est pas lu pour la porte (CTR-99).", src, ["CTR-99"])
    cle = "Porte 1 vantail" if s["configuration"] == "porte_1v" else "Porte 2 vantaux"
    ligne = d.porte.get(cle)
    dor = g.debit.dormants.get(s["dormant"])
    if not ligne or dor is None or dor["deo"] is None:
        return _controle("porte", titre, ETUDE, f"Pas de limite de porte lue pour « {cle} » avec le dormant {s['dormant']}.", src)
    col = next((h for h in ligne if (("plax" in h) == (s["finition"] == "filme"))), None)
    if col is None:
        return _controle("porte", titre, ETUDE, "Colonne de finition introuvable.", src)
    lim_l, lim_h = ligne[col]
    a = dor["deo"]
    # La largeur d'ouvrant est connue ; la hauteur d'une porte sur seuil ne l'est pas (aucune cote de débit) : on n'en a
    # qu'une borne haute, la hauteur du dormant moins la traverse haute.
    w = (s["largeur_mm"] - 2 * a) / (2 if s["configuration"] == "porte_2v" else 1)
    h_haut = s["hauteur_mm"] - a
    det = (f"{cle}, {col.lower()} : ouvrant d'au plus {_l(lim_l)} × {_l(lim_h)} mm. Largeur d'ouvrant "
           f"{'(par vantail, au plus) ' if s['configuration'] == 'porte_2v' else ''}{_l(w)} mm ; hauteur d'ouvrant au plus {_l(h_haut)} mm "
           "(la cote de débit du bas sur seuil n'existe pas dans le wiki : borne haute).")
    if s["configuration"] == "porte_2v":
        det += " Largeur par vantail : borne haute, le débit des battements d'une porte à deux vantaux n'est pas écrit."
    if w <= lim_l and h_haut <= lim_h:
        v = OK
    elif s["configuration"] == "porte_1v" and w > lim_l:
        v = HORS
    else:
        v = ETUDE
    return _controle("porte", titre, v, det, src, ["CTR-99"], {"limite": (lim_l, lim_h), "ouvrant_max": (w, h_haut)})


# ---- l'inertie d'un meneau --------------------------------------------------------------------


def _controle_meneau(snap: WikiSnapshot, g: Generation, d: Donnees, s: Dict[str, Any]) -> Dict[str, Any]:
    titre = "Inertie du meneau"
    ref = s["meneau"]
    ren = g.meneaux_renfort.get(ref)
    ine = _inertie(snap, d, s["classe_vent"])
    src = [_src(ine["page"], ine["section"], ine["localisation"])]
    if ren is None or ren["iw"] is None:
        return _controle("meneau", titre, ETUDE, f"Le renfort du meneau {ref} n'a pas d'inertie dans « {GENERATIONS[g.id]} ».", src,
                         ["INC-141"])
    src.append(_src(ren["page"], ren["section"], ren["localisation"]))
    anomalies = ["INC-141"] + (["INC-144"] if ref == "6127" else [])
    a, b = s.get("a_cm"), s.get("b_cm")
    portee = s.get("portee_cm")
    majore = portee is None
    if majore:
        portee = s["hauteur_mm"] / 10
    if a is None or b is None:
        return _controle("meneau", titre, ETUDE,
                         "Les largeurs de charge a et b se lisent sur les plans de charge (des images) : elles ne sont pas "
                         "saisies, l'inertie requise n'est pas calculée.", src, anomalies)
    grille = ine["grille"]
    portees = sorted(grille)
    largeurs = ine["largeurs"]
    if portee > portees[-1]:
        return _controle("meneau", titre, ETUDE, f"Portée de {_l(portee)} cm : au-delà de la table ({portees[-1]} cm).", src, anomalies)
    lp = min(x for x in portees if x >= portee)
    lus, manques = [], []
    for nom, larg in (("a", a), ("b", b)):
        if larg > largeurs[-1]:
            manques.append(f"{nom} = {_l(larg)} cm : au-delà de la table ({largeurs[-1]} cm)")
            continue
        gl = min(x for x in largeurs if x >= larg)
        val = grille[lp].get(gl)
        if val is None:
            manques.append(f"{nom} = {_l(larg)} cm : case vide à la portée {lp} cm (a dépasse la moitié de la portée)")
        else:
            lus.append((nom, larg, gl, val))
    if manques:
        return _controle("meneau", titre, ETUDE, "Lecture impossible : " + " ; ".join(manques) + ".", src, anomalies)
    requis = sum(v for *_, v in lus)
    marge = PRECISION_CASE_INERTIE * len(lus)
    lecture = " + ".join(f"{_l(v)} ({n} = {_l(l)} cm lu à {gl} cm)" for n, l, gl, v in lus)
    iw = ren["iw"]
    if iw >= requis + marge:
        v = OK
    elif iw >= requis - marge:
        v = ETUDE
    else:
        v = HORS
    det = (f"Meneau {ref}, renfort {ren['renfort']} : Iw = {_l(iw)} cm⁴ pour {_l(requis)} cm⁴ requis ({lecture}), "
           f"portée lue à {lp} cm, classement {ine['libelle']} ; chaque case est à ±{_fmt(PRECISION_CASE_INERTIE, 2)} près. Lecture sans interpolation : la "
           "graduation supérieure de la portée et de chaque largeur de charge."
           + (" Portée non saisie : la hauteur hors tout, qui la majore (hypothèse)." if majore else ""))
    if v == HORS:
        det += " Le renfort de ce meneau ne suffit pas."
    return _controle("meneau", titre, v, det, src, anomalies, {"requis_cm4": round(requis, 2), "iw_cm4": iw, "renfort": ren["renfort"],
                                                              "portee_lue_cm": lp})


# ---- le verdict ------------------------------------------------------------------------------


def _verifier_planches(deb: Debit, s: Dict[str, Any]) -> Optional[Tuple[str, List[str]]]:
    """VER-85 : une planche de battement ne liste ni tous les dormants, ni tous les ouvrants."""
    b, o, dor = s["battement"], s["ouvrant"], s["dormant"]
    rangs = [p for p in deb.planches or [] if b in p["exterieurs"]]
    if not rangs:
        return f"Le battement {b} n'est dessiné sur aucune planche de débit.", ["VER-85"]
    avec = [p for p in rangs if p["ouvrants"] is None or o in p["ouvrants"]]
    if not avec:
        return (f"L'ouvrant {o} n'est dessiné avec le battement {b} sur aucune planche (les ouvrants dessinés sont des ouvrants "
                "réduits)."), ["VER-85"]
    if all(dor in p["absents"] for p in avec):
        return (f"Le dormant {dor} n'apparaît pas sur la planche du battement {b} avec l'ouvrant {o} : la valeur du tableau n'est pas "
                "confirmée par la planche."), ["VER-85"]
    return None


def _controle_debit(g: Generation, d: Donnees, s: Dict[str, Any], deb: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = s["configuration"]
    titre = "Cotes de débit"
    if cfg in ("soufflet", "coulissant", "porte_1v", "porte_2v"):
        return []
    if deb["manque"]:
        return [_controle("debit", titre, ETUDE, deb["manque"][0].upper() + deb["manque"][1:] + ".", deb["sources"], deb["anomalies"],
                          {"etapes": deb["etapes"]})]
    if cfg == "fixe":
        vl, vh = deb["vitrage"]
        return [_controle("debit", titre, INFO, f"Vitrage du châssis fixe {_l(vl)} × {_l(vh)} mm (DHT − 2 × ③).", deb["sources"],
                          deb["anomalies"], {"etapes": deb["etapes"], "vitrage": deb["vitrage"]})]
    wl, wh = deb["deo"]
    vl, vh = deb["vitrage"]
    anomalies = list(deb["anomalies"])
    det = (f"Chaque vantail mesure {_l(wl)} × {_l(wh) if wh is not None else '(hauteur non donnée)'} mm hors tout d'ouvrant (DEO) ; "
           f"vitrage {_l(vl)} × {_l(vh) if vh is not None else '—'} mm.")
    if g.id == "2023" and s["ouvrant"] == "6112":
        anomalies.append("INC-125")
        c8 = d.generations["2008"].debit.ouvrants["6112"]["vitrage"]
        det += (f" Le tableau donne {_l(g.debit.ouvrants['6112']['vitrage'])} à déduire pour le 6112 ; la vignette du manuel (et le classeur "
                f"de 2008) donnent {_l(c8)}, soit {_l(wl - 2 * c8)} mm de vitrage en largeur au lieu de {_l(vl)} (INC-125).")
    if g.id == "2008":
        anomalies.append("CTR-78")
        if cfg == "2v_meneau" and s["meneau"] == "6127":
            anomalies.append("INC-217")
    if deb["notes"]:
        det += " " + " ; ".join(n[0].upper() + n[1:] for n in deb["notes"]) + "."
    statut = INFO
    if deb.get("hauteur_manque"):
        statut = ETUDE
        det += " " + deb["hauteur_manque"][0].upper() + deb["hauteur_manque"][1:] + " : sur étude."
    if cfg in CFG_BATTEMENT:
        det += " Le second vantail est un ouvrant réduit apparié au battement : ses cotes de débit ne sont pas dans le wiki."
        b = deb.get("battement") or {}
        if b:
            morceaux = [f"extérieur {_l(b['exterieur'])} mm"] if b.get("exterieur") is not None else []
            morceaux += [f"intérieur {_l(b['interieur'])} mm"] if b.get("interieur") is not None else []
            det += f" Débit du battement : {', '.join(morceaux)} ({HYPOTHESE_BATTEMENT})."
            anomalies += ["CTR-79"] + (["INC-153"] if g.id == "2023" else []) + ANOMALIES_BATTEMENT[g.id].get(s["battement"], [])
        if g.debit.planches is not None:
            pl = _verifier_planches(g.debit, s)
            if pl:
                statut = ETUDE
                det += " " + pl[0]
                anomalies += pl[1]
    return [_controle("debit", titre, statut, det, deb["sources"], sorted(set(anomalies)),
                      {"etapes": deb["etapes"], "deo": deb["deo"], "dfo": deb["dfo"], "vitrage": deb["vitrage"],
                       "battement": deb.get("battement")})]


def _controle_seuil(g: Generation, d: Donnees, s: Dict[str, Any]) -> Dict[str, Any]:
    titre = "Seuil et montants sur seuil"
    section = "Variante 1 : montage coupe 90° sur dormant, débit des seuils 9F67 et 9F68"
    if g.id != "2023":
        return _controle("seuil", titre, ETUDE, "Le tableau de débit des seuils est celui du manuel de 2023 ; aucun n'est lu pour 2008.",
                         [_src(PAGE_SEUIL, section)])
    sd = _seuil_debit(g, d, s)
    if sd is None:
        return _controle("seuil", titre, ETUDE, f"Le dormant {s['dormant']} n'a pas de ligne dans le tableau de débit des seuils (variante 1).",
                         [_src(PAGE_SEUIL, section)])
    if "largeur" not in sd:
        return _controle("seuil", titre, ETUDE, sd["manque"][0].upper() + sd["manque"][1:] + ".", [_src_ligne(sd["ligne"])])
    emb = sd["ligne"]["embouts"].get(sd["seuil"])
    return _controle("seuil", titre, INFO,
                     f"Seuil {sd['seuil']} : {_l(sd['largeur'])} mm ({sd['ligne']['largeur']} avec DHT = {_l(s['largeur_mm'])}) ; montants "
                     f"de dormant {_l(sd['montant'])} mm ({sd['ligne']['montant']} avec DHT = {_l(s['hauteur_mm'])}) ; embout "
                     f"{emb or 'non écrit'}. Variante 1, coupe 90° sur dormant.",
                     [_src_ligne(sd["ligne"])], [], {"seuil": sd["largeur"], "montant": sd["montant"], "embout": emb})


def _controle_poids(g: Generation, s: Dict[str, Any], vit, deb: Dict[str, Any]) -> Dict[str, Any]:
    """Les abaques sont établis jusqu'à un poids de vitrage ; au-delà, le renforcement systématique est conseillé."""
    ab = g.abaques
    kg_m2 = vit.verre_mm * ab.densite
    src = [_src(ab.page_regles, "Remarques importantes concernant les abaques de renforcement" if g.id == "2023" else "", "")]
    det = (f"{_l(vit.verre_mm)} mm de verre à {_l(ab.densite)} kg/m² par mm : {_l(kg_m2)} kg/m². Les abaques de "
           f"{GENERATIONS[g.id].split(',')[0]} sont établis jusqu'à {_l(ab.verre_kg_m2)} kg/m² ({_l(ab.verre_mm)} mm de verre)")
    if s["configuration"] in CFG_OUVRANT and not deb["manque"] and deb["vantaux"] and deb["vitrage"][1] is not None:
        poids = kg_m2 * (deb["vitrage"][0] / 1000) * (deb["vitrage"][1] / 1000)
        det += f" ; {_l(poids)} kg de verre par vantail ({HYPOTHESE_VERRE})"
    det += "."
    if kg_m2 > ab.verre_kg_m2 + 1e-9:
        return _controle("poids", "Poids du vitrage", ETUDE,
                         det + " Au-delà, les abaques ne s'appliquent plus : le renforcement systématique est conseillé.", src, ["CTR-80"])
    return _controle("poids", "Poids du vitrage", INFO, det, src, ["CTR-80"])


def verifier(snap: WikiSnapshot, saisie: Dict[str, Any]) -> Dict[str, Any]:
    d = donnees(snap)
    s = coercer(champs(snap), saisie)
    g = d.generations[s["generation"]]
    cfg = s["configuration"]
    vit = lire_vitrage(s["vitrage"])
    ctl: List[Dict[str, Any]] = []

    # 0. ce qu'aucun document ne dit : toujours en tête
    ctl.append(_controle("ver28", "Rattachement de la gamme au système 70", INFO, VER28 + f" Génération lue : {GENERATIONS[g.id]}.",
                         [_src(p, "", "") for p in PAGES_VER28 if p in snap.pages], ["VER-28"]))
    if g.id == "2008":
        ec = _ecarts(d, s)
        ctl.append(_controle("generation", "Écart avec le manuel de 2023", INFO,
                             "Valeurs du classeur e.VOLUTION d'août 2008, jamais mêlées à celles de 2023. "
                             + (("Écarts pour cette saisie : " + " ; ".join(ec) + ".") if ec else "Aucun écart pour cette saisie.")
                             + " Les parcloses (CTR-92) et les épaisseurs de vitrage admises (CTR-91) diffèrent aussi.",
                             [_src(g.debit.pages[0], "", ""), _src(g.abaques.page_regles, "", "")],
                             ["CTR-78"] + (["CTR-79"] if cfg in CFG_BATTEMENT else []) + ["CTR-80", "CTR-91"]))

    # 1. le débit, puis le seuil
    deb = calcul_debit(g, d, s)
    ctl += _controle_debit(g, d, s, deb)
    if s.get("bas") == "seuil" and cfg in CFG_OUVRANT and not deb["manque"]:
        ctl.append(_controle_seuil(g, d, s))

    # 2. les limites de baie
    ctl.append(_controle_porte(g, d, s) if cfg in ("porte_1v", "porte_2v") else _controle_dtd(d, s))

    # 3. l'abaque d'ouvrant et le renfort
    if cfg in TYPE_ABAQUE:
        ctl += _controle_ouvrant(g, d, s, deb)

    # 4. le poids de vitrage pour lequel les abaques sont établis
    if cfg in CFG_OUVRANT or cfg in ("porte_1v", "porte_2v", "fixe"):
        ctl.append(_controle_poids(g, s, vit, deb))

    # 5. le meneau
    if cfg == "2v_meneau":
        ctl.append(_controle_meneau(snap, g, d, s))

    # 6. la parclose
    ctl.append(_controle_parclose(g, d, s, vit))

    # 7. le ferrage que le DTD écrit, et la ferrure que le wiki ne borne nulle part
    if cfg in CFG_OUVRANT and not deb["manque"] and deb["vantaux"] and deb["deo"][1] is not None:
        h_m = deb["deo"][1] / 1000
        n = next((nb for lim, nb in d.fiches if h_m <= lim), d.fiches[-1][1])
        det = f"{n} fiches par vantail pour {_fmt(h_m, 2)} m de hauteur d'ouvrant (DTD 2.2.3.6)."
        if d.verrouillage_mm and max(deb["deo"]) >= d.verrouillage_mm:
            det += f" Verrouillage complémentaire haut et bas, centré, à partir de {_l(d.verrouillage_mm)} mm."
        ctl.append(_controle("ferrage", "Ferrage et verrouillage", INFO, det, [_src(PAGE_DTD, "2.2.3.6 Ferrage – Verrouillage")]))
    if cfg in CFG_OUVRANT or cfg in ("porte_1v", "porte_2v"):
        ctl.append(_controle("ferrure", "Ferrure et pivot", INFO,
                             "Sur étude : le wiki ne donne aucune borne de ferrure ni de pivot pour le système 70. Le DTD écrit des "
                             "crémones FERCO, « d'autres quincailleries sont possibles sur justifications » ; aucun champ d'application, "
                             "aucune charge admissible. Ce contrôle ne décide pas du verdict : il reste à faire.",
                             [_src(PAGE_DTD, "2.2.3.6 Ferrage – Verrouillage"), _src("/quincaillerie/ferco.md", "", "")], ["VER-28"]))

    # les pages en brouillon que ce calcul a lues : celles que ses sources citent
    brouillons = sorted({src["page"] for c in ctl for src in c["sources"]
                         if snap.pages.get(src["page"]) and snap.pages[src["page"]].status == "draft"})
    notes = ["Verdict sous réserve : VER-28 (le rattachement des gammes PERFORM, HYBRIDE et TEXTURAL au système 70 n'est établi dans "
             "aucun document), et la ferrure n'est pas contrôlée."]
    r = synthese(ctl, brouillons, notes=notes, generation=g.id, vitrage=vit.__dict__,
                 debit={k: deb.get(k) for k in ("deo", "dfo", "vitrage", "etapes", "vantaux", "battement")})
    return wiki_famille.relocaliser(snap, r)


# ---- la fiche de débit -----------------------------------------------------------------------


def _formule_lue(formule: str) -> str:
    """« « Débit battement (6130) = DEO – 70 mm » » → « DEO – 70 mm »."""
    return formule.replace("«", "").replace("»", "").rsplit("=", 1)[-1].strip()


def fiche_debit(snap: WikiSnapshot, saisie: Dict[str, Any]) -> Dict[str, Any]:
    """Ce que l'atelier coupe et commande, depuis les mêmes cotes que la faisabilité. Rien d'inventé : ce que le wiki ne donne pas
    (surlongueur de soudure, longueur des parcloses, débit de l'ouvrant réduit d'un battement) reste sans longueur, avec sa note."""
    d = donnees(snap)
    s = coercer(champs(snap), saisie)
    g = d.generations[s["generation"]]
    cfg = s["configuration"]
    if cfg in ("soufflet", "coulissant", "porte_1v", "porte_2v"):
        return {"indisponible": "Aucune cote de débit dans le wiki pour cette configuration : le système 70 n'y a que la limite de "
                                "dimension (onglet Faisabilité)."}
    deb = calcul_debit(g, d, s)
    if deb["manque"]:
        return {"indisponible": deb["manque"][0].upper() + deb["manque"][1:] + "."}
    ver = verifier(snap, saisie)
    ctl = {c["id"]: c for c in ver["controles"]}
    dor = g.debit.dormants[s["dormant"]]
    L, H = s["largeur_mm"], s["hauteur_mm"]
    lignes: List[Dict[str, Any]] = []
    notes: List[str] = []
    src_dor = _src_ligne(dor)

    def ligne(famille, piece, ref, qte, longueur, calcul, coupe, source, note="", image=None):
        lignes.append({"rep": len(lignes) + 1, "famille": famille, "piece": piece, "ref": ref, "qte": qte, "longueur": longueur,
                       "calcul": calcul, "coupe": coupe, "source": source, "note": note, "image": image})

    def image_de(profil: str, designation: str) -> Optional[str]:
        return next((a["image"] for a in d.accessoires.get(profil, []) if a["designation"].startswith(designation) and a["image"]), None)

    # ---- le dormant : les barres soudées sortent en cote finie (le wiki ne donne pas de surlongueur de soudure)
    sd = _seuil_debit(g, d, s) if (s.get("bas") == "seuil" and cfg in CFG_OUVRANT and g.id == "2023") else None
    sur_seuil = sd is not None and "largeur" in sd
    if s.get("bas") == "seuil" and cfg in CFG_OUVRANT and not sur_seuil:
        notes.append("Seuil : débit non calculé (" + ("le tableau des seuils est celui de 2023" if g.id == "2008" else
                                                      "dormant absent du tableau de débit des seuils, ou formule illisible") + ").")
    if sur_seuil:
        ligne("Dormant", "Traverse haute", s["dormant"], 1, L, _l(L), "45° soudé", src_dor, "cote finie", image_de(s["dormant"], "Dormant"))
        ligne("Dormant", "Montant sur seuil", s["dormant"], 2, sd["montant"], f"{sd['ligne']['montant']} avec DHT = {_l(H)}",
              "coupe 90° sur le seuil", _src_ligne(sd["ligne"]), "variante 1 : coupe 90° sur dormant")
        ligne("Seuil", f"Seuil {sd['seuil']}", sd["seuil"], 1, sd["largeur"], f"{sd['ligne']['largeur']} avec DHT = {_l(L)}", "droite",
              _src_ligne(sd["ligne"]), "embouts : " + ", ".join(f"{k} → {v}" for k, v in sd["ligne"]["embouts"].items()))
    else:
        ligne("Dormant", "Traverse haute et basse", s["dormant"], 2, L, _l(L), "45° soudé", src_dor, "cote finie", image_de(s["dormant"], "Dormant"))
        ligne("Dormant", "Montant", s["dormant"], 2, H, _l(H), "45° soudé", src_dor, "cote finie")
    r = dor["renfort"]
    renforts_dormant = [a["ref"] for a in d.accessoires.get(s["dormant"], []) if a["designation"].startswith("Renfort")]
    choix = " / ".join(renforts_dormant) if renforts_dormant else "—"
    note_r = "au choix selon la statique" if len(renforts_dormant) > 1 else ("" if renforts_dormant else "référence non donnée")
    if r is not None:
        ligne("Renfort", "Renfort de dormant, traverses", choix, 1 if sur_seuil else 2, L - 2 * r, f"{_l(L)} − 2 × {_l(r)}", "droite",
              src_dor, note_r, image_de(s["dormant"], "Renfort"))
        if sur_seuil:
            ligne("Renfort", "Renfort de dormant, montants sur seuil", choix, 2, None, "", "droite", src_dor,
                  "le débit du renfort d'un montant sur seuil n'est pas écrit")
        else:
            ligne("Renfort", "Renfort de dormant, montants", choix, 2, H - 2 * r, f"{_l(H)} − 2 × {_l(r)}", "droite", src_dor, note_r)

    # ---- le vitrage d'un châssis fixe
    if cfg == "fixe":
        vl, vh = deb["vitrage"]
        ligne("Vitrage", "Vitrage fixe", s["vitrage"], 1, None, f"{_l(vl)} × {_l(vh)} (DHT − 2 × {_l(dor['vitrage'])})", "", src_dor)

    # ---- le meneau
    if cfg == "2v_meneau":
        if dor["meneau"] is not None:
            ligne("Meneau", "Meneau", s["meneau"], 1, H - 2 * dor["meneau"], f"{_l(H)} − 2 × {_l(dor['meneau'])}",
                  "droite, assemblé mécaniquement", src_dor, "④ du dormant", image_de(s["meneau"], "Meneau"))
        rm = dor["renfort_meneau"].get(s["assemblage"])
        if rm is not None:
            ligne("Renfort", "Renfort de meneau", g.meneaux_renfort.get(s["meneau"], {}).get("renfort", "—"), 1, H - 2 * rm,
                  f"{_l(H)} − 2 × {_l(rm)}", "droite", src_dor, ASSEMBLAGES[s["assemblage"]].lower())

    # ---- les ouvrants
    n_v = deb.get("vantaux") or 0
    if n_v and deb.get("deo"):
        ouv = g.debit.ouvrants[s["ouvrant"]]
        src_o = _src_ligne(ouv)
        wl, wh = deb["deo"]
        actifs = 1 if cfg in CFG_BATTEMENT else n_v      # le second vantail d'un battement est un ouvrant réduit
        if wh is None:
            notes.append("Ouvrant sur seuil : aucune cote de débit pour sa hauteur ; les ouvrants ne sont pas débités.")
        else:
            ligne("Ouvrant", "Traverses", s["ouvrant"], 2 * actifs, wl, _l(wl), "45° soudé", src_o, "DEO ; cote finie", image_de(s["ouvrant"], "Ouvrant"))
            ligne("Ouvrant", "Montants", s["ouvrant"], 2 * actifs, wh, _l(wh), "45° soudé", src_o, "DEO ; cote finie")
            rf = ctl.get("renfort")
            val = (rf or {}).get("valeurs") or {}
            ro = ouv["renfort"]
            ref_r = " / ".join(val.get("renforts") or []) or "—"
            if rf is None:
                notes.append("Renfort d'ouvrant non déterminé : l'abaque de cet ouvrant n'est pas exploitable.")
            elif val.get("pieces") is not None and ro is not None:
                trav, mont = val["pieces"]
                if trav:
                    ligne("Renfort", "Renfort d'ouvrant, traverses", ref_r, trav * actifs, wl - 2 * ro, f"{_l(wl)} − 2 × {_l(ro)}", "droite",
                          rf["sources"][0], f"repère {val['repere']}", image_de(s["ouvrant"], "Renfort"))
                if mont:
                    ligne("Renfort", "Renfort d'ouvrant, montants", ref_r, mont * actifs, wh - 2 * ro, f"{_l(wh)} − 2 × {_l(ro)}", "droite",
                          rf["sources"][0], f"repère {val['repere']}")
                if not trav and not mont:
                    notes.append(f"Repère {val['repere']} de l'abaque : ouvrant sans renfort.")
            elif val.get("repere"):
                notes.append(f"Repère {val['repere']} : {val.get('legende') or 'légende non lue'}. Pièces renforcées à lire sur la légende "
                             f"(traverses {_l(wl - 2 * ro)} mm, montants {_l(wh - 2 * ro)} mm, soit 2 × {_l(ro)} mm retranchés).")
            vl, vh = deb["vitrage"]
            ligne("Vitrage", "Vitrage d'ouvrant", s["vitrage"], actifs, None, f"{_l(vl)} × {_l(vh)} (DEO − 2 × {_l(ouv['vitrage'])})", "", src_o)
            parc = (ctl.get("parclose") or {}).get("valeurs", {}).get("parcloses") or []
            ref_p = " / ".join(parc[0]["refs"]) if parc else "à choisir"
            img_p = parc[0]["image"] if parc else None
            ligne("Parclose", "Parclose d'ouvrant, horizontale", ref_p, 2 * actifs, None, "", "45°", src_o,
                  "longueur = dimension intérieure X (directives de vitrage), que le wiki ne relie à aucune cote de débit", img_p)
            ligne("Parclose", "Parclose d'ouvrant, verticale", ref_p, 2 * actifs, None, "", "45°", src_o,
                  "longueur = dimension intérieure X" + ("; une surlongueur est admise en 2023 (directives de vitrage)" if g.id == "2023" else ""), img_p)
            if cfg in CFG_BATTEMENT:
                b = g.debit.battements.get(s["battement"])
                bt = deb.get("battement") or {}
                if b is None:
                    notes.append(f"Le battement {s['battement']} n'a pas de formule de débit dans « {GENERATIONS[g.id]} ».")
                else:
                    src_b = _src(b["page"], b["section"], b["localisation"])
                    if bt.get("exterieur") is not None:
                        ligne("Battement", "Battement extérieur", s["battement"], 1, bt["exterieur"], f"{_formule_lue(b['exterieur'])} avec DEO = {_l(wh)}",
                              "droite", src_b, HYPOTHESE_BATTEMENT, image_de(s["battement"], "Battement"))
                    if bt.get("interieur") is not None:
                        ligne("Battement", "Battement intérieur", " / ".join(b["interieurs"]) or "—", 1, bt["interieur"],
                              f"{_formule_lue(b['interieur'])} avec DEO = {_l(wh)}", "droite", src_b,
                              "CTR-79 : l'autre document écrit DEO − " + ("12" if g.id == "2008" else "6"))
                notes.append("Le second vantail (ouvrant réduit apparié au battement) n'est pas débité : ses cotes ne sont pas dans le wiki.")

    # ---- la nomenclature : regroupée par référence et par pièce
    ordre = list(OrderedDict.fromkeys(l["famille"] for l in lignes))
    lignes.sort(key=lambda l: ordre.index(l["famille"]))
    for i, l in enumerate(lignes, 1):
        l["rep"] = i
    nomen: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()
    for l in lignes:
        sans_ref = l["ref"] in ("—", "", None)
        k = (l["piece"] if sans_ref else l["ref"], l["famille"])
        n = nomen.setdefault(k, {"ref": "—" if sans_ref else l["ref"], "designation": l["piece"], "famille": l["famille"], "qte": 0,
                                 "ml": 0.0, "image": l["image"]})
        if l["piece"] not in n["designation"].split(" + "):
            n["designation"] += " + " + l["piece"]
        n["qte"] += l["qte"]
        if l["longueur"] is not None:
            n["ml"] += l["qte"] * l["longueur"] / 1000
    nomenclature = [dict(v, ml=round(v["ml"], 2) if v["ml"] else None) for v in nomen.values()]

    # ---- les accessoires des profilés employés (les quantités ne sont pas données)
    employes = [s["dormant"]] + ([s["ouvrant"]] if n_v else []) + ([s["meneau"]] if cfg == "2v_meneau" else []) \
        + ([s["battement"]] if cfg in CFG_BATTEMENT else [])
    deja = {l["ref"] for l in lignes}
    accessoires = []
    for p in employes:
        for a in d.accessoires.get(p, []):
            if a["designation"].startswith("Renfort") or a["ref"] in deja or a["designation"].lower().startswith(("capot", "demi capot")):
                continue
            accessoires.append(dict(a, profil=p))
    notes.append("Les barres soudées (dormant, ouvrant) sont en cote finie : "
                 + ("le manuel de 2023 ne donne pas de surlongueur de soudure." if g.id == "2023" else
                    "le classeur de 2008 donne une limitation du cordon de soudure, pas une surlongueur de coupe."))
    notes.append("Parcloses : longueur = dimension intérieure X, non reliée aux cotes de débit par le wiki. Aucun débit de tapée, "
                 "d'appui ni de renfort de battement.")
    brouillons = sorted(set(ver["brouillons"]) | ({PAGE_ACCESSOIRES} if snap.pages.get(PAGE_ACCESSOIRES)
                                                  and snap.pages[PAGE_ACCESSOIRES].status == "draft" else set()))
    return wiki_famille.relocaliser(snap, {
        "verdict": ver["verdict"], "libelle": ver["libelle"], "lignes": lignes, "nomenclature": nomenclature,
        "accessoires": accessoires, "notes": notes, "debit": ver["debit"], "brouillons": brouillons})
