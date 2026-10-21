"""Le calculateur du SOLEAL PY (porte d'entrée TECHNAL de la gamme LUMINE), 21/10/2026.

Deux sorties, comme pour les autres systèmes : la **faisabilité** (``verifier``) et la **fiche de débit**
(``fiche_debit``). Toutes les valeurs sont lues dans les tableaux du wiki à chaque rechargement de l'instantané ;
seuls des chemins de pages, des titres de section, des en-têtes de colonnes et des noms de typologie sont écrits ici.

Une porte n'a ni DTA ni courbe d'utilisation au vent dans le wiki. Sa faisabilité tient donc à ce que la page des
dimensions maximales et celles des organes de rotation donnent :

- **les dimensions maximales du vantail** (hauteur Hv, largeur Lv ou Lv1 + Lv2, poids par vantail) selon la
  typologie (simple action, renforcée, va-et-vient, tube APD ; 1 ou 2 vantaux) et l'organe de rotation ;
- **le poids maxi du vantail** que l'organe de rotation porte : l'abaque des paumelles (2 lames 130 kg, 3 lames
  150 kg, porte renforcée ou non), la force du pivot frein de sol TBN1370 selon H × L ;
- **le nombre de paumelles** (cas A à D, selon la hauteur du vantail et le ferme-porte) ;
- **le vitrage** : la parclose et le joint intérieur de la planche donnent une épaisseur de remplissage exacte.

La liste de coupe lit une page d'application par type de porte et de seuil : 22 planches du catalogue de conception
4944.006 (09/2018) et 6 de la F.I.T N° 15 (12/2012), la génération antérieure du seuil PMR PY1100. Les deux
générations ne sont jamais fusionnées : ce sont des applications distinctes, la F.I.T est étiquetée et son écart avec
le catalogue de 2018 est renvoyé aux contradictions du registre.

Une application n'est calculée que si **toutes** les cotes de ses formules sont définies par sa page : H et L, et
celles que la page définit en toutes lettres (Hv1, Hv2, L1, L2 : cotes lues sur les coupes, saisies). H1 et H2 de
l'ensemble menuisé, ou L2 d'une planche qui l'écrit pour L/2 (INC-684), ne sont définies nulle part : l'application
reste « sur étude » avec la raison.

Ce que le wiki ne permet pas, et que ce module n'invente pas :

- le **poids du vantail** : ni poids des profilés ni des panneaux opaques. Le verre seul (densité transposée de la
  page des abaques du système 70) est une borne basse : il peut dépasser une limite, jamais la prouver respectée ;
- la **hauteur et la largeur du vantail** : le wiki ne les rattache pas à H et L hors tout. À défaut de saisie, H et L
  (L/2 par vantail de deux vantaux égaux) en sont des majorants : sous la limite, le vantail l'est a fortiori ; au-delà,
  « sur étude », jamais « hors domaine ». Saisis, ils décident ;
- le nombre de paumelles à 2200 mm exactement (INC-698), les poids de l'abaque entre deux graduations (jamais
  d'interpolation : la plus restrictive des graduations voisines), les quantités écrites « *** » ou « 3/ml ».
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

from app.services import wiki_famille
from app.services.faisabilite import (DonneeIntrouvable, ETUDE, HORS, INFO, OK, Table, _controle, _corps, _fmt, _nombre,
                                      _source, _texte_section, _une, tables)
from app.services.faisabilite_ga import (Application, Ligne, anomalies_citees, classer_table, construire_fiche,
                                         densite_du_verre, images_par_reference, valeur_ou_none)
from app.services.faisabilite_systemes import coercer, synthese
from app.services.wiki_service import WikiSnapshot
from app.services.wiki_tables import tables_toutes

ID = "soleal-py"
LIBELLE = "SOLEAL PY (porte LUMINE)"
GAMMES = ["LUMINE"]
PORTEE = ("Porte simple action, renforcée, va-et-vient et tube APD, 1 ou 2 vantaux : dimensions maximales du vantail, "
          "poids maxi par organe de rotation (abaques des paumelles, force du pivot), nombre de paumelles, parclose "
          "et joint de vitrage, et liste de coupe de 28 applications (catalogue de conception 2018 et F.I.T 2012). "
          "Aucun DTA ni courbe de vent pour la porte ; le poids du vantail ne peut être borné que par le verre ; une "
          "application dont une cote n'est pas définie (ensemble menuisé à meneau, rejet d'eau de la porte 2 vantaux "
          "sans FPI) reste « sur étude ».")

PAGE_LIMITES = "/portes/soleal-py-dimensions-maximales.md"
PAGE_REPARTITION = "/quincaillerie/soleal-py-paumelles-en-feuillure-et-repartition.md"
PAGE_ABAQUES = {"2": "/quincaillerie/soleal-py-paumelles-abaques-2-lames-130-kg.md",
                "3": "/quincaillerie/soleal-py-paumelles-abaques-3-lames-150-kg.md"}
PAGE_PIVOTS = "/quincaillerie/soleal-py-pivots-freins-de-sol-et-crapaudines.md"
PAGE_PARCLOSES = "/profiles/soleal-py-parcloses-et-prises-de-volume.md"
PAGE_VERRE = "/profiles/systeme-70-abaques-dimensionnels.md"

P_SA1 = "/portes/soleal-py-cotes-de-debit.md"
P_SA2 = "/portes/soleal-py-simple-action-2-vantaux-seuil-pmr-batiment.md"
P_ENSEMBLE = "/portes/soleal-py-simple-action-ensemble-menuise-et-fixe-lateral.md"
P_RPT = "/portes/soleal-py-simple-action-ouvrant-peripherique-seuil-pmr-rpt.md"
P_RC3_EXT = "/portes/soleal-py-simple-action-retard-a-l-effraction-ouverture-exterieure.md"
P_RC3_PERI = "/portes/soleal-py-simple-action-retard-a-l-effraction-ouvrant-peripherique.md"
P_PLAT = "/portes/soleal-py-simple-action-seuil-plat-1-et-2-vantaux.md"
P_RESID = "/portes/soleal-py-simple-action-seuil-residentiel.md"
P_VV = "/portes/soleal-py-va-et-vient-ouvrant-peripherique-seuil-plat.md"
P_TUBE_SA = "/portes/soleal-py-porte-tube-apd-simple-action.md"
P_TUBE_DA = "/portes/soleal-py-porte-tube-apd-double-action.md"
P_PY1100_1 = "/portes/technal-seuil-py1100-applications-1-vantail.md"
P_PY1100_2 = "/portes/technal-seuil-py1100-applications-2-vantaux.md"
P_PY1100_F = "/portes/technal-seuil-py1100-applications-fixe-lateral.md"

CONCEPTION = "catalogue de conception 4944.006 (09/2018)"
FIT = "F.I.T N° 15 (12/2012)"
# Les noms de typologie du tableau des dimensions maximales.
FAMILLES = {"sa": "Porte simple action", "vv": "Porte va-et-vient", "tube": "Porte tube APD"}
FAMILLE_LIBELLE = {"sa": "simple action", "vv": "va-et-vient", "tube": "tube APD"}


@dataclass(frozen=True)
class Def:
    """Une application : une section d'une page de nomenclature (``section`` ``None`` : la page entière)."""
    id: str
    libelle: str
    page: str
    section: Optional[str]
    famille: str            # « sa », « vv » ou « tube »
    vantaux: int
    fpi: bool = False       # ferme-porte intégré dans le titre de la planche
    document: str = CONCEPTION


def _d(id_: str, libelle: str, page: str, section: Optional[str], famille: str, vantaux: int, fpi: bool = False,
       document: str = CONCEPTION) -> Tuple[str, Def]:
    return id_, Def(id_, libelle, page, section, famille, vantaux, fpi, document)


APPLICATIONS: Dict[str, Def] = dict([
    _d("sa1_pmr", "SA 1 vantail, ouverture extérieure, plinthe et seuil PMR bâtiment", P_SA1,
       "Porte 1 vantail ouverture extérieure, plinthe et seuil PMR bâtiment", "sa", 1),
    _d("sa1_pmr_fpi", "SA 1 vantail, ouverture extérieure, plinthe et seuil PMR bâtiment, avec FPI", P_SA1,
       "Porte 1 vantail ouverture extérieure avec FPI, plinthe et seuil PMR bâtiment", "sa", 1, True),
    _d("sa2_pmr", "SA 2 vantaux, ouverture extérieure, plinthe et seuil PMR bâtiment", P_SA2,
       "Porte 2 vantaux ouverture extérieure, plinthe et seuil PMR bâtiment", "sa", 2),
    _d("sa2_pmr_fpi", "SA 2 vantaux, ouverture extérieure, plinthe et seuil PMR bâtiment, avec FPI", P_SA2,
       "Porte 2 vantaux ouverture extérieure avec FPI, plinthe et seuil PMR bâtiment", "sa", 2, True),
    _d("sa1_plat", "SA 1 vantail, ouverture extérieure, plinthe et seuil plat T525053", P_PLAT,
       "Porte 1 vantail ouverture extérieure, plinthe et seuil", "sa", 1),
    _d("sa2_plat", "SA 2 vantaux, ouverture extérieure, plinthe et seuil plat T525053", P_PLAT,
       "Porte 2 vantaux ouverture extérieure, plinthe et seuil", "sa", 2),
    _d("sa1_resid", "SA 1 vantail, ouverture intérieure, seuil résidentiel", P_RESID, None, "sa", 1),
    _d("sa1_rpt", "SA 1 vantail, ouverture intérieure, ouvrant périphérique, seuil PMR RPT", P_RPT,
       "Porte 1 vantail ouverture intérieure, ouvrant périphérique, seuil PMR RPT", "sa", 1),
    _d("sa2_rpt", "SA 2 vantaux, ouverture intérieure, ouvrant périphérique, seuil PMR RPT", P_RPT,
       "Porte 2 vantaux ouverture intérieure, ouvrant périphérique, seuil PMR RPT", "sa", 2),
    _d("sa1_rc3_ext", "SA 1 vantail, ouverture extérieure, plinthe et seuil, retard à l'effraction", P_RC3_EXT,
       "Porte 1 vantail ouverture extérieure, plinthe et seuil, retard à l'effraction", "sa", 1),
    _d("sa2_rc3_ext", "SA 2 vantaux, ouverture extérieure, plinthe et seuil, retard à l'effraction", P_RC3_EXT,
       "Porte 2 vantaux ouverture extérieure, plinthe et seuil, retard à l'effraction", "sa", 2),
    _d("sa1_rc3_peri", "SA 1 vantail, ouverture intérieure, ouvrant périphérique, seuil PMR RPT, retard à l'effraction",
       P_RC3_PERI, "Porte 1 vantail ouverture intérieure, ouvrant périphérique et seuil PMR RPT", "sa", 1),
    _d("sa2_rc3_peri", "SA 2 vantaux, ouverture intérieure, ouvrant périphérique, seuil PMR RPT, retard à l'effraction",
       P_RC3_PERI, "Porte 2 vantaux ouverture intérieure, ouvrant périphérique et seuil PMR RPT", "sa", 2),
    _d("sa1_ensemble", "SA 1 vantail dans un ensemble menuisé à meneau (fixe latéral et imposte)", P_ENSEMBLE,
       "Porte 1 vantail ouverture extérieure, plinthe et seuil bâtiment dans ensemble menuisé meneau", "sa", 1),
    _d("sa1_fixe", "SA 1 vantail, plinthe, seuil PMR bâtiment et fixe latéral", P_ENSEMBLE,
       "Porte 1 vantail ouverture extérieure, plinthe, seuil PMR bâtiment et fixe latéral", "sa", 1),
    _d("vv1", "Va-et-vient 1 vantail, ouvrant périphérique, seuil plat et FPI", P_VV,
       "Porte va-et-vient 1 vantail, ouvrant périphérique, seuil plat et FPI", "vv", 1, True),
    _d("vv2", "Va-et-vient 2 vantaux, ouvrant périphérique, seuil plat et FPI", P_VV,
       "Porte va-et-vient 2 vantaux, ouvrant périphérique, seuil plat et FPI", "vv", 2, True),
    _d("tube1_sa", "Tube APD simple action 1 vantail, ouverture extérieure, APD niveau 2 avec FPI", P_TUBE_SA,
       "Porte tube 1 vantail ouverture extérieure, APD niveau 2 avec FPI", "tube", 1, True),
    _d("tube2_sa", "Tube APD simple action 2 vantaux, ouverture extérieure, APD niveau 2 avec FPI", P_TUBE_SA,
       "Porte tube 2 vantaux ouverture extérieure, APD niveau 2 avec FPI", "tube", 2, True),
    _d("tube1_da", "Tube APD double action 1 vantail, APD niveau 2 avec FPI", P_TUBE_DA,
       "Porte tube double action 1 vantail, APD niveau 2 avec FPI", "tube", 1, True),
    _d("tube2_da", "Tube APD double action 2 vantaux, APD niveau 2 avec FPI", P_TUBE_DA,
       "Porte tube double action 2 vantaux, APD niveau 2 avec FPI", "tube", 2, True),
    _d("tube1_da_fixe", "Tube APD double action 1 vantail, APD niveau 1 avec FPI et fixe latéral", P_TUBE_DA,
       "Porte tube double action 1 vantail, APD niveau 1 avec FPI et fixe latéral", "tube", 1, True),
    _d("py1100_sa1_ext", "Seuil PMR PY1100, 1 vantail, ouverture extérieure (F.I.T 2012)", P_PY1100_1,
       "Porte à 1 vantail, ouverture extérieure", "sa", 1, document=FIT),
    _d("py1100_sa1_int", "Seuil PMR PY1100, 1 vantail, ouverture intérieure (F.I.T 2012)", P_PY1100_1,
       "Porte à 1 vantail, ouverture intérieure", "sa", 1, document=FIT),
    _d("py1100_sa2_ext", "Seuil PMR PY1100, 2 vantaux, ouverture extérieure (F.I.T 2012)", P_PY1100_2,
       "Porte à 2 vantaux, ouverture extérieure", "sa", 2, document=FIT),
    _d("py1100_sa2_int", "Seuil PMR PY1100, 2 vantaux, ouverture intérieure (F.I.T 2012)", P_PY1100_2,
       "Porte à 2 vantaux, ouverture intérieure", "sa", 2, document=FIT),
    _d("py1100_fixe_ext", "Seuil PMR PY1100, 1 vantail et fixe latéral, ouverture extérieure (F.I.T 2012)", P_PY1100_F,
       "Porte à 1 vantail et fixe latéral, ouverture extérieure", "sa", 1, document=FIT),
    _d("py1100_fixe_int", "Seuil PMR PY1100, 1 vantail et fixe latéral, ouverture intérieure (F.I.T 2012)", P_PY1100_F,
       "Porte à 1 vantail et fixe latéral, ouverture intérieure", "sa", 1, document=FIT),
])
AUTRE = "autre"
LIBELLE_AUTRE = "Autre application (porte d'habitat, autre seuil, autre ouvrant : aucune planche dans le wiki)"
PAGES_APPLICATIONS = tuple(dict.fromkeys(d.page for d in APPLICATIONS.values()))
PAGES = (PAGE_LIMITES, PAGE_REPARTITION, *PAGE_ABAQUES.values(), PAGE_PIVOTS, PAGE_PARCLOSES, PAGE_VERRE,
         *PAGES_APPLICATIONS)

# Les cotes que la page d'une application définit en toutes lettres. Une cote qu'aucune phrase ne définit (H1 et H2 :
# « se partagent la hauteur », sans dire laquelle est laquelle) n'est pas dans cette table : l'application reste sur étude.
# Chaque motif est la phrase du wiki ; si la page est réécrite, la cote cesse d'être définie, ce qui est le sens sûr.
DEFINITIONS: Dict[str, Tuple[str, str]] = {
    "Hv1": (r"Hv1 (?:est la hauteur cotée )?du côté du seuil", "Hv1 : hauteur cotée du côté du seuil, sous la traverse T215205"),
    "Hv2": (r"Hv2 (?:celle )?du côté du haut", "Hv2 : hauteur cotée du côté du haut de la porte, au-dessus de la traverse T215205"),
    "L1": (r"L1(?: est la largeur | celle |, | )du côté (?:du vantail|de l'ouvrant)", "L1 : largeur du côté du vantail"),
    "L2": (r"L2(?: est la largeur | la largeur |, | )(?:du côté du fixe|du fixe)", "L2 : largeur du côté du fixe latéral"),
}
CHAMP_DE = {"Hv1": "hv1_mm", "Hv2": "hv2_mm", "L1": "l1_mm", "L2": "l2_mm"}
DEFAUT_COTE = {"Hv1": 600, "Hv2": 1500, "L1": 900, "L2": 400}

# Les anomalies que le contrat nomme pour une application, quand la page ne les cite pas elle-même.
ANOMALIES_EN_PLUS: Dict[str, List[str]] = {
    "sa1_plat": ["CTR-258"],
    "py1100_sa1_ext": ["CTR-244"], "py1100_sa1_int": ["CTR-243", "CTR-244"],
    "py1100_sa2_ext": ["CTR-128", "CTR-129"], "py1100_sa2_int": ["CTR-128", "CTR-129"],
    "vv2": ["INC-690"], "tube2_da": ["INC-690"],
}
# Ce que la planche montre d'elle-même : une quantité par mètre linéaire sans longueur, une quantité « *** ».
# Les planches de 2018 que le registre met en face d'une planche de la F.I.T (CTR-243 ; CTR-128 et CTR-129).
EQUIVALENTS: Dict[str, List[str]] = {"py1100_sa1_int": ["sa1_rpt"], "py1100_sa2_ext": ["sa2_pmr", "sa2_rpt"],
                                     "py1100_sa2_int": ["sa2_pmr", "sa2_rpt"]}
ANOMALIES_DES_QUANTITES = ((re.compile(r"/ml?\b"), "VER-574"), (re.compile(r"^\*+$"), "VER-572"))

HYPOTHESE_VANTAIL = ("hypothèse : H et L hors tout tiennent lieu de hauteur et de largeur du vantail (L/2 pour un vantail "
                     "de deux vantaux égaux) ; le vantail est plus petit, donc une limite respectée l'est a fortiori")
HYPOTHESE_VERRE = ("hypothèse : poids du vantail = verre seul, à la densité transposée de la page des abaques du système "
                   "70 (aucune page de la porte ne donne la masse du verre, des profilés ni des panneaux)")

# La vis, le clip ou le profilé écrit « 3/ml » ne dit pas de quelle longueur : VER-574.


class DonneePorte(DonneeIntrouvable):
    """Une donnée de la porte absente ou illisible."""


# ---- les données lues dans le wiki ------------------------------------------------------------


@dataclass
class Limite:
    typologie: str
    organe: str
    refs: List[str]
    hv: float
    largeur: float
    largeur_texte: str
    poids: float


@dataclass
class Abaque:
    grille: Dict[int, Dict[int, Optional[float]]]      # hauteur du vantail -> largeur -> poids maxi (kg)
    section: str
    localisation: str


@dataclass
class Cas:
    cas: str
    ferme_porte: bool
    sens: str               # « < » ou « > »
    seuil: float
    nombre: int


@dataclass
class Pivot:
    force: int
    h: float
    l: float
    moment_2: str
    moment_90: str


@dataclass
class Parclose:
    parclose: str
    hauteur: float
    joint: str
    epaisseur_joint: float
    remplissage: float
    zone: Tuple[float, float]


@dataclass
class AppPY:
    defn: Def
    app: Application
    saisies: List[str]                    # cotes définies par la page et utilisées par ses formules
    indefinies: List[str]                 # cotes utilisées, définies nulle part
    images: Dict[str, str]
    anomalies: List[str]


@dataclass
class Donnees:
    statuts: Dict[str, str]
    limites: List[Limite]
    loc_limites: str
    abaques: Dict[Tuple[str, bool], Abaque]            # (nombre de lames, porte renforcée)
    precision_kg: float
    lames: Dict[str, str]                              # référence de paumelle -> nombre de lames
    cas: List[Cas]
    loc_cas: str
    pivots: List[Pivot]
    loc_pivots: str
    parcloses: List[Parclose]
    loc_parcloses: str
    applications: Dict[str, AppPY]
    densite_verre: Optional[float]
    images: Dict[str, str]


_CACHE: Dict[str, Any] = {"cle": None, "donnees": None}


def donnees(snap: WikiSnapshot) -> Donnees:
    cle = (id(snap), snap.loaded_at)
    if _CACHE["cle"] != cle:
        _CACHE["donnees"] = _charge(snap)
        _CACHE["cle"] = cle
    return _CACHE["donnees"]


def _propre(texte: str) -> str:
    return texte.replace(" ", " ").replace(" ", " ").replace("’", "'")


def _cle(ref: str) -> str:
    """« T591005 » et « 591005 » (la F.I.T écrit les références sans T, VER-152), « TAS0017 » et « AS0017 » : une même clé."""
    return ref[1:] if ref.startswith("T") else ref


def normaliser_formule(formule: str) -> str:
    """« 2(L/2-74) » : un nombre collé à une parenthèse est un produit, que la grammaire partagée ne lit pas."""
    return re.sub(r"(\d)\s*\((?![A-Za-z'\"]+\)\s*$)", r"\1*(", formule.strip())


# -- les tableaux d'une application


def _genre(t: Table) -> Optional[str]:
    if t.entete and t.entete[0] == "Remplissage":
        return "remplissage"
    g = classer_table(t)
    # La F.I.T intitule « Quantité » la longueur d'un joint : seule la désignation le distingue d'un accessoire.
    if g == "accessoires" and t.lignes and all(len(r) > 1 and r[1].lower().startswith("joint") for r in t.lignes):
        return "joints"
    return g


def _lignes(t: Table, genre: str) -> List[Ligne]:
    h = t.entete
    out: List[Ligne] = []
    for r, b in zip(t.lignes, t.brut):
        if not r or not r[0]:
            continue
        if genre == "vis":
            out.append(Ligne(r[0], "", normaliser_formule(r[-1]), ""))
        elif genre == "accessoires":
            out.append(Ligne(r[0], r[1], normaliser_formule(r[2]), ""))
        elif genre == "joints":
            out.append(Ligne(r[0], r[1], "1", normaliser_formule(r[-1])))
        else:
            # Une cellule de coupe « \|\| » se scinde en cellules : le débit est la dernière cellule du catalogue de 2018 ;
            # la F.I.T met le débit avant la colonne des dessins.
            idb = next(i for i, c in enumerate(h) if c.startswith("Débit"))
            debit = r[-1] if idb == len(h) - 1 else r[idb]
            if "Désignation" in h:
                designation, coupe = r[1], "|".join(b[3:-1]).strip()
            else:
                designation, coupe = "", r[1]
            out.append(Ligne(r[0], designation, normaliser_formule(r[2]), normaliser_formule(debit),
                             "" if coupe == "-" else coupe))
    return out


def _identifiants(formule: str) -> List[str]:
    f = re.sub(r"\([A-Za-z][A-Za-z0-9'\"]*\)$", "", formule)
    return re.findall(r"[A-Za-z][A-Za-z0-9]*", f)


REMPLISSAGE_PROSE = re.compile(r"[Rr]emplissage de cette porte a pour cotes (H[\w/\-.]*\d) et (L[\w/\-.]*\d)")


def _lire_application(snap: WikiSnapshot, defn: Def) -> AppPY:
    corps = _propre(_corps(snap, defn.page))
    ts = [t for t in tables_toutes(corps) if defn.section is None or defn.section in t.chemin[-2:]]
    trouves: Dict[str, Table] = {}
    remplissages: List[Dict[str, str]] = []
    for t in ts:
        genre = _genre(t)
        if genre == "remplissage":
            remplissages += [{"libelle": r[0], "hauteur": r[1], "largeur": r[2]} for r in t.lignes if len(r) > 2]
        elif genre and genre not in trouves:
            trouves[genre] = t
    nom = defn.section or "page entière"
    for genre in ("profils", "joints", "accessoires"):      # la F.I.T n'a pas de tableau de vis : ses vis sont des accessoires
        if genre not in trouves:
            raise DonneePorte(f"{defn.page} : section « {nom} », tableau des {genre} introuvable")
    texte = _texte_section(corps, defn.section) if defn.section else corps
    if not remplissages:
        remplissages = [{"libelle": "", "hauteur": m.group(1), "largeur": m.group(2)}
                        for m in REMPLISSAGE_PROSE.finditer(texte)]
    if not remplissages:
        raise DonneePorte(f"{defn.page} : section « {nom} », cotes du remplissage introuvables")
    remplissages = [dict(r, hauteur=normaliser_formule(r["hauteur"]), largeur=normaliser_formule(r["largeur"]))
                    for r in remplissages]
    profils = _lignes(trouves["profils"], "profils")
    app = Application(defn.id, defn.libelle, defn.page, defn.section or "", trouves["profils"].localisation,
                      profils, _lignes(trouves["accessoires"], "accessoires"), _lignes(trouves["joints"], "joints"),
                      _lignes(trouves["vis"], "vis") if "vis" in trouves else [], remplissages, [])
    # les cotes : celles des débits, des joints et des remplissages
    utilisees = sorted(({v for l in profils + app.joints for v in _identifiants(l.debit)}
                        | {v for r in remplissages for v in _identifiants(r["hauteur"]) + _identifiants(r["largeur"])})
                       - {"H", "L"})
    definies = [v for v in utilisees if v in DEFINITIONS and re.search(DEFINITIONS[v][0], corps)]
    indefinies = [v for v in utilisees if v not in definies]
    avant_titre = re.split(r"(?m)^#", corps, maxsplit=1)[0]
    quantites = [l.qte for l in app.accessoires + app.vis + profils]
    des_quantites = [ano for motif, ano in ANOMALIES_DES_QUANTITES if any(motif.search(q.strip()) for q in quantites)]
    anomalies = sorted(set(anomalies_citees(avant_titre + _texte_section(corps, "Lire la nomenclature") + texte)
                           + ANOMALIES_EN_PLUS.get(defn.id, []) + des_quantites), key=lambda a: (a[:3], int(a[4:])))
    # le dessin du profilé de la F.I.T : dans la dernière colonne de son tableau
    images: Dict[str, str] = {}
    if trouves["profils"].entete and trouves["profils"].entete[-1] == "Coupe" and "Désignation" not in trouves["profils"].entete:
        for r, b in zip(trouves["profils"].lignes, trouves["profils"].brut):
            m = re.search(r"\((/assets/[^)]+)\)", b[-1]) if b else None
            if m and r and r[0]:
                images.setdefault(r[0], m.group(1))
    return AppPY(defn, replace(app, anomalies=anomalies), [] if indefinies else definies, indefinies, images, anomalies)


# -- les autres pages


def _charge(snap: WikiSnapshot) -> Donnees:
    # -- dimensions maximales du vantail
    corps = _corps(snap, PAGE_LIMITES)
    t = _une(tables(corps), "Typologie | Organe de rotation | Référence | Hv maxi (mm) | Largeur maxi (mm) | "
                            "Poids maxi par vantail (kg)", PAGE_LIMITES)
    limites: List[Limite] = []
    for r in t.lignes:
        m = re.match(r"(\d+)\s*\((.*)\)", r[4])
        if not m:
            raise DonneePorte(f"{PAGE_LIMITES} : largeur maxi illisible : « {r[4]} »")
        limites.append(Limite(r[0], r[1], [x.strip() for x in re.split(r"\s+ou\s+", r[2])], _nombre(r[3]),
                              float(m.group(1)), m.group(2), _nombre(r[5])))

    # -- répartition des paumelles et nombre de lames par référence
    tr = tables(_corps(snap, PAGE_REPARTITION))
    t_cas = _une(tr, "Cas | Condition imprimée | Nombre de paumelles", PAGE_REPARTITION)
    cas: List[Cas] = []
    for r in t_cas.lignes:
        m = re.search(r"([<>])\s*(\d+)\s*mm", r[1])
        if not m:
            raise DonneePorte(f"{PAGE_REPARTITION} : condition du cas {r[0]} illisible : « {r[1]} »")
        cas.append(Cas(r[0], "ferme-porte" in r[1].lower(), m.group(1), float(m.group(2)), int(_nombre(r[2]))))
    t_ref = _une(tr, "Référence | Mention imprimée", PAGE_REPARTITION)
    lames = {r[0]: m.group(1) for r in t_ref.lignes if (m := re.search(r"(\d+) lames", r[1]))}

    # -- les abaques de poids : 2 et 3 lames, porte renforcée ou non
    abaques: Dict[Tuple[str, bool], Abaque] = {}
    precision = None
    for nb, page in PAGE_ABAQUES.items():
        corps_ab = _corps(snap, page)
        mp = re.search(r"Précision de lecture : ± (\d+) kg", corps_ab)
        if not mp:
            raise DonneePorte(f"{page} : précision de lecture des abaques (± kg) introuvable")
        precision = max(precision or 0, float(mp.group(1)))
        for tb in tables(corps_ab):
            if tb.entete[:1] != ["Hauteur vantail (mm)"] or not tb.entete[1].startswith("L = "):
                continue
            renforcee = not tb.section.startswith("Porte non renforcée")
            largeurs = [int(re.search(r"\d+", c).group(0)) for c in tb.entete[1:]]
            grille = {int(_nombre(r[0])): {l: _nombre(c) for l, c in zip(largeurs, r[1:])} for r in tb.lignes}
            abaques[(nb, renforcee)] = Abaque(grille, tb.section, tb.localisation)
    for nb in PAGE_ABAQUES:
        if (nb, False) not in abaques or (nb, True) not in abaques:
            raise DonneePorte(f"{PAGE_ABAQUES[nb]} : abaques « porte non renforcée » et « porte renforcée » introuvables")

    # -- pivot frein de sol : force selon H × L
    t_piv = _une(tables(_corps(snap, PAGE_PIVOTS)), "Force du pivot | Dimensions maxi de la porte H × L (mm)", PAGE_PIVOTS)
    pivots = []
    for r in t_piv.lignes:
        m = re.search(r"(\d+)\s*×\s*(\d+)", r[1])
        if not m:
            raise DonneePorte(f"{PAGE_PIVOTS} : dimensions du pivot de force {r[0]} illisibles : « {r[1]} »")
        pivots.append(Pivot(int(_nombre(r[0])), float(m.group(1)), float(m.group(2)), r[2], r[3]))

    # -- la matrice parclose × joint intérieur : l'épaisseur de remplissage A
    parcloses: List[Parclose] = []
    loc_parcloses = ""
    for tb in tables_toutes(_corps(snap, PAGE_PARCLOSES)):
        if tb.entete[:1] != ["Parclose"] or not tb.entete[1].startswith("Hauteur de parclose C") \
                or "pose de face" in tb.section.lower():
            continue
        loc_parcloses = loc_parcloses or tb.localisation
        joints = {}
        for k, c in enumerate(tb.entete):
            mj, mb = re.search(r"\b(T?AS\d{4}|T\d{6})\b", c), re.search(r"B = (\d+) mm", c)
            if "obtenue avec le joint intérieur" in c and mj and mb:
                joints[k] = (mj.group(1), float(mb.group(1)))
        kz = next((k for k, c in enumerate(tb.entete) if c.startswith("Prise de volume recommandée")), None)
        if not joints or kz is None:
            raise DonneePorte(f"{PAGE_PARCLOSES} : colonnes des joints ou zone recommandée introuvables")
        for r in tb.lignes:
            zone = [float(x) for x in re.findall(r"\d+", r[kz])]
            for k, (joint, b) in joints.items():
                a = _nombre(r[k])
                if a is not None:
                    parcloses.append(Parclose(r[0], _nombre(r[1]), joint, b, a, (zone[0], zone[-1]) if zone else (a, a)))
    if not parcloses:
        raise DonneePorte(f"{PAGE_PARCLOSES} : matrice des parcloses à pelle introuvable")

    # -- les applications
    images = images_par_reference(snap, "soleal-py")
    apps = {k: _lire_application(snap, defn) for k, defn in APPLICATIONS.items()}
    return Donnees(
        statuts={p: (snap.pages[p].status or "") for p in PAGES if p in snap.pages},
        limites=limites, loc_limites=t.localisation, abaques=abaques, precision_kg=precision or 0, lames=lames, cas=cas,
        loc_cas=t_cas.localisation, pivots=pivots, loc_pivots=t_piv.localisation, parcloses=parcloses,
        loc_parcloses=loc_parcloses, applications=apps, densite_verre=densite_du_verre(snap), images=images)


# ---- le formulaire ------------------------------------------------------------------------------


def _organes(d: Donnees) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for l in d.limites:
        out.setdefault(l.refs[0], f"{l.organe} {' ou '.join(l.refs)}")
    return out


def _typologie(defn: Def, renforcee: bool) -> str:
    nom = FAMILLES[defn.famille] + (" renforcée" if renforcee and defn.famille == "sa" else "")
    return f"{nom} {defn.vantaux} " + ("vantail" if defn.vantaux == 1 else "vantaux")


def options(snap: WikiSnapshot) -> Dict[str, Any]:
    d = donnees(snap)
    configurations = {k: a.defn.libelle for k, a in d.applications.items()}
    configurations[AUTRE] = LIBELLE_AUTRE
    calculables = list(d.applications)
    sa = [k for k, a in d.applications.items() if a.defn.famille == "sa"]
    organes = {"nomenclature": "Celui de la planche d'application"}
    organes.update(_organes(d))
    epaisseurs = sorted({int(p.remplissage) for p in d.parcloses})
    champs: List[Dict[str, Any]] = [
        {"id": "configuration", "libelle": "Application", "type": "select", "options": configurations,
         "defaut": "sa1_pmr"},
        {"id": "largeur_mm", "libelle": "Largeur hors tout L", "type": "nombre", "min": 400, "max": 6000,
         "defaut": 900, "unite": "mm"},
        {"id": "hauteur_mm", "libelle": "Hauteur hors tout H", "type": "nombre", "min": 1000, "max": 4000,
         "defaut": 2200, "unite": "mm"},
        {"id": "epaisseur_vitrage_mm", "libelle": "Épaisseur du remplissage (vitrage)", "type": "select",
         "options": {str(e): f"{e} mm" for e in epaisseurs}, "defaut": "24", "si": {"configuration": calculables},
         "aide": "L'épaisseur que la parclose et le joint intérieur de la planche reçoivent."},
        {"id": "verre_mm", "libelle": "Verre seul (somme des verres, sans lame)", "type": "nombre", "min": 0, "max": 60,
         "defaut": 8, "unite": "mm", "si": {"configuration": calculables},
         "aide": "4-16-4 : 8 mm de verre. Sert au poids du vantail (verre seul, borne basse)."},
        {"id": "organe", "libelle": "Organe de rotation", "type": "select", "options": organes, "defaut": "nomenclature",
         "si": {"configuration": calculables},
         "aide": "Paumelles, pivot frein de sol ou crapaudine : le tableau des dimensions maximales les distingue."},
        {"id": "renforcee", "libelle": "Porte simple action renforcée", "type": "case", "defaut": False,
         "si": {"configuration": sa},
         "aide": "Profilés à toile de 3 mm : « renforcée » n'est définie par aucune légende (VER-583)."},
        {"id": "ferme_porte", "libelle": "Ferme-porte (hors FPI de la planche)", "type": "case", "defaut": False,
         "si": {"configuration": sa}, "aide": "Le nombre de paumelles passe aux cas C et D."},
        {"id": "hauteur_vantail_mm", "libelle": "Hauteur du vantail Hv (facultatif)", "type": "nombre", "min": 500,
         "max": 4000, "defaut": None, "requis": False, "unite": "mm", "si": {"configuration": calculables},
         "aide": "Vide : H hors tout tient lieu de Hv (majorant). Saisie : le verdict décide."},
        {"id": "largeur_vantail_mm", "libelle": "Largeur d'un vantail Lv (facultatif)", "type": "nombre", "min": 300,
         "max": 3000, "defaut": None, "requis": False, "unite": "mm", "si": {"configuration": calculables},
         "aide": "Vide : L hors tout (L/2 par vantail de deux vantaux égaux). Saisie : le verdict décide."},
    ]
    for v, nom in CHAMP_DE.items():
        ids = [k for k, a in d.applications.items() if v in a.saisies]
        if ids:
            champs.append({"id": nom, "libelle": f"Cote {v}", "type": "nombre", "min": 50, "max": 4000,
                           "defaut": DEFAUT_COTE[v], "unite": "mm", "si": {"configuration": ids},
                           "aide": f"{DEFINITIONS[v][1]} (cote portée sur la coupe de la planche)."})
    pages = list(PAGES)
    return {
        "champs": champs, "configurations": configurations, "pages": pages,
        "brouillons": sorted(p for p, s in d.statuts.items() if s == "draft"),
        "avertissements": [
            "Le wiki ne donne ni DTA ni courbe de vent pour la porte SOLEAL PY : la faisabilité tient aux dimensions "
            "maximales du vantail, au poids maxi des paumelles ou du pivot et au vitrage.",
            "Le wiki ne donne ni le poids des profilés ni celui des panneaux opaques : le poids du vantail est celui du "
            "verre seul ; il peut dépasser une limite, jamais la prouver respectée.",
            "Le wiki ne rattache pas la hauteur et la largeur du vantail à H et L hors tout : sans saisie, H et L en sont "
            "des majorants (L/2 par vantail pour deux vantaux égaux).",
            "Trois origines de nomenclature : le catalogue de conception 4944.006 (09/2018), dont ce calcul lit 22 planches, "
            "le catalogue de fabrication 4899.007 (même date, non lu ici) et la F.I.T N° 15 (12/2012), lue à part et jamais "
            "fusionnée avec les autres.",
            "Les angles de coupe de la planche de 2018 sont des pictogrammes sans légende (VER-571) ; la liste de coupe "
            "donne des longueurs de coupe droites.",
        ],
    }


# ---- la faisabilité ------------------------------------------------------------------------------


def _variables(s: Dict[str, Any], a: AppPY) -> Dict[str, float]:
    v = {"H": s["hauteur_mm"], "L": s["largeur_mm"]}
    for nom in a.saisies:
        if s.get(CHAMP_DE[nom]) is not None:
            v[nom] = s[CHAMP_DE[nom]]
    return v


@dataclass
class Vantail:
    hv: float
    lv: float               # largeur d'un vantail
    somme: float            # largeur de tous les vantaux (Lv1 + Lv2)
    ferme_h: bool           # Hv saisie : elle décide ; sinon H hors tout, un majorant
    ferme_l: bool
    texte: str

    @property
    def ferme(self) -> bool:
        return self.ferme_h and self.ferme_l


def _vantail(s: Dict[str, Any], defn: Def) -> Vantail:
    n = defn.vantaux
    hv_saisi, lv_saisi = s.get("hauteur_vantail_mm"), s.get("largeur_vantail_mm")
    hv = hv_saisi if hv_saisi is not None else s["hauteur_mm"]
    lv = lv_saisi if lv_saisi is not None else s["largeur_mm"] / n
    somme = lv * n if lv_saisi is not None else s["largeur_mm"]
    manque = [nom for nom, v in (("Hv", hv_saisi), ("Lv", lv_saisi)) if v is None]
    texte = (f"Hv = {_fmt(hv)} mm, Lv = {_fmt(lv, 1 if lv % 1 else 0)} mm saisis" if not manque else
             f"{' et '.join(manque)} non saisi{'s' if len(manque) > 1 else ''} : {HYPOTHESE_VANTAIL}")
    if n > 1 and lv_saisi is not None:
        texte += " (deux vantaux égaux : Lv1 + Lv2 = 2 Lv)"
    return Vantail(hv, lv, somme, hv_saisi is not None, lv_saisi is not None, texte)


def _pire(*statuts: str) -> str:
    return max(statuts, key=[OK, INFO, ETUDE, HORS].index)


def _depasse(v: Vantail) -> str:
    """Un majorant qui dépasse la limite ne prouve rien : « sur étude » ; une cote saisie qui la dépasse : « hors »."""
    return HORS if v.ferme else ETUDE


def _organe(d: Donnees, a: AppPY, s: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """(référence de l'organe de rotation, d'où elle vient)."""
    if s["organe"] != "nomenclature":
        return s["organe"], "choisi"
    connues = {r for l in d.limites for r in l.refs}
    for l in a.app.accessoires:
        if l.ref in connues:
            return l.ref, "celui de la planche d'application"
    return None, "la planche d'application ne porte aucun organe de rotation du tableau des dimensions maximales"


def _ligne_limite(d: Donnees, a: AppPY, s: Dict[str, Any], ref: Optional[str]) -> Optional[Limite]:
    typo = _typologie(a.defn, bool(s.get("renforcee")))
    return next((l for l in d.limites if l.typologie == typo and ref in l.refs), None)


def _src_limites(d: Donnees) -> List[Dict[str, str]]:
    return [_source(PAGE_LIMITES, "Dimensions maximales du vantail", d.loc_limites)]


def _vitrages(a: AppPY) -> str:
    return " et ".join(f"{r['hauteur']} × {r['largeur']}" for r in a.app.remplissages if "fixe" not in r["libelle"].lower())


def _controle_document(d: Donnees, a: AppPY) -> List[Dict[str, Any]]:
    """La génération de la fiche, quand ce n'est pas le catalogue courant : jamais fusionnée avec l'autre. L'écart avec la
    planche de 2018 que le registre lui oppose est affiché, jamais arbitré."""
    if a.defn.document == CONCEPTION:
        return []
    ecarts = [f"remplissage {_vitrages(a)} dans la F.I.T contre {_vitrages(d.applications[e])} dans le catalogue de 2018 "
              f"({d.applications[e].defn.libelle})" for e in EQUIVALENTS.get(a.defn.id, [])]
    ctr = ", ".join(x for x in ANOMALIES_EN_PLUS.get(a.defn.id, []) if x.startswith("CTR"))
    return [_controle("document", "Génération du document", INFO,
                      f"Cette application vient de la {FIT}, la génération antérieure du seuil PMR PY1100 ; le {CONCEPTION} "
                      "est le document courant de la porte. Les deux ne sont jamais fusionnés : les références de la F.I.T "
                      "sont écrites sans T (VER-152), ses débits et ses remplissages diffèrent de ceux de 2018"
                      + (f" ({ctr})" if ctr else "") + (" : " + " ; ".join(ecarts) if ecarts else "") + ".",
                      [_source(a.defn.page, a.defn.section or "", a.app.localisation)], a.anomalies)]


def _controle_dimensions(d: Donnees, a: AppPY, s: Dict[str, Any], v: Vantail, ref: Optional[str], comment: str,
                         lim: Optional[Limite]) -> Dict[str, Any]:
    titre = "Dimensions maximales du vantail"
    typo = _typologie(a.defn, bool(s.get("renforcee")))
    anomalies = (["INC-681"] if typo.startswith("Porte tube APD 1") else []) + (["VER-583"] if s.get("renforcee") else [])
    if ref is None or lim is None:
        quoi = (f"L'organe {ref} n'est pas listé pour « {typo} » au tableau des dimensions maximales."
                if ref else f"Organe de rotation inconnu : {comment}.")
        return _controle("dimensions", titre, ETUDE, quoi, _src_limites(d), anomalies)
    deux = a.defn.vantaux > 1
    ok_h, ok_l = v.hv <= lim.hv, v.somme <= lim.largeur
    statut = _pire(OK if ok_h else (HORS if v.ferme_h else ETUDE), OK if ok_l else (HORS if v.ferme_l else ETUDE))
    detail = (f"{typo}, {lim.organe} {' ou '.join(lim.refs)} : hauteur de vantail {_fmt(v.hv)} mm pour {_fmt(lim.hv)} mm "
              f"maxi ({'dans la limite' if ok_h else 'dépassée'}) ; largeur {lim.largeur_texte} "
              f"{_fmt(v.somme, 1 if v.somme % 1 else 0)} mm{' (somme des deux vantaux)' if deux else ''} pour "
              f"{_fmt(lim.largeur)} mm maxi ({'dans la limite' if ok_l else 'dépassée'}). {v.texte}.")
    if statut == ETUDE:
        detail += (" La limite est dépassée par H ou L hors tout, qui majorent le vantail : saisir Hv et Lv pour décider.")
    return _controle("dimensions", titre, statut, detail, _src_limites(d), anomalies,
                     {"hv_max": lim.hv, "largeur_max": lim.largeur, "poids_nominal_kg": lim.poids,
                      "typologie": typo, "organe": lim.refs[0]})


def _poids_verre(d: Donnees, a: AppPY, s: Dict[str, Any], variables: Dict[str, float]) -> Tuple[Optional[float], str]:
    """Le verre d'un vantail : les remplissages de la planche hors fixe latéral. Une borne basse du poids du vantail."""
    if not d.densite_verre:
        return None, "la densité du verre n'est pas dans le wiki"
    surface = 0.0
    for r in a.app.remplissages:
        if "fixe" in r["libelle"].lower():
            continue
        h, rh = valeur_ou_none(r["hauteur"], variables)
        l, rl = valeur_ou_none(r["largeur"], variables)
        if h is None or l is None:
            return None, "remplissage non calculable : " + "; ".join(x for x in (rh, rl) if x)
        surface += h * l / 1e6
    if not surface:
        return None, "aucun remplissage de vantail"
    return (s["verre_mm"] or 0) * d.densite_verre * surface, ""


def _lire_abaque(ab: Abaque, hv: float, lv: float) -> Tuple[Optional[float], str]:
    """Le poids maxi lu sans interpolation : le plus bas des quatre croisements de graduations voisins."""
    hs = sorted(ab.grille)
    ls = sorted(next(iter(ab.grille.values())))
    if not hs[0] <= hv <= hs[-1] or not ls[0] <= lv <= ls[-1]:
        return None, (f"vantail {_fmt(hv)} × {_fmt(lv)} mm hors des graduations de l'abaque (hauteur {_fmt(hs[0])} à "
                      f"{_fmt(hs[-1])} mm, largeur {_fmt(ls[0])} à {_fmt(ls[-1])} mm)")
    h_voisins = sorted({max(h for h in hs if h <= hv), min(h for h in hs if h >= hv)})
    l_voisins = sorted({max(x for x in ls if x <= lv), min(x for x in ls if x >= lv)})
    cases = [ab.grille[h][l] for h in h_voisins for l in l_voisins]
    if any(c is None for c in cases):
        return None, "case absente de l'abaque"
    entre = "" if len(cases) == 1 else " (plus bas des graduations voisines, sans interpolation)"
    return min(cases), (f"H {'/'.join(_fmt(h) for h in h_voisins)} mm × L {'/'.join(_fmt(l) for l in l_voisins)} mm{entre}")


def _controle_organe(d: Donnees, a: AppPY, s: Dict[str, Any], v: Vantail, ref: Optional[str], lim: Optional[Limite],
                     poids: Optional[float], raison: str) -> Dict[str, Any]:
    titre = "Poids du vantail et organe de rotation"
    if lim is None or ref is None:
        return _controle("organe", titre, ETUDE, "Organe de rotation absent du tableau des dimensions maximales pour "
                         "cette typologie : le poids maxi n'est pas lu.", _src_limites(d))
    src = _src_limites(d) + [_source(PAGE_VERRE, "Remarques importantes concernant les abaques de renforcement")]
    anomalies: List[str] = []
    parties: List[str] = []
    statuts: List[str] = []
    if poids is None:
        statuts.append(ETUDE)
        parties.append(f"Poids du verre non calculé : {raison}.")
    else:
        parties.append(f"{_fmt(poids, 1)} kg de verre par vantail ({_fmt(s['verre_mm'] or 0, 1 if (s['verre_mm'] or 0) % 1 else 0)}"
                       f" mm ; {HYPOTHESE_VERRE}).")
        statuts.append(HORS if poids > lim.poids else INFO)
        parties.append(f"Poids maxi par vantail du tableau pour {lim.organe} {' ou '.join(lim.refs)} : {_fmt(lim.poids)} kg"
                       + (" : dépassé par le verre seul." if poids > lim.poids else
                          " ; les profilés ne sont pas comptés, le verre seul ne prouve pas que le vantail le respecte."))
    valeurs: Dict[str, Any] = {"poids_verre_kg": None if poids is None else round(poids, 1), "limite_nominale_kg": lim.poids}
    lames = next((d.lames[r] for r in lim.refs if r in d.lames), None)
    if lames:
        ab = d.abaques[(lames, bool(s.get("renforcee")))]
        src.append(_source(PAGE_ABAQUES[lames], ab.section, ab.localisation))
        anomalies += ["VER-600"]
        pm, lecture = _lire_abaque(ab, v.hv, v.lv)
        if pm is None:
            statuts.append(ETUDE)
            parties.append(f"Abaque de poids ({ab.section}) : {lecture}.")
        else:
            valeurs["abaque_kg"] = pm
            marge = d.precision_kg
            if poids is None:
                statuts.append(ETUDE)
                parties.append(f"Abaque de poids : {_fmt(pm)} kg maxi ({lecture}).")
            elif poids > pm + marge:
                statuts.append(_depasse(v))
                parties.append(f"Abaque de poids : {_fmt(pm)} kg maxi ({lecture}), dépassé par le verre seul"
                               + ("" if v.ferme else " ; la lecture à H et L hors tout est restrictive, saisir Hv et Lv"
                                                    " pour décider") + ".")
            elif poids > pm - marge:
                statuts.append(ETUDE)
                parties.append(f"Abaque de poids : {_fmt(pm)} kg maxi ({lecture}) à moins de la précision de lecture "
                               f"(± {_fmt(marge)} kg) du poids du verre.")
            else:
                statuts.append(INFO)
                parties.append(f"Abaque de poids : {_fmt(pm)} kg maxi ({lecture}), précision de lecture ± {_fmt(marge)} kg "
                               "; les profilés ne sont pas comptés.")
    elif lim.organe.startswith("pivot"):
        src.append(_source(PAGE_PIVOTS, "Choix des forces sur le pivot", d.loc_pivots))
        convient = [p for p in sorted(d.pivots, key=lambda p: p.force) if v.hv <= p.h and v.lv <= p.l]
        if convient:
            p = convient[0]
            statuts.append(INFO)
            valeurs["force_pivot"] = p.force
            parties.append(f"Pivot frein de sol : force {p.force} au moins pour {_fmt(v.hv)} × {_fmt(v.lv, 1 if v.lv % 1 else 0)}"
                           f" mm (H × L maxi {_fmt(p.h)} × {_fmt(p.l)} mm ; moment de fermeture {p.moment_2} N·m à 2°, "
                           f"{p.moment_90} N·m à 90°).")
        else:
            statuts.append(_depasse(v))
            parties.append(f"Pivot frein de sol : aucune force du tableau ne couvre {_fmt(v.hv)} × "
                           f"{_fmt(v.lv, 1 if v.lv % 1 else 0)} mm (plus grande porte : "
                           f"{_fmt(max(p.h for p in d.pivots))} × {_fmt(max(p.l for p in d.pivots))} mm).")
    else:
        parties.append(f"{lim.organe.capitalize()} {' ou '.join(lim.refs)} : le wiki ne donne pas d'abaque de poids, "
                       "seulement le poids maxi nominal.")
    return _controle("organe", titre, _pire(*statuts), " ".join(parties) + " " + v.texte + ".", src, anomalies, valeurs)


def _cas(d: Donnees, hv: float, ferme: bool, hv_saisie: bool) -> Tuple[Optional[Cas], str]:
    """Le cas de la répartition. Une hauteur de vantail saisie à 2200 mm exactement n'est dans aucun cas (INC-698). Faute de
    saisie, H hors tout tient lieu de Hv, et le vantail étant plus petit que H, H = 2200 mm donne un vantail sous 2200."""
    if hv_saisie and any(c.seuil == hv for c in d.cas):
        return None, (f"hauteur de vantail de {_fmt(hv)} mm : aucun cas ne la couvre, les conditions écrites sont « < » et "
                      "« > » (INC-698)")
    for c in d.cas:
        sous = hv < c.seuil or (not hv_saisie and hv == c.seuil)
        if c.ferme_porte == ferme and ((c.sens == "<" and sous) or (c.sens == ">" and hv > c.seuil)):
            return c, ""
    return None, "aucun cas ne correspond"


def _controle_paumelles(d: Donnees, a: AppPY, s: Dict[str, Any], v: Vantail, lim: Optional[Limite]) -> List[Dict[str, Any]]:
    if lim is None or not any(r in d.lames for r in lim.refs):
        return []
    titre = "Nombre de paumelles"
    ferme = a.defn.fpi or bool(s.get("ferme_porte"))
    src = [_source(PAGE_REPARTITION, "Répartition des paumelles", d.loc_cas)]
    cas, raison = _cas(d, v.hv, ferme, v.ferme_h)
    nom_cas = f"{'avec' if ferme else 'sans'} ferme-porte{' (FPI de la planche)' if a.defn.fpi else ''}"
    if cas is None:
        return [_controle("paumelles", titre, ETUDE, f"Porte {nom_cas} : {raison}.", src, ["INC-698"])]
    ligne = next((l for l in a.app.accessoires if l.ref in lim.refs), None)
    base = (f"Porte {nom_cas}, hauteur de vantail {_fmt(v.hv)} mm"
            + ("" if v.ferme_h else " (hypothèse : H hors tout tient lieu de Hv"
               + ("" if not any(c.seuil == v.hv for c in d.cas) else
                  f" ; le vantail est plus petit que H, donc à H = {_fmt(v.hv)} mm il est sous cette hauteur")
               + ")") + f" : cas {cas.cas}, {cas.nombre} paumelles par vantail.")
    statut, suite = INFO, ""
    if ligne is not None and re.fullmatch(r"\d+", ligne.qte.strip()):
        pose = int(ligne.qte) / a.defn.vantaux
        suite = f" La planche pose {ligne.qte} {ligne.ref}, soit {_fmt(pose, 1 if pose % 1 else 0)} par vantail."
        if pose < cas.nombre:
            statut = ETUDE
            suite += " C'est moins que le cas ne le demande."
    elif ligne is not None:
        suite = (f" La planche écrit « *** » pour {ligne.ref} (VER-572) : le nombre se lit sur cet abaque.")
    return [_controle("paumelles", titre, statut, base + suite, src, ["INC-698", "VER-572"],
                      {"cas": cas.cas, "nombre_par_vantail": cas.nombre})]


def _choix_parclose(d: Donnees, a: AppPY, n: int) -> Dict[str, Any]:
    """La parclose et le joint de la planche, et ce que la matrice donne à l'épaisseur n."""
    p = next((l for l in a.app.profils if l.designation.lower().startswith("parclose") or
              (not l.designation and _cle(l.ref).startswith("5910"))), None)
    j = next((l for l in a.app.joints if "parclose" in l.designation.lower()), None)
    cle_p, cle_j = (_cle(p.ref) if p else None), (_cle(j.ref) if j else None)
    propre = next((c for c in d.parcloses if _cle(c.parclose) == cle_p and _cle(c.joint) == cle_j), None)
    memes = [c for c in d.parcloses if _cle(c.parclose) == cle_p and c.remplissage == n]
    autres = [c for c in d.parcloses if c.remplissage == n and _cle(c.parclose) != cle_p]
    return {"parclose": p, "joint": j, "propre": propre, "memes": memes, "autres": autres}


def _controle_vitrage(d: Donnees, a: AppPY, s: Dict[str, Any]) -> Dict[str, Any]:
    titre = "Vitrage, parclose et joint"
    n = int(s["epaisseur_vitrage_mm"])
    src = [_source(PAGE_PARCLOSES, "Prises de volume et parcloses à pelle", d.loc_parcloses)]
    ch = _choix_parclose(d, a, n)
    if ch["parclose"] is None or ch["joint"] is None:
        return _controle("vitrage", titre, ETUDE, "La planche ne porte pas de parclose ou de joint de parclose lisible.",
                         src, ["VER-566"])
    p, j, propre = ch["parclose"], ch["joint"], ch["propre"]
    base = (f"Planche : parclose {p.ref} et {j.designation.lower() or 'joint'} {j.ref}"
            + (f" : remplissage de {_fmt(propre.remplissage)} mm (zone recommandée {_fmt(propre.zone[0])} à "
               f"{_fmt(propre.zone[1])} mm)." if propre else " : la matrice ne les croise pas."))
    valeurs = {"epaisseur": n, "parclose": p.ref, "joint_planche": j.ref}
    if propre and propre.remplissage == n:
        zone = propre.zone[0] <= n <= propre.zone[1]
        return _controle("vitrage", titre, OK, base + f" Le remplissage de {n} mm est celui de la planche"
                         + (", dans la zone recommandée." if zone else ", hors de la zone recommandée."),
                         src, ["VER-566"], dict(valeurs, joint=j.ref))
    if ch["memes"]:
        c = ch["memes"][0]
        valeurs["joint"] = c.joint
        return _controle("vitrage", titre, OK, base + f" À {n} mm la même parclose {p.ref} prend le joint intérieur {c.joint} "
                         f"(B = {_fmt(c.epaisseur_joint)} mm) ; la longueur de joint de la planche est reprise "
                         "(hypothèse : même longueur).", src, ["VER-566"], valeurs)
    if ch["autres"]:
        refs = ", ".join(sorted({f"{c.parclose} avec {c.joint}" for c in ch["autres"]}))
        return _controle("vitrage", titre, ETUDE, base + f" À {n} mm la parclose {p.ref} n'a pas de joint intérieur qui "
                         f"convienne ; la matrice le donne avec {refs}, ce qui change la parclose et les cotes de débit "
                         "et de remplissage de la planche.", src, ["VER-566"], valeurs)
    return _controle("vitrage", titre, HORS, base + f" Aucune parclose à pelle ne reçoit {n} mm.", src, ["VER-566"], valeurs)


def _controle_goupillage(snap: WikiSnapshot) -> List[Dict[str, Any]]:
    """Une préconisation du catalogue, dont la dimension n'est pas précisée : affichée, jamais appliquée."""
    m = re.search(r"Au-delà de (\d+) mm et de (\d+) kg par vantail, il est préconisé l'assemblage de l'ouvrant par goupillage",
                  _propre(_corps(snap, PAGE_LIMITES)))
    if not m:
        return []
    return [_controle("goupillage", "Goupillage de l'ouvrant", INFO,
                      f"Au-delà de {m.group(1)} mm et de {m.group(2)} kg par vantail, le catalogue préconise l'assemblage de "
                      "l'ouvrant par goupillage ; il ne dit pas de quelle dimension il s'agit, et le poids du vantail n'est "
                      "pas connu : à apprécier.", [_source(PAGE_LIMITES, "Préconisation de goupillage")])]


def _brouillons(d: Donnees, pages: List[str]) -> List[str]:
    return [p for p in dict.fromkeys(pages) if d.statuts.get(p) == "draft"]


def _raison_indefinie(a: AppPY) -> str:
    raisons = []
    if {"H1", "H2"} & set(a.indefinies):
        raisons.append("H1 et H2 « se partagent la hauteur » sans dire laquelle est laquelle")
    if "L2" in a.indefinies:
        raisons.append("L2 n'est pas défini par cette page, qui l'écrit pour L/2 (INC-684) : la correction n'est pas "
                       "appliquée sans décision du service technique")
    autres = [v for v in a.indefinies if v not in ("H1", "H2", "L2")]
    if autres:
        raisons.append(", ".join(autres) + " : aucune phrase ne les définit")
    return " ; ".join(raisons)


def _controle_application(a: AppPY) -> List[Dict[str, Any]]:
    """Les cotes de la planche qu'aucune phrase ne définit : la liste de coupe et le poids ne sont pas calculés."""
    if not a.indefinies:
        return []
    return [_controle("application", "Cotes de la planche", ETUDE,
                      f"La planche utilise {', '.join(a.indefinies)}, que sa page ne définit pas ({_raison_indefinie(a)}) : "
                      "la liste de coupe et le poids du vitrage ne sont pas calculés pour cette application.",
                      [_source(a.defn.page, a.defn.section or "", a.app.localisation)], a.anomalies)]


def verifier(snap: WikiSnapshot, saisie: Dict[str, Any]) -> Dict[str, Any]:
    d = donnees(snap)
    s = coercer(options(snap)["champs"], saisie)
    if s["configuration"] == AUTRE:
        return wiki_famille.relocaliser(snap, synthese(
            [_controle("application", "Application", ETUDE,
                       "Le wiki n'a de nomenclature que pour les 28 applications proposées : porte d'habitat, autre seuil, "
                       "autre ouvrant ou ouvrant caché ne sont ni dimensionnés ni débités.",
                       [_source(PAGE_LIMITES, "Dimensions maximales du vantail")], [])], _brouillons(d, [PAGE_LIMITES])))
    a = d.applications[s["configuration"]]
    variables = _variables(s, a)
    v = _vantail(s, a.defn)
    ref, comment = _organe(d, a, s)
    lim = _ligne_limite(d, a, s, ref)
    poids, raison = _poids_verre(d, a, s, variables)
    ctl: List[Dict[str, Any]] = []
    ctl += _controle_document(d, a)
    ctl += _controle_application(a)
    ctl.append(_controle_dimensions(d, a, s, v, ref, comment, lim))
    ctl.append(_controle_organe(d, a, s, v, ref, lim, poids, raison))
    ctl += _controle_paumelles(d, a, s, v, lim)
    ctl.append(_controle_vitrage(d, a, s))
    ctl += _controle_goupillage(snap)
    pages = [PAGE_LIMITES, PAGE_PARCLOSES, a.defn.page] + ([PAGE_REPARTITION, *PAGE_ABAQUES.values()] if lim else [])
    if lim and lim.organe.startswith("pivot"):
        pages.append(PAGE_PIVOTS)
    return wiki_famille.relocaliser(snap, synthese(
        ctl, _brouillons(d, pages), application=a.defn.libelle, document=a.defn.document,
        typologie=_typologie(a.defn, bool(s.get("renforcee"))), organe=ref,
        vantail={"hv": v.hv, "lv": v.lv, "saisi": v.ferme},
        remplissages=[{"libelle": r["libelle"], "hauteur": valeur_ou_none(r["hauteur"], variables)[0],
                       "largeur": valeur_ou_none(r["largeur"], variables)[0]} for r in a.app.remplissages],
        poids_verre_kg=None if poids is None else round(poids, 1)))


# ---- la fiche de débit ---------------------------------------------------------------------------


def _avec_paumelles(d: Donnees, a: AppPY, s: Dict[str, Any], v: Vantail, notes: List[str]) -> Application:
    """La quantité de paumelles que la planche écrit « *** » (VER-572), lue sur la répartition par cas."""
    lim_refs = {r for l in d.limites for r in l.refs if r in d.lames}
    cibles = [l for l in a.app.accessoires if l.ref in lim_refs and l.qte.strip() in ("*", "***")]
    if not cibles:
        return a.app
    cas, raison = _cas(d, v.hv, a.defn.fpi or bool(s.get("ferme_porte")), v.ferme_h)
    if cas is None:
        notes.append(f"Paumelles {', '.join(c.ref for c in cibles)} : quantité « *** » non lue ({raison}).")
        return a.app
    total = cas.nombre * a.defn.vantaux
    notes.append(f"Paumelles : « *** » est lu sur la répartition par cas : cas {cas.cas}, {cas.nombre} par vantail, "
                 f"{total} au total (hauteur de vantail {_fmt(v.hv)} mm, H hors tout tenant lieu de Hv).")
    return replace(a.app, accessoires=[replace(l, qte=str(total)) if l in cibles else l for l in a.app.accessoires])


def _avec_joint(d: Donnees, a: AppPY, app: Application, n: int, notes: List[str]) -> Application:
    ch = _choix_parclose(d, a, n)
    j, propre = ch["joint"], ch["propre"]
    if j is None or (propre and propre.remplissage == n) or not ch["memes"]:
        return app
    c = ch["memes"][0]
    notes.append(f"Joint de parclose : la planche donne {j.ref} pour {_fmt(propre.remplissage) if propre else '?'} mm ; à {n} mm "
                 f"la parclose {c.parclose} prend {c.joint} (B = {_fmt(c.epaisseur_joint)} mm), même longueur reprise : hypothèse.")
    joints = [replace(l, ref=c.joint, designation=f"Joint de parclose {_fmt(c.epaisseur_joint)} mm") if l == j else l
              for l in app.joints]
    return replace(app, joints=joints)


def fiche_debit(snap: WikiSnapshot, saisie: Dict[str, Any]) -> Dict[str, Any]:
    d = donnees(snap)
    s = coercer(options(snap)["champs"], saisie)
    if s["configuration"] == AUTRE:
        return {"indisponible": "Aucune nomenclature dans le wiki pour cette application (porte d'habitat, autre seuil, autre "
                                "ouvrant) : pas de liste de coupe."}
    a = d.applications[s["configuration"]]
    if a.indefinies:
        return {"indisponible": f"La planche « {a.defn.libelle} » utilise {', '.join(a.indefinies)}, que sa page ne définit "
                                f"pas ({_raison_indefinie(a)}) : le wiki ne permet pas de calculer sa liste de coupe."}
    ver = verifier(snap, saisie)
    v = _vantail(s, a.defn)
    n = int(s["epaisseur_vitrage_mm"])
    notes: List[str] = []
    app = _avec_paumelles(d, a, s, v, notes)
    app = _avec_joint(d, a, app, n, notes)
    images = dict(d.images)
    for k, x in list(images.items()):
        if k.startswith("T"):
            images.setdefault(k[1:], x)
    images.update(a.images)
    variables = _variables(s, a)
    nb = a.defn.vantaux if len(app.remplissages) == 1 else None
    fiche = construire_fiche(app, variables, n, {"ref": ""}, images, None, nb,
                             "un par vantail (la planche n'écrit pas de quantité ; son remplissage est écrit par vantail)")
    lignes = fiche["lignes"]
    fiche_notes = list(fiche["notes"])
    if a.defn.document == FIT:
        # La F.I.T écrit ses angles de coupe en toutes lettres : ce ne sont pas des pictogrammes sans légende.
        for l in lignes:
            l["coupe"] = re.sub(r"pictogramme « (.*) » \(non légendé\)", r"angles de coupe écrits : \1", l["coupe"])
        fiche_notes = [("Les débits et les cotes de la F.I.T sont portés sans unité (VER-151) ; les références sont écrites "
                        "sans T (VER-152) ; les angles de coupe sont ceux que la fiche écrit."
                        if x.startswith("Longueurs de coupe droites") else x) for x in fiche_notes]
    elif a.defn.vantaux > 1 and len(app.remplissages) > 1:
        for l in lignes:
            if l["famille"] == "Remplissage" and "fixe" not in l["piece"].lower():
                l["qte"] = a.defn.vantaux
                l["note"] = ("hypothèse : chaque vantail reçoit les deux vitrages ; la planche n'écrit pas de quantité")
    manquantes = [c["id"] for c in ver["controles"] if c["statut"] in (ETUDE, HORS)]
    if manquantes:
        fiche_notes.append("Faisabilité : " + ver["libelle"].lower() + " (contrôles : " + ", ".join(manquantes) + ").")
    return wiki_famille.relocaliser(snap, {
        "verdict": ver["verdict"], "libelle": ver["libelle"], "lignes": lignes, "nomenclature": _nomenclature(lignes),
        "accessoires": [], "notes": fiche_notes + notes,
        "brouillons": _brouillons(d, [a.defn.page, PAGE_PARCLOSES]), "application": a.defn.libelle,
        "document": a.defn.document})


def _nomenclature(lignes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """La nomenclature recomptée sur les lignes finales (la quantité d'un vitrage a pu être corrigée)."""
    nomen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for l in lignes:
        k = (l["piece"] if l["ref"] == "—" else l["ref"], l["famille"])
        x = nomen.setdefault(k, {"ref": l["ref"], "designation": l["piece"].strip(), "famille": l["famille"], "qte": 0,
                                 "ml": 0.0, "image": l["image"]})
        x["qte"] += l["qte"] or 0
        if l["longueur"] is not None:
            x["ml"] += (l["qte"] or 1) * l["longueur"] / 1000
    return [dict(x, ml=round(x["ml"], 2) if x["ml"] else None) for x in nomen.values()]
