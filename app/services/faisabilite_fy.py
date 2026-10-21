"""Le calculateur du SOLEAL FY (fenêtre à frappe LUMINE55, système TECHNAL SOLEAL FY 55 ; FY 65 en partie), 21/10/2026.

Même forme que ``faisabilite_ga`` et ``faisabilite_gy`` (dont il réutilise l'évaluateur de formules, la lecture
des nomenclatures, la fiche de débit et le poids de verre) : toutes les valeurs sont lues dans les tableaux du wiki
à chaque rechargement de l'instantané.

**Deux générations de la même nomenclature, jamais fusionnées.** Le catalogue 6319.003 (07/2021, quincaillerie
cachée « QC », variables H et W, modules 55 et 65) est le courant ; le catalogue 6057.003 (07/2020, quincaillerie
visible, variables H et L, module 55) reste proposé. L'utilisateur choisit la génération, le contrôle
« Génération du catalogue » dit laquelle sert, si l'autre porte la même application et ce qui change (références,
formules, quantités) : on ne mélange jamais une ligne de l'une avec une ligne de l'autre.

Ce que le wiki permet, et ce que ce module en fait :

- **le DTA 6/12-2016_V5** borne la baie (H × L) d'un SOLEAL 55 FYa (ouvrant Apparent) ou FYm (ouvrant Minimal)
  selon le type de fenêtre. Il **ne couvre que le FY 55** et **expire** : la validité est lue sur la page et
  affichée sur le contrôle, et un avis échu fait passer le verdict à « sur étude » ; il ne propose aucun seuil PMR
  (CTR-141) ;
- **l'inertie des deux montants d'un châssis 2 vantaux** (courbes de la page des dimensions maximales) : aux
  graduations de W seulement, ±0,03 m, « - » = courbe non tracée, jamais d'interpolation. Elle ne décide que hors
  DTA : le DTA borne déjà L à 1,55 m ;
- **la quincaillerie cachée (génération 6319)** : le champ de l'abaque (Wv × Hv du vantail, zone blanche hors
  champ), la zone et ses points de fermeture, la classe de poids de l'articulation (80, 110 ou 160 kg) face au
  poids de verre. Le wiki ne dit pas ce que mesure la valeur en kg/m² de l'abaque (VER-406) : elle est lue, affichée,
  jamais comparée. La quincaillerie cachée n'est pas disponible en LUMINE65 ;
- **la quincaillerie visible (génération 6057)** : poids maxi et largeur maxi du vantail de la paumelle choisie. Les
  bandes de kg/m² des abaques de paumelles ne sont pas relevées (VER-370) : le nombre de paumelles n'est pas calculé ;
- **le vitrage** : les parcloses et joints de l'épaisseur saisie, lus par ``parcloses`` ; le DTA s'arrête à 32 mm ;
- **la liste de coupe** de chaque application dont le wiki a la nomenclature : les 16 planches du catalogue 6319
  (1 et 2 vantaux, avec ou sans seuil PMR, Apparent ou Minimal, FY 55 et FY 65) et 22 planches du catalogue 6057
  (fixe, OB, soufflet, portes-fenêtres, ensembles menuisés, italienne, projection).

Hypothèses affichées (le wiki ne les écrit pas) : la baie est prise égale aux dimensions hors tout saisies ;
Hv et Wv sont les longueurs de coupe des profilés d'ouvrant ; le poids est celui du verre seul, à la masse par mm
de la page des abaques du système 70.
"""
from __future__ import annotations

import datetime
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

from app.services import parcloses, wiki_famille
from app.services.faisabilite import (DonneeIntrouvable, ETUDE, HORS, INFO, OK, Table, _borne, _controle, _corps, _fmt,
                                      _nombre, _source, _texte_section, _une, tables)
from app.services.faisabilite_ga import (Application, Ligne, anomalies_citees, construire_fiche, densite_du_verre,
                                         dimensions, images_par_reference, poids_du_verre, valeur_ou_none)
from app.services.faisabilite_systemes import coercer, synthese
from app.services.wiki_service import WikiSnapshot
from app.services.wiki_tables import tables_toutes

ID = "soleal-fy"
LIBELLE = "SOLEAL FY (fenêtre à frappe LUMINE)"
GAMMES = ["LUMINE"]
PORTEE = ("Fenêtre et porte-fenêtre à frappe SOLEAL FY : DTA 6/12-2016_V5 (dimensions maximales de baie, FY 55 seulement, "
          "validité affichée), inertie des montants d'un châssis 2 vantaux, quincaillerie cachée (champ de l'abaque, zone, "
          "classe de poids) ou visible (paumelle), vitrage, liste de coupe des catalogues 6319.003 (courant) et 6057.003 "
          "(ancien). « Sur étude » : le FY 65 (aucun DTA), le seuil PMR, le nombre de paumelles visibles et la valeur en "
          "kg/m² des abaques de quincaillerie cachée, dont le wiki ne dit pas ce qu'elle mesure.")

PAGE_DTA = "/certifications/dta-6-12-2016-soleal-fy-dossier-technique.md"
PAGE_DTA_AVIS = "/certifications/dta-6-12-2016-soleal-fy.md"
PAGE_INERTIE = "/profiles/soleal-fy-qc-dimensions-maximales-chassis-2-vantaux.md"
PAGE_ABAQUES_F = "/quincaillerie/soleal-fy-qc-abaques-fenetre-francaise.md"
PAGE_ABAQUES_OB = "/quincaillerie/soleal-fy-qc-prise-de-cotes-articulations-et-abaques-ob-bo.md"
PAGE_SOUFFLET = "/quincaillerie/soleal-fy-qc-soufflet-abaque-et-ferrures.md"
PAGE_PRINCIPE_QC = "/quincaillerie/soleal-fy-qc-principe-et-ligne-du-produit.md"
PAGE_VISIBLE = "/quincaillerie/soleal-fy-quincaillerie-visible-et-cachee.md"
PAGE_FERRURES_VISIBLES = "/quincaillerie/soleal-fy-ferrures-francaise-et-soufflet.md"
PAGE_PARCLOSES = "/profiles/soleal-fy-parcloses-et-vitrage.md"
PAGE_PARCLOSES_65 = "/profiles/soleal-fy-qc-65-parcloses-et-prises-de-volume.md"
PAGE_CALAGE = "/profiles/soleal-fy-prise-de-volume-equilibre-et-calage.md"
PAGE_VERRE = "/profiles/systeme-70-abaques-dimensionnels.md"

P_6057 = "/profiles/soleal-fy-cotes-de-debit.md"
P_6057_OB = "/profiles/soleal-fy-cotes-de-debit-apparent-ob-soufflet-et-ensemble.md"
P_6057_PF = "/profiles/soleal-fy-cotes-de-debit-portes-fenetres-apparent.md"
P_6057_MIN = "/profiles/soleal-fy-cotes-de-debit-ouvrant-minimal-fenetres.md"
P_6057_MIN_PF = "/profiles/soleal-fy-cotes-de-debit-ouvrant-minimal-portes-fenetres.md"
P_6057_EQ = "/profiles/soleal-fy-cotes-de-debit-equilibre-italienne-projection.md"


# ---- ce que l'utilisateur choisit --------------------------------------------------------------

G_6319, G_6057 = "6319", "6057"
GENERATIONS = {
    G_6319: "Catalogue 6319.003 (07/2021), quincaillerie cachée : le courant",
    G_6057: "Catalogue 6057.003 (07/2020), quincaillerie visible : l'ancien",
}
NOMS_GENERATION = {G_6319: "6319.003 (QC, courant)", G_6057: "6057.003 (visible, ancien)"}
OUVRANTS = {"Apparent": "Ouvrant Apparent (FYa)", "Minimal": "Ouvrant Minimal (FYm)"}
MODULES = {"55": "SOLEAL FY 55 (LUMINE55)", "65": "SOLEAL FY 65 (correspondance LUMINE65 non écrite, VER-392)"}

TYPOLOGIES = {
    "fixe": "Châssis fixe",
    "v1": "Fenêtre ou porte-fenêtre 1 vantail (à la française)",
    "v2": "Fenêtre ou porte-fenêtre 2 vantaux (à la française)",
    "v1_ob": "Fenêtre 1 vantail oscillo-battant (OB / BO)",
    "v2_ob": "Fenêtre 2 vantaux oscillo-battants (OB / BO)",
    "soufflet": "Fenêtre à soufflet",
    "v1_pmr": "Porte-fenêtre 1 vantail avec seuil PMR",
    "v2_pmr": "Porte-fenêtre 2 vantaux avec seuil PMR",
    "pf1": "Porte-fenêtre 1 vantail avec seuil",
    "pf2": "Porte-fenêtre 2 vantaux avec seuil",
    "pf1_serrure": "Porte-fenêtre 1 vantail avec montant serrure",
    "v2_fixe": "Fenêtre 2 vantaux et fixe latéral (DTA seulement)",
    "ensemble": "Ensemble menuisé : fenêtre française et fixe latéral sur allège",
    "equilibre": "Ensemble menuisé renfort vertical, version Equilibre",
    "italienne": "Fenêtre italienne (ouverture extérieure)",
    "projection": "Fenêtre projection (ouverture extérieure)",
}
# Nombre de vantaux ouvrants, quand il est écrit par le nom de la typologie ; 0 : pas d'ouvrant ; ``None`` : plusieurs
# vitrages dont aucune planche ne dit lequel est le vantail (VER-360).
NB_VANTAUX: Dict[str, Optional[int]] = {
    "fixe": 0, "v1": 1, "v2": 2, "v1_ob": 1, "v2_ob": 2, "soufflet": 1, "v1_pmr": 1, "v2_pmr": 2, "pf1": 1, "pf2": 2,
    "pf1_serrure": 1, "v2_fixe": 2, "ensemble": None, "equilibre": None, "italienne": 1, "projection": 1}
DEUX_VANTAUX = ("v2", "v2_ob", "v2_pmr", "pf2", "v2_fixe")
PMR = ("v1_pmr", "v2_pmr")

# La planche du catalogue 6319 d'un nombre de vantaux sert plusieurs ouvertures (trois pictogrammes en tête).
SECTION_6319 = {"v1": "Châssis 1 vantail", "v1_ob": "Châssis 1 vantail", "soufflet": "Châssis 1 vantail",
                "v2": "Châssis 2 vantaux", "v2_ob": "Châssis 2 vantaux",
                "v1_pmr": "Châssis 1 vantail avec seuil PMR", "v2_pmr": "Châssis 2 vantaux avec seuil PMR"}

# Une application du catalogue 6057 : (ouvrant, typologie) → (page, titre de la section).
APPLICATIONS_6057: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("Apparent", "fixe"): (P_6057, "Châssis fixe, ouvrant Apparent"),
    ("Apparent", "v1"): (P_6057, "Fenêtre et porte-fenêtre 1 vantail, ouvrant Apparent"),
    ("Apparent", "v2"): (P_6057, "Fenêtre et porte-fenêtre 2 vantaux, ouvrant Apparent"),
    ("Apparent", "v1_ob"): (P_6057_OB, "Fenêtre et porte-fenêtre OB 1 vantail, ouvrant Apparent"),
    ("Apparent", "v2_ob"): (P_6057_OB, "Fenêtre et porte-fenêtre OB 2 vantaux, ouvrant Apparent"),
    ("Apparent", "soufflet"): (P_6057_OB, "Fenêtre à soufflet, ouvrant Apparent"),
    ("Apparent", "ensemble"): (P_6057_OB, "Ensemble menuisé fenêtre française et fixe latéral sur allège, ouvrant Apparent"),
    ("Apparent", "pf1"): (P_6057_PF, "Porte-fenêtre 1 vantail avec seuil, ouvrant Apparent"),
    ("Apparent", "pf2"): (P_6057_PF, "Porte-fenêtre 2 vantaux avec seuil, ouvrant Apparent"),
    ("Apparent", "v1_pmr"): (P_6057_PF, "Porte-fenêtre 1 vantail ouvrant serrure avec seuil PMR, ouvrant Apparent"),
    ("Apparent", "v2_pmr"): (P_6057_PF, "Porte-fenêtre 2 vantaux ouvrant serrure avec seuil PMR, ouvrant Apparent"),
    ("Apparent", "italienne"): (P_6057_EQ, "Fenêtre italienne, ouverture extérieure, ouvrant Apparent"),
    ("Apparent", "projection"): (P_6057_EQ, "Fenêtre projection, ouverture extérieure, ouvrant Apparent"),
    ("Minimal", "v1"): (P_6057_MIN, "Fenêtre et porte-fenêtre 1 vantail, ouvrant Minimal"),
    ("Minimal", "v2"): (P_6057_MIN, "Fenêtre et porte-fenêtre 2 vantaux, ouvrant Minimal"),
    ("Minimal", "v1_ob"): (P_6057_MIN, "Fenêtre et porte-fenêtre OB 1 vantail, ouvrant Minimal"),
    ("Minimal", "v2_ob"): (P_6057_MIN, "Fenêtre et porte-fenêtre OB 2 vantaux, ouvrant Minimal"),
    ("Minimal", "soufflet"): (P_6057_MIN, "Fenêtre soufflet, ouvrant Minimal"),
    ("Minimal", "pf1"): (P_6057_MIN_PF, "Porte-fenêtre 1 vantail avec seuil, ouvrant Minimal"),
    ("Minimal", "pf1_serrure"): (P_6057_MIN_PF, "Porte-fenêtre 1 vantail avec montant serrure, ouvrant Minimal"),
    ("Minimal", "ensemble"): (P_6057_MIN_PF, "Ensemble menuisé fenêtre française et fixe latéral sur allège, ouvrant Minimal"),
    ("Minimal", "equilibre"): (P_6057_EQ, "Ensemble menuisé renfort vertical, version Equilibre"),
}

# Le type de fenêtre du tableau du DTA, par typologie. Absent : le DTA n'a pas de ligne pour elle.
DTA_TYPE = {"v1": "1 vantail OF", "pf1": "1 vantail OF", "pf1_serrure": "1 vantail OF", "v1_ob": "1 vantail OB",
            "v2": "2 vantaux OF", "pf2": "2 vantaux OF", "v2_fixe": "2 vantaux OF + fixe latéral", "soufflet": "Soufflet"}
DTA_OUVRANT = {"Apparent": "a", "Minimal": "m"}      # « Soleal 55 FYa » / « Soleal 55 FYm »

# L'abaque de quincaillerie cachée qui sert une typologie.
ABAQUE_DE = {"v1": "F", "v2": "F", "v1_pmr": "F", "v2_pmr": "F", "v1_ob": "OB", "v2_ob": "OB", "soufflet": "S"}

HYPOTHESE_BAIE = "hypothèse : la baie est prise égale aux dimensions hors tout saisies"
HYPOTHESE_VANTAIL = ("hypothèse : Hv et Wv sont les longueurs de coupe des profilés d'ouvrant (hauteur et largeur du "
                     "vantail), le wiki ne le dit pas")
JOURS_AVANT_ECHEANCE = 90


def _aujourdhui() -> datetime.date:
    """La date du jour, séparée pour qu'un test fixe l'échéance du DTA."""
    return datetime.date.today()


# ---- les données lues dans le wiki --------------------------------------------------------------


@dataclass
class Champ:
    kg_m2: Optional[float]      # ``None`` : couleur dont la valeur n'est pas imprimée
    lo: float
    hi: float
    dagger: bool


@dataclass
class AbaqueQC:
    classe: int                 # kg
    page: str
    section: str
    localisation: str
    lignes: Dict[int, Tuple[float, List[Champ]]]      # Wv → (Hv maxi du champ, champs de couleur)


@dataclass
class Zone:
    lettre: str
    wv: Tuple[float, float]
    hv: Tuple[float, float, float]                    # (Hv mini, Hv sûr jusqu'à, Hv au plus)
    wv_abaque: float                                  # Wv au plus sur l'abaque (zone D du soufflet : 2400)
    points: str


@dataclass
class Donnees:
    statuts: Dict[str, str]
    dta: Dict[Tuple[str, str], Tuple[float, float]]
    dta_localisation: str
    dta_validite: Tuple[datetime.date, datetime.date]
    dta_vitrage_max: float
    dta_sans_pmr: bool
    inertie: Dict[Tuple[str, str], Dict[str, Any]]
    inertie_precision: float
    abaques: Dict[str, Dict[int, AbaqueQC]]
    zones: Dict[str, List[Zone]]
    hv_mini: Dict[str, Dict[str, float]]
    articulations: List[List[str]]
    paumelles: Dict[str, Dict[str, Any]]
    cas_paumelles: Dict[str, List[Tuple[str, str, str]]]
    pivot_sans_pmr: bool
    pmr_minimal: Tuple[float, ...]
    precision_abaque: float                           # « précision de lecture de ±50 mm » (page des abaques)
    precision_dagger: float                           # une cellule marquée † est « lue à ±100 mm »
    soufflet_kg: int
    cale_6319: float                                  # hauteur au-delà de laquelle le catalogue 6319 prescrit une cale
    cale_6057: Tuple[float, float]                    # (hauteur, épaisseur de la cale) du catalogue 6057, version Minimale
    apps: Dict[Tuple[str, str, str, str], Application]
    cotes: Dict[str, List[str]]                       # variable de cote → typologies qui l'emploient
    densite_verre: Optional[float]
    images: Dict[str, str]


_CACHE: Dict[str, Any] = {"cle": None, "donnees": None}


def donnees(snap: WikiSnapshot) -> Donnees:
    cle = (id(snap), snap.loaded_at)
    if _CACHE["cle"] != cle:
        _CACHE["donnees"] = _charge(snap)
        _CACHE["cle"] = cle
    return _CACHE["donnees"]


def _date(texte: str) -> datetime.date:
    j, m, a = (int(x) for x in texte.split("/"))
    return datetime.date(a, m, j)


def _m_en_mm(c: str) -> Optional[float]:
    v = _nombre(c)
    return None if v is None else round(v * 1000)


def _lire_dta(snap: WikiSnapshot):
    corps = _corps(snap, PAGE_DTA)
    t = _une(tables(corps), "Type de fenêtre | Ouvrants | Hauteur de baie H (m) | Largeur de baie L (m)", PAGE_DTA)
    dta: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for r in t.lignes:
        m = re.search(r"FY([am])\s*$", r[1])
        if not m:
            raise DonneeIntrouvable(f"{PAGE_DTA} : ouvrant du tableau des dimensions maximales illisible : « {r[1]} »")
        dta[(r[0], m.group(1))] = (_m_en_mm(r[2]), _m_en_mm(r[3]))
    mv = re.search(r"valide du (\d{2}/\d{2}/\d{4}) au (\d{2}/\d{2}/\d{4})", snap.pages[PAGE_DTA].raw_text)
    if not mv:
        raise DonneeIntrouvable(f"{PAGE_DTA} : validité du DTA (« valide du … au … ») introuvable")
    vit = re.search(r"jusqu'à (\d+) mm d'épaisseur", corps)
    if not vit:
        raise DonneeIntrouvable(f"{PAGE_DTA} : épaisseur maximale du vitrage isolant introuvable")
    sans_pmr = bool(re.search(r"ne dispose pas d'une solution de seuil permettant l'accès des handicapés",
                              _corps(snap, PAGE_DTA_AVIS)))
    return dta, t.localisation, (_date(mv.group(1)), _date(mv.group(2))), float(vit.group(1)), sans_pmr


def _lire_inertie(snap: WikiSnapshot):
    corps = _corps(snap, PAGE_INERTIE)
    prec = re.search(r"précision de lecture ± ?(\d+,\d+) m", corps)
    if not prec:
        raise DonneeIntrouvable(f"{PAGE_INERTIE} : précision de lecture des courbes introuvable")
    courbes: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for t in tables_toutes(corps):
        m = re.match(r"Châssis 2 vantaux, version (Apparent|Minimal)$", t.section)
        if not (m and t.entete and t.entete[0] == "W (m)"):
            continue
        for k in range(1, len(t.entete)):
            mc = re.match(r"H maxi, courbe (\d), (FY\d+), (.+?) \(m\)$", t.entete[k])
            if not mc:
                raise DonneeIntrouvable(f"{PAGE_INERTIE} : en-tête de courbe illisible : « {t.entete[k]} »")
            valeurs = {_m_en_mm(r[0]): _m_en_mm(r[k]) for r in t.lignes}
            courbes[(m.group(1), mc.group(2))] = {
                "courbe": int(mc.group(1)), "montants": mc.group(3), "valeurs": valeurs, "section": t.section,
                "localisation": t.localisation}
    if not courbes:
        raise DonneeIntrouvable(f"{PAGE_INERTIE} : tableaux des châssis 2 vantaux Apparent et Minimal introuvables")
    return courbes, _m_en_mm(prec.group(1))


def _plage(texte: str) -> Optional[Tuple[float, float]]:
    """« 425 à 1200 » → (425, 1200) ; « jusqu'à 1200 » → (0, 1200)."""
    m = re.search(r"(\d+)\s*à\s*(\d+)", texte)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"jusqu'à\s*(\d+)", texte)
    return (0.0, float(m.group(1))) if m else None


def _lire_zones(t: Table) -> List[Zone]:
    out = []
    for r in t.lignes:
        wv, hv = _plage(r[1]), _plage(r[2])
        if wv is None or hv is None:
            raise DonneeIntrouvable(f"zone « {r[0]} » : Wv ou Hv illisible ({r[1]} ; {r[2]})")
        # une zone dont la limite haute descend (soufflet) : « 510 à 1200 en Wv 1201, jusqu'à 1075 en Wv 1600 »
        mj = re.search(r"jusqu'à (\d+) en Wv", r[2])
        sur = float(mj.group(1)) if mj else hv[1]
        # une zone dont l'abaque va plus loin que le titre des planches : « … ; l'abaque va jusqu'à 2400 »
        ma = re.search(r"l'abaque va jusqu'à (\d+)", r[1])
        out.append(Zone(r[0], wv, (hv[0], min(sur, hv[1]), hv[1]), float(ma.group(1)) if ma else wv[1], r[3]))
    return out


def _lire_abaques(snap: WikiSnapshot, page: str) -> Tuple[Dict[int, AbaqueQC], List[Zone], Dict[str, float], List[List[str]]]:
    corps = _corps(snap, page)
    abaques: Dict[int, AbaqueQC] = {}
    zones: List[Zone] = []
    mini: Dict[str, float] = {}
    artic: List[List[str]] = []
    for t in tables_toutes(corps):
        m = re.search(r"Poids Maxi vantail (\d+) kg", t.section)
        if m and t.entete[:2] == ["Wv (mm)", "Hv maxi du champ (mm)"]:
            valeurs = [(re.match(r"(\d+) kg/m²", h) or [None, None])[1] for h in t.entete[2:]]
            lignes: Dict[int, Tuple[float, List[Champ]]] = {}
            for r in t.lignes:
                champs = []
                for v, cell in zip(valeurs, r[2:]):
                    mc = re.match(r"(\d+)\s*à\s*(\d+)\s*(†)?", cell)
                    if mc:
                        champs.append(Champ(float(v) if v else None, float(mc.group(1)), float(mc.group(2)), bool(mc.group(3))))
                lignes[int(_nombre(r[0]))] = (_nombre(r[1]), champs)
            abaques[int(m.group(1))] = AbaqueQC(int(m.group(1)), page, t.section, t.localisation, lignes)
        elif t.entete[:3] == ["Zone", "Wv (mm)", "Hv (mm)"]:
            zones = _lire_zones(t)
        elif t.entete[:2] == ["Châssis", "Hv mini (mm)"]:
            mini = {r[0]: _nombre(r[1]) for r in t.lignes}
        elif t.entete[:1] == ["Poids du vantail (kg)"]:
            artic = t.lignes
    return abaques, zones, mini, artic


def _lire_visible(snap: WikiSnapshot):
    corps = _corps(snap, PAGE_VISIBLE)
    paumelles: Dict[str, Dict[str, Any]] = {}
    cas: Dict[str, List[Tuple[str, str, str]]] = {}
    ts = tables_toutes(corps)
    for t in ts:
        if t.entete[:4] == ["Référence", "Désignation", "Poids maxi du vantail (kg)", "Lv maxi (mm)"]:
            for r in t.lignes:
                if _nombre(r[2]) is not None and _nombre(r[3]) is not None:
                    paumelles[r[0]] = {"designation": r[1], "poids": _nombre(r[2]), "lv": _nombre(r[3]),
                                       "pivot": "pivot" in " ".join(t.chemin).lower(), "section": t.section,
                                       "localisation": t.localisation}
    if not paumelles:
        raise DonneeIntrouvable(f"{PAGE_VISIBLE} : tableaux « Poids maxi du vantail (kg) | Lv maxi (mm) » introuvables")
    for t in ts:
        if t.entete[:3] == ["Cas", "Référence", "Qté"]:
            for ref in paumelles:
                if ref in t.section:
                    cas.setdefault(ref, []).extend((r[0], r[1], r[2]) for r in t.lignes)
    pivot = bool(re.search(r"paumelles sur pivot ne sont pas compatibles avec le seuil PMR", corps))
    return paumelles, cas, pivot


# -- les nomenclatures

_GENRES_PREMIERE_COLONNE = {"Profilé": "profils", "Accessoire": "accessoires", "Joint": "joints", "Vis": "vis"}


def _genre(t: Table) -> Optional[str]:
    """« profils », « joints », « accessoires » ou « vis » selon la forme du tableau (les deux catalogues n'ont pas la
    même première colonne : « Référence » en 6057 ; « Profilé », « Accessoire », « Joint » ou « Vis » en 6319).

    Un tableau dont la première colonne désigne le genre mais dont les colonnes de quantité ou de débit ont changé de
    nom n'est pas lu de travers : le chargement échoue (``DonneeIntrouvable``)."""
    h = t.entete
    if not h:
        return None
    qte = [i for i, c in enumerate(h) if c.startswith(("Qté", "Quantité"))]
    debit = [i for i, c in enumerate(h) if c.startswith("Débit")]
    if h[0] in _GENRES_PREMIERE_COLONNE:
        genre = _GENRES_PREMIERE_COLONNE[h[0]]
        attendu = {"profils": qte == [2] and debit == [len(h) - 1], "accessoires": qte == [2] and not debit,
                   "joints": debit == [len(h) - 1] and not qte, "vis": qte == [1] and len(h) == 2}[genre]
        if not attendu:
            raise DonneeIntrouvable(f"tableau « {h[0]} » : colonnes de quantité ou de débit changées : {h}")
        return genre
    if h[0] != "Référence":
        return None
    if len(h) == 2:
        return "vis"
    return "profils" if debit and qte else ("joints" if debit else ("accessoires" if qte else None))


def _coupe(t: Table, b: List[str]) -> str:
    """Le type de coupe d'une ligne de profilé. 6057 l'écrit en mots (« onglet / onglet ») ; 6319 en symboles
    (`/\\`, `\\|\\|`, `\\|\\`) dont la barre échappée coupe la cellule : on recolle les morceaux avec la barre."""
    if len(t.entete) > 3 and t.entete[3] == "Type de coupe":
        return b[3].strip()
    return "|".join(b[3:-1]).strip().strip("`").replace("\\|", "|") if len(b) > 4 else ""


def _lignes(t: Table, genre: str) -> List[Ligne]:
    out: List[Ligne] = []
    for r, b in zip(t.lignes, t.brut):
        if not r or not r[0]:
            continue
        if genre == "vis":
            out.append(Ligne(r[0], "", r[-1], ""))
        elif genre == "accessoires":
            # « *** » (voir abaque) : le nettoyage des cellules retire le gras, on rend la cellule telle qu'écrite
            out.append(Ligne(r[0], r[1], b[2].strip() if b[2].strip().startswith("*") else r[2], ""))
        elif genre == "joints":
            out.append(Ligne(r[0], r[1], "1", r[-1]))
        else:
            out.append(Ligne(r[0], r[1], r[2], r[-1], _coupe(t, b)))
    return out


_REMPLISSAGE_MOTS = re.compile(r"Remplissage[^:\n]*:\s*hauteur\s+(\S+)\s+mm,\s+largeur\s+(\S+?)\s*mm")
_REMPLISSAGE_ET = re.compile(r"Remplissage[^:\n]*:\s*(H\d?-[\d,.]+)\s+et\s+((?:W|L)[\w/\-.,]*\d)")
_COTE_PREFIXEE = re.compile(r"^\d+[A-Za-z]")


def _cote_de_vitrage(formule: str) -> str:
    """Une cote de vitrage écrite avec un coefficient devant la variable (« 2H3-98 », VER-357) : le préfixe n'est défini
    nulle part, la formule est rendue illisible plutôt que lue comme 2 × H3 − 98."""
    return f"{formule} [préfixe non défini]" if _COTE_PREFIXEE.match(formule.strip()) else formule


def _lire_remplissages(texte: str, tables_section: List[Table]) -> List[Dict[str, str]]:
    rempl: List[Dict[str, str]] = []
    for m in _REMPLISSAGE_MOTS.finditer(texte):
        rempl.append({"libelle": "", "hauteur": m.group(1), "largeur": m.group(2)})
    for m in _REMPLISSAGE_ET.finditer(texte):
        rempl.append({"libelle": "", "hauteur": m.group(1), "largeur": m.group(2)})
    for t in tables_section:
        if t.entete[:2] == ["Hauteur (mm)", "Largeur (mm)"]:
            rempl += [{"libelle": "", "hauteur": r[0], "largeur": r[1]} for r in t.lignes if r and r[0]]
    return [dict(r, hauteur=_cote_de_vitrage(r["hauteur"]), largeur=_cote_de_vitrage(r["largeur"])) for r in rempl]


def lire_application(corps: str, page: str, section: str, id_: str, libelle: str) -> Application:
    """Les tableaux d'une section (sous-sections comprises) et ses cotes de remplissage. Une section peut porter
    plusieurs tableaux de profilés (l'ensemble Equilibre en a deux) : ils sont mis bout à bout."""
    ts = [t for t in tables_toutes(corps) if section in t.chemin]
    lignes: Dict[str, List[Ligne]] = {g: [] for g in ("profils", "accessoires", "joints", "vis")}
    loc = ""
    for t in ts:
        g = _genre(t)
        if g:
            lignes[g] += _lignes(t, g)
            if g == "profils" and not loc:
                loc = t.localisation
    for g in ("profils", "accessoires", "joints"):
        if not lignes[g]:
            raise DonneeIntrouvable(f"{page} : section « {section} », tableau des {g} introuvable")
    texte = _texte_section(corps, section)
    rempl = _lire_remplissages(texte, ts)
    if not rempl:
        raise DonneeIntrouvable(f"{page} : section « {section} », cotes du remplissage introuvables")
    return Application(id_, libelle, page, section, loc, lignes["profils"], lignes["accessoires"], lignes["joints"],
                       lignes["vis"], rempl, anomalies_citees(texte))


def _formules(app: Application) -> List[str]:
    f = [l.debit for l in app.profils + app.joints] + [l.qte for l in app.profils + app.accessoires + app.vis]
    for r in app.remplissages:
        f += [r["hauteur"], r["largeur"]]
    return [x for x in f if x]


def _variables_de_cote(app: Application) -> List[str]:
    """Les cotes indicées (H1, L2…) que la nomenclature emploie : la saisie doit les donner."""
    return sorted({v for f in _formules(app) for v in re.findall(r"[A-Za-z][A-Za-z0-9]*", f) if re.fullmatch(r"[HLW]\d+", v)})


def _charge(snap: WikiSnapshot) -> Donnees:
    dta, loc_dta, validite, vit_max, sans_pmr = _lire_dta(snap)
    inertie, prec = _lire_inertie(snap)
    abaques: Dict[str, Dict[int, AbaqueQC]] = {}
    zones: Dict[str, List[Zone]] = {}
    mini: Dict[str, Dict[str, float]] = {}
    articulations: List[List[str]] = []
    for cle, page in (("F", PAGE_ABAQUES_F), ("OB", PAGE_ABAQUES_OB)):
        ab, zo, mi, artic = _lire_abaques(snap, page)
        if not ab or not zo or not mi:
            raise DonneeIntrouvable(f"{page} : abaques, zones et hauteurs mini introuvables")
        abaques[cle], zones[cle], mini[cle] = ab, zo, mi
        if cle == "OB":
            articulations = artic
    zones_s = _lire_abaques(snap, PAGE_SOUFFLET)[1]
    msk = re.search(r"Poids Maxi vantail (\d+) kg", _corps(snap, PAGE_SOUFFLET))
    if not zones_s or not msk:
        raise DonneeIntrouvable(f"{PAGE_SOUFFLET} : zones et poids maxi de l'abaque du soufflet introuvables")
    zones["S"] = zones_s
    # la hauteur mini du vantail est la même sur l'abaque de la française et sur la planche de la ferrure visible
    mini_visible = _une(tables(_corps(snap, PAGE_FERRURES_VISIBLES)), "Châssis | Hv mini (mm)", PAGE_FERRURES_VISIBLES)
    mini["visible"] = {r[0]: _nombre(r[1]) for r in mini_visible.lignes}
    for cle, m in mini.items():
        if {"1 vantail", "2 vantaux"} - set(m):
            raise DonneeIntrouvable(f"hauteurs mini du vantail ({cle}) : lignes « 1 vantail » et « 2 vantaux » introuvables")
    corps_f = _corps(snap, PAGE_ABAQUES_F)
    pa = re.search(r"précision de lecture de ±(\d+) mm", corps_f)
    pd = re.search(r"†[^.]*?lue à ±(\d+) mm", corps_f)
    if not (pa and pd):
        raise DonneeIntrouvable(f"{PAGE_ABAQUES_F} : précisions de lecture des abaques (±… mm, cellule †) introuvables")
    paumelles, cas, pivot = _lire_visible(snap)
    mp = re.search(r"l'ouvrant Minimal est disponible en épaisseur de vitrage de (\d+) et (\d+) mm",
                   _corps(snap, PAGE_PRINCIPE_QC))
    c19 = re.search(r"Pour une hauteur supérieure à (\d+) mm, caler le vitrage au droit des points de verrouillage",
                    _corps(snap, PAGE_PARCLOSES_65))
    c57 = re.search(r"Pour une hauteur supérieure à (\d+) mm ajouter une cale de vitrage de (\d+) mm",
                    _corps(snap, PAGE_CALAGE))
    if not mp:
        raise DonneeIntrouvable(f"{PAGE_PRINCIPE_QC} : épaisseurs de vitrage de l'ouvrant Minimal avec seuil PMR introuvables")
    if not (c19 and c57):
        raise DonneeIntrouvable(f"{PAGE_PARCLOSES_65} et {PAGE_CALAGE} : notes de calage « pour une hauteur supérieure à … mm » "
                                "introuvables")

    apps: Dict[Tuple[str, str, str, str], Application] = {}
    corps_pages: Dict[str, str] = {}

    def corps_de(page: str) -> str:
        if page not in corps_pages:
            corps_pages[page] = _corps(snap, page)
        return corps_pages[page]

    for module in MODULES:
        for ouvrant in OUVRANTS:
            page = f"/profiles/soleal-fy-qc-{module}-applications-ouvrant-{ouvrant.lower()}.md"
            for typ, lib in SECTION_6319.items():
                section = f"{lib}, FY {module}, ouvrant {ouvrant}"
                apps[(G_6319, module, ouvrant, typ)] = lire_application(
                    corps_de(page), page, section, f"{G_6319}-{module}-{ouvrant}-{typ}", f"{TYPOLOGIES[typ]}, FY {module}, "
                    f"ouvrant {ouvrant}")
    for (ouvrant, typ), (page, section) in APPLICATIONS_6057.items():
        apps[(G_6057, "55", ouvrant, typ)] = lire_application(
            corps_de(page), page, section, f"{G_6057}-55-{ouvrant}-{typ}", f"{TYPOLOGIES[typ]}, FY 55, ouvrant {ouvrant}")
    cotes: Dict[str, List[str]] = {}
    for (gen, module, ouvrant, typ), app in apps.items():
        for v in _variables_de_cote(app):
            if typ not in cotes.setdefault(v, []):
                cotes[v].append(typ)

    pages = {PAGE_DTA, PAGE_DTA_AVIS, PAGE_INERTIE, PAGE_ABAQUES_F, PAGE_ABAQUES_OB, PAGE_SOUFFLET, PAGE_PRINCIPE_QC,
             PAGE_VISIBLE, PAGE_FERRURES_VISIBLES, PAGE_PARCLOSES, PAGE_PARCLOSES_65, PAGE_CALAGE, PAGE_VERRE}
    pages |= {a.page for a in apps.values()}
    return Donnees(
        statuts={p: (snap.pages[p].status or "") for p in pages if p in snap.pages},
        dta=dta, dta_localisation=loc_dta, dta_validite=validite, dta_vitrage_max=vit_max, dta_sans_pmr=sans_pmr,
        inertie=inertie, inertie_precision=prec, abaques=abaques, zones=zones, hv_mini=mini, articulations=articulations,
        paumelles=paumelles, cas_paumelles=cas, pivot_sans_pmr=pivot,
        pmr_minimal=tuple(float(x) for x in mp.groups()),
        precision_abaque=float(pa.group(1)), precision_dagger=float(pd.group(1)), soufflet_kg=int(msk.group(1)),
        cale_6319=float(c19.group(1)), cale_6057=(float(c57.group(1)), float(c57.group(2))), apps=apps, cotes=cotes,
        densite_verre=densite_du_verre(snap), images=images_par_reference(snap, "soleal-fy"))


# ---- le formulaire --------------------------------------------------------------------------------


def _id_cote(variable: str) -> str:
    return f"cote_{variable.lower()}_mm"


def _lib_paumelle(ref: str, p: Dict[str, Any]) -> str:
    return f"{ref}, {p['designation']} ({_fmt(p['poids'])} kg, Lv {_fmt(p['lv'])} mm)"


def options(snap: WikiSnapshot) -> Dict[str, Any]:
    d = donnees(snap)
    classes = sorted({c for a in d.abaques.values() for c in a})
    champs: List[Dict[str, Any]] = [
        {"id": "configuration", "libelle": "Typologie", "type": "select", "options": dict(TYPOLOGIES), "defaut": "v2"},
        {"id": "largeur_mm", "libelle": "Largeur hors tout L (W)", "type": "nombre", "min": 300, "max": 10000,
         "defaut": 1200, "unite": "mm"},
        {"id": "hauteur_mm", "libelle": "Hauteur hors tout H", "type": "nombre", "min": 300, "max": 10000,
         "defaut": 1350, "unite": "mm"},
        {"id": "generation", "libelle": "Catalogue de débit", "type": "select", "options": dict(GENERATIONS),
         "defaut": G_6319,
         "aide": "Les deux catalogues portent la même nomenclature FY 55 : le courant par défaut, l'ancien en choix. "
                 "Ils ne sont jamais fusionnés."},
        {"id": "module", "libelle": "Module", "type": "select", "options": dict(MODULES), "defaut": "55",
         "si": {"generation": [G_6319]}, "aide": "Le catalogue 6057 ne traite que le FY 55."},
        {"id": "ouvrant", "libelle": "Ouvrant", "type": "select", "options": dict(OUVRANTS), "defaut": "Apparent"},
        {"id": "epaisseur_vitrage_mm", "libelle": "Épaisseur du double vitrage", "type": "nombre", "min": 4, "max": 70,
         "defaut": 24, "unite": "mm"},
        {"id": "verre_mm", "libelle": "Verre seul (somme des verres, sans lame)", "type": "nombre", "min": 0, "max": 60,
         "defaut": 8, "unite": "mm", "aide": "4-16-4 : 8 mm de verre. Sert au poids du vitrage."},
        {"id": "ferrure_kg", "libelle": "Quincaillerie cachée : poids maxi du vantail", "type": "select",
         "options": {str(c): f"Articulation jusqu'à {c} kg" for c in classes}, "defaut": str(classes[0]),
         "si": {"generation": [G_6319]}},
        {"id": "paumelle", "libelle": "Paumelle (quincaillerie visible)", "type": "select",
         "options": {r: _lib_paumelle(r, p) for r, p in d.paumelles.items()}, "defaut": next(iter(d.paumelles)),
         "si": {"generation": [G_6057]}},
    ]
    for v in sorted(d.cotes):
        champs.append({"id": _id_cote(v), "libelle": f"Cote {v} portée sur la coupe", "type": "nombre", "min": 0,
                       "max": 10000, "unite": "mm", "requis": False, "si": {"configuration": sorted(d.cotes[v])},
                       "aide": "Cote de l'ensemble menuisé telle que la planche la nomme ; les relations entre L, L1, L2… "
                               "ne sont pas écrites, elles sont prises telles que saisies."})
    brouillons = sorted(p for p, s in d.statuts.items() if s == "draft")
    validite = _validite(d)[0]
    return {
        "champs": champs, "configurations": dict(TYPOLOGIES), "pages": sorted(d.statuts), "brouillons": brouillons,
        "avertissements": [
            f"{validite}. Il ne couvre que le SOLEAL FY 55 : le FY 65 n'a pas de DTA.",
            "Les abaques de quincaillerie cachée donnent une valeur en kg/m² dont le wiki ne dit pas ce qu'elle mesure "
            "(VER-406) : elle est lue, jamais comparée au poids de verre.",
            "Le wiki ne donne pas le poids des profilés ni la surlongueur de soudure ou d'onglet : le poids est celui du "
            "verre seul, les longueurs de coupe sont droites.",
            "Le nombre de paumelles de la quincaillerie visible n'est pas calculable : les bandes de kg/m² des abaques "
            "ne sont pas relevées (VER-370).",
        ],
    }


# ---- ce que la saisie désigne -----------------------------------------------------------------------


def _saisie(snap: WikiSnapshot, saisie: Dict[str, Any]) -> Dict[str, Any]:
    s = coercer(options(snap)["champs"], saisie)
    if s["generation"] == G_6057:
        s["module"] = "55"
    s["nb_vantaux"] = NB_VANTAUX[s["configuration"]]
    return s


def _variables(d: Donnees, s: Dict[str, Any]) -> Dict[str, float]:
    v = {"H": s["hauteur_mm"], "L": s["largeur_mm"], "W": s["largeur_mm"]}
    for nom in d.cotes:
        valeur = s.get(_id_cote(nom))
        if valeur is not None:
            v[nom] = valeur
    return v


def _application(d: Donnees, s: Dict[str, Any], generation: Optional[str] = None) -> Optional[Application]:
    gen = generation or s["generation"]
    module = s["module"] if gen == G_6319 else "55"
    return d.apps.get((gen, module, s["ouvrant"], s["configuration"]))


def _brouillons(d: Donnees, pages: List[str]) -> List[str]:
    return [p for p in dict.fromkeys(pages) if d.statuts.get(p) == "draft"]


def _pire(*statuts: str) -> str:
    return max(statuts, key=[OK, INFO, ETUDE, HORS].index)


def _nb(x: float) -> str:
    return _fmt(x, 1 if x % 1 else 0)


def _vantail(app: Application, variables: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Hv et Wv : les longueurs de coupe des profilés d'ouvrant (hypothèse), la ligne en hauteur et la ligne en largeur
    du profilé nommé « Ouvrant… » (hors traverse basse de seuil PMR, « … L 1500 mm »)."""
    hv = wv = None
    for l in app.profils:
        if not l.designation.lower().startswith("ouvrant") or re.search(r"L \d+ mm", l.designation):
            continue
        valeur, _ = valeur_ou_none(l.debit, variables)
        if valeur is None:
            continue
        if re.match(r"H\d*\b", l.debit.strip()) and hv is None:
            hv = valeur
        elif re.match(r"[LW]\d*\b", l.debit.strip()) and wv is None:
            wv = valeur
    return {"hv": hv, "wv": wv}


# ---- les contrôles -----------------------------------------------------------------------------------


def _ecart(a: Application, b: Application, nom_a: str, nom_b: str) -> List[str]:
    """Ce qui change entre deux nomenclatures de la même application : références seules d'un côté, formules ou
    quantités qui diffèrent pour une même référence. Variable W lue comme L (les deux catalogues nomment la largeur
    autrement)."""
    def norm(f: str) -> str:
        return re.sub(r"(?<![A-Za-z])W(?![A-Za-z])", "L", f).replace(" ", "")

    def cles(app: Application, genre: str) -> Dict[str, Counter]:
        out: Dict[str, Counter] = {}
        for l in getattr(app, genre):
            out.setdefault(l.ref, Counter())[(norm(l.debit), norm(l.qte))] += 1
        return out

    noms = {"profils": "profilés", "joints": "joints", "accessoires": "accessoires", "vis": "vis"}
    sortie: List[str] = []
    for genre, nom in noms.items():
        ca, cb = cles(a, genre), cles(b, genre)
        seul_a, seul_b = sorted(set(ca) - set(cb)), sorted(set(cb) - set(ca))
        diff = sorted(r for r in set(ca) & set(cb) if ca[r] != cb[r])
        if seul_a:
            sortie.append(f"{nom} seulement en {nom_a} : {', '.join(seul_a)}")
        if seul_b:
            sortie.append(f"{nom} seulement en {nom_b} : {', '.join(seul_b)}")
        if diff:
            sortie.append(f"{nom} dont la formule ou la quantité diffère : {', '.join(diff)}")
    ra = [(norm(r["hauteur"]), norm(r["largeur"])) for r in a.remplissages]
    rb = [(norm(r["hauteur"]), norm(r["largeur"])) for r in b.remplissages]
    if ra != rb:
        sortie.append("remplissage : " + " ; ".join(f"{h} × {l}" for h, l in ra) + f" en {nom_a} contre "
                      + " ; ".join(f"{h} × {l}" for h, l in rb) + f" en {nom_b}")
    return sortie


def _controle_generation(d: Donnees, s: Dict[str, Any], app: Optional[Application]) -> Dict[str, Any]:
    gen = s["generation"]
    autre = G_6057 if gen == G_6319 else G_6319
    nom, nom_autre = NOMS_GENERATION[gen], NOMS_GENERATION[autre]
    typ = s["configuration"]
    src = [_source(app.page, app.section, app.localisation)] if app else []
    parties = [f"Catalogue {nom}" + (" : " + app.libelle if app else "") + "."]
    if gen == G_6319 and typ in ("v1_ob", "v2_ob", "soufflet"):
        parties.append("La planche de ce catalogue sert la française, le soufflet et l'OB / BO (trois pictogrammes en tête "
                       "de planche) : la nomenclature est celle du châssis, sans ligne propre à cette ouverture.")
    # le catalogue 6057 ne traite que le FY 55 : il n'a pas de contrepartie d'un module 65
    contre = d.apps.get((autre, "55", s["ouvrant"], typ)) if s["module"] == "55" else None
    ecarts: List[str] = []
    if app is None:
        parties.append("Le wiki n'a pas de nomenclature de cette typologie dans ce catalogue"
                       + (f" ; le catalogue {nom_autre} en a une, à choisir explicitement." if contre is not None else "."))
    if contre is None:
        parties.append(f"Le catalogue {nom_autre} n'a pas cette application pour ce module et cet ouvrant.")
    elif app is not None:
        src.append(_source(contre.page, contre.section, contre.localisation))
        ecarts = _ecart(app, contre, gen, autre)
        parties.append(f"Même application dans le catalogue {nom_autre} : "
                       + ("; ".join(ecarts) if ecarts else "mêmes références, mêmes formules et mêmes quantités")
                       + ". Les deux catalogues ne sont jamais fusionnés : seule la nomenclature choisie sert au calcul.")
    anomalies = ["VER-396" if gen == G_6319 else "VER-352"]
    if typ in PMR:
        anomalies += ["CTR-220", "CTR-208" if s["ouvrant"] == "Apparent" else "CTR-209"]
        parties.append("La page des cotes de débit du catalogue 6057 porte aussi des valeurs « de l'ancienne rédaction » du "
                       "seuil PMR, contredites par les planches : elles ne sont pas lues.")
    return _controle("generation", "Génération du catalogue", INFO, " ".join(parties), src, anomalies,
                     {"generation": gen, "ecarts": ecarts})


def _validite(d: Donnees) -> Tuple[str, str, Dict[str, Any]]:
    """(texte, statut plancher, valeurs) de la validité du DTA, contre la date du jour."""
    debut, fin = d.dta_validite
    jours = (fin - _aujourdhui()).days
    base = f"DTA 6/12-2016_V5 valide du {debut.strftime('%d/%m/%Y')} au {fin.strftime('%d/%m/%Y')}"
    valeurs = {"du": debut.isoformat(), "au": fin.isoformat(), "jours_restants": jours, "expire": jours < 0}
    if jours < 0:
        return (base + f" : échu le {fin.strftime('%d/%m/%Y')} ; le wiki ne porte pas de version plus récente, la "
                "limite du tableau n'est plus garantie par un avis en cours", ETUDE, valeurs)
    if jours <= JOURS_AVANT_ECHEANCE:
        return base + f" : expire dans {jours} jour(s), le {fin.strftime('%d/%m/%Y')}", OK, valeurs
    return base, OK, valeurs


def _controle_dta(d: Donnees, s: Dict[str, Any]) -> Dict[str, Any]:
    titre = "Dimensions maximales de baie (DTA)"
    typ, module, ouvrant = s["configuration"], s["module"], s["ouvrant"]
    src = [_source(PAGE_DTA, "Dimensions maximales de baie", d.dta_localisation), _source(PAGE_DTA_AVIS, "")]
    anomalies = ["INC-402"] + (["INC-403"] if ouvrant == "Minimal" else [])
    validite, plancher, vvaleurs = _validite(d)
    if module == "65":
        return _controle("dta", titre, ETUDE,
                         "Aucun DTA pour le SOLEAL FY 65 : le dossier technique ne couvre que le SOLEAL 55 FYa - FYm, la "
                         "limite de baie n'existe pas. " + validite + " ; il ne couvre que le FY 55.", src, anomalies + ["VER-392"],
                         {"validite": vvaleurs})
    if typ in PMR:
        texte = ("Le DTA ne dispose pas d'une solution de seuil permettant l'accès des handicapés (rubrique 1.2.1.11) : "
                 if d.dta_sans_pmr else "Le seuil PMR n'est pas traité par le DTA : ")
        return _controle("dta", titre, ETUDE, texte + "aucune limite de baie ne vaut pour un châssis à seuil PMR, que "
                         "les planches de fabrication décrivent pourtant. " + validite + ".", src, anomalies + ["CTR-141"],
                         {"validite": vvaleurs})
    type_dta = DTA_TYPE.get(typ)
    ligne = d.dta.get((type_dta, DTA_OUVRANT[ouvrant])) if type_dta else None
    if ligne is None:
        return _controle("dta", titre, ETUDE,
                         f"Le tableau du DTA n'a pas de ligne pour « {TYPOLOGIES[typ].lower()} » : ni limite de baie, ni "
                         f"avis. Au-delà, seul le Certificat de Qualification du menuisier peut fixer la limite. {validite}.",
                         src, anomalies, {"validite": vvaleurs})
    h_max, l_max = ligne
    H, L = s["hauteur_mm"], s["largeur_mm"]
    dans = H <= h_max and L <= l_max
    detail = (f"Baie {_fmt(H / 1000, 3)} × {_fmt(L / 1000, 3)} m (H × L), limite {_fmt(h_max / 1000, 3)} × "
              f"{_fmt(l_max / 1000, 3)} m pour « {type_dta} » en Soleal 55 FY{DTA_OUVRANT[ouvrant]} ({HYPOTHESE_BAIE}) : "
              + ("dans la limite." if dans else "dépassée ; pour une fabrication certifiée, des dimensions supérieures "
                 "peuvent figurer au Certificat de Qualification du menuisier, que le wiki ne porte pas.")
              + " " + validite + ". Le DTA ne couvre que le FY 55.")
    statut = _pire(OK if dans else HORS, plancher)
    return _controle("dta", titre, statut, detail, src, anomalies,
                     {"limite": {"h": h_max, "l": l_max}, "type_dta": type_dta, "dans": dans, "validite": vvaleurs,
                      "hypothese": HYPOTHESE_BAIE})


def _lire_courbe_inertie(courbe: Dict[int, Optional[float]], largeur: float, hauteur: float, precision: float):
    """Verdict d'une hauteur sur une courbe relevée aux graduations de W, **sans interpolation** : entre deux graduations
    la plus restrictive ; une graduation en « - » (courbe non tracée) n'est jamais étendue ; à moins de la précision de
    lecture de la limite, « sur étude »."""
    grad = sorted(courbe)
    if largeur < grad[0] or largeur > grad[-1]:
        return ETUDE, None, (f"largeur de {_nb(largeur)} mm hors des graduations de l'abaque ({_nb(grad[0])} à "
                             f"{_nb(grad[-1])} mm)")
    bas = max(g for g in grad if g <= largeur)
    haut = min(g for g in grad if g >= largeur)
    ou = f"à W = {_nb(bas)} mm" if bas == haut else f"entre W = {_nb(bas)} et {_nb(haut)} mm, la plus restrictive"
    manquantes = [g for g in {bas, haut} if courbe[g] is None]
    if manquantes:
        return ETUDE, None, (f"la courbe n'est pas tracée à W = {_nb(min(manquantes))} mm (« - ») : aucune valeur n'est "
                             "étendue au-delà du tracé")
    lim = min(courbe[bas], courbe[haut])
    return _borne(hauteur, lim, precision), lim, f"H maxi lue {_nb(lim)} mm {ou} (précision de lecture ±{_nb(precision)} mm)"


def _controle_inertie(d: Donnees, s: Dict[str, Any], dta: Dict[str, Any]) -> Dict[str, Any]:
    titre = "Inertie des montants du châssis 2 vantaux"
    module, ouvrant = s["module"], s["ouvrant"]
    cle = (ouvrant, f"FY{module}")
    c = d.inertie.get(cle)
    if c is None:
        return _controle("inertie", titre, ETUDE, f"Aucune courbe pour le module FY {module}, ouvrant {ouvrant}.",
                         [_source(PAGE_INERTIE)])
    statut, lim, texte = _lire_courbe_inertie(c["valeurs"], s["largeur_mm"], s["hauteur_mm"], d.inertie_precision)
    detail = (f"Courbe {c['courbe']} (FY {module}, ouvrant {ouvrant}), montants {c['montants']}, flèche inférieure à 15 mm : "
              f"hauteur {_nb(s['hauteur_mm'])} mm pour W = {_nb(s['largeur_mm'])} mm. {texte[0].upper() + texte[1:]}.")
    anomalies = ["VER-392"] if module == "65" else []
    if dta["valeurs"].get("dans"):
        # le DTA borne déjà la baie : la courbe ne décide que hors DTA
        detail += (" La baie est dans les limites du DTA, qui borne déjà L : la courbe d'inertie ne décide pas dans ce "
                   "domaine, elle est donnée à titre d'information.")
        statut = INFO
    return _controle("inertie", titre, statut, detail, [_source(PAGE_INERTIE, c["section"], c["localisation"])], anomalies,
                     {"h_max": lim, "courbe": c["courbe"], "montants": c["montants"], "module": module})


# -- le vitrage


def _resume_solution(sol: Dict[str, Any]) -> str:
    joint = next((e["valeur"] for e in sol.get("elements", []) if e["role"] == "Joint intérieur"), None)
    return sol["ref"] + (f" (joint {joint})" if joint else "")


def _controle_vitrage(d: Donnees, s: Dict[str, Any], snap: WikiSnapshot) -> Dict[str, Any]:
    titre = "Vitrage, parclose et joint"
    n, verre = float(s["epaisseur_vitrage_mm"]), s["verre_mm"] or 0
    module, ouvrant, typ = s["module"], s["ouvrant"], s["configuration"]
    section = ("Matrice des parcloses droites et arrondies" if ouvrant == "Apparent"
               else "Vitrage de l'Ouvrant Minimal (FYm / OM)")
    src = [_source(PAGE_PARCLOSES, section), _source(PAGE_DTA, "Vitrages isolants admis (2.2.2.7)")]
    anomalies = ["CTR-142"] + ((["VER-395"] if ouvrant == "Apparent" else ["CTR-218"]) if module == "65" else [])
    if verre >= n:
        return _controle("vitrage", titre, ETUDE, f"Le verre seul ({_nb(verre)} mm) n'est pas inférieur à l'épaisseur du "
                                                  f"vitrage ({_nb(n)} mm).", src, anomalies)
    res = parcloses.chercher(snap, epaisseur=n)
    systeme = f"SOLEAL FY {module}"
    contexte = "Ouvrant apparent et fixe" if ouvrant == "Apparent" else "Ouvrant minimal (masqué)"
    sols = [x for g in res["groupes"] if g["systeme"] == systeme for c in g["contextes"] if c["contexte"] == contexte
            for x in c["solutions"]]
    ok = [x for x in sols if x["statut"] == "ok"]
    proches = [x for x in sols if x["statut"] == "proche"]
    statuts: List[str] = []
    parties: List[str] = []
    if ok:
        statuts.append(OK)
        parties.append(f"Double vitrage de {_nb(n)} mm, ouvrant {ouvrant} : " + ", ".join(_resume_solution(x) for x in ok[:4])
                       + (f" et {len(ok) - 4} autre(s)" if len(ok) > 4 else "") + " (prise de volume franche, la page "
                       "des parcloses et vitrages donne la matrice).")
    elif module == "65" and ouvrant == "Apparent":
        statuts.append(ETUDE)
        src.append(_source(PAGE_PARCLOSES_65, "Parcloses droites de l'ouvrant Apparent et de la partie fixe FY 65"))
        parties.append(f"Double vitrage de {_nb(n)} mm : la matrice des parcloses de l'ouvrant Apparent FY 65 n'est pas lue "
                       "par le calcul (tableau HTML de la page des parcloses FY 65).")
    elif proches:
        statuts.append(ETUDE)
        parties.append(f"Double vitrage de {_nb(n)} mm, ouvrant {ouvrant} : aucune prise de volume franche ; la plus proche "
                       + ", ".join(_resume_solution(x) for x in proches[:3]) + ".")
    else:
        statuts.append(HORS if module == "55" else ETUDE)
        parties.append(f"Double vitrage de {_nb(n)} mm, ouvrant {ouvrant} : aucune prise de volume de {systeme} ne la reçoit "
                       "dans le wiki.")
    if n > d.dta_vitrage_max:
        statuts.append(ETUDE)
        parties.append(f"Le DTA admet les doubles vitrages jusqu'à {_nb(d.dta_vitrage_max)} mm : au-delà, seul le catalogue "
                       "(plus large) les prévoit (CTR-142).")
    if typ in PMR and ouvrant == "Minimal" and n not in d.pmr_minimal:
        statuts.append(ETUDE)
        parties.append("Avec seuil PMR, l'ouvrant Minimal est disponible en " + " et ".join(_nb(x) for x in d.pmr_minimal)
                       + " mm.")
        src.append(_source(PAGE_PRINCIPE_QC, "Le châssis à seuil PMR"))
    return _controle("vitrage", titre, _pire(*statuts), " ".join(parties), src, anomalies,
                     {"epaisseur": n, "verre_mm": verre, "solutions": [x["ref"] for x in ok]})


# -- la cale de vitrage


def _controle_cale(d: Donnees, s: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Une cale de vitrage supplémentaire au-delà d'une hauteur : les deux catalogues la prescrivent autrement (CTR-219)."""
    if NB_VANTAUX[s["configuration"]] == 0:
        return []
    H, gen = s["hauteur_mm"], s["generation"]
    titre = "Cale de vitrage supplémentaire"
    if gen == G_6319:
        seuil, src = d.cale_6319, [_source(PAGE_PARCLOSES_65, "Règles de calage")]
        txt = (f"Au-delà de {_nb(seuil)} mm de hauteur, le catalogue 6319 prescrit de caler le vitrage au droit des points de "
               "verrouillage côté boîtier et articulation, en française comme en OB / BO (versions Apparent et Minimal).")
    else:
        seuil, src = d.cale_6057[0], [_source(PAGE_CALAGE, "Calage")]
        txt = (f"Au-delà de {_nb(seuil)} mm de hauteur, le catalogue 6057 prescrit, pour la version Minimale seulement, une cale "
               f"de vitrage de {_nb(d.cale_6057[1])} mm au droit de la poignée et du point de condamnation côté paumelles sur "
               "un OB et semi-fixe, sauf traverse intermédiaire.")
    if H <= seuil:
        return []
    return [_controle("cale", titre, INFO, f"Hauteur hors tout {_nb(H)} mm (le wiki ne dit pas s'il s'agit de H ou de Hv). {txt} "
                      "Les deux catalogues ne prescrivent pas la même cale selon la quincaillerie (CTR-219) : celui du "
                      "catalogue choisi est donné, l'autre n'est pas mélangé.", src, ["CTR-219"], {"seuil_mm": seuil})]


# -- le poids de verre

def _controle_poids(d: Donnees, s: Dict[str, Any], dim: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[float]]:
    poids, raison = poids_du_verre(d.densite_verre, s["verre_mm"], dim)
    titre = "Poids de verre du vantail"
    src = [_source(PAGE_VERRE, "Remarques importantes concernant les abaques de renforcement")]
    if poids is None:
        return _controle("poids", titre, ETUDE, f"Poids de verre non calculé : {raison}.", src), None
    txt = (f"{_nb(round(poids, 1))} kg de verre pour le plus petit vitrage ({_nb(s['verre_mm'])} mm × {_fmt(d.densite_verre, 1)} "
           "kg/m² par mm, hypothèse transposée de la page des abaques du système 70 : aucune page du SOLEAL FY ne donne la "
           "masse du verre). Le wiki ne donne pas le poids des profilés : c'est une borne basse du poids du vantail, pas "
           "son poids.")
    return _controle("poids", titre, INFO, txt, src, ["VER-190"], {"poids_verre_kg": round(poids, 1)}), poids


# -- la quincaillerie cachée

def _lire_champ(d: Donnees, ab: AbaqueQC, wv: float, hv: float, hv_mini: float) -> Tuple[str, str, List[float]]:
    """(statut, détail, valeurs kg/m² lues) : le vantail Wv × Hv est-il dans le champ de l'abaque ? Jamais
    d'interpolation : entre deux graduations de Wv, le plus restrictif des deux. Hors champ, la zone est blanche."""
    grad = sorted(ab.lignes)
    if wv < grad[0] or wv > grad[-1]:
        return HORS, (f"Wv = {_nb(wv)} mm hors de l'abaque ({_nb(grad[0])} à {_nb(grad[-1])} mm)"), []
    bas = max(g for g in grad if g <= wv)
    haut = min(g for g in grad if g >= wv)
    rows = sorted({bas, haut})
    axe = min(c.lo for _, champs in ab.lignes.values() for c in champs)      # « 520 est le bas de l'axe »
    hi = min(ab.lignes[g][0] for g in rows)
    lo, prec_lo = 0.0, d.precision_abaque
    for g in rows:
        champs = ab.lignes[g][1]
        if not champs:
            continue
        c_bas = min(champs, key=lambda c: c.lo)
        if c_bas.lo > lo:
            lo, prec_lo = c_bas.lo, d.precision_dagger if c_bas.dagger else d.precision_abaque
    s_haut = _borne(hv, hi, d.precision_abaque)
    if lo <= axe:
        s_bas = OK if hv >= lo else HORS
    else:
        s_bas = OK if hv >= lo + prec_lo else (ETUDE if hv > lo - prec_lo else HORS)
    s_mini = OK if hv >= hv_mini else HORS
    statut = _pire(s_haut, s_bas, s_mini)
    ou = f"à Wv = {_nb(bas)} mm" if bas == haut else f"entre Wv = {_nb(bas)} et {_nb(haut)} mm, le plus restrictif"
    detail = (f"champ de l'abaque {ou} : Hv de {_nb(max(lo, hv_mini))} à {_nb(hi)} mm (précision de lecture "
              f"±{_nb(d.precision_abaque)} mm)")
    valeurs: List[float] = []
    for g in rows:
        for c in ab.lignes[g][1]:
            p = d.precision_dagger if c.dagger else d.precision_abaque
            if c.kg_m2 is not None and c.lo - p <= hv <= c.hi + p:
                valeurs.append(c.kg_m2)
    return statut, detail, sorted(set(valeurs))


def _zones_de(zones: List[Zone], wv: float, hv: float) -> List[Zone]:
    return [z for z in zones if z.wv[0] <= wv <= z.wv[1] and z.hv[0] <= hv <= z.hv[2]]


def _classe_articulation(d: Donnees, classe: int) -> str:
    for r in d.articulations:
        m = re.search(r"(\d+)", r[0])
        if m and int(m.group(1)) == classe:
            return f"articulation droite {r[1]}, gauche {r[2]}, ouverture à {r[3]}°"
    return ""


def _controles_qc(d: Donnees, s: Dict[str, Any], app: Application, variables: Dict[str, float],
                  poids: Optional[float]) -> List[Dict[str, Any]]:
    typ, module = s["configuration"], s["module"]
    kind = ABAQUE_DE.get(typ)
    if kind is None:
        return []
    titre = "Quincaillerie cachée"
    if module == "65":
        return [_controle("ferrure", titre, ETUDE,
                          "La quincaillerie cachée n'est pas disponible en LUMINE65 : le catalogue la décrit pour les "
                          "modules 55 et 65, mais l'offre PROFERM ne la propose pas en 65. Aucun contrôle de ferrure.",
                          [_source(PAGE_PRINCIPE_QC, "La quincaillerie cachée QC du SOLEAL FY")], ["VER-392"])]
    vt = _vantail(app, variables)
    hv, wv = vt["hv"], vt["wv"]
    if hv is None or wv is None:
        return [_controle("ferrure", titre, ETUDE, "Hv ou Wv du vantail non calculable à partir de la nomenclature (formule "
                          "illisible ou cote manquante).", [_source(app.page, app.section, app.localisation)])]
    nb = s["nb_vantaux"] or 1
    hvm = d.hv_mini["F" if kind == "S" else kind].get("2 vantaux" if nb == 2 else "1 vantail")
    base = f"Vantail Hv = {_nb(hv)} mm, Wv = {_nb(wv)} mm ({HYPOTHESE_VANTAIL})."
    out: List[Dict[str, Any]] = []
    if kind == "S":
        zones = d.zones["S"]
        src = [_source(PAGE_SOUFFLET, "Abaque d'utilisation pour soufflet")]
        zs = [z for z in zones if z.wv[0] <= wv <= z.wv_abaque]
        z = zs[-1] if zs else None
        titre_champ = "Quincaillerie cachée : champ de l'abaque"
        if z is None:
            hors = wv < min(x.wv[0] for x in zones) or wv > max(x.wv_abaque for x in zones)
            out.append(_controle("ferrure_champ", titre_champ, HORS if hors else ETUDE,
                                 base + (" Wv hors de l'abaque du soufflet." if hors else
                                         " Wv entre deux zones de l'abaque du soufflet : la planche ne dit pas laquelle."),
                                 src, ["INC-619"]))
        else:
            if hv < z.hv[0] or hv > z.hv[2]:
                statut, det = HORS, f"Hv hors du champ de la zone {z.lettre} ({_nb(z.hv[0])} à {_nb(z.hv[2])} mm)"
            elif hv > z.hv[1]:
                statut, det = ETUDE, (f"la limite haute de la zone {z.lettre} descend en ligne droite de {_nb(z.hv[2])} à "
                                      f"{_nb(z.hv[1])} mm, la valeur intermédiaire n'est pas relevée")
            elif wv > z.wv[1]:
                statut, det = ETUDE, (f"Wv au-delà de la borne {_nb(z.wv[1])} du titre des planches, dans l'abaque jusqu'à "
                                      f"{_nb(z.wv_abaque)} mm")
            else:
                statut, det = OK, f"dans le champ de la zone {z.lettre}"
            out.append(_controle("ferrure_champ", titre_champ, statut, f"{base} Abaque soufflet : {det}.", src,
                                 ["INC-619", "VER-407"], {"zone": z.lettre, "wv": wv, "hv": hv}))
            out.append(_controle("ferrure_zone", "Quincaillerie cachée : zone et fermeture", INFO,
                                 f"Zone {z.lettre} : {z.points}.", src, ["INC-619"], {"zone": z.lettre}))
        classe = d.soufflet_kg
    else:
        classe = int(s["ferrure_kg"])
        ab = d.abaques[kind].get(classe)
        page = PAGE_ABAQUES_F if kind == "F" else PAGE_ABAQUES_OB
        if ab is None:
            return [_controle("ferrure", titre, ETUDE, f"Aucun abaque de {classe} kg.", [_source(page)])]
        src = [_source(page, ab.section, ab.localisation)]
        statut, detail, kg = _lire_champ(d, ab, wv, hv, hvm)
        out.append(_controle("ferrure_champ", "Quincaillerie cachée : champ de l'abaque", statut,
                             f"{base} Abaque {classe} kg ({'fenêtre française' if kind == 'F' else 'OB / BO'}) : {detail} ; "
                             f"Hv mini {_nb(hvm)} mm pour {'2 vantaux' if nb == 2 else '1 vantail'}.", src,
                             ["VER-406"] + (["INC-601"] if classe == 160 else []), {"hv": hv, "wv": wv, "classe": classe}))
        zs = _zones_de(d.zones[kind], wv, hv)
        zone_txt = ("Zone " + " ou ".join(f"{z.lettre} ({z.points})" for z in zs) + "." if zs
                    else "Aucune zone ne porte ces cotes.")
        suffixe = (" À la frontière de deux bandes : la planche ne départage pas, prendre la plus exigeante."
                   if len(zs) > 1 else "")
        valeur = ("aucune valeur lue" if not kg else
                  f"{_nb(kg[0])} kg/m²" if kg[0] == kg[-1] else f"de {_nb(kg[0])} à {_nb(kg[-1])} kg/m²")
        out.append(_controle("ferrure_zone", "Quincaillerie cachée : zone, fermeture et valeur de l'abaque", INFO,
                             f"{zone_txt}{suffixe} Valeur lue sur l'abaque : {valeur}. Le wiki ne dit pas ce que cette "
                             "valeur mesure (glossaire, VER-406) : elle n'est pas comparée au poids de verre.", src,
                             ["VER-406"], {"zones": [z.lettre for z in zs], "kg_m2": kg}))
    # la classe de poids de l'articulation face au poids de verre
    art = (_classe_articulation(d, classe) if kind != "S"
           else "articulations hautes et compas du soufflet : voir la page du soufflet")
    if poids is None:
        out.append(_controle("ferrure", titre, ETUDE, "Poids de verre non calculé : classe de poids non contrôlée.",
                             [_source(PAGE_ABAQUES_OB, "Concept des articulations")]))
    elif poids > classe:
        suffisantes = sorted(c for c in {c for a in d.abaques.values() for c in a} if c >= poids)
        out.append(_controle("ferrure", titre, HORS,
                             f"{_nb(round(poids, 1))} kg de verre dépassent la classe de {classe} kg de la quincaillerie "
                             f"cachée (classes dont le poids maxi suffit au verre seul : "
                             f"{', '.join(str(c) for c in suffisantes) or 'aucune'}).",
                             [_source(PAGE_ABAQUES_OB, "Concept des articulations")], ["INC-601"] if classe == 160 else [],
                             {"classe": classe, "poids_verre_kg": round(poids, 1)}))
    else:
        out.append(_controle("ferrure", titre, INFO,
                             f"Classe de {classe} kg : {art or 'références non lues'}. {_nb(round(poids, 1))} kg de verre "
                             "sous la classe ; le wiki ne donne pas le poids des profilés, le verre seul ne prouve pas que "
                             "le vantail la respecte.", [_source(PAGE_ABAQUES_OB, "Concept des articulations")],
                             ["INC-601"] if classe == 160 else [], {"classe": classe, "poids_verre_kg": round(poids, 1)}))
    return out


# -- la quincaillerie visible

def _controles_visible(d: Donnees, s: Dict[str, Any], app: Application, variables: Dict[str, float],
                       poids: Optional[float]) -> List[Dict[str, Any]]:
    ref = s["paumelle"]
    p = d.paumelles[ref]
    src = [_source(PAGE_VISIBLE, p["section"], p["localisation"])]
    vt = _vantail(app, variables)
    hv, wv = vt["hv"], vt["wv"]
    titre = "Quincaillerie visible : paumelle"
    anomalies = ["VER-190", "VER-370"] + (["VER-372"] if ref == "T940023" else []) + (["VER-371"] if ref == "T940035" else [])
    typ = s["configuration"]
    if typ in PMR:
        anomalies.append("VER-369")
    statuts, parties = [], [f"Paumelle {_lib_paumelle(ref, p)}."]
    if wv is None or hv is None:
        statuts.append(ETUDE)
        parties.append("Hv ou Wv du vantail non calculable à partir de la nomenclature.")
    else:
        parties.append(f"Vantail Hv = {_nb(hv)} mm, Wv = {_nb(wv)} mm ({HYPOTHESE_VANTAIL}).")
        if wv > p["lv"]:
            statuts.append(HORS)
            parties.append(f"Wv dépasse la largeur maxi Lv de la paumelle ({_nb(p['lv'])} mm).")
        else:
            statuts.append(INFO)
            parties.append(f"Wv sous la largeur maxi Lv ({_nb(p['lv'])} mm).")
        hvm = d.hv_mini["visible"].get("2 vantaux" if s["nb_vantaux"] == 2 else "1 vantail")
        if s["nb_vantaux"] and hvm is not None and typ != "soufflet":
            if hv < hvm:
                statuts.append(HORS)
                parties.append(f"Hv inférieure à la hauteur mini du vantail ({_nb(hvm)} mm).")
    if poids is None:
        statuts.append(ETUDE)
        parties.append("Poids de verre non calculé : poids maxi non contrôlé.")
    elif poids > p["poids"]:
        statuts.append(HORS)
        parties.append(f"{_nb(round(poids, 1))} kg de verre dépassent le poids maxi du vantail ({_nb(p['poids'])} kg).")
    else:
        statuts.append(INFO)
        parties.append(f"{_nb(round(poids, 1))} kg de verre sous le poids maxi ({_nb(p['poids'])} kg) ; le poids des "
                       "profilés n'est pas dans le wiki, le verre seul ne prouve pas que le vantail le respecte.")
    if p["pivot"] and typ in PMR and d.pivot_sans_pmr:
        statuts.append(HORS)
        parties.append("Les paumelles sur pivot ne sont pas compatibles avec le seuil PMR.")
    ctl = [_controle("paumelle", titre, _pire(*statuts), " ".join(parties), src, anomalies,
                     {"paumelle": ref, "poids_maxi_kg": p["poids"], "lv_maxi_mm": p["lv"], "hv": hv, "wv": wv})]
    cas = d.cas_paumelles.get(ref, [])
    ecrit = "; ".join(f"cas {c} : {q} {r}" for c, r, q in cas)
    ctl.append(_controle("paumelles_nombre", "Quincaillerie visible : nombre de paumelles", INFO,
                         "Le nombre de paumelles n'est pas calculé : il dépend du cas 1, 2 ou 3 lu sur un abaque dont les "
                         "bandes de kg/m² ne sont pas relevées (VER-370) et dont la limite « hors préconisation » n'est pas "
                         "relevée non plus." + (f" Quantités écrites par cas : {ecrit}." if ecrit else ""), src,
                         ["VER-370"], {"cas": [list(c) for c in cas]}))
    return ctl


# ---- la faisabilité ---------------------------------------------------------------------------------


def verifier(snap: WikiSnapshot, saisie: Dict[str, Any]) -> Dict[str, Any]:
    d = donnees(snap)
    s = _saisie(snap, saisie)
    typ, gen = s["configuration"], s["generation"]
    app = _application(d, s)
    variables = _variables(d, s)
    ctl = [_controle_generation(d, s, app), _controle_dta(d, s)]
    pages = [PAGE_DTA, PAGE_DTA_AVIS, PAGE_PARCLOSES, PAGE_PARCLOSES_65 if gen == G_6319 else PAGE_CALAGE]
    if typ in DEUX_VANTAUX:
        ctl.append(_controle_inertie(d, s, ctl[1]))
        pages.append(PAGE_INERTIE)
    ctl.append(_controle_vitrage(d, s, snap))
    ctl += _controle_cale(d, s)
    extra: Dict[str, Any] = {}
    if app is None:
        ctl.append(_controle("application", "Nomenclature et dimensions du vantail", ETUDE,
                             "Le wiki n'a pas de nomenclature de cette typologie dans ce catalogue : dimensions du vantail, "
                             "vitrage, poids et quincaillerie non calculés.", []))
    else:
        manquantes = [v for v in _variables_de_cote(app) if v not in variables]
        if manquantes:
            ctl.append(_controle("cotes", "Cotes de l'ensemble menuisé", ETUDE,
                                 f"Cotes non saisies : {', '.join(manquantes)} ; les vitrages et le vantail ne se calculent "
                                 "pas sans elles.", [_source(app.page, app.section, app.localisation)], ["VER-360"]))
        dim = dimensions(app, variables)
        poids_ctl, poids = _controle_poids(d, s, dim)
        pages.append(app.page)
        if NB_VANTAUX[typ] != 0:
            ctl.append(poids_ctl)
            if gen == G_6319:
                ctl += _controles_qc(d, s, app, variables, poids)
                pages += [PAGE_ABAQUES_F, PAGE_ABAQUES_OB, PAGE_SOUFFLET, PAGE_PRINCIPE_QC]
            elif NB_VANTAUX[typ] is None:
                ctl.append(_controle("paumelle", "Quincaillerie visible : paumelle", ETUDE,
                                     "Les vitrages d'un ensemble menuisé ne portent pas de repère de position (VER-360) : le "
                                     "vantail et son vitrage ne sont pas identifiables, la paumelle n'est pas contrôlée.",
                                     [_source(app.page, app.section, app.localisation)], ["VER-360"]))
            else:
                ctl += _controles_visible(d, s, app, variables, poids)
                pages.append(PAGE_VISIBLE)
        extra = {"vantail": _vantail(app, variables),
                 "remplissages": [{"libelle": r["libelle"], "hauteur": r["h"], "largeur": r["l"]} for r in dim["remplissages"]],
                 "poids_verre_kg": None if poids is None else round(poids, 1)}
    return wiki_famille.relocaliser(snap, synthese(
        ctl, _brouillons(d, pages), application=app.libelle if app else TYPOLOGIES[typ], generation=gen,
        module=s["module"], ouvrant=s["ouvrant"],
        vitrage={"epaisseur_mm": s["epaisseur_vitrage_mm"], "verre_mm": s["verre_mm"]}, **extra))


# ---- la fiche de débit ------------------------------------------------------------------------------


_SANS_NOMBRE = re.compile(r"^\*+$|voir abaque|^-$", re.I)
_PAR_METRE = re.compile(r"(\d+)/ml/([HLW]\d?)")


def fiche_debit(snap: WikiSnapshot, saisie: Dict[str, Any]) -> Dict[str, Any]:
    d = donnees(snap)
    s = _saisie(snap, saisie)
    app = _application(d, s)
    gen = s["generation"]
    if app is None:
        autre = G_6057 if gen == G_6319 else G_6319
        existe = _application(d, s, autre) is not None
        return {"indisponible": f"Le catalogue {NOMS_GENERATION[gen]} n'a pas de nomenclature de « "
                                f"{TYPOLOGIES[s['configuration']].lower()} » pour ce module et cet ouvrant."
                                + (f" Le catalogue {NOMS_GENERATION[autre]} en a une : le choisir explicitement, les deux "
                                   "catalogues ne sont pas mélangés." if existe else "")}
    variables = _variables(d, s)
    notes: List[str] = []
    # une quantité renvoyée à un abaque (« *** », « voir abaque ») n'a pas de nombre : on le dit, on ne l'invente pas
    sans = [l for l in app.accessoires + app.vis if _SANS_NOMBRE.search(l.qte.strip())]
    # « 3/ml/H » : la page du catalogue 6057 le lit « 3 vis par mètre linéaire de la hauteur H » ; même écriture en 6319
    par_metre = {l.ref: l.qte for l in app.accessoires + app.vis if _PAR_METRE.search(l.qte)}

    def converti(l: Ligne) -> Ligne:
        return replace(l, qte=_PAR_METRE.sub(r"(\1*\2/1000)", l.qte))

    app_calc = replace(app, accessoires=[converti(l) for l in app.accessoires if l not in sans],
                       vis=[converti(l) for l in app.vis if l not in sans])
    for l in sans:
        if l.qte.strip() == "-":
            notes.append(f"{l.ref} {l.designation} : quantité non imprimée (« - »), non calculée.")
        else:
            notes.append(f"{l.ref} {l.designation} : quantité « {l.qte} » renvoyée à un abaque que le wiki ne chiffre pas, "
                         "non calculée" + (" (VER-370 pour les paumelles)" if "paumelle" in l.designation.lower() else "")
                         + ".")
    manquantes = [v for v in _variables_de_cote(app) if v not in variables]
    if manquantes:
        notes.append(f"Cotes {', '.join(manquantes)} non saisies : les lignes qui en dépendent ne sont pas calculées.")
    ver = verifier(snap, saisie)
    nb = NB_VANTAUX[s["configuration"]]
    fiche = construire_fiche(app_calc, variables, int(s["epaisseur_vitrage_mm"]), {"ref": ""}, d.images, None,
                             nb or None, "un par vantail (la planche n'écrit pas de quantité de vitrages)")
    for l in fiche["lignes"]:
        if gen == G_6057 and l["coupe"]:
            # le catalogue 6057 traduit le pictogramme en mots (« onglet / onglet ») ; la planche n'imprime pas de légende
            l["coupe"] = l["coupe"].replace("pictogramme", "type de coupe").replace(
                "(non légendé)", "(traduit des pictogrammes par le wiki, sans légende imprimée : VER-352)")
        if l["ref"] in par_metre:
            l["calcul"] = f"{par_metre[l['ref']]} : quantité par mètre linéaire de la cote (le wiki ne dit pas l'arrondi)"
    fiche["notes"] += notes
    if s["configuration"] in PMR:
        fiche["notes"].append("Les valeurs « de l'ancienne rédaction » du seuil PMR que porte la page des cotes de débit du "
                              "catalogue 6057 sont contredites par les planches (CTR-208, CTR-209, CTR-220) : elles ne sont "
                              "pas lues.")
    fiche["notes"].append(f"Catalogue {NOMS_GENERATION[gen]} ; les mentions de la coupe sont celles des planches, sans légende "
                          "(VER-352 en 6057, VER-396 en 6319).")
    return wiki_famille.relocaliser(snap, {
        "verdict": ver["verdict"], "libelle": ver["libelle"], "lignes": fiche["lignes"],
        "nomenclature": fiche["nomenclature"], "accessoires": [], "notes": fiche["notes"],
        "brouillons": _brouillons(d, [app.page]), "application": app.libelle, "generation": gen})
