"""Le calculateur ASKEY Frappe 65 Ouvrant Caché : le débit d'une application et la ferrure Ferco (21/10/2026).

Le catalogue de conception de la Frappe 65 OC ne porte **ni DTA, ni limite de baie, ni abaque de vent, ni
inertie requise** : seuls des procès-verbaux d'essai et des inerties de profilé sans valeur à leur comparer.
Aucun contrôle d'inertie n'est donc possible, et ce calculateur ne l'invente pas. Ce que le wiki permet :

1. le **débit** d'une application (un châssis complet : fenêtre un vantail, fixe, deux vantaux…), lu dans
   les seize applications des planches de débit. Les formules s'écrivent en L et H du châssis, avec LCO et HCO
   (le cadre d'ouvrant) définis par la ligne de l'ouvrant (« LCO=L-48 »). Seules six applications sont calculées
   (1 à 3 et 12 à 14, le contrat de données) ; les dix autres portent des variables que le wiki ne définit pas
   (L1, L2, H1 à H3, X, Y, x) ou des cotes que ses propres anomalies contestent : elles restent « sur étude »,
   avec la raison ;
2. la **zone de la ferrure Ferco Unijet**, par les inéquations de la planche « Exécution quincaillerie »
   (qui font foi), le nombre de points de fermeture qui en découle, et le poids de vantail maximal par
   exécution. L'abaque n'est imprimé que pour l'exécution C (VER-818) : C130 et D restent sur étude.

Deux hypothèses, affichées sur les contrôles qu'elles touchent : la largeur et la hauteur de fond de feuillure
(Lff, Hff) sont prises égales à la largeur et à la hauteur du cadre d'ouvrant (LCO, HCO) — le wiki ne
rapproche pas les deux notions — et le poids du vantail est celui du verre seul. On peut saisir Lff et Hff
mesurés : ils remplacent alors l'hypothèse.

Une prudence propre à la zone : le graphique de la planche trace, pour le bas de la zone, une diagonale que les
inéquations n'écrivent pas (VER-817). Un point que les inéquations placent dans une zone mais que la diagonale
laisse dehors, ou à moins de la précision de lecture, reste « sur étude » : la planche se contredit.

Toutes les valeurs viennent des tableaux du wiki, relus à chaque rechargement de l'instantané. Les pages de ce
système sont en brouillon : chaque verdict le dit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services import faisabilite as fa
from app.services import faisabilite_systemes as fs
from app.services import wiki_famille
from app.services.faisabilite import DonneeIntrouvable, _controle, _fmt, _nombre, _source, _texte_section
from app.services.wiki_service import WikiSnapshot
from app.services.wiki_tables import tables_toutes

ID = "askey-oc"
LIBELLE = "ASKEY Frappe 65 Ouvrant Caché"
GAMMES = ["Frappe 65 Ouvrant Caché"]
PORTEE = ("Débit des applications 1 à 3 et 12 à 14 (cotes de coupe, joints, remplissage), zone de la ferrure Ferco Unijet "
          "(points de fermeture, poids de vantail). Pas de limite de baie, de vent ni d'inertie dans le wiki : les autres "
          "applications, les exécutions C130 et D et les points litigieux de l'abaque restent sur étude.")

PAGES_DEBIT = (
    "/profiles/askey-frappe-65-oc-cotes-de-debit.md",
    "/profiles/askey-frappe-65-oc-cotes-de-debit-pmr-renovation-et-double-dormant.md",
    "/profiles/askey-frappe-65-oc-cotes-de-debit-imposte-pmr-et-renovation-36.md",
    "/profiles/askey-frappe-65-oc-cotes-de-debit-pv-36-imposte-et-double-dormant.md",
    "/profiles/askey-frappe-65-oc-cotes-de-debit-pv-36-c45-2-vantaux-et-fixe.md",
    "/profiles/askey-frappe-65-oc-cotes-de-debit-fixe-sur-seuil-pmr.md",
)
PAGE_FERCO = "/quincaillerie/askey-frappe-65-oc-abaques-ferco-unijet-et-limites-de-cales.md"
# Les coupes des profilés : pour l'image d'une référence, jamais pour une valeur.
PAGES_COUPES = (
    "/profiles/askey-frappe-65-oc-dormants-et-ouvrants.md",
    "/profiles/askey-frappe-65-oc-dormants-65-hi-et-vitrage-36-mm.md",
    "/profiles/askey-frappe-65-oc-dormants-monoblocs.md",
    "/profiles/askey-frappe-65-oc-meneaux-feuillures-rapportees-et-parcloses-de-fixe.md",
    "/profiles/askey-frappe-65-oc-profiles-complementaires-renovation-et-ite.md",
    "/profiles/askey-frappe-65-oc-profiles-de-liaison-et-poteaux-d-angle.md",
    "/profiles/askey-frappe-65-oc-seuils-et-rejets-d-eau.md",
)
PAGES = (*PAGES_DEBIT, PAGE_FERCO, fa.PAGE_ROTO)

OK, HORS, ETUDE, INFO = fs.OK, fs.HORS, fs.ETUDE, fs.INFO

# Le contrat de données : les applications dont toutes les formules ne portent que L, H, LCO et HCO, sans cote
# contestée par une anomalie ouverte. Les autres restent sur étude, avec leur raison.
APPLICATIONS_CALCULABLES = (1, 2, 3, 12, 13, 14)
EXECUTION_DE_REFERENCE = "C"      # « doc. ASKEY … (C) » : le seul abaque imprimé
HYPOTHESE_LFF = "hypothèse : Lff / Hff = LCO / HCO du cadre d'ouvrant"
HYPOTHESE_VERRE = "hypothèse : poids du vantail = verre seul ; le wiki ne donne pas le poids des profilés"
ANOMALIES_ZONE = ("VER-817", "VER-818", "CTR-287")
ANOMALIES_POIDS = ("CTR-286", "VER-818")

ENTETE_PROFILS = ["Référence", "Désignation", "Qté", "Débit (mm)"]
ENTETES_COUPE = ("Coupe (°)", "Coupes (°)")
ENTETES_JOINTS = (["Référence", "Désignation", "Qté", "Débit (mm)"], ["Référence", "Désignation", "Débit"])
ENTETE_ACCESSOIRES = ["Référence", "Désignation", "Qté"]
ENTETE_REMPLISSAGE = ["Remplissage", "Qté", "Dimensions (mm)"]
ENTETE_POIDS = ["Exécution Unijet", "Poids maxi vantail (kg)"]
ENTETE_ZONES = ["Zone", "Lff (mm)", "Hff (mm)"]
COLONNES_ZONES = {"gauche": "Points sur le montant de gauche", "droite": "Points sur le montant de droite",
                  "haute": "Point en traverse haute", "basse": "Point en traverse basse",
                  "total": "Total de points", "poignee": "Poignée"}
SECTION_ABAQUE = "Abaque des dimensions (zones 1 à 6)"
SECTION_POIDS = "Poids maxi de vantail par exécution"
SECTION_ZONES = "Exécution quincaillerie : zones, points de fermeture et paumelles"


# ---- l'évaluateur de formules de débit ----------------------------------------------------------
# Grammaire explicite : nombres (virgule ou point), variables nommées, + − × ÷ ( ), « 2H » pour 2 × H, une
# définition « LCO=L-48 » en tête. Jamais d'``eval`` : une formule qu'il ne comprend pas lève une erreur, et la
# ligne n'est pas calculée.

class FormuleIllisible(ValueError):
    pass


class VariableInconnue(FormuleIllisible):
    def __init__(self, nom: str):
        super().__init__(f"variable « {nom} » non définie par le wiki")
        self.nom = nom


_JETON = re.compile(r"\s*(?:(?P<n>\d+(?:[.,]\d+)?)|(?P<id>[A-Za-z]+\d*)|(?P<op>[-+−×÷*/()=]))")
_DEFINITION = re.compile(r"^\s*([A-Za-z]+\d*)\s*=\s*(.+?)\s*$")


def _nb(x: float) -> str:
    return f"{round(x, 3):.3f}".rstrip("0").rstrip(".").replace(".", ",")


def _jetons(texte: str) -> List[Tuple[str, Any]]:
    t, pos, out = texte.strip(), 0, []
    while pos < len(t):
        m = _JETON.match(t, pos)
        if not m or m.end() == pos:
            raise FormuleIllisible(f"« {texte} » : caractère « {t[pos]} » non compris")
        if m.group("n"):
            out.append(("n", float(m.group("n").replace(",", "."))))
        elif m.group("id"):
            out.append(("id", m.group("id")))
        else:
            out.append(("op", {"−": "-", "×": "*", "÷": "/"}.get(m.group("op"), m.group("op"))))
        pos = m.end()
    if not out:
        raise FormuleIllisible("formule vide")
    return out


class _Parseur:
    """Descente récursive : somme de produits de facteurs. Rend la valeur et la formule avec ses variables
    remplacées (« 952 − 59 »), qui sert d'écriture du calcul."""

    def __init__(self, jetons: List[Tuple[str, Any]], valeurs: Dict[str, float], source: str):
        self.j, self.v, self.i, self.source = jetons, valeurs, 0, source

    def _vu(self) -> Optional[Tuple[str, Any]]:
        return self.j[self.i] if self.i < len(self.j) else None

    def _op(self, *ops: str) -> Optional[str]:
        t = self._vu()
        return t[1] if t and t[0] == "op" and t[1] in ops else None

    def expression(self) -> Tuple[float, str]:
        val, txt = self.terme()
        while (op := self._op("+", "-")):
            self.i += 1
            v2, t2 = self.terme()
            val = val + v2 if op == "+" else val - v2
            txt += f" {'+' if op == '+' else '−'} {t2}"
        return val, txt

    def terme(self) -> Tuple[float, str]:
        nombre_litteral = self._vu() is not None and self._vu()[0] == "n"
        val, txt = self.facteur()
        while True:
            if (op := self._op("*", "/")):
                self.i += 1
                v2, t2 = self.facteur()
                if op == "/":
                    if v2 == 0:
                        raise FormuleIllisible(f"« {self.source} » : division par zéro")
                    val, txt = val / v2, f"{txt} ÷ {t2}"
                else:
                    val, txt = val * v2, f"{txt} × {t2}"
                nombre_litteral = False
                continue
            t = self._vu()
            # « 2H », « 2 L2 », « 2(H1+H2) » : un nombre collé à un nom ou à une parenthèse multiplie
            if nombre_litteral and t is not None and (t[0] == "id" or (t[0] == "op" and t[1] == "(")):
                v2, t2 = self.facteur()
                val, txt = val * v2, f"{txt} × {t2}"
                nombre_litteral = False
                continue
            return val, txt

    def facteur(self) -> Tuple[float, str]:
        t = self._vu()
        if t is None:
            raise FormuleIllisible(f"« {self.source} » : formule incomplète")
        self.i += 1
        if t[0] == "n":
            return t[1], _nb(t[1])
        if t[0] == "id":
            if t[1] not in self.v:
                raise VariableInconnue(t[1])
            return self.v[t[1]], _nb(self.v[t[1]])
        if t[1] == "-":
            val, txt = self.facteur()
            return -val, f"−{txt}"
        if t[1] == "(":
            val, txt = self.expression()
            if self._op(")") is None:
                raise FormuleIllisible(f"« {self.source} » : parenthèse non fermée")
            self.i += 1
            return val, f"({txt})"
        raise FormuleIllisible(f"« {self.source} » : « {t[1]} » inattendu")


def evaluer(formule: str, valeurs: Dict[str, float]) -> Tuple[float, str]:
    """(valeur, calcul écrit) d'une formule sans définition ; ``FormuleIllisible`` si elle n'est pas comprise,
    ``VariableInconnue`` si elle porte une variable que ``valeurs`` ne donne pas."""
    p = _Parseur(_jetons(formule), valeurs, formule)
    val, txt = p.expression()
    if p.i != len(p.j):
        raise FormuleIllisible(f"« {formule} » : « {p.j[p.i][1]} » inattendu")
    return round(val, 6), txt


def variables_de(formule: str) -> List[str]:
    return [v for k, v in _jetons(_DEFINITION.sub(r"\2", formule)) if k == "id"]


# ---- les données, telles que le wiki les écrit ----------------------------------------------

@dataclass
class Ligne:
    ref: str
    designation: str
    qte: Optional[int]
    qte_brut: str
    debit: str
    coupe: str


@dataclass
class Application:
    numero: int
    titre: str
    page: str
    texte: str                                  # la section de l'application, pour ses anomalies
    anomalies: List[str]
    profils: List[Ligne] = field(default_factory=list)
    joints: List[Ligne] = field(default_factory=list)
    accessoires: List[Ligne] = field(default_factory=list)
    remplissages: List[Dict[str, Any]] = field(default_factory=list)
    localisations: Dict[str, str] = field(default_factory=dict)   # nature de tableau -> « schéma: … » qui le suit
    localisation: str = ""                                        # la planche de l'application (la ligne « schéma: » est
                                                                  # imprimée une fois, après le dernier tableau)
    sections: Dict[str, str] = field(default_factory=dict)        # nature de tableau -> titre de section


@dataclass
class Zone:
    numero: int
    lff: Tuple[float, bool, float, bool]        # mini, inclus ?, maxi, inclus ?
    hff: Tuple[float, bool, float, bool]
    gauche: int
    droite: int
    haute: int
    basse: int
    total: int
    poignee: str

    def contient(self, lff: float, hff: float) -> bool:
        def dans(b: Tuple[float, bool, float, bool], x: float) -> bool:
            return (x >= b[0] if b[1] else x > b[0]) and (x <= b[2] if b[3] else x < b[2])
        return dans(self.lff, lff) and dans(self.hff, hff)


@dataclass
class Ferco:
    poids_max: Dict[str, float]
    gache_kg: Optional[float]
    zones: List[Zone]
    diagonale: Tuple[Tuple[float, float], Tuple[float, float]]   # (Lff, Hff) de ses deux extrémités
    precision_mm: float
    loc_poids: str
    loc_zones: str
    loc_abaque: str


@dataclass
class Donnees:
    applications: Dict[int, Application]
    ferco: Ferco
    kg_par_m2_mm: Optional[float]
    images: Dict[str, str]                      # 7 caractères de la référence -> coupe
    statuts: Dict[str, str]


_CACHE: Dict[str, Any] = {"cle": None, "donnees": None}
_ANOMALIE = re.compile(r"\b(?:INC|VER|CTR)-\d+\b")


def anomalies_de(texte: str) -> List[str]:
    vus: List[str] = []
    for a in _ANOMALIE.findall(texte):
        if a not in vus:
            vus.append(a)
    return vus


def _entier(c: str) -> Optional[int]:
    return int(c.strip()) if re.fullmatch(r"\d+", c.strip()) else None


def _ligne(r: List[str], avec_qte: bool) -> Ligne:
    if avec_qte:
        return Ligne(r[0], r[1], _entier(r[2]), r[2], r[3].strip(), r[4].strip() if len(r) > 4 else "")
    return Ligne(r[0], r[1], None, "", r[2].strip(), "")


def _charge_applications(snap: WikiSnapshot) -> Dict[int, Application]:
    apps: Dict[int, Application] = {}
    for page in PAGES_DEBIT:
        corps = fa._corps(snap, page)
        intro = corps.split("\n# ", 1)[0]
        for t in tables_toutes(corps):
            titre = next((c for c in t.chemin if re.match(r"Application \d+ :", c)), None)
            if titre is None:
                continue
            n = int(re.match(r"Application (\d+) :", titre).group(1))
            app = apps.get(n)
            if app is None:
                texte = _texte_section(corps, titre)
                app = apps[n] = Application(n, titre, page, texte, anomalies_de(texte) + [
                    a for a in anomalies_de(intro) if a not in anomalies_de(texte)])
            sous = t.section
            if re.fullmatch(r"Profilés de l'application \d+", sous):
                if t.entete[:4] != ENTETE_PROFILS or t.entete[4:5] not in [[c] for c in ENTETES_COUPE]:
                    raise DonneeIntrouvable(f"{page} : en-têtes des profilés de l'application {n} changés : {t.entete}")
                app.profils = [_ligne(r, True) for r in t.lignes]
                nature = "profils"
            elif re.fullmatch(r"Joints de l'application \d+", sous):
                if t.entete not in ENTETES_JOINTS:
                    raise DonneeIntrouvable(f"{page} : en-têtes des joints de l'application {n} changés : {t.entete}")
                app.joints = [_ligne(r, len(t.entete) == 4) for r in t.lignes]
                nature = "joints"
            elif re.fullmatch(r"Accessoires de l'application \d+", sous):
                if t.entete != ENTETE_ACCESSOIRES:
                    raise DonneeIntrouvable(f"{page} : en-têtes des accessoires de l'application {n} changés : {t.entete}")
                app.accessoires = [Ligne(r[0], r[1], _entier(r[2]), r[2], "", "") for r in t.lignes]
                nature = "accessoires"
            elif re.fullmatch(r"Remplissages? de l'application \d+", sous):
                if t.entete != ENTETE_REMPLISSAGE:
                    raise DonneeIntrouvable(f"{page} : en-têtes du remplissage de l'application {n} changés : {t.entete}")
                app.remplissages = [{"libelle": r[0], "qte": _entier(r[1]), "dimensions": r[2]} for r in t.lignes]
                nature = "remplissages"
            else:
                continue
            app.localisations[nature], app.sections[nature] = t.localisation, sous
    for app in apps.values():
        app.localisation = next((v for v in app.localisations.values() if v), "")
    manquantes = set(APPLICATIONS_CALCULABLES) - set(apps)
    if manquantes:
        raise DonneeIntrouvable(f"cotes de débit de la Frappe 65 OC : applications {sorted(manquantes)} introuvables")
    for n in APPLICATIONS_CALCULABLES:
        if not apps[n].profils or not apps[n].remplissages:
            raise DonneeIntrouvable(f"cotes de débit de la Frappe 65 OC : application {n} sans profilés ou sans remplissage")
    return apps


def _borne(texte: str, variable: str) -> Tuple[float, bool, float, bool]:
    """« 500 ≤ Lff < 900 » : la borne telle que la planche l'écrit ; ses signes disent si les extrémités comptent."""
    m = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*([≤<])\s*" + variable + r"\s*([≤<])\s*(\d+(?:[.,]\d+)?)\s*", texte)
    if not m:
        raise DonneeIntrouvable(f"{PAGE_FERCO} : borne « {texte} » illisible")
    return (float(m.group(1).replace(",", ".")), m.group(2) == "≤", float(m.group(4).replace(",", ".")), m.group(3) == "≤")


def _charge_ferco(snap: WikiSnapshot) -> Ferco:
    corps = fa._corps(snap, PAGE_FERCO)
    poids: Dict[str, float] = {}
    zones: List[Zone] = []
    loc_poids = loc_zones = ""
    for t in tables_toutes(corps):
        if t.entete[:2] == ENTETE_POIDS:
            poids = {r[0]: _nombre(r[1]) for r in t.lignes}
            loc_poids = t.localisation
        elif t.entete[:3] == ENTETE_ZONES:
            col = {h: k for k, h in enumerate(t.entete)}
            try:
                for r in t.lignes:
                    zones.append(Zone(int(_nombre(r[0])), _borne(r[1], "Lff"), _borne(r[2], "Hff"),
                                      *(int(_nombre(r[col[COLONNES_ZONES[k]]])) for k in ("gauche", "droite", "haute", "basse", "total")),
                                      r[col[COLONNES_ZONES["poignee"]]]))
            except KeyError as exc:
                raise DonneeIntrouvable(f"{PAGE_FERCO} : colonne {exc} du tableau des zones introuvable")
            loc_zones = t.localisation
    if not poids or not zones:
        raise DonneeIntrouvable(f"{PAGE_FERCO} : tableau des poids maxi ou des zones introuvable")
    texte_abaque = _texte_section(corps, SECTION_ABAQUE)
    md = re.search(r"diagonale qui part de Lff (\d+) et de Hff (\d+) environ et monte vers la droite jusqu'à Lff (\d+) et "
                   r"Hff (\d+) environ", texte_abaque)
    mp = re.search(r"précision de lecture ±(\d+) mm", texte_abaque)
    if not md or not mp:
        raise DonneeIntrouvable(f"{PAGE_FERCO} : diagonale ou précision de lecture de l'abaque introuvable")
    x0, y0, x1, y1 = (float(g) for g in md.groups())
    mg = re.search(r"poids > (\d+) ?Kg utiliser une gâche releveur", _texte_section(corps, SECTION_POIDS), re.I)
    loc_abaque = next((t.localisation for t in tables_toutes(corps) if t.entete[:1] == ["Hff (mm)"]), loc_zones)
    return Ferco(poids, float(mg.group(1)) if mg else None, zones, ((x0, y0), (x1, y1)), float(mp.group(1)),
                 loc_poids, loc_zones, loc_abaque)


def _charge_coupes(snap: WikiSnapshot) -> Dict[str, str]:
    """Une coupe par référence (les 7 caractères après la lettre initiale, qui varie : INC-790 et suivantes)."""
    images: Dict[str, str] = {}
    for page in PAGES_COUPES:
        if page not in snap.pages:
            continue
        for t in tables_toutes(fa._corps(snap, page)):
            if t.entete[:1] != ["Référence"] or t.entete[-1:] != ["Coupe"]:
                continue
            for r, b in zip(t.lignes, t.brut):
                img = re.search(r"\((/assets/[^)]+)\)", b[-1])
                if img and re.fullmatch(r"[WY][0-9A-Z]{7}", r[0]):
                    images.setdefault(r[0][1:], img.group(1))
    return images


def _charge(snap: WikiSnapshot) -> Donnees:
    kg = re.search(r"1 mm/m² d'épaisseur de vitre ≙ ([\d,]+) kg", fa._corps(snap, fa.PAGE_ROTO))
    return Donnees(_charge_applications(snap), _charge_ferco(snap), _nombre(kg.group(1)) if kg else None,
                   _charge_coupes(snap), {p: (snap.pages[p].status or "") for p in PAGES})


def donnees(snap: WikiSnapshot) -> Donnees:
    cle = (id(snap), snap.loaded_at)
    if _CACHE["cle"] != cle:
        _CACHE["donnees"] = _charge(snap)
        _CACHE["cle"] = cle
    return _CACHE["donnees"]


# ---- le débit d'une application -----------------------------------------------------------------

def variables_manquantes(app: Application) -> List[str]:
    """Les variables des formules de l'application que ni L, H ni une définition de ligne ne donnent."""
    connues = {"L", "H"} | {m.group(1) for l in app.profils if (m := _DEFINITION.match(l.debit))}
    vues: List[str] = []
    cellules = [l.debit for l in app.profils + app.joints] + [x for r in app.remplissages for x in r["dimensions"].split(";")]
    for c in cellules:
        try:
            noms = variables_de(c)
        except FormuleIllisible:
            continue
        vues += [v for v in noms if v not in connues and v not in vues]
    return vues


def raison_sur_etude(app: Application) -> Optional[str]:
    """Pourquoi l'application n'est pas calculée ; ``None`` si elle l'est."""
    if app.numero in APPLICATIONS_CALCULABLES:
        return None
    manque = variables_manquantes(app)
    if manque:
        return (f"Les formules de l'application {app.numero} portent des variables que le wiki ne définit pas "
                f"({', '.join(manque)}) : leurs lignes ne sont pas calculées.")
    return (f"L'application {app.numero} n'est pas retenue par le contrat de données : ses cotes sont contestées par "
            f"{', '.join(anomalies_de(app.texte))} dans le wiki.")


def calcul_application(app: Application, largeur: float, hauteur: float) -> Dict[str, Any]:
    """Toutes les lignes de l'application pour un châssis L × H ; une ligne qui n'est pas comprise porte son erreur."""
    valeurs: Dict[str, float] = {"L": largeur, "H": hauteur}
    definitions: Dict[int, Tuple[float, str]] = {}      # rang de la ligne -> valeur, calcul écrit
    par_nom: Dict[str, Tuple[float, str]] = {}
    notes: List[str] = []
    for k, l in enumerate(app.profils):
        m = _DEFINITION.match(l.debit)
        if not m:
            continue
        try:
            v, txt = evaluer(m.group(2), valeurs)
        except FormuleIllisible:
            continue
        definitions[k] = (v, txt)
        par_nom.setdefault(m.group(1), (v, txt))
        if m.group(1) not in valeurs:
            valeurs[m.group(1)] = v
        elif abs(valeurs[m.group(1)] - v) > 1e-6:
            notes.append(f"{m.group(1)} est défini deux fois avec des valeurs différentes ({_nb(valeurs[m.group(1)])} puis "
                         f"{_nb(v)}) : la première sert aux autres lignes.")

    def calcule(l: Ligne, definition: Optional[Tuple[float, str]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"ligne": l, "longueur": None, "calcul": l.debit, "erreur": None}
        try:
            if definition is not None:
                v, txt = definition
                nom = _DEFINITION.match(l.debit).group(1)
                out.update(longueur=v, calcul=f"{nom} = {txt}")
            else:
                v, txt = evaluer(_DEFINITION.sub(r"\2", l.debit), valeurs)
                out.update(longueur=v, calcul=re.sub(r"\s+", " ", l.debit.strip()) + (f" : {txt}" if txt != _nb(v) else ""))
            if v <= 0:
                out.update(longueur=None, erreur=f"cote négative ou nulle ({_nb(v)} mm) : châssis trop petit pour cette formule")
        except FormuleIllisible as exc:
            out["erreur"] = str(exc)
        return out

    profils = [calcule(l, definitions.get(k)) for k, l in enumerate(app.profils)]
    joints = [calcule(l, None) for l in app.joints]
    remplissages = []
    for r in app.remplissages:
        out = {"libelle": r["libelle"], "qte": r["qte"], "hauteur": None, "largeur": None, "calcul": r["dimensions"],
               "erreur": None}
        try:
            parties = [p.strip() for p in r["dimensions"].split(";")]
            if len(parties) != 2:
                raise FormuleIllisible(f"« {r['dimensions']} » : deux cotes attendues, séparées par « ; »")
            lus = {}
            for p in parties:
                v, txt = evaluer(p, valeurs)
                nom = (variables_de(p) or [""])[0]
                # l'ordre change d'une application à l'autre (« L2-103 ; H-100,5 ») : on lit la lettre, pas la position
                axe = {"H": "hauteur", "L": "largeur"}.get(nom[:1])
                if axe is None or axe in lus:
                    raise FormuleIllisible(f"« {r['dimensions']} » : hauteur et largeur non distinguées")
                lus[axe] = (v, txt)
            out.update(hauteur=lus["hauteur"][0], largeur=lus["largeur"][0],
                       calcul=f"{r['dimensions']} : {lus['hauteur'][1]} ; {lus['largeur'][1]}")
        except FormuleIllisible as exc:
            out["erreur"] = str(exc)
        remplissages.append(out)
    return {"valeurs": valeurs, "definitions": par_nom,
            "profils": profils, "joints": joints, "remplissages": remplissages, "notes": notes}


def _lignes_en_erreur(calc: Dict[str, Any]) -> List[str]:
    return [f"{x['ligne'].ref} « {x['ligne'].debit} » : {x['erreur']}" for x in calc["profils"] + calc["joints"] if x["erreur"]] + \
           [f"remplissage « {x['libelle']} » : {x['erreur']}" for x in calc["remplissages"] if x["erreur"]]


# ---- la zone de la ferrure ----------------------------------------------------------------------

def _diagonale(f: Ferco, lff: float) -> Optional[float]:
    (x0, y0), (x1, y1) = f.diagonale
    return y0 + (lff - x0) * (y1 - y0) / (x1 - x0) if x0 <= lff <= x1 else None


def _borne_texte(b: Tuple[float, bool, float, bool], var: str) -> str:
    return f"{_fmt(b[0])} {'≤' if b[1] else '<'} {var} {'≤' if b[3] else '<'} {_fmt(b[2])}"


def zone_ferco(snap: WikiSnapshot, lff: float, hff: float) -> Dict[str, Any]:
    """La zone de l'abaque Ferco pour un fond de feuillure Lff × Hff (mm), par les inéquations de la planche.

    ``statut`` : ok (une zone, que la diagonale du graphique confirme), sur étude (une zone par les inéquations
    mais le graphique la contredit ou la laisse à moins de sa précision de lecture), hors (aucune zone)."""
    f = donnees(snap).ferco
    trouvees = [z for z in f.zones if z.contient(lff, hff)]
    base = {"lff": lff, "hff": hff, "zone": None, "points": None, "ecart_diagonale_mm": None}
    cote = f"Lff {_fmt(lff, 1 if lff % 1 else 0)} × Hff {_fmt(hff, 1 if hff % 1 else 0)} mm"
    if not trouvees:
        return dict(base, statut=HORS, detail=f"{cote} : aucune des zones 1 à 6 de l'abaque ne couvre ce point (non conforme).")
    if len(trouvees) > 1:
        return dict(base, statut=ETUDE, detail=f"{cote} : les inéquations de la planche placent ce point dans plusieurs zones "
                                              f"({', '.join(str(z.numero) for z in trouvees)}).")
    z = trouvees[0]
    pts = (f"{z.total} points de fermeture ({z.gauche} sur le montant de gauche, {z.droite} sur celui de droite, "
           f"{z.haute} en traverse haute, {z.basse} en traverse basse), poignée sur le {z.poignee}")
    zone = {"zone": z.numero, "points": z.total, "poignee": z.poignee,
            "lff_bornes": _borne_texte(z.lff, "Lff"), "hff_bornes": _borne_texte(z.hff, "Hff")}
    diag = _diagonale(f, lff)
    if diag is None:
        return dict(base, **zone, statut=ETUDE,
                    detail=f"{cote} : zone {z.numero} par les inéquations de la planche, mais hors de la plage où le graphique "
                           f"trace sa diagonale ; sur étude.")
    ecart = hff - diag
    if ecart < f.precision_mm:
        pos = (f"à {_fmt(-ecart)} mm sous" if ecart < 0 else f"à {_fmt(ecart)} mm au-dessus de")
        return dict(base, **zone, ecart_diagonale_mm=round(ecart, 1), statut=ETUDE,
                    detail=f"{cote} : zone {z.numero} par les inéquations de la planche ({zone['lff_bornes']} ; {zone['hff_bornes']}), "
                           f"mais le point est {pos} la diagonale qui borde le bas de la zone sur le graphique (précision de "
                           f"lecture ±{_fmt(f.precision_mm)} mm) : la planche se contredit, sur étude.")
    return dict(base, **zone, ecart_diagonale_mm=round(ecart, 1), statut=OK,
                detail=f"{cote} : zone {z.numero} ({zone['lff_bornes']} ; {zone['hff_bornes']}), {pts}.")


# ---- le formulaire --------------------------------------------------------------------------------

def _libelle_application(app: Application) -> str:
    return app.titre + ("" if app.numero in APPLICATIONS_CALCULABLES else " — sur étude")


def options(snap: WikiSnapshot) -> Dict[str, Any]:
    d = donnees(snap)
    configurations = {str(n): _libelle_application(a) for n, a in sorted(d.applications.items())}
    executions = {k: f"Unijet {k} ({_fmt(v)} kg maxi)" for k, v in d.ferco.poids_max.items()}
    champs = [
        {"id": "configuration", "libelle": "Application", "type": "select", "options": configurations, "defaut": "1"},
        {"id": "largeur_mm", "libelle": "Largeur du châssis (L)", "type": "nombre", "unite": "mm", "min": 300, "max": 6000,
         "pas": 10, "defaut": 1000},
        {"id": "hauteur_mm", "libelle": "Hauteur du châssis (H)", "type": "nombre", "unite": "mm", "min": 300, "max": 6000,
         "pas": 10, "defaut": 1200},
        {"id": "execution", "libelle": "Exécution de la ferrure Ferco", "type": "select", "options": executions,
         "defaut": EXECUTION_DE_REFERENCE,
         "aide": "L'abaque n'est imprimé que pour l'exécution C ; sans objet pour un châssis fixe."},
        {"id": "verre_mm", "libelle": "Épaisseur de verre cumulée (facultatif)", "type": "nombre", "unite": "mm", "min": 0,
         "max": 100, "pas": 0.5, "defaut": None, "requis": False,
         "aide": "Somme des verres, sans les lames d'air : sert au contrôle du poids de vantail."},
        {"id": "lff_mm", "libelle": "Lff mesuré (facultatif)", "type": "nombre", "unite": "mm", "min": 1, "max": 6000,
         "pas": 1, "defaut": None, "requis": False,
         "aide": "Largeur de fond de feuillure quincaillerie. Vide : prise égale à LCO (hypothèse affichée)."},
        {"id": "hff_mm", "libelle": "Hff mesuré (facultatif)", "type": "nombre", "unite": "mm", "min": 1, "max": 6000,
         "pas": 1, "defaut": None, "requis": False,
         "aide": "Hauteur de fond de feuillure quincaillerie. Vide : prise égale à HCO (hypothèse affichée)."},
    ]
    return {
        "champs": champs, "configurations": configurations, "pages": list(PAGES),
        "brouillons": sorted(p for p, s in d.statuts.items() if s == "draft"),
        "avertissements": [
            "Le wiki n'a ni limite de baie, ni abaque de vent, ni inertie requise pour cette gamme : aucun contrôle de ce genre.",
            "Débit calculé pour les applications " + ", ".join(str(n) for n in APPLICATIONS_CALCULABLES)
            + " ; les autres restent sur étude, avec la raison.",
            f"Ferrure : {HYPOTHESE_LFF}.",
            "Parcloses et joints de vitrage : voir la page « Parcloses ». Supports de cale non contrôlés.",
        ],
    }


# ---- le verdict -------------------------------------------------------------------------------------

def _l(x: float) -> str:
    return _fmt(x, 1 if round(x, 3) % 1 else 0)


def _src_app(app: Application, nature: str) -> Dict[str, str]:
    return _source(app.page, app.sections.get(nature, ""), app.localisations.get(nature) or app.localisation)


def _poids_vantail(d: Donnees, s: Dict[str, Any], calc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Le contrôle du poids du verre contre le poids maxi de l'exécution ; ``None`` sans épaisseur saisie."""
    f = d.ferco
    pmax = f.poids_max.get(s["execution"])
    src = [_source(PAGE_FERCO, SECTION_POIDS, f.loc_poids)]
    if not s.get("verre_mm"):
        return _controle("poids", "Poids de vantail (Unijet)", INFO,
                         f"Unijet {s['execution']} : {_fmt(pmax)} kg maxi par vantail"
                         + (f", gâche releveur au-delà de {_fmt(f.gache_kg)} kg" if f.gache_kg else "")
                         + ". Épaisseur de verre non saisie : le poids n'est pas contrôlé.",
                         src, list(ANOMALIES_POIDS), {"poids_max_kg": pmax})
    verre = next((r for r in calc["remplissages"] if r["hauteur"] and r["largeur"]), None)
    if d.kg_par_m2_mm is None or verre is None:
        return _controle("poids", "Poids de vantail (Unijet)", ETUDE,
                         "Le poids du verre ne se calcule pas : facteur kg/m²/mm ou dimensions de remplissage illisibles.",
                         src, list(ANOMALIES_POIDS))
    poids = s["verre_mm"] * d.kg_par_m2_mm * verre["hauteur"] / 1000 * verre["largeur"] / 1000
    src.append(_source(fa.PAGE_ROTO, "Convertir une épaisseur de vitrage en poids de vantail"))
    gache = f" Au-delà de {_fmt(f.gache_kg)} kg, une gâche releveur est exigée." if f.gache_kg and poids > f.gache_kg else ""
    detail = (f"{_fmt(poids, 1)} kg de verre par vantail ({_fmt(s['verre_mm'], 1)} mm × {_fmt(d.kg_par_m2_mm, 1)} kg/m²/mm × "
              f"{_l(verre['hauteur'])} × {_l(verre['largeur'])} mm de remplissage) pour {_fmt(pmax)} kg maxi en Unijet "
              f"{s['execution']} ({HYPOTHESE_VERRE}).{gache}")
    if poids > pmax:
        detail += " Le verre seul dépasse déjà le poids maxi."
    return _controle("poids", "Poids de vantail (Unijet)", HORS if poids > pmax else OK, detail, src, list(ANOMALIES_POIDS),
                     {"poids_kg": round(poids, 1), "poids_max_kg": pmax, "hypothese": HYPOTHESE_VERRE})


def verifier(snap: WikiSnapshot, saisie: Dict[str, Any]) -> Dict[str, Any]:
    d = donnees(snap)
    s = fs.coercer(options(snap)["champs"], saisie)
    app = d.applications[int(s["configuration"])]
    L, H = s["largeur_mm"], s["hauteur_mm"]
    ctl: List[Dict[str, Any]] = []
    pages = [app.page]
    notes = ["Parcloses, joints de vitrage et supports de cale ne sont pas contrôlés ici (page « Parcloses »)."]

    raison = raison_sur_etude(app)
    if raison:
        ctl.append(_controle("debit", "Débit de l'application", ETUDE, raison + " Calculées : "
                             + ", ".join(str(n) for n in APPLICATIONS_CALCULABLES) + ".",
                             [_src_app(app, "profils")], app.anomalies))
        brouillons = [p for p in pages if d.statuts.get(p) == "draft"]
        return wiki_famille.relocaliser(snap, fs.synthese(ctl, brouillons, notes=notes, debit=None))

    calc = calcul_application(app, L, H)
    erreurs = _lignes_en_erreur(calc)
    etapes = [{"etape": nom, "calcul": txt, "resultat": _l(v)} for nom, (v, txt) in calc["definitions"].items()]
    etapes += [{"etape": f"Remplissage « {r['libelle']} »", "calcul": r["calcul"],
                "resultat": f"{_l(r['hauteur'])} × {_l(r['largeur'])}"} for r in calc["remplissages"] if not r["erreur"]]
    if erreurs:
        ctl.append(_controle("debit", "Débit de l'application", ETUDE,
                             f"{app.titre} : lignes non calculées — " + " ; ".join(erreurs) + ".",
                             [_src_app(app, "profils")], app.anomalies, {"etapes": etapes}))
    else:
        cadre = (f"cadre d'ouvrant LCO {_l(calc['valeurs']['LCO'])} × HCO {_l(calc['valeurs']['HCO'])} mm, "
                 if "LCO" in calc["valeurs"] and "HCO" in calc["valeurs"] else "châssis fixe, ")
        remplissage = " ; ".join(f"{r['qte'] or '?'} × {_l(r['hauteur'])} × {_l(r['largeur'])} mm" for r in calc["remplissages"])
        ctl.append(_controle("debit", "Débit de l'application", INFO,
                             f"{app.titre} pour {_l(L)} × {_l(H)} mm : {cadre}{len(app.profils)} lignes de profilé, "
                             f"remplissage (hauteur × largeur) {remplissage}. Le wiki n'imprime aucune dimension minimale "
                             "ni maximale pour l'application.",
                             [_src_app(app, "profils"), _src_app(app, "remplissages")], app.anomalies, {"etapes": etapes}))
    notes += calc["notes"]

    # la ferrure : sans ouvrant, pas de ferrure
    lco, hco = calc["valeurs"].get("LCO"), calc["valeurs"].get("HCO")
    if lco is None or hco is None:
        ctl.append(_controle("ferco", "Ferrure Ferco Unijet", INFO,
                             "Châssis fixe : aucun ouvrant, aucune ferrure à contrôler.", [_src_app(app, "profils")]))
    elif s["execution"] != EXECUTION_DE_REFERENCE:
        ctl.append(_controle("ferco", "Ferrure Ferco Unijet", ETUDE,
                             f"Exécution {s['execution']} : l'abaque et les zones ne sont imprimés que pour l'exécution "
                             f"{EXECUTION_DE_REFERENCE} (VER-818), aucune plage de Lff et de Hff n'est donnée pour C130 ni pour D.",
                             [_source(PAGE_FERCO, SECTION_ABAQUE, d.ferco.loc_abaque)], list(ANOMALIES_ZONE)))
        pages.append(PAGE_FERCO)
    else:
        lff, hff = s.get("lff_mm") or lco, s.get("hff_mm") or hco
        z = zone_ferco(snap, lff, hff)
        hypo = [n for n, mesure in (("Lff", s.get("lff_mm")), ("Hff", s.get("hff_mm"))) if not mesure]
        detail = z["detail"] + (f" ({HYPOTHESE_LFF} ; {' et '.join(hypo)} non mesuré{'s' if len(hypo) > 1 else ''})." if hypo else "")
        ctl.append(_controle("ferco", "Ferrure Ferco Unijet", z["statut"], detail,
                             [_source(PAGE_FERCO, SECTION_ZONES, d.ferco.loc_zones),
                              _source(PAGE_FERCO, SECTION_ABAQUE, d.ferco.loc_abaque)],
                             list(ANOMALIES_ZONE), {k: z[k] for k in ("zone", "points", "lff", "hff", "ecart_diagonale_mm")}
                             | ({"hypothese": HYPOTHESE_LFF} if hypo else {})))
        pages.append(PAGE_FERCO)
    if lco is not None and hco is not None:
        poids = _poids_vantail(d, s, calc)
        if poids is not None:
            ctl.append(poids)
            pages += [PAGE_FERCO] + ([fa.PAGE_ROTO] if s.get("verre_mm") else [])

    brouillons = sorted({p for p in pages if d.statuts.get(p) == "draft"})
    debit = {"application": app.numero, "variables": {k: v for k, v in calc["valeurs"].items() if k not in ("L", "H")},
             "remplissages": [{"libelle": r["libelle"], "qte": r["qte"], "hauteur": r["hauteur"], "largeur": r["largeur"]}
                              for r in calc["remplissages"]]}
    return wiki_famille.relocaliser(snap, fs.synthese(ctl, brouillons, notes=notes, debit=debit))


# ---- la fiche de débit -------------------------------------------------------------------------------

def fiche_debit(snap: WikiSnapshot, saisie: Dict[str, Any]) -> Dict[str, Any]:
    d = donnees(snap)
    s = fs.coercer(options(snap)["champs"], saisie)
    app = d.applications[int(s["configuration"])]
    raison = raison_sur_etude(app)
    if raison:
        return {"indisponible": f"Liste de coupe non calculée : {raison}"}
    calc = calcul_application(app, s["largeur_mm"], s["hauteur_mm"])
    erreurs = _lignes_en_erreur(calc)
    ver = verifier(snap, saisie)
    lignes: List[Dict[str, Any]] = []
    notes = ["Les longueurs sont les débits imprimés par le catalogue de conception ; le wiki ne donne aucune surlongueur "
             "de soudure ni de sertissage.",
             "Le wiki n'imprime aucune dimension minimale ni maximale du châssis pour cette application."]

    def ajoute(famille: str, x: Dict[str, Any], nature: str) -> None:
        l: Ligne = x["ligne"]
        note = "" if l.qte is not None else f"quantité non numérique : « {l.qte_brut} »"
        lignes.append({"rep": len(lignes) + 1, "famille": famille, "piece": l.designation, "ref": l.ref,
                       "qte": l.qte if l.qte is not None else 0, "longueur": x["longueur"], "calcul": x["calcul"],
                       "coupe": f"{l.coupe}°" if l.coupe else "",
                       "source": _src_app(app, nature), "note": note, "image": d.images.get(l.ref[1:])})

    for x in calc["profils"]:
        if not x["erreur"]:
            ajoute("Profilés", x, "profils")
    for x in calc["joints"]:
        if not x["erreur"]:
            ajoute("Joints", x, "joints")
    for r in calc["remplissages"]:
        if not r["erreur"]:
            lignes.append({"rep": len(lignes) + 1, "famille": "Vitrage", "piece": r["libelle"], "ref": "—",
                           "qte": r["qte"] or 0, "longueur": None, "calcul": r["calcul"], "coupe": "",
                           "source": _src_app(app, "remplissages"),
                           "note": f"hauteur × largeur : {_l(r['hauteur'])} × {_l(r['largeur'])} mm", "image": None})
    for e in erreurs:
        notes.append(f"Ligne non calculée : {e}.")
    notes += calc["notes"]
    if app.anomalies:
        notes.append("Anomalies ouvertes sur cette application : " + ", ".join(app.anomalies) + ".")

    nomen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for l in lignes:
        sans_ref = l["ref"] in ("—", "", None)
        n = nomen.setdefault((l["piece"] if sans_ref else l["ref"], l["famille"]),
                             {"ref": "—" if sans_ref else l["ref"], "designation": l["piece"], "famille": l["famille"],
                              "qte": 0, "ml": 0.0, "image": l["image"]})
        n["qte"] += l["qte"]
        if l["longueur"] is not None:
            n["ml"] += l["qte"] * l["longueur"] / 1000
    nomenclature = [dict(v, ml=round(v["ml"], 2) if v["ml"] else None) for v in nomen.values()]
    accessoires = [{"ref": a.ref, "designation": a.designation, "qte": a.qte, "profil": f"qté {a.qte_brut}",
                    "image": None, "section": app.sections.get("accessoires", "")} for a in app.accessoires]
    return wiki_famille.relocaliser(snap, {
        "verdict": ver["verdict"], "libelle": ver["libelle"], "lignes": lignes, "nomenclature": nomenclature,
        "accessoires": accessoires, "notes": notes, "debit": ver["debit"], "brouillons": ver["brouillons"]})
