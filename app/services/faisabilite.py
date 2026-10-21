"""Le vérificateur de faisabilité PERFORM76 (audit du 22/09, § 9.1) — premier jet.

On saisit une menuiserie (configuration, dimension hors tout, profilés, couleur, vitrage) et on
obtient **réalisable**, **hors domaine** ou **sur étude**, contrôle par contrôle, chacun avec la
page et la section du wiki qui le porte. Aucun modèle : tout est calculé, et **toutes les valeurs
sont lues dans le wiki** à chaque rechargement de l'instantané — les tableaux sont la donnée.

Chaîne de calcul, dans l'ordre du manuel profine :

1. les **cotes de débit** transforment la dimension hors tout en dimension extérieure d'ouvrant
   (DEO = X − (a + b)), puis en feuillure et en vitrage ;
2. le **DTA** borne la baie, l'**abaque de dormant** borne le cadre selon la couleur ;
3. l'**abaque d'ouvrant** borne chaque vantail : limite de couleur, règle des 25 %, courbe
   d'épaisseur de verre (J079 : deux courbes plus bas), zone de renforcement ;
4. la **parclose** se choisit à l'épaisseur exacte du vitrage ;
5. la **tapée** se choisit à l'épaisseur d'isolant, dans la colonne du dormant employé ;
6. la **ferrure Roto NX** borne la feuillure et le poids du vantail, le **pivot bas** son poids.

Compléments du 21/10 : l'abaque d'ouvrant simple des 76272 / 76279 (aucune courbe de verre), le poids
d'ouvrant admissible selon la ferrure et l'acier du dormant, les courbes Roto NX de hauteur maxi par
poids de vitrage (OB 130 et 150 kg, Designo II 80 kg), l'AluClip Pro (76271 capoté A072), le soufflet
(abaque lu en inversant hauteur et largeur), le fixe latéral (DTA seul), la distance entre points de
verrouillage (INC-10), l'élargissement du meneau entre deux ouvrants.

Une hypothèse, affichée sur chaque verdict qu'elle touche : la LFF / HFF de Roto (fond de
feuillure du vantail) est prise égale à la DFO de profine (dimension de feuillure d'ouvrant). Aucune
source ne l'écrit ; c'est la lecture naturelle des deux définitions.

Trois règles de prudence, parce qu'un outil qui calcule donne l'air d'une certitude :

- **jamais d'interpolation** : entre deux graduations d'une courbe, la plus restrictive des deux ;
- **la précision de lecture compte** : à moins de ±3 cm d'une courbe ou ±2 cm d'un coin
  (précisions écrites sur la page des abaques), le verdict est « sur étude » ;
- **une case absente reste absente** : « - » en fin de courbe, abaque manquant, combinaison non
  nommée par la source → « sur étude », avec la raison.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from app.services import wiki_famille
from app.services.wiki_service import WikiSnapshot

PAGE_ABAQUES = "/profiles/systeme-76-abaques-dimensionnels.md"
PAGE_DEBIT = "/profiles/systeme-76-cotes-de-debit.md"
PAGE_DTA = "/certifications/dta-6-16-2334.md"
PAGE_PARCLOSES = "/profiles/perform76-parcloses.md"
PAGE_DORMANTS = "/profiles/perform76-dormants.md"
PAGE_OUVRANTS = "/profiles/perform76-ouvrants-et-battements.md"
PAGE_ROTO = "/quincaillerie/roto-nx-champs-application.md"
PAGE_TAPEES = "/profiles/perform76-tapees-et-isolation.md"
PAGE_PIVOT = "/quincaillerie/perform76-poignee-et-pivot.md"
PAGE_MENEAUX = "/profiles/perform76-meneaux.md"
PAGE_FIXE_LATERAL = "/procedures/porte-fenetre-fixe-lateral-systeme-76.md"
PAGES = (PAGE_ABAQUES, PAGE_DEBIT, PAGE_DTA, PAGE_PARCLOSES, PAGE_DORMANTS, PAGE_OUVRANTS,
         PAGE_TAPEES, PAGE_ROTO, PAGE_PIVOT)
HYPOTHESE_LFF = "hypothèse : LFF / HFF Roto = DFO profine"

PAUMELLES = {"P": "Paumelles P", "designo": "Designo II"}
VERSIONS_P = {"130": "130 kg", "150": "150 kg"}
VERSIONS_DESIGNO = {"80": "80 kg", "100": "100 kg", "report": "Report de charge, 80 à 150 kg"}
SECURITES = {"base": "Sécurité de base", "cdr1n": "CDR 1 N", "cdr2": "CDR 2 et CDR 2 N", "cdr3": "CDR 3"}

# Précisions de lecture écrites sur la page des abaques.
PRECISION_COURBE_CM = 3
PRECISION_COIN_CM = 2
PRECISION_DORMANT_M = 0.05
PRECISION_ROTO_MM = 25          # « précision de lecture ±25 mm » écrite sous chaque diagramme Roto NX
COURBES_VERRE = (12, 16, 20, 24, 28)

OK, HORS, ETUDE, INFO = "ok", "hors", "etude", "info"

CONFIGURATIONS = {
    "fixe": "Châssis fixe",
    "1v_of": "1 vantail OF",
    "1v_ob": "1 vantail OB",
    "2v_meneau": "2 vantaux OF sur meneau",
    "2v_battement": "2 vantaux OF à battement",
    "2v_fixe_lateral": "2 vantaux OF + fixe latéral",
    "soufflet": "Soufflet (basculant, 1 vantail)",
}
COULEURS = {
    "blanc": "Blanc",
    "standard": "Couleur standard",
    "ir_reflex": "Couleur IR-Reflex",
    "capot_alu": "Capot aluminium",
}


class DonneeIntrouvable(RuntimeError):
    """Un tableau attendu n'est plus dans le wiki, ou plus sous la forme attendue."""


# ---- lecture des tableaux markdown ---------------------------------------------------------

@dataclass
class Table:
    chemin: List[str]          # titres des sections englobantes, du plus haut au plus proche
    entete: List[str]
    lignes: List[List[str]]
    brut: List[List[str]]      # cellules non nettoyées (images, gras)
    localisation: str = ""     # la ligne « (schéma: …) » qui suit

    @property
    def section(self) -> str:
        return self.chemin[-1] if self.chemin else ""


def _cellules(ligne: str) -> List[str]:
    return [c.strip() for c in ligne.strip().strip("|").split("|")]


def _nettoie(c: str) -> str:
    c = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", c)
    c = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", c)
    return c.replace("**", "").replace("\\*", "*").strip()


def tables(body: str) -> List[Table]:
    out: List[Table] = []
    pile: List[Tuple[int, str]] = []
    lignes = body.splitlines()
    i = 0
    while i < len(lignes):
        l = lignes[i]
        m = re.match(r"^(#{1,6})\s+(.*?)\s*$", l)
        if m:
            niveau = len(m.group(1))
            pile = [(n, t) for n, t in pile if n < niveau] + [(niveau, m.group(2))]
            i += 1
            continue
        if l.strip().startswith("|") and i + 1 < len(lignes) and re.match(r"^\s*\|\s*:?-{2,}", lignes[i + 1]):
            entete = [_nettoie(c) for c in _cellules(l)]
            j, brut = i + 2, []
            while j < len(lignes) and lignes[j].strip().startswith("|"):
                brut.append(_cellules(lignes[j]))
                j += 1
            loc = ""
            for k in range(j, min(j + 4, len(lignes))):
                if lignes[k].strip().startswith("(schéma:"):
                    loc = lignes[k].strip()[1:-1]
                    break
            out.append(Table([t for _, t in pile], entete, [[_nettoie(c) for c in r] for r in brut], brut, loc))
            i = j
            continue
        i += 1
    return out


def _une(ts: List[Table], debut: str, page: str) -> Table:
    for t in ts:
        if " | ".join(t.entete).startswith(debut):
            return t
    raise DonneeIntrouvable(f"{page} : tableau « {debut}… » introuvable")


def _nombre(c: str) -> Optional[float]:
    c = c.strip().replace(" ", "").replace(" ", "").replace(",", ".")
    return float(c) if re.fullmatch(r"-?\d+(?:\.\d+)?", c) else None


def _ref(c: str) -> str:
    m = re.search(r"\b\d{5}\b", c)
    return m.group(0) if m else c.strip()


def _texte_section(body: str, titre: str) -> str:
    """Le texte d'une section (sans ses sous-sections plus profondes que la suivante de même niveau)."""
    m = re.search(r"^(#{1,6})\s+" + re.escape(titre) + r"\s*$", body, re.M)
    if not m:
        return ""
    niveau = len(m.group(1))
    fin = re.search(r"^#{1," + str(niveau) + r"}\s", body[m.end():], re.M)
    return body[m.end(): m.end() + fin.start()] if fin else body[m.end():]


# ---- les données, telles que le wiki les écrit ------------------------------------------------

@dataclass
class LimiteCouleur:
    hauteur_maxi: float
    largeur_a_hauteur_maxi: float
    coin_largeur: float
    coin_hauteur: float
    largeur_maxi: float
    hauteur_mini_a_largeur_maxi: float


@dataclass
class Abaque:
    ouvrants: List[str]
    renfort: str
    section: str
    localisation: str
    couleurs: Dict[str, LimiteCouleur]
    zones: Tuple[float, float]                       # (largeur, hauteur) de séparation, cm
    verre: Dict[int, Dict[int, Optional[float]]]     # largeur -> {courbe: hauteur maxi} ; vide si la page n'en trace aucune
    equivalents: List[Tuple[str, str]] = field(default_factory=list)   # (section, localisation) d'abaques aux limites identiques


@dataclass
class AbaqueBattement:
    ouvrants: List[str]
    renfort: str
    battements: List[str]
    section: str
    localisation: str
    courbes: Dict[str, Dict[str, Dict[int, Optional[float]]]]  # vent -> colonne -> largeur -> h


@dataclass
class CourbesRoto:
    """Un diagramme d'application Roto NX relevé tous les 100 mm de LFF : une courbe par poids de
    vitrage (kg/m²), plus la droite basse. Une hauteur absente (None) est « hors champ »."""
    maxi: Dict[int, Dict[int, Optional[float]]]   # kg/m² -> {LFF: HFF maxi}
    bas: Dict[int, float]                         # LFF -> HFF mini (0 : « bord bas », pas de minimum)
    deuxieme_compas: List[int]                    # LFF dont la zone est « 2ᵉ compas nécessaire »
    section: str
    localisation: str


@dataclass
class Donnees:
    statuts: Dict[str, str]
    dormant_limites: Dict[str, List[Tuple[float, float]]]
    abaques: List[Abaque]
    abaques_battement: List[AbaqueBattement]
    paumelles: List[Tuple[float, float, int]]
    debit_dormants: Dict[str, Dict[str, Any]]
    debit_meneaux: Dict[str, Dict[str, Any]]
    debit_ouvrants: Dict[str, Dict[str, Any]]
    debit_battements: Dict[Tuple[str, str], Dict[str, float]]
    renfort_battement: Dict[str, str]
    debit_seuil: float
    dta: List[Tuple[str, float, float]]
    dta_localisation: str
    parcloses_ouvrant: List[Dict[str, Any]]
    parcloses_dormant: List[Dict[str, Any]]
    dormants_proferm: List[str]
    ouvrants_proferm: List[str]
    poids_verre_kg_m2_mm: Optional[float]
    tapees: List[Dict[str, Any]]                       # une ligne par tapée : iso par dormant, coupe
    appuis_par_tapee: Dict[str, List[str]]             # tapée (sans .1) -> pièces d'appui compatibles
    iso_76171: Dict[float, Dict[str, str]]             # iso -> appui, dormant bas
    roto_p: Dict[Tuple[str, str], Dict[str, Any]]      # (version, sécurité) -> bornes, sources
    roto_designo: Dict[str, Dict[str, Any]]            # version -> bornes, OF admis
    pivot_kg: Optional[float]
    aluclip_pro: Dict[str, Any] = field(default_factory=dict)                          # abaques de l'ouvrant 76271 capoté A072
    roto_courbes: Dict[str, CourbesRoto] = field(default_factory=dict)                 # « P130 », « P150 », « D80 »
    roto_soufflet: Dict[str, Any] = field(default_factory=dict)                        # limite oblique, LFF « uniquement avec crémone verrou »
    meneau_compositions: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # meneau -> élargissements dessinés
    part_ouvrant: Dict[str, float] = field(default_factory=dict)                       # ouvrant -> largeur hors tout de son recouvrement au meneau
    poids_ferrure: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)  # (ferrure kg, acier) -> poids admissible
    verrouillage_cm: Dict[str, float] = field(default_factory=dict)                    # « paumelle » / « galets » -> distance maxi
    localisations: Dict[str, str] = field(default_factory=dict)


_CACHE: Dict[str, Any] = {"cle": None, "donnees": None}


def donnees(snap: WikiSnapshot) -> Donnees:
    cle = (id(snap), snap.loaded_at)
    if _CACHE["cle"] != cle:
        _CACHE["donnees"] = _charge(snap)
        _CACHE["cle"] = cle
    return _CACHE["donnees"]


def _corps(snap: WikiSnapshot, page: str) -> str:
    """Le corps de la page, suites comprises : une page coupée par le wiki se lit d'un seul tenant."""
    p = snap.pages.get(page)
    if p is None or p.missing:
        raise DonneeIntrouvable(f"page absente du wiki : {page}")
    return wiki_famille.corps_famille(snap, page)


def _charge(snap: WikiSnapshot) -> Donnees:
    loc: Dict[str, str] = {}
    # -- abaques
    corps = _corps(snap, PAGE_ABAQUES)
    ta = tables(corps)
    t = _une(ta, "Couleur | Sommets de la limite", PAGE_ABAQUES)
    loc["dormant"] = t.localisation
    dormant = {}
    for r in t.lignes:
        pts = [(float(a.replace(",", ".")), float(b.replace(",", ".")))
               for a, b in re.findall(r"\(\s*([\d,]+)\s*;\s*([\d,]+)\s*\)", r[1])]
        dormant[r[0]] = pts

    abaques: List[Abaque] = []
    batt: List[AbaqueBattement] = []
    sections: Dict[str, List[Table]] = {}
    for tb in ta:
        sections.setdefault(" > ".join(tb.chemin), []).append(tb)
    for cle_sec, tbs in sections.items():
        chemin = tbs[0].chemin
        titre = chemin[-1]
        m = re.fullmatch(r"Ouvrant (.+?) avec renfort (\S+)", titre)
        if m and len(chemin) >= 2 and chemin[-2].startswith("Abaques d'ouvrant simple"):
            couleurs, verre, loc_ab = {}, {}, ""
            for tb in tbs:
                h = " | ".join(tb.entete)
                if h.startswith("Couleur | Hauteur maxi"):
                    for r in tb.lignes:
                        v = [_nombre(c) for c in r[1:7]]
                        if None not in v:
                            # « blanc et IR-Reflex » : une ligne pour deux catégories
                            for nom in re.split(r"\s+et\s+", r[0].lower()):
                                couleurs[nom] = LimiteCouleur(*v)
                elif h.startswith("Largeur d'ouvrant (cm) | Verre 12"):
                    cols = [int(re.search(r"(\d+) mm", c).group(1)) for c in tb.entete[1:]]
                    for r in tb.lignes:
                        verre[int(_nombre(r[0]))] = {cols[k]: _nombre(c) for k, c in enumerate(r[1:])}
                    loc_ab = tb.localisation
                if not loc_ab:
                    loc_ab = tb.localisation
            zm = re.search(r"séparation entre A / C et B / D à (\d+) cm de largeur et (\d+) cm de hauteur",
                           _texte_section(corps, titre))
            # Les ouvrants 76272 / 76279 : « aucune courbe d'épaisseur de verre » écrit par la page ;
            # la table de verre est alors vide, pas absente (le titre sépare les références par « , » ou « / »).
            if couleurs and zm and (verre or "aucune courbe d'épaisseur de verre" in _texte_section(corps, titre)):
                abaques.append(Abaque([x.strip() for x in re.split(r"[/,]", m.group(1)) if x.strip()], m.group(2),
                                      titre, loc_ab, couleurs, (float(zm.group(1)), float(zm.group(2))), verre))
        mb = re.fullmatch(r"Ouvrant (.+?) avec (V[\w.]+), battement ([\d, ]+)", chemin[-2] if len(chemin) >= 2 else "")
        if mb and re.fullmatch(r"[\d,]+ kN/m²", titre):
            ab = next((b for b in batt if b.section == chemin[-2]), None)
            if ab is None:
                ab = AbaqueBattement([x.strip() for x in mb.group(1).split("/")], mb.group(2),
                                     re.findall(r"\d{5}", mb.group(3)), chemin[-2], "", {})
                batt.append(ab)
            for tb in tbs:
                if tb.entete[0].startswith("Largeur d'ouvrant"):
                    cols = [re.sub(r"\s*\(cm\)", "", c) for c in tb.entete[1:]]
                    ab.courbes[titre] = {c: {int(_nombre(r[0])): _nombre(r[k + 1]) for r in tb.lignes}
                                         for k, c in enumerate(cols)}
                    ab.localisation = ab.localisation or tb.localisation
    # la localisation des tableaux à deux vantaux suit le dernier tableau de la section
    for ab in batt:
        for tb in ta:
            if tb.chemin and tb.chemin[-2:-1] == [ab.section] and tb.localisation:
                ab.localisation = tb.localisation

    tp = _une(ta, "Hauteur d'ouvrant (cm) | Nombre de paumelles", PAGE_ABAQUES)
    paumelles = []
    for r in tp.lignes:
        a, b = re.findall(r"\d+", r[0])[:2]
        paumelles.append((float(a), float(b), int(_nombre(r[1]))))

    # -- AluClip Pro (ouvrant 76271 capoté A072, sans renfort acier) : la limite « dimension maxi » est
    # écrite en prose (simple et deux vantaux), les courbes de verre et de vent sont des tableaux
    ap: Dict[str, Any] = {}
    titre_simple, titre_2v = "AluClip Pro avec capot A072", "AluClip Pro à deux vantaux"
    prose_simple = re.sub(r"\s+", " ", _texte_section(corps, titre_simple).split("\n## ")[0])
    prose_2v = re.sub(r"\s+", " ", _texte_section(corps, titre_2v))
    cherche = re.compile(r"(\d+) cm de haut jusqu'à (\d+) cm de large, oblique jusqu'à \((\d+) ; (\d+)\), "
                         r"verticale à (\d+) cm(?: de large)? jusqu'à (\d+) cm(?: de haut)?")
    for tb in ta:
        if tb.chemin[-1:] == [titre_simple] and tb.entete[0].startswith("Largeur d'ouvrant") and tb.entete[1].startswith("Verre"):
            cols = [int(re.search(r"(\d+) mm", c).group(1)) for c in tb.entete[1:]]
            ap["verre"] = {int(_nombre(r[0])): {cols[k]: _nombre(c) for k, c in enumerate(r[1:])} for r in tb.lignes}
            ap["localisation"] = tb.localisation
    for cle, prose in (("limite", prose_simple), ("limite_2v", prose_2v)):
        mm = cherche.search(prose)
        if mm:
            h, lh, cl, ch, lm, hm = (float(x) for x in mm.groups())
            ap[cle] = LimiteCouleur(h, lh, cl, ch, lm, hm)
    mi = re.search(r"A072(?:\*\*)? \(IW ([\d,]+) cm⁴, IG ([\d,]+) cm⁴\)", prose_simple)
    mj = re.search(r"poids d'ouvrant de (\d+) kg il faut impérativement rajouter l'équerre J079", prose_simple)
    ap["iw"], ap["ig"] = (_nombre(mi.group(1)), _nombre(mi.group(2))) if mi else (None, None)
    ap["j079_kg"] = float(mj.group(1)) if mj else None
    ap["section"], ap["section_2v"] = titre_simple, titre_2v
    ap["battements"] = {}
    texte_2v = _texte_section(corps, titre_2v)
    morceaux = re.split(r"(!\[AluClip Pro, battement \d{5} avec V\d+\]\([^)]*\))", texte_2v)
    for k in range(1, len(morceaux) - 1, 2):
        mb2 = re.search(r"battement (\d{5}) avec (V\d+)", morceaux[k])
        for tb in tables(morceaux[k + 1]):
            if tb.entete[0].startswith("Largeur d'ouvrant") and tb.entete[1].startswith("0,8 kN/m²"):
                courbes = {re.match(r"([\d,]+) kN/m²", h).group(1): {int(_nombre(r[0])): _nombre(r[j + 1]) for r in tb.lignes}
                           for j, h in enumerate(tb.entete[1:])}
                ap["battements"][mb2.group(1)] = {"renfort": mb2.group(2), "courbes": courbes}
    ml = re.search(r"\((schéma: [^)]+)\)", texte_2v)
    ap["localisation_2v"] = ml.group(1) if ml else ""

    # -- poids d'ouvrant admissible selon la ferrure et l'acier du dormant (essais ift, directive TBDK)
    t = _une(ta, "Ferrure | Acier dans le dormant/meneau", PAGE_ABAQUES)
    loc["poids_ferrure"] = t.localisation
    poids_ferrure: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in t.lignes:
        mf, ma = re.match(r"(\d+) kg", r[0]), re.match(r"(sans|court|long)", r[1])
        if mf and ma:
            poids_ferrure[(mf.group(1), ma.group(1))] = {"kg": _nombre(r[4]), "acier": r[1], "rapport": r[5]}
    # -- distance entre points de verrouillage : deux phrases du manuel (INC-10)
    t = _une(ta, "Page | Formulation du manuel | Valeur", PAGE_ABAQUES)
    verrouillage = {}
    for r in t.lignes:
        mo = re.search(r"verrouillage \((paumelle|galets)", r[1])
        v = _nombre(r[2].replace("cm", ""))
        if mo and v:
            verrouillage[mo.group(1)] = v

    # -- cotes de débit
    td = tables(_corps(snap, PAGE_DEBIT))
    t = _une(td, "Dormant | DEO (mm) | DFO (mm)", PAGE_DEBIT)
    loc["debit_dormant"] = t.localisation
    dd = {r[0]: {"deo": _nombre(r[1]), "dfo": _nombre(r[2]), "section": t.section} for r in t.lignes}
    t = _une(td, "Meneau | DEO (mm)", PAGE_DEBIT)
    loc["debit_meneau"] = t.localisation
    dm = {r[0]: {"deo": _nombre(r[1]), "section": t.section} for r in t.lignes}
    t = _une(td, "Ouvrant | DFO (mm) | Vitrage (mm)", PAGE_DEBIT)
    loc["debit_ouvrant"] = t.localisation
    do = {r[0]: {"dfo": _nombre(r[1]), "vitrage": _nombre(r[2]), "section": t.section} for r in t.lignes}
    t = _une(td, "Dormant | Battement | DEO (mm)", PAGE_DEBIT)
    loc["debit_battement"] = t.localisation
    db = {}
    for r in t.lignes:
        a, b = re.search(r"X\s*[−-]\s*(\d+)", r[2]), re.search(r"X\s*[−-]\s*(\d+)", r[3])
        if a and b:
            db[(r[0], r[1])] = {"deo": float(a.group(1)), "dfo": float(b.group(1))}
    t = _une(td, "Battement | Débit du battement | Renfort associé", PAGE_DEBIT)
    rb = {_ref(r[0]): r[2] for r in t.lignes if r[2] not in ("—", "-", "")}
    t = _une(td, "Cas | Seuils concernés | DEO (mm)", PAGE_DEBIT)
    seuil = {_nombre(r[2]) for r in t.lignes}
    if len(seuil) != 1:
        raise DonneeIntrouvable(f"{PAGE_DEBIT} : DEO du seuil différente selon les cas")

    # -- DTA
    t = _une(tables(_corps(snap, PAGE_DTA)), "Configuration | H maxi (m) | L maxi (m)", PAGE_DTA)
    dta = [(r[0], _nombre(r[1]), _nombre(r[2])) for r in t.lignes]

    # -- parcloses
    tpar = tables(_corps(snap, PAGE_PARCLOSES))

    def parcloses(section: str) -> List[Dict[str, Any]]:
        for tb in tpar:
            if tb.section == section and tb.entete[:2] == ["Parclose", "Épaisseur de vitrage (mm)"]:
                out = []
                for r, b in zip(tb.lignes, tb.brut):
                    img = re.search(r"\((/assets/[^)]+)\)", b[-1])
                    out.append({"ref": r[0], "epaisseur": _nombre(r[1]), "cote": _nombre(r[2]),
                                "image": img.group(1) if img else None, "section": section})
                return out
        raise DonneeIntrouvable(f"{PAGE_PARCLOSES} : section « {section} » introuvable")

    # -- les références que PROFERM propose
    dp = [_ref(r[0]) for r in _une(tables(_corps(snap, PAGE_DORMANTS)), "Dormant | Usage", PAGE_DORMANTS).lignes]
    op = [_ref(r[0]) for r in _une(tables(_corps(snap, PAGE_OUVRANTS)), "Ouvrant | Profil", PAGE_OUVRANTS).lignes]

    # -- Roto NX : conversion du poids, champs d'application
    corps_roto = _corps(snap, PAGE_ROTO)
    m = re.search(r"1 mm/m² d'épaisseur de vitre ≙ ([\d,]+) kg", corps_roto)
    poids = _nombre(m.group(1)) if m else None
    tr = tables(corps_roto)
    bornes = lambda r: {"lff_min": _nombre(r[0]), "lff_max": _nombre(r[1]), "hff_min": _nombre(r[2]),
                        "hff_max": _nombre(r[3]), "pv_max": _nombre(r[4])}
    roto_p: Dict[Tuple[str, str], Dict[str, Any]] = {}
    cle_sec = {v: k for k, v in SECURITES.items()}

    def ajoute(version: str, securite: str, b: Dict[str, float], source: str, loc_: str) -> None:
        k = (version, cle_sec.get(securite, securite))
        cur = roto_p.get(k)
        if cur is None:
            roto_p[k] = dict(b, sources=[(source, loc_)])
            return
        # Deux documents Roto qui divergent (CTR-18) : la page dit de retenir les bornes les plus basses.
        for c in ("lff_min", "hff_min"):
            cur[c] = max(cur[c], b[c])
        for c in ("lff_max", "hff_max", "pv_max"):
            cur[c] = min(cur[c], b[c])
        cur["sources"].append((source, loc_))
        cur["ctr18"] = True

    t = _une(tr, "Type d'ouverture | Classe de sécurité | LFF mini", PAGE_ROTO)
    loc["roto_p"] = t.localisation
    for r in t.lignes:
        mv = re.fullmatch(r"Oscillo-battant, fenêtre rectangulaire, version (\d+) kg", r[0])
        if mv:
            ajoute(mv.group(1), r[1], bornes(r[2:]), "Cotes des champs d'application, côté paumelles P", t.localisation)
    for r in t.lignes:
        if r[0] == "Soufflet, fenêtre rectangulaire":
            ajoute("soufflet", r[1], bornes(r[2:]), "Cotes des champs d'application, côté paumelles P", t.localisation)
    t = _une(tr, "Version | Classe de sécurité | LFF mini", PAGE_ROTO)
    for r in t.lignes:
        mv = re.match(r"(\d+) kg", r[0])
        if mv:
            ajoute(mv.group(1), r[1], bornes(r[2:]), "Le catalogue de juin 2023 donne d'autres bornes, et une classe de plus",
                   t.localisation)
    t = _une(tr, "Configuration Designo II | LFF mini", PAGE_ROTO)
    roto_designo: Dict[str, Dict[str, Any]] = {}
    for r in t.lignes:
        if "report de charge" in r[0] and "sans report" not in r[0]:
            k = "report"
        else:
            mk = re.search(r"(\d+) kg$", r[0])
            k = mk.group(1) if mk else r[0]
        roto_designo[k] = dict(bornes(r[1:]), of=r[0].startswith("Fenêtre à la française"), libelle=r[0],
                               localisation=t.localisation)

    # -- diagrammes d'application : une courbe HFF maxi par poids de vitrage, relevés tous les 100 mm de LFF
    cles_diagramme = {"Version 130 kg": "P130", "Version 150 kg": "P150",
                      "Designo II, OF et OB sans report de charge, 80 kg": "D80"}
    roto_courbes: Dict[str, CourbesRoto] = {}
    for tb in tr:
        cle = cles_diagramme.get(tb.section)
        if cle is None or tb.entete[0] != "LFF (mm)" or not tb.entete[1].startswith("HFF maxi"):
            continue
        cols = {int(re.search(r"(\d+) kg/m²", h).group(1)): k for k, h in enumerate(tb.entete) if h.startswith("HFF maxi") and "droite" not in h}
        k_bas = next(k for k, h in enumerate(tb.entete) if "droite basse" in h)
        k_zone = tb.entete.index("Zone")
        maxi = {kg: {} for kg in cols}
        bas, compas = {}, []
        for r in tb.lignes:
            lff = int(_nombre(r[0]))
            for kg, k in cols.items():
                maxi[kg][lff] = None if r[k] == "hors champ" else _nombre(r[k])
            bas[lff] = 0.0 if r[k_bas] == "bord bas" else _nombre(r[k_bas])
            if r[k_zone].startswith("2ᵉ compas"):
                compas.append(lff)
        roto_courbes[cle] = CourbesRoto(maxi, bas, compas, tb.section, tb.localisation)

    # -- soufflet : la limite oblique du diagramme (segment droit) et les LFF réservées à la crémone verrou
    texte_roto = re.sub(r"[\s\u00a0\u202f]+", " ", corps_roto)
    chiffre = lambda x: float(re.sub(r"\D", "", x))
    mo = re.search(r"part du coin haut \(LFF ([\d ]+), HFF ([\d ]+)\) et descend jusqu'à environ ([\d ]+) mm de HFF au bord droit \(LFF ([\d ]+)\)", texte_roto)
    roto_soufflet = {
        "oblique": tuple(chiffre(g) for g in (mo.group(1), mo.group(2), mo.group(3), mo.group(4))) if mo else None,   # x0, y0, y1, x1
        "cremone": [(float(a), float(b_)) for a, b_ in re.findall(r"LFF (\d+) – (\d+) mm uniquement avec crémone verrou", texte_roto)]}

    # -- meneaux : élargissements dessinés au cahier PERFORM76, et part d'un ouvrant des paires basse et haute
    corps_men = _corps(snap, PAGE_MENEAUX)
    compos: Dict[str, List[Dict[str, Any]]] = {}
    for tb in tables(corps_men):
        mm_ = re.fullmatch(r"Cotes des élargissements du meneau (\d{5})", tb.section)
        if mm_ and tb.entete[:3] == ["Composition", "Largeur hors tout (mm)", "Vue (mm)"]:
            compos[mm_.group(1)] = [{"composition": r[0], "largeur": _nombre(r[1]), "vue": r[2], "section": tb.section,
                                     "localisation": tb.localisation} for r in tb.lignes]
    texte_men = re.sub(r"\s+", " ", corps_men)
    mp = re.search(r"\*\*(\d+)\*\* est la part d'un ouvrant de la paire basse, \*\*(\d+)\*\* celle d'un ouvrant de la paire haute", texte_men)
    corps_ouv = _corps(snap, PAGE_OUVRANTS)
    mh = re.search(r"les (\d+) mm de la paire basse et les (\d+) mm de la paire haute", re.sub(r"\s+", " ", corps_ouv))
    part_ouvrant: Dict[str, float] = {}
    if mp and mh:
        t = _une(tables(corps_ouv), "Ouvrant | Profil", PAGE_OUVRANTS)
        for r in t.lignes:
            hauteur = _nombre(r[5]) if len(r) > 5 else None
            if hauteur == float(mh.group(1)):
                part_ouvrant[_ref(r[0])] = float(mp.group(1))
            elif hauteur == float(mh.group(2)):
                part_ouvrant[_ref(r[0])] = float(mp.group(2))

    # -- tapées et isolation
    tt = tables(_corps(snap, PAGE_TAPEES))
    t = _une(tt, "Tapée | Cote propre (mm) | Iso sur", PAGE_TAPEES)
    loc["tapees"] = t.localisation
    colonnes = [(k, re.findall(r"\d{5}", h)) for k, h in enumerate(t.entete) if h.startswith("Iso sur")]
    tapees = []
    for r, b in zip(t.lignes, t.brut):
        img = re.search(r"\((/assets/[^)]+)\)", b[-1])
        iso = {dor: _nombre(r[k]) for k, dors in colonnes for dor in dors}
        tapees.append({"ref": r[0], "cote": _nombre(r[1]), "iso": iso, "image": img.group(1) if img else None})
    t = _une(tt, "Tapée | Épaisseur tapée (mm) | Pièce d'appui", PAGE_TAPEES)
    appuis = [re.search(r"Pièce d'appui (\S+)", h).group(1) for h in t.entete[2:]]
    appuis_par_tapee = {r[0].split(".")[0]: [a for a, c in zip(appuis, r[2:]) if c.strip().upper() == "X"]
                        for r in t.lignes}
    t = _une(tt, "Iso sur 76171 (mm) | Appui | Dormant bas", PAGE_TAPEES)
    iso_76171 = {_nombre(r[0]): {"appui": r[1], "dormant_bas": r[2]} for r in t.lignes}

    # -- pivot bas
    m = re.search(r"pivot bas d'une PERFORM76 est de \*\*(\d+) kg par ouvrant", _corps(snap, PAGE_PIVOT))
    pivot = float(m.group(1)) if m else None

    return Donnees(
        statuts={p: (snap.pages[p].status or "") for p in PAGES},
        dormant_limites=dormant, abaques=abaques, abaques_battement=batt, paumelles=paumelles,
        debit_dormants=dd, debit_meneaux=dm, debit_ouvrants=do, debit_battements=db,
        renfort_battement=rb, debit_seuil=seuil.pop(), dta=dta, dta_localisation=t.localisation,
        parcloses_ouvrant=parcloses("Cotes des parcloses d'ouvrant"),
        parcloses_dormant=parcloses("Cotes des parcloses de dormant"),
        dormants_proferm=[d for d in dp if d in dd], ouvrants_proferm=[o for o in op if o in do],
        poids_verre_kg_m2_mm=poids, tapees=tapees, appuis_par_tapee=appuis_par_tapee, iso_76171=iso_76171,
        roto_p=roto_p, roto_designo=roto_designo, pivot_kg=pivot, aluclip_pro=ap, roto_courbes=roto_courbes, roto_soufflet=roto_soufflet, meneau_compositions=compos,
        part_ouvrant=part_ouvrant, poids_ferrure=poids_ferrure,
        verrouillage_cm=verrouillage, localisations=loc,
    )


# ---- vitrage -------------------------------------------------------------------------------

@dataclass
class Vitrage:
    composition: str
    verre_mm: float        # somme des verres, sans intercalaires ni films : ce que lisent les abaques
    total_mm: float        # épaisseur totale : ce que tient la parclose


def lire_vitrage(composition: str) -> Vitrage:
    """« 4-16-4 », « 44.2-16-4 », « 4-12-4-12-4 » : verre, lame, verre… Un feuilleté « 44.2 » compte
    deux verres de 4 mm pour les abaques (le manuel additionne les couches de verre) et deux films
    de 0,38 mm pour l'épaisseur totale."""
    parts = [p for p in re.split(r"[-/ x+]+", composition.strip().replace(",", ".")) if p]
    if len(parts) % 2 == 0 or not parts:
        raise ValueError("composition attendue : verre-lame-verre, par exemple 4-16-4 ou 44.2-16-4")
    verre = total = 0.0
    for k, p in enumerate(parts):
        if k % 2 == 0:
            m = re.fullmatch(r"(\d)(\d)\.(\d)", p)
            if m:
                verre += int(m.group(1)) + int(m.group(2))
                total += int(m.group(1)) + int(m.group(2)) + 0.38 * int(m.group(3))
            elif re.fullmatch(r"\d+(?:\.\d+)?", p):
                verre += float(p)
                total += float(p)
            else:
                raise ValueError(f"verre illisible : « {p} »")
        else:
            if not re.fullmatch(r"\d+(?:\.\d+)?", p):
                raise ValueError(f"lame illisible : « {p} »")
            total += float(p)
    return Vitrage(composition, round(verre, 2), round(total, 2))


# ---- la saisie et le résultat --------------------------------------------------------------

@dataclass
class Saisie:
    configuration: str
    largeur_mm: float
    hauteur_mm: float
    dormant: str
    ouvrant: Optional[str] = None
    meneau: Optional[str] = None
    battement: Optional[str] = None
    bas: str = "dormant"          # « dormant » ou « seuil » (seuil alu A076)
    couleur: str = "blanc"
    vitrage: str = "4-16-4"
    j079: bool = False
    vent: str = "0,8"             # charge de vent des deux vantaux à battement, kN/m²
    isolant_mm: Optional[float] = None   # isolant intérieur à rattraper par une tapée ; vide = pas de contrôle
    paumelles: str = "P"                 # « P » ou « designo »
    version: str = "130"                 # P : 130 / 150 ; Designo II : 80 / 100 / report
    securite: str = "base"               # P seulement : base, cdr1n, cdr2, cdr3
    ferrure_poids: Optional[str] = None  # ferrure de 100 ou 130 kg (poids admissible) ; vide = pas de contrôle
    acier_dormant: str = "sans"          # acier dans le dormant ou le meneau : sans, court (55 mm), long
    distance_verrouillage_cm: Optional[float] = None   # plus grand écart entre deux points de verrouillage ; vide = pas de contrôle
    aluclip_pro: bool = False            # ouvrant 76271 capoté A072 (AluClip Pro), sans renfort acier


def _source(page: str, section: str = "", localisation: str = "") -> Dict[str, str]:
    return {"page": page, "section": section, "localisation": localisation}


def _controle(id_: str, titre: str, statut: str, detail: str, sources: List[Dict[str, str]],
              anomalies: Optional[List[str]] = None, valeurs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"id": id_, "titre": titre, "statut": statut, "detail": detail, "sources": sources,
            "anomalies": anomalies or [], "valeurs": valeurs or {}}


def _borne(valeur: float, limite: float, precision: float) -> str:
    """ok franchement sous la limite ; sur étude à moins de la précision de lecture ; hors au-delà."""
    if valeur <= limite - precision:
        return OK
    if valeur < limite + precision:
        return ETUDE
    return HORS


def _fmt(x: float, d: int = 0) -> str:
    s = f"{x:.{d}f}".replace(".", ",")
    return s


def _dans_polygone(x: float, y: float, pts: List[Tuple[float, float]]) -> bool:
    poly = pts + [(0.0, 0.0)]
    dedans = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x <= xi:
                dedans = not dedans
    return dedans


def _limite_polygone(pts: List[Tuple[float, float]], x: float, y: float, marge: float) -> str:
    """Point dans le polygone de limite, avec la précision de lecture en marge."""
    dedans = _dans_polygone(x, y, pts)
    # proche du bord : on teste le point décalé de la marge dans les quatre directions
    voisins = [_dans_polygone(x + dx, y + dy, pts) for dx, dy in ((marge, 0), (0, marge), (-marge, 0), (0, -marge))]
    if dedans and all(voisins):
        return OK
    if not dedans and not any(voisins):
        return HORS
    return ETUDE


def calcul_debit(d: Donnees, s: Saisie) -> Dict[str, Any]:
    """Dimension hors tout → DEO, DFO, vitrage de chaque vantail, étapes écrites comme le manuel."""
    dor = d.debit_dormants.get(s.dormant)
    if dor is None:
        raise ValueError(f"dormant {s.dormant} absent des cotes de débit")
    a = dor["deo"]
    a_bas = d.debit_seuil if s.bas == "seuil" else a
    etapes: List[Dict[str, str]] = []
    res: Dict[str, Any] = {"etapes": etapes, "vantaux": 0, "manque": None}
    if s.configuration == "fixe":
        return res
    ouv = d.debit_ouvrants.get(s.ouvrant or "")
    if ouv is None:
        raise ValueError(f"ouvrant {s.ouvrant} absent des cotes de débit")
    L, H = s.largeur_mm, s.hauteur_mm
    if s.configuration in ("1v_of", "1v_ob", "soufflet"):
        deo_l = L - 2 * a
        etapes.append({"etape": "Largeur extérieure d'ouvrant", "calcul": f"DEO = {_fmt(L)} − 2 × {_fmt(a)}", "resultat": _fmt(deo_l)})
        res["vantaux"] = 1
    elif s.configuration == "2v_meneau":
        men = d.debit_meneaux.get(s.meneau or "")
        if men is None:
            raise ValueError(f"meneau {s.meneau} absent des cotes de débit")
        x = L / 2
        deo_l = x - (a + men["deo"])
        etapes.append({"etape": "Demi-largeur au meneau", "calcul": f"X = {_fmt(L)} / 2", "resultat": _fmt(x, 1 if x % 1 else 0)})
        etapes.append({"etape": "Largeur extérieure d'ouvrant", "calcul": f"DEO = X − (a + b) = {_fmt(x, 1 if x % 1 else 0)} − ({_fmt(a)} + {_fmt(men['deo'])})", "resultat": _fmt(deo_l, 1 if deo_l % 1 else 0)})
        res["vantaux"] = 2
    elif s.configuration == "2v_battement":
        x = L / 2
        cle = (s.dormant, s.battement or "")
        if cle not in d.debit_battements:
            res["manque"] = (f"aucune cote de débit pour le dormant {s.dormant} avec le battement {s.battement} : "
                             "le dormant n'apparaît sur aucune planche de battement")
            return res
        n = d.debit_battements[cle]["deo"]
        deo_l = x - n
        etapes.append({"etape": "Axe du châssis au bord du dormant", "calcul": f"X = {_fmt(L)} / 2", "resultat": _fmt(x, 1 if x % 1 else 0)})
        etapes.append({"etape": "Largeur extérieure d'ouvrant", "calcul": f"DEO = X − {_fmt(n)}", "resultat": _fmt(deo_l, 1 if deo_l % 1 else 0)})
        res["vantaux"] = 2
    elif s.configuration == "2v_fixe_lateral":
        res["manque"] = ("aucune cote de débit pour deux vantaux avec un fixe latéral : le wiki ne donne le fixe latéral que pour une "
                         "porte-fenêtre à un seul ouvrant 76281 (Axe, X, Y), et le débit à battement se prend à l'axe du châssis")
        return res
    else:
        raise ValueError(f"configuration inconnue : {s.configuration}")
    deo_h = H - a - a_bas
    etapes.append({"etape": "Hauteur extérieure d'ouvrant",
                   "calcul": f"DEO = {_fmt(H)} − {_fmt(a)} − {_fmt(a_bas)}" + (" (seuil)" if s.bas == "seuil" else ""),
                   "resultat": _fmt(deo_h)})
    c = ouv["vitrage"]
    vl, vh = deo_l - 2 * c, deo_h - 2 * c
    etapes.append({"etape": "Vitrage de l'ouvrant", "calcul": f"DEO − 2 × {_fmt(c)} = {_fmt(vl, 1 if vl % 1 else 0)} × {_fmt(vh)}", "resultat": f"{_fmt(vl, 1 if vl % 1 else 0)} × {_fmt(vh)}"})
    res.update(deo=(deo_l, deo_h), dfo=(deo_l - 2 * ouv["dfo"], deo_h - 2 * ouv["dfo"]), vitrage=(vl, vh))
    return res


def abaque_simple(d: Donnees, ouvrant: Optional[str]) -> Optional[Abaque]:
    """L'abaque d'ouvrant simple d'un ouvrant. Quand la page en donne plusieurs (un par renfort) :
    s'ils ont exactement les mêmes limites, les renforts s'additionnent (76272 / 76279 : V326.Z ou
    V314.Z, mêmes limites, aucune courbe de verre) ; sinon le premier, comme avant — la saisie ne dit
    pas quel renfort l'atelier monte."""
    candidats = [a for a in d.abaques if ouvrant in a.ouvrants]
    if not candidats:
        return None
    premier = candidats[0]
    memes = [a for a in candidats[1:] if (a.couleurs, a.zones, a.verre) == (premier.couleurs, premier.zones, premier.verre)]
    if len(memes) != len(candidats) - 1:
        return premier
    return replace(premier, renfort=" ou ".join(a.renfort for a in candidats),
                   equivalents=[(a.section, a.localisation) for a in memes])


def _hauteur_courbe(courbe: Dict[int, Optional[float]], largeur_cm: float) -> Tuple[Optional[float], str]:
    """Hauteur maxi lue sur une courbe relevée tous les 10 cm, SANS interpolation : la plus
    restrictive des deux graduations voisines. Un « - » avant la première valeur est une courbe
    au-dessus du cadre (non limitante) ; après la dernière, une courbe non légendée (inconnue)."""
    grad = sorted(courbe)
    if largeur_cm < grad[0] or largeur_cm > grad[-1]:
        return None, "hors des graduations de l'abaque"
    bas = max(g for g in grad if g <= largeur_cm)
    haut = min(g for g in grad if g >= largeur_cm)
    valeurs = [v for v in courbe.values() if v is not None]
    premiere = min((g for g in grad if courbe[g] is not None), default=None)
    lus = []
    for g in {bas, haut}:
        v = courbe[g]
        if v is None:
            if premiere is not None and g < premiere:
                continue  # au-dessus du cadre : ne limite pas
            return None, f"courbe non relevée à {g} cm"
        lus.append(v)
    if not valeurs:
        return None, "aucune valeur relevée"
    if not lus:
        return float("inf"), "courbe au-dessus du cadre de l'abaque"
    return min(lus), (f"entre {bas} et {haut} cm, la plus restrictive" if bas != haut else f"à {bas} cm")


def _voisines(grad: List[int], x: float) -> List[int]:
    """Les deux graduations qui encadrent x (une seule si x tombe dessus ou sort des graduations)."""
    bas = [g for g in grad if g <= x]
    haut = [g for g in grad if g >= x]
    return sorted({max(bas) if bas else min(grad), min(haut) if haut else max(grad)})


def _sous_courbe(c: CourbesRoto, kgm2: int, lff: float, hff: float) -> bool:
    """HFF sous la courbe de ce poids de vitrage, sans interpolation : la plus restrictive des deux
    graduations de LFF voisines ; « hors champ » à l'une d'elles = hors champ."""
    hauteurs = [c.maxi[kgm2][g] for g in _voisines(sorted(c.maxi[kgm2]), lff)]
    return None not in hauteurs and hff <= min(hauteurs)


def _sur_droite_basse(c: CourbesRoto, lff: float, hff: float) -> bool:
    return hff >= max(c.bas[g] for g in _voisines(sorted(c.bas), lff))


def _avec_marge(test, lff: float, hff: float) -> str:
    """OK si le test passe au point et à ±25 mm dans les quatre directions ; HORS s'il échoue partout."""
    m = PRECISION_ROTO_MM
    res = [test(lff + dx, hff + dy) for dx, dy in ((0, 0), (m, 0), (-m, 0), (0, m), (0, -m))]
    return OK if all(res) else (HORS if not any(res) else ETUDE)


def champ_courbes(c: CourbesRoto, lff: float, hff: float, kgm2: float) -> Tuple[str, str, Dict[str, Any]]:
    """Le diagramme d'application à courbes par poids de vitrage : statut, phrase, valeurs.

    Entre deux courbes on prend la plus lourde (comme le manuel pour l'épaisseur de verre). Plus léger
    que la plus légère : sa région contient celle de cette courbe, donc dedans = ok, dehors = on ne
    sait pas (sur étude). Plus lourd que la plus lourde : région contenue dans la sienne, donc dehors =
    hors, dedans = on ne sait pas. La droite basse ne dépend pas du poids."""
    poids = sorted(c.maxi)
    lourde = next((p for p in poids if p >= kgm2), None)
    if lourde is not None and (lourde == poids[0] and kgm2 < poids[0]):
        courbe, mode = poids[0], "leger"
    elif lourde is None:
        courbe, mode = poids[-1], "lourd"
    else:
        courbe, mode = lourde, "exact"
    v_c = _avec_marge(lambda x, y: _sous_courbe(c, courbe, x, y), lff, hff)
    if mode == "leger":
        v_c = OK if v_c == OK else ETUDE
    elif mode == "lourd":
        v_c = HORS if v_c == HORS else ETUDE
    v_b = _avec_marge(lambda x, y: _sur_droite_basse(c, x, y), lff, hff)
    statut = max((v_c, v_b), key=[OK, ETUDE, HORS].index)
    h_lim = [c.maxi[courbe][g] for g in _voisines(sorted(c.maxi[courbe]), lff)]
    h_bas = max(c.bas[g] for g in _voisines(sorted(c.bas), lff))
    kg = _fmt(kgm2, 1 if kgm2 % 1 else 0)
    quoi = {"exact": f"courbe {courbe} kg/m² (vitrage de {kg} kg/m²)",
            "leger": f"courbe {courbe} kg/m², la plus légère tracée (vitrage de {kg} kg/m² : plus léger, le champ réel est plus large)",
            "lourd": f"courbe {courbe} kg/m², la plus lourde tracée (vitrage de {kg} kg/m² : plus lourd, le champ réel est plus étroit)"}[mode]
    haut = "hors champ" if None in h_lim else f"{_fmt(min(h_lim))} mm"
    phrase = (f"diagramme d'application, {quoi} : HFF {_fmt(hff)} mm pour {haut} maxi à LFF {_fmt(lff, 1 if lff % 1 else 0)} mm "
              f"(plus restrictive des deux graduations de 100 mm) et {_fmt(h_bas)} mm mini (droite basse), lecture ±{PRECISION_ROTO_MM} mm")
    compas = any(g in c.deuxieme_compas for g in _voisines(sorted(c.bas), lff))
    if compas:
        phrase += " ; 2ᵉ compas nécessaire à cette largeur"
    return statut, phrase, {"courbe_kg_m2": courbe, "mode": mode, "hff_maxi": None if None in h_lim else min(h_lim),
                            "hff_mini": h_bas, "deuxieme_compas": compas}


def _soufflet_roto(d: Donnees, b: Dict[str, Any], lff: float, hff: float, v: str, det: str,
                   anomalies: List[str]) -> Tuple[str, str, List[str]]:
    """Le diagramme du soufflet : au-delà de LFF 1 200 mm, la HFF est bornée par un segment droit
    descendant (lu à ±25 mm), et les LFF les plus étroites exigent la crémone verrou (INC-193)."""
    obl = d.roto_soufflet.get("oblique")
    if obl is not None:
        x0, y0, y1, x1 = obl
        v_o = _avec_marge(lambda x, y: x <= x0 or y <= y0 + (x - x0) * (y1 - y0) / (x1 - x0), lff, hff)
        if v_o != OK:
            det += (f" Limite oblique du diagramme (de LFF {_fmt(x0)} × HFF {_fmt(y0)} à LFF {_fmt(x1)} × HFF {_fmt(y1)}, "
                    f"lecture ±{PRECISION_ROTO_MM} mm) : " + ("hors champ." if v_o == HORS else "à moins de la précision de lecture."))
        v = max((v, v_o), key=[OK, ETUDE, HORS].index)
    for bas_, haut_ in d.roto_soufflet.get("cremone", []):
        if bas_ == b["lff_min"] and bas_ <= lff <= haut_:
            det += f" LFF {_fmt(bas_)} à {_fmt(haut_)} mm uniquement avec crémone verrou."
            anomalies = anomalies + ["INC-193"]
    return v, det, anomalies


def _roto(d: Donnees, s: Saisie, deb: Dict[str, Any], poids: Optional[float], verre_mm: float) -> Dict[str, Any]:
    """Champ d'application Roto NX sur la feuillure du vantail, avec LFF / HFF = DFO : la boîte de
    bornes, puis, quand le manuel en trace, les courbes de hauteur maxi par poids de vitrage (kg/m²)
    et la droite basse. Les courbes s'ajoutent à la boîte, jamais à sa place : c'est l'intersection,
    donc plus restrictive. Le wiki ne dit pas si ces diagrammes valent aussi pour les classes CDR
    (leurs bornes de boîte sont plus serrées de toute façon)."""
    lff, hff = deb["dfo"]
    titre = "Ferrure Roto NX"
    ob = s.configuration == "1v_ob"
    soufflet = s.configuration == "soufflet"
    if soufflet and s.paumelles == "designo":
        return _controle("roto", titre, ETUDE,
                         "Le champ d'application du soufflet côté paumelles Designo II n'est pas relevé en tableau dans le wiki : "
                         "choisir les paumelles P, ou voir la page des configurations Roto NX KSR du soufflet.",
                         [_source(PAGE_ROTO, "Cotes des champs d'application, côté paumelles Designo II")])
    if s.paumelles == "designo":
        b = d.roto_designo.get(s.version)
        srcs = [_source(PAGE_ROTO, "Cotes des champs d'application, côté paumelles Designo II", b["localisation"] if b else "")]
        if b is None:
            return _controle("roto", titre, ETUDE, f"Aucun champ Designo II pour « {s.version} ».", srcs)
        if not ob and not b["of"]:
            return _controle("roto", titre, HORS, f"« {b['libelle']} » ne s'applique qu'en oscillo-battant.", srcs)
        lib, anomalies = f"Designo II, {VERSIONS_DESIGNO.get(s.version, s.version)}", []
    else:
        srcs = [_source(PAGE_ROTO, "Cotes des champs d'application, côté paumelles P", d.localisations.get("roto_p", ""))]
        if not ob and not soufflet:
            # un trou de la ferrure choisie, pas de la menuiserie : il ne décide pas du verdict
            return _controle("roto", titre, INFO,
                             "Non contrôlée : côté paumelles P, les champs d'application ne sont donnés que pour "
                             "l'oscillo-battant. Pour un ouvrant à la française, choisir Designo II.",
                             srcs)
        b = d.roto_p.get(("soufflet", "base") if soufflet else (s.version, s.securite))
        if b is None:
            return _controle("roto", titre, ETUDE,
                             f"Aucun champ Roto NX pour la version {s.version} kg en {SECURITES.get(s.securite, s.securite)}.", srcs)
        srcs = [_source(PAGE_ROTO, sec, loc_) for sec, loc_ in b["sources"]]
        lib = ("paumelles P, soufflet, sécurité de base" if soufflet else
               f"paumelles P, version {s.version} kg, {SECURITES.get(s.securite, s.securite)}")
        anomalies = (["CTR-18"] if b.get("ctr18") else []) + (["INC-13"] if s.version == "150" and not soufflet else [])
    ecarts = []
    if lff < b["lff_min"] or lff > b["lff_max"]:
        ecarts.append(f"LFF {_fmt(lff, 1 if lff % 1 else 0)} mm hors de {_fmt(b['lff_min'])} à {_fmt(b['lff_max'])} mm")
    if hff < b["hff_min"] or hff > b["hff_max"]:
        ecarts.append(f"HFF {_fmt(hff)} mm hors de {_fmt(b['hff_min'])} à {_fmt(b['hff_max'])} mm")
    if poids is not None and poids > b["pv_max"]:
        ecarts.append(f"{_fmt(poids, 1)} kg de verre pour {_fmt(b['pv_max'])} kg de poids de vantail maxi")
    v = HORS if ecarts else OK
    det = (f"{lib} : " + (" ; ".join(ecarts) if ecarts else
           f"feuillure {_fmt(lff, 1 if lff % 1 else 0)} × {_fmt(hff)} mm dans {_fmt(b['lff_min'])}–{_fmt(b['lff_max'])} × "
           f"{_fmt(b['hff_min'])}–{_fmt(b['hff_max'])} mm, {_fmt(b['pv_max'])} kg maxi")
           + f" ({HYPOTHESE_LFF}" + (" ; verre seul pour le poids)." if poids is not None else ")."))
    cle = None if soufflet else (("P" + s.version) if s.paumelles != "designo" else ("D" + s.version))
    courbes = d.roto_courbes.get(cle)
    valeurs: Dict[str, Any] = {}
    if soufflet:
        v, det, anomalies = _soufflet_roto(d, b, lff, hff, v, det, anomalies)
    if courbes is not None and d.poids_verre_kg_m2_mm:
        kgm2 = verre_mm * d.poids_verre_kg_m2_mm
        v_c, phrase, valeurs = champ_courbes(courbes, lff, hff, kgm2)
        det += " Puis le " + phrase + "."
        srcs = srcs + [_source(PAGE_ROTO, "Diagrammes d'application, ferrure OB, fenêtre rectangulaire" if cle[0] == "P" else courbes.section,
                               courbes.localisation)]
        v = max((v, v_c), key=[OK, ETUDE, HORS].index)
    elif courbes is None and not soufflet:
        det += (" Les courbes de poids de vitrage de ce diagramme ne sont pas relevées dans le wiki en tableau : "
                "seules les bornes ci-dessus sont contrôlées.")
    return _controle("roto", titre, v, det, srcs, anomalies,
                     dict(valeurs, lff=lff, hff=hff, bornes={k: b[k] for k in ("lff_min", "lff_max", "hff_min", "hff_max", "pv_max")},
                          hypothese=HYPOTHESE_LFF))


def _poids_verre(d: Donnees, vit: Vitrage, deb: Dict[str, Any]) -> Optional[float]:
    """Poids du verre d'un vantail, kg : épaisseur de verre × kg/mm/m² (Roto) × surface de vitrage."""
    if not d.poids_verre_kg_m2_mm or not deb.get("vitrage"):
        return None
    vl, vh = deb["vitrage"]
    return vit.verre_mm * d.poids_verre_kg_m2_mm * (vl / 1000) * (vh / 1000)


def abaque_aluclip(d: Donnees, s: Saisie) -> Optional[Abaque]:
    """L'abaque AluClip Pro (ouvrant 76271 capoté A072) sous la forme d'un abaque d'ouvrant simple :
    la limite « dimension maxi » unique, comptée comme celle du blanc (le capot aluminium se dimensionne
    comme un ouvrant blanc), la table de verre, pas de zone. Deux vantaux à battement : sa limite commune."""
    ap = d.aluclip_pro
    deux = s.configuration == "2v_battement"
    lim = ap.get("limite_2v" if deux else "limite")
    if lim is None or not ap.get("verre"):
        return None
    return Abaque(["76271"], "capot A072, sans renfort acier", ap["section_2v"] if deux else ap["section"],
                  (ap["localisation_2v"] if deux else ap["localisation"]), {"blanc": lim}, (0.0, 0.0), ap["verre"])


def _vent_aluclip(d: Donnees, s: Saisie, wl: float, wh: float) -> Dict[str, Any]:
    """Courbe de vent de l'AluClip Pro à deux vantaux : une courbe par battement (76471 / 76472), lue sur
    la coupe dessinée — le titre du premier abaque se trompe de battement (INC-54)."""
    ap = d.aluclip_pro
    titre = f"Charge de vent {s.vent} kN/m²"
    srcs = [_source(PAGE_ABAQUES, ap["section_2v"], ap["localisation_2v"])]
    b = ap["battements"].get(s.battement or "")
    if b is None:
        return _controle("vent", titre, ETUDE,
                         f"L'AluClip Pro à deux vantaux n'a d'abaque que pour les battements {' et '.join(sorted(ap['battements']))} : "
                         f"rien pour le battement {s.battement}.", srcs, ["INC-54"])
    courbe = b["courbes"].get(s.vent)
    if courbe is None:
        return _controle("vent", titre, ETUDE, f"Pas de courbe {s.vent} kN/m² sur l'abaque AluClip Pro du battement {s.battement}.",
                         srcs, ["INC-54"])
    h_c, comment = _hauteur_courbe(courbe, wl)
    if h_c is None:
        v, det = ETUDE, f"Courbe {s.vent} kN/m², battement {s.battement} avec {b['renfort']} : {comment}."
    elif h_c == float("inf"):
        v, det = OK, f"Courbe {s.vent} kN/m², battement {s.battement} avec {b['renfort']}, au-dessus du cadre à cette largeur : elle ne limite pas."
    else:
        v = _borne(wh, h_c, PRECISION_COURBE_CM)
        det = (f"Hauteur {_fmt(wh, 1)} cm pour {_fmt(h_c)} cm sur la courbe {s.vent} kN/m² du battement {s.battement} avec "
               f"{b['renfort']} ({comment}).")
    return _controle("vent", titre, v, det, srcs, ["INC-54"])


def _renfort_aluclip(d: Donnees, s: Saisie, e: float, poids: Optional[float]) -> List[Dict[str, Any]]:
    """AluClip Pro : pas de renfort acier, et l'équerre J079 des quatre angles dès 40 kg d'ouvrant.
    Au-delà de 12 mm de verre le manuel exige le renforcement total, que ce profilé n'a pas : sur étude."""
    ap = d.aluclip_pro
    srcs = [_source(PAGE_ABAQUES, ap["section"], ap["localisation"]), _source(PAGE_ABAQUES, "Équerre de feuillure J079")]
    inerties = (f" (IW {_fmt(ap['iw'], 1)} cm⁴, IG {_fmt(ap['ig'], 1)} cm⁴)" if ap.get("iw") is not None else "")
    ctl: List[Dict[str, Any]] = []
    if e > 12:
        ctl.append(_controle("renfort", "Renforcement exigé", ETUDE,
                             f"Plus de 12 mm de verre ({_fmt(e, 1 if e % 1 else 0)} mm) : le manuel exige le renforcement total, "
                             "or l'ouvrant 76271 capoté A072 n'a pas de renfort acier ; il ne dit pas comment les concilier.",
                             srcs, valeurs={"zone": None, "renfort": None}))
    else:
        ctl.append(_controle("renfort", "Renforcement exigé", INFO,
                             f"Pas de renfort acier : l'ouvrant 76271 est capoté par l'aluminium A072{inerties}, le manuel dit le "
                             "renforcement des ouvrants fondamentalement pas nécessaire.", srcs, valeurs={"zone": None, "renfort": None}))
    seuil = ap.get("j079_kg")
    if seuil is not None and poids is not None:
        pk = _fmt(poids, 1)
        if poids >= seuil and not s.j079:
            ctl.append(_controle("j079", "Équerre de feuillure J079", HORS,
                                 f"Le verre seul pèse {pk} kg : à partir de {_fmt(seuil)} kg d'ouvrant, l'équerre J079 est "
                                 "impérative dans les 4 angles. La cocher dans la saisie.", srcs))
        elif poids >= seuil:
            ctl.append(_controle("j079", "Équerre de feuillure J079", OK,
                                 f"Le verre seul pèse {pk} kg (≥ {_fmt(seuil)} kg) : J079 dans les 4 angles, comme exigé.", srcs))
        else:
            ctl.append(_controle("j079", "Équerre de feuillure J079", INFO,
                                 f"Le verre seul pèse {pk} kg ; le poids de l'ouvrant avec profilé et capot n'est pas connu (hypothèse : "
                                 f"verre seul). J079 impérative dès {_fmt(seuil)} kg d'ouvrant.", srcs))
    return ctl


def _poids_ferrure(d: Donnees, s: Saisie, poids: Optional[float]) -> Dict[str, Any]:
    """Poids d'ouvrant admissible de la ferrure (100 ou 130 kg) selon l'acier du dormant ou du meneau.
    Valeurs indicatives (essais ift Rosenheim, TBDK) : le fabricant doit les garantir. Le wiki ne donne
    pas le poids des profilés, donc seul le verre est compté (hypothèse affichée) : un dépassement par le
    verre seul est certain, un « sous la limite » ne dit pas que l'ouvrant complet y est."""
    titre = "Poids d'ouvrant selon la ferrure"
    srcs = [_source(PAGE_ABAQUES, "Poids d'ouvrant admissible selon la ferrure", d.localisations.get("poids_ferrure", ""))]
    anomalies = ["VER-25"] if s.acier_dormant == "long" else []
    b = d.poids_ferrure.get((s.ferrure_poids or "", s.acier_dormant))
    if b is None:
        return _controle("poids_ferrure", titre, ETUDE,
                         f"Aucune ligne « ferrure {s.ferrure_poids} kg, acier {s.acier_dormant} » dans le tableau des poids admissibles.",
                         srcs, anomalies)
    lib = f"ferrure {s.ferrure_poids} kg, acier {b['acier']} dans le dormant ou le meneau"
    if poids is None:
        return _controle("poids_ferrure", titre, ETUDE,
                         f"{lib} : {_fmt(b['kg'])} kg admissibles ; le poids du verre n'a pas pu être calculé.", srcs, anomalies,
                         {"admissible_kg": b["kg"]})
    v = HORS if poids > b["kg"] else OK
    det = (f"{lib} : {_fmt(poids, 1)} kg de verre pour {_fmt(b['kg'])} kg d'ouvrant admissibles"
           + (" : le verre seul dépasse la limite." if v == HORS else
              " ; profilés et renfort non comptés (hypothèse : verre seul), valeur indicative à faire garantir par le fabricant."))
    return _controle("poids_ferrure", titre, v, det, srcs, anomalies,
                     {"admissible_kg": b["kg"], "poids_kg": round(poids, 1), "rapport": b["rapport"]})


def _meneau(d: Donnees, s: Saisie) -> List[Dict[str, Any]]:
    """Deux informations sur le meneau, aucune ne décide du verdict : l'élargissement dessiné au cahier PERFORM76
    entre les deux ouvrants (largeur hors tout, vue) et l'absence de toute courbe de vent pour un meneau."""
    srcs = [_source(PAGE_MENEAUX, f"Cotes des élargissements du meneau {s.meneau}")]
    ctl = []
    lignes = d.meneau_compositions.get(s.meneau or "", [])
    part = d.part_ouvrant.get(s.ouvrant or "")
    if lignes and part is not None:
        seul = lignes[0]["composition"]
        cible = f"{_fmt(part)} + {seul} + {_fmt(part)}"
        trouve = next((r for r in lignes if r["composition"] == cible), None)
        srcs = [_source(PAGE_MENEAUX, lignes[0]["section"], lignes[0]["localisation"])]
        if trouve:
            ctl.append(_controle("meneau", f"Meneau {s.meneau} entre deux ouvrants", INFO,
                                 f"{cible} : {_fmt(trouve['largeur'])} mm hors tout, vue {trouve['vue']} mm entre les deux ouvrants {s.ouvrant}.",
                                 srcs, valeurs={"composition": cible, "largeur": trouve["largeur"], "vue": trouve["vue"]}))
        else:
            ctl.append(_controle("meneau", f"Meneau {s.meneau} entre deux ouvrants", INFO,
                                 f"Le cahier PERFORM76 ne dessine pas la composition {cible} pour le meneau {s.meneau} "
                                 "(il ne donne que : " + ", ".join(r["composition"] for r in lignes[1:]) + ").", srcs))
    ctl.append(_controle("vent_meneau", "Charge de vent sur le meneau", INFO,
                         "Le wiki ne donne aucune courbe de vent pour le meneau : les abaques à courbes de vent (0,8 et 1,2 kN/m²) "
                         "sont ceux des deux vantaux à battement. Chaque vantail est vérifié sur l'abaque d'ouvrant simple ; "
                         "la tenue du meneau et de son renfort au vent n'est pas contrôlée.",
                         [_source(PAGE_ABAQUES, "Abaques à deux vantaux avec battement")]))
    return ctl


def _verrouillage(d: Donnees, s: Saisie) -> Dict[str, Any]:
    """Distance entre points de verrouillage : le manuel écrit 70 cm (paumelles) et 80 cm (galets) — INC-10.
    En dessous de la plus petite : bon pour les deux lectures ; entre les deux : sur étude."""
    srcs = [_source(PAGE_ABAQUES, "Points de verrouillage")]
    p, g = d.verrouillage_cm.get("paumelle"), d.verrouillage_cm.get("galets")
    if p is None or g is None:
        return _controle("verrouillage", "Points de verrouillage", ETUDE,
                         "Les distances maximales du manuel n'ont pas pu être lues.", srcs, ["INC-10"])
    bas, haut = min(p, g), max(p, g)
    x = s.distance_verrouillage_cm
    xs = _fmt(x, 1 if x % 1 else 0)
    v = OK if x <= bas else (ETUDE if x <= haut else HORS)
    det = {OK: f"{xs} cm entre points de verrouillage : dans les deux limites du manuel ({_fmt(p)} cm paumelles, {_fmt(g)} cm galets).",
           ETUDE: f"{xs} cm entre points de verrouillage : le manuel écrit {_fmt(p)} cm pour les paumelles et {_fmt(g)} cm pour les galets, "
                  f"sans dire de quel organe parle chacune (INC-10) ; retenir {_fmt(bas)} cm en attendant la réponse de profine.",
           HORS: f"{xs} cm entre points de verrouillage : au-delà des deux limites du manuel ({_fmt(p)} et {_fmt(g)} cm)."}[v]
    return _controle("verrouillage", "Points de verrouillage", v, det, srcs, ["INC-10"], {"paumelle_cm": p, "galets_cm": g})


def _isolant(d: Donnees, s: Saisie) -> Dict[str, Any]:
    """La tapée qui rattrape l'isolant, lue dans la colonne du dormant employé."""
    iso = float(s.isolant_mm)
    srcs = [_source(PAGE_TAPEES, "Cotes des tapées par dormant", d.localisations.get("tapees", ""))]
    offertes = [(t, t["iso"].get(s.dormant)) for t in d.tapees]
    offertes = [(t, e) for t, e in offertes if e is not None]
    if not offertes:
        return _controle("isolant", "Isolant et tapée", ETUDE,
                         f"Le tableau des tapées PERFORM76 n'a pas de colonne pour le dormant {s.dormant}.", srcs)
    exact = next((t for t, e in offertes if abs(e - iso) < 0.5), None)
    if exact is None:
        dessous = max((e for _, e in offertes if e < iso), default=None)
        dessus = min((e for _, e in offertes if e > iso), default=None)
        propo = " ou ".join(_fmt(x) for x in (dessous, dessus) if x is not None)
        return _controle("isolant", "Isolant et tapée", ETUDE,
                         f"Aucune tapée ne donne {_fmt(iso)} mm d'isolant sur le dormant {s.dormant} : "
                         f"la demande se traite en {propo} mm. Épaisseurs possibles sur ce dormant : "
                         + ", ".join(_fmt(e) for _, e in offertes) + " mm.",
                         srcs, valeurs={"tapees": []})
    nom = "sans tapée" if exact["ref"] == "sans tapée" else f"tapée {exact['ref']}"
    det = f"{_fmt(iso)} mm d'isolant sur le dormant {s.dormant} : {nom}"
    if exact["ref"] != "sans tapée":
        det += f" (cote propre {_fmt(exact['cote'])} mm)"
        appuis = d.appuis_par_tapee.get(exact["ref"].split(".")[0], [])
        if appuis:
            det += ", pièces d'appui compatibles " + ", ".join(appuis)
            srcs.append(_source(PAGE_TAPEES, "Correspondance avec les pièces d'appui"))
    det += "."
    ligne = d.iso_76171.get(iso) if s.dormant == "76171" else None
    if ligne:
        srcs.append(_source(PAGE_TAPEES, "La contrainte du 76171 au-delà de 155 mm"))
        if ligne["dormant_bas"] != "76171":
            det += f" Au-delà de 155 mm sur un 76171, le dormant bas devient un {ligne['dormant_bas']}, avec l'appui {ligne['appui']}."
        else:
            det += f" Appui {ligne['appui']}."
    return _controle("isolant", "Isolant et tapée", OK, det, srcs,
                     valeurs={"tapees": [dict(exact, epaisseur=iso)] if exact.get("image") else []})


def verifier(snap: WikiSnapshot, s: Saisie) -> Dict[str, Any]:
    d = donnees(snap)
    if s.configuration not in CONFIGURATIONS:
        raise ValueError("configuration inconnue")
    if s.couleur not in COULEURS:
        raise ValueError("couleur inconnue")
    vit = lire_vitrage(s.vitrage)
    if s.aluclip_pro and (s.ouvrant != "76271" or s.couleur != "capot_alu" or s.configuration == "fixe"):
        raise ValueError("AluClip Pro : ouvrant 76271 capoté A072, couleur « Capot aluminium », avec au moins un vantail")
    ctl: List[Dict[str, Any]] = []
    L, H = s.largeur_mm, s.hauteur_mm

    # 1. le débit : de la dimension hors tout à l'ouvrant
    deb = calcul_debit(d, s)
    if deb.get("manque"):
        batt_ = s.configuration == "2v_battement"
        ctl.append(_controle("debit", "Cotes de débit", ETUDE, deb["manque"][0].upper() + deb["manque"][1:] + ".",
                             [_source(PAGE_DEBIT, "Cotes de débit d'un dormant recevant un battement", d.localisations.get("debit_battement", ""))]
                             if batt_ else [_source(PAGE_FIXE_LATERAL, "Cotes de débit")],
                             ["VER-21"] if batt_ and s.dormant == "76185" else []))
    elif deb["vantaux"]:
        ctl.append(_controle("debit", "Cotes de débit", INFO,
                             f"Chaque vantail mesure {_fmt(deb['deo'][0], 1 if deb['deo'][0] % 1 else 0)} × {_fmt(deb['deo'][1])} mm hors tout d'ouvrant (DEO).",
                             [_source(PAGE_DEBIT, d.debit_dormants[s.dormant]["section"], d.localisations.get("debit_dormant", ""))],
                             ["VER-22"] if s.bas == "seuil" else [],
                             {"etapes": deb["etapes"]}))

    # 2. le DTA : la baie
    lib = {"1v_of": ["1 vantail OF"], "1v_ob": ["1 vantail OB"], "2v_meneau": ["2 vantaux OF"],
           "2v_battement": ["2 vantaux OF"], "2v_fixe_lateral": ["2 vantaux OF + fixe latéral"],
           "soufflet": ["Soufflet (loqueteaux)"]}.get(s.configuration, [])
    lignes = [(c, h, l) for c, h, l in d.dta if c in lib]
    src_dta = [_source(PAGE_DTA, "2.2.3.7 Dimensions maximales", d.dta_localisation)]
    if not lignes:
        ctl.append(_controle("dta", "Dimensions maximales de baie (DTA)", INFO,
                             "Le DTA ne donne pas de dimension maximale pour un châssis fixe.", src_dta))
    else:
        verdicts = []
        for c, h, l in lignes:
            v = HORS if (H / 1000 > h or L / 1000 > l) else OK
            verdicts.append((v, c, h, l))
        meilleur = min(verdicts, key=lambda t: t[0] != OK)
        v, c, h, l = meilleur
        couples = " ou ".join(f"{_fmt(hh, 2)} × {_fmt(ll, 2)} m" for _, _, hh, ll in verdicts)
        ctl.append(_controle("dta", "Dimensions maximales de baie (DTA)", v,
                             (f"{_fmt(H/1000, 2)} × {_fmt(L/1000, 2)} m (H × L) dans la limite {couples} du {c}."
                              if v == OK else
                              f"{_fmt(H/1000, 2)} × {_fmt(L/1000, 2)} m (H × L) dépasse {couples} du {c}. "
                              "Au-delà, seul un Certificat de Qualification du menuisier peut l'admettre."),
                             src_dta, valeurs={"h_maxi": h, "l_maxi": l}))
        if s.configuration == "soufflet":
            ctl[-1]["detail"] += " Le DTA donne ce soufflet manœuvré par loqueteaux ; il est appliqué ici à tout soufflet (hypothèse)."

    # 3. l'abaque de dormant
    nom_dormant = {"blanc": "Blanc", "ir_reflex": "Couleur IR-Reflex", "capot_alu": "Capot aluminium",
                   "standard": "Couleur standard"}[s.couleur]
    pts = d.dormant_limites.get(nom_dormant)
    src_dor = [_source(PAGE_ABAQUES, "Dimensions maximales de dormant", d.localisations.get("dormant", ""))]
    if not pts:
        ctl.append(_controle("dormant", "Abaque de dormant", ETUDE, f"Pas de limite de dormant pour « {nom_dormant} ».", src_dor))
    else:
        v = _limite_polygone(pts, L / 1000, H / 1000, PRECISION_DORMANT_M)
        ctl.append(_controle("dormant", "Abaque de dormant", v,
                             {OK: f"Le cadre {_fmt(L/1000, 2)} × {_fmt(H/1000, 2)} m (L × H) est dans la limite {COULEURS[s.couleur].lower()}.",
                              ETUDE: f"Le cadre {_fmt(L/1000, 2)} × {_fmt(H/1000, 2)} m est à moins de {_fmt(PRECISION_DORMANT_M, 2)} m de la limite {COULEURS[s.couleur].lower()} (précision de lecture).",
                              HORS: f"Le cadre {_fmt(L/1000, 2)} × {_fmt(H/1000, 2)} m sort de la limite {COULEURS[s.couleur].lower()}."}[v],
                             src_dor, valeurs={"limite": pts, "point": [L / 1000, H / 1000]}))

    if s.configuration == "2v_meneau" and not deb.get("manque"):
        ctl.extend(_meneau(d, s))

    # 4. l'abaque d'ouvrant, 5. le verre, 6. le renforcement
    graphe = None
    if deb["vantaux"] and not deb.get("manque"):
        wl, wh = deb["deo"][0] / 10, deb["deo"][1] / 10   # cm
        if s.configuration == "soufflet":
            wl, wh = wh, wl   # « châssis basculants : mêmes diagrammes en inversant la hauteur et la largeur »
        cat = "blanc" if s.couleur in ("blanc", "capot_alu") else ("ir-reflex" if s.couleur == "ir_reflex" else "standard")
        simple = abaque_aluclip(d, s) if s.aluclip_pro else abaque_simple(d, s.ouvrant)
        batt, batt_ambigu = None, False
        if s.configuration == "2v_battement" and simple is not None and not s.aluclip_pro:
            # un abaque à battement par renfort : si la saisie ne tranche pas entre deux renforts
            # (76272 / 76279 : V326.Z ou V314.Z), on ne devine pas la courbe
            candidats = [b for b in d.abaques_battement if s.ouvrant in b.ouvrants and s.battement in b.battements
                         and b.renfort in simple.renfort.split(" ou ")]
            batt, batt_ambigu = (candidats[0] if len(candidats) == 1 else None), len(candidats) > 1
        if simple is None:
            ctl.append(_controle("ouvrant", "Abaque d'ouvrant", ETUDE,
                                 ("L'abaque AluClip Pro n'a pas pu être lu dans le wiki." if s.aluclip_pro else
                                  f"Aucun abaque d'ouvrant simple pour l'ouvrant {s.ouvrant} dans le manuel."),
                                 [_source(PAGE_ABAQUES, "Abaques d'ouvrant simple")]))
        else:
            lim = simple.couleurs.get(cat)
            src_ab = [_source(PAGE_ABAQUES, simple.section, simple.localisation)] + \
                     [_source(PAGE_ABAQUES, sec, loc_) for sec, loc_ in simple.equivalents]
            anomalies = []
            if s.configuration == "2v_battement":
                anomalies.append("INC-54" if s.aluclip_pro else "INC-53")
            if lim is None:
                ctl.append(_controle("ouvrant", "Abaque d'ouvrant", ETUDE, f"Pas de limite « {cat} » sur l'abaque {simple.section}.", src_ab))
            else:
                # limite de couleur : hauteur maxi, oblique, largeur maxi, règle des 25 %
                if wl <= lim.largeur_a_hauteur_maxi:
                    h_lim = lim.hauteur_maxi
                elif wl <= lim.coin_largeur and lim.coin_largeur > lim.largeur_a_hauteur_maxi:
                    f = (wl - lim.largeur_a_hauteur_maxi) / (lim.coin_largeur - lim.largeur_a_hauteur_maxi)
                    h_lim = lim.hauteur_maxi + f * (lim.coin_hauteur - lim.hauteur_maxi)
                else:
                    h_lim = lim.coin_hauteur
                v_h = _borne(wh, h_lim, PRECISION_COIN_CM)
                v_l = _borne(wl, lim.largeur_maxi, PRECISION_COIN_CM)
                v_25 = _borne(wl, 1.25 * wh, PRECISION_COIN_CM)
                pire = max((v_h, v_l, v_25), key=[OK, ETUDE, HORS].index)
                raisons = []
                if v_h != OK:
                    raisons.append(f"hauteur {_fmt(wh, 1)} cm pour {_fmt(h_lim, 0)} cm admis à cette largeur")
                if v_l != OK:
                    raisons.append(f"largeur {_fmt(wl, 1)} cm pour {_fmt(lim.largeur_maxi)} cm maxi")
                if v_25 != OK:
                    raisons.append(f"la largeur dépasse la hauteur de plus de 25 % (hauteur mini {_fmt(wl / 1.25, 1)} cm)")
                ctl.append(_controle("ouvrant", "Abaque d'ouvrant", pire,
                                     (f"Vantail {_fmt(wl, 1)} × {_fmt(wh, 1)} cm (L × H) dans la limite {cat} "
                                      f"de l'abaque {s.ouvrant} avec renfort {simple.renfort}."
                                      if pire == OK else
                                      ("À moins de la précision de lecture : " if pire == ETUDE else "Hors limite : ") + " ; ".join(raisons) + ".")
                                     + (" Soufflet : le manuel dit les châssis basculants lus sur les mêmes diagrammes en inversant "
                                        "hauteur et largeur ; la règle des 25 % est lue de même (hypothèse)." if s.configuration == "soufflet" else ""),
                                     src_ab + ([_source(PAGE_ABAQUES, batt.section, batt.localisation)] if batt else []),
                                     anomalies, {"renfort": simple.renfort}))
                graphe = {"limite": lim.__dict__, "point": [wl, wh], "courbe": None, "titre": f"{s.ouvrant} · {simple.renfort} · {cat}"}

            # courbe de vent des deux vantaux à battement
            if s.configuration == "2v_battement" and s.aluclip_pro:
                ctl.append(_vent_aluclip(d, s, wl, wh))
            elif s.configuration == "2v_battement":
                renfort_b = d.renfort_battement.get(s.battement or "")
                courbes = batt.courbes.get(f"{s.vent} kN/m²") if batt else None
                if batt is None or renfort_b is None or courbes is None:
                    ctl.append(_controle("vent", f"Charge de vent {s.vent} kN/m²", ETUDE,
                                         (f"Le battement {s.battement} n'a pas de renfort : sa courbe est l'une des vignettes 3 ou 4, "
                                          "dont la source ne nomme pas la combinaison." if renfort_b is None else
                                          f"L'ouvrant {s.ouvrant} a un abaque à battement par renfort ({simple.renfort.replace(' ou ', ' et ')}) "
                                          "aux courbes différentes : la saisie ne dit pas lequel est monté." if batt_ambigu else
                                          "Aucun abaque à battement relevé pour cette combinaison."),
                                         [_source(PAGE_ABAQUES, "Abaques à deux vantaux avec battement")], ["INC-53"]))
                elif renfort_b not in courbes:
                    ctl.append(_controle("vent", f"Charge de vent {s.vent} kN/m²", OK,
                                         f"À {s.vent} kN/m², la courbe du battement {s.battement} (renfort {renfort_b}) n'a pas de tracé "
                                         "dans le cadre de l'abaque : elle ne limite pas.",
                                         [_source(PAGE_ABAQUES, batt.section, batt.localisation)], ["INC-53"]))
                else:
                    h_c, comment = _hauteur_courbe(courbes[renfort_b], wl)
                    if h_c is None:
                        v, det = ETUDE, f"Courbe {renfort_b} : {comment}."
                    elif h_c == float("inf"):
                        v, det = OK, f"Courbe {renfort_b} au-dessus du cadre à cette largeur : elle ne limite pas."
                    else:
                        v = _borne(wh, h_c, PRECISION_COURBE_CM)
                        det = f"Hauteur {_fmt(wh, 1)} cm pour {_fmt(h_c)} cm sur la courbe {renfort_b} ({comment})."
                    ctl.append(_controle("vent", f"Charge de vent {s.vent} kN/m²", v, det,
                                         [_source(PAGE_ABAQUES, batt.section, batt.localisation)], ["INC-53"]))

            # épaisseur de verre
            e = vit.verre_mm
            src_v = [_source(PAGE_ABAQUES, "Épaisseur de verre"), _source(PAGE_ABAQUES, simple.section, simple.localisation)]
            if not simple.verre and e >= 12:
                ctl.append(_controle("verre", "Épaisseur de verre", ETUDE,
                                     f"{_fmt(e, 1 if e % 1 else 0)} mm de verre : à partir de 12 mm les dimensions d'ouvrant sont "
                                     f"restreintes, mais l'abaque {simple.section.split(' avec ')[0].lower()} ne porte aucune courbe d'épaisseur de verre.",
                                     src_v))
            elif e < 12:
                ctl.append(_controle("verre", "Épaisseur de verre", OK,
                                     f"{_fmt(e, 1 if e % 1 else 0)} mm de verre : sous 12 mm, les abaques ne restreignent pas.", src_v))
            else:
                niveau = next((i for i, c in enumerate(COURBES_VERRE) if c >= e), None)
                if niveau is None:
                    ctl.append(_controle("verre", "Épaisseur de verre", ETUDE,
                                         f"{_fmt(e, 1 if e % 1 else 0)} mm de verre : au-delà de la dernière courbe (28 mm).", src_v))
                else:
                    brut = COURBES_VERRE[niveau]
                    if s.j079:
                        niveau -= 2
                    if niveau < 0:
                        ctl.append(_controle("verre", "Épaisseur de verre", OK,
                                             f"{_fmt(e, 1 if e % 1 else 0)} mm de verre, courbe {brut} mm ; l'équerre J079 décale de deux "
                                             "courbes, sous la première : pas de restriction.", src_v + [_source(PAGE_ABAQUES, "Équerre de feuillure J079")]))
                    else:
                        courbe = COURBES_VERRE[niveau]
                        h_c, comment = _hauteur_courbe({g: r.get(courbe) for g, r in simple.verre.items()}, wl)
                        lib_c = f"courbe {courbe} mm" + (f" (verre {brut} mm, décalé de deux courbes par J079)" if s.j079 else
                                                         (f" (verre de {_fmt(e, 1 if e % 1 else 0)} mm, courbe supérieure)" if brut != e else ""))
                        if h_c is None:
                            v, det = ETUDE, f"{lib_c.capitalize()} : {comment}."
                        elif h_c == float("inf"):
                            v, det = OK, f"{lib_c.capitalize()} au-dessus du cadre à cette largeur : elle ne limite pas."
                        else:
                            v = _borne(wh, h_c, PRECISION_COURBE_CM)
                            det = f"Hauteur {_fmt(wh, 1)} cm pour {_fmt(h_c)} cm admis sur la {lib_c}, {comment}."
                        ctl.append(_controle("verre", "Épaisseur de verre", v, det,
                                             src_v + ([_source(PAGE_ABAQUES, "Équerre de feuillure J079")] if s.j079 else [])))
                        if graphe is not None:
                            graphe["courbe"] = {"nom": f"verre {courbe} mm",
                                                "points": [[g, r.get(courbe)] for g, r in sorted(simple.verre.items())]}

            # renforcement exigé
            if s.aluclip_pro:
                ctl.extend(_renfort_aluclip(d, s, e, _poids_verre(d, vit, deb)))
            else:
                zl, zh = simple.zones
                zone = ("A" if wl <= zl else "B") if wh <= zh else ("C" if wl <= zl else "D")
                libelle_zone = {"A": "sans renfort", "B": "renforcement horizontal", "C": "renforcement vertical", "D": "renforcement total"}[zone]
                if e > 12:
                    exige, pourquoi = "renforcement total", f"plus de 12 mm de verre ({_fmt(e, 1 if e % 1 else 0)} mm)"
                elif s.couleur in ("standard", "ir_reflex"):
                    exige, pourquoi = "renforcement systématique", "profilé de couleur"
                elif s.couleur == "capot_alu":
                    exige, pourquoi = f"zone {zone} : {libelle_zone}", "capot aluminium : dimensions du blanc, traitement des couleurs (vissage du renfort)"
                else:
                    exige, pourquoi = f"zone {zone} : {libelle_zone}", f"blanc, vantail {_fmt(wl, 1)} × {_fmt(wh, 1)} cm, zones séparées à {_fmt(zl)} cm et {_fmt(zh)} cm"
                ctl.append(_controle("renfort", "Renforcement exigé", INFO,
                                     f"{exige[0].upper() + exige[1:]} ({pourquoi}), renfort {simple.renfort}.",
                                     [_source(PAGE_ABAQUES, "Zones de renforcement"), _source(PAGE_ABAQUES, "Catégories de couleur")],
                                     valeurs={"zone": zone, "renfort": simple.renfort}))
                if graphe is not None:
                    graphe["zones"] = [zl, zh]

        # le DTA borne la baie : ramené au vantail par les mêmes cotes de débit, pour l'abaque
        if graphe is not None:
            graphe["dta"] = []
            for c, h, l in lignes:
                deo = calcul_debit(d, replace(s, largeur_mm=l * 1000, hauteur_mm=h * 1000))["deo"]
                graphe["dta"].append({"configuration": c, "largeur": deo[1 if s.configuration == "soufflet" else 0] / 10,
                                      "hauteur": deo[0 if s.configuration == "soufflet" else 1] / 10})

    # 7. la parclose
    src_p = [_source(PAGE_PARCLOSES, "Comment choisir une parclose")]
    t = vit.total_mm
    if t > 50:
        ctl.append(_controle("parclose", "Vitrage et parclose", HORS,
                             f"Vitrage de {_fmt(t, 2 if t % 1 else 0)} mm : le DTA admet jusqu'à 50 mm.",
                             src_dta + src_p, ["CTR-17"]))
    else:
        serie = d.parcloses_dormant if s.configuration == "fixe" else d.parcloses_ouvrant
        ou = "de dormant" if s.configuration == "fixe" else "d'ouvrant"
        # La page PERFORM76 donne une épaisseur par parclose, sans tolérance : correspondance à
        # ±0,5 mm, et « sur étude » avec les candidates jusqu'à ±1,5 mm.
        trouvees = [p for p in serie if abs(p["epaisseur"] - t) <= 0.5]
        voisines = [p for p in serie if abs(p["epaisseur"] - t) <= 1.5]
        anomalies = ["CTR-19"] if any(p["ref"] in ("76508", "2454", "2433", "2638") for p in trouvees or voisines) else []
        if not trouvees:
            proches = sorted({p["epaisseur"] for p in serie}, key=lambda x: abs(x - t))[:2]
            ctl.append(_controle("parclose", "Vitrage et parclose", ETUDE if voisines else HORS,
                                 f"Vitrage de {_fmt(t, 2 if t % 1 else 0)} mm : aucune parclose {ou} PERFORM76 à cette épaisseur "
                                 f"(les plus proches : {', '.join(_fmt(x) for x in sorted(proches))} mm)"
                                 + (" ; la page ne donne pas de tolérance." if voisines else "."),
                                 [_source(PAGE_PARCLOSES, f"Cotes des parcloses {ou}")], anomalies,
                                 {"parcloses": voisines}))
        else:
            refs = ", ".join(p["ref"] for p in trouvees)
            ctl.append(_controle("parclose", "Vitrage et parclose", OK,
                                 f"Vitrage de {_fmt(t, 2 if t % 1 else 0)} mm ({_fmt(vit.verre_mm, 1 if vit.verre_mm % 1 else 0)} mm de verre) : "
                                 f"parclose {ou} {refs}" + (" — le critère entre les deux n'est pas donné." if len(trouvees) > 1 else "."),
                                 [_source(PAGE_PARCLOSES, f"Cotes des parcloses {ou}")], anomalies,
                                 {"parcloses": trouvees}))

    # 8. paumelles et poids, pour information
    if deb["vantaux"] and not deb.get("manque") and s.configuration not in ("1v_ob", "soufflet"):
        wh = deb["deo"][1] / 10
        n = next((k for a, b, k in d.paumelles if wh <= b), d.paumelles[-1][2] if wh > d.paumelles[-1][1] else None)
        if n is not None:
            ctl.append(_controle("paumelles", "Paumelles", INFO, f"{n} paumelles pour {_fmt(wh, 1)} cm de hauteur d'ouvrant.",
                                 [_source(PAGE_ABAQUES, "Nombre de paumelles")]))
    # 9. le poids du verre, la ferrure Roto NX et le pivot bas
    avec_vantaux = deb["vantaux"] and not deb.get("manque")
    poids = _poids_verre(d, vit, deb) if avec_vantaux else None
    if poids is not None:
        vl, vh = deb["vitrage"]
        ctl.append(_controle("poids", "Poids du vitrage", INFO,
                             f"{_fmt(poids, 1)} kg de verre par vantail ({_fmt(vl, 0)} × {_fmt(vh, 0)} mm, "
                             f"{_fmt(d.poids_verre_kg_m2_mm, 1)} kg par mm et par m²). Le wiki ne donne pas le poids "
                             "des profilés : les contrôles de poids ci-dessous portent sur le verre seul.",
                             [_source(PAGE_ROTO, "Convertir une épaisseur de vitrage en poids de vantail")],
                             valeurs={"poids_kg": round(poids, 1)}))
    if avec_vantaux:
        ctl.append(_roto(d, s, deb, poids, vit.verre_mm))
    if avec_vantaux and d.pivot_kg and poids is not None and s.configuration != "soufflet":
        v = HORS if poids > d.pivot_kg else OK
        ctl.append(_controle("pivot", "Pivot bas", v,
                             (f"{_fmt(poids, 1)} kg de verre pour {_fmt(d.pivot_kg)} kg admis par ouvrant"
                              + (" : le verre seul dépasse la charge du pivot." if v == HORS else
                                 " ; profilés et renfort non comptés.")),
                             [_source(PAGE_PIVOT, "Charge admissible sur le pivot bas")], ["CTR-01"],
                             {"charge_kg": d.pivot_kg}))

    if avec_vantaux and s.ferrure_poids:
        ctl.append(_poids_ferrure(d, s, poids))
    if avec_vantaux and s.distance_verrouillage_cm:
        ctl.append(_verrouillage(d, s))

    # 10. l'isolant et la tapée
    if s.isolant_mm:
        ctl.append(_isolant(d, s))

    rang = [OK, INFO, ETUDE, HORS]
    decisifs = [c for c in ctl if c["statut"] in (HORS, ETUDE)]
    verdict = HORS if any(c["statut"] == HORS for c in ctl) else (ETUDE if decisifs else OK)
    brouillons = sorted(p for p, st in d.statuts.items() if st == "draft")
    return wiki_famille.relocaliser(snap, {
        "verdict": verdict,
        "libelle": {OK: "Réalisable", ETUDE: "Sur étude", HORS: "Hors domaine d'emploi"}[verdict],
        "contrainte": next((c for c in sorted(decisifs, key=lambda c: -rang.index(c["statut"]))), None),
        "controles": ctl,
        "debit": {k: deb.get(k) for k in ("deo", "dfo", "vitrage", "etapes", "vantaux")},
        "vitrage": vit.__dict__,
        "graphe": graphe,
        "brouillons": brouillons,
    })


def options(snap: WikiSnapshot) -> Dict[str, Any]:
    d = donnees(snap)
    return {
        "configurations": CONFIGURATIONS,
        "couleurs": COULEURS,
        "dormants": d.dormants_proferm,
        "ouvrants": d.ouvrants_proferm,
        "meneaux": sorted(d.debit_meneaux),
        "battements": sorted({b for _, b in d.debit_battements}),
        "vents": sorted({v.replace(" kN/m²", "") for b in d.abaques_battement for v in b.courbes}),
        "brouillons": sorted(p for p, st in d.statuts.items() if st == "draft"),
        "pages": list(PAGES),
        "paumelles": PAUMELLES,
        "versions": {"P": {k: v for k, v in VERSIONS_P.items() if any(kk[0] == k for kk in d.roto_p)},
                     "designo": {k: v for k, v in VERSIONS_DESIGNO.items() if k in d.roto_designo}},
        "securites": {k: v for k, v in SECURITES.items() if any(kk[1] == k for kk in d.roto_p)},
        "aluclip_pro": {"ouvrant": "76271", "couleur": "capot_alu", "battements": sorted(d.aluclip_pro.get("battements", {}))},
        "ferrures_poids": sorted({f for f, _ in d.poids_ferrure}),
        "aciers_dormant": {a: next(v["acier"] for (f, aa), v in d.poids_ferrure.items() if aa == a)
                           for a in ("sans", "court", "long") if any(aa == a for _, aa in d.poids_ferrure)},
        "isolants": {dor: [e for e in (t["iso"].get(dor) for t in d.tapees) if e is not None] for dor in d.dormants_proferm},
    }
