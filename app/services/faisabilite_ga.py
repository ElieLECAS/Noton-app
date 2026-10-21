"""Le calculateur du LUMEAL GA (coulissant à ouvrant caché, LUMÉAL55 de la gamme LUMINE), 21/10/2026.

Deux sorties, comme pour la PERFORM76 : la **faisabilité** (``verifier``) et la **fiche de débit**
(``fiche_debit``). Toutes les valeurs sont lues dans les tableaux du wiki à chaque rechargement de
l'instantané ; seuls des chemins de pages, des titres de section et des en-têtes de colonnes sont écrits ici.

Ce que le wiki permet, et ce que ce module en fait :

- **le DTA 6/14-2166** borne la baie d'une *fenêtre* à 2 vantaux (ou 1 vantail + fixe) selon la paire de
  montants centraux et le recouvrement, mini ou maxi. Le recouvrement n'est défini nulle part (VER-169) : quand
  l'utilisateur ne le dit pas, une baie qui tient dans la limite « maxi » (la plus basse) passe, une baie qui
  dépasse la limite « mini » est hors domaine, entre les deux le verdict est « sur étude » ;
- **les courbes d'utilisation au vent** (V*A2 : 800 Pa, V*A3 : 1 200 Pa) bornent la hauteur selon la largeur
  et la paire de montants. Jamais d'interpolation : entre deux graduations de L, la plus restrictive ; à moins
  de la précision de lecture écrite par la page, « sur étude » ;
- **les limites imprimées sous les courbes** (300 kg par ouvrant, demi-périmètre de 5 m) : le wiki ne donne ni
  le poids des profilés ni la largeur du cadre ouvrant, donc le poids de verre seul et une borne basse du
  demi-périmètre ne peuvent que **dépasser** la limite (hors domaine) ; sous la limite, l'information est
  donnée, jamais un « oui » ;
- **le vitrage** : épaisseur du double vitrage, joint de la prise de volume, plage de la traverse basse ;
- **les roulettes** (unité du poids maxi non imprimée, VER-239) et **la cale** TGA3817 (condition ambiguë,
  CTR-198) : information, « sur étude » si même la plus forte roulette ne suffit pas ;
- **la liste de coupe** de chaque application dont le wiki a la nomenclature (huit sur dix) : les formules de
  débit sont du texte (« L/2-65 », « L-Lf-241 ») que ``evaluer`` lit avec une grammaire explicite, jamais ``eval``.

Ce module porte aussi ce que ``faisabilite_gy`` réutilise : l'évaluateur de formules, la lecture d'une courbe
d'utilisation (``lire_sur_courbe``), la lecture d'une nomenclature (``lire_nomenclature``), la table des
roulettes et les images par référence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.services import wiki_famille
from app.services.faisabilite import (DonneeIntrouvable, ETUDE, HORS, INFO, OK, Table, _borne, _controle, _corps,
                                      _fmt, _nombre, _source, _texte_section, _une, tables)
from app.services.faisabilite_systemes import coercer, synthese
from app.services.wiki_service import WikiSnapshot
from app.services.wiki_tables import tables_toutes

ID = "lumeal-ga"
LIBELLE = "LUMEAL GA (coulissant LUMINE)"
GAMMES = ["LUMINE"]
PORTEE = ("Fenêtre 2 vantaux : DTA 6/14-2166 (dimensions maximales de baie), courbes d'utilisation au vent A2 et "
          "A3, limites de charge et de demi-périmètre, vitrage, roulettes et liste de coupe. Portes-fenêtres : "
          "courbes et liste de coupe, le DTA est « sur étude » (son tableau est celui d'une fenêtre). 6 vantaux et "
          "2 vantaux + 2 fixes latéraux : aucune nomenclature dans le wiki, « sur étude ».")

PAGE_DEBIT = "/profiles/lumeal-ga-cotes-de-debit.md"
PAGE_DEBIT_RC3 = "/profiles/lumeal-ga-cotes-de-debit-2-vantaux-rc3-et-seuil-pmr.md"
PAGE_DEBIT_4V = "/profiles/lumeal-ga-cotes-de-debit-4-vantaux-et-3-rails.md"
PAGE_DEBIT_1V = "/profiles/lumeal-ga-cotes-de-debit-1-vantail-et-fixe-central.md"
PAGE_COURBES = "/profiles/lumeal-ga-courbes-d-utilisation-au-vent.md"
PAGE_DTA = "/certifications/dta-6-14-2166-lumeal-ga-dossier-technique.md"
PAGE_DTA_AVIS = "/certifications/dta-6-14-2166-lumeal-ga.md"
PAGE_OUVRANTS = "/profiles/lumeal-ga-dormants-et-ouvrants.md"
PAGE_VOLUME = "/profiles/lumeal-ga-prises-de-volume-et-joints-de-vitrage.md"
PAGE_ROULETTES = "/quincaillerie/lumeal-ga-roulements-et-fermetures.md"
PAGE_VERRE = "/profiles/systeme-70-abaques-dimensionnels.md"
PAGES = (PAGE_DTA, PAGE_DTA_AVIS, PAGE_COURBES, PAGE_DEBIT, PAGE_DEBIT_RC3, PAGE_DEBIT_4V, PAGE_DEBIT_1V,
         PAGE_OUVRANTS, PAGE_VOLUME, PAGE_ROULETTES, PAGE_VERRE)

# Une application = une section d'une page de nomenclature : (libellé, page, titre de la section).
APPLICATIONS: Dict[str, Tuple[str, str, str]] = {
    "fenetre2": ("Fenêtre 2 vantaux", PAGE_DEBIT, "Fenêtre 2 vantaux"),
    "pf2": ("Porte-fenêtre 2 vantaux", PAGE_DEBIT, "Porte-fenêtre 2 vantaux"),
    "pf2_rc3": ("Porte-fenêtre 2 vantaux retard à l'effraction (RC3)", PAGE_DEBIT_RC3,
                "Porte-fenêtre 2 vantaux retard à l'effraction"),
    "pf2_pmr": ("Porte-fenêtre 2 vantaux seuil PMR", PAGE_DEBIT_RC3, "Porte-fenêtre 2 vantaux seuil PMR"),
    "pf1_fixe": ("Porte-fenêtre 1 vantail + fixe", PAGE_DEBIT_1V, "Porte-fenêtre 1 vantail + fixe"),
    "pf2_fixe_central": ("Porte-fenêtre 2 vantaux + fixe central", PAGE_DEBIT_1V,
                         "Porte-fenêtre 2 vantaux + fixe central"),
    "pf4": ("Porte-fenêtre 4 vantaux 2 rails (percussion centrale)", PAGE_DEBIT_4V, "Porte-fenêtre 4 vantaux 2 rails"),
    "pf3": ("Porte-fenêtre 3 vantaux 3 rails", PAGE_DEBIT_4V, "Porte-fenêtre 3 vantaux 3 rails"),
}
AUTRE = "autre"
LIBELLE_AUTRE = "Autre application (6 vantaux 3 rails, 2 vantaux + 2 fixes latéraux)"
# Le tableau du DTA est celui d'une fenêtre ; les courbes sont tracées « pour 2 vantaux et 1 vantail + fixe ».
APPLICATIONS_DTA = ("fenetre2",)
APPLICATIONS_COURBES = ("fenetre2", "pf2", "pf2_rc3", "pf2_pmr", "pf1_fixe")
AVEC_LF = ("pf1_fixe",)

HYPOTHESE_BAIE = "hypothèse : la baie est prise égale aux dimensions hors tout saisies"
HYPOTHESE_UNITE = "hypothèse : le poids maxi du tableau est en kg par vantail (unité non imprimée, VER-239)"


# ---- l'évaluateur de formules de débit --------------------------------------------------------


class FormuleIllisible(ValueError):
    """Une formule que la grammaire ne comprend pas : la ligne n'est pas calculée."""


@dataclass(frozen=True)
class Evaluation:
    valeur: float
    suffixe: str = ""          # la lettre de coupe collée à la formule, « (AB) » : sans légende, jamais interprétée


_JETON = re.compile(r"(?P<nb>\d+(?:[.,]\d+)?)|(?P<id>[A-Za-z][A-Za-z0-9]*)|(?P<op>[-+*/()×÷])")


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
        out.append((m.lastgroup or "", m.group(0)))
        pos = m.end()
    return out


class _Analyseur:
    """nombres, variables nommées, + − × ÷ et parenthèses ; « 2L » se lit comme un produit."""

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
        while (j := self._suivant()) and j[1] in "+-" and j[0] == "op":
            self.i += 1
            droite = self._terme()
            valeur = valeur + droite if j[1] == "+" else valeur - droite
        return valeur

    def _terme(self) -> float:
        valeur, etait_nombre = self._facteur()
        while (j := self._suivant()) is not None:
            if j[0] == "op" and j[1] in "*×":
                self.i += 1
                valeur *= self._facteur()[0]
            elif j[0] == "op" and j[1] in "/÷":
                self.i += 1
                diviseur = self._facteur()[0]
                if diviseur == 0:
                    raise FormuleIllisible("division par zéro")
                valeur /= diviseur
            elif etait_nombre and j[0] == "id":
                valeur *= self._facteur()[0]
            else:
                break
            etait_nombre = False
        return valeur

    def _facteur(self) -> Tuple[float, bool]:
        j = self._suivant()
        if j is None:
            raise FormuleIllisible("formule incomplète")
        self.i += 1
        genre, texte = j
        if genre == "nb":
            return float(texte.replace(",", ".")), True
        if genre == "id":
            if texte in self.v:
                return float(self.v[texte]), False
            raise FormuleIllisible(f"variable « {texte} » inconnue")
        if texte == "-":
            return -self._facteur()[0], False
        if texte == "(":
            valeur = self._expression()
            fin = self._suivant()
            if fin is None or fin[1] != ")":
                raise FormuleIllisible("parenthèse non fermée")
            self.i += 1
            return valeur, False
        raise FormuleIllisible(f"« {texte} » n'est pas attendu ici")


def evaluer(formule: str, variables: Dict[str, float]) -> Evaluation:
    """Le résultat d'une formule de débit ou de quantité ; ``FormuleIllisible`` si la grammaire ne la comprend pas.

    Une lettre de coupe collée au bout (« L/2-55(AB) ») est séparée du calcul et rendue telle quelle : aucune
    source ne la définit (VER-459), elle ne sert pas à calculer."""
    texte = formule.strip().replace("−", "-")
    suffixe = ""
    m = re.fullmatch(r"(.*[\d)])\(([A-Za-z][A-Za-z0-9'\"]*)\)", texte)
    if m:      # une lettre entre parenthèses collée à un nombre n'est jamais un produit : « 3L/4-179.5(H) » du GY
        texte, suffixe = m.group(1), m.group(2)
    jetons = _jetons(texte)
    # « x » entre deux nombres ou parenthèses est le signe de multiplication, sauf si c'est une variable
    jetons = [("op", "*") if g == "id" and t == "x" and "x" not in variables else (g, t) for g, t in jetons]
    if not jetons:
        raise FormuleIllisible("formule vide")
    if len(jetons) > 1 and all(g == "id" for g, _ in jetons):
        raise FormuleIllisible("du texte, pas une formule")
    return Evaluation(round(_Analyseur(jetons, variables).analyser(), 4), suffixe)


def valeur_ou_none(formule: str, variables: Dict[str, float]) -> Tuple[Optional[float], str]:
    """(valeur, raison) : la raison dit pourquoi une formule n'est pas calculée."""
    try:
        return evaluer(formule, variables).valeur, ""
    except FormuleIllisible as exc:
        return None, f"« {formule} » : {exc}"


# ---- lire une nomenclature -------------------------------------------------------------------


@dataclass
class Ligne:
    ref: str
    designation: str
    qte: str                   # texte du wiki : « 2 », « (L-Lf)/300 »
    debit: str                 # texte du wiki ; vide pour un accessoire ou une vis
    coupe: str = ""            # le pictogramme imprimé (GY), jamais interprété


@dataclass
class Application:
    id: str
    libelle: str
    page: str
    section: str
    localisation: str
    profils: List[Ligne]
    accessoires: List[Ligne]
    joints: List[Ligne]
    vis: List[Ligne]
    remplissages: List[Dict[str, str]]      # {"libelle", "hauteur", "largeur"} : formules du wiki
    anomalies: List[str]                    # celles que la page cite pour cette application


def classer_table(t: Table) -> Optional[str]:
    """« profils », « joints », « accessoires » ou « vis » selon la forme du tableau ; ``None`` sinon."""
    h = t.entete
    if not h or h[0] != "Référence":
        return None
    if len(h) == 2:
        return "vis"
    debit = any(c.startswith("Débit") for c in h)
    qte = any(c.startswith(("Qté", "Quantité")) for c in h)
    if debit and qte:
        return "profils"
    return "joints" if debit else ("accessoires" if qte else None)


def lire_lignes(t: Table, genre: str) -> List[Ligne]:
    """Les lignes d'un tableau de nomenclature. Une cellule de coupe « || » non échappée se scinde en cellules
    vides (VER-458) : la quantité est donc toujours la troisième cellule et le débit toujours la dernière."""
    out: List[Ligne] = []
    for r, b in zip(t.lignes, t.brut):
        if not r or not r[0]:
            continue
        if genre == "vis":
            out.append(Ligne(r[0], "", r[-1], ""))
        elif genre == "accessoires":
            out.append(Ligne(r[0], r[1], r[2], ""))
        elif genre == "joints":
            out.append(Ligne(r[0], r[1], "1", r[-1]))
        else:
            milieu = [c for c in b[3:-1]]
            coupe = "".join(milieu).strip() or ("||" if len(milieu) > 1 else "")
            out.append(Ligne(r[0], r[1], r[2], r[-1], coupe))
    return out


REMPLISSAGE_RE = re.compile(
    r"(?:(le vantail|la partie fixe)[, ]+)?(H-\d+)(?:\s+en hauteur)?\s+et\s+(?:en largeur\s+à\s+)?"
    r"((?:Lf|L)[\w/\-.]*\d)")


def lire_remplissages(texte: str) -> List[Dict[str, str]]:
    """Les cotes du remplissage, écrites en phrase (« H-143 et L/2-66 »), avec le libellé quand la phrase en donne."""
    return [{"libelle": m.group(1) or "", "hauteur": m.group(2), "largeur": m.group(3)}
            for m in REMPLISSAGE_RE.finditer(texte)]


def anomalies_citees(texte: str) -> List[str]:
    return sorted(set(re.findall(r"\b(?:INC|CTR|VER)-\d+\b", texte)), key=lambda a: (a[:3], int(a[4:])))


def lire_nomenclature(corps: str, section: Optional[str], page: str, remplissage_dans: Optional[str] = None,
                      id_: str = "", libelle: str = "") -> Application:
    """Les quatre tableaux d'une nomenclature et la phrase de remplissage.

    ``section`` : le titre de la section qui porte les quatre tableaux (une page de LUMEAL GA en porte plusieurs,
    une par application) ; ``None`` quand la page entière est une application (SOLEAL GY : une section par
    tableau). ``remplissage_dans`` : le titre de la section qui porte la phrase de remplissage (la même par défaut)."""
    ts = [t for t in tables_toutes(corps) if section is None or (t.chemin and t.chemin[-1] == section)]
    trouves: Dict[str, Table] = {}
    for t in ts:
        genre = classer_table(t)
        if genre and genre not in trouves:
            trouves[genre] = t
    nom = section or "page entière"
    for genre in ("profils", "accessoires", "joints", "vis"):
        if genre not in trouves:
            raise DonneeIntrouvable(f"{page} : section « {nom} », tableau des {genre} introuvable")
    titre_phrase = remplissage_dans or section
    texte = _texte_section(corps, titre_phrase) if titre_phrase else corps
    remplissages = lire_remplissages(texte)
    if not remplissages:
        raise DonneeIntrouvable(f"{page} : section « {nom} », cotes du remplissage introuvables")
    if section is None:
        texte = corps
    return Application(id_, libelle, page, nom, trouves["profils"].localisation,
                       lire_lignes(trouves["profils"], "profils"), lire_lignes(trouves["accessoires"], "accessoires"),
                       lire_lignes(trouves["joints"], "joints"), lire_lignes(trouves["vis"], "vis"),
                       remplissages, anomalies_citees(texte))


def normaliser_ref(ref: str) -> str:
    """« T141037 » et « 141037 » nomment le même profilé (INC-444) ; « TGY1207 » reste tel quel."""
    ref = ref.strip()
    return ref[1:] if re.fullmatch(r"T\d+", ref) else ref


REF_PAIRE_RE = re.compile(r"\b(?:(?:TGY|TGA|T)\d{4,6}|\d{6})\b")


def paire_de(entete: str) -> Optional[Tuple[str, str]]:
    """La paire de montants centraux nommée par un en-tête de colonne ; ``None`` si l'en-tête n'en nomme pas deux."""
    refs = [normaliser_ref(r) for r in REF_PAIRE_RE.findall(entete)]
    return tuple(sorted(refs)) if len(refs) == 2 else None  # type: ignore[return-value]


def lire_colonnes(t: Table) -> List[Tuple[str, Dict[int, Optional[float]]]]:
    """Un tableau de courbes : (en-tête, {largeur: hauteur}) par colonne ; la première colonne est L, « - » reste ``None``."""
    largeurs = [int(_nombre(r[0])) for r in t.lignes]
    return [(t.entete[k], {l: _nombre(r[k]) for l, r in zip(largeurs, t.lignes)}) for k in range(1, len(t.entete))]


# ---- lire une courbe, sans interpoler --------------------------------------------------------


@dataclass
class LectureCourbe:
    statut: str
    h_max: Optional[float]     # ``None`` : non lisible ; ``inf`` : au-dessus du cadre de l'abaque
    detail: str


def lire_sur_courbe(courbe: Dict[int, Optional[float]], largeur: float, hauteur: float, cadre: Optional[float],
                    precision: float) -> LectureCourbe:
    """Verdict d'une hauteur sur une courbe relevée aux graduations de L, **sans interpolation**.

    Entre deux graduations, la plus restrictive. Un « - » avant la première valeur d'une courbe est une courbe
    au-dessus du cadre : elle ne limite pas, dans le cadre ; après la dernière valeur, c'est une courbe arrêtée
    ou non relevée : inconnue. À moins de la précision de lecture de la limite, « sur étude »."""
    grad = sorted(courbe)
    if largeur < grad[0] or largeur > grad[-1]:
        return LectureCourbe(ETUDE, None, f"largeur de {_fmt(largeur)} mm hors des graduations de l'abaque "
                                          f"({_fmt(grad[0])} à {_fmt(grad[-1])} mm)")
    bas = max(g for g in grad if g <= largeur)
    haut = min(g for g in grad if g >= largeur)
    premiere = min((g for g in grad if courbe[g] is not None), default=None)
    lus, au_dessus = [], False
    for g in sorted({bas, haut}):
        v = courbe[g]
        if v is None:
            if premiere is not None and g < premiere:
                au_dessus = True
                continue
            return LectureCourbe(ETUDE, None, f"la courbe n'est pas relevée à L = {_fmt(g)} mm (courbe arrêtée)")
        lus.append(v)
    ou = (f"à L = {_fmt(bas)} mm" if bas == haut else f"entre L = {_fmt(bas)} et {_fmt(haut)} mm, la plus restrictive")
    if not lus:
        if cadre is None:
            return LectureCourbe(ETUDE, float("inf"), f"courbe au-dessus du cadre de l'abaque {ou}, cadre non lu")
        statut = OK if hauteur <= cadre - precision else ETUDE
        return LectureCourbe(statut, float("inf"),
                             f"courbe au-dessus du cadre de l'abaque {ou} : elle ne limite pas jusqu'à "
                             f"{_fmt(cadre)} mm" + ("" if statut == OK else f", la hauteur de {_fmt(hauteur)} mm est "
                                                  "à moins de la précision de lecture du cadre ou au-delà"))
    h_lim = min(lus)
    if au_dessus:
        ou += " (l'autre graduation est au-dessus du cadre)"
    statut = _borne(hauteur, h_lim, precision)
    return LectureCourbe(statut, h_lim, f"H maxi lue {_fmt(h_lim)} mm {ou} (précision de lecture ±{_fmt(precision)} mm)")


# ---- les autres lectures communes -------------------------------------------------------------


def lire_roulettes(t: Table) -> Dict[str, Dict[str, Any]]:
    """{référence: {"poids", "reglage", "designation"}} ; le poids maxi par vantail est imprimé sans unité."""
    k = next((i for i, c in enumerate(t.entete) if c.startswith("Poids maxi par vantail")), None)
    if k is None:
        raise DonneeIntrouvable("tableau des roulettes : colonne « Poids maxi par vantail » introuvable")
    return {r[0]: {"poids": _nombre(r[k]), "designation": r[1], "reglage": r[2] if k > 2 else ""}
            for r in t.lignes if r and _nombre(r[k]) is not None}


def lire_plage(texte: str) -> Optional[Tuple[float, float]]:
    """« 24 à 28 », « 24 à 28 mm », « 32 » → (min, max) ; ``None`` si aucune épaisseur n'est écrite."""
    n = [float(x) for x in re.findall(r"\d+", texte)]
    if not n:
        return None
    return (n[0], n[1]) if len(n) >= 2 else (n[0], n[0])


def images_par_reference(snap: WikiSnapshot, fragment: str) -> Dict[str, str]:
    """La coupe de chaque référence : l'image dont le texte alternatif est la référence, ou « Profilé RÉF »,
    « Joint RÉF », « Accessoire RÉF », « Vis RÉF », « Roulette RÉF » (pages dont le chemin porte ``fragment``)."""
    out: Dict[str, str] = {}
    motif = re.compile(r"^(?:(?:Profilé complémentaire|Profilé PVC|Profilé|Accessoire|Joint|Vis|Roulette|Dormant|"
                       r"Ouvrant|Montant central|Montant latéral|Montant|Parclose|Seuil|Bouclier)\s+)?"
                       r"((?:TGY|TGA|TVE|TAS|T)\d{3,6})$")
    for chemin in sorted(snap.pages):
        if fragment not in chemin or snap.pages[chemin].missing:
            continue
        for alt, src in re.findall(r"!\[([^\]]+)\]\((/assets/[^)]+)\)", snap.pages[chemin].body):
            m = motif.match(alt.strip())
            if m:
                out.setdefault(m.group(1), src)
    return out


def densite_du_verre(snap: WikiSnapshot) -> Optional[float]:
    """kg/m² par mm de verre, lu sur la page des abaques du système 70 : aucune page LUMEAL ne le donne (hypothèse)."""
    if PAGE_VERRE not in snap.pages:
        return None
    m = re.search(r"masse de \*\*([\d,]+) kg/m² par mm", snap.pages[PAGE_VERRE].body)
    return _nombre(m.group(1)) if m else None


def limites_sous_les_courbes(corps: str, page: str) -> Dict[str, float]:
    """Charge maxi par ouvrant (kg), demi-périmètre maxi (mm), précision de lecture (mm), cadre des abaques (mm)."""
    lim: Dict[str, float] = {}
    for cle, motif, facteur in (
            ("kg", r"limitée à (\d+)\s*kg", 1), ("demi_perimetre", r"limité à (\d+)\s*ml", 1000),
            ("precision", r"[Pp]récision de lecture (?:est de )?±\s*(\d+)\s*mm", 1),
            ("cadre", r"(?:cadre des abaques s.arrête à H = |axe vertical H \(mm\) de [\d\s  ]+ à )"
                      r"([\d\s  ]+?)\s*mm", 1)):
        m = re.search(motif, corps)
        if not m:
            raise DonneeIntrouvable(f"{page} : « {cle} » introuvable dans le texte des courbes")
        lim[cle] = float(re.sub(r"\D", "", m.group(1))) * facteur
    return lim


def cale_ou_vide(t: Table, cle: str) -> List[List[str]]:
    return [r for r in t.lignes if r and r[0].strip() == cle]


def evaluer_application(app: Application, variables: Dict[str, float]) -> Dict[str, Any]:
    """Les cotes de la fiche : chaque ligne calculée ou la raison pour laquelle elle ne l'est pas."""
    lignes: List[Dict[str, Any]] = []
    for genre in ("profils", "joints", "accessoires", "vis"):
        for l in getattr(app, genre):
            q, raison_q = valeur_ou_none(l.qte, variables)
            if genre == "joints":
                q, raison_q = 1.0, ""
            d, raison_d = (valeur_ou_none(l.debit, variables) if l.debit else (None, ""))
            lignes.append({"genre": genre, "ligne": l, "qte": q, "longueur": d,
                           "raison": "; ".join(r for r in (raison_q, raison_d) if r),
                           "suffixe": (evaluer(l.debit, variables).suffixe if d is not None and l.debit else "")})
    return {"lignes": lignes}


# ---- les données lues dans le wiki -------------------------------------------------------------


@dataclass
class LigneDta:
    paire: Tuple[str, str]
    libelle: str               # « 2 x 141037 », « 141037 + 141038 » tel que le DTA l'écrit
    type_: str
    h_mm: float
    l_mm: float


@dataclass
class Donnees:
    statuts: Dict[str, str]
    dta: Dict[str, List[LigneDta]]                      # « mini » / « maxi »
    dta_titres: Dict[str, str]
    dta_localisation: Dict[str, str]
    dta_validite: str
    dta_vitrages: List[int]
    courbes: Dict[str, Dict[str, Any]]                  # « A2 » → section, localisation, pression, colonnes
    limites: Dict[str, float]
    applications: Dict[str, Application]
    plages_traverses: Dict[str, Optional[Tuple[float, float]]]
    joints_volume: Dict[int, Dict[str, str]]            # épaisseur → référence, repère couleur
    loc_volume: str
    roulettes: Dict[str, Dict[str, Any]]
    loc_roulettes: str
    cales: List[Table]
    densite_verre: Optional[float]
    images: Dict[str, str]


_CACHE: Dict[str, Any] = {"cle": None, "donnees": None}


def donnees(snap: WikiSnapshot) -> Donnees:
    cle = (id(snap), snap.loaded_at)
    if _CACHE["cle"] != cle:
        _CACHE["donnees"] = _charge(snap)
        _CACHE["cle"] = cle
    return _CACHE["donnees"]


def _charge(snap: WikiSnapshot) -> Donnees:
    # -- le DTA : une table par recouvrement
    corps = _corps(snap, PAGE_DTA)
    dta: Dict[str, List[LigneDta]] = {}
    titres: Dict[str, str] = {}
    loc_dta: Dict[str, str] = {}
    for t in tables(corps):
        m = re.fullmatch(r"Pose en applique avec un recouvrement (mini|maxi)", t.section)
        if not m:
            continue
        if t.entete != ["Fenêtre", "Montants centraux", "H (m)", "L (m)"]:
            raise DonneeIntrouvable(f"{PAGE_DTA} : en-têtes du tableau « {t.section} » changés : {t.entete}")
        lignes = []
        for r in t.lignes:
            refs = [normaliser_ref(x) for x in re.findall(r"\d{6}", r[1])]
            paire = (refs[0], refs[0]) if len(refs) == 1 and re.match(r"\s*2\s*x", r[1]) else tuple(sorted(refs))
            if len(paire) != 2:
                raise DonneeIntrouvable(f"{PAGE_DTA} : montants centraux illisibles : « {r[1]} »")
            lignes.append(LigneDta(paire, r[1], r[0], round(_nombre(r[2]) * 1000), round(_nombre(r[3]) * 1000)))
        dta[m.group(1)] = lignes
        titres[m.group(1)] = t.section
        loc_dta[m.group(1)] = t.localisation
    if set(dta) != {"mini", "maxi"}:
        raise DonneeIntrouvable(f"{PAGE_DTA} : tableaux « recouvrement mini » et « recouvrement maxi » introuvables")
    mv = re.search(r"Isolant double de ([\d, ou]+?) mm d'épaisseur", corps)
    vitrages = [int(x) for x in re.findall(r"\d+", mv.group(1))] if mv else []
    mval = re.search(r"valide du (\d{2}/\d{2}/\d{4}) au (\d{2}/\d{2}/\d{4})", snap.pages[PAGE_DTA_AVIS].raw_text
                     if PAGE_DTA_AVIS in snap.pages else "")
    validite = f"{mval.group(1)} au {mval.group(2)}" if mval else ""

    # -- les courbes d'utilisation
    corps = _corps(snap, PAGE_COURBES)
    limites = limites_sous_les_courbes(corps, PAGE_COURBES)
    courbes: Dict[str, Dict[str, Any]] = {}
    for t in tables_toutes(corps):
        m = re.match(r"Abaque V\*(A\d) : vent (\d+) Pa", t.section)
        if not (m and t.entete and t.entete[0].startswith("L hors tout")):
            continue
        colonnes: Dict[Tuple[str, str], Dict[int, Optional[float]]] = {}
        entetes: Dict[Tuple[str, str], str] = {}
        for entete, courbe in lire_colonnes(t):
            paire = paire_de(entete)
            if paire:
                colonnes[paire] = courbe
                entetes[paire] = entete.split(" / ")[-1]
        courbes[m.group(1)] = {"section": t.section, "localisation": t.localisation, "pression": int(m.group(2)),
                               "colonnes": colonnes, "entetes": entetes}
    if not courbes:
        raise DonneeIntrouvable(f"{PAGE_COURBES} : abaques V*A2 / V*A3 introuvables")

    # -- les nomenclatures
    corps_par_page = {p: _corps(snap, p) for p in {a[1] for a in APPLICATIONS.values()}}
    apps = {k: lire_nomenclature(corps_par_page[page], section, page, id_=k, libelle=lib)
            for k, (lib, page, section) in APPLICATIONS.items()}

    # -- les traverses basses : plage de vitrage
    corps_ouv = _corps(snap, PAGE_OUVRANTS)
    t = _une(tables(corps_ouv), "Référence | Rôle et cinématique | Face vue (mm) | Largeur (mm) | Épaisseur vitrage admissible",
             PAGE_OUVRANTS)
    plages = {r[0]: lire_plage(r[4]) for r in t.lignes if r}

    # -- les joints de vitrage
    corps_vol = _corps(snap, PAGE_VOLUME)
    tv = _une(tables(corps_vol), "Référence du joint | Prise de volume (mm)", PAGE_VOLUME)
    joints = {int(_nombre(r[1])): {"ref": r[0], "couleur": r[2]} for r in tv.lignes if _nombre(r[1]) is not None}

    # -- les roulettes et la cale
    tr = tables(_corps(snap, PAGE_ROULETTES))
    t_rou = _une(tr, "Référence | Désignation imprimée | Réglage | Poids maxi par vantail", PAGE_ROULETTES)
    t_cale = _une(tr, "Traverse basse | Plage de vitrage écrite", PAGE_ROULETTES)

    return Donnees(
        statuts={p: (snap.pages[p].status or "") for p in PAGES if p in snap.pages},
        dta=dta, dta_titres=titres, dta_localisation=loc_dta, dta_validite=validite, dta_vitrages=vitrages,
        courbes=courbes, limites=limites, applications=apps, plages_traverses=plages, joints_volume=joints,
        loc_volume=tv.localisation, roulettes=lire_roulettes(t_rou), loc_roulettes=t_rou.localisation,
        cales=[t_cale], densite_verre=densite_du_verre(snap), images=images_par_reference(snap, "lumeal-ga"))


# ---- le formulaire ------------------------------------------------------------------------------


def _paires(d: Donnees) -> List[Tuple[str, str]]:
    ps = {l.paire for lignes in d.dta.values() for l in lignes}
    for c in d.courbes.values():
        ps |= set(c["colonnes"])
    return sorted(ps)


def _cle_paire(p: Tuple[str, str]) -> str:
    return "+".join(p)


def _lib_paire(p: Tuple[str, str]) -> str:
    return " + ".join(f"T{r}" if re.fullmatch(r"\d{6}", r) else r for r in p)


def options(snap: WikiSnapshot) -> Dict[str, Any]:
    d = donnees(snap)
    configurations = {k: v.libelle for k, v in d.applications.items()}
    configurations[AUTRE] = LIBELLE_AUTRE
    avec_vitrage = [k for k in configurations if k != AUTRE]
    montants = {"nomenclature": "Ceux de la nomenclature de l'application"}
    montants.update({_cle_paire(p): _lib_paire(p) for p in _paires(d)})
    recouvrements = {"inconnu": "Non précisé (le DTA ne définit pas le recouvrement)"}
    recouvrements.update({k: f"Recouvrement {k}" for k in ("mini", "maxi")})
    vents = {c: f"{c} : vent {v['pression']} Pa" for c, v in sorted(d.courbes.items())}
    epaisseurs = {str(e): f"{e} mm" + ("" if e in d.dta_vitrages else " (catalogue seulement, hors DTA)")
                  for e in sorted(d.joints_volume)}
    champs = [
        {"id": "configuration", "libelle": "Application", "type": "select", "options": configurations,
         "defaut": "fenetre2"},
        {"id": "largeur_mm", "libelle": "Largeur hors tout L", "type": "nombre", "min": 500, "max": 10000,
         "defaut": 1800, "unite": "mm"},
        {"id": "hauteur_mm", "libelle": "Hauteur hors tout H", "type": "nombre", "min": 500, "max": 10000,
         "defaut": 1250, "unite": "mm"},
        {"id": "lf_mm", "libelle": "Largeur de la partie fixe Lf", "type": "nombre", "min": 100, "max": 9000,
         "defaut": 900, "unite": "mm", "si": {"configuration": list(AVEC_LF)},
         "aide": "Lf n'est défini nulle part dans le wiki (VER-167) : saisir la cote portée sur la coupe."},
        {"id": "montants", "libelle": "Montants centraux d'ouvrant", "type": "select", "options": montants,
         "defaut": "nomenclature", "si": {"configuration": list(APPLICATIONS_COURBES)},
         "aide": "Paire côté EXT / côté INT, lue par le DTA et par les courbes."},
        {"id": "recouvrement", "libelle": "Pose en applique, recouvrement", "type": "select",
         "options": recouvrements, "defaut": "inconnu", "si": {"configuration": list(APPLICATIONS_DTA)}},
        {"id": "vent", "libelle": "Classe de vent", "type": "select", "options": vents, "defaut": "A2",
         "si": {"configuration": list(APPLICATIONS_COURBES)}},
        {"id": "epaisseur_vitrage_mm", "libelle": "Épaisseur du double vitrage", "type": "select",
         "options": epaisseurs, "defaut": "24", "si": {"configuration": avec_vitrage}},
        {"id": "verre_mm", "libelle": "Verre seul (somme des verres, sans lame)", "type": "nombre", "min": 0,
         "max": 60, "defaut": 8, "unite": "mm", "si": {"configuration": avec_vitrage},
         "aide": "4-16-4 : 8 mm de verre. Sert au poids du vitrage."},
    ]
    brouillons = sorted(p for p, s in d.statuts.items() if s == "draft")
    return {
        "champs": champs, "configurations": configurations, "pages": list(PAGES), "brouillons": brouillons,
        "avertissements": [
            "Le wiki ne donne ni le poids des profilés ni la largeur du cadre ouvrant : les contrôles de poids et de "
            "demi-périmètre ne peuvent que constater un dépassement.",
            "Les angles de coupe ne sont pas écrits par la source : la liste de coupe donne des longueurs de coupe "
            "droites, sans surlongueur de soudure ni onglet.",
            *([f"Validité du DTA : du {d.dta_validite}."] if d.dta_validite else [])],
    }


# ---- la faisabilité ------------------------------------------------------------------------------


def _variables(s: Dict[str, Any]) -> Dict[str, float]:
    v = {"H": s["hauteur_mm"], "L": s["largeur_mm"]}
    if s.get("lf_mm") is not None:
        v["Lf"] = s["lf_mm"]
    return v


def ligne_de(app: Application, designation: str) -> Optional[Ligne]:
    return next((l for l in app.profils if l.designation.lower().startswith(designation.lower())), None)


def paire_nomenclature(app: Application, variables: Dict[str, float]) -> Optional[Tuple[str, str]]:
    """La paire de montants centraux de la nomenclature : ses deux montants, une fois par pièce."""
    refs: List[str] = []
    for l in app.profils:
        if l.designation.lower().startswith("montant central"):
            q, _ = valeur_ou_none(l.qte, variables)
            refs += [normaliser_ref(l.ref)] * int(q or 0)
    return tuple(sorted(refs)) if len(refs) == 2 else None  # type: ignore[return-value]


def _montants(d: Donnees, s: Dict[str, Any], app: Application, variables: Dict[str, float]) -> Tuple[Optional[Tuple[str, str]], str]:
    if s["montants"] in (None, "nomenclature"):
        p = paire_nomenclature(app, variables)
        return p, "ceux de la nomenclature" + ("" if p else " : pas une paire")
    a, b = s["montants"].split("+")
    return (a, b), "ceux choisis"


def dimensions(app: Application, variables: Dict[str, float]) -> Dict[str, Any]:
    """Cotes des remplissages, de la traverse basse et du montant latéral, quand les formules se calculent."""
    rempl = []
    for r in app.remplissages:
        h, _ = valeur_ou_none(r["hauteur"], variables)
        l, _ = valeur_ou_none(r["largeur"], variables)
        rempl.append({"libelle": r["libelle"], "h": h, "l": l})
    traverse, montant = ligne_de(app, "Traverse basse"), ligne_de(app, "Montant latéral")
    return {
        "remplissages": rempl,
        "traverse": valeur_ou_none(traverse.debit, variables)[0] if traverse else None,
        "montant": valeur_ou_none(montant.debit, variables)[0] if montant else None,
        "nb_vantaux": int(valeur_ou_none(traverse.qte, variables)[0] or 0) if traverse else 0,
    }


def _controle_dta(d: Donnees, s: Dict[str, Any], app: Application, variables: Dict[str, float]) -> Dict[str, Any]:
    titre = "Dimensions maximales de baie (DTA)"
    src = [_source(PAGE_DTA, "Dimensions maximales (2.2.3.6)", d.dta_localisation.get("maxi", ""))]
    anomalies = ["VER-169", "CTR-135"]
    if s["configuration"] not in APPLICATIONS_DTA:
        return _controle("dta", titre, ETUDE,
                         "Le tableau du DTA donne les dimensions maximales d'une fenêtre à 2 vantaux ou à 1 vantail + "
                         "fixe ; une porte-fenêtre n'y figure pas. Au-delà, seul le Certificat de Qualification du "
                         "menuisier peut fixer la limite.", src, anomalies)
    paire, quels = _montants(d, s, app, variables)
    if paire is None:
        return _controle("dta", titre, ETUDE, f"Montants centraux {quels} : la paire n'est pas lisible.", src, anomalies)
    lignes = {k: next((l for l in v if l.paire == paire), None) for k, v in d.dta.items()}
    if not any(lignes.values()):
        return _controle("dta", titre, ETUDE,
                         f"La paire {_lib_paire(paire)} ne figure pas au tableau des dimensions maximales du DTA.",
                         src, anomalies)
    H, L = s["hauteur_mm"], s["largeur_mm"]
    dans = {k: (l is not None and H <= l.h_mm and L <= l.l_mm) for k, l in lignes.items()}
    rec = s["recouvrement"] or "inconnu"
    lim = " ; ".join(f"recouvrement {k} : {_fmt(l.h_mm / 1000, 2)} × {_fmt(l.l_mm / 1000, 2)} m" for k, l in sorted(lignes.items()) if l)
    base = (f"Baie {_fmt(H / 1000, 2)} × {_fmt(L / 1000, 2)} m (H × L) avec {_lib_paire(paire)} ({quels}) ; limites du "
            f"DTA : {lim} ({HYPOTHESE_BAIE}).")
    if rec != "inconnu":
        statut = OK if dans.get(rec) else HORS
        detail = base + (" Dans la limite." if statut == OK else " Dépasse la limite : au-delà, seul le Certificat de "
                                                                  "Qualification du menuisier peut l'admettre.")
    elif dans.get("maxi"):
        statut, detail = OK, base + " Dans la limite du recouvrement maxi, donc dans les deux."
    elif dans.get("mini"):
        statut, detail = ETUDE, base + (" Dans la limite du recouvrement mini seulement : le DTA ne définit pas le "
                                        "recouvrement (VER-169), choisir mini ou maxi.")
    else:
        statut, detail = HORS, base + " Dépasse même la limite du recouvrement mini."
    if d.dta_validite:
        detail += f" DTA valide du {d.dta_validite}."
    return _controle("dta", titre, statut, detail, src, anomalies,
                     {"paire": list(paire), "limites": {k: {"h": l.h_mm, "l": l.l_mm} for k, l in lignes.items() if l},
                      "hypothese": HYPOTHESE_BAIE})


def _controle_courbe(d: Donnees, s: Dict[str, Any], app: Application, variables: Dict[str, float]) -> Dict[str, Any]:
    titre = "Résistance au vent (courbes d'utilisation)"
    if s["configuration"] not in APPLICATIONS_COURBES:
        return _controle("courbe", titre, ETUDE,
                         "Les courbes sont tracées pour 2 vantaux et 1 vantail + fixe ; cette application n'a pas de "
                         "courbe dans le wiki.", [_source(PAGE_COURBES, "Conditions d'emploi des courbes")])
    classe = s["vent"] or "A2"
    cb = d.courbes.get(classe)
    if cb is None:
        return _controle("courbe", titre, ETUDE, f"Aucun abaque V*{classe} dans le wiki.", [_source(PAGE_COURBES)])
    src = [_source(PAGE_COURBES, cb["section"], cb["localisation"])]
    paire, quels = _montants(d, s, app, variables)
    if paire is None or paire not in cb["colonnes"]:
        tracees = " ; ".join(_lib_paire(p) for p in sorted(cb["colonnes"]))
        return _controle("courbe", titre, ETUDE,
                         f"La paire {_lib_paire(paire) if paire else quels} n'a pas de courbe sur l'abaque V*{classe} "
                         f"(tracées : {tracees}).", src)
    lecture = lire_sur_courbe(cb["colonnes"][paire], s["largeur_mm"], s["hauteur_mm"], d.limites["cadre"],
                              d.limites["precision"])
    detail = (f"V*{classe} ({cb['pression']} Pa), courbe {cb['entetes'][paire]} : hauteur {_fmt(s['hauteur_mm'])} mm "
              f"pour L = {_fmt(s['largeur_mm'])} mm. {lecture.detail[0].upper() + lecture.detail[1:]}. "
              "Domaine admis : H sous la courbe (« H hors tout maximal »).")
    return _controle("courbe", titre, lecture.statut, detail, src, ["INC-444"] if "141024" in paire else [],
                     {"h_max": None if lecture.h_max in (None, float("inf")) else lecture.h_max,
                      "paire": list(paire), "classe": classe})


def graphe_courbes(cb: Dict[str, Any], paire: Any, largeur: float, hauteur: float, statut: str) -> Dict[str, Any]:
    """Le tracé d'un abaque de courbes pour la page : toutes les courbes relevées (hauteur maximale selon L), celle de
    la paire saisie mise en avant, et le point saisi. Les graduations sont celles du wiki, jamais interpolées."""
    courbes = []
    for p, courbe in sorted(cb["colonnes"].items()):
        courbes.append({"nom": cb["entetes"][p], "choisie": p == tuple(paire),
                        "points": [[l, h] for l, h in sorted(courbe.items()) if h is not None]})
    return {"titre": f"Courbes d'utilisation, {cb['pression']} Pa", "x": "Largeur L (mm)", "y": "Hauteur H maximale (mm)",
            "courbes": courbes, "point": [largeur, hauteur], "statut": statut}


def graphe_du_controle(ctl: List[Dict[str, Any]], cb_de_classe: Any, largeur: float, hauteur: float) -> Optional[Dict[str, Any]]:
    """Le tracé du contrôle « courbe » s'il a pu être lu ; ``cb_de_classe(classe)`` rend l'abaque de la classe."""
    c = next((c for c in ctl if c["id"] == "courbe"), None)
    if not c or not c["valeurs"].get("classe") or not c["valeurs"].get("paire"):
        return None
    return graphe_courbes(cb_de_classe(c["valeurs"]["classe"]), c["valeurs"]["paire"], largeur, hauteur, c["statut"])


def poids_du_verre(densite: Optional[float], verre_mm: float, dim: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """Poids de verre du plus petit remplissage de vantail : une borne basse, jamais le poids du vantail."""
    if not densite:
        return None, "la densité du verre n'est pas dans le wiki"
    vantaux = [r for r in dim["remplissages"] if r["libelle"] == "le vantail"] or dim["remplissages"]
    poids = [(verre_mm or 0) * densite * (r["h"] / 1000) * (r["l"] / 1000) for r in vantaux if r["h"] and r["l"]]
    return (min(poids), "") if poids else (None, "remplissage non calculable")


def controles_limites(densite: Optional[float], limites: Dict[str, float], verre_mm: float, dim: Dict[str, Any],
                      sources: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Les deux limites imprimées sous les courbes : la charge de l'ouvrant et son demi-périmètre.

    Le wiki ne donne ni le poids des profilés ni la largeur du cadre ouvrant : le verre seul et la borne basse du
    demi-périmètre ne peuvent que **dépasser** la limite (hors domaine) ; en dessous, ``info``, jamais un « oui »."""
    out: List[Dict[str, Any]] = []
    poids, raison = poids_du_verre(densite, verre_mm, dim)
    kg = limites["kg"]
    if poids is None:
        out.append(_controle("poids", "Charge de l'ouvrant", ETUDE, f"Poids de verre non calculé : {raison}.", sources))
    else:
        verdict = HORS if poids > kg else INFO
        txt = (f"{_fmt(poids, 1)} kg de verre par vantail ({_fmt(verre_mm, 1 if verre_mm % 1 else 0)} mm × "
               f"{_fmt(densite, 1)} kg/m² par mm, hypothèse transposée de la page des abaques du système 70 : "
               f"aucune page de ce système ne donne la masse du verre) pour {_fmt(kg)} kg maxi par ouvrant. "
               + ("La charge est dépassée par le verre seul." if verdict == HORS else
                  "Le wiki ne donne pas le poids des profilés : sous la limite, le verre seul ne prouve pas que "
                  "l'ouvrant la respecte."))
        out.append(_controle("poids", "Charge de l'ouvrant", verdict, txt,
                             sources + [_source(PAGE_VERRE, "Remarques importantes concernant les abaques de renforcement")],
                             [], {"poids_verre_kg": round(poids, 1), "limite_kg": kg}))
    if dim["traverse"] is not None and dim["montant"] is not None:
        demi = dim["traverse"] + dim["montant"]
        limite = limites["demi_perimetre"]
        txt = (f"Traverse basse {_fmt(dim['traverse'], 1 if dim['traverse'] % 1 else 0)} + montant latéral "
               f"{_fmt(dim['montant'], 1 if dim['montant'] % 1 else 0)} = {_fmt(demi, 1 if demi % 1 else 0)} mm : borne "
               f"basse du demi-périmètre du vantail (la largeur des montants n'est pas comptée, le wiki ne donne pas le "
               f"cadre ouvrant), pour {_fmt(limite)} mm maxi.")
        out.append(_controle("demi_perimetre", "Demi-périmètre de l'ouvrant", HORS if demi > limite else INFO, txt,
                             sources, [], {"borne_basse_mm": demi, "limite_mm": limite}))
    return out


def controle_roulettes(roulettes: Dict[str, Dict[str, Any]], sources: List[Dict[str, str]], app: Application,
                       poids: Optional[float], raison: str, anomalies: List[str],
                       hypothese: str = HYPOTHESE_UNITE) -> Dict[str, Any]:
    """La roulette de la nomenclature face au poids de verre. Le poids maxi du tableau est imprimé sans unité
    (VER-239, VER-203) : hypothèse « kg par vantail », affichée ; jamais plus que « info », ou « sur étude »."""
    titre = "Roulettes"
    rou = next((l for l in app.accessoires if l.designation.lower().startswith("roulette")), None)
    if rou is None or rou.ref not in roulettes or poids is None:
        return _controle("roulettes", titre, ETUDE, "Roulette ou poids non lisible : " + (raison or
                         "la nomenclature ne porte pas de roulette connue du tableau."), sources, anomalies)
    m = roulettes[rou.ref]["poids"]
    convient = sorted(r for r, v in roulettes.items() if v["poids"] >= poids)
    if poids <= m:
        return _controle("roulettes", titre, INFO,
                         f"Roulette {rou.ref} de la nomenclature ({rou.designation}) : poids maxi {_fmt(m)} pour "
                         f"{_fmt(poids, 1)} kg de verre par vantail ({hypothese} ; profilés non comptés).",
                         sources, anomalies, {"roulette": rou.ref})
    if convient:
        return _controle("roulettes", titre, ETUDE,
                         f"{_fmt(poids, 1)} kg de verre par vantail dépassent le poids maxi {_fmt(m)} de la roulette "
                         f"{rou.ref} de la nomenclature ({hypothese}) ; roulettes dont le poids maxi est au moins "
                         f"{_fmt(poids, 1)} : {', '.join(convient)}.", sources, anomalies,
                         {"roulette": rou.ref, "roulettes_possibles": convient})
    return _controle("roulettes", titre, ETUDE,
                     f"{_fmt(poids, 1)} kg de verre par vantail dépassent le poids maxi de toutes les roulettes du "
                     f"tableau ({hypothese}).", sources, anomalies, {"roulette": rou.ref})


def _controle_cale(d: Donnees, app: Application) -> List[Dict[str, Any]]:
    """La cale TGA3817 : deux catalogues, deux conditions qui ne s'accordent pas (CTR-198) : information."""
    traverse = ligne_de(app, "Traverse basse")
    lignes = cale_ou_vide(d.cales[0], traverse.ref) if traverse else []
    if not lignes:
        return []
    r = lignes[0]
    return [_controle("cale", "Cale de vitrage TGA3817", INFO,
                      f"Traverse basse {traverse.ref} : catalogue de conception : {r[2]} ; catalogue de "
                      f"fabrication : {r[3]}. Les deux ne s'accordent pas, la condition n'est pas décidée par ce "
                      "calcul.", [_source(PAGE_ROULETTES, "Charge admissible et calage : conception et fabrication")],
                      ["CTR-198", "CTR-196"], {"traverse": traverse.ref})]


def plage_de_nomenclature(d: Donnees, app: Application) -> Tuple[Optional[Ligne], Optional[Tuple[float, float]]]:
    """La traverse basse de la nomenclature et l'épaisseur de vitrage que le wiki lui attribue."""
    traverse = ligne_de(app, "Traverse basse")
    return traverse, (d.plages_traverses.get(traverse.ref) if traverse else None)


def _controle_vitrage(d: Donnees, s: Dict[str, Any], app: Application, paire: Optional[Tuple[str, str]]) -> Dict[str, Any]:
    titre = "Vitrage et joint"
    n = int(s["epaisseur_vitrage_mm"])
    j = d.joints_volume.get(n)
    src = [_source(PAGE_VOLUME, "Joints de l'ouvrant", d.loc_volume), _source(PAGE_DTA, "Vitrage (2.2.3.5)")]
    traverse, plage = plage_de_nomenclature(d, app)
    verre = s["verre_mm"] or 0
    anomalies = ["CTR-134"]
    if j is None:
        return _controle("vitrage", titre, HORS, f"Aucune prise de volume de {n} mm.", src, anomalies)
    base = f"Double vitrage de {n} mm : joint {j['ref']} (repère {j['couleur']})."
    valeurs = {"joint": j["ref"], "couleur": j["couleur"], "epaisseur": n}
    if verre >= n:
        return _controle("vitrage", titre, ETUDE, base + f" Le verre seul ({_fmt(verre, 1)} mm) n'est pas inférieur à "
                                                         "l'épaisseur du vitrage.", src, anomalies, valeurs)
    # chaque montant central dont le wiki écrit l'épaisseur de vitrage admissible doit la respecter
    hors = []
    for ref in paire or ():
        p = d.plages_traverses.get(f"T{ref}" if ref.isdigit() else ref)
        if p and not p[0] <= n <= p[1]:
            hors.append(f"T{ref} ({_fmt(p[0])} à {_fmt(p[1])} mm)" if p[0] != p[1] else f"T{ref} ({_fmt(p[0])} mm)")
    if hors:
        return _controle("vitrage", titre, ETUDE,
                         base + f" Épaisseur admissible imprimée pour le montant central {', '.join(hors)} : le montant "
                                "choisi ne convient pas à cette épaisseur.",
                         src + [_source(PAGE_OUVRANTS, "3. Profilés ouvrants (ouvrant caché)")], anomalies, valeurs)
    if n not in d.dta_vitrages:
        return _controle("vitrage", titre, ETUDE,
                         base + f" Le DTA n'admet que {', '.join(str(x) for x in d.dta_vitrages)} mm : cette épaisseur "
                                "n'est que dans le catalogue (CTR-134).", src, anomalies, valeurs)
    suite = ""
    if plage and not plage[0] <= n <= plage[1]:
        suite = (f" La nomenclature de cette application porte la traverse basse {traverse.ref} ({_fmt(plage[0])} à "
                 f"{_fmt(plage[1])} mm) : la liste de coupe n'est pas établie à cette épaisseur.")
    return _controle("vitrage", titre, OK, base + " Épaisseur admise par le DTA." + suite, src, anomalies, valeurs)


def verifier(snap: WikiSnapshot, saisie: Dict[str, Any]) -> Dict[str, Any]:
    d = donnees(snap)
    s = coercer(options(snap)["champs"], saisie)
    if s["configuration"] == AUTRE:
        return wiki_famille.relocaliser(snap, synthese(
            [_controle("application", "Application", ETUDE,
                       "Aucune nomenclature dans le wiki pour la porte-fenêtre 6 vantaux 3 rails (désignation seule) ni "
                       "pour 2 vantaux + 2 fixes latéraux : ni DTA, ni courbe, ni liste de coupe.",
                       [_source(PAGE_DEBIT, "Désignation symbolique des châssis")], ["VER-242"])],
            _brouillons(d, [PAGE_DEBIT])))
    app = d.applications[s["configuration"]]
    variables = _variables(s)
    ctl: List[Dict[str, Any]] = []
    if s["configuration"] in AVEC_LF and not 0 < s["lf_mm"] < s["largeur_mm"]:
        raise ValueError("Largeur de la partie fixe Lf : doit être comprise entre 0 et la largeur hors tout")
    if s["configuration"] in AVEC_LF:
        ctl.append(_controle("lf", "Largeur de la partie fixe", INFO,
                             f"Lf = {_fmt(s['lf_mm'])} mm saisi ; sa définition n'est pas écrite par la source (VER-167).",
                             [_source(app.page, app.section, app.localisation)], ["VER-167"]))
    dim = dimensions(app, variables)
    ctl.append(_controle_dta(d, s, app, variables))
    ctl.append(_controle_courbe(d, s, app, variables))
    ctl += controles_limites(d.densite_verre, d.limites, s["verre_mm"], dim,
                             [_source(PAGE_COURBES, "Conditions d'emploi des courbes")])
    paire, _ = _montants(d, s, app, variables)
    ctl.append(_controle_vitrage(d, s, app, paire))
    poids, raison = poids_du_verre(d.densite_verre, s["verre_mm"], dim)
    ctl.append(controle_roulettes(d.roulettes, [_source(PAGE_ROULETTES, "Roulettes", d.loc_roulettes)], app, poids, raison,
                                  ["VER-239", "CTR-197", "VER-332"]))
    ctl += _controle_cale(d, app)
    pages = [PAGE_DTA, PAGE_COURBES, PAGE_VOLUME, PAGE_ROULETTES, PAGE_OUVRANTS, app.page]
    return wiki_famille.relocaliser(snap, synthese(
        ctl, _brouillons(d, pages), application=app.libelle,
        vitrage={"epaisseur_mm": int(s["epaisseur_vitrage_mm"]), "verre_mm": s["verre_mm"],
                 "joint": d.joints_volume[int(s["epaisseur_vitrage_mm"])]["ref"]},
        remplissages=[{"libelle": r["libelle"], "hauteur": r["h"], "largeur": r["l"]} for r in dim["remplissages"]],
        graphe_courbes=graphe_du_controle(ctl, lambda classe: d.courbes[classe], s["largeur_mm"], s["hauteur_mm"]),
        poids_verre_kg=None if poids is None else round(poids, 1)))


def _brouillons(d: Donnees, pages: List[str]) -> List[str]:
    return [p for p in dict.fromkeys(pages) if d.statuts.get(p) == "draft"]


# ---- la fiche de débit ---------------------------------------------------------------------------

FAMILLES = {"profils": "Profilés", "joints": "Joints", "accessoires": "Accessoires", "vis": "Visserie"}


def construire_fiche(app: Application, variables: Dict[str, float], epaisseur: int, joint_volume: Dict[str, str],
                     images: Dict[str, str], sections: Optional[Dict[str, str]] = None,
                     nb_vitrages: Optional[int] = None, raison_nb: str = "") -> Dict[str, Any]:
    """La liste de coupe, la nomenclature et les notes d'une application (``faisabilite_gy`` la réutilise).

    Chaque ligne du wiki est calculée ou, si une formule échappe à la grammaire, listée dans les notes avec sa
    raison. Le joint de vitrage de la nomenclature est celui d'une seule épaisseur : à une autre épaisseur la ligne
    prend le joint de sa prise de volume avec la même longueur, et le dit (hypothèse).

    Le nombre de vitrages d'un remplissage unique est ``nb_vitrages`` (avec sa ``raison_nb``) ; à défaut la quantité de
    la traverse basse d'ouvrant, une par vantail."""
    ev = evaluer_application(app, variables)
    lignes: List[Dict[str, Any]] = []
    notes: List[str] = []
    non_calculees: List[str] = []
    sections = sections or {}

    def ajoute(genre: str, famille: str, piece: str, ref: str, qte: Optional[float], longueur: Optional[float],
               calcul: str, coupe: str = "", note: str = "") -> None:
        lignes.append({"rep": len(lignes) + 1, "famille": famille, "piece": piece, "ref": ref, "qte": qte,
                       "longueur": longueur, "calcul": calcul, "coupe": coupe,
                       "source": {"page": app.page, "section": sections.get(genre, app.section),
                                  "localisation": app.localisation},
                       "note": note, "image": images.get(ref)})

    for e in ev["lignes"]:
        l: Ligne = e["ligne"]
        if e["raison"] or e["qte"] is None:
            non_calculees.append(f"{l.ref} {l.designation} : {e['raison'] or 'quantité illisible'}")
            continue
        note_l, ref = "", l.ref
        if e["genre"] == "joints":
            mj = re.fullmatch(r"Joint vitrage (\d+)\s*mm", l.designation.strip())
            if mj and int(mj.group(1)) != epaisseur:
                ref = joint_volume["ref"]
                note_l = (f"joint de la prise de volume {epaisseur} mm ; la nomenclature ne donne la longueur que "
                          f"pour {mj.group(1)} mm ({l.ref}), même longueur reprise : hypothèse")
        calcul = l.debit if l.debit else (l.qte if not re.fullmatch(r"\d+", l.qte) else "")
        if e["genre"] == "joints":
            calcul = f"{l.debit} : longueur totale de joint"
        qte = round(e["qte"], 2)
        if qte != int(qte) and e["genre"] != "joints":
            note_l = (note_l + " ; " if note_l else "") + f"quantité écrite en formule ({l.qte}) : arrondi non écrit"
        ajoute(e["genre"], FAMILLES[e["genre"]], l.designation or l.ref, ref, int(qte) if qte == int(qte) else qte,
               e["longueur"], calcul, f"pictogramme « {l.coupe} » (non légendé)" if l.coupe else "", note_l)
    # le remplissage : une ligne par cote imprimée
    dim = dimensions(app, variables)
    nb = nb_vitrages or dim["nb_vantaux"] or None
    for r, rr in zip(app.remplissages, dim["remplissages"]):
        if rr["h"] is None or rr["l"] is None:
            non_calculees.append(f"remplissage {r['hauteur']} × {r['largeur']} : formule non calculée")
            continue
        un = len(app.remplissages) == 1 and nb
        ajoute("remplissage", "Remplissage", f"Remplissage (vitrage de {epaisseur} mm) {r['libelle']}".strip(), "—",
               nb if un else 1, None,
               f"{r['hauteur']} × {r['largeur']} = {_fmt(rr['h'], 1 if rr['h'] % 1 else 0)} × "
               f"{_fmt(rr['l'], 1 if rr['l'] % 1 else 0)} mm", "",
               (raison_nb or "un par vantail (quantité de la traverse basse d'ouvrant)" if un else
                "quantité non écrite par la planche : une ligne par cote imprimée" +
                ("" if r["libelle"] else ", la planche ne dit pas laquelle est le vantail")))
    ordre = list(dict.fromkeys(l["famille"] for l in lignes))
    lignes.sort(key=lambda l: ordre.index(l["famille"]))
    for i, l in enumerate(lignes, 1):
        l["rep"] = i

    nomen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for l in lignes:
        k = (l["piece"] if l["ref"] == "—" else l["ref"], l["famille"])
        x = nomen.setdefault(k, {"ref": l["ref"], "designation": l["piece"].strip(), "famille": l["famille"], "qte": 0,
                                 "ml": 0.0, "image": l["image"]})
        x["qte"] += l["qte"] or 0
        if l["longueur"] is not None:
            x["ml"] += (l["qte"] or 1) * l["longueur"] / 1000
    notes.append("Longueurs de coupe droites, sans surlongueur : la source n'écrit pas les angles de coupe ni de "
                 "surlongueur de soudure ou d'onglet.")
    if app.anomalies:
        notes.append("Anomalies ouvertes citées par la page pour cette application : " + ", ".join(app.anomalies) + ".")
    if non_calculees:
        notes.append("Lignes non calculées : " + " ; ".join(non_calculees) + ".")
    return {"lignes": lignes, "notes": notes,
            "nomenclature": [dict(v, ml=round(v["ml"], 2) if v["ml"] else None) for v in nomen.values()]}


def fiche_debit(snap: WikiSnapshot, saisie: Dict[str, Any]) -> Dict[str, Any]:
    d = donnees(snap)
    s = coercer(options(snap)["champs"], saisie)
    if s["configuration"] == AUTRE:
        return {"indisponible": "Aucune nomenclature dans le wiki pour cette application (6 vantaux 3 rails, "
                                "2 vantaux + 2 fixes latéraux) : pas de liste de coupe."}
    app = d.applications[s["configuration"]]
    n = int(s["epaisseur_vitrage_mm"])
    traverse, plage = plage_de_nomenclature(d, app)
    if plage and not plage[0] <= n <= plage[1]:
        return {"indisponible": f"La nomenclature de cette application porte la traverse basse {traverse.ref} "
                                f"({_fmt(plage[0])} à {_fmt(plage[1])} mm) : le wiki n'a pas la liste de coupe à "
                                f"{n} mm pour elle."}
    ver = verifier(snap, saisie)
    fiche = construire_fiche(app, _variables(s), n, d.joints_volume[n], d.images)
    if (s["montants"] or "nomenclature") != "nomenclature":
        fiche["notes"].append("Le choix des montants centraux ne modifie pas la nomenclature : elle est celle de "
                              "l'application.")
    return wiki_famille.relocaliser(snap, {
        "verdict": ver["verdict"], "libelle": ver["libelle"], "lignes": fiche["lignes"],
        "nomenclature": fiche["nomenclature"], "accessoires": [], "notes": fiche["notes"],
        "brouillons": _brouillons(d, [app.page, PAGE_VOLUME]), "application": app.libelle})
