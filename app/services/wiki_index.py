"""L'index de navigation du wiki : recherche lexicale, carte, fiche de la question, anomalies.

Le wiki ne tient pas dans une fenêtre de contexte (plus de 300 pages, ~2 M tokens). Le prompt
permanent porte ``index.md`` (une ligne par page) et un vocabulaire ; GLM navigue : il formule
une recherche, lit une **carte**, puis lit les sections qu'il choisit. Les entrées d'anomalie
rapprochées des pages lues sont poussées par le serveur.

La recherche est **lexicale** (BM25 sur texte intégral, par section et par page), pas
vectorielle : deux pages qui se contredisent doivent toutes les deux remonter, alors qu'un
top-k sémantique les met en concurrence. Quelques choix méritent d'être rappelés :

* les **facettes remontent une page, elles ne l'excluent pas**. Une facette mal choisie
  cachait la bonne page : « garantie structure LUMINE65 » filtré par ``tags=coulissant``
  écartait la page des garanties. Il ne reste que ``gamme`` et ``systeme``. Le ``type`` ne
  compte plus dans le classement : le modèle le devine (« Profilé » pour une limite que fixe un
  DTA ou une page de gamme) et, compté double, il reléguait la page qui répond — 68 % des
  recherches réelles de Mistral trouvaient la bonne page dans les trois premières, 85 % sans lui
  (30/09/2026) ;
* une **référence** (76526, NT1947, A076) vaut trois mots ordinaires : c'est le signal le
  plus sûr de la question d'un menuisier, et elle ne figure dans aucun tag. Une **cote** de
  la question (« 1 800 mm », « 1 200 de large ») a la forme d'une référence mais n'en est
  pas une : elle ne reçoit pas ce poids, sans quoi « SoftOpen 1800 » ramenait les grands
  tableaux de ferrures devant la page du coulissant. Un nom de gamme suivi de son épaisseur
  (PERFORM76, LUMINE55) non plus : c'est un produit, pas une pièce (``_reference``) ;
* les mots sont **ramenés à une graphie** : sans accent ni ligature (« manœuvre » =
  « manoeuvre »), chiffres groupés recollés (« 487 206 » = « 487206 »), un nom suivi d'un
  nombre également collé (« LUMINE 65 » = « LUMINE65 »), pluriel en -s ou -x des mots de plus
  de quatre lettres retiré (« parcloses » = « parclose »). Ce n'est pas une lemmatisation :
  « vantaux » ne trouve pas « vantail » ;
* la recherche se fait **par section** et rend une **carte** (02/10/2026). Les pages sont
  découpées à l'indexation (``decouper_sections``), jamais dans le wiki ; une page est classée
  par ses meilleures sections ; plusieurs formulations se fusionnent par rang réciproque
  (``classer_multi``). La carte donne douze résultats d'environ un millier de caractères — la
  page, ses sections et les lignes qui répondent avec l'en-tête de leur tableau — et ne livre
  que la page qui domine nettement le classement. Mesuré hors ligne : la section de la preuve
  est désignée dans 100 % du golden et 93 % du banc pour ~17 000 caractères, là où six pages
  livrées en coûtaient 52 000 ;
* la **fiche de la question** (``fiche_question``) est calculée avant tout appel au modèle :
  les références rares avec leurs emplacements exacts, celles qu'aucune page ne porte, les cotes ;
* les registres d'``anomalies/`` ne sont pas livrés — ils se lisent entrée par entrée, et les
  entrées utiles sont injectées par le serveur — et les pages ``sources/`` pèsent moitié moins
  que les pages concept : une source résume un document, une page concept porte la valeur.

L'index est construit avec l'instantané (``wiki_service.load_snapshot``) et jeté avec lui dès
qu'un fichier change. Il ne relit pas les fichiers : il travaille sur les pages déjà lues.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import log
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

TOKEN_RE = re.compile(r"[a-z0-9]+")
LIEN_RE = re.compile(r"\((/[^)\s]+\.md)\)")
ENTREE_RE = re.compile(r"\A(INC|CTR|VER)-\d+\Z")

ANOMALY_PAGES = (
    "/anomalies/incoherences-internes.md",
    "/anomalies/contradictions-entre-sources.md",
    "/anomalies/informations-a-verifier.md",
)

K1, B = 1.5, 0.75

VIDES = frozenset({
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "ou", "que", "qui",
    "quel", "quelle", "quels", "quelles", "est", "sont", "pour", "sur", "en",
    "dans", "par", "au", "aux", "ce", "cette", "avec", "peut", "on", "il",
    "elle", "se", "sa", "son", "ses", "plus", "pas", "ne", "quoi", "comment",
    "combien", "faut", "doit", "entre", "chez", "sans", "leur", "lors", "a",
})


def deaccent(texte: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texte) if unicodedata.category(c) != "Mn"
    )


# « œ » et « æ » ne se décomposent pas en NFD : sans cette table, « manœuvre » ne trouvait pas
# « manoeuvre ».
LIGATURES = str.maketrans({"œ": "oe", "æ": "ae"})
# Un nombre imprimé en groupes de trois chiffres : « 487 206 », « 1 800 ». Les virgules et les
# points en sont exclus de part et d'autre, pour ne pas recoller « 2,15 1,00 ».
GROUPES_RE = re.compile(r"(?<![\d,.])(\d{1,3})((?:[   ]\d{3})+)(?![\d,])")


def _recolle(texte: str) -> str:
    return GROUPES_RE.sub(lambda m: m.group(1) + re.sub(r"\D", "", m.group(2)), texte)


def _racine(jeton: str) -> str:
    """Le pluriel en -s ou -x d'un mot de plus de quatre lettres, rien d'autre."""
    if len(jeton) > 4 and jeton[-1] in "sx" and jeton.isalpha() and not jeton.endswith("plus"):
        return jeton[:-1]
    return jeton


# « PERFORM+ » et « HYBRIDE+ » sont des gammes à part : sans cette règle, le « + » tombait et la
# PERFORM+ se confondait avec la PERFORM (Q38, 01/10).
PLUS_RE = re.compile(r"(?<=[a-z0-9])\+")


def mots(texte: Any) -> List[str]:
    """Les mots tels qu'ils sont écrits, ramenés à une graphie — sans les noms collés."""
    plat = PLUS_RE.sub("plus", _recolle(deaccent(str(texte).lower().translate(LIGATURES))))
    return [_racine(t) for t in TOKEN_RE.findall(plat)]


def tokenise(texte: Any) -> List[str]:
    """Les mots d'un texte, ramenés à une graphie (voir l'en-tête du module).

    Appliquée à l'identique aux pages et aux requêtes : c'est ce qui rend les deux graphies
    équivalentes, et c'est pourquoi aucune n'est jamais réécrite dans le wiki.
    """
    jetons = mots(texte)
    colles = [
        a + b
        for a, b in zip(jetons, jetons[1:])
        if a.isalpha() and len(a) >= 3 and a not in VIDES and b.isdigit() and 2 <= len(b) <= 3
    ]
    return jetons + colles


def normalise(valeur: Any) -> List[str]:
    """Rend une liste de chaînes, que la valeur soit une chaîne, un entier ou une liste.

    ``gamme`` et ``systeme`` sont saisis tantôt en chaîne (``PERFORM``), tantôt en entier
    (``55``), tantôt en liste (``[55, 65]``) : tout lecteur passe par ici.
    """
    if valeur is None:
        return []
    if isinstance(valeur, (list, tuple, set)):
        return [str(v).strip() for v in valeur if str(v).strip()]
    texte = str(valeur).strip()
    return [texte] if texte else []


def est_reference(token: str) -> bool:
    """Un token de référence : 76526, nt1947, a076 — ce que le menuisier demande."""
    return len(token) >= 3 and any(c.isdigit() for c in token)


def references(texte: str) -> Set[str]:
    return set(REFERENCE_RE.findall(texte or ""))


# Une dimension dans la question : un nombre suivi de son unité, de « de large / de haut », ou
# pris dans un « L × H ». Le nombre précédé de lettres (PERFORM76) n'en est pas une.
_CHIFFRES = r"(\d+)(?:[.,]\d+)?"
_NOMBRE = r"(?<![a-z0-9])" + _CHIFFRES
COTE_RES = (
    re.compile(_NOMBRE + r"\s*(?:mm|cm|m|kg|dan|pa|°)(?![a-z])"),
    re.compile(_NOMBRE + r"\s*(?:de\s+)?(?:large|haut|long|largeur|hauteur|longueur|epaisseur)\b"),
    re.compile(_NOMBRE + r"\s*[x×*]\s*(?=\d)"),
    re.compile(r"[x×*]\s*" + _CHIFFRES),
)


def cotes_de(question: str) -> Set[str]:
    """Les nombres que la question donne comme dimensions, dans leur forme de jeton."""
    plat = _recolle(deaccent(str(question or "").lower()))
    return {m.group(1) for motif in COTE_RES for m in motif.finditer(plat)}


def references_de(question: str) -> List[str]:
    """Les références que porte la question (TGY3702, 76507, LUMINE65), cotes exclues."""
    cotes = cotes_de(question)
    vues: List[str] = []
    # Les mots écrits, pas les noms collés : « vitrage 24 » ne fait pas une référence.
    for jeton in mots(question):
        if est_reference(jeton) and jeton not in cotes and jeton not in vues:
            vues.append(jeton)
    return vues


# Une référence de menuiserie est un nombre de 3 à 6 chiffres (76507, 2636), parfois précédé
# de lettres (A076, NT1947). Seuls les chiffres servent au rapprochement : c'est ce que
# portent les noms de fichiers (parclose-76507.png).
REFERENCE_RE = re.compile(r"\b[A-Za-z]{0,3}(\d{3,6})\b")


def _sac_de_tokens(page: Any) -> List[str]:
    """Les champs forts sont répétés : pondération par champ sans BM25 multi-champs."""
    sac = tokenise(page.title) * 5
    for tag in page.tags:
        sac += tokenise(tag) * 4
    sac += tokenise(page.description) * 3
    for valeurs in (page.gamme, page.systeme, page.famille):
        for valeur in valeurs:
            sac += tokenise(valeur) * 3
    return sac + tokenise(page.body)


# ---------------------------------------------------------------------------
# Sections : la découpe se fait ici, à l'indexation, jamais dans le wiki
# ---------------------------------------------------------------------------

# Une section au-delà de cette taille est recoupée. 2 500, 5 000 et 8 000 ont été mesurés le
# 01/10 : ±3 points, le budget de livraison compte bien plus que le grain.
SECTION_MAX = 5_000
# Une page plus petite est livrée entière : une donnée = une page, et ses exceptions (« sur
# étude », « à ne pas confondre ») sont souvent dans un paragraphe voisin du tableau. 15, 20 et
# 30 k mesurés le 01/10 : 15 k fait aussi bien ou mieux sur le golden et le banc, avec moins de
# volume (preuve livrée 95 % et 91 %, 52 k caractères par recherche).
PAGE_ENTIERE_MAX = 15_000
# Classement d'une page par ses sections : la meilleure, une part de la deuxième, et le score
# de la page entière (l'index historique) pour départager.
W_TITRES = 4
W_TITRE_PAGE = 2
W_SECONDE = 0.5
W_PAGE = 0.3
POIDS_SOURCE = 0.5
SOMMAIRE_MAX = 40
# La carte (02/10/2026) : ce que lit un modèle qui navigue au lieu de recevoir des pages. Douze
# résultats d'environ un millier de caractères chacun (la page, ses sections qui répondent et les
# lignes qui contiennent les mots cherchés, avec l'en-tête de leur tableau) contre six pages
# livrées entières : mesuré hors ligne le 02/10, la section de la preuve y est désignée pour 100 %
# du golden et 93 % du banc, pour ~17 000 caractères au lieu de 52 000.
CARTE_PAGES = 12
CARTE_SECTIONS = 5
CARTE_SECTIONS_AVEC_LIGNES = 3
CARTE_LIGNES = 4
CARTE_BUDGET = 24_000
# Plusieurs formulations d'une même question sont fusionnées par rang réciproque (RRF), sur les
# trente premières pages de chacune : une relance coûte un appel, la fusion n'en coûte aucun.
RRF_K = 10
RRF_PROFONDEUR = 30
# Une référence n'a de sens pour une recherche exacte que si peu de sections la portent : au-delà
# (PERFORM76, 2024, un RAL dans un nuancier), c'est un terme du wiki, pas une pièce.
REFERENCE_RARE_MAX = 10
# Une page dont la suivante ne vaut pas la moitié du classement est LA page : elle arrive entière.
DOMINANCE = 0.5

TITRE_RE = re.compile(r"^(#{1,4})\s+(.*?)\s*#*\s*$")


def _lignes_frontmatter(lignes: Sequence[str]) -> int:
    """Le nombre de lignes de la frontmatter (0 si la page n'en a pas)."""
    if lignes and lignes[0].strip() == "---":
        for n in range(1, len(lignes)):
            if lignes[n].strip() == "---":
                return n + 1
    return 0


def decouper_sections(chemin: str, texte: str, limite: int = SECTION_MAX) -> List[Dict[str, Any]]:
    """Les sections d'une page, dans l'ordre, numérotées à partir de 1.

    * une section commence à un titre ``#`` à ``####`` ; ce qui précède le premier titre est la
      section d'ouverture ;
    * au-delà de ``limite`` caractères, elle est recoupée à une fin de paragraphe, à une ligne de
      tableau markdown ou à une fin de ligne HTML ``</tr>`` — jamais au milieu d'une ligne ;
    * **un tableau coupé garde son en-tête** : la ligne d'en-tête et le séparateur en markdown,
      toutes les lignes ``<th>`` avant la première ``<td>`` en HTML (les en-têtes à plusieurs
      niveaux, comme la matrice des parcloses SOLEAL FY, restent lisibles) ;
    * ``debut`` et ``fin`` sont les lignes du fichier, frontmatter comprise : ce sont celles que
      citent les preuves et que ``lire`` reconstitue.
    """
    lignes = texte.splitlines()
    premier = _lignes_frontmatter(lignes)
    sections: List[Dict[str, Any]] = []
    pile: List[Tuple[int, str]] = []
    courant: List[str] = []
    debut, taille, suite = premier + 1, 0, False
    tableau: Optional[Tuple[str, List[str]]] = None  # ("md" | "html", lignes d'en-tête)
    entete_html_fini = False

    def emettre(fin: int, coupe: bool) -> None:
        nonlocal courant, debut, taille
        if any(l.strip() for l in courant):
            corps = "\n".join(courant)
            if coupe and tableau and tableau[0] == "html":
                corps += "\n</tbody></table>"
            sections.append({
                "chemin": chemin,
                "numero": len(sections) + 1,
                "debut": debut,
                "fin": fin,
                "titres": [t for _, t in pile],
                "texte": corps,
                "suite": suite,
                "coupe_tableau": coupe,
            })
        courant, debut, taille = [], fin + 1, 0

    for n, ligne in enumerate(lignes[premier:], start=premier + 1):
        bas = ligne.lower()
        titre = TITRE_RE.match(ligne)
        if titre and not (tableau and tableau[0] == "html"):
            # Un titre suivi aussitôt d'un autre titre n'est pas une section : il rejoint la
            # suivante (sinon « Prescriptions », vide, arrivait parmi les meilleures sections).
            seulement_titres = courant and all(TITRE_RE.match(l) or not l.strip() for l in courant)
            if seulement_titres:
                courant, debut_report = courant, debut
            else:
                emettre(n - 1, False)
                courant, debut_report = [], n
            niveau = len(titre.group(1))
            pile = [(a, t) for a, t in pile if a < niveau] + [(niveau, titre.group(2))]
            courant = courant + [ligne]
            debut, taille, suite, tableau = debut_report, sum(len(l) + 1 for l in courant), False, None
            continue
        if "<table" in bas and tableau is None:
            tableau, entete_html_fini = ("html", [ligne]), False
        elif tableau and tableau[0] == "html" and not entete_html_fini:
            if "<td" in bas:
                entete_html_fini = True
            else:
                tableau[1].append(ligne)
        if ligne.strip().startswith("|") and not (tableau and tableau[0] == "html"):
            if tableau is None:
                tableau = ("md", [ligne])
            elif len(tableau[1]) < 2:
                tableau[1].append(ligne)
        elif tableau and tableau[0] == "md":
            tableau = None
        courant.append(ligne)
        taille += len(ligne) + 1
        if tableau and tableau[0] == "html" and "</table>" in bas:
            tableau = None
        if taille <= limite:
            continue
        if tableau is None:
            point_sur = not ligne.strip()
        elif tableau[0] == "md":
            point_sur = len(tableau[1]) == 2 and ligne.strip().startswith("|")
        else:
            point_sur = entete_html_fini and "</tr>" in bas
        if point_sur:
            dans_tableau = tableau is not None
            emettre(n, dans_tableau)
            suite = True
            if dans_tableau:
                courant = list(tableau[1])
                taille = sum(len(l) + 1 for l in courant)
    emettre(len(lignes), False)
    return sections


def _entete_tableau(lignes: Sequence[str], i: int) -> List[str]:
    """Les lignes d'en-tête du tableau qui porte la ligne ``i`` (markdown ou HTML), ou rien si la
    ligne n'est pas dans un tableau ou est elle-même l'en-tête. Une section recoupée dans un
    tableau commence par son en-tête (``decouper_sections``) : il est toujours là."""
    ligne = lignes[i].strip()
    if ligne.startswith("|"):
        debut = i
        while debut > 0 and lignes[debut - 1].strip().startswith("|"):
            debut -= 1
        return [l.strip() for l in lignes[debut:debut + 2]] if debut < i else []
    if "<td" in ligne.lower():
        debut = i
        while debut > 0 and "<table" not in lignes[debut].lower():
            debut -= 1
        if "<table" not in lignes[debut].lower():
            return []
        entete: List[str] = []
        for l in lignes[debut:i]:
            if "<td" in l.lower():
                break
            entete.append(l.strip())
        return entete
    return []


def titre_section(section: Dict[str, Any], profondeur: int = 3) -> str:
    """Le chemin de titres, sans les marques de mise en forme, raccourci aux derniers niveaux."""
    titres = [re.sub(r"[*`]", "", t).strip() for t in section["titres"]][-profondeur:]
    return " > ".join(titres) or "(ouverture)"


@dataclass
class Livraison:
    """Ce qu'un tour a déjà livré au modèle : rien n'est envoyé deux fois.

    Sérialisable (``vers_dict`` / ``depuis_dict``) pour le banc qui rejoue les outils appel par
    appel (``app/scripts/naviguer_wiki.py``).
    """

    sections: Set[str] = field(default_factory=set)  # "chemin#numero"
    pages: Set[str] = field(default_factory=set)  # pages livrées entières
    caracteres: int = 0

    def vers_dict(self) -> Dict[str, Any]:
        return {"sections": sorted(self.sections), "pages": sorted(self.pages), "caracteres": self.caracteres}

    @classmethod
    def depuis_dict(cls, data: Optional[Dict[str, Any]]) -> "Livraison":
        data = data or {}
        return cls(set(data.get("sections") or ()), set(data.get("pages") or ()), int(data.get("caracteres") or 0))


def _milliers(n: int) -> str:
    return f"{n:,}".replace(",", " ")


class WikiIndex:
    """Index lexical sur les pages concept du wiki, plus les registres d'anomalies."""

    def __init__(self, pages: Iterable[Any]):
        self.entries: List[Dict[str, Any]] = []
        self.df: Counter = Counter()
        self.longueur_moyenne: float = 1.0
        self.anomalies: Dict[str, Dict[str, Any]] = {}
        # Les jetons que le wiki écrit tels quels. Un nom collé par la recherche (« vitrage 24 »
        # → vitrage24) n'a le poids d'une référence que si une page l'écrit ainsi (LUMINE65) :
        # sinon les pages pleines de « vitrage 24 » passaient devant la bonne (30/09/2026).
        self.natifs: Set[str] = set()
        retenues: List[Any] = []

        for page in pages:
            if page.reserved or page.missing:
                continue
            retenues.append(page)
            sac = _sac_de_tokens(page)
            self.natifs.update(mots(" ".join([page.title or "", page.description or "", page.body or ""])))
            self.entries.append({
                "chemin": page.id,
                "type": page.type or "-",
                "titre": page.title,
                "description": page.description or "-",
                "tags": list(page.tags),
                "gamme": list(page.gamme),
                "systeme": list(page.systeme),
                "famille": list(page.famille),
                "statut": page.status or "-",
                "corps": page.body,
                "tf": Counter(sac),
                "longueur": len(sac),
            })
            if page.id in ANOMALY_PAGES:
                self._charger_anomalies(page)

        # Les noms de gamme, en lettres seules : « PERFORM », « LUMINE » (voir ``_reference``).
        self.noms_gamme: Set[str] = {
            re.sub(r"[^a-z]", "", deaccent(str(g).lower()))
            for e in self.entries for g in e["gamme"]
        } - {""}

        self.entries.sort(key=lambda e: e["chemin"])
        for entry in self.entries:
            self.df.update(entry["tf"].keys())
        if self.entries:
            self.longueur_moyenne = sum(e["longueur"] for e in self.entries) / len(self.entries)
        self.par_chemin: Dict[str, Dict[str, Any]] = {e["chemin"]: e for e in self.entries}
        self._indexer_sections(retenues)

    def _indexer_sections(self, pages: Sequence[Any]) -> None:
        """Les sections de chaque page et leur index inversé (BM25 par section).

        Le sac d'une section : son texte, son chemin de titres (×4) et le titre de sa page (×2) —
        le titre porte le sens que le texte d'un tableau n'a pas. Les registres d'anomalies ne
        sont pas découpés : ils ne sont jamais livrés.
        """
        self.sections: List[Dict[str, Any]] = []
        self.sections_par_page: Dict[str, List[int]] = defaultdict(list)
        self._inverse: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._s_longueur: List[int] = []
        self._s_df: Counter = Counter()
        for page in sorted(pages, key=lambda p: p.id):
            if page.id in ANOMALY_PAGES:
                continue
            titre_page = tokenise(page.title) * W_TITRE_PAGE
            for section in decouper_sections(page.id, page.raw_text):
                sid = len(self.sections)
                section["id"] = sid
                self.sections.append(section)
                self.sections_par_page[page.id].append(sid)
                sac = tokenise(section["texte"]) + tokenise(" ".join(section["titres"])) * W_TITRES + titre_page
                tf = Counter(sac)
                self._s_longueur.append(len(sac))
                for token, n in tf.items():
                    self._inverse[token].append((sid, n))
                self._s_df.update(tf.keys())
        self._s_moyenne = (sum(self._s_longueur) / len(self._s_longueur)) if self._s_longueur else 1.0

    # ---- construction --------------------------------------------------

    def _charger_anomalies(self, page: Any) -> None:
        """Une entrée par ligne de table dont la première cellule est un identifiant.

        La dernière colonne des tableaux d'entrées, ``Pages du wiki``, sert au rapprochement
        (``liens``) et à ``lire_anomalie`` ; l'entrée injectée d'office (``ligne``) en est
        allégée — un cinquième de son poids — et ses mots ne comptent pas dans le rapprochement
        par sujet : les titres des pages liées n'en sont pas le sujet.
        """
        avec_pages = False
        for ligne in page.body.splitlines():
            if not ligne.lstrip().startswith("|"):
                continue
            ligne = ligne.strip()
            cellules = [c.strip() for c in ligne.strip("|").split("|")]
            if cellules and cellules[0] == "ID":
                avec_pages = cellules[-1] == "Pages du wiki"
            if not cellules or not ENTREE_RE.match(cellules[0]):
                continue
            identifiant = cellules[0]
            if identifiant in self.anomalies:
                continue
            allegee = ligne[: ligne.rstrip("|").rfind("|") + 1] if avec_pages else ligne
            self.anomalies[identifiant] = {
                "id": identifiant,
                "sujet": cellules[1] if len(cellules) > 1 else "-",
                "registre": page.id,
                "ligne": allegee,
                "complete": ligne,
                "liens": set(LIEN_RE.findall(ligne)),
                "tokens": set(tokenise(allegee)),
            }

    # ---- recherche -----------------------------------------------------

    def _idf(self, token: str) -> float:
        n = len(self.entries)
        df = self.df.get(token, 0)
        if not df:
            return 0.0
        return log(1 + (n - df + 0.5) / (df + 0.5))

    def _facette_ok(self, entry: Dict[str, Any], champ: str, demande: Any) -> bool:
        voulus = [deaccent(v.lower()) for v in normalise(demande)]
        if not voulus:
            return True
        disponibles = [deaccent(v.lower()) for v in entry[champ]]
        # Comparaison souple dans les deux sens : des gammes saisies tantôt « LUMINE », tantôt
        # « LUMINE65 ».
        return any(v in d or d in v for v in voulus for d in disponibles)

    def _score_bm25(
        self, requete: Sequence[str], entry: Dict[str, Any], cotes: Set[str] = frozenset()
    ) -> float:
        score = 0.0
        for token in requete:
            tf = entry["tf"].get(token, 0)
            if not tf:
                continue
            norme = 1 - B + B * entry["longueur"] / self.longueur_moyenne
            contribution = self._idf(token) * tf * (K1 + 1) / (tf + K1 * norme)
            # Une référence trouvée est un signal bien plus sûr qu'un mot courant ; une cote de
            # la question en a la forme, pas la valeur.
            score += contribution * (3.0 if self._reference(token, cotes) else 1.0)
        return score

    def _reference(self, token: str, cotes: Set[str] = frozenset()) -> bool:
        """Le poids ×3 est pour une référence de pièce (TGY3702, 76507, NT1947), pas pour :

        * une cote de la question (``cotes``) ;
        * un nom que la recherche a collé elle-même (« vitrage 24 » → vitrage24) et qu'aucune
          page n'écrit ainsi ;
        * un nom de gamme suivi de son épaisseur (PERFORM76, LUMINE55) : c'est un produit, que
          la page de gamme, le nuancier et l'argumentaire répètent sans porter la pièce
          demandée — compté triple, il les faisait passer devant les parcloses SOLEAL FY.
        """
        if not est_reference(token) or token in cotes or token not in self.natifs:
            return False
        nom = re.fullmatch(r"([a-z]+)\d+", token)
        return not (nom and nom.group(1) in self.noms_gamme)

    def _compte_facettes(self, entry: Dict[str, Any], facettes: Dict[str, Any]) -> int:
        return sum(1 for champ, demande in facettes.items() if self._facette_ok(entry, champ, demande))

    # ---- recherche par sections -----------------------------------------

    def _s_idf(self, token: str) -> float:
        n, df = len(self.sections), self._s_df.get(token, 0)
        return log(1 + (n - df + 0.5) / (df + 0.5)) if df else 0.0

    def _scores_sections(self, requete: Sequence[str], cotes: Set[str] = frozenset()) -> Dict[int, float]:
        scores: Dict[int, float] = defaultdict(float)
        for token in requete:
            idf = self._s_idf(token)
            if not idf:
                continue
            poids = 3.0 if self._reference(token, cotes) else 1.0
            for sid, tf in self._inverse.get(token, ()):
                norme = 1 - B + B * self._s_longueur[sid] / self._s_moyenne
                scores[sid] += idf * tf * (K1 + 1) / (tf + K1 * norme) * poids
        return scores

    def classer(
        self,
        mots_cles: str,
        gamme: Optional[str] = None,
        systeme: Optional[str] = None,
        cotes: Set[str] = frozenset(),
    ) -> List[Dict[str, Any]]:
        """Les pages classées par leurs sections : ``[{"entree", "score", "sections": [(id, score)…]}]``.

        Score d'une section : BM25 de la section (normalisé par la meilleure) + ``W_PAGE`` × BM25
        de sa page entière (normalisé). Score d'une page : sa meilleure section + ``W_SECONDE``
        × la deuxième. Les facettes (gamme, système) doublent le score sans rien exclure, une
        page ``sources/`` pèse ``POIDS_SOURCE``.
        """
        requete = [t for t in tokenise(mots_cles) if t not in VIDES]
        scores = self._scores_sections(requete, cotes)
        if not scores:
            return []
        meilleur = max(scores.values())
        par_page_bm25 = {}
        for entry in self.entries:
            s = self._score_bm25(requete, entry, cotes)
            if s > 0:
                par_page_bm25[entry["chemin"]] = s
        plafond = max(par_page_bm25.values(), default=1.0) or 1.0
        facettes = {k: v for k, v in (("gamme", gamme), ("systeme", systeme)) if v}
        notes: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for sid, s in scores.items():
            chemin = self.sections[sid]["chemin"]
            notes[chemin].append((sid, s / meilleur + W_PAGE * par_page_bm25.get(chemin, 0.0) / plafond))
        classement = []
        for chemin, liste in notes.items():
            entree = self.par_chemin.get(chemin)
            if entree is None:
                continue
            liste.sort(key=lambda n: -n[1])
            score = liste[0][1] + (W_SECONDE * liste[1][1] if len(liste) > 1 else 0.0)
            score *= 2.0 ** self._compte_facettes(entree, facettes)
            if chemin.startswith("/sources/"):
                score *= POIDS_SOURCE
            classement.append({"entree": entree, "score": score, "sections": liste})
        classement.sort(key=lambda c: (-c["score"], c["entree"]["chemin"]))
        return classement

    def _cle(self, sid: int) -> str:
        s = self.sections[sid]
        return f"{s['chemin']}#{s['numero']}"

    def fiche(self, chemin: str) -> str:
        """Titre, description et phrase d'ouverture : ce qui dit de quoi parle la page."""
        entree = self.par_chemin[chemin]
        ouverture = ""
        for bloc in re.split(r"\n\s*\n", entree["corps"]):
            b = bloc.strip()
            if b and not b.startswith(("#", "|", "<", "!", ">")):
                ouverture = " ".join(b.split())
                break
        if len(ouverture) > 400:
            ouverture = ouverture[:400].rsplit(" ", 1)[0] + " …"
        lignes = [f"Fiche : {entree['titre']} — {entree['description']}"]
        if ouverture:
            lignes.append(f"Ouverture : {ouverture}")
        return "\n".join(lignes)

    def _groupes(self, chemin: str) -> List[List[int]]:
        """Les sections d'une page par titre : une section recoupée tient sur une ligne du sommaire."""
        groupes: List[List[int]] = []
        for sid in self.sections_par_page.get(chemin, []):
            s = self.sections[sid]
            if groupes and s["suite"] and self.sections[groupes[-1][-1]]["titres"] == s["titres"]:
                groupes[-1].append(sid)
            else:
                groupes.append([sid])
        return groupes

    def sommaire(self, chemin: str, livrees: Iterable[int] = (), deja: Set[str] = frozenset(),
                 complet: bool = False) -> str:
        """Le sommaire de la page : ce qui existe, ce qui vient d'être livré, ce qui l'a déjà été.

        C'est lui qui évite de conclure à une absence sur une page lue en partie : le modèle voit
        les sections qu'il n'a pas reçues.
        """
        livrees = set(livrees)
        groupes = self._groupes(chemin)
        retenus = list(range(len(groupes)))
        if not complet and len(groupes) > SOMMAIRE_MAX:
            utiles = [i for i, g in enumerate(groupes) if any(sid in livrees or self._cle(sid) in deja for sid in g)]
            courts = [i for i, g in enumerate(groupes) if len(self.sections[g[0]]["titres"]) <= 2 and i not in utiles]
            retenus = sorted((utiles + courts)[:SOMMAIRE_MAX])
        lignes = []
        for i in retenus:
            g = groupes[i]
            a, b = self.sections[g[0]], self.sections[g[-1]]
            numero = f"§{a['numero']}" if len(g) == 1 else f"§{a['numero']}–§{b['numero']}"
            if any(sid in livrees for sid in g):
                marque = " ← livrée"
            elif any(self._cle(sid) in deja for sid in g):
                marque = " (déjà livrée)"
            else:
                marque = ""
            lignes.append(f"  {numero} {titre_section(a)} (l. {a['debut']}–{b['fin']}){marque}")
        if len(retenus) < len(groupes):
            lignes.append(
                f"  … {len(groupes) - len(retenus)} autres sections : "
                f"lire(lectures=[{{\"chemin\": \"{chemin}\", \"sections\": [\"sommaire\"]}}]) pour le sommaire complet."
            )
        return "\n".join(lignes)

    def _rendu_section(self, sid: int) -> str:
        s = self.sections[sid]
        return f"--- §{s['numero']} — {titre_section(s)} (lignes {s['debut']}–{s['fin']}) ---\n{s['texte'].strip()}"

    def _description(self, sids: Iterable[int]) -> List[str]:
        return [f"§{self.sections[sid]['numero']} {titre_section(self.sections[sid], 2)}" for sid in sids]

    def page_entiere(self, chemin: str, livraison: Livraison, entete: str = "PAGE") -> Tuple[str, Dict[str, Any]]:
        """Une page livrée entière (la page dominante) : ``(texte, {"chemin", "mode", "sections"})``.

        Le serveur ne la livre que si elle tient (``page_dominante``) et ne l'a pas déjà livrée.
        """
        corps = self.par_chemin[chemin]["corps"].strip()
        livraison.pages.add(chemin)
        livraison.caracteres += len(corps)
        entete_page = f"===== {entete} : {chemin} — page entière ====="
        return f"{entete_page}\n{corps}", {"chemin": chemin, "mode": "page", "sections": []}

    def lire(self, page: Any, section: Any, livraison: Livraison, reste: int) -> Tuple[str, Dict[str, Any]]:
        """``lire`` : la page complète (``section`` vide), son sommaire, ou une section.

        ``section`` : « §13 », « 13 », un morceau de titre, ou « sommaire ». Dans la page complète,
        ce qui a déjà été livré est remplacé par un renvoi ; ``reste`` est ce que le tour peut
        encore lire.
        """
        chemin = page.id
        demande = str(section or "").strip()
        sommaire_complet = lambda: self.sommaire(chemin, deja=livraison.sections, complet=True)  # noqa: E731
        if not demande:
            if chemin in livraison.pages:
                return (f"La page {chemin} a déjà été livrée entière plus haut.",
                        {"chemin": chemin, "mode": "page", "sections": []})
            lignes = page.raw_text.splitlines()
            fournies = [sid for sid in self.sections_par_page.get(chemin, []) if self._cle(sid) in livraison.sections]
            for sid in sorted(fournies, key=lambda s: -self.sections[s]["debut"]):
                s = self.sections[sid]
                lignes[s["debut"] - 1:s["fin"]] = [f"[§{s['numero']} {titre_section(s)} : déjà fourni plus haut]"]
            texte = "\n".join(lignes)
            if len(texte) > reste:
                return (
                    f"La page {chemin} fait {_milliers(len(texte))} caractères, plus que ce qui reste à lire "
                    f"dans ce tour ({_milliers(max(reste, 0))}). Lis la section utile avec "
                    f"lire avec une ou plusieurs sections du sommaire.\n\n{self.fiche(chemin)}\nSommaire :\n{sommaire_complet()}",
                    {"chemin": chemin, "mode": "sommaire", "sections": []},
                )
            livraison.pages.add(chemin)
            livraison.caracteres += len(texte)
            return texte, {"chemin": chemin, "mode": "page", "sections": []}
        if deaccent(demande.lower()) == "sommaire":
            return (f"{self.fiche(chemin)}\nSommaire de {chemin} :\n{sommaire_complet()}",
                    {"chemin": chemin, "mode": "sommaire", "sections": []})
        groupes = self._groupes(chemin)
        numero = re.fullmatch(r"§?\s*(\d+)", demande)
        if numero:
            choisis = [g for g in groupes if any(self.sections[sid]["numero"] == int(numero.group(1)) for sid in g)]
        else:
            cle = deaccent(demande.lower())
            choisis = [g for g in groupes if cle in deaccent(" > ".join(self.sections[g[0]]["titres"]).lower())][:2]
        if not choisis:
            return (f"Aucune section « {demande} » dans {chemin}. Sommaire :\n{sommaire_complet()}",
                    {"chemin": chemin, "mode": "sommaire", "sections": []})
        rendus: List[str] = []
        nouvelles: List[int] = []
        taille = 0
        for sid in (sid for g in choisis for sid in g):
            s = self.sections[sid]
            if self._cle(sid) in livraison.sections:
                rendus.append(f"[§{s['numero']} {titre_section(s)} : déjà fourni plus haut]")
                continue
            if taille + len(s["texte"]) > reste and nouvelles:
                rendus.append(f"[§{s['numero']} non livrée : budget de lecture du tour atteint]")
                continue
            rendus.append(self._rendu_section(sid))
            nouvelles.append(sid)
            taille += len(s["texte"])
        livraison.sections.update(self._cle(sid) for sid in nouvelles)
        livraison.caracteres += taille
        # Le sommaire accompagne toute lecture partielle : il montre au modèle ce qu'il n'a pas reçu
        # — la légende d'un tableau, la règle qui dit comment l'employer — et l'empêche de conclure
        # d'une page lue en partie (consigne 0). Sa suppression avec l'ancienne livraison a fait lire
        # à GLM les tableaux de déductions sans leur mode d'emploi (02/10, Q26 : « Oui » au lieu de « Non »).
        texte = (
            f"===== {chemin} — section(s) demandée(s) =====\n{self.fiche(chemin)}\n"
            "Sommaire de la page (« ← livrée » : lue dans ce résultat ; sans marque : pas encore lue) :\n"
            f"{self.sommaire(chemin, livrees=nouvelles, deja=livraison.sections)}\n\n"
            + "\n\n".join(rendus)
        )
        return texte, {"chemin": chemin, "mode": "sections", "sections": self._description(nouvelles)}

    # ---- la carte : ce que lit un modèle qui navigue ---------------------

    def jetons_requete(self, requetes: Iterable[str]) -> List[str]:
        """Les mots d'une ou de plusieurs formulations, sans doublon ni mot vide."""
        vus: List[str] = []
        for requete in requetes:
            for jeton in tokenise(requete):
                if jeton not in VIDES and jeton not in vus:
                    vus.append(jeton)
        return vus

    def classer_multi(
        self,
        requetes: Sequence[str],
        gamme: Optional[str] = None,
        systeme: Optional[str] = None,
        cotes: Set[str] = frozenset(),
    ) -> List[Dict[str, Any]]:
        """Plusieurs formulations, un seul classement : ``[{"entree", "score", "relatif", "sections"}]``.

        Chaque formulation est classée par ``classer`` ; les pages sont fusionnées par rang
        réciproque (``RRF_K``, ``RRF_PROFONDEUR``), leurs sections réunies ; **l'ordre des
        formulations est une priorité** : la question de l'utilisateur en premier, les
        reformulations ensuite. ``relatif`` est le
        meilleur score de la page rapporté à la première de la formulation qui la place le mieux :
        c'est lui qui dit si une page domine (``page_dominante``). Avec une seule formulation,
        l'ordre est celui de ``classer``.
        """
        pages: Dict[str, Dict[str, Any]] = {}
        for requete in requetes:
            classement = self.classer(requete, gamme=gamme, systeme=systeme, cotes=cotes)
            if not classement:
                continue
            haut = classement[0]["score"] or 1.0
            for rang, item in enumerate(classement[:RRF_PROFONDEUR]):
                chemin = item["entree"]["chemin"]
                page = pages.setdefault(
                    chemin, {"entree": item["entree"], "score": 0.0, "relatif": 0.0, "sections": {}}
                )
                page["score"] += 1.0 / (RRF_K + rang)
                page["relatif"] = max(page["relatif"], item["score"] / haut)
                # Le score d'une section est celui de la première formulation qui la classe : l'ordre
                # des formulations est une priorité (les mots de l'utilisateur d'abord). Mesuré le
                # 02/10 sur le banc, le maximum perdait 4 questions de plus sur 56 (« toutes les
                # pages de preuve » 62,5 % contre 69,6 %) : une reformulation en mots-clés fait
                # monter des sections qui ressemblent à la question sans la porter.
                for sid, note in item["sections"]:
                    page["sections"].setdefault(sid, note)
        fusion = []
        for page in pages.values():
            page["sections"] = sorted(page["sections"].items(), key=lambda s: (-s[1], s[0]))
            fusion.append(page)
        fusion.sort(key=lambda p: (-p["score"], p["entree"]["chemin"]))
        return fusion

    def page_dominante(self, classement: Sequence[Dict[str, Any]], livraison: Livraison) -> Optional[Dict[str, Any]]:
        """La première page du classement, si la suivante ne vaut pas la moitié, qu'elle tient
        entière (``PAGE_ENTIERE_MAX``) et qu'elle n'a pas déjà été livrée. Sinon rien : la carte
        suffit, et le modèle choisit."""
        if not classement:
            return None
        premiere = classement[0]
        if len(classement) > 1 and classement[1]["relatif"] >= DOMINANCE:
            return None
        chemin = premiere["entree"]["chemin"]
        if chemin in livraison.pages or len(premiere["entree"]["corps"].strip()) > PAGE_ENTIERE_MAX:
            return None
        return premiere

    def lignes_qui_repondent(self, sid: int, requete: Sequence[str], n: int = CARTE_LIGNES) -> List[str]:
        """Les lignes d'une section qui portent les mots cherchés, avec l'en-tête de leur tableau.

        Une ligne de tableau ne se lit qu'avec son en-tête (``| Référence | Cote |``) : sans lui,
        « 76507 | 44 | 14 » ne dit rien. Sont gardées les ``n`` meilleures lignes, pourvu qu'elles
        valent 60 % de la meilleure ; l'en-tête est donné une fois par tableau.
        """
        lignes = self.sections[sid]["texte"].splitlines()
        cherches = set(requete)
        notees: List[Tuple[float, int]] = []
        for i, ligne in enumerate(lignes):
            brut = ligne.strip()
            if not brut or TITRE_RE.match(ligne) or set(brut) <= set("|-: "):
                continue
            jetons = set(tokenise(brut))
            note = sum(self._s_idf(t) for t in cherches if t in jetons)
            if note > 0:
                notees.append((note, i))
        if not notees:
            return []
        meilleure = max(note for note, _ in notees)
        gardees = sorted((x for x in notees if x[0] >= 0.6 * meilleure), key=lambda x: (-x[0], x[1]))[:n]
        rendues: List[str] = []
        vues: Set[str] = set()
        for _, i in sorted(gardees, key=lambda x: x[1]):
            for ligne in _entete_tableau(lignes, i) + [lignes[i].strip()]:
                if ligne not in vues:
                    rendues.append(ligne[:700])
                    vues.add(ligne)
        return rendues

    def carte(
        self,
        classement: Sequence[Dict[str, Any]],
        requete: Sequence[str],
        livraison: Livraison,
        pages: int = CARTE_PAGES,
        budget: int = CARTE_BUDGET,
    ) -> Tuple[str, List[str]]:
        """La carte d'une recherche : ``(texte, chemins listés)``.

        Une entrée par page — chemin, titre, taille, produit, pertinence relative —, puis ses
        sections qui répondent, et pour les trois meilleures les lignes qui contiennent les mots
        cherchés. Ce qui a déjà été livré est marqué « lue » et n'est pas redonné. Une ligne de
        carte situe une réponse, elle ne la prouve pas : c'est la lecture qui la prouve.
        """
        blocs: List[str] = []
        listees: List[str] = []
        total = 0
        for rang, item in enumerate(classement[:pages], 1):
            entree = item["entree"]
            chemin = entree["chemin"]
            corps = entree["corps"].strip()
            produit = ", ".join(
                [str(g) for g in entree["gamme"]]
                + [f"système {s}" if str(s).isdigit() else str(s) for s in entree["systeme"]]
            )
            entiere = chemin in livraison.pages
            taille = f"{max(1, round(len(corps) / 1000))} k car., {len(self.sections_par_page[chemin])} sections"
            bloc = [
                f"{rang}. {chemin} — {entree['titre']} ({taille}{', ' + produit if produit else ''})"
                f" · pertinence {item.get('relatif', 1.0):.2f}" + (" · page déjà lue entière" if entiere else "")
            ]
            for j, (sid, _) in enumerate(item["sections"][:CARTE_SECTIONS]):
                section = self.sections[sid]
                lue = entiere or self._cle(sid) in livraison.sections
                bloc.append(f"   §{section['numero']} {titre_section(section)}" + (" (lue)" if lue else ""))
                if j < CARTE_SECTIONS_AVEC_LIGNES and not lue:
                    bloc += ["      " + ligne for ligne in self.lignes_qui_repondent(sid, requete)]
            texte = "\n".join(bloc)
            if len(blocs) >= 3 and total + len(texte) > budget:
                break
            blocs.append(texte)
            listees.append(chemin)
            total += len(texte)
        if not blocs:
            return "Aucune page ne correspond. Reformule avec les mots du wiki, ou cherche la référence seule.", []
        entete = f"===== CARTE — {len(blocs)} pages sur {len(classement)} trouvées ====="
        pied = ""
        if len(blocs) < len(classement):
            pied = f"\n… {len(classement) - len(blocs)} autres pages classées plus bas, non affichées."
        return entete + "\n" + "\n".join(blocs) + pied, listees

    def fiche_question(self, question: str) -> Dict[str, Any]:
        """Ce que le serveur sait de la question avant tout appel au modèle, sans modèle.

        * les **références rares** — au plus ``REFERENCE_RARE_MAX`` sections les portent — avec
          leurs emplacements exacts (page, section, ligne et en-tête) ;
        * les références **absentes** de tout le wiki, ou citées seulement dans un registre
          d'anomalies : le modèle n'a pas à chercher six fois ce qui n'existe pas ;
        ``texte`` est ce qu'on ajoute à la question ; il est vide quand il n'y a rien à dire. Il ne
        porte que ce qui change la recherche : les cotes de la question (ce ne sont pas des
        références) et les produits nommés (PERFORM76, LUMINE : la lettre des mots, aucune
        équivalence de système n'est déduite, VER-28) sont calculés et rendus à part, mais ne
        sont pas écrits dans le message du modèle.
        """
        cotes = cotes_de(question)
        rares: List[str] = []
        absentes: List[str] = []
        registres: List[str] = []
        for ref in references_de(question):
            en_sections = self._s_df.get(ref, 0)
            if not en_sections and not self.df.get(ref):
                absentes.append(ref)
            elif not en_sections:
                registres.append(ref)
            elif en_sections <= REFERENCE_RARE_MAX and self._reference(ref, cotes):
                rares.append(ref)
        produits = []
        for jeton in mots(question):
            nom = re.fullmatch(r"([a-z]+)\d*", jeton)
            if nom and nom.group(1) in self.noms_gamme and jeton.upper() not in produits:
                produits.append(jeton.upper())

        lignes: List[str] = []
        emplacements: Dict[str, List[Dict[str, Any]]] = {}
        for ref in rares:
            sids = sorted(self._inverse.get(ref, ()), key=lambda x: (-x[1], x[0]))
            emplacements[ref] = []
            lignes.append(f"- {ref.upper()} — {len(sids)} section(s) :")
            for sid, _ in sids:
                section = self.sections[sid]
                extraits = self.lignes_qui_repondent(sid, [ref], 2)
                emplacements[ref].append({"chemin": section["chemin"], "section": section["numero"]})
                lignes.append(f"    {section['chemin']} §{section['numero']} {titre_section(section)}")
                lignes += ["        " + e for e in extraits]
        if absentes:
            lignes.append(
                f"- Absent de tout le wiki : {', '.join(r.upper() for r in absentes)}. "
                "Aucune page ne porte cette référence."
            )
        if registres:
            lignes.append(
                f"- Cité seulement dans les registres d'anomalies : {', '.join(r.upper() for r in registres)}."
            )
        texte = ""
        if lignes:
            texte = "===== FICHE DE LA QUESTION (calculée par le serveur, avant ta première recherche) =====\n" + "\n".join(lignes)
        return {
            "texte": texte,
            "rares": rares,
            "absentes": absentes,
            "registres": registres,
            "cotes": sorted(cotes),
            "produits": produits,
            "emplacements": emplacements,
        }

    # ---- prompt permanent ----------------------------------------------

    def vocabulaire(self) -> str:
        """Les mots que le wiki emploie pour nommer les choses : les valeurs des facettes de
        ``chercher`` (gamme, système) et tous les tags, du plus fréquent au moins fréquent. Le tag
        est l'endroit où le mot du métier se range quand une source en emploie un autre."""
        tags = Counter(t for e in self.entries for t in e["tags"])
        gammes = sorted({g for e in self.entries for g in e["gamme"]})
        systemes = sorted({str(s) for e in self.entries for s in e["systeme"]})
        return (
            "GAMMES (valeurs du paramètre gamme) : " + ", ".join(gammes)
            + "\n\nSYSTÈMES (valeurs du paramètre systeme) : " + ", ".join(systemes)
            + f"\n\nTAGS ({len(tags)} tags distincts, du plus fréquent au moins fréquent) : "
            + ", ".join(t for t, _ in tags.most_common())
        )

    def anomalie(self, identifiant: str) -> str:
        entree = self.anomalies.get(str(identifiant).strip().upper())
        if not entree:
            return (
                f"Aucune entrée {identifiant}. N'appelle lire_anomalie que sur un identifiant "
                "écrit dans une page lue ou dans les entrées qui t'ont été fournies."
            )
        return f"{entree['registre']} :\n{entree['complete']}"

    def match_anomalies(
        self, question: str, pages_lues: Sequence[Dict[str, Any]], limite: int = 6
    ) -> List[Dict[str, Any]]:
        """Les entrées qui concernent la réponse en cours, sans que le modèle ait à demander.

        La règle 2 des consignes est non négociable : on ne la laisse pas dépendre de la
        discipline du modèle, le serveur fait le rapprochement.
        """
        q_tokens = {t for t in tokenise(question) if t not in VIDES}
        q_refs = {t for t in q_tokens if est_reference(t)}
        q_sujet = {t for t in q_tokens if len(t) >= 4}

        chemins_lus = {p["chemin"] for p in pages_lues}
        refs_lues: Set[str] = set()
        for page in pages_lues:
            refs_lues |= {t for t in page["tf"] if est_reference(t)}

        classees: List[Tuple[int, Dict[str, Any]]] = []
        for entree in self.anomalies.values():
            score = 0
            if entree["liens"] & chemins_lus:
                score += 10
            score += 6 * len(q_refs & entree["tokens"])
            if refs_lues & entree["tokens"] & q_tokens:
                score += 4
            communs = q_sujet & entree["tokens"]
            if len(communs) >= 2:
                score += 2 * len(communs)
            if score >= 4:
                classees.append((score, entree))

        classees.sort(key=lambda c: c[0], reverse=True)
        return [entree for _, entree in classees[:limite]]
