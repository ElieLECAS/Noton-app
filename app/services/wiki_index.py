"""L'index de navigation du wiki : recherche lexicale et registres d'anomalies.

Le wiki ne tient plus dans une fenêtre de contexte (297 pages). Le prompt permanent ne
porte donc qu'un **vocabulaire** (types, tags, gammes, systèmes) ; les pages se trouvent par
l'outil ``chercher``, qui interroge cet index, et les entrées d'anomalie rapprochées des pages
lues sont poussées par le serveur.

La recherche est **lexicale** (BM25 sur texte intégral + facettes de métadonnées), pas
vectorielle : deux pages qui se contredisent doivent toutes les deux remonter, alors qu'un
top-k sémantique les met en concurrence. Trois choix méritent d'être rappelés :

* les **facettes remontent une page, elles ne l'excluent pas**. Une facette mal choisie
  cachait la bonne page : « garantie structure LUMINE65 » filtré par ``tags=coulissant``
  écartait la page des garanties. Le ``type`` ne compte même plus dans le classement : le
  modèle le devine (« Profilé » pour une limite que fixe un DTA ou une page de gamme) et,
  compté double, il reléguait la page qui répond — 68 % des recherches réelles de Mistral
  trouvaient la bonne page dans les trois premières, 85 % sans lui (30/09/2026). Il ne sert
  plus qu'à lister une catégorie sans mot-clé ;
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
* la recherche se fait **par section** et se livre **par page** (01/10/2026). Le wiki compte
  23 pages de plus de 50 000 caractères : trois pages entières en coûtaient jusqu'à 300 000.
  Les pages sont découpées à l'indexation (``decouper_sections``), jamais dans le wiki ; une page
  est classée par ses meilleures sections, livrée **entière** si elle est petite, sinon par sa
  fiche, son sommaire et les sections trouvées. Mesuré hors ligne sur le banc : la preuve
  arrive dans 98 % des questions pour 60 000 caractères, contre 91 % pour 69 000 avec trois
  pages entières ;
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
# Ce que livre une recherche : peu de pages, mais complètes. 60 000 caractères de sections
# suffisent (98 % des preuves livrées sur le banc), et 24 pages d'une ou deux sections chacune
# faisaient conclure au modèle qu'une page ne donnait pas ce qu'elle donnait (Q18, 01/10).
PAGES_PAR_RECHERCHE = 6
SECTIONS_PAR_PAGE = 3
BUDGET_RECHERCHE = 60_000
# Classement d'une page par ses sections : la meilleure, une part de la deuxième, et le score
# de la page entière (l'index historique) pour départager.
W_TITRES = 4
W_TITRE_PAGE = 2
W_SECONDE = 0.5
W_PAGE = 0.3
POIDS_SOURCE = 0.5
SOMMAIRE_MAX = 40

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
      citent les preuves et que ``lire_page`` reconstitue.
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
        if champ == "type":
            disponibles = [deaccent(str(entry["type"]).lower())]
        else:
            disponibles = [deaccent(v.lower()) for v in entry[champ]]
        # Comparaison souple dans les deux sens : des centaines de tags, et des gammes
        # saisies tantôt « LUMINE », tantôt « LUMINE65 ».
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

    def search(
        self,
        mots_cles: str = "",
        type: Optional[str] = None,
        tags: Optional[str] = None,
        gamme: Optional[str] = None,
        systeme: Optional[str] = None,
        statut: Optional[str] = None,
        limite: int = 10,
        cotes: Set[str] = frozenset(),
    ) -> List[Dict[str, Any]]:
        """``cotes`` : les nombres que la question donne comme dimensions (``cotes_de``)."""
        requete = [t for t in tokenise(mots_cles) if t not in VIDES]
        demandees = {
            k: v
            for k, v in (("type", type), ("tags", tags), ("gamme", gamme), ("systeme", systeme))
            if v
        }

        entries = self.entries
        if statut:
            voulu = deaccent(str(statut).lower())
            entries = [e for e in entries if voulu in deaccent(str(e["statut"]).lower())]

        # Sans mots-clés il n'y a rien à classer : la facette redevient un filtre.
        if not requete:
            return [
                e for e in entries if self._compte_facettes(e, demandees) == len(demandees)
            ][:limite]

        # Les facettes remontent une page, elles ne l'excluent pas : les familles jumelles
        # se départagent en les voyant toutes les deux, pas en en masquant une. Le type, lui,
        # ne classe pas (voir l'en-tête du module).
        classantes = {k: v for k, v in demandees.items() if k != "type"}
        classees: List[Tuple[float, Dict[str, Any]]] = []
        for entry in entries:
            score = self._score_bm25(requete, entry, cotes)
            if score > 0:
                classees.append((score * 2.0 ** self._compte_facettes(entry, classantes), entry))
        classees.sort(key=lambda c: c[0], reverse=True)
        return [entry for _, entry in classees[:limite]]

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
        tags: Optional[str] = None,
        cotes: Set[str] = frozenset(),
    ) -> List[Dict[str, Any]]:
        """Les pages classées par leurs sections : ``[{"entree", "score", "sections": [(id, score)…]}]``.

        Score d'une section : BM25 de la section (normalisé par la meilleure) + ``W_PAGE`` × BM25
        de sa page entière (normalisé). Score d'une page : sa meilleure section + ``W_SECONDE``
        × la deuxième. Les facettes doublent le score sans rien exclure, le ``type`` ne compte
        pas, une page ``sources/`` pèse ``POIDS_SOURCE``.
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
        facettes = {k: v for k, v in (("gamme", gamme), ("systeme", systeme), ("tags", tags)) if v}
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
                "lire_page(chemin, section=\"sommaire\") pour le sommaire complet."
            )
        return "\n".join(lignes)

    def _rendu_section(self, sid: int) -> str:
        s = self.sections[sid]
        return f"--- §{s['numero']} — {titre_section(s)} (lignes {s['debut']}–{s['fin']}) ---\n{s['texte'].strip()}"

    def _description(self, sids: Iterable[int]) -> List[str]:
        return [f"§{self.sections[sid]['numero']} {titre_section(self.sections[sid], 2)}" for sid in sids]

    def livrer(
        self,
        classement: Sequence[Dict[str, Any]],
        livraison: Livraison,
        budget: int = BUDGET_RECHERCHE,
        pages_max: int = PAGES_PAR_RECHERCHE,
        sections_par_page: int = SECTIONS_PAR_PAGE,
        seuil_page: int = PAGE_ENTIERE_MAX,
        autres_max: int = 8,
        entete: str = "PAGE",
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Le résultat de ``chercher`` : page entière si elle est petite, sinon fiche, sommaire et
        sections trouvées ; rien de ce que ``livraison`` a déjà vu. Rend le texte et, page par
        page, ce qui a été livré (``{"chemin", "mode": "page" | "sections", "sections"}``).
        """
        blocs: List[str] = []
        livrees: List[Dict[str, Any]] = []
        prises: Set[str] = set()
        total = 0
        for item in classement:
            if len(livrees) >= pages_max:
                break
            entree = item["entree"]
            chemin = entree["chemin"]
            if chemin in livraison.pages:
                continue
            corps = entree["corps"].strip()
            deja_en_partie = any(self._cle(sid) in livraison.sections for sid in self.sections_par_page.get(chemin, []))
            numero = len(livrees) + 1
            # Une page courte qui ne tient plus entière dans le budget est livrée par ses
            # sections : la sauter faisait passer une page moins bien classée devant elle.
            tient = total + len(corps) <= budget or not livrees
            if len(corps) <= seuil_page and not deja_en_partie and tient:
                blocs.append(f"===== {entete} {numero} : {chemin} — page entière =====\n{corps}")
                livraison.pages.add(chemin)
                prises.add(chemin)
                total += len(corps)
                livrees.append({"chemin": chemin, "mode": "page", "sections": []})
                continue
            choix: List[int] = []
            for sid, _ in item["sections"]:
                if self._cle(sid) not in livraison.sections:
                    choix.append(sid)
                if len(choix) >= sections_par_page:
                    break
            # Un tableau coupé : sa suite immédiate vient avec, une fois.
            for sid in list(choix):
                voisin = sid + 1
                if (self.sections[sid]["coupe_tableau"] and voisin < len(self.sections)
                        and self.sections[voisin]["chemin"] == chemin and voisin not in choix
                        and self._cle(voisin) not in livraison.sections):
                    choix.append(voisin)
                    break
            garde: List[int] = []
            taille = 0
            for sid in choix:
                t = len(self.sections[sid]["texte"])
                if total + taille + t > budget and (livrees or garde):
                    continue
                garde.append(sid)
                taille += t
            if not garde:
                continue
            garde.sort(key=lambda sid: self.sections[sid]["numero"])
            tete = (
                f"===== {entete} {numero} : {chemin} — {len(garde)} section(s) sur "
                f"{len(self.sections_par_page[chemin])} (page de {_milliers(len(corps))} car.) =====\n"
                f"{self.fiche(chemin)}\n"
                "Sommaire — lire_page(chemin, section) pour une autre section, lire_page(chemin) pour la page complète :\n"
                f"{self.sommaire(chemin, livrees=garde, deja=livraison.sections)}"
            )
            blocs.append(tete + "\n\n" + "\n\n".join(self._rendu_section(sid) for sid in garde))
            livraison.sections.update(self._cle(sid) for sid in garde)
            prises.add(chemin)
            total += taille + len(tete)
            livrees.append({"chemin": chemin, "mode": "sections", "sections": self._description(garde)})
        autres = [
            item for item in classement
            if item["entree"]["chemin"] not in prises and item["entree"]["chemin"] not in livraison.pages
        ][:autres_max]
        if autres:
            cellule = lambda s: str(s).replace("|", "/").replace("\n", " ")  # noqa: E731
            lignes = [
                "===== AUTRES RÉSULTATS =====",
                "Non livrés : lire_page(chemin) pour la page, lire_page(chemin, section) pour une section.",
                "",
                "| Chemin | Titre | Section la plus proche |",
                "| --- | --- | --- |",
            ]
            for item in autres:
                s = self.sections[item["sections"][0][0]]
                lignes.append(
                    f"| {item['entree']['chemin']} | {cellule(item['entree']['titre'])} "
                    f"| §{s['numero']} {cellule(titre_section(s, 2))} |"
                )
            blocs.append("\n".join(lignes))
        livraison.caracteres += total
        return "\n\n".join(blocs), livrees

    def lire(self, page: Any, section: Any, livraison: Livraison, reste: int) -> Tuple[str, Dict[str, Any]]:
        """``lire_page`` : la page complète (``section`` vide), son sommaire, ou une section.

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
                    f"lire_page(chemin, section).\n\n{self.fiche(chemin)}\nSommaire :\n{sommaire_complet()}",
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
        texte = f"===== {chemin} — section(s) demandée(s) =====\n{self.fiche(chemin)}\n\n" + "\n\n".join(rendus)
        return texte, {"chemin": chemin, "mode": "sections", "sections": self._description(nouvelles)}

    # ---- prompt permanent ----------------------------------------------

    def vocabulaire(self, nb_tags: int = 60) -> str:
        types = Counter(e["type"] for e in self.entries)
        tags = Counter(t for e in self.entries for t in e["tags"])
        gammes = sorted({g for e in self.entries for g in e["gamme"]})
        systemes = sorted({s for e in self.entries for s in e["systeme"]})
        return (
            "TYPES (valeur exacte du champ type) : "
            + ", ".join(f"{t} ({n})" for t, n in types.most_common())
            + f"\n\nTAGS les plus fréquents ({len(tags)} tags distincts au total) : "
            + ", ".join(t for t, _ in tags.most_common(nb_tags))
            + "\n\nGAMMES : " + ", ".join(gammes)
            + "\n\nSYSTÈMES : " + ", ".join(systemes)
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


# ---------------------------------------------------------------------------
# Liste par facettes (recherche sans mot-clé)
# ---------------------------------------------------------------------------


def formate_liste(pages: Sequence[Dict[str, Any]]) -> str:
    """Sans mot-clé il n'y a rien à classer : la facette liste une catégorie, en métadonnées."""
    pages = [p for p in pages if not p["chemin"].startswith("/anomalies/")]
    if not pages:
        return "Aucune page ne correspond. Élargis la requête : retire une facette, ou cherche la référence seule."
    cellule = lambda s: str(s).replace("|", "/").replace("\n", " ")  # noqa: E731
    lignes = [
        "===== PAGES =====",
        "Métadonnées seules : lire_page(chemin) pour lire une page.",
        "",
        "| Chemin | Type | Titre | Description |",
        "| --- | --- | --- | --- |",
    ]
    for page in pages:
        lignes.append(
            f"| {page['chemin']} | {cellule(page['type'])} | {cellule(page['titre'])} | {cellule(page['description'])} |"
        )
    return "\n".join(lignes)
