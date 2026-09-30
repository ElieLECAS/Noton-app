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
* ``trier_resultats`` écarte les registres d'``anomalies/`` — ils se lisent entrée par
  entrée, et les entrées utiles sont injectées par le serveur — et fait passer les pages
  concept **avant** les pages ``sources/`` : une source résume un document, une page concept
  porte la valeur technique.

L'index est construit avec l'instantané (``wiki_service.load_snapshot``) et jeté avec lui dès
qu'un fichier change. Il ne relit pas les fichiers : il travaille sur les pages déjà lues.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
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
    if len(jeton) > 4 and jeton[-1] in "sx" and jeton.isalpha():
        return jeton[:-1]
    return jeton


def mots(texte: Any) -> List[str]:
    """Les mots tels qu'ils sont écrits, ramenés à une graphie — sans les noms collés."""
    plat = _recolle(deaccent(str(texte).lower().translate(LIGATURES)))
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

        for page in pages:
            if page.reserved or page.missing:
                continue
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

    def extrait(self, entry: Dict[str, Any], mots_cles: str, largeur: int = 150) -> str:
        """Le passage qui a fait remonter la page, pour décider sans deviner."""
        corps = entry["corps"]
        plat = deaccent(corps.lower())
        requete = [t for t in tokenise(mots_cles) if t not in VIDES]
        requete.sort(key=lambda t: (est_reference(t), self._idf(t)), reverse=True)
        for token in requete:
            position = plat.find(token)
            if position == -1:
                continue
            debut = max(0, position - largeur // 3)
            morceau = corps[debut:debut + largeur]
            return " ".join(morceau.split())
        return " ".join(corps.split())[:largeur]

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
# Mise en forme des résultats
# ---------------------------------------------------------------------------


def trier_resultats(pages: Sequence[Dict[str, Any]], completes: int = 3) -> Tuple[List, List]:
    """Répartit les résultats entre pages livrées entières et reste.

    Trois règles, dans cet ordre :

    * les registres d'``anomalies/`` sont **écartés** : ils se lisent entrée par entrée
      (``lire_anomalie``) et celles qui concernent les pages lues sont injectées par le
      serveur. En livrer un revenait à payer des dizaines de milliers de tokens ;
    * les pages concept passent **avant** les pages ``sources/`` : une page source résume un
      document, une page concept porte la valeur technique ;
    * à catégorie égale, l'ordre du classement est conservé.
    """
    retenues = [p for p in pages if not p["chemin"].startswith("/anomalies/")]
    concepts = [p for p in retenues if not p["chemin"].startswith("/sources/")]
    sources = [p for p in retenues if p["chemin"].startswith("/sources/")]

    entieres = (concepts + sources)[:completes]
    gardees = {p["chemin"] for p in entieres}
    reste = [p for p in retenues if p["chemin"] not in gardees]
    return entieres, reste


def formate_resultats(
    index: WikiIndex, pages: Sequence[Dict[str, Any]], mots_cles: str, completes: int = 3
) -> str:
    """Contenu entier des ``completes`` premières pages, métadonnées pour les suivantes.

    La bonne page ne sort pas toujours en tête, mais elle est presque toujours dans les trois
    premières ; le modèle, lui, n'ouvre qu'une page par question. Livrer d'emblée les
    premières pages entières supprime cet écart, et supprime aussi un aller-retour.
    """
    if not pages:
        return (
            "Aucune page ne correspond. Élargis la requête : retire une facette, ou cherche "
            "la référence seule."
        )

    entieres, reste = trier_resultats(pages, completes)

    blocs = [
        f"===== PAGE {i} : {page['chemin']} =====\n{page['corps'].strip()}"
        for i, page in enumerate(entieres, 1)
    ]

    if reste:
        cellule = lambda s: str(s).replace("|", "/").replace("\n", " ")  # noqa: E731
        lignes = [
            "===== AUTRES RÉSULTATS =====",
            "Métadonnées seules. Appelle lire_page(chemin) si l'une d'elles est nécessaire.",
            "",
            "| Chemin | Type | Titre | Tags | Description | Extrait |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for page in reste:
            lignes.append(
                f"| {page['chemin']} | {cellule(page['type'])} | {cellule(page['titre'])} "
                f"| {cellule(', '.join(page['tags'][:6]))} | {cellule(page['description'])} "
                f"| {cellule(index.extrait(page, mots_cles))} |"
            )
        blocs.append("\n".join(lignes))

    return "\n\n".join(blocs)
