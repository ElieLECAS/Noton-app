"""Les sommaires du wiki, pour le modèle : une racine (N0) et un sommaire par produit (N1).

Le prompt permanent porte ``index.md`` entier : 940 lignes rangées par type, ~122 000 tokens
relus à chaque appel du tour, et les pages d'un même système éparpillées dans six rubriques.
Ici, deux niveaux calculés depuis la frontmatter, jamais écrits dans le wiki :

- la **racine** (N0) : les gammes PROFERM et les systèmes qu'elles emploient, puis les systèmes
  par fournisseur, chacun avec son nombre de pages par type et le chemin de son sommaire ;
- un **sommaire par produit** (N1) : toutes ses pages, une ligne chacune au format d'``index.md``,
  rangées par type puis par famille.

Le produit d'une page : son ``systeme``, sinon sa ``gamme`` (ASKEY ne remplit que la gamme),
sinon « transverse ». Une page de plusieurs systèmes figure dans chacun : un sommaire oriente, il
n'exclut rien. Le fournisseur d'un produit se lit sur ses pages, hors pages de gamme — une page
de gamme qui cite la ferrure Roto NX ne fait pas de Roto NX un produit KÖMMERLING ; un produit
dont la plupart des pages ne nomment aucun fournisseur est une gamme PROFERM.

Recalculés avec les pages, ils ne dérivent jamais d'elles. Ce qui les fausserait — deux produits
sur le même chemin, un sommaire trop gros — va dans ``alertes``.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from app.services.wiki_carte import TYPE_PLURIEL, _cle, sys_label
from app.services.wiki_service import WikiPage

PREFIXE = "/index/"
TRANSVERSE = "transverse"
PROFERM = "PROFERM"
# Les limites d'abord (gamme, DTA), puis les pièces, les gestes, ce qui se vend ; les cartes des
# documents en dernier : elles disent d'où vient une valeur, elles ne la portent pas.
ORDRE_TYPES = (
    "Gamme", "Certification", "Norme", "Profilé", "Quincaillerie", "Vitrage", "Porte d'entrée",
    "Équipement", "Procédure", "Machine", "Entretien", "Coloris", "Commercial", "Garantie", "Tarif",
    "Référence", "Fournisseur", "Anomalie", "Document source",
)
PLURIEL = {**TYPE_PLURIEL, "Norme": "Normes", "Machine": "Machines", "Tarif": "Tarifs"}
# ~30 000 tokens : au-delà, un sommaire coûte le quart de l'index plat qu'il remplace.
SOMMAIRE_MAX = 90_000
# Les types les plus fournis d'un produit, dans sa ligne de la racine.
TYPES_RACINE = 4


@dataclass
class Sommaire:
    produit: str
    chemin: str
    fournisseur: str
    pages: List[WikiPage]
    texte: str = ""

    @property
    def libelle(self) -> str:
        return "Pages sans produit" if self.produit == TRANSVERSE else sys_label(self.produit)

    def extrait(self, demande: str) -> Optional[str]:
        """Les rubriques dont le titre contient ``demande`` (« profilés », « quincaillerie »),
        sous l'en-tête du sommaire ; ``None`` si aucune ne convient. Sans demande, tout."""
        cle = _plat(demande)
        if not cle or cle == "sommaire":
            return self.texte
        tete, *blocs = re.split(r"(?m)^(?=## )", self.texte)
        retenus = [b for b in blocs if cle in _plat(b.splitlines()[0])]
        return (tete + "".join(retenus)) if retenus else None

    def rubriques(self) -> List[str]:
        return re.findall(r"(?m)^## (.+)$", self.texte)


@dataclass
class Sommaires:
    racine: str
    par_chemin: Dict[str, Sommaire]
    alertes: List[str] = field(default_factory=list)

    def du_produit(self, produit: str) -> Optional[Sommaire]:
        return next((s for s in self.par_chemin.values() if s.produit == produit), None)

    def de_page(self, chemin: str) -> List[Sommaire]:
        return [s for s in self.par_chemin.values() if any(p.id == chemin for p in s.pages)]


def _plat(texte: str) -> str:
    return unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode().lower().strip()


def produits_de(page: WikiPage) -> List[str]:
    return list(page.systeme) or list(page.gamme)


def chemin_du_sommaire(produit: str) -> str:
    """« 76 » → /index/systeme-76.md ; le « + » se dit, sinon HYBRIDE+ écrase HYBRIDE."""
    texte = sys_label(produit).replace("+", " plus")
    return PREFIXE + re.sub(r"[^a-z0-9]+", "-", _plat(texte)).strip("-") + ".md"


def _rang_type(t: str) -> tuple:
    return (ORDRE_TYPES.index(t) if t in ORDRE_TYPES else len(ORDRE_TYPES), _cle(t))


def _fournisseur(produit: str, pages: List[WikiPage]) -> str:
    """Celui que nomment la plupart des pages du produit, hors pages de gamme : PERFORM+ déclare
    la ferrure Roto NX, elle ne fait pas de Roto NX un produit KÖMMERLING. Quand la plupart n'en
    nomment aucun, c'est une gamme PROFERM (les portes LUMINE, HYBRIDE…)."""
    comptees = [p for p in pages if p.type != "Gamme"] or pages
    compte = Counter(p.fournisseur[0] if p.fournisseur else "" for p in comptees)
    return compte.most_common(1)[0][0] or PROFERM


def _pages(n: int) -> str:
    return f"{n} page" if n == 1 else f"{n} pages"


def _ligne(p: WikiPage) -> str:
    famille = f" `{p.famille[0]}`" if p.famille else ""
    return f"* [{p.title}]({p.id}){famille} - {' '.join(p.description.split())}"


def _repartition(pages: List[WikiPage], n: Optional[int] = None) -> str:
    compte = Counter(p.type for p in pages).most_common(n)
    return ", ".join(f"{PLURIEL.get(t, t).lower()} {k}" for t, k in compte)


def _texte_sommaire(s: Sommaire, tous: Dict[str, Sommaire]) -> str:
    lignes = [f"# Sommaire : {s.libelle}", ""]
    tete = _pages(len(s.pages))
    if s.produit == TRANSVERSE:
        fournisseurs = Counter(p.fournisseur[0] if p.fournisseur else "sans fournisseur" for p in s.pages)
        tete += " · " + ", ".join(f"{f} {k}" for f, k in sorted(fournisseurs.items(), key=lambda fk: _cle(fk[0])))
    else:
        tete += f" · fournisseur {s.fournisseur}"
        gammes = sorted({g for p in s.pages for g in p.gamme if g != s.produit}, key=_cle)
        if gammes:
            tete += " · gammes PROFERM : " + ", ".join(gammes)
    lignes.append(tete)
    # Les produits dont ce sommaire partage des pages : la question qui touche les deux y passe.
    voisins = Counter(v for p in s.pages for v in produits_de(p) if v != s.produit)
    if voisins:
        lignes.append("Pages communes avec : " + " ; ".join(
            f"{tous[chemin_du_sommaire(v)].libelle} ({k}) {chemin_du_sommaire(v)}"
            for v, k in sorted(voisins.items(), key=lambda vk: (-vk[1], _cle(vk[0])))
            if chemin_du_sommaire(v) in tous))
    par_type: Dict[str, List[WikiPage]] = {}
    for p in s.pages:
        par_type.setdefault(p.type, []).append(p)
    for t in sorted(par_type, key=_rang_type):
        lignes += ["", f"## {PLURIEL.get(t, t)} ({len(par_type[t])})", ""]
        # Par famille, puis par titre : les parcloses ensemble, les cotes de débit ensemble.
        rangees = sorted(par_type[t], key=lambda p: (not p.famille, _cle(p.famille[0]) if p.famille else (), _cle(p.title), p.id))
        lignes += [_ligne(p) for p in rangees]
    return "\n".join(lignes) + "\n"


def _pages_de_gamme(pages: List[WikiPage]) -> Dict[str, WikiPage]:
    """La page d'une gamme : celle dont le titre EST la gamme, sinon la première (comme la carte)."""
    out: Dict[str, WikiPage] = {}
    for p in pages:
        if p.type == "Gamme":
            for g in p.gamme:
                actuelle = out.get(g)
                if actuelle is None or (p.title == g and actuelle.title != g):
                    out[g] = p
    return out


def systemes_des_gammes(pages: Iterable[WikiPage]) -> Dict[str, List[str]]:
    """Chaque gamme qui a sa page → les systèmes sur lesquels ses pages sont écrites, déclarés par
    la page de gamme ou portés avec elle par une page (LUMINE : SOLEAL FY, GY, PY et LUMEAL GA,
    quand sa page n'en déclare qu'un). Le plus employé d'abord."""
    pages = [p for p in pages if p.is_concept]
    out: Dict[str, List[str]] = {}
    for g, page in sorted(_pages_de_gamme(pages).items(), key=lambda gp: _cle(gp[0])):
        systemes = Counter(s for p in pages if g in p.gamme for s in p.systeme)
        for s in page.systeme:
            systemes.setdefault(s, 0)
        out[g] = [s for s, _ in sorted(systemes.items(), key=lambda sk: (-sk[1], _cle(sk[0])))]
    return out


def _texte_racine(par_chemin: Dict[str, Sommaire], pages: List[WikiPage]) -> str:
    lignes = [
        "# Carte du wiki",
        "",
        f"{len(pages)} pages, rangées par produit. Chaque produit a son sommaire, une page que `lire` "
        "ouvre : toutes ses pages, une ligne chacune (titre, chemin, ce que la page couvre), rangées par "
        "type. `lire` avec `sections` = un type (« Profilés », « Quincaillerie ») n'en rend que cette "
        "rubrique. Une page commune à plusieurs produits figure dans le sommaire de chacun.",
    ]
    liees: set = set()

    # Les gammes : ce que vend PROFERM, et les systèmes sur lesquels ses pages sont écrites.
    pages_de_gamme = _pages_de_gamme(pages)
    if pages_de_gamme:
        lignes += ["", "## Gammes PROFERM et systèmes employés", ""]
    for g, systemes in systemes_des_gammes(pages).items():
        page = pages_de_gamme[g]
        morceaux = [f"[{page.title}]({page.id})"]
        propre = par_chemin.get(chemin_du_sommaire(g))
        if propre is not None and propre.produit == g:
            morceaux.append(f"{propre.fournisseur}, {_pages(len(propre.pages))} propres à la gamme : {propre.chemin}")
            liees.add(propre.chemin)
        sommaires_sys = [par_chemin[c] for c in map(chemin_du_sommaire, systemes) if c in par_chemin]
        if sommaires_sys:
            morceaux.append("systèmes : " + ", ".join(
                f"{x.libelle} ({_pages(len(x.pages))}) {x.chemin}" for x in sommaires_sys))
        lignes.append(f"- **{g}** — " + " · ".join(morceaux))

    # Les systèmes, par fournisseur : le plus fourni d'abord.
    restants = [s for c, s in par_chemin.items() if c not in liees and s.produit != TRANSVERSE]
    par_fournisseur: Dict[str, List[Sommaire]] = {}
    for s in restants:
        par_fournisseur.setdefault(s.fournisseur, []).append(s)
    if par_fournisseur:
        lignes += ["", "## Systèmes par fournisseur"]
    for f in sorted(par_fournisseur, key=lambda f: (-sum(len(s.pages) for s in par_fournisseur[f]), _cle(f))):
        lignes += ["", f"### {f}", ""]
        for s in sorted(par_fournisseur[f], key=lambda s: (-len(s.pages), _cle(s.produit))):
            lignes.append(f"- **{s.libelle}** — {_pages(len(s.pages))} ({_repartition(s.pages, TYPES_RACINE)}) : {s.chemin}")

    transverse = par_chemin.get(chemin_du_sommaire(TRANSVERSE))
    if transverse is not None:
        lignes += ["", "## Pages sans produit", "",
                   f"- {_pages(len(transverse.pages))} ({_repartition(transverse.pages)}) : {transverse.chemin}"]
    return "\n".join(lignes) + "\n"


def sommaires(pages: Iterable[WikiPage]) -> Sommaires:
    """N0 et N1 pour les pages données (les pages réservées et les fantômes sont ignorés)."""
    concept = sorted((p for p in pages if p.is_concept), key=lambda p: p.id)
    alertes: List[str] = []

    par_produit: Dict[str, List[WikiPage]] = {}
    for p in concept:
        for produit in produits_de(p) or [TRANSVERSE]:
            par_produit.setdefault(produit, []).append(p)

    par_chemin: Dict[str, Sommaire] = {}
    for produit in sorted(par_produit, key=_cle):
        chemin = chemin_du_sommaire(produit)
        if chemin in par_chemin:
            # Une collision perdrait un produit entier : on la signale plutôt que d'écraser.
            alertes.append(f"{chemin} : « {par_chemin[chemin].produit} » et « {produit} » sur le même chemin")
            chemin = chemin[:-3] + f"-{sum(c.startswith(chemin[:-3]) for c in par_chemin) + 1}.md"
        liste = par_produit[produit]
        fournisseur = _fournisseur(produit, liste) if produit != TRANSVERSE else ""
        par_chemin[chemin] = Sommaire(produit, chemin, fournisseur, liste)

    for s in par_chemin.values():
        s.texte = _texte_sommaire(s, par_chemin)
        if len(s.texte) > SOMMAIRE_MAX:
            alertes.append(f"{s.chemin} : {len(s.texte):,} car., au-delà de {SOMMAIRE_MAX:,}".replace(",", " "))

    return Sommaires(_texte_racine(par_chemin, concept), par_chemin, alertes)
