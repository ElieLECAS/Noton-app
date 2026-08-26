"""Thésaurus métier pour l'expansion des requêtes BM25 (menuiserie).

POURQUOI. Depuis le retrait de la voie dense texte (2026-08-25), BM25 est le SEUL canal
texte du retriever. Le dictionnaire ``french`` de Postgres gère la morphologie
(ouvrant / ouvrants) mais pas la synonymie : l'utilisateur écrit « poignée » là où la
notice dit « béquille », et plus rien ne rattrape l'écart.

POURQUOI UN DICTIONNAIRE ET PAS UN LLM. Le domaine est FERMÉ — un vocabulaire fini de
gammes, profils et ferrures. Sur un domaine fermé un thésaurus curé bat une expansion
LLM : déterministe, auditable, 0 ms, 0 €, et on peut expliquer pourquoi une page est
sortie. L'expansion LLM se justifie sur les domaines ouverts.

OÙ IL S'APPLIQUE. UNIQUEMENT sur les paliers de repli OR de ``retrieve_bm25_pages``,
jamais sur le AND strict : si la requête stricte matche, elle est fiable et on n'y touche
pas. Le reranker étant éteint, rien en aval ne rattrape un pool pollué — l'expansion doit
donc rester cantonnée au cas où la requête stricte a déjà rendu zéro.

CURATION. Garder ce fichier PETIT et SÛR : un mauvais synonyme pollue toutes les requêtes
qui touchent ce terme. Mieux vaut 30 groupes justes que 200 approximatifs. Les flexions
courantes (pluriels, variantes accentuées) vont directement dans les groupes — le
stemming ``french`` s'applique de toute façon après, à la construction du tsquery.
"""
from __future__ import annotations

import unicodedata
from typing import Dict, FrozenSet, List, Set

# Chaque groupe = des termes que le métier considère comme équivalents pour la RECHERCHE.
# N'ajouter un terme que si l'on accepte qu'une requête sur n'importe quel membre du
# groupe ramène des pages parlant des autres.
_SYNONYM_GROUPS: List[Set[str]] = [
    # --- Menuiserie : parties de l'ouvrage ---
    {"ouvrant", "ouvrants", "vantail", "vantaux", "battant", "battants"},
    {"dormant", "dormants", "cadre", "cadres", "bati", "bâti", "bâtis"},
    {"traverse", "traverses"},
    {"montant", "montants"},
    {"seuil", "seuils", "rejingot", "rejingots"},
    {"vitrage", "vitrages", "verre", "verres", "double vitrage"},
    {"parclose", "parcloses"},
    {"chassis", "châssis", "menuiserie", "menuiseries"},
    # --- Quincaillerie / ferrures ---
    {"ferrure", "ferrures", "quincaillerie", "quincailleries"},
    {"beguille", "béquille", "beguilles", "béquilles", "poignee", "poignée", "poignees", "poignées"},
    {"paumelle", "paumelles", "charniere", "charnière", "charnieres", "charnières", "fiche", "fiches"},
    {"gache", "gâche", "gaches", "gâches"},
    {"cremone", "crémone", "cremones", "crémones", "espagnolette", "espagnolettes"},
    {"galet", "galets", "rouleau", "rouleaux"},
    {"compas", "compas d'aeration", "compas d'aération"},
    {"serrure", "serrures", "verrou", "verrous"},
    # --- Étanchéité / réglage ---
    {"joint", "joints", "etancheite", "étanchéité", "etancheites", "étanchéités"},
    {"compression", "serrage", "pression"},
    {"reglage", "réglage", "reglages", "réglages", "ajustement", "ajustements"},
    {"calage", "cales", "cale", "calages"},
    # --- Modes d'ouverture (sigles métier omniprésents dans les notices) ---
    {"of", "ouverture a la francaise", "ouverture à la française", "francaise", "française"},
    {"ob", "oscillo-battant", "oscillo battant", "oscillobattant"},
    {"coulissant", "coulissants", "coulissante", "galandage"},
    {"soufflet", "abattant"},
    # --- Opérations ---
    {"pose", "poses", "montage", "montages", "installation", "installations", "fixation", "fixations"},
    {"depose", "dépose", "demontage", "démontage"},
    {"percage", "perçage", "percages", "perçages"},
    {"entretien", "maintenance", "nettoyage"},
    # --- Matériaux ---
    {"pvc", "polychlorure de vinyle"},
    {"alu", "aluminium"},
    # --- Volets / stores ---
    {"volet roulant", "volets roulants", "vr"},
    {"tablier", "tabliers"},
    {"manivelle", "manivelles", "treuil", "treuils"},
    {"lame", "lames"},
]


# Articles élidés à retirer avant correspondance. En français métier l'élision est la
# règle plus que l'exception (« réglage de l'ouvrant », « pose d'un dormant ») : sans
# ça, le thésaurus rate la majorité des occurrences réelles de son propre vocabulaire.
_ELISIONS = ("l'", "d'", "qu'", "n'", "s'", "j'", "m'", "t'", "c'")


def _normalize_token(token: str) -> str:
    """Clé de correspondance : minuscules, sans accent, sans article élidé, NFKC.

    Le repli d'accents est indispensable : les notices écrivent « béquille » et les
    utilisateurs tapent souvent « bequille ».
    """
    raw = (token or "").strip().lower().replace("’", "'")
    for elision in _ELISIONS:
        if raw.startswith(elision):
            raw = raw[len(elision):]
            break
    folded = unicodedata.normalize("NFKD", raw)
    return "".join(c for c in folded if not unicodedata.combining(c))


def _build_index() -> Dict[str, FrozenSet[str]]:
    """token normalisé -> ensemble des termes du groupe (forme d'origine)."""
    index: Dict[str, Set[str]] = {}
    for group in _SYNONYM_GROUPS:
        for member in group:
            key = _normalize_token(member)
            if not key:
                continue
            index.setdefault(key, set()).update(group)
    return {k: frozenset(v) for k, v in index.items()}


_INDEX: Dict[str, FrozenSet[str]] = _build_index()


def expand_term(term: str) -> List[str]:
    """Synonymes d'un terme, terme d'origine EXCLU (déjà présent dans la requête).

    Retourne une liste vide si le terme n'est pas dans le thésaurus — cas très majoritaire,
    notamment pour les références produit.
    """
    key = _normalize_token(term)
    if not key:
        return []
    group = _INDEX.get(key)
    if not group:
        return []
    return sorted(m for m in group if _normalize_token(m) != key)


def is_discriminant(term: str) -> bool:
    """Un token discriminant (référence produit, marque) ne s'étend jamais.

    « TGY3702 » n'a pas de synonyme : le diluer dans un OR ramènerait des pages sans
    rapport et ferait perdre au canal lexical sa seule vraie force — la correspondance
    exacte sur les codes.
    """
    return any(c.isdigit() for c in (term or ""))


def expand_terms(terms: List[str], *, max_total: int = 20) -> List[str]:
    """Étend une liste de termes, dédupliquée et PLAFONNÉE.

    Le plafond porte sur le TOTAL après expansion, pas sur les termes d'entrée : six
    termes qui s'étendent chacun à quatre synonymes font un OR de vingt-quatre lexèmes,
    où la précision s'effondre.

    Ordre de priorité, pour que la troncature coupe au bon endroit :
      1. tous les termes d'origine (jamais sacrifiés) ;
      2. puis les synonymes, dans l'ordre des termes qui les ont produits.
    """
    out: List[str] = []
    seen: Set[str] = set()

    def _add(value: str) -> bool:
        key = _normalize_token(value)
        if not key or key in seen:
            return False
        if len(out) >= max_total:
            return False
        seen.add(key)
        out.append(value)
        return True

    for term in terms:
        _add(term)

    for term in terms:
        if is_discriminant(term):
            continue
        for synonym in expand_term(term):
            if len(out) >= max_total:
                return out
            _add(synonym)

    return out
