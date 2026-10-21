"""Une page coupée en plusieurs : la page de tête et ses suites.

Le 21/10/2026 les pages de plus de 20 000 caractères du wiki ont été coupées (protocole
``wiki_llm/CLAUDE.md``, « Writing to be found », règle 1) : la première partie garde l'ancien chemin,
les suivantes ont un chemin voisin, une phrase d'ouverture qui renvoie à la première
(« … est sur [titre](/ancien/chemin.md) ») et, dans leur « Voir aussi », la phrase « Cette page est une
partie d'un sujet qui en compte N ». Les calculateurs lisent leurs tableaux par le titre de section et
l'en-tête des colonnes, jamais par la page : ils lisent donc la tête et ses suites d'un seul tenant.

La famille se reconnaît à ces deux marques, pas au nom des fichiers : ``systeme-70-profiles-complementaires``
et ``systeme-70-profiles-complementaires-manuel`` sont deux pages différentes.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from app.services.wiki_service import WikiPage, WikiSnapshot

PHRASE = "Cette page est une partie d'un sujet qui en compte"
LIEN_RE = re.compile(r"\]\((/[^)\s#]+\.md)")
TITRE_RE = re.compile(r"(?m)^#{1,6} +(.+?)\s*$")

_cache: Dict[int, Dict[str, List[str]]] = {}


def _tete_de(page: WikiPage) -> Optional[str]:
    """La tête d'une suite : le premier lien de sa phrase d'ouverture (avant le premier titre)."""
    if PHRASE not in page.body:
        return None
    avant_titre = TITRE_RE.split(page.body, maxsplit=1)[0]
    m = LIEN_RE.search(avant_titre)
    return m.group(1) if m else None


def familles(snap: WikiSnapshot) -> Dict[str, List[str]]:
    """Chemin de tête → chemins de ses suites, dans l'ordre du « Voir aussi » de la tête."""
    cle = id(snap)
    if cle not in _cache:
        _cache.clear()
        suites: Dict[str, List[str]] = {}
        for p in snap.pages.values():
            if p.is_concept and (tete := _tete_de(p)) and tete in snap.pages and tete != p.id:
                suites.setdefault(tete, []).append(p.id)
        for tete, liste in suites.items():
            ordre = LIEN_RE.findall(snap.pages[tete].body.rsplit("# Voir aussi", 1)[-1])
            liste.sort(key=lambda c: (ordre.index(c) if c in ordre else len(ordre), c))
        _cache[cle] = suites
    return _cache[cle]


def suites(snap: WikiSnapshot, chemin: str) -> List[str]:
    return list(familles(snap).get(chemin, []))


def corps_famille(snap: WikiSnapshot, chemin: str) -> str:
    """Le corps de la page et de ses suites, bout à bout. Une page entière si elle n'a pas été coupée."""
    p = snap.pages[chemin]
    return "\n\n".join([p.body] + [snap.pages[c].body for c in suites(snap, chemin)])


def relocaliser(snap: WikiSnapshot, objet):
    """Toute source ``{page, section}`` d'un résultat désigne la partie qui porte la section, pas la tête."""
    if isinstance(objet, dict):
        if isinstance(objet.get("page"), str) and "section" in objet and objet["page"] in snap.pages:
            objet = dict(objet, page=partie_de_section(snap, objet["page"], objet.get("section") or ""))
        return {k: relocaliser(snap, v) for k, v in objet.items()}
    if isinstance(objet, list):
        return [relocaliser(snap, v) for v in objet]
    return objet


def partie_de_section(snap: WikiSnapshot, chemin: str, section: str) -> str:
    """Le chemin de la partie de la famille qui porte un titre de ce nom (la tête par défaut)."""
    if not section:
        return chemin
    for c in [chemin] + suites(snap, chemin):
        if any(t.strip() == section.strip() for t in TITRE_RE.findall(snap.pages[c].body)):
            return c
    return chemin
