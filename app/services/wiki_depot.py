"""Le dépôt du wiki — mettre à jour ``wiki/`` et ``raw/`` depuis l'administration.

Le wiki s'écrit HORS de l'application (Claude Code + git, protocole ``wiki_llm/CLAUDE.md``) ;
il arrive sur le serveur par ce dépôt : on glisse le dossier ``wiki_llm/`` dans l'onglet Wiki de
l'administration, le navigateur annonce ce qu'il a, le serveur répond ce qui lui manque, les
fichiers montent un par un. Ni rebuild, ni pull, ni ssh.

Deux dossiers, deux contrats — ils n'ont pas la même nature :

  * ``wiki/`` (les pages et les PNG de ``assets/``, ~3 Mo, versionné) est un **miroir** : ce qui
    est déposé DEVIENT le wiki, une page absente du dépôt est supprimée — sans quoi une page
    retirée de la rédaction continuerait de répondre. Tout monte dans un dossier de préparation
    (``.depot/<id>/wiki``) et la bascule est un couple de renommages, verrou de l'instantané
    tenu : si l'envoi casse en route, le wiki servi n'a pas bougé d'un octet.
  * ``raw/`` (les PDF sources, ~800 Mo, hors git) s'**accumule** : le navigateur annonce chemin
    et taille, le serveur ne réclame que ce qui manque ou a changé de taille, et ne supprime
    jamais. Sans ce différentiel, corriger une page coûterait 800 Mo d'envoi.

Un dépôt qui ne contient aucun fichier de ``wiki/`` ne touche pas au wiki : déposer ``raw/``
seul ajoute des PDF, un point c'est tout. C'est ce qui rend la suppression en miroir sûre.

Le dépôt n'a **aucun état en mémoire** : son manifeste est écrit à côté des fichiers préparés.
Un redémarrage au milieu d'un envoi ne perd rien, et un dépôt abandonné est ramassé au suivant.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator, Dict, List, Tuple

from app.services import wiki_service

logger = logging.getLogger(__name__)

# Les dépôts en cours vivent à côté du wiki (même système de fichiers : la bascule est un
# renommage, jamais une copie). Le dossier est ignoré par git et par la signature de
# l'instantané, qui ne regarde que ``wiki/**/*.md`` et ``raw/*.pdf``.
DEPOTS = ".depot"
RACINES = ("wiki", "raw")
# Ce que le wiki contient réellement : des pages et les images de ``assets/``.
EXT_WIKI = frozenset({".md", ".png", ".jpg", ".jpeg", ".svg", ".webp"})
# ``raw/`` ne sert qu'à une chose : rendre le PDF source d'une page. Le reste (exports, zip,
# json de travail) reste sur le poste du rédacteur.
EXT_RAW = frozenset({".pdf"})
TAILLE_MAX = 256 * 1024 * 1024
MORCEAU = 1024 * 1024
# Un dépôt ouvert et jamais validé est ramassé à l'ouverture du suivant.
DEPOT_PERIME_S = 12 * 3600
# Garde-fou sur l'annonce : le manifeste est lu en une fois, il ne doit pas pouvoir enfler.
ANNONCES_MAX = 50_000
CARACTERE_INTERDIT = re.compile(r"[\x00-\x1f<>:\"|?*]")
ID_DEPOT = re.compile(r"\A[0-9a-f]{32}\Z")


class DepotInconnu(LookupError):
    """Le dépôt n'existe pas (ou a été ramassé)."""


class DepotRefuse(ValueError):
    """Le fichier envoyé n'est pas celui que le dépôt attendait."""


class DepotIncomplet(RuntimeError):
    """Des fichiers annoncés ne sont pas arrivés : on ne bascule pas un wiki à trous."""

    def __init__(self, manquants: List[str]):
        super().__init__(f"{len(manquants)} fichier(s) annoncé(s) jamais reçu(s)")
        self.manquants = manquants


@dataclass(frozen=True)
class Classement:
    """Ce qu'un chemin annoncé par le navigateur devient — ou pourquoi il est écarté."""

    racine: str = ""  # "wiki" | "raw" ; vide = écarté
    relatif: str = ""
    raison: str = ""

    @property
    def cle(self) -> str:
        return f"{self.racine}/{self.relatif}" if self.racine else ""


# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------


def classer(brut: Any) -> Classement:
    """Le chemin relatif d'un fichier déposé → son emplacement dans la racine du wiki.

    Le navigateur envoie ce qu'il voit : ``wiki_llm/wiki/profiles/x.md`` si on a glissé
    ``wiki_llm/``, ``raw/moustiquaires/y.pdf`` si on a glissé ``raw/``. On cherche donc le
    premier segment ``wiki`` ou ``raw`` — ce qui précède est le nom du dossier glissé, ce qui
    suit est le chemin dans le wiki. Tout le reste (``CLAUDE.md``, ``a_faire/``, ``.git/``) est
    écarté avec sa raison.

    Aucun chemin ne peut sortir de sa racine : les segments sont validés un par un, pas résolus
    sur disque.
    """
    chemin = str(brut or "").replace("\\", "/").strip()
    segments = [s for s in chemin.split("/") if s not in ("", ".")]
    if len(segments) < 2:
        return Classement(raison="hors de wiki/ et de raw/")
    if any(s == ".." for s in segments):
        return Classement(raison="remontée de dossier")
    if any(CARACTERE_INTERDIT.search(s) for s in segments):
        return Classement(raison="caractère interdit dans le chemin")
    if any(s != s.strip(" .") for s in segments):
        return Classement(raison="nom de fichier invalide")

    for i, segment in enumerate(segments[:-1]):  # jamais le fichier lui-même
        if segment in RACINES:
            racine, relatif = segment, "/".join(segments[i + 1:])
            break
    else:
        return Classement(raison="hors de wiki/ et de raw/")

    extension = PurePosixPath(relatif).suffix.lower()
    autorisees = EXT_WIKI if racine == "wiki" else EXT_RAW
    if extension not in autorisees:
        return Classement(raison=f"{extension or 'sans extension'} : pas un fichier de {racine}/")
    return Classement(racine=racine, relatif=relatif)


def _lister(dossier: Path, extensions: frozenset) -> Dict[str, int]:
    """Les fichiers retenus d'une arborescence : chemin relatif POSIX → taille."""
    trouves: Dict[str, int] = {}
    if not dossier.is_dir():
        return trouves
    for chemin in dossier.rglob("*"):
        if chemin.is_file() and chemin.suffix.lower() in extensions:
            trouves[chemin.relative_to(dossier).as_posix()] = chemin.stat().st_size
    return trouves


def _empreintes(dossier: Path) -> Dict[str, str]:
    """Chemin relatif → sha256 : le seul moyen honnête de dire « modifiée »."""
    empreintes: Dict[str, str] = {}
    if not dossier.is_dir():
        return empreintes
    for chemin in dossier.rglob("*"):
        if chemin.is_file() and chemin.suffix.lower() in EXT_WIKI:
            empreintes[chemin.relative_to(dossier).as_posix()] = hashlib.sha256(
                chemin.read_bytes()
            ).hexdigest()
    return empreintes


# ---------------------------------------------------------------------------
# Le dépôt sur disque
# ---------------------------------------------------------------------------


def _dossier_depots(root: Path) -> Path:
    return root / DEPOTS


def _ramasser(root: Path) -> None:
    """Jette les dépôts ouverts il y a plus de ``DEPOT_PERIME_S`` et jamais validés."""
    base = _dossier_depots(root)
    if not base.is_dir():
        return
    limite = time.time() - DEPOT_PERIME_S
    for dossier in base.iterdir():
        try:
            if dossier.is_dir() and dossier.stat().st_mtime < limite:
                shutil.rmtree(dossier, ignore_errors=True)
                logger.info("[dépôt] %s périmé — ramassé", dossier.name)
        except OSError:
            continue


def _charger(root: Path, depot_id: str) -> Tuple[Path, Dict[str, Any]]:
    """Le dossier du dépôt et son manifeste — l'identifiant est un hexa de 32, jamais un
    chemin : il ne peut désigner que ``.depot/<id>``."""
    if not ID_DEPOT.match(str(depot_id or "")):
        raise DepotInconnu("identifiant de dépôt invalide")
    dossier = _dossier_depots(root) / depot_id
    manifeste_path = dossier / "manifeste.json"
    if not manifeste_path.is_file():
        raise DepotInconnu("dépôt inconnu ou expiré")
    return dossier, json.loads(manifeste_path.read_text(encoding="utf-8"))


def _cible(root: Path, dossier: Path, classement: Classement) -> Path:
    """Où atterrit un fichier : le wiki se prépare à part, les PDF vont directement à leur
    place (ils ne se suppriment pas, une arrivée partielle ne casse rien)."""
    if classement.racine == "wiki":
        return dossier / "wiki" / classement.relatif
    return root / "raw" / classement.relatif


# ---------------------------------------------------------------------------
# Les trois temps : ouvrir, déposer, valider
# ---------------------------------------------------------------------------


def ouvrir(root: Path, annonces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Le navigateur annonce ce qu'il a ; le serveur répond ce qu'il veut, et ce que ça change.

    Retourne le plan AVANT tout envoi : les pages ajoutées, celles qui disparaîtront, les PDF
    manquants, et ce qui a été écarté. C'est ce que l'administration affiche pour faire
    confirmer — un miroir qui supprime doit se voir avant, pas après.
    """
    if len(annonces) > ANNONCES_MAX:
        raise DepotRefuse(f"dépôt trop large : {len(annonces)} fichiers annoncés")
    _ramasser(root)

    retenus: Dict[str, int] = {}
    # La clé normalisée → le chemin tel que le navigateur l'a annoncé : c'est CE chemin qu'on
    # lui réclame, pour qu'il retrouve son fichier sans avoir à rejouer le classement.
    origines: Dict[str, str] = {}
    ignores: List[Dict[str, str]] = []
    for annonce in annonces:
        brut = str(annonce.get("chemin") or "")
        classement = classer(brut)
        taille = max(0, int(annonce.get("taille") or 0))
        if not classement.racine:
            ignores.append({"chemin": brut, "raison": classement.raison})
        elif taille > TAILLE_MAX:
            ignores.append({"chemin": brut, "raison": f"plus lourd que {TAILLE_MAX // 2**20} Mo"})
        else:
            retenus[classement.cle] = taille
            origines[classement.cle] = brut

    pages = {c[5:]: t for c, t in retenus.items() if c.startswith("wiki/")}
    pdfs = {c[4:]: t for c, t in retenus.items() if c.startswith("raw/")}
    # Un dépôt sans aucune page ne touche pas au wiki : on n'efface pas 198 pages parce que
    # quelqu'un a glissé le dossier des PDF.
    miroir = bool(pages)

    en_ligne_pages = _lister(root / "wiki", EXT_WIKI)
    en_ligne_pdfs = _lister(root / "raw", EXT_RAW)

    # Le wiki monte en entier (quelques mégaoctets) : c'est ce qui permet de comparer les
    # contenus à la bascule au lieu de croire une taille annoncée.
    attendus: Dict[str, int] = {f"wiki/{rel}": t for rel, t in pages.items()}
    # Les PDF, eux, ne montent que s'ils manquent ou ont changé de taille.
    nouveaux, remplaces = [], []
    for rel, taille in sorted(pdfs.items()):
        presente = en_ligne_pdfs.get(rel)
        if presente == taille:
            continue
        attendus[f"raw/{rel}"] = taille
        (remplaces if presente is not None else nouveaux).append(rel)

    depot_id = uuid.uuid4().hex
    dossier = _dossier_depots(root) / depot_id
    (dossier / "wiki").mkdir(parents=True, exist_ok=True)
    manifeste = {
        "id": depot_id,
        "cree": datetime.now().isoformat(timespec="seconds"),
        "miroir_wiki": miroir,
        "attendus": attendus,
        "pdf_nouveaux": nouveaux,
        "pdf_remplaces": remplaces,
    }
    (dossier / "manifeste.json").write_text(
        json.dumps(manifeste, ensure_ascii=False), encoding="utf-8"
    )

    plan = {
        "miroir_wiki": miroir,
        "pages_deposees": len(pages),
        "pages_en_ligne": len(en_ligne_pages),
        "pages_ajoutees": sorted(set(pages) - set(en_ligne_pages)) if miroir else [],
        "pages_supprimees": sorted(set(en_ligne_pages) - set(pages)) if miroir else [],
        "pdf_deposes": len(pdfs),
        "pdf_en_ligne": len(en_ligne_pdfs),
        "pdf_nouveaux": nouveaux,
        "pdf_remplaces": remplaces,
    }
    logger.info(
        "[dépôt] %s ouvert — %d page(s), %d PDF annoncés ; %d fichier(s) réclamé(s) (%.1f Mo), "
        "%d écarté(s)",
        depot_id,
        len(pages),
        len(pdfs),
        len(attendus),
        sum(attendus.values()) / 2**20,
        len(ignores),
    )
    return {
        "depot": depot_id,
        "attendus": [
            {"chemin": origines[cle], "cle": cle, "octets": taille}
            for cle, taille in sorted(attendus.items())
        ],
        "octets_attendus": sum(attendus.values()),
        "plan": plan,
        "ignores": ignores[:60],
        "ignores_total": len(ignores),
    }


async def deposer(
    root: Path, depot_id: str, chemin: str, morceaux: AsyncIterator[bytes]
) -> Dict[str, Any]:
    """Écrit un fichier du dépôt, morceau par morceau : un catalogue de 55 Mo ne passe jamais en
    entier par la mémoire — ni côté serveur, ni dans un tampon de proxy.

    Seul un chemin **annoncé à l'ouverture** est accepté : le manifeste est la liste blanche. Le
    fichier s'écrit à côté (``part-…``) puis est renommé : personne ne lit un PDF à moitié reçu.
    """
    dossier, manifeste = _charger(root, depot_id)
    classement = classer(chemin)
    cle = classement.cle
    if not cle or cle not in manifeste["attendus"]:
        raise DepotRefuse(f"fichier hors du dépôt annoncé : {chemin}")

    cible = _cible(root, dossier, classement)
    provisoire = dossier / f"part-{uuid.uuid4().hex}"
    ecrits = 0
    try:
        with provisoire.open("wb") as sortie:
            async for morceau in morceaux:
                if not morceau:
                    continue
                ecrits += len(morceau)
                if ecrits > TAILLE_MAX:
                    raise DepotRefuse(f"{cle} dépasse {TAILLE_MAX // 2**20} Mo")
                sortie.write(morceau)
        cible.parent.mkdir(parents=True, exist_ok=True)
        os.replace(provisoire, cible)
    finally:
        provisoire.unlink(missing_ok=True)
    return {"chemin": cle, "octets": ecrits}


def _manquants(root: Path, dossier: Path, manifeste: Dict[str, Any]) -> List[str]:
    absents = []
    for cle in manifeste["attendus"]:
        classement = classer(cle)
        if not classement.racine or not _cible(root, dossier, classement).is_file():
            absents.append(cle)
    return absents


def valider(root: Path, depot_id: str) -> Dict[str, Any]:
    """Bascule le wiki préparé à la place du wiki servi, puis recharge l'instantané.

    La bascule est un couple de renommages sous le verrou de l'instantané : entre les deux, un
    tour de chat qui recharge attend au lieu de lire un dossier à moitié remplacé. Les PDF, eux,
    sont déjà à leur place — ils n'ont rien à basculer.
    """
    dossier, manifeste = _charger(root, depot_id)
    absents = _manquants(root, dossier, manifeste)
    if absents:
        raise DepotIncomplet(sorted(absents))

    resume: Dict[str, Any] = {
        "pages_ajoutees": [],
        "pages_modifiees": [],
        "pages_supprimees": [],
        "pdf_nouveaux": list(manifeste.get("pdf_nouveaux") or []),
        "pdf_remplaces": list(manifeste.get("pdf_remplaces") or []),
        "miroir_wiki": bool(manifeste.get("miroir_wiki")),
    }

    if manifeste.get("miroir_wiki"):
        vivant = root / "wiki"
        prepare = dossier / "wiki"
        avant, apres = _empreintes(vivant), _empreintes(prepare)
        resume["pages_ajoutees"] = sorted(set(apres) - set(avant))
        resume["pages_supprimees"] = sorted(set(avant) - set(apres))
        resume["pages_modifiees"] = sorted(
            cle for cle in set(avant) & set(apres) if avant[cle] != apres[cle]
        )
        ancien = dossier / "ancien"
        # Le verrou n'est PAS tenu pendant le rechargement : il n'est pas réentrant.
        with wiki_service.reload_lock():
            if vivant.exists():
                os.rename(vivant, ancien)
            os.rename(prepare, vivant)
        shutil.rmtree(ancien, ignore_errors=True)

    shutil.rmtree(dossier, ignore_errors=True)
    wiki_service.reset_snapshot()
    instantane = wiki_service.get_snapshot()
    logger.info(
        "[dépôt] %s validé — %d page(s) ajoutée(s), %d modifiée(s), %d supprimée(s), "
        "%d PDF nouveau(x), %d remplacé(s) ; le wiki sert %d pages",
        depot_id,
        len(resume["pages_ajoutees"]),
        len(resume["pages_modifiees"]),
        len(resume["pages_supprimees"]),
        len(resume["pdf_nouveaux"]),
        len(resume["pdf_remplaces"]),
        len(instantane.concept_pages),
    )
    return {"resume": resume, "stats": instantane.stats()}


def annuler(root: Path, depot_id: str) -> None:
    """Abandonne un dépôt : le wiki préparé est jeté, les PDF déjà montés restent (ils ne
    remplacent rien qu'on veuille défaire)."""
    dossier, _ = _charger(root, depot_id)
    shutil.rmtree(dossier, ignore_errors=True)
    logger.info("[dépôt] %s abandonné", depot_id)
