"""Lecture déléguée d'une page en IMAGE — un appel vision cadré, une réponse structurée.

Mesuré le 2026-09-12 sur la planche des parcloses (doc 438 p.8, cotation bleue = épaisseur
vitrage, vérité terrain 2452 → 16 mm et 2636 → 30 mm) :

  * page entière en PNG + question ciblée, en appel ISOLÉ → mistral-small répond juste sur
    les deux références, en ~800 ms, à 150 comme à 220 dpi ; large aussi ;
  * la MÊME page, lue par large au milieu de 47 000 caractères de contexte, 5 images et une
    question diluée → « 27 mm » (le chiffre NOIR au lieu du bleu).

Ce n'est donc ni une limite de vision, ni une question de résolution, ni une question de
modèle : c'est le contexte qui tue la lecture. D'où ce module — une page, une question,
rien d'autre — et le fait que l'image ne remonte JAMAIS dans la conversation principale
(elle ne consomme plus le plafond de 8 images par requête de l'API).

Le danger de la délégation, mesuré lui aussi : interrogé sur une référence ABSENTE de
l'image, le modèle invente une valeur sans hésiter (« 2452 = 38 mm » sur un recadrage qui
ne contient pas 2452). La valeur fabriquée reviendrait alors en texte propre, indétectable.
D'où le contrat de sortie : ``absent`` est une réponse de premier rang, et toute valeur
doit être accompagnée du libellé du repère lu à côté d'elle — ce qui rend l'erreur visible.

RETIRÉ le 2026-09-13 : la désambiguïsation déterministe par la couleur (second rendu de la
page avec tout sauf une couleur effacé, puis intersection d'ensembles). Elle rendait juste
la planche des parcloses, mais c'était un mécanisme bâti pour UNE page : sur les 26 pages
du dossier Perform76, une seule porte une légende de couleur. Décision d'Elie : le lecteur
lit la page depuis le PNG, sans règle ajoutée. Conséquence mesurée et assumée — sur cette
planche, le repère 2636 est lu 27 (le nombre noir voisin) au lieu de 30, de façon stable.
La parade générique reste le contrat de sortie : plusieurs valeurs sans règle lisible sur
la page → ``ambigu``, jamais un nombre choisi au hasard.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

# Plafonds de la réponse structurée (elle reste dans l'historique du lecteur).
MAX_ANSWER_CHARS = 600
MAX_CITATIONS = 8
MAX_CITATION_CHARS = 120
# Le relevé peut couvrir une planche entière (30 schémas) : c'est voulu, c'est lui qui
# force la lecture méthodique. Il ne remonte PAS entier au lecteur, seulement l'entrée utile.
MAX_SURVEY_ENTRIES = 80

PAGE_READER_SYSTEM = (
    "Tu lis UNE page d'un document technique, fournie en image. Tu réponds UNIQUEMENT à "
    "partir de ce qui est visible sur cette image.\n"
    "\n"
    "PROCÈDE DANS CET ORDRE, sans sauter d'étape. L'ordre est ce qui fait la justesse : "
    "chercher directement la réponse sur une planche dense fait lire le mauvais schéma.\n"
    "\n"
    "1. LÉGENDE. Une planche cotée dit presque toujours comment la lire : un encadré, une "
    "note, une couleur (« Cotation en bleu = épaisseur vitrage », « cotes en rouge = "
    "entraxes »…). Recopie-la VERBATIM dans « convention », ou laisse vide s'il n'y en a pas.\n"
    "\n"
    "2. RELEVÉ SYSTÉMATIQUE — l'étape la plus importante. Parcours la page ENTIÈRE, "
    "méthodiquement : colonne par colonne, de haut en bas, un schéma après l'autre. Pour "
    "CHAQUE repère (référence, libellé de profil, cellule de tableau), note toutes les "
    "valeurs qui lui sont attachées, avec leur couleur. Ne saute aucun élément et ne "
    "cherche PAS encore la réponse à la question — contente-toi de relever ce que tu vois.\n"
    '   Format : "releve": [{"repere": "Parclose 2452", "valeurs": [{"valeur": "16", '
    '"couleur": "bleu"}, {"valeur": "41.5", "couleur": "noir"}]}, …]\n'
    "\n"
    "3. RÉPONSE. Cherche maintenant le repère demandé DANS TON PROPRE RELEVÉ, et applique "
    "la convention de l'étape 1 pour choisir la bonne valeur.\n"
    "   - Le repère n'est pas dans ton relevé → absent=true, « reponse » vide.\n"
    "   - La convention désigne une couleur → c'est elle qui tranche, PAS la plausibilité, "
    "PAS l'ordre de lecture, PAS la taille du nombre.\n"
    "   - Aucune convention et plusieurs valeurs possibles → ambigu=true, « reponse » vide. "
    "Une ambiguïté déclarée est utile ; un nombre choisi au hasard est une faute.\n"
    "\n"
    "INTERDITS\n"
    "- Inventer une valeur par analogie avec un repère voisin ou par plausibilité.\n"
    "- Reprendre une note générale de la page (« valable pour une feuillure de 62 mm ») "
    "comme si c'était la cote du repère demandé.\n"
    "- Décrire la page en prose : le relevé et la réponse suffisent.\n"
    "\n"
    "Recopie dans « citations » le libellé EXACT du repère et la valeur retenue, séparément "
    '(ex. « Parclose 2636 », « 30 »).\n'
    "\n"
    'Réponds en JSON strict, sans texte autour : {"convention": str, "releve": '
    '[{"repere": str, "valeurs": [{"valeur": str, "couleur": str}]}], "absent": bool, '
    '"ambigu": bool, "reponse": str, "citations": [str]}'
)


@dataclass
class PageReading:
    """Ce qu'un modèle vision a lu sur UNE page, pour UNE question."""

    document_id: int
    page_no: int
    answer: str = ""
    citations: List[str] = field(default_factory=list)
    absent: bool = False
    ambiguous: bool = False
    # Règle de lecture inscrite sur la page (« Cotation en bleu = épaisseur vitrage »).
    convention: str = ""
    # Relevé systématique de la page : [{repere, valeurs: [{valeur, couleur}]}].
    survey: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    ms: int = 0

    @property
    def ok(self) -> bool:
        return (
            self.error is None
            and not self.absent
            and not self.ambiguous
            and bool(self.answer)
        )


def _coerce_citations(raw: Any) -> List[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        text = " ".join(str(item).split())
        if not text:
            continue
        out.append(text[:MAX_CITATION_CHARS])
        if len(out) >= MAX_CITATIONS:
            break
    return out


def _clean(value: Any, limit: int = MAX_CITATION_CHARS) -> str:
    return " ".join(str(value or "").split())[:limit]


def _coerce_survey(raw: Any) -> List[Dict[str, Any]]:
    """Normalise le relevé systématique : [{repere, valeurs: [{valeur, couleur}]}].

    Tolérante à la forme : certains modèles rendent ``valeurs`` comme une liste de chaînes,
    ou collent la couleur dans la valeur. Un relevé mal formé ne doit pas faire perdre la
    lecture — c'est lui qui porte la justesse.
    """
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        repere = _clean(item.get("repere") or item.get("repère") or item.get("ref"))
        if not repere:
            continue
        values: List[Dict[str, str]] = []
        rawvals = item.get("valeurs") or item.get("valeur") or []
        if not isinstance(rawvals, list):
            rawvals = [rawvals]
        for v in rawvals:
            if isinstance(v, dict):
                entry = {"valeur": _clean(v.get("valeur")), "couleur": _clean(v.get("couleur"))}
            else:
                entry = {"valeur": _clean(v), "couleur": ""}
            if entry["valeur"]:
                values.append(entry)
        out.append({"repere": repere, "valeurs": values})
        if len(out) >= MAX_SURVEY_ENTRIES:
            break
    return out


def survey_entry_for(survey: Sequence[Dict[str, Any]], needle: str) -> Optional[Dict[str, Any]]:
    """Entrée du relevé correspondant à un repère cherché (comparaison lâche)."""
    key = re.sub(r"[^a-z0-9]", "", (needle or "").lower())
    if not key:
        return None
    for entry in survey:
        flat = re.sub(r"[^a-z0-9]", "", str(entry.get("repere") or "").lower())
        if key and (key in flat or flat in key):
            return entry
    return None


def parse_page_reading(raw: str, *, document_id: int, page_no: int) -> PageReading:
    """Parse la réponse JSON du lecteur de page. Ne lève jamais (fonction pure, testable).

    Une sortie illisible n'est PAS traitée comme une lecture vide : elle devient une erreur
    explicite, sinon un JSON cassé se présenterait au lecteur comme « rien sur cette page ».
    """
    text = (raw or "").strip()
    if not text:
        return PageReading(document_id, page_no, error="réponse vide du lecteur de page")

    # Les modèles encadrent volontiers leur JSON d'un bloc ``` ou d'une phrase.
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    elif not text.startswith("{"):
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)

    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return PageReading(
            document_id, page_no, error=f"réponse non JSON du lecteur de page : {text[:120]}"
        )
    if not isinstance(data, dict):
        return PageReading(document_id, page_no, error="réponse JSON de forme inattendue")

    answer = _clean(data.get("reponse"), MAX_ANSWER_CHARS)
    absent = bool(data.get("absent"))
    ambiguous = bool(data.get("ambigu"))
    citations = _coerce_citations(data.get("citations"))
    convention = _clean(data.get("convention"), MAX_ANSWER_CHARS)
    survey = _coerce_survey(data.get("releve") or data.get("relevé"))
    # Un modèle qui répond « absent » ou « ambigu » tout en donnant une valeur se contredit :
    # le doute prime, c'est le sens sûr (on préfère renvoyer le lecteur chercher ailleurs
    # qu'entériner un nombre choisi au hasard).
    if absent or ambiguous:
        answer = ""
    return PageReading(
        document_id=document_id,
        page_no=page_no,
        answer=answer,
        citations=citations,
        absent=absent,
        ambiguous=ambiguous,
        convention=convention,
        survey=survey,
    )


async def read_page_image(
    *,
    pdf_path: str,
    document_id: int,
    page_no: int,
    question: str,
    needle: str = "",
    model: Optional[str] = None,
    dpi: Optional[int] = None,
) -> PageReading:
    """Rend la page en PNG et la fait lire par un modèle vision, sur UNE question."""
    import time

    from app.services.mistral_service import chat
    from app.services.multimodal_page_service import render_page_png_cached

    model = model or settings.READER_PAGE_MODEL
    dpi = dpi or settings.CAG_IMAGE_DPI
    t0 = time.perf_counter()

    try:
        png = await asyncio.to_thread(render_page_png_cached, pdf_path, page_no, dpi=dpi)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[lecteur de page] rendu PNG doc=%s p.%s échoué : %s", document_id, page_no, exc)
        return PageReading(document_id, page_no, error=f"rendu de la page impossible : {exc}")

    b64 = base64.b64encode(png).decode("utf-8")
    context: List[Dict[str, Any]] = [
        {"role": "system", "content": PAGE_READER_SYSTEM},
        {"role": "user", "content": question, "images": [b64]},
    ]
    try:
        result = await chat(
            message="",
            model=model,
            context=context,
            temperature=0.0,
            max_tokens=2500,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[lecteur de page] appel %s échoué (doc=%s p.%s) : %s", model, document_id, page_no, exc)
        return PageReading(document_id, page_no, error=f"lecture de la page indisponible : {exc}")

    choice = (result.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    if isinstance(content, list):  # contenu multipart
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))

    reading = parse_page_reading(str(content), document_id=document_id, page_no=page_no)
    reading.ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "[lecteur de page] doc=%s p.%s model=%s %dms → %s",
        document_id,
        page_no,
        model,
        reading.ms,
        "absent" if reading.absent else (reading.error or reading.answer[:80] or "(ambigu)"),
    )
    return reading
