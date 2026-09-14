"""Lecture d'une page en IMAGE — un appel vision cadré, une réponse structurée.

Le principe tient en une phrase : **une page, une question, un appel, rien d'autre.**

Pourquoi la page est lue SEULE, et jamais au milieu du contexte du tour (mesuré le
2026-09-12 sur la planche des parcloses, doc 438 p.8) : la même page, lue par le même
modèle au milieu de 47 000 caractères, 5 images et une question diluée, rend le mauvais
chiffre ; lue seule avec sa question, elle rend le bon. Ce n'est ni le modèle ni la
résolution, c'est le cadrage de l'appel. C'est aussi pourquoi l'image ne remonte jamais
dans la conversation principale : elle y serait relue au hasard, et elle consommerait le
plafond de 8 images par requête de l'API.

**Aucune règle de lecture n'est ajoutée** — ni filtrage de couleur, ni recadrage, ni
relecture à d'autres résolutions, ni vote. Ces mécanismes ont été construits puis retirés
le 2026-09-14 : chacun rendait juste le cas pour lequel il avait été écrit et introduisait
son propre mode de défaillance ailleurs (une cote « corrigée » de 14 à 0 sur une page sans
rapport, des étiquettes de couleur qui contredisaient la valeur retenue). Décision d'Elie :
la page PNG et la question suffisent, et si le modèle lit mal, c'est le MODÈLE de lecture
qu'on change (``READER_PAGE_MODEL``), pas l'échafaudage autour.

Le danger propre à la délégation, mesuré lui aussi : interrogé sur une référence ABSENTE de
l'image, un modèle invente une valeur sans hésiter, et elle revient en texte propre,
indétectable. D'où le contrat de sortie : ``present=false`` et ``ambigu=true`` sont des
réponses de PREMIER RANG, et une page qui ne répond pas reste utile par son contenu
restitué — c'est au lecteur principal de faire le croisement (suivre une plage, lire la
ligne d'un tableau) à partir de cette matière.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import statistics
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
# Restitution fidèle de la page (tableaux en Markdown, lignes intactes). C'est ELLE qui
# rend la page utilisable quand la question demande un croisement que le lecteur de page ne
# peut pas faire seul — une plage de valeurs, une ligne de tableau, une procédure.
MAX_CONTENT_CHARS = 3500

PAGE_READER_SYSTEM = (
    "Tu lis UNE page d'un document technique, fournie en image. Tout ce que tu écris vient "
    "de cette image, et de rien d'autre.\n"
    "\n"
    "DEUX RÔLES, dans cet ordre :\n"
    "  (A) RESTITUER la partie utile de la page, pour que quelqu'un qui ne la voit pas "
    "puisse y travailler ;\n"
    "  (B) répondre à la question, si la page le permet.\n"
    "Une page dont tu n'as pas la réponse reste utile : restitue-la quand même.\n"
    "\n"
    "1. CONTENU — obligatoire. Rends en Markdown la partie de la page qui concerne la "
    "question, fidèlement.\n"
    "   - Un TABLEAU se rend en tableau Markdown, lignes et colonnes INTACTES. Ne l'aplatis "
    "pas, ne le résume pas, ne le réordonne pas : une ligne qui perd ses colonnes ne veut "
    "plus rien dire. S'il tient en entier, rends-le en entier.\n"
    "   - Les titres, notes, légendes et unités sont recopiés tels quels.\n"
    "   - Une suite d'étapes se rend en liste numérotée, dans l'ordre.\n"
    "   - Sur une planche de schémas, ne décris QUE les schémas utiles à la question, avec "
    "leur repère et les valeurs qui y sont écrites.\n"
    "\n"
    "2. RÉPONSE. Regarde la page et réponds.\n"
    "   - Ce qui est demandé ne figure pas sur la page → present=false, « reponse » vide.\n"
    "   - La page ne permet pas de trancher → ambigu=true, « reponse » vide. Une ambiguïté "
    "déclarée est utile ; un nombre choisi au hasard est une faute.\n"
    "   - La question demande un croisement que tu ne peux pas faire seul (une plage, un "
    "calcul, une taille intermédiaire) → « reponse » vide et present=true : le tableau que "
    "tu as restitué permettra de conclure.\n"
    "\n"
    "3. SCORE de 0 à 10 : à quel point CETTE page sert à répondre à CETTE question. "
    "10 = elle porte la réponse ; 5 = elle porte de la matière utile mais pas la réponse ; "
    "0 = hors sujet. Sois sévère : ce score décide des pages retenues.\n"
    "\n"
    "INTERDITS\n"
    "- Attribuer à un repère une valeur écrite à côté d'un AUTRE repère, même voisin.\n"
    "- Reprendre une note générale de la page (« valable pour une feuillure de 62 mm ») "
    "comme si c'était la valeur du repère demandé.\n"
    "- Résumer, interpréter ou commenter : tu restitues, tu ne rédiges pas.\n"
    "- Dessiner un schéma en caractères (encadrés, barres, flèches ASCII) : ça casse le "
    "format de sortie.\n"
    "\n"
    "Recopie dans « citations » ce que tu as lu mot pour mot sur la page à l'appui de ta "
    "réponse.\n"
    "\n"
    'Réponds en JSON strict, sans texte autour : {"type_page": str, "contenu": str, '
    '"present": bool, "ambigu": bool, "reponse": str, "citations": [str], "score": int}\n'
    '« type_page » : "tableau" | "planche cotée" | "texte" | "schéma" | "mixte" | "sommaire".'
)


@dataclass
class PageReading:
    """Ce qu'un modèle vision a lu sur UNE page, pour UNE question."""

    document_id: int
    page_no: int
    # Restitution fidèle de la page (Markdown, tableaux intacts). C'est la matière : une
    # page sans réponse reste exploitable par elle.
    content: str = ""
    page_type: str = ""
    answer: str = ""
    citations: List[str] = field(default_factory=list)
    absent: bool = False
    ambiguous: bool = False
    # Relevé systématique de la page : [{repere, valeurs: [{valeur, couleur}]}].
    # Plus demandé au lecteur ; conservé pour les lectures anciennes et le rendu du pack.
    survey: List[Dict[str, Any]] = field(default_factory=list)
    # Pertinence de la page pour CETTE question (0-10, motif RCS de PaperQA2) : décide
    # quelles pages sont retenues pour la rédaction.
    score: int = 0
    error: Optional[str] = None
    ms: int = 0

    @property
    def ok(self) -> bool:
        """La page a répondu à la question posée."""
        return (
            self.error is None
            and not self.absent
            and not self.ambiguous
            and bool(self.answer)
        )

    @property
    def usable(self) -> bool:
        """La page a été RESTITUÉE, qu'elle réponde ou non.

        C'est ce qui compte pour le pack : une page dont le tableau a été restitué permet
        au lecteur de conclure lui-même sur une plage ou une taille intermédiaire, même si
        la lecture n'a pas su répondre (régression du 13/09 sur « ouvrant de 700 mm »).
        """
        return self.error is None and bool(self.content or self.answer or self.survey)


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


def _coerce_values(raw: Any) -> List[Dict[str, str]]:
    """Normalise ``valeurs_du_repere`` : [{valeur, couleur}]. Tolérante à la forme."""
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            entry = {
                "valeur": _clean(item.get("valeur"), 40),
                "couleur": _clean(item.get("couleur"), 20),
            }
        else:
            entry = {"valeur": _clean(item, 40), "couleur": ""}
        if entry["valeur"]:
            out.append(entry)
        if len(out) >= 12:
            break
    return out


def _coerce_score(raw: Any) -> int:
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return 0
    return max(0, min(10, value))


# Nombre d'une réponse de lecture (« 30 », « 41,5 mm », « 6/18/4 » → premier nombre).
_FIRST_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def answer_number(answer: str) -> Optional[float]:
    """Valeur numérique portée par une réponse de lecture, ou None.

    Sert au consensus : seules les réponses CHIFFRÉES se votent. Une réponse en mots
    (« clipper la parclose ») n'a pas de médiane et se relit sans bénéfice.
    """
    match = _FIRST_NUMBER.search(answer or "")
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


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

    # Les modèles encadrent volontiers leur JSON d'un bloc ``` ou d'une phrase — mais depuis
    # que « contenu » restitue du Markdown, la réponse CONTIENT elle-même des ``` et des
    # tableaux. Chercher un bloc de code en premier découpait alors au milieu du JSON et
    # perdait la page entière (mesuré le 13/09 sur les pages 6 et 21 du dossier Perform76,
    # dont la sortie était pourtant complète et valide). On ne déballe donc que ce qui n'est
    # PAS déjà un objet JSON.
    if not text.startswith("{"):
        fenced = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        else:
            brace = re.search(r"\{.*\}", text, re.DOTALL)
            if brace:
                text = brace.group(0)

    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        # Repli TOLÉRANT : ``strict=False`` accepte les caractères de contrôle bruts dans
        # une chaîne. Mesuré le 13/09 sur la page 21 du dossier Perform76 — le modèle y
        # dessine un schéma en caractères, avec de vrais retours à la ligne non échappés,
        # et la page entière était perdue pour un défaut d'échappement. Une page perdue
        # coûte beaucoup plus cher qu'un contenu un peu sale.
        try:
            data = json.loads(text, strict=False)
        except (TypeError, ValueError):
            return PageReading(
                document_id, page_no, error=f"réponse non JSON du lecteur de page : {text[:120]}"
            )
    if not isinstance(data, dict):
        return PageReading(document_id, page_no, error="réponse JSON de forme inattendue")

    content = str(data.get("contenu") or "").strip()[:MAX_CONTENT_CHARS]
    page_type = _clean(data.get("type_page") or data.get("type"), 40)
    answer = _clean(data.get("reponse"), MAX_ANSWER_CHARS)
    # ``present`` (contrat du 14/09) est l'inverse de ``absent`` (contrat précédent). Les
    # deux sont acceptés : un modèle qui rend l'ancienne clé ne doit pas faire perdre la
    # page. Défaut prudent : sans aucune des deux clés, la page est considérée présente
    # (c'est ``reponse`` vide qui dira qu'elle n'a pas répondu).
    if "present" in data:
        absent = not bool(data.get("present"))
    else:
        absent = bool(data.get("absent"))
    ambiguous = bool(data.get("ambigu"))
    citations = _coerce_citations(data.get("citations"))
    survey = _coerce_survey(data.get("releve") or data.get("relevé"))
    score = _coerce_score(data.get("score"))
    # Un modèle qui répond « absent » ou « ambigu » tout en donnant une valeur se contredit :
    # le doute prime, c'est le sens sûr (on préfère renvoyer le lecteur chercher ailleurs
    # qu'entériner un nombre choisi au hasard).
    if absent or ambiguous:
        answer = ""
    return PageReading(
        document_id=document_id,
        page_no=page_no,
        content=content,
        page_type=page_type,
        answer=answer,
        citations=citations,
        absent=absent,
        ambiguous=ambiguous,
        survey=survey,
        score=score,
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
            max_tokens=4000,
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

    # JSON cassé : une SECONDE tentative, avec un rappel de format. Une page perdue coûte
    # bien plus cher qu'un appel de plus — c'est toute sa matière qui disparaît du tour.
    if reading.error and "JSON" in (reading.error or ""):
        logger.info(
            "[lecteur de page] doc=%s p.%s — sortie non JSON, seconde tentative",
            document_id,
            page_no,
        )
        try:
            retry = await chat(
                message="",
                model=model,
                context=context
                + [
                    {"role": "assistant", "content": str(content)[:500]},
                    {
                        "role": "system",
                        "content": "Ta sortie n'était pas un JSON valide. Recommence en "
                        "respectant EXACTEMENT le format demandé. N'utilise aucun dessin en "
                        "caractères, aucun retour à la ligne non échappé dans une chaîne.",
                    },
                ],
                temperature=0.0,
                max_tokens=4000,
                response_format={"type": "json_object"},
            )
            retry_choice = (retry.get("choices") or [{}])[0]
            retry_content = (retry_choice.get("message") or {}).get("content") or ""
            if isinstance(retry_content, list):
                retry_content = " ".join(
                    p.get("text", "") for p in retry_content if isinstance(p, dict)
                )
            retried = parse_page_reading(
                str(retry_content), document_id=document_id, page_no=page_no
            )
            if not retried.error:
                reading = retried
        except Exception as exc:  # noqa: BLE001
            logger.warning("[lecteur de page] seconde tentative échouée : %s", exc)

    reading.ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "[lecteur de page] doc=%s p.%s model=%s %dms → %s",
        document_id,
        page_no,
        model,
        reading.ms,
        reading.error
        or (reading.answer[:60] if reading.answer else ("absent" if reading.absent else "restitué"))
        + f" [{reading.page_type or '?'}, {len(reading.content)} car.]",
    )
    return reading
