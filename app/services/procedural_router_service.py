"""Génération d'un AIGUILLAGE procédural (une étape + des choix), pas une réponse finale.

Contrairement à la génération RAG one-shot, ce service produit, à partir des passages
récupérés et de l'historique du parcours, UNE seule étape (instruction ou question) et
2 à 5 choix mutuellement exclusifs. L'utilisateur choisit, et le moteur (guided_flow_service)
relance un tour. Grounding strict : aucune étape/cote/geste inventé hors des PASSAGES.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.config import settings
from app.services.mistral_service import chat

logger = logging.getLogger(__name__)

STEP_TYPES = ("instruction", "question", "diagnosis", "resolution", "escalation")

# Choix par défaut injectés si le LLM n'en fournit pas pour une étape non terminale.
_DEFAULT_HOWTO_CHOICES = [
    {"label": "C'est fait, étape suivante", "value": "next", "hint": ""},
    {"label": "Je suis bloqué sur cette étape", "value": "stuck", "hint": ""},
]
_DEFAULT_DIAGNOSTIC_CHOICES = [
    {"label": "Oui", "value": "yes", "hint": ""},
    {"label": "Non", "value": "no", "hint": ""},
    {"label": "Je ne sais pas", "value": "unknown", "hint": ""},
]


class RoutingChoice(BaseModel):
    label: str
    value: str
    hint: str = ""


class EscalationRecap(BaseModel):
    summary: str = ""
    steps_tested: List[str] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list)


class RoutingStep(BaseModel):
    step_type: str = "instruction"
    message: str = ""
    choices: List[RoutingChoice] = Field(default_factory=list)
    cited_pages: List[Dict[str, Any]] = Field(default_factory=list)
    is_terminal: bool = False
    escalation_recap: Optional[EscalationRecap] = None


GUIDED_ROUTER_SYSTEM_PROMPT = """Tu es LIA, l'aiguilleuse technique de PROFERM (menuiserie, volets roulants).
Ton rôle n'est PAS de donner la réponse complète, mais de GUIDER l'utilisateur PAS À PAS.

RÈGLE FONDAMENTALE :
- Tu produis UNE SEULE étape à la fois (une instruction OU une question), puis 2 à 5 choix
  mutuellement exclusifs pour que l'utilisateur indique où il en est.
- Tu ne déroules JAMAIS toute la procédure ni tout le diagnostic d'un coup.

GROUNDING STRICT (sécurité) :
- Fonde-toi EXCLUSIVEMENT sur les PASSAGES fournis. N'invente aucun geste, cote, rotation, clic,
  pièce ou valeur absent des passages.
- Restitue les verbes d'action et composants MOT POUR MOT depuis les passages.
- Si les passages ne couvrent pas la suite, propose une étape de VÉRIFICATION prudente
  ou escalade vers le SAV. Ne devine jamais.
- Cite tes sources : pour chaque fait, renseigne cited_pages avec {document_title, page_no}
  uniquement à partir des passages réellement fournis.
- FILTRE PRODUIT : le sujet (SUJET) fixe le produit cible. Si des passages parlent d'un autre
  produit ou d'une autre gamme, IGNORE-LES complètement. Ne mélange jamais deux produits.

CONTINUITÉ DU PARCOURS :
- Les ÉTAPES DÉJÀ DONNÉES sont acquises. Ne les répète pas, ne les reformule pas.
- La prochaine étape doit faire AVANCER l'utilisateur, pas revenir sur ce qui est fait.
- Si l'utilisateur signale un blocage sur la même micro-tâche plusieurs fois de suite,
  passe à une alternative concrète OU escalade — ne reformule pas la même instruction.

MODE "howto" (pose / montage sur chantier) :
- Déroule la procédure dans l'ORDRE. Donne l'étape SUIVANTE non encore présentée (cf. ÉTAPES DÉJÀ DONNÉES).
- Choix typiques : « C'est fait, étape suivante » (value "next"), « Je suis bloqué » (value "stuck").
- Quand la dernière étape utile des passages a été donnée → step_type "resolution", is_terminal true,
  message de clôture court, choices vide.

MODE "diagnostic" (SAV, problème sur produit posé) :
- Chaque étape est une QUESTION de vérification discriminante. Les choix sont les observations possibles
  (ex : « J'ai du courant » / « Pas de courant » / « Je ne sais pas »).
- Quand la cause est identifiée ET que les passages décrivent la correction → step_type "resolution",
  is_terminal true.
- Si bloqué / hors périmètre des passages / cas non résolu → step_type "escalation", is_terminal true,
  et remplis escalation_recap {summary, steps_tested[], observations[]} en synthétisant le parcours.

CONCISION : message court, direct, en français. Pas de blabla, pas de listes inutiles.

Retourne UNIQUEMENT un JSON :
{
  "step_type": "instruction | question | diagnosis | resolution | escalation",
  "message": "texte de l'étape (avec citations [Doc, page X])",
  "choices": [{"label": "...", "value": "...", "hint": ""}],
  "cited_pages": [{"document_title": "...", "page_no": 8}],
  "is_terminal": false,
  "escalation_recap": null
}
"""


def format_passages_for_prompt(passages: List[Dict[str, Any]], *, max_passage_chars: int = 1200) -> str:
    """Sérialise les passages récupérés pour le prompt d'aiguillage."""
    if not passages:
        return "(aucun passage pertinent trouvé)"
    blocks: List[str] = []
    for i, p in enumerate(passages, 1):
        text = str(p.get("passage_raw") or p.get("passage") or "").strip()
        if not text:
            continue
        if len(text) > max_passage_chars:
            text = text[: max_passage_chars - 1] + "…"
        title = p.get("document_title") or "Document sans titre"
        page_no = p.get("page_no") or p.get("page_start")
        page_info = f", page {page_no}" if page_no else ""
        score = float(p.get("score", 0.0) or 0.0)
        blocks.append(f"[{i}] ({score:.2f}) {title}{page_info}\n{text}")
    return "\n---\n".join(blocks) if blocks else "(aucun passage pertinent trouvé)"


def _format_path_for_prompt(path: List[Dict[str, Any]]) -> str:
    """Récapitule les étapes déjà présentées et les réponses de l'utilisateur."""
    if not path:
        return "(début du parcours, aucune étape encore donnée)"
    lines: List[str] = []
    for rec in path:
        idx = rec.get("step_index")
        msg = str(rec.get("message") or "").strip()
        lines.append(f"Étape {idx} (LIA) : {msg}")
        sel = rec.get("user_selection") or {}
        free = rec.get("free_text")
        if sel and sel.get("label"):
            lines.append(f"  → Réponse utilisateur : {sel.get('label')}")
        elif free:
            lines.append(f"  → Réponse utilisateur (texte libre) : {free}")
    return "\n".join(lines)


def _allowed_page_keys(passages: List[Dict[str, Any]]) -> set:
    keys = set()
    for p in passages:
        title = (p.get("document_title") or "").strip().lower()
        page_no = p.get("page_no") or p.get("page_start")
        keys.add((title, page_no))
        keys.add((title, None))  # autoriser citation document sans page
    return keys


def _sanitize_step(
    step: RoutingStep,
    *,
    flow_kind: str,
    passages: List[Dict[str, Any]],
) -> RoutingStep:
    """Normalise/borne la sortie LLM : type valide, choix bornés, citations validées."""
    if step.step_type not in STEP_TYPES:
        step.step_type = "instruction"

    terminal_types = {"resolution", "escalation"}
    if step.step_type in terminal_types:
        step.is_terminal = True
    # Une étape escalation doit toujours porter un récap (même minimal)
    if step.step_type == "escalation" and step.escalation_recap is None:
        step.escalation_recap = EscalationRecap(summary=step.message or "")

    if step.is_terminal:
        step.choices = []
    else:
        # Borne le nombre de choix ; injecte des défauts si le LLM n'en fournit pas.
        if not step.choices:
            defaults = _DEFAULT_HOWTO_CHOICES if flow_kind == "howto" else _DEFAULT_DIAGNOSTIC_CHOICES
            step.choices = [RoutingChoice(**c) for c in defaults]
        if len(step.choices) < max(2, settings.GUIDED_MIN_CHOICES):
            # Compléter avec une porte de sortie générique
            existing_values = {c.value for c in step.choices}
            if "stuck" not in existing_values:
                step.choices.append(RoutingChoice(label="Je suis bloqué", value="stuck"))
        if len(step.choices) > settings.GUIDED_MAX_CHOICES:
            step.choices = step.choices[: settings.GUIDED_MAX_CHOICES]

    # Valider les citations contre les passages réellement fournis (anti-hallucination)
    allowed = _allowed_page_keys(passages)
    validated_pages: List[Dict[str, Any]] = []
    for cp in step.cited_pages:
        if not isinstance(cp, dict):
            continue
        title = (cp.get("document_title") or "").strip().lower()
        page_no = cp.get("page_no")
        if (title, page_no) in allowed or (title, None) in allowed:
            validated_pages.append(cp)
    step.cited_pages = validated_pages

    return step


async def generate_routing_step(
    *,
    flow_kind: str,
    topic: str,
    passages: List[Dict[str, Any]],
    path: List[Dict[str, Any]],
    step_index: int,
    force_terminal: bool = False,
    model: Optional[str] = None,
) -> RoutingStep:
    """Génère la prochaine étape d'aiguillage à partir des passages et du parcours.

    force_terminal : impose une clôture (escalade) quand GUIDED_MAX_STEPS est atteint.
    """
    passages_block = format_passages_for_prompt(passages)
    path_block = _format_path_for_prompt(path)

    terminal_instruction = ""
    if force_terminal:
        terminal_instruction = (
            "\n\nCONTRAINTE : le nombre maximal d'étapes est atteint. Tu DOIS clôturer : "
            "step_type=\"escalation\", is_terminal=true, et remplir escalation_recap en synthétisant "
            "les étapes testées et observations recueillies."
        )

    user_prompt = (
        f"MODE : {flow_kind}\n"
        f"SUJET : {topic or '(non précisé)'}\n"
        f"INDEX DE L'ÉTAPE À PRODUIRE : {step_index}\n\n"
        f"ÉTAPES DÉJÀ DONNÉES ET RÉPONSES :\n{path_block}\n\n"
        f"PASSAGES (source unique de vérité) :\n{passages_block}\n\n"
        "Produis la PROCHAINE étape d'aiguillage (une seule), en respectant strictement le grounding."
        f"{terminal_instruction}"
    )

    try:
        response = await chat(
            "",
            model=model or settings.MODEL_FAST,
            context=[
                {"role": "system", "content": GUIDED_ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"].get("content", "{}")
        data = json.loads(content)
        step = RoutingStep(
            step_type=str(data.get("step_type") or "instruction").strip().lower(),
            message=str(data.get("message") or "").strip(),
            choices=[
                RoutingChoice(
                    label=str(c.get("label") or "").strip(),
                    value=str(c.get("value") or c.get("label") or "").strip(),
                    hint=str(c.get("hint") or "").strip(),
                )
                for c in (data.get("choices") or [])
                if isinstance(c, dict) and (c.get("label") or c.get("value"))
            ],
            cited_pages=[c for c in (data.get("cited_pages") or []) if isinstance(c, dict)],
            is_terminal=bool(data.get("is_terminal")),
            escalation_recap=(
                EscalationRecap(
                    summary=str((data.get("escalation_recap") or {}).get("summary") or "").strip(),
                    steps_tested=[
                        str(s).strip()
                        for s in ((data.get("escalation_recap") or {}).get("steps_tested") or [])
                        if str(s).strip()
                    ],
                    observations=[
                        str(o).strip()
                        for o in ((data.get("escalation_recap") or {}).get("observations") or [])
                        if str(o).strip()
                    ],
                )
                if isinstance(data.get("escalation_recap"), dict)
                else None
            ),
        )
    except Exception as exc:
        logger.error("[procedural_router] génération étape échouée: %s", exc)
        # Repli sûr : escalade vers le SAV plutôt que d'inventer une étape.
        return RoutingStep(
            step_type="escalation",
            message=(
                "Je ne parviens pas à poursuivre le guidage automatiquement. "
                "Je transmets votre demande au SAV."
            ),
            is_terminal=True,
            escalation_recap=EscalationRecap(summary=f"Erreur technique du moteur de guidage: {exc}"),
        )

    if force_terminal and not step.is_terminal:
        step.step_type = "escalation"
        step.is_terminal = True
        if step.escalation_recap is None:
            step.escalation_recap = EscalationRecap(summary=step.message or "")

    return _sanitize_step(step, flow_kind=flow_kind, passages=passages)
