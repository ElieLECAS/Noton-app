"""Runtime des parcours SAV guidés — refonte « Arbre SAV » (2026-07-30).

Un parcours = un arbre PUBLIÉ (snapshot GuidedTreeVersion épinglé par la session).
Traversée 100 % déterministe : zéro retrieval, zéro LLM par clic. Le seul appel LLM
possible est le mapping d'une réponse TAPÉE vers les choix du nœud courant (périmètre :
2-5 choix, jamais le corpus).

Entrées d'un parcours (chat.py) :
  - start_guided_session : bouton « Diagnostic SAV » / chip d'invitation / deep-link ;
  - run_guided_turn      : reprise d'une session active (clic de choix, précédent,
                           quitter, texte libre, feedback de feuille).

Le guidé ne démarre plus JAMAIS depuis une classification LLM du message libre —
le RAG répond toujours ; l'arbre est proposé, pas imposé.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.models.conversation import Conversation
from app.models.document import Document
from app.models.guided_session import GuidedSession
from app.models.guided_tree import GuidedTree
from app.services.authored_tree_service import (
    BACK_VALUE,
    FEEDBACK_NO_VALUE,
    FEEDBACK_YES_VALUE,
    QUIT_VALUE,
    breadcrumb_from_path,
    get_snapshot_node,
    load_published_snapshot,
    resolve_next_key,
    step_payload_from_node,
)

logger = logging.getLogger(__name__)

# Contact affiché dans le récap d'escalade (en dur — pas de flag).
SAV_CONTACT = "Service SAV PROFERM — contactez votre interlocuteur habituel."


class GuidedTurnResult(BaseModel):
    step: Dict[str, Any]
    guided_session_id: int
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    message_text: str = ""
    is_terminal: bool = False
    thinking: str = ""


def load_active_guided_state(query_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Retourne le bloc 'guided' de query_context si un parcours est actif, sinon None."""
    if not isinstance(query_context, dict):
        return None
    guided = query_context.get("guided")
    if (
        isinstance(guided, dict)
        and guided.get("active_session_id")
        and guided.get("phase") == "guided_active"
    ):
        return guided
    return None


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------


def _sources_from_attachments(session: Session, attachments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mappe les pièces jointes du nœud sur le format 'sources' du chat (ouverture PDF)."""
    if not attachments:
        return []
    doc_ids = list({int(a["document_id"]) for a in attachments if a.get("document_id")})
    has_file: Dict[int, bool] = {}
    if doc_ids:
        for d in session.exec(select(Document).where(Document.id.in_(doc_ids))).all():
            has_file[d.id] = bool(d.source_file_path)
    sources: List[Dict[str, Any]] = []
    for i, att in enumerate(attachments):
        did = int(att.get("document_id") or 0)
        sources.append(
            {
                "index": i + 1,
                "document_id": did,
                "document_title": att.get("document_title") or f"Document {did}",
                "excerpt": att.get("caption") or "Pièce jointe du diagnostic",
                "passage_full": att.get("caption") or "",
                "score": 1.0,
                "page_no": att.get("page_start"),
                "page_start": att.get("page_start"),
                "page_end": att.get("page_end"),
                "section": att.get("kind") or "notice",
                "has_source_file": has_file.get(did, False),
            }
        )
    return sources


def _record_answer(path: List[Dict[str, Any]], *, value: str, label: str, free_text: str = "") -> None:
    if not path:
        return
    last = path[-1]
    if last.get("user_selection") or last.get("free_text"):
        return
    if value:
        last["user_selection"] = {"value": value, "label": label or value}
        if label:
            last.setdefault("observations", []).append(str(label))
    if free_text:
        last["free_text"] = free_text
        last.setdefault("observations", []).append(free_text)


def _append_or_refresh_step(path: List[Dict[str, Any]], node_key: str, payload: Dict[str, Any]) -> None:
    """Ajoute l'étape présentée au parcours ; si le même nœud est re-présenté sans
    réponse (texte libre non mappé, retour), on rafraîchit l'entrée au lieu de dupliquer."""
    if path:
        last = path[-1]
        if last.get("node_key") == node_key and not last.get("user_selection") and not last.get("free_text"):
            last["message"] = payload.get("message") or ""
            last["timestamp"] = datetime.utcnow().isoformat()
            return
    path.append(
        {
            "step_index": len(path),
            "node_key": node_key,
            "step_type": payload.get("step_type"),
            "message": payload.get("message") or "",
            "choices": payload.get("choices") or [],
            "user_selection": None,
            "free_text": None,
            "observations": [],
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


def _collect_observations(path: List[Dict[str, Any]]) -> List[str]:
    obs: List[str] = []
    for rec in path:
        obs.extend(rec.get("observations") or [])
    return obs


def _build_escalation_recap(gsession: GuidedSession, node: Dict[str, Any]) -> Dict[str, Any]:
    """Récapitulatif SAV : chemin Q/R structuré + observations + photos + contact."""
    path = list(gsession.path or [])
    qa: List[Dict[str, str]] = []
    for rec in path:
        sel = rec.get("user_selection") or {}
        answer = sel.get("label") or rec.get("free_text") or ""
        if rec.get("message") and answer:
            qa.append({"question": str(rec["message"])[:300], "answer": str(answer)[:200]})
    photos = [f.get("path") for f in (gsession.uploaded_files or []) if f.get("path")]
    return {
        "summary": node.get("message") or node.get("title") or "Diagnostic non résolu, transmission au SAV.",
        "topic": gsession.topic or "",
        "steps_tested": [q["question"] for q in qa],
        "qa_path": qa,
        "observations": _collect_observations(path),
        "photos": photos,
        "contact": SAV_CONTACT,
    }


# Solutions déjà rédigées par LIA, par (arbre, version, cas). Une version publiée est
# figée : le texte est donc stable pour tous les clients, et calculé une seule fois.
_LEAF_ANSWER_CACHE: Dict[str, str] = {}
_LEAF_ANSWER_CACHE_MAX = 500

_LEAF_ANSWER_PROMPT = """Tu es LIA, l'assistante technique PROFERM (menuiserie, volets roulants).

Un technicien SAV a construit un arbre de dépannage. Pour la cause identifiée, il te donne son
INTITULÉ, sa DESCRIPTION (écrite par lui, c'est ta consigne de fond) et, quand il en a joint, les
EXTRAITS DE NOTICE qu'il a lui-même sélectionnés.

Rédige, pour le client, ce qu'il doit faire — 2 à 5 phrases, ton direct et concret, en
VOUVOIEMENT (« vérifiez », jamais « vérifie »), en texte brut (pas de JSON, pas de guillemets
autour de la réponse). Règles absolues :
- reste fidèle à la DESCRIPTION du technicien : c'est elle qui dit quoi faire ;
- pour les détails techniques (références, cotes, outils), appuie-toi UNIQUEMENT sur les extraits
  fournis ; n'invente rien qui n'y figure pas ;
- si la description et les extraits ne suffisent pas à décrire le geste, dis ce qui est certain et
  invite à consulter la notice jointe, sans broder ;
- pas de formule d'accueil ni de signature, va droit au but ;
- pas de titre, pas de liste à puces sauf si le geste est une vraie suite d'étapes ordonnées."""


def compose_leaf_answer(
    session: Session,
    *,
    tree_id: Optional[int],
    version: Optional[int],
    node: Dict[str, Any],
) -> str:
    """Texte de la solution donnée au client.

    Le SAV ne saisit que deux choses : le NOM du cas et sa DESCRIPTION. La description
    est la consigne de fond ; quand une notice est rattachée, LIA l'étoffe avec les pages
    choisies (CAG si la plage tient dans le budget, sinon recherche bornée à ces documents).
    Sans notice : la description telle quelle. Sans description : l'intitulé. Jamais d'invention.
    """
    title = str(node.get("title") or "").strip()
    description = str(node.get("message") or "").strip()
    attachments = node.get("attachments") or []
    if not attachments:
        return description or title

    cache_key = f"{tree_id}:{version}:{node.get('node_key')}"
    cached = _LEAF_ANSWER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from app.config import settings
        from app.services.guided_attachment_context_service import build_node_context
        from app.services.multimodal_page_service import _mistral_chat_completion

        context = build_node_context(session, attachments, question=description or title)
        source_text = (context.get("text") or "").strip()
        if not source_text:
            return description or title

        raw = _mistral_chat_completion(
            [
                {"role": "system", "content": _LEAF_ANSWER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Cause identifiée : {title}\n"
                        f"Description du technicien : {description or '(non renseignée)'}\n\n"
                        f"Extraits de notice sélectionnés par le SAV :\n{source_text}"
                    ),
                },
            ],
            page_no=0,
            max_tokens=500,
            temperature=0.1,
            response_format_json=False,   # prose : sinon l'aide renvoie {"message": ...}
            timeout_seconds=45,
            model=settings.MODEL_FAST,
        )
        answer = _strip_filler((raw or "").strip())
        if not answer:
            return description or title

        composed = f"**{title}**\n\n{answer}" if title else answer
        if len(_LEAF_ANSWER_CACHE) >= _LEAF_ANSWER_CACHE_MAX:
            _LEAF_ANSWER_CACHE.clear()
        _LEAF_ANSWER_CACHE[cache_key] = composed
        logger.info(
            "[guided_flow] solution rédigée par LIA (%s, mode=%s) pour « %s »",
            cache_key,
            context.get("mode"),
            title[:60],
        )
        return composed
    except Exception as exc:  # noqa: BLE001
        logger.warning("[guided_flow] rédaction de la solution échouée (%s) : %s", title[:40], exc)
        return title


_STEP_PROMPT = """Tu es LIA, l'assistante technique PROFERM. Tu accompagnes un client au téléphone
sur un dépannage de menuiserie / volet roulant, comme le ferait un technicien SAV expérimenté.

On te donne : le problème signalé, le chemin déjà parcouru (ce que le client a répondu), la
DESCRIPTION de l'étape en cours écrite par le SAV, et la liste des SITUATIONS proposées ensuite
avec leur description.

Tu produis DEUX choses : le message dit au client, et le libellé de chaque bouton de réponse.

Les situations sont nommées par le technicien avec SON vocabulaire (souvent la CAUSE : « fin de
course mal réglée », « condensateur HS »). Le client, lui, ne peut pas savoir quelle cause est la
sienne — c'est justement ce qu'on cherche. Tu dois donc réécrire chaque libellé en une
OBSERVATION que le client peut confirmer lui-même, dans ses mots.

Réponds en JSON strict :
{"message": "...", "options": [{"value": "<value fourni>", "label": "<libellé client>"}]}

Le message doit :
1. rebondir en une demi-phrase sur ce que le client vient de répondre (sans le répéter mot pour mot) ;
2. dire ce qu'on cherche à vérifier maintenant, et COMMENT le vérifier concrètement — quoi
   regarder, écouter ou toucher pour pouvoir trancher entre les situations proposées ;
3. terminer en l'invitant à choisir.

Les libellés doivent :
- décrire ce que le client CONSTATE, jamais la cause ni le diagnostic (« Rien ne bloque, il
  s'arrête net » et non « Fin de course mal réglée ») ;
- être courts (2 à 8 mots), concrets, et s'exclure mutuellement ;
- répondre directement à la question posée dans le message ;
- rester tels quels si l'intitulé du technicien est DÉJÀ compréhensible par un client
  (ex. « Le moteur ne fait aucun bruit ») ;
- reprendre EXACTEMENT le "value" fourni pour chaque situation, sans en inventer ni en oublier ;
- ne jamais contenir de jargon (référence, nom de pièce technique) que le client ne peut pas voir ;
- être COHÉRENTS entre eux et non contradictoires : chaque libellé doit être une réponse possible
  à la question posée, et une seule doit pouvoir être vraie à la fois.

Enfin, dans le message, ne fais JAMAIS référence à la position ou au nombre des boutons
(« la première option », « les deux choix ci-dessous ») : leur ordre peut changer.

Règles absolues pour le message :
- ta question doit permettre de DISTINGUER les situations proposées, et RIEN D'AUTRE. Regarde ce
  qui les différencie et parle de ça uniquement. Ne demande jamais de vérifier ce qui relève d'une
  étape ultérieure : le diagnostic avance pas à pas, une distinction à la fois ;
- si les situations se distinguent par une simple OBSERVATION (ce que le client constate déjà :
  le volet bouge ou pas, il y a un bruit ou pas), ne fais RIEN manipuler : demande simplement ce
  qu'il constate. Ne propose une manipulation que si c'est indispensable pour trancher ;
- s'il y a plus de deux situations, ne pose PAS de question fermée (oui/non) : demande laquelle
  correspond ;
- N'ÉNUMÈRE PAS les situations proposées : elles s'affichent déjà en boutons juste en dessous ;
- ne recopie pas les descriptions : reformule-les en langage parlé, utile et court ;
- 2 à 4 phrases, vouvoiement, ton calme et concret, aucune formule d'accueil ni de signature ;
- INTERDIT de commencer par « D'accord », « Très bien », « Parfait », « Entendu » ou « OK » :
  entre directement dans le sujet ;
- n'invente aucune référence, cote, ni manipulation qui ne soit pas dans les descriptions ;
- SÉCURITÉ : ne demande JAMAIS de toucher, manipuler ou tester un câble, un connecteur, un
  bornier ou une pièce sous tension. Pour l'électrique, ne fais constater que des signes
  VISIBLES ou AUDIBLES (voyant allumé ou éteint, bruit, absence de réaction) ; toute
  intervention sur le câblage relève d'un professionnel et doit être annoncée comme telle ;
- si le geste demandé présente un risque, rappelle la précaution en quelques mots (hors tension…)."""


_FILLER_RE = re.compile(
    r"^\s*(?:d['’]accord|tr[èe]s bien|parfait|entendu|ok|bien|compris)\s*[,.:;!…\-–]+\s*",
    re.IGNORECASE,
)


def _strip_filler(text: str) -> str:
    """Retire l'ouverture creuse (« D'accord, », « Très bien, »…).

    Le modèle la remet malgré la consigne, et répétée à chaque tour elle donne
    l'impression d'un automate. On la coupe côté code : c'est fiable.
    """
    out = _FILLER_RE.sub("", text or "", count=1).lstrip()
    return (out[0].upper() + out[1:]) if out else text


def _path_labels(path: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for rec in path:
        sel = rec.get("user_selection") or {}
        label = str(sel.get("label") or "").strip()
        if label and not label.startswith("__"):
            out.append(label)
    return out


def compose_step_message(
    session: Session,
    *,
    tree_id: Optional[int],
    version: Optional[int],
    snapshot: Dict[str, Any],
    node: Dict[str, Any],
    options: List[Dict[str, Any]],
    path: List[Dict[str, Any]],
    client_words: str = "",
) -> Dict[str, Any]:
    """Fait rédiger le tour par LIA au lieu de réciter la description du SAV.

    L'arbre reste déterministe : la STRUCTURE (les situations proposées, l'ordre) vient
    entièrement du SAV. Seule la FORMULATION est générée, à partir de la description de
    l'étape, de celles des situations en dessous, et du chemin déjà parcouru — c'est ce
    qui rend le tour guidant (« voilà ce qu'on vérifie, voilà comment ») au lieu d'être
    une définition suivie de boutons.

    Mémorisé par (arbre, version, étape, chemin) : deux clients qui suivent le même
    cheminement lisent le même texte, et on ne paie la rédaction qu'une fois.
    """
    fallback = {
        "message": str(node.get("message") or "").strip() or str(node.get("title") or "").strip(),
        "labels": {},
    }
    real_options = [o for o in options if not str(o.get("value") or "").startswith("__")]
    if not real_options:
        return fallback

    trail = _path_labels(path)
    # Le premier tour est personnalisé par les mots du client : il n'est donc pas mémorisé.
    first_turn = not trail and bool(client_words)
    cache_key = "|".join(
        [str(tree_id), str(version), str(node.get("node_key")), ">".join(trail)]
    )
    if not first_turn:
        cached = _LEAF_ANSWER_CACHE.get(cache_key)
        if cached is not None:
            try:
                return json.loads(cached)
            except Exception:  # noqa: BLE001
                pass

    try:
        from app.config import settings
        from app.services.multimodal_page_service import (
            _mistral_chat_completion,
            _parse_json_with_repair,
        )

        nodes = snapshot.get("nodes") or {}
        opt_lines = []
        for o in real_options:
            target = nodes.get(str(o.get("next_node_key") or "")) or {}
            desc = str(target.get("message") or "").strip()
            opt_lines.append(
                f'- value="{o.get("value")}" · nom technique : {o.get("label")}'
                + (f" · description : {desc}" if desc else "")
            )

        step_desc = str(node.get("message") or "").strip()

        # La notice rattachée à CETTE étape sert aussi à guider (où regarder, quoi écouter),
        # pas seulement à rédiger la solution finale.
        notice = ""
        if node.get("attachments"):
            try:
                from app.services.guided_attachment_context_service import build_node_context

                ctx = build_node_context(
                    session, node["attachments"], question=step_desc or str(node.get("title") or ""),
                    max_chars=1500,
                )
                notice = (ctx.get("text") or "").strip()[:1500]
            except Exception:  # noqa: BLE001
                notice = ""

        # Au premier tour, la description est celle du PROBLÈME (elle énumère les causes
        # possibles) : sans cette consigne, LIA posait déjà la question du tour suivant.
        opening = (
            "C'est le TOUT PREMIER tour : contente-toi de faire préciser ce que fait "
            "l'appareil, sans faire manipuler quoi que ce soit et sans parler des causes.\n\n"
            if not trail else ""
        )
        user = (
            opening
            + f"Problème signalé : {snapshot.get('title') or ''}\n"
            + (f"Contexte du problème (à NE PAS transformer en question) : {str(snapshot.get('description') or '').strip()}\n" if snapshot.get("description") else "")
            + (f"Ce que le client a écrit : « {client_words.strip()[:300]} »\n" if client_words.strip() else "")
            + f"Chemin déjà parcouru : {' > '.join(trail) if trail else '(début du diagnostic)'}\n\n"
            + f"Étape en cours : {node.get('title') or ''}\n"
            + f"Description du SAV pour cette étape : {step_desc or '(non renseignée)'}\n\n"
            + (f"Extraits de notice rattachés à cette étape :\n{notice}\n\n" if notice else "")
            + "Situations proposées ensuite (NE PAS les énumérer, elles sont en boutons) :\n"
            + "\n".join(opt_lines)
            + (
                f"\n\nRAPPEL : il y a {len(real_options)} situations à distinguer — demande "
                "explicitement laquelle correspond à ce que le client constate, jamais une "
                "question fermée oui/non."
                if len(real_options) > 2
                else "\n\nRAPPEL : deux situations seulement — la question doit trancher entre "
                     "les deux, sans en évoquer d'autres."
            )
        )
        raw = _mistral_chat_completion(
            [{"role": "system", "content": _STEP_PROMPT}, {"role": "user", "content": user}],
            page_no=0,
            max_tokens=600,
            temperature=0.4,
            response_format_json=True,
            timeout_seconds=30,
            model=settings.MODEL_FAST,
        )
        data = _parse_json_with_repair(raw) or {}
        text = _strip_filler(str(data.get("message") or "").strip())
        if not text:
            return fallback

        # Les libellés client ne sont acceptés que pour des "value" existants : la
        # traversée reste pilotée par les valeurs du SAV, jamais par le texte généré.
        allowed = {str(o.get("value")) for o in real_options}
        labels: Dict[str, str] = {}
        for item in data.get("options") or []:
            if not isinstance(item, dict):
                continue
            v, lab = str(item.get("value") or ""), str(item.get("label") or "").strip()
            if v in allowed and lab:
                labels[v] = lab[:90]

        out = {"message": text, "labels": labels}
        if not first_turn:
            if len(_LEAF_ANSWER_CACHE) >= _LEAF_ANSWER_CACHE_MAX:
                _LEAF_ANSWER_CACHE.clear()
            _LEAF_ANSWER_CACHE[cache_key] = json.dumps(out, ensure_ascii=False)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("[guided_flow] rédaction du tour échouée (%s) : %s", node.get("node_key"), exc)
        return fallback


def map_free_text_to_choice(
    node_message: str, choices: List[Dict[str, Any]], user_text: str
) -> Optional[str]:
    """Rattache une réponse tapée à l'un des choix du nœud (1 appel LLM léger, JSON).
    Retourne la value du choix, ou None si aucune correspondance sûre."""
    real_choices = [c for c in choices if c.get("value") and not str(c["value"]).startswith("__")]
    if not real_choices or not user_text.strip():
        return None
    try:
        from app.config import settings
        from app.services.multimodal_page_service import (
            _mistral_chat_completion,
            _parse_json_with_repair,
        )

        options = "\n".join(f"- {c['value']} : {c['label']}" for c in real_choices)
        messages = [
            {
                "role": "system",
                "content": (
                    "Tu rattaches la réponse d'un client à l'une des options d'une question de "
                    "diagnostic SAV (menuiserie). Renvoie UNIQUEMENT un JSON "
                    '{"matched_value": "<value>"} ou {"matched_value": null} si aucune option ne '
                    "correspond clairement. En cas de doute → null."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question posée : {node_message}\n\nOptions :\n{options}\n\n"
                    f"Réponse du client : {user_text.strip()[:500]}"
                ),
            },
        ]
        raw = _mistral_chat_completion(
            messages,
            page_no=0,
            max_tokens=100,
            temperature=0.0,
            response_format_json=True,
            timeout_seconds=20,
            model=settings.MODEL_FAST,
        )
        data = _parse_json_with_repair(raw)
        matched = data.get("matched_value")
        valid = {str(c["value"]) for c in real_choices}
        return str(matched) if matched and str(matched) in valid else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[guided_flow] mapping texte libre échoué : %s", exc)
        return None


def _persist_turn(
    session: Session,
    gsession: GuidedSession,
    conversation_id: int,
    *,
    terminal: bool,
) -> None:
    gsession.updated_at = datetime.utcnow()
    session.add(gsession)
    conv = session.get(Conversation, conversation_id)
    if conv is not None:
        qc = dict(conv.query_context or {})
        if terminal:
            qc.pop("guided", None)
        else:
            qc["guided"] = {
                "active_session_id": gsession.id,
                "phase": "guided_active",
                "flow_kind": gsession.flow_kind,
                "topic": gsession.topic,
            }
        conv.query_context = qc
        conv.updated_at = datetime.utcnow()
        session.add(conv)
    session.commit()
    session.refresh(gsession)


def _terminal_info_result(message: str, session_id: int = 0) -> GuidedTurnResult:
    return GuidedTurnResult(
        step={
            "step_type": "instruction",
            "message": message,
            "choices": [],
            "attachments": [],
            "is_terminal": True,
            "breadcrumb": [],
            "can_go_back": False,
        },
        guided_session_id=session_id,
        message_text=message,
        is_terminal=True,
    )


def _present_node(
    session: Session,
    gsession: GuidedSession,
    snapshot: Dict[str, Any],
    node: Dict[str, Any],
    *,
    conversation_id: int,
    perimeter: Optional[Dict[str, Any]],
    preface: str = "",
) -> GuidedTurnResult:
    """Construit l'étape du nœud courant, persiste la session + le pointeur, retourne."""
    path = list(gsession.path or [])
    is_leaf = bool(node.get("is_terminal"))
    is_resolution_leaf = is_leaf and node.get("termination_type") != "escalation"
    is_escalation_leaf = is_leaf and node.get("termination_type") == "escalation"

    payload = step_payload_from_node(
        node,
        snapshot,
        perimeter=perimeter,
        path=path,
        can_go_back=bool(path) and not is_escalation_leaf,
        awaiting_feedback=is_resolution_leaf,
    )
    # C'est LIA qui parle, jamais la fiche recopiée : sur une fin de parcours elle rédige
    # la solution depuis la notice rattachée, et sur un embranchement elle explique ce
    # qu'on vérifie et comment, à partir des descriptions du SAV.
    if is_leaf:
        payload["message"] = compose_leaf_answer(
            session,
            tree_id=gsession.authored_tree_id,
            version=gsession.tree_version,
            node=node,
        )
    else:
        turn = compose_step_message(
            session,
            tree_id=gsession.authored_tree_id,
            version=gsession.tree_version,
            snapshot=snapshot,
            node=node,
            options=payload.get("choices") or [],
            path=path,
            client_words=str((gsession.accumulated_signals or {}).get("client_words") or ""),
        )
        payload["message"] = turn.get("message") or payload["message"]
        # Boutons en langage client : le SAV nomme ses cas par la CAUSE (« fin de course
        # mal réglée »), que le client ne peut pas identifier. LIA les réécrit en
        # OBSERVATIONS. Seul le libellé change — la valeur, donc le chemin, est intacte.
        labels = turn.get("labels") or {}
        for choice in payload.get("choices") or []:
            new_label = labels.get(str(choice.get("value")))
            if new_label:
                choice["hint"] = choice.get("label") or ""   # le nom du SAV reste en infobulle
                choice["label"] = new_label
    if preface:
        payload["message"] = f"{preface}\n\n{payload['message']}".strip()

    if is_escalation_leaf:
        payload["escalation_recap"] = _build_escalation_recap(gsession, node)
        gsession.status = "escalated"
    else:
        gsession.status = "active"

    _append_or_refresh_step(path, str(node.get("node_key")), payload)
    payload["step_index"] = len(path) - 1
    gsession.path = path
    gsession.current_node_key = str(node.get("node_key"))

    terminal = bool(payload.get("is_terminal"))
    _persist_turn(session, gsession, conversation_id, terminal=terminal)

    return GuidedTurnResult(
        step=payload,
        guided_session_id=gsession.id,
        sources=_sources_from_attachments(session, payload.get("attachments") or []),
        message_text=payload.get("message") or "",
        is_terminal=terminal,
    )


# ---------------------------------------------------------------------------
# Démarrage d'un parcours (bouton / chip / deep-link)
# ---------------------------------------------------------------------------


def start_guided_session(
    session: Session,
    *,
    space_id: int,
    user_id: int,
    conversation_id: int,
    tree_slug: str = "",
    tree_id: Optional[int] = None,
    entry_node_key: Optional[str] = None,
    perimeter: Optional[Dict[str, Any]] = None,
) -> GuidedTurnResult:
    """Démarre un parcours sur un arbre publié. Toute session guidée active de la
    conversation est close (une seule à la fois)."""
    tree: Optional[GuidedTree] = None
    if tree_id is not None:
        tree = session.get(GuidedTree, tree_id)
    elif tree_slug:
        tree = session.exec(select(GuidedTree).where(GuidedTree.slug == tree_slug)).first()
    if tree is None or tree.status != "published" or not tree.current_version:
        return _terminal_info_result(
            "Ce diagnostic n'est pas (ou plus) disponible. Décrivez votre problème, je vous réponds avec la documentation."
        )
    if tree.space_id is not None and tree.space_id != space_id:
        return _terminal_info_result("Ce diagnostic n'appartient pas à cet espace.")

    snapshot = load_published_snapshot(session, tree.id, tree.current_version)
    if not snapshot:
        return _terminal_info_result("Version d'arbre introuvable — republier l'arbre depuis l'admin.")

    # Clore une éventuelle session active préexistante sur la conversation.
    conv = session.get(Conversation, conversation_id)
    prior = load_active_guided_state(conv.query_context if conv else None)
    if prior and prior.get("active_session_id"):
        old = session.get(GuidedSession, int(prior["active_session_id"]))
        if old is not None and old.status == "active":
            old.status = "abandoned"
            session.add(old)

    entry_key = entry_node_key or snapshot.get("root_node_key") or "root"
    node = get_snapshot_node(snapshot, entry_key)
    if node is None:
        entry_key = snapshot.get("root_node_key") or "root"
        node = get_snapshot_node(snapshot, entry_key)
    if node is None:
        return _terminal_info_result("Arbre sans racine — vérifier la publication.")

    # Les mots du client (son dernier message avant de lancer le diagnostic) : LIA s'en
    # sert pour accrocher le premier tour au lieu de démarrer à froid.
    client_words = ""
    try:
        from app.models.message import Message

        last = session.exec(
            select(Message)
            .where(Message.conversation_id == conversation_id, Message.role == "user")
            .order_by(Message.id.desc())
        ).first()
        text = str((last.content if last else "") or "").strip()
        if text and not text.startswith("🔧"):   # pas le libellé de démarrage lui-même
            client_words = text[:400]
    except Exception:  # noqa: BLE001
        client_words = ""

    gsession = GuidedSession(
        conversation_id=conversation_id,
        space_id=space_id,
        user_id=user_id,
        topic=tree.title[:300],
        flow_kind=tree.flow_kind or "diagnostic",
        source_mode="authored",
        authored_tree_id=tree.id,
        tree_version=tree.current_version,
        current_node_key=entry_key,
        status="active",
        path=[],
        accumulated_signals={
            "tree_slug": tree.slug,
            "entry_node_key": entry_key,
            "client_words": client_words,
        },
    )
    session.add(gsession)
    session.flush()

    logger.info(
        "[guided_flow] démarrage tree=%s v%s entry=%s conv=%s",
        tree.slug,
        tree.current_version,
        entry_key,
        conversation_id,
    )
    return _present_node(
        session, gsession, snapshot, node, conversation_id=conversation_id, perimeter=perimeter
    )


# ---------------------------------------------------------------------------
# Reprise d'un parcours actif (clic / précédent / quitter / texte libre / feedback)
# ---------------------------------------------------------------------------


async def run_guided_turn(
    *,
    session: Session,
    space_id: int,
    user_id: int,
    conversation_id: int,
    user_message: str,
    guided_choice: Optional[Dict[str, Any]] = None,
    active_state: Optional[Dict[str, Any]] = None,
    perimeter: Optional[Dict[str, Any]] = None,
    **_legacy_kwargs: Any,
) -> GuidedTurnResult:
    """Un tour de reprise sur la session active. (async conservé pour compat chat.py.)"""
    gsession: Optional[GuidedSession] = None
    if active_state and active_state.get("active_session_id"):
        gsession = session.get(GuidedSession, int(active_state["active_session_id"]))
    if gsession is None or gsession.status not in ("active", "escalated"):
        result = _terminal_info_result(
            "Le diagnostic n'est plus actif. Relancez-le depuis le bouton « Diagnostic SAV »."
        )
        conv = session.get(Conversation, conversation_id)
        if conv is not None:
            qc = dict(conv.query_context or {})
            qc.pop("guided", None)
            conv.query_context = qc
            session.add(conv)
            session.commit()
        return result

    snapshot = load_published_snapshot(session, gsession.authored_tree_id, gsession.tree_version)
    if not snapshot:
        gsession.status = "abandoned"
        _persist_turn(session, gsession, conversation_id, terminal=True)
        return _terminal_info_result(
            "La version de ce diagnostic n'est plus disponible — relancez-le depuis le bouton « Diagnostic SAV ».",
            gsession.id,
        )

    path: List[Dict[str, Any]] = list(gsession.path or [])
    current_key = gsession.current_node_key or snapshot.get("root_node_key") or "root"
    current = get_snapshot_node(snapshot, current_key)
    if current is None:
        gsession.status = "abandoned"
        _persist_turn(session, gsession, conversation_id, terminal=True)
        return _terminal_info_result("Nœud introuvable — arbre republier depuis l'admin.", gsession.id)

    value = str((guided_choice or {}).get("value") or "")
    label = str((guided_choice or {}).get("label") or "")

    # --- Quitter ---
    if value == QUIT_VALUE:
        gsession.status = "abandoned"
        gsession.path = path
        _persist_turn(session, gsession, conversation_id, terminal=True)
        return GuidedTurnResult(
            step={
                "step_type": "instruction",
                "message": "Diagnostic interrompu — je reste disponible, posez votre question librement.",
                "choices": [],
                "attachments": [],
                "is_terminal": True,
                "breadcrumb": breadcrumb_from_path(path),
                "can_go_back": False,
            },
            guided_session_id=gsession.id,
            message_text="Diagnostic interrompu — je reste disponible, posez votre question librement.",
            is_terminal=True,
        )

    # --- Précédent ---
    if value == BACK_VALUE:
        if path:
            path.pop()  # étape courante (en attente de réponse)
        if path:
            prev = path[-1]
            prev["user_selection"] = None
            prev["free_text"] = None
            prev["observations"] = []
            prev_key = str(prev.get("node_key") or snapshot.get("root_node_key") or "root")
        else:
            prev_key = snapshot.get("root_node_key") or "root"
        gsession.path = path
        node = get_snapshot_node(snapshot, prev_key) or get_snapshot_node(
            snapshot, snapshot.get("root_node_key") or "root"
        )
        return _present_node(
            session, gsession, snapshot, node, conversation_id=conversation_id, perimeter=perimeter
        )

    # --- Feedback de feuille résolution ---
    if value in (FEEDBACK_YES_VALUE, FEEDBACK_NO_VALUE):
        _record_answer(path, value=value, label=label or ("Résolu" if value == FEEDBACK_YES_VALUE else "Non résolu"))
        gsession.path = path
        if value == FEEDBACK_YES_VALUE:
            gsession.resolved_feedback = True
            gsession.status = "resolved"
            _persist_turn(session, gsession, conversation_id, terminal=True)
            msg = "Parfait — problème résolu ✅. Bonne continuation, et n'hésitez pas si autre chose se présente."
            return GuidedTurnResult(
                step={
                    "step_type": "resolution",
                    "message": msg,
                    "choices": [],
                    "attachments": [],
                    "is_terminal": True,
                    "breadcrumb": breadcrumb_from_path(path),
                    "can_go_back": False,
                },
                guided_session_id=gsession.id,
                message_text=msg,
                is_terminal=True,
            )
        # Non résolu → escalade avec récap complet du parcours.
        gsession.resolved_feedback = False
        gsession.status = "escalated"
        recap = _build_escalation_recap(gsession, current)
        msg = "Le problème persiste — je transmets un récapitulatif complet au SAV."
        payload = {
            "step_type": "escalation",
            "message": msg,
            "choices": [],
            "attachments": list(current.get("attachments") or []),
            "is_terminal": True,
            "escalation_recap": recap,
            "breadcrumb": breadcrumb_from_path(path),
            "can_go_back": False,
        }
        gsession.path = path
        _persist_turn(session, gsession, conversation_id, terminal=True)
        return GuidedTurnResult(
            step=payload,
            guided_session_id=gsession.id,
            sources=_sources_from_attachments(session, payload["attachments"]),
            message_text=msg,
            is_terminal=True,
        )

    # --- Choix normal ou texte libre ---
    preface = ""
    next_key: Optional[str] = None
    if value:
        _record_answer(path, value=value, label=label)
        next_key = resolve_next_key(current, value)
        if next_key is None:
            preface = "Je n'ai pas reconnu ce choix — reprenons :"
    elif user_message.strip():
        if bool(current.get("allow_free_text", True)):
            matched = map_free_text_to_choice(
                current.get("message") or "", current.get("choices") or [], user_message
            )
            if matched:
                matched_label = next(
                    (str(c.get("label")) for c in current.get("choices") or [] if str(c.get("value")) == matched),
                    matched,
                )
                _record_answer(path, value=matched, label=matched_label, free_text=user_message.strip())
                next_key = resolve_next_key(current, matched)
            else:
                if path:
                    path[-1].setdefault("observations", []).append(user_message.strip()[:300])
                preface = "J'ai noté votre remarque. Pour avancer, choisissez l'option la plus proche :"
        else:
            preface = "Choisissez l'une des options proposées :"

    gsession.path = path
    if next_key:
        node = get_snapshot_node(snapshot, next_key)
        if node is None:
            gsession.status = "abandoned"
            _persist_turn(session, gsession, conversation_id, terminal=True)
            return _terminal_info_result("Branche introuvable — arbre à republier.", gsession.id)
        return _present_node(
            session, gsession, snapshot, node, conversation_id=conversation_id, perimeter=perimeter
        )

    # Re-présenter le nœud courant (choix inconnu / texte libre non mappé).
    return _present_node(
        session,
        gsession,
        snapshot,
        current,
        conversation_id=conversation_id,
        perimeter=perimeter,
        preface=preface,
    )
