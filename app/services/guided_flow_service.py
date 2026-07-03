"""Orchestrateur du guidage procédural ("aiguillage" SAV / chantier).

Boucle par tour HTTP (le multi-tours est porté par la persistance DB, pas par un graphe
en mémoire — il ne survivrait pas entre deux requêtes SSE) :
  charger/créer la GuidedSession → enregistrer la réponse de l'utilisateur au tour précédent
  → récupération documentaire scopée par catégorie → générer l'étape d'aiguillage suivante
  → persister (GuidedSession + pointeur dans Conversation.query_context) → retourner.

Phase 1 : génération 100 % dynamique (les arbres validés priment en Phase 2 via
authored_tree_service ; le hook match_authored_tree est volontairement absent ici).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.config import settings
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.guided_session import GuidedSession
from app.services.category_catalog import suggested_categories_for_intent
from app.services.procedural_router_service import (
    RoutingChoice,
    RoutingStep,
    generate_routing_step,
)
from app.services.query_signals_schemas import LightweightQuerySignals

# node_key réservé de l'étape 0 d'identification du produit (slot-filling).
PRODUCT_IDENTIFICATION_NODE_KEY = "product_identification"
# Valeur du choix permanent « changer de produit / recommencer ».
RESTART_PRODUCT_VALUE = "restart_product"

logger = logging.getLogger(__name__)


class GuidedTurnResult(BaseModel):
    step: Dict[str, Any]
    guided_session_id: int
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    message_text: str = ""
    is_terminal: bool = False


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


def _synth_signals(
    flow_kind: str, accumulated: Dict[str, Any], topic: str = ""
) -> LightweightQuerySignals:
    """Synthétise des signaux par étape pour scoper la récupération (boost catégories).

    Le topic (ex. "pose joint périphérique KÖMMERLING 70") est injecté comme première
    detected_reference pour ancrer le retrieval sur le bon produit/document à chaque tour.
    """
    intent = "installation" if flow_kind == "howto" else "troubleshooting"
    inferred = suggested_categories_for_intent(intent)
    entity_texts = [str(t) for t in (accumulated.get("entity_texts") or []) if str(t).strip()]
    detected_refs = list(entity_texts)
    if topic and topic not in detected_refs:
        detected_refs.insert(0, topic)
    return LightweightQuerySignals(
        intent=intent,
        inferred_categories=inferred,
        entity_texts=entity_texts,
        detected_references=detected_refs,
    )


def _collect_observations(path: List[Dict[str, Any]]) -> List[str]:
    """Aplatit les observations de tout le parcours.

    _record_user_answer écrit déjà label et free_text dans rec["observations"] →
    ne pas les relire depuis user_selection/free_text (doublon).
    """
    obs: List[str] = []
    for rec in path:
        obs.extend(rec.get("observations") or [])
    return obs


def _build_sources(session: Session, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Construit la liste de sources (citations → PDF) à partir des passages récupérés."""
    if not passages:
        return []
    doc_ids = list({p.get("document_id") for p in passages if p.get("document_id")})
    has_file: Dict[int, bool] = {}
    if doc_ids:
        docs = session.exec(select(Document).where(Document.id.in_(doc_ids))).all()
        has_file = {d.id: bool(d.source_file_path) for d in docs}

    sources: List[Dict[str, Any]] = []
    for i, p in enumerate(passages):
        raw = str(p.get("passage_raw") or p.get("passage") or "")
        did = p.get("document_id")
        sources.append(
            {
                "index": i + 1,
                "document_id": did,
                "document_title": p.get("document_title", "Document sans titre"),
                "chunk_id": p.get("chunk_id"),
                "chunk_index": p.get("chunk_index"),
                "excerpt": (raw[:200] + "...") if len(raw) > 200 else raw,
                "passage_full": raw,
                "score": round(float(p.get("score", 0) or 0), 2),
                "page_no": p.get("page_no") or p.get("page_start"),
                "page_start": p.get("page_start"),
                "page_end": p.get("page_end"),
                "section": p.get("section"),
                "has_source_file": has_file.get(did, False),
            }
        )
    return sources


async def _retrieve_warranty_excerpt(
    session: Session, space_id: int, user_id: int, topic: str
) -> str:
    """Récupère un extrait garantie pour enrichir le récap d'escalade (best-effort)."""
    try:
        from app.services.space_search_service import search_technical_passages

        warranty_signals = LightweightQuerySignals(
            intent="regulatory", inferred_categories=["warranty"]
        )
        retrieval = await search_technical_passages(
            session=session,
            space_id=space_id,
            query_text=f"garantie {topic}".strip(),
            user_id=user_id,
            k=2,
            signals=warranty_signals,
        )
        passages = retrieval.get("passages") or []
        if passages:
            raw = str(passages[0].get("passage_raw") or passages[0].get("passage") or "")
            return raw[:400]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[guided_flow] récupération garantie échouée: %s", exc)
    return ""


async def _build_escalation_recap(
    session: Session,
    *,
    space_id: int,
    user_id: int,
    topic: str,
    prior_path: List[Dict[str, Any]],
    step: RoutingStep,
) -> Dict[str, Any]:
    """Assemble le récapitulatif SAV : étapes testées + observations + garantie + contact."""
    base = step.escalation_recap.model_dump() if step.escalation_recap else {}
    steps_tested = [str(r.get("message") or "") for r in prior_path if r.get("message")]
    observations = _collect_observations(prior_path)

    warranty_excerpt = await _retrieve_warranty_excerpt(session, space_id, user_id, topic)
    return {
        "summary": base.get("summary") or "Diagnostic non résolu, transmission au SAV.",
        "topic": topic,
        "steps_tested": steps_tested or base.get("steps_tested", []),
        "observations": observations or base.get("observations", []),
        "warranty_excerpt": warranty_excerpt,
        "contact": settings.GUIDED_SAV_CONTACT,
    }


def _collect_product_options(
    session: Session, passages: List[Dict[str, Any]], *, max_options: int
) -> List[str]:
    """Libellés de produits/gammes candidats depuis les documents du retrieval (ordre de
    pertinence) : gamme PROFERM si renseignée, sinon titre du document. Dédupliqués."""
    doc_ids: List[int] = []
    for p in passages:
        did = p.get("document_id")
        if isinstance(did, int) and did not in doc_ids:
            doc_ids.append(did)
    if not doc_ids:
        return []
    docs = {
        d.id: d for d in session.exec(select(Document).where(Document.id.in_(doc_ids))).all()
    }
    options: List[str] = []
    seen: set = set()
    for did in doc_ids:
        doc = docs.get(did)
        if not doc:
            continue
        gammes = list(getattr(doc, "proferm_gammes", None) or [])
        label = (gammes[0] if gammes else (doc.title or "")).strip()
        if len(label) > 60:
            label = label[:57] + "…"
        key = label.lower()
        if not label or key in seen:
            continue
        seen.add(key)
        options.append(label)
        if len(options) >= max_options:
            break
    return options


def _build_identification_step(
    session: Session, passages: List[Dict[str, Any]], flow_kind: str
) -> RoutingStep:
    """Étape 0 obligatoire quand l'utilisateur n'a pas nommé le produit : question
    d'identification dont les choix viennent des documents candidats du retrieval.
    Le SUJET du parcours n'est verrouillé qu'après la réponse — jamais déduit du retrieval."""
    max_products = max(1, settings.GUIDED_MAX_CHOICES - 1)
    options = _collect_product_options(session, passages, max_options=max_products)
    choices = [
        RoutingChoice(label=opt, value=f"product_{i + 1}") for i, opt in enumerate(options)
    ]
    choices.append(RoutingChoice(label="Autre / je ne sais pas", value="unknown"))
    action = "vous guider" if flow_kind == "howto" else "commencer le diagnostic"
    message = (
        f"Avant de {action} : sur quel produit ou quelle gamme intervenez-vous ? "
        "Cela m'évite de vous orienter vers la mauvaise notice."
    )
    return RoutingStep(
        step_type="question",
        message=message,
        choices=choices,
        cited_pages=[],
        is_terminal=False,
    )


def _record_user_answer(
    path: List[Dict[str, Any]],
    *,
    user_message: str,
    guided_choice: Optional[Dict[str, Any]],
) -> None:
    """Enregistre la réponse de l'utilisateur dans la dernière étape (en attente)."""
    if not path:
        return
    last = path[-1]
    if last.get("user_selection") or last.get("free_text"):
        return  # déjà répondu (idempotence)

    is_free_text = not (guided_choice and guided_choice.get("value")) or bool(
        guided_choice and guided_choice.get("free_text")
    )
    if guided_choice and guided_choice.get("value") and not guided_choice.get("free_text"):
        label = guided_choice.get("label") or guided_choice.get("value")
        last["user_selection"] = {"value": guided_choice.get("value"), "label": label}
        if label:
            last.setdefault("observations", []).append(str(label))
    elif is_free_text and user_message:
        last["free_text"] = user_message
        last.setdefault("observations", []).append(user_message)


async def run_guided_turn(
    *,
    session: Session,
    space_id: int,
    user_id: int,
    conversation_id: int,
    user_message: str,
    guided_choice: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    active_state: Optional[Dict[str, Any]] = None,
    flow_kind: str = "howto",
    topic: str = "",
    symptom: str = "",
    product_named: bool = True,
    anchor_document_ids: Optional[List[int]] = None,
) -> GuidedTurnResult:
    """Exécute un tour de guidage et retourne l'étape d'aiguillage à streamer.

    Écrit la GuidedSession et le pointeur Conversation.query_context['guided'] via la
    session de requête (encore ouverte au niveau routeur). La persistance du message
    assistant reste gérée par l'appelant (session fraîche dans le générateur SSE).
    """
    # 1. Charger ou créer la session de guidage
    gsession: Optional[GuidedSession] = None
    if active_state and active_state.get("active_session_id"):
        gsession = session.get(GuidedSession, int(active_state["active_session_id"]))
        if gsession and gsession.status != "active":
            gsession = None  # session close → on en recrée une

    resuming = gsession is not None
    if gsession is None:
        norm_flow = flow_kind if flow_kind in ("howto", "diagnostic") else "howto"
        # Phase 2 : un arbre validé prime sur la génération dynamique s'il correspond.
        from app.services.authored_tree_service import match_authored_tree

        inferred_for_match = suggested_categories_for_intent(
            "installation" if norm_flow == "howto" else "troubleshooting"
        )
        if symptom:
            inferred_for_match = [symptom, *inferred_for_match]
        matched_tree = match_authored_tree(
            session,
            space_id=space_id,
            flow_kind=norm_flow,
            symptom=symptom or None,
            inferred_categories=inferred_for_match,
            topic=topic or user_message,
            user_message=user_message,
        )
        # Étape 0 d'identification produit : uniquement en mode dynamique (un arbre
        # validé porte son propre cheminement) et quand l'utilisateur n'a PAS nommé le
        # produit — on ne verrouille JAMAIS un produit déduit du retrieval (cf. dérive
        # INNOSLIDE : « comment régler la hauteur de poignée ? » plongeait dans une gamme
        # jamais mentionnée par l'utilisateur).
        needs_identification = not matched_tree and not product_named
        gsession = GuidedSession(
            conversation_id=conversation_id,
            space_id=space_id,
            user_id=user_id,
            topic=topic or user_message[:300],
            flow_kind=norm_flow,
            source_mode="authored" if matched_tree else "dynamic",
            authored_tree_id=matched_tree.id if matched_tree else None,
            current_node_key=matched_tree.root_node_key if matched_tree else None,
            status="active",
            path=[],
            accumulated_signals={
                "original_user_message": user_message,
                "entity_texts": [],
                "symptom": symptom or "",
                # Étiquette de TÂCHE (sans produit) : sert à recomposer le sujet après
                # identification ou après « changer de produit ».
                "task_topic": (topic or user_message[:300]),
                "pending_product_identification": needs_identification,
                "confirmed_product": "",
            },
        )
        session.add(gsession)
        session.flush()  # obtenir l'id avant d'écrire le pointeur query_context

    flow_kind = gsession.flow_kind
    topic = gsession.topic
    path: List[Dict[str, Any]] = list(gsession.path or [])
    accumulated: Dict[str, Any] = dict(gsession.accumulated_signals or {})

    # 2. Enregistrer la réponse de l'utilisateur à l'étape précédente (reprise)
    if resuming:
        _record_user_answer(path, user_message=user_message, guided_choice=guided_choice)

        choice_value = str((guided_choice or {}).get("value") or "")
        last_step = path[-1] if path else None

        if choice_value == RESTART_PRODUCT_VALUE:
            # « Autre produit / recommencer » : on rouvre l'identification, le sujet
            # retombe sur l'étiquette de tâche (sans produit).
            accumulated["pending_product_identification"] = True
            accumulated["confirmed_product"] = ""
            topic = str(accumulated.get("task_topic") or topic or user_message[:300])
            gsession.topic = topic[:300]
        elif last_step and last_step.get("node_key") == PRODUCT_IDENTIFICATION_NODE_KEY:
            # Réponse à l'étape 0 : verrouiller (ou pas) le produit CONFIRMÉ par l'utilisateur.
            accumulated["pending_product_identification"] = False
            label = ""
            if choice_value and choice_value != "unknown":
                label = str((guided_choice or {}).get("label") or "").strip()
            elif not choice_value and user_message.strip():
                # Texte libre (l'utilisateur a tapé le nom de son produit).
                label = user_message.strip()[:120]
            if label:
                accumulated["confirmed_product"] = label
                entity_texts = [str(t) for t in (accumulated.get("entity_texts") or [])]
                if label not in entity_texts:
                    entity_texts.insert(0, label)
                accumulated["entity_texts"] = entity_texts
                base_topic = str(accumulated.get("task_topic") or "").strip()
                topic = f"{base_topic} — {label}".strip(" —") or label
                gsession.topic = topic[:300]
            else:
                # « Je ne sais pas » : pas de produit verrouillé — le routeur posera des
                # questions d'OBSERVATION discriminantes, sans présumer d'une gamme.
                accumulated["confirmed_product"] = ""
                last_step.setdefault("observations", []).append(
                    "Produit non identifié par l'utilisateur"
                )

    # 3. Résoudre le nœud d'arbre validé (Phase 2) si la session est en mode authored.
    authored_node = None
    if gsession.source_mode == "authored":
        from app.services.authored_tree_service import resolve_authored_node

        authored_node = resolve_authored_node(
            session, gsession, guided_choice=guided_choice, resuming=resuming
        )
        if authored_node is None:
            logger.info(
                "[guided_flow] arbre %s en impasse → repli génération dynamique",
                gsession.authored_tree_id,
            )

    # 4. Récupération documentaire scopée (par le nœud si authored, sinon par l'intent)
    observations = _collect_observations(path)
    base_query = topic or accumulated.get("original_user_message") or user_message
    query_text = " ".join([base_query] + observations[-3:]).strip() or user_message
    if authored_node is not None:
        from app.services.authored_tree_service import signals_for_authored_node

        synth_signals = signals_for_authored_node(flow_kind, accumulated, topic, authored_node)
        if authored_node.message:
            query_text = (query_text + " " + authored_node.message).strip()
    else:
        synth_signals = _synth_signals(flow_kind, accumulated, topic=topic)

    from app.services.space_search_service import search_technical_passages

    retrieval = await search_technical_passages(
        session=session,
        space_id=space_id,
        query_text=query_text,
        user_id=user_id,
        k=settings.GUIDED_RETRIEVAL_K,
        signals=synth_signals,
        # Continuité : les documents d'ancre de la conversation biaisent le retrieval
        # de chaque étape (un tour guidé ne doit pas sauter de produit).
        anchor_document_ids=anchor_document_ids or None,
    )
    passages = retrieval.get("passages") or []

    # 5. Construire l'étape : identification produit (étape 0, déterministe), nœud validé
    # (authored), ou génération dynamique.
    step_index = len(path)
    pending_identification = (
        gsession.source_mode != "authored"
        and bool(accumulated.get("pending_product_identification"))
    )
    if pending_identification:
        step = _build_identification_step(session, passages, flow_kind)
    elif authored_node is not None:
        from app.services.authored_tree_service import routing_step_from_node

        step = routing_step_from_node(authored_node, passages)
    else:
        force_terminal = step_index >= settings.GUIDED_MAX_STEPS
        step = await generate_routing_step(
            flow_kind=flow_kind,
            topic=topic,
            passages=passages,
            path=path,
            step_index=step_index,
            force_terminal=force_terminal,
        )
        # Choix permanent « changer de produit » quand le produit a été fixé via l'étape 0 :
        # l'utilisateur peut signaler que le parcours s'est engagé sur le mauvais produit
        # et rouvrir l'identification. (Pas de bruit quand il a nommé le produit lui-même.)
        if (
            not step.is_terminal
            and accumulated.get("confirmed_product")
            and not any(c.value == RESTART_PRODUCT_VALUE for c in step.choices)
        ):
            step.choices.append(
                RoutingChoice(
                    label="Ce n'est pas mon produit / recommencer",
                    value=RESTART_PRODUCT_VALUE,
                    hint="",
                )
            )

    # 5. Enrichir le récap d'escalade à partir du parcours réel (étapes testées avant celle-ci)
    escalation_recap = step.escalation_recap.model_dump() if step.escalation_recap else None
    if step.is_terminal and step.step_type == "escalation":
        escalation_recap = await _build_escalation_recap(
            session,
            space_id=space_id,
            user_id=user_id,
            topic=topic,
            prior_path=path,
            step=step,
        )

    sources = _build_sources(session, passages)

    # 6. Construire le payload d'étape (SSE + persistance message assistant)
    step_payload: Dict[str, Any] = {
        "step_type": step.step_type,
        "message": step.message,
        "choices": [c.model_dump() for c in step.choices],
        "cited_pages": step.cited_pages,
        "is_terminal": step.is_terminal,
        "escalation_recap": escalation_recap,
        "flow_kind": flow_kind,
        "topic": topic,
        "step_index": step_index,
    }

    # 7. Ajouter la nouvelle étape (en attente de réponse) au parcours
    path.append(
        {
            "step_index": step_index,
            "step_type": step.step_type,
            "message": step.message,
            "choices": step_payload["choices"],
            "cited_pages": step.cited_pages,
            "user_selection": None,
            "free_text": None,
            "observations": [],
            "node_key": (
                PRODUCT_IDENTIFICATION_NODE_KEY
                if pending_identification
                else (authored_node.node_key if authored_node is not None else None)
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    # 8. Persister la GuidedSession (réassignations explicites → détection JSON)
    gsession.path = path
    gsession.accumulated_signals = accumulated
    # Conserver le pointeur de nœud en mode authored (resolve_authored_node l'a mis à jour) ;
    # le réinitialiser pour le mode dynamique.
    if gsession.source_mode != "authored":
        gsession.current_node_key = None
    gsession.updated_at = datetime.utcnow()
    if step.is_terminal:
        gsession.status = "escalated" if step.step_type == "escalation" else "resolved"
    else:
        gsession.status = "active"
    session.add(gsession)

    # 9. Mettre à jour le pointeur de routage dans Conversation.query_context
    conv = session.get(Conversation, conversation_id)
    if conv is not None:
        qc = dict(conv.query_context or {})
        if step.is_terminal:
            qc.pop("guided", None)
        else:
            qc["guided"] = {
                "active_session_id": gsession.id,
                "phase": "guided_active",
                "flow_kind": flow_kind,
                "topic": topic,
            }
        conv.query_context = qc
        conv.updated_at = datetime.utcnow()
        session.add(conv)

    session.commit()
    session.refresh(gsession)

    logger.info(
        "[guided_flow] tour terminé — session=%s flow=%s step_index=%s type=%s terminal=%s passages=%d",
        gsession.id,
        flow_kind,
        step_index,
        step.step_type,
        step.is_terminal,
        len(passages),
    )

    return GuidedTurnResult(
        step=step_payload,
        guided_session_id=gsession.id,
        sources=sources,
        message_text=step.message,
        is_terminal=step.is_terminal,
    )
