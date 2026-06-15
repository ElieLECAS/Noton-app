"""
Reranker vision LLM pour les pages candidates ColPali.

ColPali a un bon recall mais une precision perfectible : les top_k pages
contiennent souvent du bruit. Comme l'indexation est vision-only (le texte des
pages est un placeholder), un reranker texte n'a aucun signal exploitable.

Ce service rend les pages candidates en PNG (cache partage avec le chat) et
demande a un LLM vision (Mistral Small 4 par defaut) de noter la pertinence de
chaque page vis-a-vis de la question, en un seul appel batch. Les pages sous le
seuil sont ecartees avant d'etre transmises au LLM de generation.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import Dict, List, Optional

from llama_index.core.schema import NodeWithScore
from sqlmodel import Session

from app.config import settings
from app.models.document import Document
from app.services import mistral_service
from app.services.multimodal_page_service import (
    _parse_json_with_repair,
    render_page_png_cached,
)

logger = logging.getLogger(__name__)

# L'API Mistral accepte au maximum 8 images par requete.
_MAX_IMAGES_PER_CALL = 8

_RERANK_SYSTEM_PROMPT = (
    "Tu es un evaluateur de pertinence pour un systeme de recherche documentaire. "
    "On te donne une question d'utilisateur et plusieurs pages de documents techniques "
    "(sous forme d'images numerotees Page 1, Page 2, ...). "
    "Pour chaque page, evalue a quel point elle est UTILE pour repondre precisement a la question, "
    "sur une echelle de 0 a 5 :\n"
    "0 = totalement hors-sujet / aucun rapport\n"
    "1-2 = sujet proche mais ne repond pas a la question (bruit)\n"
    "3 = contient des elements partiellement utiles\n"
    "4-5 = contient directement l'information demandee\n\n"
    "Sois strict : une page qui parle du meme produit mais pas du detail demande n'est PAS pertinente. "
    "Reponds UNIQUEMENT avec un objet JSON de la forme "
    '{"scores": {"1": <int>, "2": <int>, ...}} sans aucun texte additionnel.'
)


def _node_page_ref(node_with_score: NodeWithScore) -> Optional[Dict]:
    """Extrait (document_id, page_no) d'un noeud, ou None si indisponible."""
    meta = dict(node_with_score.node.metadata or {})
    doc_id = meta.get("document_id")
    page_no = meta.get("page_no") or meta.get("page_start")
    if doc_id is None or page_no is None:
        return None
    try:
        return {"document_id": int(doc_id), "page_no": int(page_no)}
    except (TypeError, ValueError):
        return None


def _render_candidate_png(
    session: Session, doc_id: int, page_no: int, dpi: int
) -> Optional[str]:
    """Rend une page candidate en PNG base64 (avec cache). None si echec."""
    doc = session.get(Document, doc_id)
    if not doc or not doc.source_file_path or not os.path.exists(doc.source_file_path):
        logger.warning(
            "[vision_rerank] Document %s introuvable sur disque (page %s)",
            doc_id,
            page_no,
        )
        return None
    try:
        png = render_page_png_cached(doc.source_file_path, page_no, dpi)
        return base64.b64encode(png).decode("utf-8")
    except Exception as exc:
        logger.warning(
            "[vision_rerank] Echec rendu page %s du document %s : %s",
            page_no,
            doc_id,
            exc,
        )
        return None


async def _score_batch(
    query_text: str, images_b64: List[str], model: str
) -> Optional[Dict[int, float]]:
    """Appelle le LLM vision sur un batch d'images numerotees. Retourne {index_1based: score}."""
    listing = "\n".join(f"- Page {i + 1}" for i in range(len(images_b64)))
    user_text = (
        f"Question de l'utilisateur :\n{query_text}\n\n"
        f"Pages a evaluer (dans l'ordre des images jointes) :\n{listing}\n\n"
        "Donne un score 0-5 a chaque page."
    )
    context = [
        {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
        {"role": "user", "content": user_text, "images": images_b64},
    ]
    try:
        result = await mistral_service.chat(
            message=user_text,
            model=model,
            context=context,
            temperature=0.0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        logger.warning("[vision_rerank] Appel LLM vision echoue : %s", exc)
        return None

    choice = ((result or {}).get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    try:
        parsed = _parse_json_with_repair(content)
    except Exception as exc:
        logger.warning("[vision_rerank] Parsing JSON echoue (%s) : %r", exc, content[:200])
        return None

    raw_scores = parsed.get("scores", parsed) if isinstance(parsed, dict) else {}
    scores: Dict[int, float] = {}
    if isinstance(raw_scores, dict):
        for key, value in raw_scores.items():
            try:
                scores[int(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return scores


async def rerank_pages_vision(
    session: Session,
    query_text: str,
    nodes: List[NodeWithScore],
    *,
    model: Optional[str] = None,
    min_score: Optional[float] = None,
    max_pages: Optional[int] = None,
    dpi: Optional[int] = None,
) -> List[NodeWithScore]:
    """
    Reranke les pages candidates ColPali via un LLM vision et filtre le bruit.

    Strategie robuste : en cas d'echec (rendu, appel LLM, parsing), on retourne
    les noeuds d'origine inchanges pour ne jamais casser le RAG.

    Le score de rerank (0-5) est ecrit dans node.metadata['rerank_score'].
    """
    if not nodes:
        return nodes

    model = model or settings.VISION_RERANK_MODEL
    min_score = settings.VISION_RERANK_MIN_SCORE if min_score is None else min_score
    max_pages = settings.VISION_RERANK_MAX_PAGES if max_pages is None else max_pages
    dpi = settings.VISION_RERANK_DPI if dpi is None else dpi

    # Rendu PNG des candidats (off-thread, cache partage)
    refs: List[Optional[Dict]] = [_node_page_ref(n) for n in nodes]

    async def _render(ref: Optional[Dict]) -> Optional[str]:
        if ref is None:
            return None
        return await asyncio.to_thread(
            _render_candidate_png, session, ref["document_id"], ref["page_no"], dpi
        )

    images = await asyncio.gather(*[_render(r) for r in refs])

    # Conserver uniquement les noeuds dont le PNG a pu etre rendu
    renderable = [(node, img) for node, img in zip(nodes, images) if img]
    if not renderable:
        logger.warning("[vision_rerank] Aucune page rendue, rerank ignore")
        return nodes

    # Scoring par batches de <= 8 images (limite API Mistral)
    scores_by_node: Dict[int, float] = {}
    any_success = False
    for start in range(0, len(renderable), _MAX_IMAGES_PER_CALL):
        batch = renderable[start : start + _MAX_IMAGES_PER_CALL]
        batch_scores = await _score_batch(query_text, [img for _, img in batch], model)
        if batch_scores is None:
            continue
        any_success = True
        for local_idx, (node, _img) in enumerate(batch):
            score = batch_scores.get(local_idx + 1)
            if score is not None:
                scores_by_node[id(node)] = score
                node.node.metadata["rerank_score"] = score

    if not any_success:
        logger.warning("[vision_rerank] Aucun batch score, rerank ignore")
        return nodes

    # Filtrage par seuil + tri (rerank_score desc, score ColPali en tie-break)
    kept: List[NodeWithScore] = []
    for node in nodes:
        score = scores_by_node.get(id(node))
        if score is None:
            # Page non notee (ex: non rendue) : on la conserve par prudence
            continue
        if score >= min_score:
            kept.append(node)

    # Garde-fou : si tout est filtre, garder la meilleure page notee
    if not kept:
        scored_nodes = [n for n in nodes if id(n) in scores_by_node]
        if scored_nodes:
            best = max(scored_nodes, key=lambda n: scores_by_node[id(n)])
            logger.info(
                "[vision_rerank] Tout filtre, on conserve la meilleure page (score=%.1f)",
                scores_by_node[id(best)],
            )
            kept = [best]
        else:
            return nodes

    kept.sort(
        key=lambda n: (scores_by_node.get(id(n), 0.0), n.score),
        reverse=True,
    )

    logger.info(
        "[vision_rerank] %d candidats -> %d retenus (modele=%s, seuil=%.1f)",
        len(nodes),
        len(kept[:max_pages]),
        model,
        min_score,
    )
    return kept[:max_pages]
