"""
Service de description d'images : OpenAI (async) et Mistral Pixtral (sync) pour le RAG.

Les descriptions Pixtral enrichissent les chunks feuilles « picture » après Docling.
"""

import base64
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_CHUNK_LIKE = Any  # NoteChunk | DocumentChunk (SQLModel, champs homogènes)


def _technical_vision_prompt_parts(context: str = "", caption: str = "") -> list[str]:
    parts = [
        "Décris ce schéma, graphique ou image technique de manière détaillée et structurée.",
        "Inclus :",
        "- Les éléments visuels principaux (formes, connexions, flux, axes, légendes)",
        "- Le texte visible dans l'image (OCR factuel)",
        "- Les relations entre les éléments et l'idée illustrée",
    ]
    if context:
        parts.append(f"\nContexte du document (section) : {context}")
    if caption:
        parts.append(f"\nLégende ou titre fourni par le document : {caption}")
    parts.append("\nRéponds en français, de façon concise mais complète, pour faciliter la recherche sémantique.")
    return parts


def _build_visual_neighbor_context(
    leaf: _CHUNK_LIKE,
    chunks: Sequence[_CHUNK_LIKE],
    window: int = 2,
    max_chars: int = 900,
) -> str:
    """
    Construit un contexte local autour d'un chunk image à partir des chunks voisins.
    Le but est d'éviter une analyse "hors sol" de l'image.
    """
    parent_node_id = getattr(leaf, "parent_node_id", None)
    leaf_index = getattr(leaf, "chunk_index", None)
    if parent_node_id is None or leaf_index is None:
        return ""

    sibling_candidates: list[_CHUNK_LIKE] = []
    for c in chunks:
        if not getattr(c, "is_leaf", True):
            continue
        if getattr(c, "parent_node_id", None) != parent_node_id:
            continue
        cidx = getattr(c, "chunk_index", None)
        if not isinstance(cidx, int):
            continue
        if abs(cidx - leaf_index) <= window and cidx != leaf_index:
            sibling_candidates.append(c)

    sibling_candidates.sort(key=lambda c: c.chunk_index)
    snippets: list[str] = []
    total = 0
    for sib in sibling_candidates:
        raw = (getattr(sib, "content", "") or "").strip()
        if not raw:
            continue
        meta = dict(getattr(sib, "metadata_json", {}) or {})
        label = str(meta.get("label") or "texte").strip()
        excerpt = raw[:260]
        if len(raw) > 260:
            excerpt += "..."
        block = f"[voisin:{label}] {excerpt}"
        if total + len(block) > max_chars:
            break
        snippets.append(block)
        total += len(block)

    return "\n".join(snippets)


def describe_image_with_mistral_sync(
    image_path: str,
    context: str = "",
    caption: str = "",
) -> Optional[str]:
    """
    Appel synchrone à l'API Mistral Chat Completions (Pixtral) pour décrire une image.
    """
    if not settings.MISTRAL_API_KEY:
        logger.warning("MISTRAL_API_KEY non configurée, analyse Pixtral ignorée")
        return None

    path = Path(image_path)
    if not path.exists():
        logger.error("Image non trouvée: %s", image_path)
        return None

    try:
        with open(path, "rb") as f:
            image_data = f.read()
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        suffix = path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/png")

        prompt = "\n".join(_technical_vision_prompt_parts(context, caption))
        data_uri = f"data:{mime_type};base64,{image_b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                ],
            }
        ]

        base_url = (settings.MISTRAL_BASE_URL or "https://api.mistral.ai").rstrip("/")
        model = settings.VISION_MODEL or "pixtral-12b-2409"
        max_tokens = getattr(settings, "VISION_MAX_TOKENS", 1500) or 1500

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                },
            )

        if response.status_code != 200:
            logger.error(
                "Erreur API Mistral vision (%d): %s",
                response.status_code,
                response.text[:2000],
            )
            return None

        data = response.json()
        choice0 = (data.get("choices") or [{}])[0]
        msg = choice0.get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text") or "")
            description = "\n".join(text_parts).strip()
        elif isinstance(content, str):
            description = content.strip()
        else:
            description = ""

        if not description:
            logger.warning("Réponse Pixtral vide pour %s", image_path)
            return None

        logger.debug(
            "Pixtral OK: %s (%d caractères)",
            image_path,
            len(description),
        )
        return description

    except httpx.TimeoutException:
        logger.error("Timeout Pixtral pour l'image: %s", image_path)
        return None
    except Exception as e:
        logger.error(
            "Erreur Pixtral pour l'image %s: %s",
            image_path,
            e,
            exc_info=True,
        )
        return None


def _meta_page(meta: dict) -> int:
    p = meta.get("page_no")
    if p is None:
        return 10**9
    try:
        return int(p)
    except (TypeError, ValueError):
        return 10**9


def _image_sort_key(img: dict) -> tuple:
    p = img.get("page_no")
    try:
        pn = int(p) if p is not None else 10**9
    except (TypeError, ValueError):
        pn = 10**9
    bbox = img.get("bbox") or {}
    try:
        top = float(bbox.get("t", 0))
    except (TypeError, ValueError):
        top = 0.0
    idx = img.get("index")
    try:
        iidx = int(idx) if idx is not None else 0
    except (TypeError, ValueError):
        iidx = 0
    return (pn, top, iidx)


def _pair_visual_leaves_with_images(
    picture_leaves: Sequence[_CHUNK_LIKE],
    images_info: list[dict],
) -> list[tuple[_CHUNK_LIKE, dict]]:
    """Apparie feuilles picture/figure et images Docling par page (ordre chunk_index vs bbox.t)."""
    leaves_by_page: dict[int, list] = defaultdict(list)
    for leaf in picture_leaves:
        meta = dict(leaf.metadata_json or {})
        pg = _meta_page(meta)
        leaves_by_page[pg].append(leaf)

    for pg in leaves_by_page:
        leaves_by_page[pg].sort(key=lambda c: (c.chunk_index, c.node_id or ""))

    imgs_by_page: dict[int, list] = defaultdict(list)
    for img in images_info:
        k = _image_sort_key(img)
        pg = k[0]
        imgs_by_page[pg].append(img)

    for pg in imgs_by_page:
        imgs_by_page[pg].sort(key=_image_sort_key)

    all_pages = sorted(set(leaves_by_page.keys()) | set(imgs_by_page.keys()))
    pairs: list[tuple[_CHUNK_LIKE, dict]] = []

    for pg in all_pages:
        L = leaves_by_page.get(pg, [])
        I = imgs_by_page.get(pg, [])
        n = min(len(L), len(I))
        for i in range(n):
            pairs.append((L[i], I[i]))
        if len(L) != len(I):
            logger.warning(
                "Appariement visuel page %s: %d feuilles picture vs %d images Docling "
                "(appariement 1:1 partiel)",
                pg if pg < 10**9 else "inconnue",
                len(L),
                len(I),
            )
    return pairs


def rebuild_docling_parent_chunk_contents(chunks: Sequence[_CHUNK_LIKE]) -> None:
    """Recalcule content/text des parents à partir des feuilles enrichies."""
    children_by_parent: dict[str, list] = defaultdict(list)
    parents_by_id: dict[str, _CHUNK_LIKE] = {}

    for c in chunks:
        if getattr(c, "is_leaf", True):
            pid = getattr(c, "parent_node_id", None)
            if pid:
                children_by_parent[pid].append(c)
        else:
            nid = getattr(c, "node_id", None)
            if nid:
                parents_by_id[nid] = c

    for pid, parent in parents_by_id.items():
        kids = children_by_parent.get(pid, [])
        kids.sort(key=lambda x: (x.chunk_index, x.node_id or ""))
        merged = "\n\n".join(
            (k.content or "").strip() for k in kids if (k.content or "").strip()
        )
        if merged:
            parent.content = merged
            parent.text = merged


def enrich_visual_chunks_with_pixtral(
    chunks: Sequence[_CHUNK_LIKE],
    images_info: Optional[list[dict]],
) -> None:
    """
    Enrichit les feuilles Docling « picture » / « figure » avec une description Pixtral (Mistral).

    Mutate chunk.content, chunk.text, chunk.metadata_json / metadata_ en place.
    """
    if not images_info:
        return
    if not getattr(settings, "MULTIMODAL_ENABLED", False):
        return
    if not settings.MISTRAL_API_KEY:
        logger.info("MULTIMODAL activé mais pas de MISTRAL_API_KEY — skip Pixtral")
        return

    picture_leaves: list = []
    for c in chunks:
        if not getattr(c, "is_leaf", True):
            continue
        meta = dict(c.metadata_json or {})
        if meta.get("content_type") == "table_row":
            continue
        lbl = meta.get("label")
        if lbl not in ("picture", "figure"):
            continue
        picture_leaves.append(c)

    if not picture_leaves:
        return

    pairs = _pair_visual_leaves_with_images(picture_leaves, images_info)
    cap = getattr(settings, "VISION_MAX_IMAGES_PER_DOCUMENT", None)
    if cap is not None and cap > 0 and len(pairs) > cap:
        logger.info(
            "VISION_MAX_IMAGES_PER_DOCUMENT=%s — analyse de %d/%d paires",
            cap,
            cap,
            len(pairs),
        )
        pairs = pairs[:cap]

    model = settings.VISION_MODEL or "pixtral-12b-2409"
    ts = datetime.now(timezone.utc).isoformat()

    for leaf, img in pairs:
        path = img.get("path")
        if not path:
            continue
        meta = dict(leaf.metadata_json or {})
        context = (meta.get("parent_heading") or meta.get("heading") or "").strip()
        if meta.get("figure_title"):
            context = f"{context} | figure: {meta.get('figure_title')}".strip(" |")
        local_neighbor_context = _build_visual_neighbor_context(leaf, chunks)
        if local_neighbor_context:
            context = (
                f"{context}\nContexte textuel voisin (même section):\n"
                f"{local_neighbor_context}"
            ).strip()
        raw_cap = img.get("caption") or meta.get("caption") or ""
        caption = str(raw_cap).strip() if raw_cap is not None else ""

        description = describe_image_with_mistral_sync(
            image_path=path,
            context=context,
            caption=caption,
        )
        if not description:
            continue

        block = f"\n\n## Analyse visuelle (Pixtral)\n{description.strip()}"
        base = (leaf.content or "").rstrip()
        leaf.content = f"{base}{block}".strip()
        leaf.text = leaf.content

        meta["vision_provider"] = "mistral"
        meta["vision_model"] = model
        meta["vision_enriched_at"] = ts
        meta["image_path"] = str(Path(path).name)
        leaf.metadata_json = meta
        leaf.metadata_ = meta

        logger.info(
            "Chunk feuille enrichi Pixtral (node_id=%s, image=%s)",
            meta.get("node_id") or leaf.node_id,
            Path(path).name,
        )

    rebuild_docling_parent_chunk_contents(chunks)


async def describe_image(
    image_path: str,
    context: str = "",
    caption: str = "",
) -> Optional[str]:
    """
    Génère une description textuelle d'une image avec GPT-4o Vision.
    
    Args:
        image_path: Chemin vers le fichier image
        context: Contexte du document (ex: section/heading parent)
        caption: Légende existante de l'image si disponible
        
    Returns:
        Description textuelle de l'image ou None en cas d'erreur
    """
    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY non configurée, description d'image ignorée")
        return None
    
    path = Path(image_path)
    if not path.exists():
        logger.error("Image non trouvée: %s", image_path)
        return None
    
    try:
        with open(path, "rb") as f:
            image_data = f.read()
        
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        
        # Déterminer le type MIME
        suffix = path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/png")
        
        # Construire le prompt
        prompt_parts = [
            "Décris ce schéma ou cette image technique de manière détaillée et structurée.",
            "Inclus :",
            "- Les éléments visuels principaux (formes, connexions, flux)",
            "- Le texte visible dans l'image",
            "- Les relations entre les différents éléments",
            "- L'objectif ou le concept illustré",
        ]
        
        if context:
            prompt_parts.append(f"\nContexte du document : {context}")
        
        if caption:
            prompt_parts.append(f"\nLégende existante : {caption}")
        
        prompt_parts.append("\nRéponds en français de manière concise mais complète.")
        
        prompt = "\n".join(prompt_parts)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]
        
        vision_model = getattr(settings, "VISION_MODEL", "gpt-4o")
        
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": vision_model,
                    "messages": messages,
                    "max_tokens": getattr(settings, "VISION_MAX_TOKENS", 1500),
                    "temperature": 0.3,
                },
            )
            
            if response.status_code != 200:
                logger.error(
                    "Erreur API OpenAI Vision (%d): %s",
                    response.status_code,
                    response.text,
                )
                return None
            
            data = response.json()
            description = data["choices"][0]["message"]["content"]
            
            logger.debug(
                "Image décrite avec succès: %s (%d caractères)",
                image_path,
                len(description),
            )
            
            return description
            
    except httpx.TimeoutException:
        logger.error("Timeout lors de la description de l'image: %s", image_path)
        return None
    except Exception as e:
        logger.error(
            "Erreur lors de la description de l'image %s: %s",
            image_path,
            e,
            exc_info=True,
        )
        return None


async def describe_images_batch(
    images_info: list,
    context: str = "",
) -> list:
    """
    Décrit plusieurs images en séquence.
    
    Args:
        images_info: Liste de dictionnaires avec 'path' et optionnellement 'caption'
        context: Contexte global du document
        
    Returns:
        Liste des images_info avec le champ 'description' ajouté
    """
    results = []
    
    for img in images_info:
        image_path = img.get("path")
        caption = img.get("caption", "")
        
        description = await describe_image(
            image_path=image_path,
            context=context,
            caption=caption,
        )
        
        img_with_desc = dict(img)
        img_with_desc["description"] = description
        results.append(img_with_desc)
        
        if description:
            logger.info(
                "✅ Image décrite: %s",
                Path(image_path).name if image_path else "unknown",
            )
    
    return results
