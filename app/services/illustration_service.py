"""
Extraction d'illustration pour la réponse du chat — découpe ANCRÉE SUR LA RÉFÉRENCE.

Principe (refonte) :
  Le modèle de vision ne place plus le cadre (il régresse mal des coordonnées et
  confond deux coupes voisines). On positionne la découpe de façon DÉTERMINISTE sur
  le texte de la page :

  1. Ancre exacte : `page.search_for("6104")` → bbox pixel-précise du code cible.
  2. Labels frères : tous les codes de la page (motif ILLUSTRATION_REFERENCE_PATTERN).
  3. Extent du schéma : on attribue chaque trait/image au label de code le plus proche
     (Voronoï) et on garde l'union des traits possédés par la cible.
  4. Bornage : la fenêtre est coupée au point milieu vers chaque code frère.
  5. Contrôle anti-fuite : aucun code frère ne doit tomber dans le crop final.
  6. Garde-fou de lecture (vision, sur le crop seul) : confirme que la cible est
     présente et sujet principal. Ne peut que rejeter.

  Abstention stricte : sans code cible ancrable → aucune image.
"""
import os
import io
import re
import json
import base64
import hashlib
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from app.config import settings
from app.services import mistral_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de découpe
# ---------------------------------------------------------------------------
CROP_DPI = 200                 # DPI du crop final
MIN_CROP_PX = 80               # Dimension min du crop final (px)
MIN_PIXEL_VARIANCE = 50.0      # Variance min des pixels (écarte les zones vides)

MIN_DIAGRAM_PTS = 55.0         # Dimension min (w ET h) d'un schéma 2D valide (~1.9 cm)
MIN_OWNED_STROKES = 6          # Nb min de traits/visuels attribués à la cible
MIN_STROKE_PTS = 3.0           # Taille min d'un trait pour compter (élimine le bruit)
MAX_FULLPAGE_RATIO = 0.95      # Rect couvrant ~toute la page = arrière-plan, ignoré


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm_code(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).strip().lower()


def _reference_pattern() -> "re.Pattern":
    raw = getattr(settings, "ILLUSTRATION_REFERENCE_PATTERN", r"\b\d{3,5}[A-Za-z]?\b")
    try:
        return re.compile(raw)
    except re.error:
        logger.warning("[illustration] motif de référence invalide (%r), repli par défaut", raw)
        return re.compile(r"\b\d{3,5}[A-Za-z]?\b")


def _rect_center(r) -> Tuple[float, float]:
    return ((r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0)


def _overlaps(r, win) -> bool:
    """Chevauchement tolérant aux rectangles dégénérés (traits fins de largeur/hauteur nulle),
    pour lesquels fitz.Rect.intersects() renvoie False."""
    return not (r.x1 < win.x0 or r.x0 > win.x1 or r.y1 < win.y0 or r.y0 > win.y1)


def find_code_labels(page, pattern: "Optional[re.Pattern]" = None) -> List[Tuple[Any, str]]:
    """Retourne [(fitz.Rect, code)] pour chaque mot de la page matchant le motif de code."""
    import fitz

    if pattern is None:
        pattern = _reference_pattern()
    labels: List[Tuple[Any, str]] = []
    try:
        words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
    except Exception as exc:
        logger.debug("[illustration] get_text('words') échec: %s", exc)
        return labels
    for w in words:
        text = (w[4] or "").strip()
        if text and pattern.fullmatch(text):
            labels.append((fitz.Rect(w[0], w[1], w[2], w[3]), text))
    return labels


def codes_inside(crop_rect, code_labels: List[Tuple[Any, str]]) -> Set[str]:
    """Codes dont le centre du label tombe dans le crop."""
    out: Set[str] = set()
    for r, t in code_labels:
        cx, cy = _rect_center(r)
        if crop_rect.x0 <= cx <= crop_rect.x1 and crop_rect.y0 <= cy <= crop_rect.y1:
            out.add(t)
    return out


def _content_rects(page, win) -> List[Any]:
    """Rects de contenu visuel (traits vectoriels + images raster) intersectant la fenêtre,
    hors arrière-plans pleine page et hors traits microscopiques."""
    import fitz

    page_w, page_h = page.rect.width, page.rect.height
    rects: List[Any] = []

    try:
        for d in page.get_drawings():
            r = d.get("rect")
            if not r:
                continue
            rr = fitz.Rect(r)
            if rr.width < MIN_STROKE_PTS and rr.height < MIN_STROKE_PTS:
                continue
            if rr.width > page_w * MAX_FULLPAGE_RATIO and rr.height > page_h * MAX_FULLPAGE_RATIO:
                continue
            if _overlaps(rr, win):
                rects.append(rr)
    except Exception as exc:
        logger.debug("[illustration] get_drawings échec: %s", exc)

    try:
        for img in page.get_image_info():
            b = img.get("bbox")
            if not b:
                continue
            rr = fitz.Rect(b)
            if rr.width > page_w * MAX_FULLPAGE_RATIO and rr.height > page_h * MAX_FULLPAGE_RATIO:
                continue
            if _overlaps(rr, win):
                rects.append(rr)
    except Exception as exc:
        logger.debug("[illustration] get_image_info échec: %s", exc)

    return rects


def build_anchored_crop(page, target: str, anchor_rect, code_labels: List[Tuple[Any, str]]):
    """
    Construit le rectangle de découpe (en points PDF) centré sur l'ancre du code `target`,
    borné par les codes frères et limité à l'extent du schéma possédé par la cible.
    Retourne un fitz.Rect, ou None si rien de fiable n'est isolable (abstention).
    """
    import fitz

    page_w, page_h = page.rect.width, page.rect.height
    max_w = page_w * settings.ILLUSTRATION_MAX_WINDOW_RATIO
    max_h = page_h * settings.ILLUSTRATION_MAX_WINDOW_RATIO
    acx, acy = _rect_center(anchor_rect)

    # 1) Fenêtre initiale centrée sur l'ancre, plafonnée
    win = fitz.Rect(
        max(0.0, acx - max_w / 2.0),
        max(0.0, acy - max_h / 2.0),
        min(page_w, acx + max_w / 2.0),
        min(page_h, acy + max_h / 2.0),
    )

    # 2) Bornage par les frères (codes != target) : coupe au point milieu, côté du frère
    for r, t in code_labels:
        if _norm_code(t) == _norm_code(target):
            continue
        scx, scy = _rect_center(r)
        if not (win.x0 <= scx <= win.x1 and win.y0 <= scy <= win.y1):
            continue  # frère déjà hors fenêtre : aucune contrainte
        dx, dy = scx - acx, scy - acy
        if abs(dy) >= abs(dx):
            mid = (acy + scy) / 2.0
            if dy >= 0:
                win.y1 = min(win.y1, mid)
            else:
                win.y0 = max(win.y0, mid)
        else:
            mid = (acx + scx) / 2.0
            if dx >= 0:
                win.x1 = min(win.x1, mid)
            else:
                win.x0 = max(win.x0, mid)

    if win.is_empty or win.width <= 1.0 or win.height <= 1.0:
        return None

    # 3) Voronoï : attribuer chaque rect de contenu au label de code le plus proche ;
    #    garder ceux dont le label le plus proche est la cible.
    labels_for_nearest = list(code_labels) + [(anchor_rect, target)]
    centers = [(_rect_center(r), _norm_code(t)) for r, t in labels_for_nearest]
    target_norm = _norm_code(target)

    owned: List[Any] = []
    for rr in _content_rects(page, win):
        cx, cy = _rect_center(rr)
        nearest_text = None
        nearest_d = None
        for (lx, ly), t in centers:
            d = (cx - lx) ** 2 + (cy - ly) ** 2
            if nearest_d is None or d < nearest_d:
                nearest_d = d
                nearest_text = t
        if nearest_text == target_norm:
            owned.append(rr)

    if len(owned) < MIN_OWNED_STROKES:
        logger.info(
            "[illustration] '%s' : %d trait(s) possédé(s) (<%d) → abstention",
            target, len(owned), MIN_OWNED_STROKES,
        )
        return None

    # 4) Extent du schéma = union des traits possédés ∪ label de l'ancre, clampé à la fenêtre
    diagram = fitz.Rect(owned[0])
    for rr in owned[1:]:
        diagram |= rr
    diagram |= anchor_rect
    diagram &= win

    if diagram.width < MIN_DIAGRAM_PTS or diagram.height < MIN_DIAGRAM_PTS:
        logger.info(
            "[illustration] '%s' : schéma trop petit (%.0fx%.0f pts) → abstention",
            target, diagram.width, diagram.height,
        )
        return None

    # 5) Padding
    pad = settings.ILLUSTRATION_ANCHOR_PADDING_PTS
    crop = fitz.Rect(
        max(0.0, diagram.x0 - pad),
        max(0.0, diagram.y0 - pad),
        min(page_w, diagram.x1 + pad),
        min(page_h, diagram.y1 + pad),
    )

    # 6) Contrôle anti-fuite : aucun code frère ne doit tomber dans le crop
    leaked = codes_inside(crop, code_labels) - {target}
    # tolérer d'autres occurrences du même code, rejeter tout AUTRE code
    leaked = {c for c in leaked if _norm_code(c) != target_norm}
    if leaked:
        # rétrécir au schéma sans padding et re-tester
        crop = fitz.Rect(diagram)
        leaked = {c for c in (codes_inside(crop, code_labels) - {target}) if _norm_code(c) != target_norm}
        if leaked:
            logger.info("[illustration] '%s' : code(s) frère(s) %s dans le crop → abstention", target, leaked)
            return None

    return crop


# ---------------------------------------------------------------------------
# Rendu / validation / cache du crop
# ---------------------------------------------------------------------------
def _make_crop_image(pdf_path: str, page_no: int, crop_rect) -> Optional[Image.Image]:
    """Rend la page en haute résolution, découpe le rect, valide taille et variance."""
    from app.services.multimodal_page_service import render_pdf_page_png

    try:
        high_res = render_pdf_page_png(pdf_path, page_no - 1, dpi=CROP_DPI)
        img = Image.open(io.BytesIO(high_res))
    except Exception as exc:
        logger.error("[illustration] rendu haute résolution échoué: %s", exc, exc_info=True)
        return None

    scale = CROP_DPI / 72.0
    px0 = max(0, int(crop_rect.x0 * scale))
    py0 = max(0, int(crop_rect.y0 * scale))
    px1 = min(img.width, int(crop_rect.x1 * scale))
    py1 = min(img.height, int(crop_rect.y1 * scale))

    if (px1 - px0) < MIN_CROP_PX or (py1 - py0) < MIN_CROP_PX:
        logger.info("[illustration] crop trop petit (%dx%d px) → rejet", px1 - px0, py1 - py0)
        return None

    cropped = img.crop((px0, py0, px1, py1))
    try:
        arr = np.array(cropped.convert("L"), dtype=np.float32)
        if float(np.var(arr)) < MIN_PIXEL_VARIANCE:
            logger.info("[illustration] crop quasi-vide (variance faible) → rejet")
            return None
    except Exception as exc:
        logger.debug("[illustration] variance non calculable (non bloquant): %s", exc)

    return cropped


def _cache_dir() -> str:
    cache_dir = os.path.join(os.path.dirname(settings.LANCED_DB_DIR), "illustration_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _cache_crop_image(pdf_path: str, page_no: int, crop_rect, image: Image.Image) -> str:
    """Sauvegarde le crop et retourne le nom de fichier (clé déterministe)."""
    cache_dir = _cache_dir()
    key = (
        f"{pdf_path}_{page_no}_"
        f"{round(crop_rect.x0, 1)}_{round(crop_rect.y0, 1)}_"
        f"{round(crop_rect.x1, 1)}_{round(crop_rect.y1, 1)}"
    )
    fname = f"crop_{hashlib.md5(key.encode('utf-8')).hexdigest()}.png"
    image.save(os.path.join(cache_dir, fname), format="PNG")
    logger.info("[illustration] crop sauvegardé: %s (%dx%d px)", fname, image.width, image.height)
    return fname


# ---------------------------------------------------------------------------
# Garde-fou de lecture (vision sur le crop seul)
# ---------------------------------------------------------------------------
_GATE_PROMPT = """Tu reçois une petite image découpée d'une page de document technique (menuiserie).
On cherche à illustrer le profilé / la référence «{target}».

Analyse UNIQUEMENT cette image et réponds par un JSON strict :
{{
  "visible_codes": ["<codes/références lisibles dans l'image>"],
  "present": <true si «{target}» est lisible dans l'image>,
  "dominant": <true si «{target}» est le sujet principal (le schéma/la coupe montré), pas une mention secondaire>
}}

Règle : si un AUTRE code de profilé domine l'image, "dominant" doit être false."""


async def _vision_read_gate(crop_image: Image.Image, target: str) -> bool:
    """Confirme via vision que le crop montre bien `target` comme sujet principal.
    Échec API ou doute → False (abstention prudente, conforme à « rien plutôt que faux »)."""
    buf = io.BytesIO()
    crop_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    try:
        response = await mistral_service.chat(
            message="",
            model=getattr(settings, "MULTIMODAL_EXTRACT_MODEL", "mistral-large-latest"),
            context=[{"role": "user", "content": _GATE_PROMPT.format(target=target), "images": [b64]}],
            response_format={"type": "json_object"},
        )
        content = (response.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        data = json.loads(content)
    except Exception as exc:
        logger.warning("[illustration] garde-fou vision en échec (%s) → rejet prudent", exc)
        return False

    present = bool(data.get("present"))
    dominant = bool(data.get("dominant"))
    target_norm = _norm_code(target)
    others = [
        c for c in (data.get("visible_codes") or [])
        if _norm_code(str(c)) and _norm_code(str(c)) != target_norm
    ]
    ok = present and dominant
    logger.info(
        "[illustration] garde-fou '%s' : present=%s dominant=%s autres=%s → %s",
        target, present, dominant, others, "OK" if ok else "REJET",
    )
    return ok


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
async def extract_reference_illustration(
    *,
    targets: List[str],
    pdf_path: str,
    page_no: int,
    doc_title: str,
) -> Optional[Dict[str, Any]]:
    """
    Tente de produire une illustration ancrée sur l'un des codes `targets` (ordre de priorité)
    présent comme texte sur la page `page_no` de `pdf_path`.

    Retourne {url, title, page_no, document_title, reference} ou None (abstention).
    """
    import fitz

    if not targets:
        return None

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.error("[illustration] ouverture PDF échouée %s: %s", pdf_path, exc)
        return None

    try:
        if page_no < 1 or page_no > len(doc):
            logger.debug("[illustration] page %s hors limites pour %s", page_no, pdf_path)
            return None

        page = doc[page_no - 1]
        pattern = _reference_pattern()
        code_labels = find_code_labels(page, pattern)

        for target in targets:
            if not target or not target.strip():
                continue
            try:
                anchors = page.search_for(target)
            except Exception as exc:
                logger.debug("[illustration] search_for(%r) échec: %s", target, exc)
                continue
            if not anchors:
                continue

            logger.info(
                "[illustration] cible '%s' : %d ancre(s) sur page %d de '%s'",
                target, len(anchors), page_no, doc_title,
            )
            for anchor in anchors:
                crop_rect = build_anchored_crop(page, target, anchor, code_labels)
                if crop_rect is None:
                    continue

                crop_img = _make_crop_image(pdf_path, page_no, crop_rect)
                if crop_img is None:
                    continue

                if settings.ILLUSTRATION_VISION_GATE_ENABLED:
                    if not await _vision_read_gate(crop_img, target):
                        continue

                fname = _cache_crop_image(pdf_path, page_no, crop_rect, crop_img)
                logger.info(
                    "[illustration] illustration retenue: '%s' page %d de '%s'",
                    target, page_no, doc_title,
                )
                return {
                    "url": f"/api/chats/illustrations/{fname}",
                    "title": f"Profilé {target}",
                    "page_no": page_no,
                    "document_title": doc_title,
                    "reference": target,
                }
    finally:
        doc.close()

    return None
