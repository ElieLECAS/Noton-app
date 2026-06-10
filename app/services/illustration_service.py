import os
import hashlib
import json
import logging
import base64
import io
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image

from app.config import settings
from app.services import mistral_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de filtrage des candidats
# ---------------------------------------------------------------------------
MIN_IMAGE_SIZE_PTS = 50        # Taille min (largeur ET hauteur) pour images raster (~1.8 cm)
MIN_CLUSTER_SIZE_PTS = 80      # Taille min pour clusters vectoriels (~2.8 cm)
MIN_DRAWING_RECT_PTS = 15      # Taille min pour un rect vectoriel individuel avant clustering
CLUSTER_DILATION_PTS = 30      # Dilatation pour le clustering spatial
MIN_ASPECT_RATIO = 0.08        # Ratio min(w,h)/max(w,h) pour éliminer les lignes
HEADER_ZONE_RATIO = 0.08       # Zone d'exclusion en haut (8% de la hauteur)
FOOTER_ZONE_RATIO = 0.06       # Zone d'exclusion en bas (6% de la hauteur)
MIN_RELATIVE_AREA = 0.02       # Surface min relative à la page (2%)
MAX_RELATIVE_AREA = 0.85       # Surface max relative à la page (85%)
MAX_FULLPAGE_RATIO = 0.95      # Seuil pour exclure les dessins couvrant quasi toute la page
DEDUP_AREA_TOLERANCE = 100     # Tolérance en pts² pour considérer deux candidats de même taille
CROP_DPI = 200                 # DPI pour le crop final
ANALYSIS_DPI = 150             # DPI pour l'image envoyée au modèle de vision
MIN_CROP_PX = 80               # Dimension minimum du crop final en pixels
MIN_PIXEL_VARIANCE = 50.0      # Variance minimum des pixels pour détecter une image non vide


def rect_width(r):
    return r[2] - r[0]

def rect_height(r):
    return r[3] - r[1]

def rect_area(r):
    return rect_width(r) * rect_height(r)

def rect_center_y(r):
    return (r[1] + r[3]) / 2.0

def rect_center_x(r):
    return (r[0] + r[2]) / 2.0

def dilate_rect(r, padding):
    import fitz
    return fitz.Rect(r[0] - padding, r[1] - padding, r[2] + padding, r[3] + padding)

def union_rect(r1, r2):
    import fitz
    return fitz.Rect(
        min(r1[0], r2[0]),
        min(r1[1], r2[1]),
        max(r1[2], r2[2]),
        max(r1[3], r2[3])
    )


def _is_in_header_footer(bbox, page_h: float) -> bool:
    """Vérifie si le centre vertical du candidat est dans la zone header ou footer."""
    cy = rect_center_y(bbox)
    header_limit = page_h * HEADER_ZONE_RATIO
    footer_limit = page_h * (1.0 - FOOTER_ZONE_RATIO)
    return cy < header_limit or cy > footer_limit


def _aspect_ratio(w: float, h: float) -> float:
    """Ratio min/max pour détecter les éléments filiformes (lignes décoratives)."""
    if max(w, h) == 0:
        return 0.0
    return min(w, h) / max(w, h)


def _significance_score(cand: Dict[str, Any], page_w: float, page_h: float) -> float:
    """
    Score de significativité d'un candidat visuel (0-1).
    Plus le score est élevé, plus le candidat est probablement un schéma/tableau important.
    """
    bbox = cand["bbox"]
    w = rect_width(bbox)
    h = rect_height(bbox)
    page_area = page_w * page_h

    # 1. Surface relative (poids fort : un gros visuel est probablement important)
    area_ratio = (w * h) / page_area if page_area > 0 else 0
    area_score = min(area_ratio / 0.3, 1.0)  # Saturé à 30% de la page

    # 2. Position : le centre de la page est plus probable pour un schéma que les marges
    cx_norm = rect_center_x(bbox) / page_w if page_w > 0 else 0.5
    cy_norm = rect_center_y(bbox) / page_h if page_h > 0 else 0.5
    # Distance au centre normalisée (0=centre, 1=coin)
    dist_center = ((cx_norm - 0.5) ** 2 + (cy_norm - 0.5) ** 2) ** 0.5
    position_score = max(0, 1.0 - dist_center * 1.5)

    # 3. Type : tables et images sont plus fiables que drawing_clusters
    type_weights = {"table": 1.0, "image": 0.9, "drawing_cluster": 0.7}
    type_score = type_weights.get(cand.get("type", ""), 0.5)

    # 4. Ratio d'aspect (un carré ou rectangle raisonnable > une bande étroite)
    aspect = _aspect_ratio(w, h)
    aspect_score = min(aspect / 0.3, 1.0)

    # Pondération finale
    score = (
        0.40 * area_score +
        0.20 * position_score +
        0.25 * type_score +
        0.15 * aspect_score
    )
    return round(score, 3)


def get_visual_candidates(pdf_path: str, page_no: int) -> List[Dict[str, Any]]:
    """
    Extrait les bboxes candidates (images matricielles, tableaux et dessins vectoriels groupés)
    présents sur une page de PDF à l'aide de PyMuPDF (fitz).
    
    Applique des filtres stricts pour éliminer le bruit (logos, icônes, lignes décoratives,
    en-têtes/pieds de page) et scorer chaque candidat par significativité.
    """
    import fitz

    logger.info(f"Extracting visual candidates from {pdf_path} page {page_no}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path}: {e}")
        return []

    if page_no < 1 or page_no > len(doc):
        doc.close()
        logger.warning(f"Page number {page_no} out of range for {pdf_path}")
        return []

    page = doc[page_no - 1]
    page_w, page_h = page.rect.width, page.rect.height
    page_area = page_w * page_h
    candidates = []

    # 1. Trouver les tableaux
    try:
        tables = page.find_tables()
        for idx, table in enumerate(tables):
            bbox = list(table.bbox)
            w, h = rect_width(bbox), rect_height(bbox)
            rel_area = (w * h) / page_area if page_area > 0 else 0

            # Filtres : taille minimale, pas dans header/footer, surface relative raisonnable
            if w < MIN_IMAGE_SIZE_PTS or h < MIN_IMAGE_SIZE_PTS:
                continue
            if _is_in_header_footer(bbox, page_h):
                continue
            if rel_area < MIN_RELATIVE_AREA or rel_area > MAX_RELATIVE_AREA:
                continue

            candidates.append({
                "type": "table",
                "bbox": bbox,
                "description": f"Tableau numéro {idx + 1} ({int(w)}×{int(h)} pts, {rel_area*100:.0f}% de la page)"
            })
    except Exception as e:
        logger.warning(f"Failed to find tables in fitz: {e}")

    # 2. Trouver les images matricielles (avec filtrage strict)
    try:
        images = page.get_image_info()
        img_idx = 0
        for img in images:
            bbox = img.get("bbox")
            if not bbox:
                continue

            w, h = rect_width(bbox), rect_height(bbox)
            rel_area = (w * h) / page_area if page_area > 0 else 0

            # Filtres stricts
            if w < MIN_IMAGE_SIZE_PTS or h < MIN_IMAGE_SIZE_PTS:
                continue
            if _aspect_ratio(w, h) < MIN_ASPECT_RATIO:
                continue
            if _is_in_header_footer(bbox, page_h):
                continue
            if rel_area < MIN_RELATIVE_AREA or rel_area > MAX_RELATIVE_AREA:
                continue

            img_idx += 1
            candidates.append({
                "type": "image",
                "bbox": list(bbox),
                "description": f"Image matricielle numéro {img_idx} ({int(w)}×{int(h)} pts, {rel_area*100:.0f}% de la page)"
            })
    except Exception as e:
        logger.warning(f"Failed to find image info in fitz: {e}")

    # 3. Regrouper (clustering) les dessins vectoriels proches
    try:
        drawings = page.get_drawings()
        drawing_rects = []
        for d in drawings:
            r = d.get("rect")
            if r and rect_width(r) > MIN_DRAWING_RECT_PTS and rect_height(r) > MIN_DRAWING_RECT_PTS:
                # Exclure les rectangles géants qui couvrent toute la page (arrière-plans)
                if rect_width(r) > page_w * MAX_FULLPAGE_RATIO and rect_height(r) > page_h * MAX_FULLPAGE_RATIO:
                    continue
                drawing_rects.append(r)

        clusters: List[fitz.Rect] = []
        for rect in drawing_rects:
            merged = False
            for i, c in enumerate(clusters):
                dilated_c = dilate_rect(c, CLUSTER_DILATION_PTS)
                f_rect = fitz.Rect(rect)
                if dilated_c.intersects(f_rect):
                    clusters[i] = union_rect(c, rect)
                    merged = True
                    break

            if not merged:
                clusters.append(fitz.Rect(rect))

        # Ne garder que les clusters significatifs avec filtres stricts
        drawing_idx = 1
        for c in clusters:
            w = rect_width(c)
            h = rect_height(c)
            rel_area = (w * h) / page_area if page_area > 0 else 0

            # Filtres stricts pour les clusters vectoriels
            if w < MIN_CLUSTER_SIZE_PTS or h < MIN_CLUSTER_SIZE_PTS:
                continue
            if _aspect_ratio(w, h) < MIN_ASPECT_RATIO:
                continue
            if _is_in_header_footer([c[0], c[1], c[2], c[3]], page_h):
                continue
            if rel_area < MIN_RELATIVE_AREA or rel_area > MAX_RELATIVE_AREA:
                continue

            candidates.append({
                "type": "drawing_cluster",
                "bbox": [c[0], c[1], c[2], c[3]],
                "description": f"Schéma vectoriel numéro {drawing_idx} ({int(w)}×{int(h)} pts, {rel_area*100:.0f}% de la page)"
            })
            drawing_idx += 1
    except Exception as e:
        logger.warning(f"Failed to process drawings in fitz: {e}")

    doc.close()

    # 4. Supprimer les doublons ou inclusions
    import fitz as fitz_mod
    unique_candidates = []
    for cand in candidates:
        r1 = fitz_mod.Rect(cand["bbox"])
        is_subsumed = False
        for other in candidates:
            if cand is other:
                continue
            r2 = fitz_mod.Rect(other["bbox"])
            if r2.contains(r1):
                if abs(rect_area(r2) - rect_area(r1)) < DEDUP_AREA_TOLERANCE:
                    continue  # Si de taille presque identique, on ne rejette pas
                is_subsumed = True
                break
        if not is_subsumed:
            unique_candidates.append(cand)

    # 5. Calculer le score de significativité et trier par score décroissant
    for c in unique_candidates:
        c["significance_score"] = _significance_score(c, page_w, page_h)

    unique_candidates.sort(key=lambda c: c["significance_score"], reverse=True)

    # Assigner un index final et arrondir les coordonnées pour la clarté du prompt
    for idx, c in enumerate(unique_candidates):
        c["index"] = idx
        c["bbox"] = [round(v, 1) for v in c["bbox"]]

    logger.info(
        f"Found {len(unique_candidates)} significant candidates on page {page_no} "
        f"(scores: {[c['significance_score'] for c in unique_candidates]})"
    )
    return unique_candidates


def has_significant_visuals(pdf_path: str, page_no: int) -> bool:
    """
    Vérification rapide (sans appel LLM) : la page contient-elle au moins
    un candidat visuel significatif après filtrage strict ?
    
    Utilisé par le chat router pour pré-filtrer les pages avant d'appeler
    le modèle de vision (coûteux).
    """
    candidates = get_visual_candidates(pdf_path, page_no)
    return len(candidates) > 0


async def determine_and_crop_illustration(
    query: str,
    answer: str,
    pdf_path: str,
    page_no: int,
    doc_title: str
) -> Optional[Dict[str, Any]]:
    """
    Rend la page sous forme d'image, demande au modèle de vision de sélectionner le visuel 
    le plus adapté pour illustrer la réponse, effectue le détourage et le met en cache.
    
    Améliorations v2 :
    - Prompt avec critères objectifs de pertinence et de rejet
    - Chain-of-thought pour améliorer la qualité de la décision
    - Padding adaptatif proportionnel à la taille du visuel
    - Validation du crop (dimensions min, image non vide)
    """
    from app.services.multimodal_page_service import render_pdf_page_png

    # 1. Récupérer les candidats (déjà filtrés et scorés)
    candidates = get_visual_candidates(pdf_path, page_no)
    if not candidates:
        logger.info(f"No significant visual candidates found on page {page_no}. Skipping illustration.")
        return None

    # 2. Rendre la page en PNG pour l'analyse visuelle
    try:
        png_bytes = render_pdf_page_png(pdf_path, page_no - 1, dpi=ANALYSIS_DPI)
        base64_image = base64.b64encode(png_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to render page {page_no} of {pdf_path}: {e}", exc_info=True)
        return None

    # 3. Récupérer les dimensions de la page
    import fitz
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_no - 1]
        width_pts, height_pts = page.rect.width, page.rect.height
        doc.close()
    except Exception:
        width_pts, height_pts = 595.0, 842.0  # Fallback A4

    # 4. Préparer le prompt pour le modèle de vision (v2 - avec critères stricts)
    candidates_desc = []
    for c in candidates:
        candidates_desc.append(
            f"- Élément #{c['index']}: Type: {c['type']}, "
            f"Description: {c['description']}, "
            f"Score de significativité: {c['significance_score']}, "
            f"Coordonnées [x0, y0, x1, y1]: {c['bbox']}"
        )
    candidates_text = "\n".join(candidates_desc)

    # Tronquer la réponse pour ne pas saturer le contexte du modèle
    answer_truncated = answer[:2000] + "..." if len(answer) > 2000 else answer

    prompt = f"""Tu es un expert en analyse visuelle de documents techniques. 
Tu reçois l'image d'une page de document, ainsi que la question de l'utilisateur et la réponse fournie par l'assistant.

**Question de l'utilisateur** : "{query}"

**Réponse de l'assistant** : "{answer_truncated}"

**Éléments visuels détectés sur cette page** (coordonnées en points, page = [0, 0, {width_pts:.0f}, {height_pts:.0f}]) :
{candidates_text}

---

## Ta mission

Tu dois décider si UN des éléments visuels ci-dessus est **directement pertinent** pour illustrer et enrichir la réponse textuelle de l'assistant.

## Critères de REJET (illustration_needed = false)

Tu DOIS rejeter et répondre `illustration_needed: false` si l'un de ces cas s'applique :
- La question est purement d'ordre général, théorique ou conversationnel (salutation, remerciement, question de fonctionnement)
- Les éléments visuels sont des **logos, en-têtes, pieds de page, numéros de page, bordures décoratives, icônes, flèches isolées ou rectangles de mise en page**
- Le visuel traite d'un **sujet différent** de ce qui est décrit dans la réponse (même s'il est dans le même domaine général)
- Le visuel n'ajoute **aucune information complémentaire** à ce que la réponse textuelle dit déjà — il ne fait que répéter sous forme visuelle un contenu trivial
- Le visuel est trop générique pour être utile (ex: une photo d'ambiance, un arrière-plan décoratif)

## Critères de SÉLECTION (illustration_needed = true)

Tu ne dois sélectionner un visuel QUE s'il remplit TOUS ces critères :
1. Il illustre un **élément spécifique** mentionné ou décrit dans la réponse (une mesure, un processus, une configuration, un composant, des données chiffrées)
2. Il apporte une **valeur ajoutée concrète** : le lecteur comprend mieux la réponse grâce au visuel
3. Il est suffisamment **lisible et complet** (pas un fragment coupé, pas une image floue)

## Format de réponse

Réponds UNIQUEMENT par un objet JSON valide. Tu dois d'abord raisonner brièvement (champ "reasoning") avant de décider.

Si un visuel est pertinent :
{{
  "reasoning": "Explication en 1-2 phrases de POURQUOI ce visuel illustre spécifiquement la réponse",
  "illustration_needed": true,
  "selected_candidate_index": <int ou null>,
  "custom_bbox": [<ymin>, <xmin>, <ymax>, <xmax>] ou null,
  "caption": "Légende claire et descriptive en français décrivant précisément le contenu du schéma/tableau"
}}

Si aucun visuel n'est pertinent :
{{
  "reasoning": "Explication en 1 phrase de pourquoi aucun visuel n'est retenu",
  "illustration_needed": false,
  "selected_candidate_index": null,
  "custom_bbox": null,
  "caption": ""
}}

Notes :
- "selected_candidate_index" : l'indice de l'élément dans la liste ci-dessus
- "custom_bbox" : uniquement si l'élément pertinent n'est pas correctement couvert par la liste, format [ymin, xmin, ymax, xmax] normalisé de 0 à 100 (pourcentages de la hauteur et largeur de la page)
- En cas de doute, préfère NE PAS illustrer plutôt que d'illustrer avec un visuel non pertinent
"""

    model_name = getattr(settings, "MULTIMODAL_EXTRACT_MODEL", "mistral-large-latest")
    logger.info(f"Sending page {page_no} image and prompt to {model_name} (candidates: {len(candidates)})...")

    try:
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": [base64_image]
            }
        ]
        
        response = await mistral_service.chat(
            message="",
            model=model_name,
            context=messages,
            response_format={"type": "json_object"}
        )
        
        choice = (response.get("choices") or [{}])[0]
        content = choice.get("message", {}).get("content", "").strip()
        decision = json.loads(content)
    except Exception as e:
        logger.error(f"Error calling vision API: {e}", exc_info=True)
        return None

    reasoning = decision.get("reasoning", "")
    logger.info(f"Vision model decision: illustration_needed={decision.get('illustration_needed')}, reasoning={reasoning}")

    if not decision.get("illustration_needed"):
        logger.info("Vision model decided that an illustration is not relevant.")
        return None

    # 5. Déterminer la bbox à découper (en fitz points)
    bbox = None
    selected_idx = decision.get("selected_candidate_index")
    custom_bbox = decision.get("custom_bbox")

    if selected_idx is not None and 0 <= selected_idx < len(candidates):
        bbox = candidates[selected_idx]["bbox"]
        logger.info(f"Using candidate #{selected_idx} bbox: {bbox}")
    elif custom_bbox and len(custom_bbox) == 4:
        # Convertir les coordonnées normalisées 0-100 en points fitz
        ymin, xmin, ymax, xmax = custom_bbox
        bbox = [
            (xmin * width_pts) / 100.0,
            (ymin * height_pts) / 100.0,
            (xmax * width_pts) / 100.0,
            (ymax * height_pts) / 100.0
        ]
        logger.info(f"Using custom bbox (converted from %): {bbox}")

    if not bbox:
        logger.warning("Vision model requested an illustration but failed to provide valid coordinates.")
        return None

    # 6. Effectuer le détourage avec padding adaptatif et validation
    try:
        high_res_bytes = render_pdf_page_png(pdf_path, page_no - 1, dpi=CROP_DPI)
        img = Image.open(io.BytesIO(high_res_bytes))

        scale = CROP_DPI / 72.0
        x0, y0, x1, y1 = bbox

        # Padding adaptatif : proportionnel à la taille du visuel
        visual_max_dim = max(rect_width(bbox), rect_height(bbox))
        padding = max(5.0, min(20.0, 0.03 * visual_max_dim))
        
        x0 = max(0.0, x0 - padding)
        y0 = max(0.0, y0 - padding)
        x1 = min(width_pts, x1 + padding)
        y1 = min(height_pts, y1 + padding)

        # Convertir en pixels
        px0 = int(x0 * scale)
        py0 = int(y0 * scale)
        px1 = int(x1 * scale)
        py1 = int(y1 * scale)

        # Clamp aux dimensions de l'image
        px0 = max(0, px0)
        py0 = max(0, py0)
        px1 = min(img.width, px1)
        py1 = min(img.height, py1)

        cropped_img = img.crop((px0, py0, px1, py1))

        # 7. Validation du crop
        crop_w, crop_h = cropped_img.size
        if crop_w < MIN_CROP_PX or crop_h < MIN_CROP_PX:
            logger.warning(
                f"Cropped image too small ({crop_w}×{crop_h} px, min {MIN_CROP_PX}). Discarding."
            )
            return None

        # Vérifier que l'image n'est pas quasi-vide (page blanche, rectangle uni)
        try:
            arr = np.array(cropped_img.convert("L"), dtype=np.float32)
            variance = float(np.var(arr))
            if variance < MIN_PIXEL_VARIANCE:
                logger.warning(
                    f"Cropped image has very low variance ({variance:.1f} < {MIN_PIXEL_VARIANCE}). "
                    "Likely a blank or solid-color region. Discarding."
                )
                return None
        except Exception as var_err:
            logger.debug(f"Could not compute pixel variance (non-critical): {var_err}")

        # 8. Sauvegarder dans le cache d'illustrations
        cache_dir = os.path.join(os.path.dirname(settings.LANCED_DB_DIR), "illustration_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Nom unique basé sur le hash MD5 du fichier, page et coordonnées
        key_str = f"{pdf_path}_{page_no}_{round(x0,1)}_{round(y0,1)}_{round(x1,1)}_{round(y1,1)}"
        filename_hash = hashlib.md5(key_str.encode("utf-8")).hexdigest()
        filename = f"crop_{filename_hash}.png"
        file_path = os.path.join(cache_dir, filename)

        cropped_img.save(file_path, format="PNG")
        logger.info(f"Cropped illustration cached: {file_path} ({crop_w}×{crop_h} px)")

        return {
            "url": f"/api/chats/illustrations/{filename}",
            "title": decision.get("caption") or "Schéma d'illustration",
            "page_no": page_no,
            "document_title": doc_title
        }

    except Exception as e:
        logger.error(f"Error during cropping or image saving: {e}", exc_info=True)
        return None
