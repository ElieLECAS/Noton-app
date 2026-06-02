import os
import hashlib
import json
import logging
import base64
import io
from typing import List, Dict, Any, Optional
from PIL import Image

from app.config import settings
from app.services import mistral_service

logger = logging.getLogger(__name__)

def rect_width(r):
    return r[2] - r[0]

def rect_height(r):
    return r[3] - r[1]

def rect_area(r):
    return rect_width(r) * rect_height(r)

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

def get_visual_candidates(pdf_path: str, page_no: int) -> List[Dict[str, Any]]:
    """
    Extrait les bboxes candidates (images matricielles, tableaux et dessins vectoriels groupés)
    présents sur une page de PDF à l'aide de PyMuPDF (fitz).
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
    candidates = []

    # 1. Trouver les tableaux
    try:
        tables = page.find_tables()
        for idx, table in enumerate(tables):
            candidates.append({
                "type": "table",
                "bbox": list(table.bbox),
                "description": f"Tableau numéro {idx + 1}"
            })
    except Exception as e:
        logger.warning(f"Failed to find tables in fitz: {e}")

    # 2. Trouver les images matricielles
    try:
        # get_image_info() sans arguments est compatible avec toutes les versions
        images = page.get_image_info()
        for idx, img in enumerate(images):
            bbox = img.get("bbox")
            if bbox:
                candidates.append({
                    "type": "image",
                    "bbox": list(bbox),
                    "description": f"Image matricielle numéro {idx + 1}"
                })
    except Exception as e:
        logger.warning(f"Failed to find image info in fitz: {e}")

    # 3. Regrouper (clustering) les dessins vectoriels proches
    try:
        drawings = page.get_drawings()
        drawing_rects = []
        for d in drawings:
            r = d.get("rect")
            if r and rect_width(r) > 15 and rect_height(r) > 15:
                drawing_rects.append(r)

        clusters: List[fitz.Rect] = []
        for rect in drawing_rects:
            # Exclure les rectangles géants qui couvrent toute la page (arrière-plans)
            if rect_width(rect) > page_w * 0.9 and rect_height(rect) > page_h * 0.9:
                continue

            merged = False
            for i, c in enumerate(clusters):
                # Si le rectangle croise le cluster dilaté de 30 points, on fusionne
                dilated_c = dilate_rect(c, 30)
                f_rect = fitz.Rect(rect)
                if dilated_c.intersects(f_rect):
                    clusters[i] = union_rect(c, rect)
                    merged = True
                    break

            if not merged:
                clusters.append(fitz.Rect(rect))

        # Ne garder que les clusters d'une taille significative
        drawing_idx = 1
        for c in clusters:
            w = rect_width(c)
            h = rect_height(c)
            if w > 40 and h > 40:
                if w < page_w * 0.95 and h < page_h * 0.95:
                    candidates.append({
                        "type": "drawing_cluster",
                        "bbox": [c[0], c[1], c[2], c[3]],
                        "description": f"Schéma vectoriel numéro {drawing_idx}"
                    })
                    drawing_idx += 1
    except Exception as e:
        logger.warning(f"Failed to process drawings in fitz: {e}")

    doc.close()

    # 4. Supprimer les doublons ou inclusions
    unique_candidates = []
    for cand in candidates:
        r1 = fitz.Rect(cand["bbox"])
        is_subsumed = False
        for other in candidates:
            if cand == other:
                continue
            r2 = fitz.Rect(other["bbox"])
            if r2.contains(r1):
                if abs(rect_area(r2) - rect_area(r1)) < 100:
                    continue  # Si de taille presque identique, on ne rejette pas
                is_subsumed = True
                break
        if not is_subsumed:
            unique_candidates.append(cand)

    # Assigner un index final et arrondir les coordonnées pour la clarté du prompt
    for idx, c in enumerate(unique_candidates):
        c["index"] = idx
        c["bbox"] = [round(v, 1) for v in c["bbox"]]

    logger.info(f"Found {len(unique_candidates)} unique candidates on page {page_no}")
    return unique_candidates

async def determine_and_crop_illustration(
    query: str,
    answer: str,
    pdf_path: str,
    page_no: int,
    doc_title: str
) -> Optional[Dict[str, Any]]:
    """
    Rend la page sous forme d'image, demande à Mistral Large de sélectionner le visuel 
    le plus adapté pour illustrer la réponse, effectue le détourage et le met en cache.
    """
    from app.services.multimodal_page_service import render_pdf_page_png

    # 1. Récupérer les candidats
    candidates = get_visual_candidates(pdf_path, page_no)
    if not candidates:
        logger.info(f"No visual candidates found on page {page_no}. Skipping illustration.")
        return None

    # 2. Rendre la page en PNG
    try:
        # Utiliser une résolution moyenne de 150 DPI pour l'analyse visuelle
        png_bytes = render_pdf_page_png(pdf_path, page_no - 1, dpi=150)
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

    # 4. Préparer le prompt pour le modèle de vision
    candidates_desc = []
    for c in candidates:
        candidates_desc.append(
            f"- Élément #{c['index']}: Type: {c['type']}, "
            f"Description: {c['description']}, Coordonnées [x0, y0, x1, y1]: {c['bbox']}"
        )
    candidates_text = "\n".join(candidates_desc)

    prompt = f"""Tu es un expert technique de PROFERM en notices techniques de pose de menuiseries et en vision par ordinateur.
Tu reçois l'image d'une page de notice de pose, ainsi que la question de l'utilisateur et la réponse fournie par notre assistant.

Question de l'utilisateur : "{query}"
Réponse de l'assistant : "{answer}"

Voici la liste des éléments visuels détectés sur cette page (coordonnées en points sur la page [0, 0, {width_pts}, {height_pts}]) :
{candidates_text}

Règles absolues d'évaluation :
1. Évalue si un schéma, tableau technique ou diagramme présent sur cette page est TRÈS PERTINENT pour illustrer et enrichir la réponse textuelle de l'assistant.
2. Si la question est purement d'ordre général/théorique, ou si aucun schéma n'illustre spécifiquement ce qui est décrit dans la réponse, réponds que l'illustration n'est pas nécessaire.
3. Si un élément est pertinent, sélectionne son indice "selected_candidate_index" dans la liste ci-dessus.
4. Si l'élément pertinent n'est pas correctement couvert par la liste mais est bien visible sur l'image, tu peux définir une boîte de délimitation personnalisée "custom_bbox" au format [ymin, xmin, ymax, xmax] normalisé de 0 à 100 (pourcentages de la hauteur et largeur de la page).

Tu dois répondre UNIQUEMENT par un objet JSON valide sous ce format :
{{
  "illustration_needed": true,
  "selected_candidate_index": <int ou null>,
  "custom_bbox": [<ymin>, <xmin>, <ymax>, <xmax>] ou null,
  "caption": "Une légende très claire et descriptive en Français décrivant précisément le schéma/tableau détouré"
}}

Si aucune illustration n'est nécessaire ou si les éléments visuels de la page ne sont pas pertinents, retourne :
{{
  "illustration_needed": false,
  "selected_candidate_index": null,
  "custom_bbox": null,
  "caption": ""
}}
"""

    model_name = getattr(settings, "MULTIMODAL_EXTRACT_MODEL", "mistral-large-latest")
    logger.info(f"Sending page {page_no} image and prompt to {model_name}...")

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
        logger.error(f"Error calling Mistral Large vision API: {e}", exc_info=True)
        return None

    logger.info(f"Mistral Large decision: {decision}")

    if not decision.get("illustration_needed"):
        logger.info("Mistral Large decided that an illustration is not relevant.")
        return None

    # 5. Déterminer la bbox à découper (en fitz points)
    bbox = None
    selected_idx = decision.get("selected_candidate_index")
    custom_bbox = decision.get("custom_bbox")

    if selected_idx is not None and 0 <= selected_idx < len(candidates):
        bbox = candidates[selected_idx]["bbox"]
    elif custom_bbox and len(custom_bbox) == 4:
        # Convertir les coordonnées normalisées 0-100 en points fitz
        ymin, xmin, ymax, xmax = custom_bbox
        bbox = [
            (xmin * width_pts) / 100.0,
            (ymin * height_pts) / 100.0,
            (xmax * width_pts) / 100.0,
            (ymax * height_pts) / 100.0
        ]

    if not bbox:
        logger.warning("Vision model requested an illustration but failed to provide coordinates.")
        return None

    # 6. Effectuer le détourage avec une résolution de qualité (200 DPI)
    try:
        crop_dpi = 200
        high_res_bytes = render_pdf_page_png(pdf_path, page_no - 1, dpi=crop_dpi)
        img = Image.open(io.BytesIO(high_res_bytes))

        scale = crop_dpi / 72.0
        x0, y0, x1, y1 = bbox

        # Ajouter une petite marge (ex: 10 points) autour de la boîte
        padding = 10.0
        x0 = max(0.0, x0 - padding)
        y0 = max(0.0, y0 - padding)
        x1 = min(width_pts, x1 + padding)
        y1 = min(height_pts, y1 + padding)

        # Convertir en pixels
        px0 = int(x0 * scale)
        py0 = int(y0 * scale)
        px1 = int(x1 * scale)
        py1 = int(y1 * scale)

        cropped_img = img.crop((px0, py0, px1, py1))

        # 7. Sauvegarder dans le cache d'illustrations
        # Le dossier de cache se situe dans la racine du dossier data de LanceDB
        cache_dir = os.path.join(os.path.dirname(settings.LANCED_DB_DIR), "illustration_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Nom unique basé sur le hash MD5 du fichier, page et coordonnées
        key_str = f"{pdf_path}_{page_no}_{round(x0,1)}_{round(y0,1)}_{round(x1,1)}_{round(y1,1)}"
        filename_hash = hashlib.md5(key_str.encode("utf-8")).hexdigest()
        filename = f"crop_{filename_hash}.png"
        file_path = os.path.join(cache_dir, filename)

        cropped_img.save(file_path, format="PNG")
        logger.info(f"Cropped illustration cached successfully: {file_path}")

        return {
            "url": f"/api/chats/illustrations/{filename}",
            "title": decision.get("caption") or "Schéma d'illustration",
            "page_no": page_no,
            "document_title": doc_title
        }

    except Exception as e:
        logger.error(f"Error during cropping or image saving: {e}", exc_info=True)
        return None
