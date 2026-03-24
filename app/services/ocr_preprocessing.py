"""
Module de prétraitement d'images pour améliorer la qualité OCR.

Ce module applique des transformations adaptatives aux images extraites de PDF
scannés pour améliorer la précision de l'OCR (débruitage, amélioration du contraste,
binarisation, correction d'inclinaison).
"""

import logging
from typing import Optional, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Seuil de qualité en dessous duquel le prétraitement est appliqué
QUALITY_THRESHOLD = 0.6  # 0.0 = mauvaise qualité, 1.0 = excellente qualité


def assess_image_quality(img_gray: np.ndarray) -> float:
    """
    Évalue la qualité d'une image en niveaux de gris.
    
    Mesure plusieurs facteurs :
    - Contraste (écart-type des valeurs de pixels)
    - Netteté (variance du Laplacien - détection de flou)
    - Luminosité (moyenne des pixels)
    
    Args:
        img_gray: Image en niveaux de gris (numpy array)
    
    Returns:
        Score de qualité entre 0.0 (mauvaise) et 1.0 (excellente)
    """
    try:
        import cv2
        
        # 1. Contraste (écart-type normalisé)
        contrast = np.std(img_gray) / 255.0
        
        # 2. Netteté (variance du Laplacien pour détecter le flou)
        laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 500.0, 1.0)  # Normaliser à [0,1]
        
        # 3. Luminosité (pénaliser les images trop sombres ou trop claires)
        brightness = np.mean(img_gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2.0  # Optimal à 0.5
        
        # Score composite (pondéré)
        quality_score = (
            0.4 * contrast +
            0.4 * sharpness +
            0.2 * brightness_score
        )
        
        logger.debug(
            "Qualité image - Contraste: %.2f, Netteté: %.2f, Luminosité: %.2f, Score final: %.2f",
            contrast, sharpness, brightness_score, quality_score
        )
        
        return quality_score
        
    except Exception as e:
        logger.warning("Erreur lors de l'évaluation de la qualité d'image: %s", e)
        return 1.0  # Par défaut, considérer l'image comme bonne


def deskew_image(img_gray: np.ndarray) -> np.ndarray:
    """
    Corrige l'inclinaison d'une image (deskewing).
    
    Utilise la détection de lignes de Hough pour estimer l'angle d'inclinaison
    et applique une rotation pour redresser l'image.
    
    Args:
        img_gray: Image en niveaux de gris
    
    Returns:
        Image redressée
    """
    try:
        import cv2
        
        # Détection des contours
        edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
        
        # Détection de lignes avec transformée de Hough
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None or len(lines) == 0:
            return img_gray
        
        # Calculer l'angle moyen des lignes détectées
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            # Filtrer les angles proches de l'horizontale
            if abs(angle) < 45:
                angles.append(angle)
        
        if not angles:
            return img_gray
        
        # Angle médian pour robustesse aux outliers
        median_angle = np.median(angles)
        
        # Ne corriger que si l'inclinaison est significative
        if abs(median_angle) < 0.5:
            return img_gray
        
        # Appliquer la rotation
        (h, w) = img_gray.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            img_gray, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        logger.debug("Image redressée (angle: %.2f°)", median_angle)
        return rotated
        
    except Exception as e:
        logger.warning("Erreur lors du deskewing: %s", e)
        return img_gray


def preprocess_image_for_ocr(
    image: Image.Image,
    force_preprocess: bool = False
) -> Tuple[Image.Image, dict]:
    """
    Applique un prétraitement adaptatif à une image pour améliorer l'OCR.
    
    Le prétraitement est appliqué uniquement si la qualité de l'image est
    jugée insuffisante, sauf si force_preprocess=True.
    
    Pipeline de prétraitement :
    1. Conversion en niveaux de gris
    2. Évaluation de la qualité
    3. Si qualité < seuil ou force_preprocess :
       - Redressement (deskewing)
       - Débruitage
       - Amélioration du contraste (CLAHE)
       - Binarisation adaptative
    
    Args:
        image: Image PIL à traiter
        force_preprocess: Si True, applique le prétraitement même si la qualité est bonne
    
    Returns:
        Tuple (image_traitée, métadonnées) où métadonnées contient :
        - quality_score: score de qualité initial
        - preprocessing_applied: booléen indiquant si le prétraitement a été appliqué
        - steps_applied: liste des étapes de traitement appliquées
    """
    try:
        import cv2
        
        metadata = {
            "quality_score": 1.0,
            "preprocessing_applied": False,
            "steps_applied": []
        }
        
        # Convertir PIL Image en numpy array
        img_array = np.array(image)
        
        # Convertir en niveaux de gris si nécessaire
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        # Évaluer la qualité de l'image
        quality_score = assess_image_quality(img_gray)
        metadata["quality_score"] = quality_score
        
        # Décider si le prétraitement est nécessaire
        needs_preprocessing = force_preprocess or quality_score < QUALITY_THRESHOLD
        
        if not needs_preprocessing:
            logger.debug(
                "Qualité image suffisante (%.2f >= %.2f), pas de prétraitement",
                quality_score, QUALITY_THRESHOLD
            )
            return image, metadata
        
        logger.info(
            "Application du prétraitement (qualité: %.2f, seuil: %.2f, forcé: %s)",
            quality_score, QUALITY_THRESHOLD, force_preprocess
        )
        
        metadata["preprocessing_applied"] = True
        processed = img_gray.copy()
        
        # Étape 1: Redressement (deskewing)
        try:
            processed = deskew_image(processed)
            metadata["steps_applied"].append("deskew")
        except Exception as e:
            logger.debug("Deskewing ignoré: %s", e)
        
        # Étape 2: Débruitage
        try:
            processed = cv2.fastNlMeansDenoising(processed, h=10)
            metadata["steps_applied"].append("denoise")
            logger.debug("Débruitage appliqué")
        except Exception as e:
            logger.debug("Débruitage ignoré: %s", e)
        
        # Étape 3: Amélioration du contraste (CLAHE)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
            metadata["steps_applied"].append("clahe")
            logger.debug("Amélioration contraste (CLAHE) appliquée")
        except Exception as e:
            logger.debug("CLAHE ignoré: %s", e)
        
        # Étape 4: Binarisation adaptative
        try:
            processed = cv2.adaptiveThreshold(
                processed,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            metadata["steps_applied"].append("adaptive_threshold")
            logger.debug("Binarisation adaptative appliquée")
        except Exception as e:
            logger.debug("Binarisation ignorée: %s", e)
        
        # Convertir le résultat en PIL Image
        result_image = Image.fromarray(processed)
        
        logger.info(
            "Prétraitement terminé - Étapes appliquées: %s",
            ", ".join(metadata["steps_applied"])
        )
        
        return result_image, metadata
        
    except Exception as e:
        logger.error(
            "Erreur lors du prétraitement d'image: %s, retour image originale",
            e,
            exc_info=True
        )
        return image, {
            "quality_score": 0.0,
            "preprocessing_applied": False,
            "steps_applied": [],
            "error": str(e)
        }


def preprocess_images_batch(
    images: list[Image.Image],
    force_preprocess: bool = False
) -> Tuple[list[Image.Image], list[dict]]:
    """
    Applique le prétraitement à un lot d'images.
    
    Args:
        images: Liste d'images PIL à traiter
        force_preprocess: Si True, force le prétraitement pour toutes les images
    
    Returns:
        Tuple (images_traitées, métadonnées_par_image)
    """
    processed_images = []
    all_metadata = []
    
    for i, img in enumerate(images):
        processed_img, metadata = preprocess_image_for_ocr(img, force_preprocess)
        processed_images.append(processed_img)
        all_metadata.append(metadata)
        
        logger.debug(
            "Image %d/%d - Qualité: %.2f, Prétraitement: %s",
            i + 1, len(images),
            metadata["quality_score"],
            "Oui" if metadata["preprocessing_applied"] else "Non"
        )
    
    # Statistiques globales
    preprocessed_count = sum(1 for m in all_metadata if m["preprocessing_applied"])
    avg_quality = sum(m["quality_score"] for m in all_metadata) / len(all_metadata)
    
    logger.info(
        "Batch prétraitement terminé - %d/%d images traitées, qualité moyenne: %.2f",
        preprocessed_count, len(images), avg_quality
    )
    
    return processed_images, all_metadata
