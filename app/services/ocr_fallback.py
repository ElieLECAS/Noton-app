"""
Module de fallback OCR multi-moteurs pour améliorer l'extraction de texte.

Ce module orchestre plusieurs moteurs OCR (Docling/EasyOCR et Tesseract)
pour maximiser la qualité d'extraction, particulièrement pour les documents
scannés de mauvaise qualité.
"""

import logging
from typing import Optional, Tuple
from pathlib import Path
import time

logger = logging.getLogger(__name__)


def count_pdf_pages(file_path: str) -> int:
    """
    Compte le nombre de pages dans un PDF.
    
    Args:
        file_path: Chemin vers le fichier PDF
    
    Returns:
        Nombre de pages, ou 1 si impossible à déterminer
    """
    try:
        from PyPDF2 import PdfReader
        
        reader = PdfReader(file_path)
        return len(reader.pages)
    except Exception as e:
        logger.debug("Impossible de compter les pages PDF: %s", e)
        # Fallback : essayer avec pdf2image
        try:
            from pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(file_path)
            return info.get("Pages", 1)
        except Exception as e2:
            logger.debug("Fallback comptage pages échoué: %s", e2)
            return 1


def is_ocr_result_sufficient(
    markdown_content: str,
    file_path: str,
    min_chars_per_page: int = 50
) -> bool:
    """
    Évalue si le résultat OCR est suffisant ou si un fallback est nécessaire.
    
    Critères d'évaluation :
    - Longueur minimale absolue de texte
    - Ratio caractères/pages suffisant
    - Présence de contenu significatif (pas seulement whitespace)
    
    Args:
        markdown_content: Contenu markdown extrait par Docling
        file_path: Chemin du fichier PDF source
        min_chars_per_page: Nombre minimum de caractères par page attendu
    
    Returns:
        True si le résultat est jugé suffisant, False sinon
    """
    if not markdown_content or not markdown_content.strip():
        logger.warning("Résultat OCR vide, fallback nécessaire")
        return False
    
    # Compter les caractères significatifs (hors whitespace)
    significant_chars = len(markdown_content.strip())
    
    # Compter les pages
    num_pages = count_pdf_pages(file_path)
    
    # Calculer le ratio
    chars_per_page = significant_chars / max(num_pages, 1)
    
    # Seuil minimum absolu
    min_total_chars = 30
    
    is_sufficient = (
        significant_chars >= min_total_chars and
        chars_per_page >= min_chars_per_page
    )
    
    logger.info(
        "Évaluation OCR Docling - Pages: %d, Caractères: %d, Ratio: %.1f char/page, "
        "Seuil: %d char/page, Suffisant: %s",
        num_pages, significant_chars, chars_per_page, min_chars_per_page, is_sufficient
    )
    
    return is_sufficient


def extract_pdf_images_with_preprocessing(file_path: str) -> list:
    """
    Extrait les images d'un PDF et applique un prétraitement adaptatif.
    
    Args:
        file_path: Chemin vers le fichier PDF
    
    Returns:
        Liste d'images PIL prétraitées
    """
    try:
        from pdf2image import convert_from_path
        from app.services.ocr_preprocessing import preprocess_images_batch
        
        logger.info("Extraction des images du PDF: %s", file_path)
        
        # Convertir PDF en images (haute résolution pour meilleur OCR)
        images = convert_from_path(
            file_path,
            dpi=300,  # Haute résolution pour OCR
            fmt="png"
        )
        
        logger.info("PDF converti en %d image(s), application du prétraitement", len(images))
        
        # Appliquer le prétraitement adaptatif
        processed_images, metadata = preprocess_images_batch(images, force_preprocess=False)
        
        return processed_images
        
    except Exception as e:
        logger.error(
            "Erreur lors de l'extraction d'images du PDF: %s",
            e,
            exc_info=True
        )
        return []


def run_tesseract_ocr(
    images: list,
    tesseract_config: str = r'--oem 3 --psm 6 -l fra+eng'
) -> str:
    """
    Exécute Tesseract OCR sur une liste d'images.
    
    Args:
        images: Liste d'images PIL
        tesseract_config: Configuration Tesseract
            --oem 3 = LSTM OCR Engine Mode (meilleur qualité)
            --psm 6 = Assume a single uniform block of text
            -l fra+eng = Langues français + anglais
    
    Returns:
        Texte extrait (markdown format)
    """
    if not images:
        return ""
    
    try:
        import pytesseract
        
        logger.info(
            "Exécution Tesseract OCR sur %d image(s) avec config: %s",
            len(images), tesseract_config
        )
        
        all_text = []
        
        for i, img in enumerate(images):
            try:
                # Extraire le texte avec Tesseract
                text = pytesseract.image_to_string(img, config=tesseract_config)
                
                if text and text.strip():
                    all_text.append(f"## Page {i + 1}\n\n{text.strip()}")
                    logger.debug("Page %d: %d caractères extraits", i + 1, len(text.strip()))
                else:
                    logger.debug("Page %d: aucun texte extrait", i + 1)
                    
            except Exception as e:
                logger.warning("Erreur OCR Tesseract page %d: %s", i + 1, e)
                continue
        
        result = "\n\n".join(all_text)
        
        logger.info(
            "Tesseract OCR terminé - %d caractères extraits au total",
            len(result)
        )
        
        return result
        
    except ImportError:
        logger.error(
            "pytesseract non installé, impossible d'exécuter le fallback Tesseract"
        )
        return ""
    except Exception as e:
        logger.error(
            "Erreur lors de l'exécution de Tesseract OCR: %s",
            e,
            exc_info=True
        )
        return ""


def merge_or_select_best(
    docling_result: str,
    tesseract_result: str
) -> str:
    """
    Fusionne ou sélectionne le meilleur résultat entre Docling et Tesseract.
    
    Stratégie :
    - Si un seul résultat non-vide : le retourner
    - Si les deux sont non-vides : retourner le plus long (généralement meilleur)
    - Possibilité future : fusion intelligente basée sur la confiance
    
    Args:
        docling_result: Résultat de Docling/EasyOCR
        tesseract_result: Résultat de Tesseract
    
    Returns:
        Meilleur résultat ou résultat fusionné
    """
    docling_len = len(docling_result.strip()) if docling_result else 0
    tesseract_len = len(tesseract_result.strip()) if tesseract_result else 0
    
    # Si un seul résultat est valide
    if docling_len == 0 and tesseract_len > 0:
        logger.info("Sélection résultat Tesseract (Docling vide)")
        return tesseract_result
    
    if tesseract_len == 0 and docling_len > 0:
        logger.info("Sélection résultat Docling (Tesseract vide)")
        return docling_result
    
    if docling_len == 0 and tesseract_len == 0:
        logger.warning("Aucun texte extrait par Docling ni Tesseract")
        return ""
    
    # Les deux ont du contenu : choisir le plus complet
    # Note: on pourrait améliorer avec une fusion intelligente
    if tesseract_len > docling_len * 1.5:
        # Tesseract a extrait significativement plus de texte
        logger.info(
            "Sélection résultat Tesseract (%d chars) > Docling (%d chars)",
            tesseract_len, docling_len
        )
        return tesseract_result
    else:
        # Garder Docling (généralement meilleure structure)
        logger.info(
            "Sélection résultat Docling (%d chars), Tesseract (%d chars)",
            docling_len, tesseract_len
        )
        return docling_result


def extract_with_fallback(
    file_path: str,
    docling_result: str,
    docling_doc: Optional[object] = None,
    fallback_enabled: bool = True,
    tesseract_config: Optional[str] = None
) -> str:
    """
    Fonction principale d'extraction OCR avec fallback multi-moteurs.
    
    Workflow :
    1. Évaluer la qualité du résultat Docling
    2. Si insuffisant et fallback activé :
       - Extraire images du PDF avec prétraitement
       - Exécuter Tesseract OCR
       - Fusionner ou sélectionner le meilleur résultat
    3. Retourner le résultat optimal
    
    Args:
        file_path: Chemin vers le fichier PDF source
        docling_result: Résultat markdown de Docling/EasyOCR
        docling_doc: Document Docling (optionnel, pour extraction future)
        fallback_enabled: Si False, désactive le fallback
        tesseract_config: Configuration Tesseract personnalisée
    
    Returns:
        Meilleur résultat markdown après évaluation et fallback éventuel
    """
    from app.config import settings
    
    # Vérifier si le fallback est désactivé
    if not fallback_enabled:
        logger.debug("Fallback OCR désactivé, retour résultat Docling")
        return docling_result
    
    start_time = time.time()
    
    # Évaluer la qualité du résultat Docling
    min_chars_per_page = getattr(settings, "OCR_MIN_TEXT_LENGTH", 50)
    
    if is_ocr_result_sufficient(docling_result, file_path, min_chars_per_page):
        logger.info("✅ Résultat OCR Docling suffisant, pas de fallback nécessaire")
        return docling_result
    
    # Résultat insuffisant : activer le fallback Tesseract
    logger.warning(
        "⚠️  OCR Docling insuffisant pour %s, activation fallback Tesseract",
        Path(file_path).name
    )
    
    try:
        # Extraire images avec prétraitement
        images = extract_pdf_images_with_preprocessing(file_path)
        
        if not images:
            logger.error("Impossible d'extraire les images du PDF, retour résultat Docling")
            return docling_result
        
        # Configuration Tesseract
        if tesseract_config is None:
            tesseract_config = getattr(
                settings,
                "OCR_TESSERACT_CONFIG",
                r'--oem 3 --psm 6 -l fra+eng'
            )
        
        # Exécuter Tesseract OCR
        tesseract_result = run_tesseract_ocr(images, tesseract_config)
        
        # Fusionner ou sélectionner le meilleur
        final_result = merge_or_select_best(docling_result, tesseract_result)
        
        elapsed_time = time.time() - start_time
        
        logger.info(
            "✅ Fallback OCR terminé en %.2fs - Résultat final: %d caractères",
            elapsed_time, len(final_result)
        )
        
        return final_result
        
    except Exception as e:
        logger.error(
            "Erreur lors du fallback OCR: %s, retour résultat Docling",
            e,
            exc_info=True
        )
        return docling_result
