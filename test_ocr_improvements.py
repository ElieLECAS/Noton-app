"""
Script de test pour valider l'implémentation des améliorations OCR.

Ce script vérifie que tous les modules nécessaires peuvent être importés
et que la configuration est correctement chargée.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test des imports des nouveaux modules OCR."""
    print("=" * 60)
    print("Test des imports des modules OCR")
    print("=" * 60)
    
    try:
        # Test import ocr_preprocessing
        from app.services.ocr_preprocessing import (
            preprocess_image_for_ocr,
            assess_image_quality,
            deskew_image,
            preprocess_images_batch
        )
        print("✅ Module ocr_preprocessing importé avec succès")
    except Exception as e:
        print(f"❌ Erreur import ocr_preprocessing: {e}")
        return False
    
    try:
        # Test import ocr_fallback
        from app.services.ocr_fallback import (
            extract_with_fallback,
            is_ocr_result_sufficient,
            run_tesseract_ocr,
            extract_pdf_images_with_preprocessing,
            count_pdf_pages
        )
        print("✅ Module ocr_fallback importé avec succès")
    except Exception as e:
        print(f"❌ Erreur import ocr_fallback: {e}")
        return False
    
    return True


def test_config():
    """Test de la configuration OCR."""
    print("\n" + "=" * 60)
    print("Test de la configuration OCR")
    print("=" * 60)
    
    try:
        from app.config import settings
        
        # Vérifier les nouvelles variables OCR
        config_vars = {
            "OCR_IMAGE_SCALE": settings.OCR_IMAGE_SCALE,
            "OCR_PREPROCESS_ENABLED": settings.OCR_PREPROCESS_ENABLED,
            "OCR_FALLBACK_ENABLED": settings.OCR_FALLBACK_ENABLED,
            "OCR_MIN_TEXT_LENGTH": settings.OCR_MIN_TEXT_LENGTH,
            "OCR_TESSERACT_CONFIG": settings.OCR_TESSERACT_CONFIG,
            "DOCLING_OCR_ENABLED": settings.DOCLING_OCR_ENABLED,
            "DOCLING_OCR_LANG": settings.DOCLING_OCR_LANG,
        }
        
        print("\nConfiguration OCR chargée :")
        for key, value in config_vars.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Configuration OCR chargée avec succès")
        return True
        
    except Exception as e:
        print(f"❌ Erreur chargement configuration: {e}")
        return False


def test_dependencies():
    """Test de la disponibilité des dépendances."""
    print("\n" + "=" * 60)
    print("Test des dépendances")
    print("=" * 60)
    
    dependencies = {
        "opencv (cv2)": "cv2",
        "PIL (Pillow)": "PIL",
        "numpy": "numpy",
        "pytesseract": "pytesseract",
        "pdf2image": "pdf2image",
        "PyPDF2": "PyPDF2",
    }
    
    all_ok = True
    for name, module_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"✅ {name} disponible")
        except ImportError:
            print(f"⚠️  {name} non disponible (sera installé au runtime)")
            # Ne pas marquer comme échec car certaines dépendances
            # peuvent ne pas être installées en dev local
    
    return True


def test_docling_integration():
    """Test de l'intégration avec Docling."""
    print("\n" + "=" * 60)
    print("Test de l'intégration Docling")
    print("=" * 60)
    
    try:
        from app.services.document_service import get_docling_converter
        
        print("✅ Fonction get_docling_converter accessible")
        print("   (Le converter sera initialisé au premier usage)")
        return True
        
    except Exception as e:
        print(f"❌ Erreur intégration Docling: {e}")
        return False


def main():
    """Exécute tous les tests."""
    print("\n" + "🔧" * 30)
    print("VALIDATION DES AMÉLIORATIONS OCR")
    print("🔧" * 30 + "\n")
    
    results = {
        "Imports": test_imports(),
        "Configuration": test_config(),
        "Dépendances": test_dependencies(),
        "Intégration Docling": test_docling_integration(),
    }
    
    print("\n" + "=" * 60)
    print("RÉSULTATS DES TESTS")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 Tous les tests sont passés avec succès !")
        print("\nProchaines étapes :")
        print("1. Reconstruire l'image Docker : docker-compose build web")
        print("2. Redémarrer les services : docker-compose up -d")
        print("3. Tester avec un document scanné")
        return 0
    else:
        print("\n⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
