from typing import Optional
import logging
import os
from pathlib import Path
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)


def process_document(file_path: str) -> Optional[str]:
    """
    Traite un document avec docling et le convertit en markdown.
    
    Args:
        file_path: Chemin vers le fichier à traiter
        
    Returns:
        Contenu markdown du document ou None en cas d'erreur
    """
    if not os.path.exists(file_path):
        logger.error(f"Fichier non trouvé: {file_path}")
        return None
    
    try:
        # Créer le convertisseur docling avec configuration par défaut
        # Docling gère automatiquement différents formats (PDF, DOCX, XLSX, PPTX, images, etc.)
        converter = DocumentConverter()
        
        # Convertir le document en markdown
        logger.info(f"Traitement du document: {file_path}")
        result = converter.convert(file_path)
        
        # Extraire le markdown depuis le résultat
        # Docling retourne un objet DocumentPipelineOutput avec un attribut document
        if hasattr(result, 'document'):
            markdown_content = result.document.export_to_markdown()
        elif hasattr(result, 'export_to_markdown'):
            markdown_content = result.export_to_markdown()
        else:
            # Fallback : essayer d'accéder directement au contenu
            logger.warning("Format de résultat docling inattendu, tentative d'extraction alternative")
            markdown_content = str(result)
        
        if not markdown_content or not markdown_content.strip():
            logger.warning(f"Le document {file_path} a été traité mais le contenu markdown est vide")
            return None
        
        logger.info(f"Document converti avec succès en markdown ({len(markdown_content)} caractères)")
        return markdown_content
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement du document {file_path}: {e}", exc_info=True)
        return None


def save_uploaded_file(file_content: bytes, filename: str, upload_dir: str = "media/documents") -> Optional[str]:
    """
    Sauvegarde un fichier uploadé sur le disque.
    
    Args:
        file_content: Contenu binaire du fichier
        filename: Nom du fichier original
        upload_dir: Répertoire de destination
        
    Returns:
        Chemin complet du fichier sauvegardé ou None en cas d'erreur
    """
    try:
        # Créer le répertoire s'il n'existe pas
        upload_path = Path(upload_dir)
        upload_path.mkdir(parents=True, exist_ok=True)
        
        # Générer un nom de fichier unique pour éviter les collisions
        file_extension = Path(filename).suffix
        import uuid
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = upload_path / unique_filename
        
        # Sauvegarder le fichier
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        logger.info(f"Fichier sauvegardé: {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du fichier {filename}: {e}", exc_info=True)
        return None

