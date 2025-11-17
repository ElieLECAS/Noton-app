"""
Service de parsing de documents avec Docling.
Gère l'extraction intelligente de contenu depuis divers formats de fichiers.
"""
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import ConversionResult
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("Docling n'est pas installé. Le parsing de documents sera limité.")


class ParsedDocument:
    """Représente un document parsé avec Docling"""
    
    def __init__(self, file_path: str, file_type: str):
        self.file_path = file_path
        self.file_type = file_type
        self.title: str = Path(file_path).stem
        self.content: str = ""
        self.elements: List[Dict[str, Any]] = []  # Éléments structurés (paragraphes, tableaux, etc.)
        self.metadata: Dict[str, Any] = {}
        self.page_count: int = 0
        
    def get_text_content(self) -> str:
        """Récupérer tout le contenu textuel du document"""
        return self.content
    
    def get_elements_by_type(self, element_type: str) -> List[Dict[str, Any]]:
        """Récupérer les éléments d'un type spécifique (paragraph, table, list, etc.)"""
        return [el for el in self.elements if el.get('type') == element_type]


class DoclingService:
    """Service de parsing de documents avec Docling"""
    
    def __init__(self):
        self.converter: Optional[DocumentConverter] = None
        if DOCLING_AVAILABLE:
            try:
                self.converter = DocumentConverter()
                logger.info("✅ Docling DocumentConverter initialisé")
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'initialisation de Docling: {e}")
                self.converter = None
    
    def parse_document(self, file_path: str, file_type: str) -> Optional[ParsedDocument]:
        """
        Parser un document avec Docling.
        
        Args:
            file_path: Chemin vers le fichier
            file_type: Type MIME du fichier
            
        Returns:
            ParsedDocument ou None si erreur
        """
        if not DOCLING_AVAILABLE or not self.converter:
            logger.warning("Docling non disponible, utilisation du parser basique")
            return self._parse_basic(file_path, file_type)
        
        try:
            logger.info(f"📄 Parsing de {file_path} (type: {file_type})")
            
            # Convertir le document
            result: ConversionResult = self.converter.convert(file_path)
            
            # Créer l'objet ParsedDocument
            parsed_doc = ParsedDocument(file_path, file_type)
            
            # Extraire le contenu et la structure
            if result.document:
                # Récupérer le contenu textuel complet
                parsed_doc.content = result.document.export_to_markdown()
                
                # Extraire les métadonnées
                if hasattr(result.document, 'metadata'):
                    parsed_doc.metadata = {
                        'num_pages': getattr(result.document, 'num_pages', 0),
                        'title': getattr(result.document, 'title', parsed_doc.title),
                    }
                    parsed_doc.page_count = parsed_doc.metadata.get('num_pages', 0)
                
                # Extraire les éléments structurés
                parsed_doc.elements = self._extract_elements(result)
                
                logger.info(f"✅ Document parsé: {len(parsed_doc.elements)} éléments, "
                           f"{len(parsed_doc.content)} caractères")
                
            return parsed_doc
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du parsing avec Docling: {e}", exc_info=True)
            # Fallback sur le parser basique
            return self._parse_basic(file_path, file_type)
    
    def _extract_elements(self, result: 'ConversionResult') -> List[Dict[str, Any]]:
        """
        Extraire les éléments structurés du document Docling.
        """
        elements = []
        
        try:
            if not hasattr(result, 'document') or not result.document:
                return elements
            
            doc = result.document
            
            # Parcourir les éléments du document
            if hasattr(doc, 'body') and doc.body:
                for idx, item in enumerate(doc.body):
                    element = {
                        'index': idx,
                        'type': 'text',
                        'content': '',
                        'metadata': {}
                    }
                    
                    # Déterminer le type d'élément
                    item_type = type(item).__name__.lower()
                    
                    if 'table' in item_type:
                        element['type'] = 'table'
                        # Extraire le contenu du tableau en markdown
                        if hasattr(item, 'export_to_markdown'):
                            element['content'] = item.export_to_markdown()
                        elif hasattr(item, 'to_markdown'):
                            element['content'] = item.to_markdown()
                    elif 'list' in item_type:
                        element['type'] = 'list'
                        element['content'] = str(item)
                    elif 'heading' in item_type or 'title' in item_type:
                        element['type'] = 'title'
                        element['content'] = str(item)
                    else:
                        element['type'] = 'paragraph'
                        element['content'] = str(item)
                    
                    # Extraire les métadonnées si disponibles
                    if hasattr(item, 'page'):
                        element['metadata']['page'] = item.page
                    if hasattr(item, 'bbox'):
                        element['metadata']['bbox'] = item.bbox
                    
                    if element['content']:
                        elements.append(element)
            
        except Exception as e:
            logger.warning(f"Erreur lors de l'extraction des éléments: {e}")
        
        return elements
    
    def _parse_basic(self, file_path: str, file_type: str) -> Optional[ParsedDocument]:
        """
        Parser basique pour les formats simples sans Docling.
        Utilisé comme fallback ou pour les formats texte simples.
        """
        try:
            parsed_doc = ParsedDocument(file_path, file_type)
            
            # Texte simple
            if file_type in ['text/plain', 'text/markdown']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    parsed_doc.content = f.read()
                    parsed_doc.elements = [{
                        'index': 0,
                        'type': 'text',
                        'content': parsed_doc.content,
                        'metadata': {}
                    }]
            
            # CSV
            elif file_type == 'text/csv':
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if rows:
                        # Convertir en tableau markdown
                        headers = rows[0]
                        markdown_table = "| " + " | ".join(headers) + " |\n"
                        markdown_table += "| " + " | ".join(['---'] * len(headers)) + " |\n"
                        for row in rows[1:]:
                            markdown_table += "| " + " | ".join(row) + " |\n"
                        
                        parsed_doc.content = markdown_table
                        parsed_doc.elements = [{
                            'index': 0,
                            'type': 'table',
                            'content': markdown_table,
                            'metadata': {'rows': len(rows), 'cols': len(headers)}
                        }]
            
            # JSON
            elif file_type == 'application/json':
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convertir JSON en format lisible
                    parsed_doc.content = json.dumps(data, indent=2, ensure_ascii=False)
                    parsed_doc.elements = [{
                        'index': 0,
                        'type': 'text',
                        'content': parsed_doc.content,
                        'metadata': {}
                    }]
            
            else:
                logger.warning(f"Type de fichier non supporté par le parser basique: {file_type}")
                return None
            
            logger.info(f"✅ Document parsé (mode basique): {len(parsed_doc.content)} caractères")
            return parsed_doc
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du parsing basique: {e}", exc_info=True)
            return None
    
    def is_supported_format(self, file_type: str) -> bool:
        """
        Vérifier si le format de fichier est supporté.
        """
        supported_formats = [
            # Documents
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # DOCX
            'application/msword',  # DOC
            'application/vnd.ms-word.document.macroEnabled.12',  # DOCM
            
            # Feuilles de calcul
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # XLSX
            'application/vnd.ms-excel',  # XLS
            'text/csv',
            
            # Présentations
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',  # PPTX
            'application/vnd.ms-powerpoint',  # PPT
            
            # Images (avec OCR)
            'image/png',
            'image/jpeg',
            'image/jpg',
            'image/tiff',
            'image/bmp',
            
            # Texte
            'text/plain',
            'text/markdown',
            'application/json',
            'text/html',
        ]
        
        return file_type in supported_formats


# Instance singleton
_docling_service: Optional[DoclingService] = None


def get_docling_service() -> DoclingService:
    """Récupérer l'instance singleton du service Docling"""
    global _docling_service
    if _docling_service is None:
        _docling_service = DoclingService()
    return _docling_service

