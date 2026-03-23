"""
Service de gestion du stockage de fichiers uploadés.
Gère la sauvegarde, la suppression et l'organisation des fichiers.
"""
from pathlib import Path
from typing import Optional, Tuple
import shutil
import logging
import mimetypes
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

# Dossier racine pour les uploads
UPLOAD_ROOT = Path("app/uploads")

# Taille maximale de fichier : 50 MB
MAX_FILE_SIZE = 50 * 1024 * 1024


class FileStorageService:
    """Service de gestion des fichiers uploadés"""
    
    def __init__(self, upload_root: Path = UPLOAD_ROOT):
        self.upload_root = upload_root
        self.upload_root.mkdir(parents=True, exist_ok=True)
    
    def get_note_storage_path(self, user_id: int, project_id: int, note_id: int) -> Path:
        """
        Obtenir le chemin de stockage pour une note.
        Structure: uploads/user_{user_id}/project_{project_id}/note_{note_id}/
        """
        path = self.upload_root / f"user_{user_id}" / f"project_{project_id}" / f"note_{note_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def save_uploaded_file(
        self,
        file_content: bytes,
        original_filename: str,
        user_id: int,
        project_id: int,
        note_id: int
    ) -> Tuple[str, str]:
        """
        Sauvegarder un fichier uploadé.
        
        Args:
            file_content: Contenu binaire du fichier
            original_filename: Nom original du fichier
            user_id: ID de l'utilisateur
            project_id: ID du projet
            note_id: ID de la note
            
        Returns:
            Tuple (file_path, mime_type)
        """
        try:
            # Vérifier la taille
            if len(file_content) > MAX_FILE_SIZE:
                raise ValueError(f"Fichier trop volumineux. Taille max: {MAX_FILE_SIZE / (1024*1024):.0f} MB")
            
            # Obtenir le chemin de stockage
            storage_path = self.get_note_storage_path(user_id, project_id, note_id)
            
            # Nettoyer le nom de fichier (enlever les caractères dangereux)
            safe_filename = self._sanitize_filename(original_filename)
            
            # Ajouter un timestamp pour éviter les conflits
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename_parts = safe_filename.rsplit('.', 1)
            if len(filename_parts) == 2:
                final_filename = f"original_{timestamp}_{filename_parts[0]}.{filename_parts[1]}"
            else:
                final_filename = f"original_{timestamp}_{safe_filename}"
            
            # Chemin complet du fichier
            file_path = storage_path / final_filename
            
            # Sauvegarder le fichier
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Déterminer le type MIME
            mime_type = self._get_mime_type(str(file_path), original_filename)
            
            logger.info(f"✅ Fichier sauvegardé: {file_path} (type: {mime_type}, taille: {len(file_content)} bytes)")
            
            return str(file_path), mime_type
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde du fichier: {e}", exc_info=True)
            raise
    
    def delete_note_files(self, user_id: int, project_id: int, note_id: int) -> bool:
        """
        Supprimer tous les fichiers d'une note.
        
        Args:
            user_id: ID de l'utilisateur
            project_id: ID du projet
            note_id: ID de la note
            
        Returns:
            True si succès, False sinon
        """
        try:
            storage_path = self.get_note_storage_path(user_id, project_id, note_id)
            
            if storage_path.exists():
                shutil.rmtree(storage_path)
                logger.info(f"🗑️ Fichiers supprimés: {storage_path}")
                return True
            else:
                logger.warning(f"Dossier inexistant: {storage_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la suppression des fichiers: {e}", exc_info=True)
            return False
    
    def get_file_path(self, relative_path: str) -> Optional[Path]:
        """
        Obtenir le chemin absolu d'un fichier à partir du chemin relatif.
        
        Args:
            relative_path: Chemin relatif du fichier
            
        Returns:
            Path absolu ou None si inexistant
        """
        try:
            file_path = Path(relative_path)
            
            if file_path.exists() and file_path.is_file():
                return file_path
            else:
                logger.warning(f"Fichier inexistant: {relative_path}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la résolution du chemin: {e}")
            return None
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Nettoyer un nom de fichier pour le rendre sûr.
        Enlève les caractères dangereux et les chemins relatifs.
        """
        # Enlever les chemins relatifs
        filename = Path(filename).name
        
        # Remplacer les caractères dangereux
        dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Limiter la longueur
        if len(filename) > 200:
            name_part, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name_part[:190] + ('.' + ext if ext else '')
        
        return filename
    
    def _get_mime_type(self, file_path: str, original_filename: str) -> str:
        """
        Déterminer le type MIME d'un fichier.
        """
        # Essayer avec python-magic (plus précis)
        try:
            import magic
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(file_path)
            if mime_type:
                return mime_type
        except (ImportError, Exception) as e:
            logger.debug(f"python-magic non disponible ou erreur: {e}")
        
        # Fallback : utiliser mimetypes standard
        mime_type, _ = mimetypes.guess_type(original_filename)
        if mime_type:
            return mime_type
        
        # Fallback par extension
        extension_map = {
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xls': 'application/vnd.ms-excel',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.txt': 'text/plain',
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
        }
        
        file_ext = Path(original_filename).suffix.lower()
        return extension_map.get(file_ext, 'application/octet-stream')
    
    def get_storage_stats(self, user_id: int, project_id: Optional[int] = None) -> dict:
        """
        Obtenir des statistiques sur l'espace de stockage utilisé.
        
        Args:
            user_id: ID de l'utilisateur
            project_id: ID du projet (optionnel, pour stats par projet)
            
        Returns:
            Dictionnaire avec les statistiques
        """
        try:
            if project_id:
                base_path = self.upload_root / f"user_{user_id}" / f"project_{project_id}"
            else:
                base_path = self.upload_root / f"user_{user_id}"
            
            if not base_path.exists():
                return {'total_size': 0, 'file_count': 0}
            
            total_size = 0
            file_count = 0
            
            for file_path in base_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            return {
                'total_size': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'file_count': file_count
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du calcul des statistiques: {e}")
            return {'total_size': 0, 'file_count': 0}


# Instance singleton
_file_storage_service: Optional[FileStorageService] = None


def get_file_storage_service() -> FileStorageService:
    """Récupérer l'instance singleton du service de stockage"""
    global _file_storage_service
    if _file_storage_service is None:
        _file_storage_service = FileStorageService()
    return _file_storage_service

