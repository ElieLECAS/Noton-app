from typing import Optional
import logging
import os
from pathlib import Path
import threading
import time
from queue import Queue
from sqlmodel import Session
from app.config import settings
from app.database import engine
from app.models.note import Note
from app.services.chunk_service import create_chunks_for_note, generate_embeddings_for_chunks_async
from datetime import datetime

logger = logging.getLogger(__name__)

# Files d'attente par projet pour traiter les documents séquentiellement par projet
project_queues: dict[int, Queue] = {}
project_locks: dict[int, threading.Lock] = {}
_queues_lock = threading.Lock()  # Lock pour protéger les dictionnaires de queues et locks
document_workers = []
_document_workers_lock = threading.Lock()  # Lock pour la synchronisation des workers

# Reader Docling LlamaIndex partagé (singleton)
_docling_reader = None
_docling_reader_lock = threading.Lock()

# Configurer PyTorch pour CPU avant l'import de docling
# Cela évite les warnings sur pin_memory et optimise pour CPU
try:
    import torch
    import warnings
    
    # Désactiver cuDNN (inutile sans GPU)
    torch.backends.cudnn.enabled = False
    
    # Filtrer les avertissements PyTorch liés à pin_memory et dataloader
    # Ces avertissements apparaissent quand pin_memory=True mais pas de GPU disponible
    warnings.filterwarnings('ignore', message='.*pin_memory.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*dataloader.*pin_memory.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*accelerator.*', category=UserWarning)
    # Filtrer aussi les avertissements généraux de torch.utils.data.dataloader
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.utils.data.dataloader')
    
    # Configurer le nombre de threads PyTorch
    # Par défaut, utiliser tous les cœurs disponibles pour maximiser les performances
    if settings.TORCH_NUM_THREADS is not None:
        torch.set_num_threads(settings.TORCH_NUM_THREADS)
        logger.info(f"PyTorch configuré avec {settings.TORCH_NUM_THREADS} threads (valeur explicite)")
    else:
        # Utiliser tous les cœurs disponibles pour maximiser les performances
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if settings.USE_ALL_CPU_CORES:
            default_threads = cpu_count
            logger.info(f"PyTorch configuré avec {default_threads} threads (tous les {cpu_count} cœurs disponibles)")
        else:
            # Fallback: limiter à la moitié si USE_ALL_CPU_CORES est False
            default_threads = max(1, cpu_count // 2)
            logger.info(f"PyTorch configuré avec {default_threads} threads (moitié des {cpu_count} cœurs disponibles)")
        torch.set_num_threads(default_threads)
    
    # Configurer les variables d'environnement pour OpenMP (utilisé par PyTorch)
    # Supprimer OMP_NUM_THREADS si elle est définie avec une valeur invalide
    if 'OMP_NUM_THREADS' in os.environ:
        omp_value = os.environ['OMP_NUM_THREADS'].strip()
        # Vérifier si la valeur est vide ou invalide
        if not omp_value or not omp_value.isdigit() or int(omp_value) <= 0:
            invalid_value = omp_value
            del os.environ['OMP_NUM_THREADS']
            logger.debug(f"OMP_NUM_THREADS supprimé de l'environnement (valeur invalide: '{invalid_value}')")
    
    # Configurer OMP_NUM_THREADS depuis les settings
    if settings.OMP_NUM_THREADS is not None and settings.OMP_NUM_THREADS > 0:
        os.environ['OMP_NUM_THREADS'] = str(settings.OMP_NUM_THREADS)
        logger.info(f"OMP_NUM_THREADS configuré à {settings.OMP_NUM_THREADS} (valeur explicite)")
    elif 'OMP_NUM_THREADS' not in os.environ:
        # Si OMP_NUM_THREADS n'est pas défini, utiliser tous les cœurs par défaut
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if settings.USE_ALL_CPU_CORES:
            default_omp_threads = cpu_count
            logger.info(f"OMP_NUM_THREADS configuré par défaut à {default_omp_threads} (tous les {cpu_count} cœurs disponibles)")
        else:
            # Fallback: limiter à la moitié si USE_ALL_CPU_CORES est False
            default_omp_threads = max(1, cpu_count // 2)
            logger.info(f"OMP_NUM_THREADS configuré par défaut à {default_omp_threads} (moitié des {cpu_count} cœurs disponibles)")
        os.environ['OMP_NUM_THREADS'] = str(default_omp_threads)
    
    # Gérer le GPU optionnel selon la configuration
    if settings.DOCLING_USE_GPU is False or (settings.DOCLING_CPU_ONLY and settings.DOCLING_USE_GPU is None):
        # Forcer CPU uniquement
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logger.info("GPU désactivé pour Docling (mode CPU uniquement)")
    elif settings.DOCLING_USE_GPU is True:
        # Forcer GPU - ne pas définir CUDA_VISIBLE_DEVICES pour permettre l'accès GPU
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        logger.info("GPU activé pour Docling")
    # Si DOCLING_USE_GPU est None et DOCLING_CPU_ONLY est False, auto-détection (ne rien faire)
    
except ImportError:
    logger.warning("PyTorch non disponible, certaines optimisations CPU ne seront pas appliquées")

from llama_index.readers.docling import DoclingReader


def get_docling_reader():
    """
    Récupère ou crée le reader Docling partagé (singleton).
    Le reader est créé une seule fois et réutilisé pour tous les documents.
    Cela évite de recharger les modèles à chaque document (gain de performance massif).
    
    Returns:
        DoclingReader partagé
    """
    global _docling_reader
    
    with _docling_reader_lock:
        if _docling_reader is None:
            import time
            init_start = time.time()
            logger.info("Initialisation du DoclingReader LlamaIndex avec optimisations (une seule fois)...")
            try:
                # DoclingReader peut accepter des paramètres de configuration dans le futur
                # Pour l'instant, utiliser la configuration par défaut
                _docling_reader = DoclingReader()
                init_time = time.time() - init_start
                logger.info(f"✅ DoclingReader LlamaIndex initialisé en {init_time:.2f}s et prêt à être réutilisé")
            except Exception as e:
                logger.warning(f"Impossible de configurer DoclingReader avec optimisations: {e}")
                _docling_reader = DoclingReader()
                init_time = time.time() - init_start
                logger.info(f"DoclingReader initialisé avec configuration par défaut en {init_time:.2f}s")
        
        return _docling_reader


def process_document(file_path: str) -> Optional[str]:
    """
    Traite un document et le convertit en texte/markdown.
    Stratégie unifiée:
    - Tous les formats passent par DoclingReader (LlamaIndex)
    - Sortie normalisée en markdown/texte exploitable par l'indexation hiérarchique
    
    Args:
        file_path: Chemin vers le fichier à traiter
        
    Returns:
        Contenu du document ou None en cas d'erreur
    """
    import time
    start_time = time.time()
    
    if not os.path.exists(file_path):
        logger.error(f"Fichier non trouvé: {file_path}")
        return None
    
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"Démarrage du traitement via DoclingReader: {file_path} ({file_size_mb:.2f} MB)")
        
        # Récupérer le reader partagé (créé une seule fois)
        reader_init_start = time.time()
        reader = get_docling_reader()
        reader_init_time = time.time() - reader_init_start
        if reader_init_time > 1:
            logger.info(f"DoclingReader initialisé en {reader_init_time:.2f}s")
        
        # Désactiver les messages de progression de Docling
        import sys
        
        class ProgressFilter:
            def __init__(self, original_stream):
                self.original_stream = original_stream
            
            def write(self, text):
                # Filtrer les messages de progression
                if 'Progress:' not in text and 'Complete' not in text and '|' not in text[:20]:
                    self.original_stream.write(text)
            
            def flush(self):
                self.original_stream.flush()
        
        progress_filter_stdout = ProgressFilter(sys.stdout)
        progress_filter_stderr = ProgressFilter(sys.stderr)
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            sys.stdout = progress_filter_stdout
            sys.stderr = progress_filter_stderr
            
            # Conversion unifiée via LlamaIndex DoclingReader
            conversion_start = time.time()
            docs = reader.load_data(file_path=Path(file_path))
            conversion_time = time.time() - conversion_start
            logger.info(f"Conversion Docling terminée en {conversion_time:.2f}s ({conversion_time/60:.2f} min)")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        markdown_parts = []
        for doc in docs or []:
            if hasattr(doc, "get_content"):
                content = doc.get_content() or ""
            else:
                content = str(doc)
            if content.strip():
                markdown_parts.append(content.strip())
        
        markdown_content = "\n\n".join(markdown_parts).strip()
        
        if not markdown_content or not markdown_content.strip():
            logger.warning(f"Le document {file_path} a été traité mais le contenu est vide")
            return None
        
        total_time = time.time() - start_time
        logger.info(f"✅ Document converti avec succès en {total_time:.2f}s ({total_time/60:.2f} min) - {len(markdown_content)} caractères extraits")
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


def _process_document_worker():
    """
    Worker thread qui traite les documents depuis les files d'attente par projet.
    Ce worker est isolé pour ne pas bloquer la navigation de l'application.
    """
    # Essayer de réduire la priorité du thread pour laisser de la place à FastAPI
    try:
        import os
        if hasattr(os, 'nice'):
            # Réduire la priorité du thread (Linux/Unix)
            os.nice(5)  # Augmenter la "niceness" (priorité plus basse)
    except Exception:
        pass  # Ignorer si non disponible (Windows)
    
    logger.info("Worker de traitement de documents démarré et en attente de tâches...")
    while True:
        task = None
        project_id = None
        project_lock = None
        lock_acquired = False
        try:
            # Parcourir les projets qui ont des documents en attente
            with _queues_lock:
                available_projects = [
                    pid for pid, queue in project_queues.items() 
                    if not queue.empty()
                ]
            
            if not available_projects:
                # Aucun projet avec des documents, attendre un peu avant de réessayer
                # Utiliser un sleep plus long pour réduire la consommation CPU quand il n'y a rien à faire
                time.sleep(2)
                continue
            
            # Essayer d'acquérir le verrou d'un projet disponible
            for pid in available_projects:
                with _queues_lock:
                    if pid not in project_locks:
                        # Créer le lock si nécessaire
                        project_locks[pid] = threading.Lock()
                    project_lock = project_locks[pid]
                
                # Essayer d'acquérir le verrou de manière non-bloquante
                if project_lock.acquire(blocking=False):
                    lock_acquired = True
                    try:
                        # Prendre un document de la queue du projet
                        with _queues_lock:
                            if pid not in project_queues or project_queues[pid].empty():
                                project_lock.release()
                                lock_acquired = False
                                continue
                            try:
                                task = project_queues[pid].get(block=False)
                            except Exception:
                                # Queue vide entre-temps, libérer le lock et continuer
                                project_lock.release()
                                lock_acquired = False
                                continue
                        
                        if task is None:  # Signal d'arrêt
                            project_lock.release()
                            lock_acquired = False
                            logger.info("Signal d'arrêt reçu, arrêt du worker de documents")
                            return
                        
                        project_id = pid
                        note_id, file_path = task
                        logger.info(f"Worker traite le document pour la note {note_id} (projet {project_id})")
                        
                        # Traiter le document
                        _process_document_for_note(note_id, file_path)
                        
                        # Marquer la tâche comme terminée
                        with _queues_lock:
                            if project_id in project_queues:
                                project_queues[project_id].task_done()
                        
                        logger.info(f"Worker a terminé le traitement du document pour la note {note_id} (projet {project_id})")
                        
                        # Délai entre les documents pour éviter de surcharger le CPU et laisser de la place pour la navigation
                        # Augmenter le délai pour mieux isoler le traitement
                        time.sleep(1.0)
                        break  # Sortir de la boucle des projets après avoir traité un document
                        
                    except Exception as e:
                        logger.error(f"Erreur dans le worker de traitement de documents: {e}", exc_info=True)
                        # Marquer la tâche comme terminée même en cas d'erreur
                        if task and project_id:
                            try:
                                with _queues_lock:
                                    if project_id in project_queues:
                                        project_queues[project_id].task_done()
                            except Exception:
                                pass
                        raise
                    finally:
                        # Libérer le verrou du projet si acquis
                        if lock_acquired:
                            try:
                                project_lock.release()
                            except Exception:
                                pass
                            lock_acquired = False
                else:
                    # Le verrou est déjà pris par un autre worker, essayer le projet suivant
                    continue
            
        except Exception as e:
            logger.error(f"Erreur dans le worker de traitement de documents: {e}", exc_info=True)
            if lock_acquired and project_lock:
                try:
                    project_lock.release()
                except Exception:
                    pass


def _process_document_for_note(note_id: int, file_path: str):
    """
    Traiter un document pour une note (appelé par le worker).
    
    Args:
        note_id: ID de la note
        file_path: Chemin vers le fichier à traiter
    """
    try:
        logger.info(f"Démarrage du traitement du document pour la note {note_id}")
        
        # Créer une nouvelle session pour ce thread
        with Session(engine) as session:
            # Récupérer la note
            note = session.get(Note, note_id)
            if not note:
                logger.error(f"Note {note_id} non trouvée pour traitement de document")
                return
            
            # Mettre à jour le statut à 'processing'
            note.processing_status = 'processing'
            note.processing_progress = 10
            note.updated_at = datetime.utcnow()
            session.add(note)
            session.commit()
            
            # Traiter le document avec docling
            markdown_content = process_document(file_path)
            
            if not markdown_content:
                # Erreur lors du traitement
                note.processing_status = 'failed'
                note.processing_progress = max(note.processing_progress or 0, 10)
                note.content = "❌ Erreur lors du traitement du document. Le fichier peut être corrompu ou dans un format non supporté."
                note.updated_at = datetime.utcnow()
                session.add(note)
                session.commit()
                logger.error(f"Échec du traitement du document pour la note {note_id}")
                return
            
            # Étape extraction Docling terminée
            note.content = markdown_content
            note.processing_status = 'processing'
            note.processing_progress = 55
            note.updated_at = datetime.utcnow()
            session.add(note)
            session.commit()
            
            logger.info(f"Document traité avec succès pour la note {note_id} ({len(markdown_content)} caractères)")
            
            # Créer les chunks (sans embeddings, rapide)
            try:
                chunks = create_chunks_for_note(session, note, generate_embeddings=False)
                logger.info(f"Créé {len(chunks)} chunks pour la note {note_id}")
                note.processing_progress = 75
                note.updated_at = datetime.utcnow()
                session.add(note)
                session.commit()
                
                # Ajouter la génération d'embeddings à la file d'embeddings
                if chunks:
                    note.processing_progress = 85
                    note.updated_at = datetime.utcnow()
                    session.add(note)
                    session.commit()
                    generate_embeddings_for_chunks_async(note.id, note.project_id)
                    logger.info(f"Tâche de génération d'embeddings ajoutée à la file pour la note {note_id}")
                else:
                    note.processing_status = 'completed'
                    note.processing_progress = 100
                    note.updated_at = datetime.utcnow()
                    session.add(note)
                    session.commit()
            except Exception as e:
                logger.error(f"Erreur lors de la création des chunks pour la note {note_id}: {e}", exc_info=True)
                note.processing_status = 'failed'
                note.processing_progress = max(note.processing_progress or 0, 55)
                note.updated_at = datetime.utcnow()
                session.add(note)
                session.commit()
            
            # Supprimer le fichier original maintenant que le contenu est extrait et stocké en BDD
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"✅ Fichier original supprimé: {file_path}")
                else:
                    logger.warning(f"Fichier déjà supprimé ou introuvable: {file_path}")
            except Exception as e:
                logger.error(f"Erreur lors de la suppression du fichier {file_path}: {e}", exc_info=True)
                # Ne pas faire échouer le traitement si la suppression échoue
            
    except Exception as e:
        logger.error(f"Erreur lors du traitement du document pour la note {note_id}: {e}", exc_info=True)
        # Marquer la note comme failed en cas d'erreur
        try:
            with Session(engine) as session:
                note = session.get(Note, note_id)
                if note:
                    note.processing_status = 'failed'
                    note.processing_progress = max(note.processing_progress or 0, 10)
                    note.content = f"❌ Erreur lors du traitement du document: {str(e)}"
                    note.updated_at = datetime.utcnow()
                    session.add(note)
                    session.commit()
        except Exception as update_error:
            logger.error(f"Erreur lors de la mise à jour du statut d'erreur pour la note {note_id}: {update_error}")


def process_document_async(note_id: int, file_path: str):
    """
    Ajouter un document à la file d'attente pour traitement en arrière-plan.
    Cette fonction est non-bloquante et retourne immédiatement.
    
    Args:
        note_id: ID de la note
        file_path: Chemin vers le fichier à traiter
    """
    # Récupérer le project_id depuis la note en BDD
    try:
        with Session(engine) as session:
            note = session.get(Note, note_id)
            if not note:
                logger.error(f"Note {note_id} non trouvée pour ajout à la queue")
                return
            project_id = note.project_id
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du project_id pour la note {note_id}: {e}", exc_info=True)
        return
    
    # S'assurer que les workers sont démarrés
    _ensure_document_workers()
    
    # Obtenir ou créer la queue et le lock pour ce projet
    with _queues_lock:
        if project_id not in project_queues:
            project_queues[project_id] = Queue()
            project_locks[project_id] = threading.Lock()
            logger.debug(f"Queue et lock créés pour le projet {project_id}")
        
        # Ajouter la tâche à la queue du projet
        project_queues[project_id].put((note_id, file_path))
        queue_size = project_queues[project_id].qsize()
    
    logger.info(f"✅ Tâche de traitement de document ajoutée à la file du projet {project_id} pour la note {note_id} (taille de la file: {queue_size})")


def _ensure_document_workers():
    """S'assurer que les workers de traitement de documents sont démarrés"""
    global document_workers
    
    with _document_workers_lock:
        if not document_workers or not any(w.is_alive() for w in document_workers):
            # Redémarrer les workers s'ils sont morts
            document_workers = []
            num_workers = settings.MAX_CONCURRENT_DOCUMENTS
            for i in range(num_workers):
                worker = threading.Thread(target=_process_document_worker, daemon=True)
                worker.start()
                document_workers.append(worker)
                logger.info(f"Worker de traitement de documents {i+1} démarré")

