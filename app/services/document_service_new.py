from typing import Optional, List
import logging
import os
from pathlib import Path
import threading
import time
from queue import Queue
from sqlmodel import Session, select
from app.config import settings
from app.database import engine
from app.models.document import Document, DocumentCreate, DocumentUpdate
from app.models.document_space import DocumentSpace
from app.models.document_chunk import DocumentChunk
from datetime import datetime

logger = logging.getLogger(__name__)

document_queues: dict[int, Queue] = {}
document_locks: dict[int, threading.Lock] = {}
_queues_lock = threading.Lock()
document_workers = []
_document_workers_lock = threading.Lock()

_docling_converter = None
_docling_converter_lock = threading.Lock()

try:
    import torch
    import warnings

    torch.backends.cudnn.enabled = False
    warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*dataloader.*pin_memory.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*accelerator.*", category=UserWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

    if settings.TORCH_NUM_THREADS is not None:
        torch.set_num_threads(settings.TORCH_NUM_THREADS)
        logger.info("PyTorch configuré avec %d threads (valeur explicite)", settings.TORCH_NUM_THREADS)
    else:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if settings.USE_ALL_CPU_CORES:
            default_threads = cpu_count
            logger.info("PyTorch configuré avec %d threads (tous les %d cœurs disponibles)", default_threads, cpu_count)
        else:
            default_threads = max(1, cpu_count // 2)
            logger.info("PyTorch configuré avec %d threads (moitié des %d cœurs disponibles)", default_threads, cpu_count)
        torch.set_num_threads(default_threads)

    if "OMP_NUM_THREADS" in os.environ:
        omp_value = os.environ["OMP_NUM_THREADS"].strip()
        if not omp_value or not omp_value.isdigit() or int(omp_value) <= 0:
            del os.environ["OMP_NUM_THREADS"]

    if settings.OMP_NUM_THREADS is not None and settings.OMP_NUM_THREADS > 0:
        os.environ["OMP_NUM_THREADS"] = str(settings.OMP_NUM_THREADS)
        logger.info("OMP_NUM_THREADS configuré à %d (valeur explicite)", settings.OMP_NUM_THREADS)
    elif "OMP_NUM_THREADS" not in os.environ:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if settings.USE_ALL_CPU_CORES:
            default_omp_threads = cpu_count
        else:
            default_omp_threads = max(1, cpu_count // 2)
        os.environ["OMP_NUM_THREADS"] = str(default_omp_threads)

    if settings.DOCLING_USE_GPU is False or (settings.DOCLING_CPU_ONLY and settings.DOCLING_USE_GPU is None):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("GPU désactivé pour Docling (mode CPU uniquement)")
    elif settings.DOCLING_USE_GPU is True:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        logger.info("GPU activé pour Docling")

except ImportError:
    logger.warning("PyTorch non disponible, certaines optimisations CPU ne seront pas appliquées")


def get_docling_converter():
    """Récupère ou crée le DocumentConverter Docling partagé (singleton)."""
    global _docling_converter

    with _docling_converter_lock:
        if _docling_converter is None:
            init_start = time.time()
            logger.info("Initialisation du DocumentConverter Docling (une seule fois)...")
            try:
                from docling.document_converter import DocumentConverter

                format_options = None
                ocr_enabled = getattr(settings, "DOCLING_OCR_ENABLED", False)
                if True:
                    try:
                        from docling.datamodel.base_models import InputFormat
                        from docling.datamodel.pipeline_options import PdfPipelineOptions
                        from docling.document_converter import PdfFormatOption

                        pipeline_options = PdfPipelineOptions(
                            do_ocr=ocr_enabled,
                            generate_picture_images=True,
                            images_scale=2.0,
                        )

                        if ocr_enabled:
                            ocr_lang = getattr(settings, "DOCLING_OCR_LANG", None)
                            if ocr_lang:
                                lang_list = [x.strip().lower() for x in ocr_lang.replace("+", ",").split(",") if x.strip()]
                                if lang_list:
                                    try:
                                        from docling.datamodel.pipeline_options import EasyOcrOptions
                                        pipeline_options.ocr_options = EasyOcrOptions(lang=lang_list, use_gpu=settings.DOCLING_USE_GPU is True)
                                        logger.info("OCR Docling activé (EasyOCR, lang=%s)", lang_list)
                                    except ImportError:
                                        try:
                                            from docling.datamodel.pipeline_options import TesseractOcrOptions
                                            pipeline_options.ocr_options = TesseractOcrOptions(lang=ocr_lang)
                                            logger.info("OCR Docling activé (Tesseract, lang=%s)", ocr_lang)
                                        except ImportError:
                                            logger.debug("OCR options non disponibles, do_ocr=True sans ocr_options")
                            else:
                                logger.info("OCR Docling activé (langues par défaut)")

                        format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
                    except Exception as ie:
                        logger.warning("Configuration OCR Docling non disponible (%s), conversion sans OCR", ie)

                _docling_converter = DocumentConverter(format_options=format_options) if format_options else DocumentConverter()
                init_time = time.time() - init_start
                logger.info("✅ DocumentConverter Docling initialisé en %.2fs et prêt à être réutilisé", init_time)
            except Exception as e:
                logger.error("Erreur lors de l'initialisation du DocumentConverter: %s", e, exc_info=True)
                raise

        return _docling_converter


def process_document_file(file_path: str) -> tuple[Optional[str], Optional[list], Optional[object]]:
    """
    Traite un document et retourne (markdown, llama_docs_json, docling_doc).
    """
    import time as _time

    start_time = _time.time()

    if not os.path.exists(file_path):
        logger.error("Fichier non trouvé: %s", file_path)
        return None, None, None

    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info("Démarrage du traitement Docling: %s (%.2f MB)", file_path, file_size_mb)

        converter = get_docling_converter()

        import sys

        class ProgressFilter:
            def __init__(self, original_stream):
                self.original_stream = original_stream

            def write(self, text):
                if "Progress:" not in text and "Complete" not in text and "|" not in text[:20]:
                    self.original_stream.write(text)

            def flush(self):
                self.original_stream.flush()
            
            def fileno(self):
                return self.original_stream.fileno()

        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = ProgressFilter(sys.stdout)
            sys.stderr = ProgressFilter(sys.stderr)

            conversion_start = _time.time()
            result = converter.convert(file_path)
            conversion_time = _time.time() - conversion_start
            logger.info("Conversion Docling terminée en %.2fs (%.2f min)", conversion_time, conversion_time / 60)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        docling_doc = result.document
        markdown_content = docling_doc.export_to_markdown().strip()

        if not markdown_content:
            logger.warning("Le document %s a été traité mais le contenu markdown est vide", file_path)
            return None, None, None

        from llama_index.core import Document as LlamaDocument

        llama_docs = [
            LlamaDocument(
                text=docling_doc.model_dump_json(),
                metadata={"source": str(file_path)},
            )
        ]

        total_time = _time.time() - start_time
        logger.info(
            "✅ Document converti en %.2fs (%.2f min) — %d caractères markdown, %d document(s) JSON Docling",
            total_time, total_time / 60, len(markdown_content), len(llama_docs)
        )
        return markdown_content, llama_docs, docling_doc

    except Exception as e:
        logger.error("Erreur lors du traitement du document %s: %s", file_path, e, exc_info=True)
        return None, None, None


def save_uploaded_file(file_content: bytes, filename: str, upload_dir: str = "media/documents") -> Optional[str]:
    """Sauvegarde un fichier uploadé sur le disque."""
    try:
        upload_path = Path(upload_dir)
        upload_path.mkdir(parents=True, exist_ok=True)

        file_extension = Path(filename).suffix
        import uuid

        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = upload_path / unique_filename

        with open(file_path, "wb") as f:
            f.write(file_content)

        logger.info("Fichier sauvegardé: %s", file_path)
        return str(file_path)

    except Exception as e:
        logger.error("Erreur lors de la sauvegarde du fichier %s: %s", filename, e, exc_info=True)
        return None


def extract_and_save_images(docling_doc, document_id: int) -> list:
    """Extrait les images du document Docling et les sauvegarde sur disque."""
    images_info = []
    images_dir = Path(f"media/images/{document_id}")
    
    try:
        pictures = getattr(docling_doc, "pictures", None)
        if not pictures:
            logger.debug("Aucune image trouvée dans le document pour document_id=%s", document_id)
            return []
        
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, picture in enumerate(pictures):
            try:
                pil_image = None
                
                if hasattr(picture, "get_image") and callable(getattr(picture, "get_image")):
                    try:
                        pil_image = picture.get_image(docling_doc)
                    except Exception as e:
                        logger.debug("get_image() a échoué pour image %d: %s", idx, e)
                
                if pil_image is None:
                    image_ref = getattr(picture, "image", None)
                    if image_ref is not None:
                        if hasattr(image_ref, "pil_image"):
                            pil_image = image_ref.pil_image
                        elif hasattr(image_ref, "save"):
                            pil_image = image_ref
                
                if pil_image is None:
                    logger.warning("Image %d sans données PIL exploitables pour document_id=%s", idx, document_id)
                    continue
                
                image_filename = f"image_{idx}.png"
                image_path = images_dir / image_filename
                pil_image.save(str(image_path), "PNG")
                
                prov = getattr(picture, "prov", None)
                page_no = None
                bbox = None
                if prov and len(prov) > 0:
                    page_no = getattr(prov[0], "page_no", None)
                    bbox_obj = getattr(prov[0], "bbox", None)
                    if bbox_obj:
                        bbox = {
                            "l": getattr(bbox_obj, "l", 0),
                            "t": getattr(bbox_obj, "t", 0),
                            "r": getattr(bbox_obj, "r", 0),
                            "b": getattr(bbox_obj, "b", 0),
                        }
                
                caption = getattr(picture, "caption", "") or ""
                
                images_info.append({
                    "path": str(image_path),
                    "filename": image_filename,
                    "page_no": page_no,
                    "caption": caption,
                    "bbox": bbox,
                    "index": idx,
                })
                
                logger.debug("Image %d extraite : %s (page %s)", idx, image_path, page_no)
                
            except Exception as img_err:
                logger.warning("Erreur extraction image %d pour document_id=%s: %s", idx, document_id, img_err)
                continue
        
        if images_info:
            logger.info("✅ %d image(s) extraite(s) pour document_id=%s dans %s", len(images_info), document_id, images_dir)
        
    except Exception as e:
        logger.error("Erreur lors de l'extraction des images pour document_id=%s: %s", document_id, e, exc_info=True)
    
    return images_info


def create_document(
    session: Session,
    document_create: DocumentCreate,
    library_id: int,
    user_id: int,
    space_ids: List[int]
) -> Optional[Document]:
    """Crée un nouveau document et l'associe aux espaces spécifiés."""
    from app.services.library_service import get_library_by_id
    from app.services.space_service import get_space_by_id
    from app.services.document_space_service import link_document_to_space
    
    library = get_library_by_id(session, library_id, user_id)
    if not library:
        logger.error(f"Bibliothèque {library_id} non trouvée")
        return None
    
    for space_id in space_ids:
        space = get_space_by_id(session, space_id, user_id)
        if not space:
            logger.error(f"Espace {space_id} non trouvé")
            return None
    
    document = Document(
        title=document_create.title,
        content=document_create.content,
        document_type=document_create.document_type,
        source_file_path=document_create.source_file_path,
        processing_status=document_create.processing_status,
        processing_progress=document_create.processing_progress,
        is_paid=document_create.is_paid,
        folder_id=document_create.folder_id,
        library_id=library_id,
        user_id=user_id
    )
    session.add(document)
    session.commit()
    session.refresh(document)
    
    for space_id in space_ids:
        link_document_to_space(session, document.id, space_id, user_id)
    
    logger.info(f"Document créé: {document.title} (ID: {document.id}) accessible dans {len(space_ids)} espace(s)")
    return document


def get_document_by_id(session: Session, document_id: int, user_id: int) -> Optional[Document]:
    """Récupère un document par son ID si il appartient à l'utilisateur."""
    statement = select(Document).where(
        Document.id == document_id,
        Document.user_id == user_id
    )
    return session.exec(statement).first()


def get_documents_by_folder(session: Session, folder_id: Optional[int], library_id: int, user_id: int) -> List[Document]:
    """Récupère tous les documents d'un dossier (ou racine si folder_id est None)."""
    statement = select(Document).where(
        Document.library_id == library_id,
        Document.user_id == user_id,
        Document.folder_id == folder_id
    ).order_by(Document.created_at.desc())
    return list(session.exec(statement).all())


def get_documents_by_library(session: Session, library_id: int, user_id: int) -> List[Document]:
    """Récupère tous les documents d'une bibliothèque."""
    statement = select(Document).where(
        Document.library_id == library_id,
        Document.user_id == user_id
    ).order_by(Document.created_at.desc())
    return list(session.exec(statement).all())


def get_documents_by_space(session: Session, space_id: int, user_id: int) -> List[Document]:
    """Récupère tous les documents accessibles dans un espace."""
    from app.services.document_space_service import get_documents_for_space
    return get_documents_for_space(session, space_id, user_id)


def update_document(
    session: Session,
    document_id: int,
    document_update: DocumentUpdate,
    user_id: int
) -> Optional[Document]:
    """Met à jour un document."""
    document = get_document_by_id(session, document_id, user_id)
    if not document:
        return None
    
    update_data = document_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(document, key, value)
    
    document.updated_at = datetime.utcnow()
    session.add(document)
    session.commit()
    session.refresh(document)
    
    return document


def move_document(
    session: Session,
    document_id: int,
    new_folder_id: Optional[int],
    user_id: int
) -> Optional[Document]:
    """Déplace un document vers un nouveau dossier."""
    from app.services.folder_service import get_folder_by_id
    
    document = get_document_by_id(session, document_id, user_id)
    if not document:
        return None
    
    if new_folder_id:
        folder = get_folder_by_id(session, new_folder_id, user_id)
        if not folder:
            return None
        
        if folder.library_id != document.library_id:
            logger.error(f"Impossible de déplacer le document {document_id} : le dossier n'est pas dans la même bibliothèque")
            return None
    
    document.folder_id = new_folder_id
    document.updated_at = datetime.utcnow()
    session.add(document)
    session.commit()
    session.refresh(document)
    
    return document


def delete_document(session: Session, document_id: int, user_id: int) -> bool:
    """Supprime un document, ses chunks, et toutes ses associations."""
    from app.services.chunk_service import delete_chunks_for_document
    from app.services.document_space_service import get_document_spaces
    
    document = get_document_by_id(session, document_id, user_id)
    if not document:
        return False
    
    delete_chunks_for_document(session, document_id, commit=False)
    
    doc_spaces = get_document_spaces(session, document_id, user_id)
    for doc_space in doc_spaces:
        session.delete(doc_space)
    
    if document.source_file_path and os.path.exists(document.source_file_path):
        try:
            os.remove(document.source_file_path)
            logger.info(f"Fichier source supprimé: {document.source_file_path}")
        except Exception as e:
            logger.warning(f"Impossible de supprimer le fichier source: {e}")
    
    images_dir = Path(f"media/images/{document_id}")
    if images_dir.exists():
        try:
            import shutil
            shutil.rmtree(images_dir)
            logger.info(f"Images supprimées: {images_dir}")
        except Exception as e:
            logger.warning(f"Impossible de supprimer le dossier d'images: {e}")
    
    session.delete(document)
    session.commit()
    
    logger.info(f"Document supprimé: {document.title} (ID: {document_id})")
    return True


def add_document_to_spaces(
    session: Session,
    document_id: int,
    space_ids: List[int],
    user_id: int
) -> bool:
    """Ajoute un document à plusieurs espaces et déclenche l'extraction KAG."""
    from app.services.document_space_service import link_document_to_space
    from app.services.kag_graph_service import process_kag_for_document_space
    
    document = get_document_by_id(session, document_id, user_id)
    if not document:
        return False
    
    for space_id in space_ids:
        link_document_to_space(session, document_id, space_id, user_id)
        
        if document.processing_status == "completed":
            process_kag_for_document_space(session, document_id, space_id)
    
    return True


def remove_document_from_spaces(
    session: Session,
    document_id: int,
    space_ids: List[int],
    user_id: int
) -> bool:
    """Retire un document de plusieurs espaces."""
    from app.services.document_space_service import unlink_document_from_space
    
    for space_id in space_ids:
        unlink_document_from_space(session, document_id, space_id, user_id)
    
    return True


def _process_document_worker():
    """Worker thread qui traite les documents depuis les files d'attente."""
    try:
        import os
        if hasattr(os, "nice"):
            os.nice(5)
    except Exception:
        pass

    logger.info("Worker de traitement de documents démarré et en attente de tâches...")
    while True:
        task = None
        library_id = None
        library_lock = None
        lock_acquired = False
        try:
            with _queues_lock:
                available_libraries = [lid for lid, queue in document_queues.items() if not queue.empty()]

            if not available_libraries:
                time.sleep(2)
                continue

            for lid in available_libraries:
                with _queues_lock:
                    if lid not in document_locks:
                        document_locks[lid] = threading.Lock()
                    library_lock = document_locks[lid]

                if library_lock.acquire(blocking=False):
                    lock_acquired = True
                    try:
                        with _queues_lock:
                            if lid not in document_queues or document_queues[lid].empty():
                                library_lock.release()
                                lock_acquired = False
                                continue
                            try:
                                task = document_queues[lid].get(block=False)
                            except Exception:
                                library_lock.release()
                                lock_acquired = False
                                continue

                        if task is None:
                            library_lock.release()
                            lock_acquired = False
                            logger.info("Signal d'arrêt reçu, arrêt du worker de documents")
                            return

                        library_id = lid
                        document_id, file_path = task
                        logger.info("Worker traite le document %d (bibliothèque %d)", document_id, library_id)

                        _process_document_for_id(document_id, file_path)

                        with _queues_lock:
                            if library_id in document_queues:
                                document_queues[library_id].task_done()

                        logger.info("Worker a terminé le traitement du document %d", document_id)
                        time.sleep(1.0)
                        break

                    except Exception as e:
                        logger.error("Erreur dans le worker de traitement de documents: %s", e, exc_info=True)
                        if task and library_id:
                            try:
                                with _queues_lock:
                                    if library_id in document_queues:
                                        document_queues[library_id].task_done()
                            except Exception:
                                pass
                        raise
                    finally:
                        if lock_acquired:
                            try:
                                library_lock.release()
                            except Exception:
                                pass
                            lock_acquired = False
                else:
                    continue

        except Exception as e:
            logger.error("Erreur dans le worker de traitement de documents: %s", e, exc_info=True)
            if lock_acquired and library_lock:
                try:
                    library_lock.release()
                except Exception:
                    pass


def _process_document_for_id(document_id: int, file_path: str):
    """Traite un document pour un ID donné."""
    from app.services.chunk_service import create_chunks_for_document, create_chunks_for_document_from_docling, generate_embeddings_for_chunks_async
    from app.services.document_space_service import get_document_spaces
    from app.services.kag_graph_service import process_kag_for_document_space
    
    try:
        logger.info("Démarrage du traitement du document %d", document_id)

        with Session(engine) as session:
            document = session.get(Document, document_id)
            if not document:
                logger.error("Document %d non trouvé pour traitement", document_id)
                return

            document.processing_status = "processing"
            document.processing_progress = 10
            document.updated_at = datetime.utcnow()
            session.add(document)
            session.commit()

            markdown_content, llama_docs, docling_doc = process_document_file(file_path)

            if not markdown_content:
                document.processing_status = "failed"
                document.processing_progress = max(document.processing_progress or 0, 10)
                document.content = "❌ Erreur lors du traitement du document. Le fichier peut être corrompu ou dans un format non supporté."
                document.updated_at = datetime.utcnow()
                session.add(document)
                session.commit()
                logger.error("Échec du traitement du document %d", document_id)
                return

            file_extension = Path(file_path).suffix.lower()
            permanent_path = Path(f"media/documents/{document_id}{file_extension}")
            permanent_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.move(file_path, str(permanent_path))
            document.source_file_path = str(permanent_path)
            logger.info("Fichier déplacé vers chemin permanent: %s", permanent_path)
            
            document.content = markdown_content
            document.processing_status = "processing"
            document.processing_progress = 55
            document.updated_at = datetime.utcnow()
            session.add(document)
            session.commit()

            logger.info("Document traité avec succès pour document_id %d (%d caractères extraits)", document_id, len(markdown_content))
            
            if docling_doc:
                images_info = extract_and_save_images(docling_doc, document_id)
                if images_info:
                    logger.info("%d image(s) extraite(s) pour le document %d", len(images_info), document_id)

            try:
                if llama_docs:
                    chunks = create_chunks_for_document_from_docling(session, document, llama_docs, generate_embeddings=False)
                    logger.info("Créé %d chunks (DoclingNodeParser) pour le document %d", len(chunks), document_id)
                else:
                    chunks = create_chunks_for_document(session, document, generate_embeddings=False)
                    logger.info("Créé %d chunks (HierarchicalNodeParser fallback) pour le document %d", len(chunks), document_id)

                document.processing_progress = 75
                document.updated_at = datetime.utcnow()
                session.add(document)
                session.commit()

                if chunks:
                    document.processing_progress = 85
                    document.updated_at = datetime.utcnow()
                    session.add(document)
                    session.commit()
                    generate_embeddings_for_chunks_async(document.id, document.library_id)
                    logger.info("Tâche de génération d'embeddings ajoutée à la file pour le document %d", document_id)
                else:
                    document.processing_status = "completed"
                    document.processing_progress = 100
                    document.updated_at = datetime.utcnow()
                    session.add(document)
                    session.commit()

            except Exception as e:
                logger.error("Erreur lors de la création des chunks pour le document %d: %s", document_id, e, exc_info=True)
                document.processing_status = "failed"
                document.processing_progress = max(document.processing_progress or 0, 55)
                document.updated_at = datetime.utcnow()
                session.add(document)
                session.commit()

    except Exception as e:
        logger.error("Erreur lors du traitement du document %d: %s", document_id, e, exc_info=True)
        try:
            with Session(engine) as session:
                document = session.get(Document, document_id)
                if document:
                    document.processing_status = "failed"
                    document.processing_progress = max(document.processing_progress or 0, 10)
                    document.content = f"❌ Erreur lors du traitement du document: {str(e)}"
                    document.updated_at = datetime.utcnow()
                    session.add(document)
                    session.commit()
        except Exception as update_error:
            logger.error("Erreur lors de la mise à jour du statut d'erreur pour le document %d: %s", document_id, update_error)


def process_document_async(document_id: int, file_path: str):
    """Ajoute un document à la file d'attente pour traitement en arrière-plan."""
    try:
        with Session(engine) as session:
            document = session.get(Document, document_id)
            if not document:
                logger.error("Document %d non trouvé pour ajout à la queue", document_id)
                return
            library_id = document.library_id
    except Exception as e:
        logger.error("Erreur lors de la récupération du library_id pour le document %d: %s", document_id, e, exc_info=True)
        return

    _ensure_document_workers()

    with _queues_lock:
        if library_id not in document_queues:
            document_queues[library_id] = Queue()
            document_locks[library_id] = threading.Lock()

        document_queues[library_id].put((document_id, file_path))
        queue_size = document_queues[library_id].qsize()

    logger.info("✅ Tâche de traitement de document ajoutée à la file de la bibliothèque %d pour le document %d (taille de la file: %d)", library_id, document_id, queue_size)


def _ensure_document_workers():
    """S'assurer que les workers de traitement de documents sont démarrés."""
    global document_workers

    with _document_workers_lock:
        if not document_workers or not any(w.is_alive() for w in document_workers):
            document_workers = []
            # Traitement volontairement séquentiel: 1 seul worker global.
            num_workers = 1
            for i in range(num_workers):
                worker = threading.Thread(target=_process_document_worker, daemon=True)
                worker.start()
                document_workers.append(worker)
                logger.info("Worker de traitement de documents %d démarré", i + 1)
