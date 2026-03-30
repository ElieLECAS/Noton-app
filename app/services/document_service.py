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
from app.services.chunk_service import (
    create_chunks_for_note,
    create_chunks_for_note_from_docling,
    complete_note_embeddings_and_kag_sync,
    generate_embeddings_for_chunks_async,
)
from datetime import datetime

logger = logging.getLogger(__name__)

# Files d'attente par projet pour traiter les documents séquentiellement par projet
project_queues: dict[int, Queue] = {}
project_locks: dict[int, threading.Lock] = {}
_queues_lock = threading.Lock()
document_workers = []
_document_workers_lock = threading.Lock()

# DocumentConverter Docling partagé (singleton) — remplace DoclingReader
_docling_converter = None
_docling_converter_generic = None
_docling_converter_lock = threading.Lock()

# Configurer PyTorch pour CPU avant l'import de docling
try:
    import torch
    import warnings

    torch.backends.cudnn.enabled = False

    warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)
    warnings.filterwarnings(
        "ignore", message=".*dataloader.*pin_memory.*", category=UserWarning
    )
    warnings.filterwarnings("ignore", message=".*accelerator.*", category=UserWarning)
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.data.dataloader"
    )

    if settings.TORCH_NUM_THREADS is not None:
        torch.set_num_threads(settings.TORCH_NUM_THREADS)
        logger.info(
            "PyTorch configuré avec %d threads (valeur explicite)",
            settings.TORCH_NUM_THREADS,
        )
    else:
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        if settings.USE_ALL_CPU_CORES:
            default_threads = cpu_count
            logger.info(
                "PyTorch configuré avec %d threads (tous les %d cœurs disponibles)",
                default_threads,
                cpu_count,
            )
        else:
            default_threads = max(1, cpu_count // 2)
            logger.info(
                "PyTorch configuré avec %d threads (moitié des %d cœurs disponibles)",
                default_threads,
                cpu_count,
            )
        torch.set_num_threads(default_threads)

    if "OMP_NUM_THREADS" in os.environ:
        omp_value = os.environ["OMP_NUM_THREADS"].strip()
        if not omp_value or not omp_value.isdigit() or int(omp_value) <= 0:
            del os.environ["OMP_NUM_THREADS"]

    if settings.OMP_NUM_THREADS is not None and settings.OMP_NUM_THREADS > 0:
        os.environ["OMP_NUM_THREADS"] = str(settings.OMP_NUM_THREADS)
        logger.info(
            "OMP_NUM_THREADS configuré à %d (valeur explicite)", settings.OMP_NUM_THREADS
        )
    elif "OMP_NUM_THREADS" not in os.environ:
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        if settings.USE_ALL_CPU_CORES:
            default_omp_threads = cpu_count
        else:
            default_omp_threads = max(1, cpu_count // 2)
        os.environ["OMP_NUM_THREADS"] = str(default_omp_threads)

    if settings.DOCLING_USE_GPU is False or (
        settings.DOCLING_CPU_ONLY and settings.DOCLING_USE_GPU is None
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("GPU désactivé pour Docling (mode CPU uniquement)")
    elif settings.DOCLING_USE_GPU is True:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        logger.info("GPU activé pour Docling")

except ImportError:
    logger.warning(
        "PyTorch non disponible, certaines optimisations CPU ne seront pas appliquées"
    )


def get_docling_converter(file_path: Optional[str] = None):
    """
    Récupère ou crée le DocumentConverter Docling partagé (singleton).

    Utiliser le DocumentConverter directement (plutôt que DoclingReader) permet
    d'obtenir en une seule passe :
    - le markdown pour note.content (export_to_markdown)
    - le JSON sérialisé pour DoclingNodeParser (model_dump_json)
    """
    global _docling_converter, _docling_converter_generic

    with _docling_converter_lock:
        # Utiliser un convertisseur générique pour les formats non-PDF
        # afin d'éviter de restreindre Docling aux seules options PDF.
        suffix = Path(file_path).suffix.lower() if file_path else ""
        use_generic_converter = suffix and suffix != ".pdf"

        if use_generic_converter:
            if _docling_converter_generic is None:
                init_start = time.time()
                logger.info(
                    "Initialisation du DocumentConverter Docling (générique)..."
                )
                try:
                    from docling.document_converter import DocumentConverter

                    _docling_converter_generic = DocumentConverter()
                    init_time = time.time() - init_start
                    logger.info(
                        "✅ DocumentConverter Docling générique initialisé en %.2fs",
                        init_time,
                    )
                except Exception as e:
                    logger.error(
                        "Erreur lors de l'initialisation du DocumentConverter générique: %s",
                        e,
                        exc_info=True,
                    )
                    raise
            return _docling_converter_generic

        if _docling_converter is None:
            init_start = time.time()
            logger.info(
                "Initialisation du DocumentConverter Docling (une seule fois)..."
            )
            try:
                from docling.document_converter import DocumentConverter

                format_options = None
                ocr_enabled = getattr(settings, "DOCLING_OCR_ENABLED", False)
                # Toujours générer les images pour extraction et stockage (sans Vision/chunks)
                if True:
                    try:
                        from docling.datamodel.base_models import InputFormat
                        from docling.datamodel.pipeline_options import (
                            PdfPipelineOptions,
                        )
                        from docling.document_converter import PdfFormatOption

                        # Échelle d'image ajustable pour OCR (3.0 par défaut pour meilleure qualité)
                        image_scale = getattr(settings, "OCR_IMAGE_SCALE", 3.0)
                        
                        pipeline_options = PdfPipelineOptions(
                            do_ocr=ocr_enabled,
                            generate_picture_images=True,
                            images_scale=image_scale,
                        )

                        # Configuration OCR si activé
                        if ocr_enabled:
                            ocr_lang = getattr(settings, "DOCLING_OCR_LANG", None)
                            if ocr_lang:
                                lang_list = [
                                    x.strip().lower() for x in ocr_lang.replace("+", ",").split(",")
                                    if x.strip()
                                ]
                                if lang_list:
                                    try:
                                        from docling.datamodel.pipeline_options import (
                                            EasyOcrOptions,
                                        )

                                        pipeline_options.ocr_options = EasyOcrOptions(
                                            lang=lang_list,
                                            use_gpu=settings.DOCLING_USE_GPU is True,
                                        )
                                        logger.info(
                                            "OCR Docling activé (EasyOCR, lang=%s)",
                                            lang_list,
                                        )
                                    except ImportError:
                                        try:
                                            from docling.datamodel.pipeline_options import (
                                                TesseractOcrOptions,
                                            )

                                            pipeline_options.ocr_options = (
                                                TesseractOcrOptions(lang=ocr_lang)
                                            )
                                            logger.info(
                                                "OCR Docling activé (Tesseract, lang=%s)",
                                                ocr_lang,
                                            )
                                        except ImportError:
                                            logger.debug(
                                                "OCR options non disponibles, do_ocr=True sans ocr_options"
                                            )
                            else:
                                logger.info(
                                    "OCR Docling activé (langues par défaut)"
                                )

                        format_options = {
                            InputFormat.PDF: PdfFormatOption(
                                pipeline_options=pipeline_options
                            ),
                        }
                    except Exception as ie:
                        logger.warning(
                            "Configuration OCR Docling non disponible (%s), conversion sans OCR",
                            ie,
                        )

                _docling_converter = (
                    DocumentConverter(format_options=format_options)
                    if format_options
                    else DocumentConverter()
                )
                init_time = time.time() - init_start
                logger.info(
                    "✅ DocumentConverter Docling initialisé en %.2fs et prêt à être réutilisé",
                    init_time,
                )
            except Exception as e:
                logger.error(
                    "Erreur lors de l'initialisation du DocumentConverter: %s",
                    e,
                    exc_info=True,
                )
                raise

        return _docling_converter


def ensure_pdf_for_docling(file_path: str) -> str:
    """
    Convertit les formats bureautiques (ODT/DOCX/DOC/...) en PDF via LibreOffice
    avant de les passer à Docling.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return file_path

    convertible_exts = {
        ".odt",
        ".odm",
        ".odg",
        ".odp",
        ".ods",
        ".odf",
        ".doc",
        ".docx",
        ".rtf",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
    }

    # Pour tout autre format (ex: EPUB), on laisse Docling gérer nativement.
    if suffix not in convertible_exts:
        logger.info(
            "Format %s traité directement par Docling sans conversion PDF: %s",
            suffix or "(sans extension)",
            file_path,
        )
        return file_path

    import subprocess

    pdf_path = Path(file_path).with_suffix(".pdf")
    try:
        if pdf_path.exists():
            pdf_path.unlink()
    except Exception:
        pass

    logger.info("Conversion LibreOffice vers PDF: %s", file_path)
    cmd = [
        "libreoffice",
        "--headless",
        "--nologo",
        "--nolockcheck",
        "--convert-to",
        "pdf",
        "--outdir",
        str(pdf_path.parent),
        str(file_path),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-2000:]
        stdout_tail = (proc.stdout or "").strip()[-2000:]
        details = stderr_tail or stdout_tail or f"code={proc.returncode}"
        raise RuntimeError(f"LibreOffice a échoué pour {file_path}: {details}")

    if not pdf_path.exists():
        candidates = sorted(
            pdf_path.parent.glob(f"{Path(file_path).stem}*.pdf"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"Aucun PDF généré pour {file_path}")
        pdf_path = candidates[0]

    return str(pdf_path)


def process_document(file_path: str) -> tuple[Optional[str], Optional[list], Optional[object]]:
    """
    Traite un document et retourne (markdown, llama_docs_json, docling_doc).

    Une seule passe Docling produit :
    - markdown : texte lisible stocké dans note.content (pour l'affichage)
    - llama_docs : liste de LlamaIndex Document en format JSON Docling,
      prête à être consommée par DoclingNodeParser pour un chunking sémantique
    - docling_doc : document Docling original (pour extraction d'images multimodal)

    Args:
        file_path: Chemin vers le fichier à traiter

    Returns:
        (markdown_content, llama_docs, docling_doc) ou (None, None, None) en cas d'erreur
    """
    import time as _time

    start_time = _time.time()

    if not os.path.exists(file_path):
        logger.error("Fichier non trouvé: %s", file_path)
        return None, None, None

    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(
            "Démarrage du traitement Docling: %s (%.2f MB)", file_path, file_size_mb
        )

        converter = get_docling_converter(file_path)

        # Filtrer les messages de progression Docling
        import sys

        class ProgressFilter:
            def __init__(self, original_stream):
                self.original_stream = original_stream

            def write(self, text):
                if (
                    "Progress:" not in text
                    and "Complete" not in text
                    and "|" not in text[:20]
                ):
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
            logger.info(
                "Conversion Docling terminée en %.2fs (%.2f min)",
                conversion_time,
                conversion_time / 60,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        docling_doc = result.document

        # Markdown pour note.content (affichage)
        markdown_content = docling_doc.export_to_markdown().strip()

        # Logique de fallback OCR si activée et résultat insuffisant
        if getattr(settings, "OCR_FALLBACK_ENABLED", True):
            try:
                from app.services.ocr_fallback import extract_with_fallback
                
                # Appliquer le fallback si nécessaire
                markdown_content = extract_with_fallback(
                    file_path=file_path,
                    docling_result=markdown_content,
                    docling_doc=docling_doc,
                    fallback_enabled=True
                )
            except Exception as e:
                logger.warning(
                    "Erreur lors du fallback OCR, utilisation résultat Docling: %s",
                    e
                )

        if not markdown_content:
            logger.warning(
                "Le document %s a été traité mais le contenu markdown est vide",
                file_path,
            )
            return None, None, None

        # Format JSON obligatoire pour DoclingNodeParser : préserve la structure
        # hiérarchique des tableaux (colonnes/lignes). Le parser re-parse ce JSON en interne
        # (double coût connu ; l'API n'accepte pas d'objet DoclingDocument natif).
        from llama_index.core import Document as LlamaDocument

        llama_docs = [
            LlamaDocument(
                text=docling_doc.model_dump_json(),
                metadata={"source": str(file_path)},
            )
        ]

        total_time = _time.time() - start_time
        logger.info(
            "✅ Document converti en %.2fs (%.2f min) — %d caractères markdown, "
            "%d document(s) JSON Docling",
            total_time,
            total_time / 60,
            len(markdown_content),
            len(llama_docs),
        )
        return markdown_content, llama_docs, docling_doc

    except Exception as e:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".epub":
            logger.warning(
                "Docling n'a pas pu traiter l'EPUB (%s). Fallback extraction texte EPUB.",
                e,
            )
            fallback_markdown = extract_text_from_epub(file_path)
            if fallback_markdown:
                logger.info(
                    "✅ Fallback EPUB réussi: %d caractères extraits",
                    len(fallback_markdown),
                )
                return fallback_markdown, None, None

        logger.error(
            "Erreur lors du traitement du document %s: %s", file_path, e, exc_info=True
        )
        return None, None, None


def extract_text_from_epub(file_path: str) -> Optional[str]:
    """
    Extrait le texte brut d'un EPUB via stdlib (zip+xml/html), sans dépendance externe.
    """
    import re
    import zipfile
    from html import unescape

    try:
        chunks: list[str] = []
        with zipfile.ZipFile(file_path, "r") as zf:
            names = [
                n for n in zf.namelist()
                if n.lower().endswith((".xhtml", ".html", ".htm"))
            ]
            for name in sorted(names):
                try:
                    raw = zf.read(name)
                    text = raw.decode("utf-8", errors="ignore")
                except Exception:
                    continue

                # Retirer scripts/styles puis balises HTML.
                text = re.sub(
                    r"<(script|style)\b[^>]*>.*?</\1>",
                    " ",
                    text,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                text = re.sub(r"<[^>]+>", " ", text)
                text = unescape(text)
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    chunks.append(text)

        if not chunks:
            return None

        return "\n\n".join(chunks)
    except Exception as e:
        logger.error("Fallback EPUB échoué pour %s: %s", file_path, e, exc_info=True)
        return None


def save_uploaded_file(
    file_content: bytes, filename: str, upload_dir: str = "media/documents"
) -> Optional[str]:
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
        logger.error(
            "Erreur lors de la sauvegarde du fichier %s: %s", filename, e, exc_info=True
        )
        return None


def extract_and_save_images(docling_doc, note_id: int) -> list:
    """
    Extrait les images du document Docling et les sauvegarde sur disque.
    
    Args:
        docling_doc: Document Docling contenant les images extraites
        note_id: ID de la note pour organiser les images
        
    Returns:
        Liste de dictionnaires avec les infos de chaque image:
        - path: chemin du fichier image
        - page_no: numéro de page
        - caption: légende de l'image
        - bbox: bounding box dans le document
    """
    images_info = []
    images_dir = Path(f"media/images/{note_id}")
    
    try:
        # Vérifier si le document a des images
        pictures = getattr(docling_doc, "pictures", None)
        if not pictures:
            logger.debug("Aucune image trouvée dans le document pour note_id=%s", note_id)
            return []
        
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, picture in enumerate(pictures):
            try:
                # Récupérer l'image PIL via l'API Docling
                pil_image = None
                
                # Méthode 1: PictureItem.get_image(document) - API Docling 2.x recommandée
                if hasattr(picture, "get_image") and callable(getattr(picture, "get_image")):
                    try:
                        pil_image = picture.get_image(docling_doc)
                    except Exception as e:
                        logger.debug("get_image() a échoué pour image %d: %s", idx, e)
                
                # Méthode 2: Accès via picture.image.pil_image
                if pil_image is None:
                    image_ref = getattr(picture, "image", None)
                    if image_ref is not None:
                        if hasattr(image_ref, "pil_image"):
                            pil_image = image_ref.pil_image
                        elif hasattr(image_ref, "save"):
                            pil_image = image_ref
                
                if pil_image is None:
                    logger.warning("Image %d sans données PIL exploitables pour note_id=%s", idx, note_id)
                    continue
                
                # Sauvegarder l'image
                image_filename = f"image_{idx}.png"
                image_path = images_dir / image_filename
                pil_image.save(str(image_path), "PNG")
                
                # Extraire les métadonnées
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
                
                logger.debug(
                    "Image %d extraite : %s (page %s)",
                    idx, image_path, page_no
                )
                
            except Exception as img_err:
                logger.warning(
                    "Erreur extraction image %d pour note_id=%s: %s",
                    idx, note_id, img_err
                )
                continue
        
        if images_info:
            logger.info(
                "✅ %d image(s) extraite(s) pour note_id=%s dans %s",
                len(images_info), note_id, images_dir
            )
        
    except Exception as e:
        logger.error(
            "Erreur lors de l'extraction des images pour note_id=%s: %s",
            note_id, e, exc_info=True
        )
    
    return images_info


def _process_document_worker():
    """
    Worker thread qui traite les documents depuis les files d'attente par projet.
    """
    try:
        import os

        if hasattr(os, "nice"):
            os.nice(5)
    except Exception:
        pass

    logger.info("Worker de traitement de documents démarré et en attente de tâches...")
    while True:
        task = None
        project_id = None
        project_lock = None
        lock_acquired = False
        try:
            with _queues_lock:
                available_projects = [
                    pid
                    for pid, queue in project_queues.items()
                    if not queue.empty()
                ]

            if not available_projects:
                time.sleep(2)
                continue

            for pid in available_projects:
                with _queues_lock:
                    if pid not in project_locks:
                        project_locks[pid] = threading.Lock()
                    project_lock = project_locks[pid]

                if project_lock.acquire(blocking=False):
                    lock_acquired = True
                    try:
                        with _queues_lock:
                            if pid not in project_queues or project_queues[pid].empty():
                                project_lock.release()
                                lock_acquired = False
                                continue
                            try:
                                task = project_queues[pid].get(block=False)
                            except Exception:
                                project_lock.release()
                                lock_acquired = False
                                continue

                        if task is None:
                            project_lock.release()
                            lock_acquired = False
                            logger.info(
                                "Signal d'arrêt reçu, arrêt du worker de documents"
                            )
                            return

                        project_id = pid
                        note_id, file_path = task
                        logger.info(
                            "Worker traite le document pour la note %d (projet %d)",
                            note_id,
                            project_id,
                        )

                        _process_document_for_note(note_id, file_path)

                        with _queues_lock:
                            if project_id in project_queues:
                                project_queues[project_id].task_done()

                        logger.info(
                            "Worker a terminé le traitement du document pour la note %d (projet %d)",
                            note_id,
                            project_id,
                        )

                        time.sleep(1.0)
                        break

                    except Exception as e:
                        logger.error(
                            "Erreur dans le worker de traitement de documents: %s",
                            e,
                            exc_info=True,
                        )
                        if task and project_id:
                            try:
                                with _queues_lock:
                                    if project_id in project_queues:
                                        project_queues[project_id].task_done()
                            except Exception:
                                pass
                        raise
                    finally:
                        if lock_acquired:
                            try:
                                project_lock.release()
                            except Exception:
                                pass
                            lock_acquired = False
                else:
                    continue

        except Exception as e:
            logger.error(
                "Erreur dans le worker de traitement de documents: %s", e, exc_info=True
            )
            if lock_acquired and project_lock:
                try:
                    project_lock.release()
                except Exception:
                    pass


def _process_document_for_note(
    note_id: int, file_path: str, sync_embeddings: bool = False
):
    """
    Traiter un document pour une note (appelé par le worker).

    Utilise DocumentConverter Docling directement pour obtenir en une seule passe
    le markdown (note.content) et les LlamaIndex Documents JSON (pour DoclingNodeParser).
    """
    try:
        logger.info("Démarrage du traitement du document pour la note %d", note_id)
        run_embeddings_sync = False
        project_id_for_sync = None

        with Session(engine) as session:
            note = session.get(Note, note_id)
            if not note:
                logger.error(
                    "Note %d non trouvée pour traitement de document", note_id
                )
                return

            note.processing_status = "processing"
            note.processing_progress = 10
            note.updated_at = datetime.utcnow()
            session.add(note)
            session.commit()

            original_ext = Path(file_path).suffix.lower()
            pdf_input_path = file_path
            converted_to_pdf = False
            try:
                pdf_input_path = ensure_pdf_for_docling(file_path)
                converted_to_pdf = pdf_input_path != file_path
            except Exception as e:
                logger.error(
                    "Conversion vers PDF impossible pour la note %d (%s): %s",
                    note_id,
                    file_path,
                    e,
                    exc_info=True,
                )
                raise

            # Une seule passe Docling → markdown + llama_docs JSON + docling_doc (pour images)
            markdown_content, llama_docs, docling_doc = process_document(pdf_input_path)

            if not markdown_content:
                note.processing_status = "failed"
                note.processing_progress = max(note.processing_progress or 0, 10)
                note.content = (
                    "❌ Erreur lors du traitement du document. "
                    "Le fichier peut être corrompu ou dans un format non supporté."
                )
                note.updated_at = datetime.utcnow()
                session.add(note)
                session.commit()
                logger.error("Échec du traitement du document pour la note %d", note_id)
                return

            output_ext = Path(pdf_input_path).suffix.lower() or ".bin"
            permanent_output_path = Path(f"media/documents/{note_id}{output_ext}")
            permanent_output_path.parent.mkdir(parents=True, exist_ok=True)

            import shutil

            # Retenter proprement sur relances/erreurs précédentes
            if permanent_output_path.exists():
                permanent_output_path.unlink()

            # Si conversion effectuée, on conserve aussi l'original dans le dossier du document.
            if converted_to_pdf:
                original_permanent_path = Path(
                    f"media/documents/{note_id}{original_ext}"
                )
                original_permanent_path.parent.mkdir(parents=True, exist_ok=True)
                if original_permanent_path.exists():
                    original_permanent_path.unlink()
                shutil.move(file_path, str(original_permanent_path))

            shutil.move(pdf_input_path, str(permanent_output_path))
            note.source_file_path = str(permanent_output_path)
            logger.info(
                "Fichier traité déplacé vers chemin permanent: %s",
                permanent_output_path,
            )
            
            # Garder le markdown dans note.content pour le RAG (chunking, fallback, recherche).
            # L’UI affiche le PDF via source_file_path, pas le contenu éditable.
            note.content = markdown_content
            note.processing_status = "processing"
            note.processing_progress = 55
            note.updated_at = datetime.utcnow()
            session.add(note)
            session.commit()

            logger.info(
                "Document traité avec succès pour la note %d (%d caractères extraits)",
                note_id,
                len(markdown_content),
            )
            
            # Extraction et stockage des images (sans Vision ni chunks image)
            if docling_doc:
                images_info = extract_and_save_images(docling_doc, note_id)
                if images_info:
                    logger.info(
                        "%d image(s) extraite(s) pour la note %d",
                        len(images_info), note_id
                    )

            try:
                # Chunking sémantique via DoclingNodeParser si llama_docs disponibles,
                # sinon fallback sur HierarchicalNodeParser
                if llama_docs:
                    chunks = create_chunks_for_note_from_docling(
                        session, note, llama_docs,
                        generate_embeddings=False,
                    )
                    logger.info(
                        "Créé %d chunks (DoclingNodeParser) pour la note %d",
                        len(chunks),
                        note_id,
                    )
                else:
                    chunks = create_chunks_for_note(
                        session, note, generate_embeddings=False
                    )
                    logger.info(
                        "Créé %d chunks (HierarchicalNodeParser fallback) pour la note %d",
                        len(chunks),
                        note_id,
                    )

                note.processing_progress = 75
                note.updated_at = datetime.utcnow()
                session.add(note)
                session.commit()

                if chunks:
                    note.processing_progress = 85
                    note.updated_at = datetime.utcnow()
                    session.add(note)
                    session.commit()
                    if sync_embeddings:
                        run_embeddings_sync = True
                        project_id_for_sync = note.project_id
                        logger.info(
                            "Embeddings/KAG synchrones activés pour la note %d",
                            note_id,
                        )
                    else:
                        generate_embeddings_for_chunks_async(note.id, note.project_id)
                        logger.info(
                            "Tâche de génération d'embeddings ajoutée à la file pour la note %d",
                            note_id,
                        )
                else:
                    note.processing_status = "completed"
                    note.processing_progress = 100
                    note.updated_at = datetime.utcnow()
                    session.add(note)
                    session.commit()

            except Exception as e:
                logger.error(
                    "Erreur lors de la création des chunks pour la note %d: %s",
                    note_id,
                    e,
                    exc_info=True,
                )
                note.processing_status = "failed"
                note.processing_progress = max(note.processing_progress or 0, 55)
                note.updated_at = datetime.utcnow()
                session.add(note)
                session.commit()

            # Le fichier traité est conservé de manière permanente dans media/documents/{note_id}{ext}
            # (PDF converti ou format natif Docling selon le type d'entrée).

        if run_embeddings_sync and project_id_for_sync is not None:
            complete_note_embeddings_and_kag_sync(note_id, project_id_for_sync)

    except Exception as e:
        logger.error(
            "Erreur lors du traitement du document pour la note %d: %s",
            note_id,
            e,
            exc_info=True,
        )
        try:
            with Session(engine) as session:
                note = session.get(Note, note_id)
                if note:
                    note.processing_status = "failed"
                    note.processing_progress = max(note.processing_progress or 0, 10)
                    note.content = (
                        f"❌ Erreur lors du traitement du document: {str(e)}"
                    )
                    note.updated_at = datetime.utcnow()
                    session.add(note)
                    session.commit()
        except Exception as update_error:
            logger.error(
                "Erreur lors de la mise à jour du statut d'erreur pour la note %d: %s",
                note_id,
                update_error,
            )


def enqueue_project_document_thread(note_id: int, file_path: str):
    """Ajoute un document à la file thread du projet (sans Celery)."""
    try:
        with Session(engine) as session:
            note = session.get(Note, note_id)
            if not note:
                logger.error(
                    "Note %d non trouvée pour ajout à la queue", note_id
                )
                return
            project_id = note.project_id
    except Exception as e:
        logger.error(
            "Erreur lors de la récupération du project_id pour la note %d: %s",
            note_id,
            e,
            exc_info=True,
        )
        return

    _ensure_document_workers()

    with _queues_lock:
        if project_id not in project_queues:
            project_queues[project_id] = Queue()
            project_locks[project_id] = threading.Lock()

        project_queues[project_id].put((note_id, file_path))
        queue_size = project_queues[project_id].qsize()

    logger.info(
        "✅ Tâche de traitement de document ajoutée à la file du projet %d "
        "pour la note %d (taille de la file: %d)",
        project_id,
        note_id,
        queue_size,
    )


def process_document_async(note_id: int, file_path: str):
    """Délègue à Celery ou file thread selon TASK_BACKEND_MODE."""
    from app.services.task_dispatch import dispatch_project_document

    dispatch_project_document(note_id, file_path)


def _ensure_document_workers():
    """S'assurer que les workers de traitement de documents sont démarrés."""
    global document_workers

    with _document_workers_lock:
        if not document_workers or not any(w.is_alive() for w in document_workers):
            document_workers = []
            # Par défaut: traitement séquentiel (1 document à la fois).
            # La valeur est bornée à >= 1 pour éviter toute config invalide.
            num_workers = max(1, settings.MAX_CONCURRENT_DOCUMENTS)
            for i in range(num_workers):
                worker = threading.Thread(
                    target=_process_document_worker, daemon=True
                )
                worker.start()
                document_workers.append(worker)
                logger.info("Worker de traitement de documents %d démarré", i + 1)
