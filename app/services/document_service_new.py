from typing import Optional, List
import logging
import os
from pathlib import Path
import threading
import time
from queue import Queue, Empty
from sqlmodel import Session, select
from app.config import settings
from app.database import engine
from app.library_document_logging import get_library_document_logger
from app.models.document import Document, DocumentCreate, DocumentUpdate
from app.models.document_space import DocumentSpace
from app.models.document_chunk import DocumentChunk
from datetime import datetime

logger = logging.getLogger(__name__)

# Une seule file globale : ordre FIFO strict, un document terminé entièrement avant le suivant.
document_task_queue: Queue = Queue()
_document_queue_lock = threading.Lock()
document_workers = []
_document_workers_lock = threading.Lock()
_cancelled_document_ids: set[int] = set()
_cancelled_documents_lock = threading.Lock()

_docling_converter = None
_docling_converter_generic = None
_docling_converter_lock = threading.Lock()


def _mark_document_processing_cancelled(document_id: int) -> None:
    """Marque un document comme annulé pour stopper son traitement asynchrone."""
    with _cancelled_documents_lock:
        _cancelled_document_ids.add(document_id)


def _is_document_processing_cancelled(document_id: int) -> bool:
    with _cancelled_documents_lock:
        return document_id in _cancelled_document_ids


def _clear_document_processing_cancelled(document_id: int) -> None:
    with _cancelled_documents_lock:
        _cancelled_document_ids.discard(document_id)


def _should_abort_processing(document_id: int) -> bool:
    """
    Indique si le traitement doit être interrompu:
    - suppression demandée explicitement
    - document supprimé en base
    """
    if _is_document_processing_cancelled(document_id):
        return True

    try:
        with Session(engine) as session:
            return session.get(Document, document_id) is None
    except Exception:
        # En cas d'erreur transitoire DB, on n'interrompt pas par défaut.
        return False

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


def get_docling_converter(file_path: Optional[str] = None):
    """Récupère le convertisseur Docling adapté au type de fichier."""
    global _docling_converter, _docling_converter_generic

    with _docling_converter_lock:
        # Utiliser un convertisseur générique pour les formats non-PDF
        # afin d'éviter de restreindre Docling aux seules options PDF.
        suffix = Path(file_path).suffix.lower() if file_path else ""
        use_generic_converter = suffix and suffix != ".pdf"

        if use_generic_converter:
            if _docling_converter_generic is None:
                init_start = time.time()
                logger.info("Initialisation du DocumentConverter Docling (générique)...")
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

                        # Échelle d'image ajustable pour OCR (3.0 par défaut pour meilleure qualité)
                        image_scale = getattr(settings, "OCR_IMAGE_SCALE", 3.0)
                        
                        pipeline_options = PdfPipelineOptions(
                            do_ocr=ocr_enabled,
                            generate_picture_images=True,
                            images_scale=image_scale,
                        )

                        if ocr_enabled:
                            ocr_lang = getattr(settings, "DOCLING_OCR_LANG", None)
                            if ocr_lang:
                                lang_list = [x.strip().lower() for x in ocr_lang.replace("+", ",").split(",") if x.strip()]
                                if lang_list:
                                    try:
                                        from docling.datamodel.pipeline_options import EasyOcrOptions
                                        pipeline_options.ocr_options = EasyOcrOptions(
                                            lang=lang_list,
                                            use_gpu=settings.DOCLING_USE_GPU is True,
                                        )
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


def ensure_pdf_for_docling(file_path: str) -> str:
    """
    Docling est configuré/optimisé pour traiter le PDF.

    Pour les formats bureautiques (ODT/DOCX/DOC/...), on convertit d'abord en PDF
    via LibreOffice (headless), puis on traite le PDF résultant.
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
        # Si on ne peut pas supprimer, on tentera quand même la conversion
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
        # LibreOffice peut parfois produire un nom légèrement différent.
        candidates = sorted(
            pdf_path.parent.glob(f"{Path(file_path).stem}*.pdf"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"Aucun PDF généré pour {file_path}")
        pdf_path = candidates[0]

    return str(pdf_path)


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

        converter = get_docling_converter(file_path)

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

        logger.error("Erreur lors du traitement du document %s: %s", file_path, e, exc_info=True)
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


def reindex_library_document(document_id: int, user_id: int) -> dict:
    """
    Re-extrait le texte (Docling), rechunke (hiérarchie Docling ou fallback fixe),
    ré-embed les feuilles et relance le KAG pour les espaces liés — sans réupload.

    Utilise ``document.source_file_path`` (fichier déjà stocké sous media/documents).
    """
    ld = get_library_document_logger()
    ld.info(
        "[Réindex] Démarrage document_id=%s user_id=%s — pipeline : PDF/Docling → "
        "markdown + llama_docs → chunks → embeddings → KAG (espaces liés).",
        document_id,
        user_id,
    )
    from app.services.chunk_service import (
        complete_document_embeddings_and_kag_sync,
        create_chunks_for_document,
        create_chunks_for_document_from_docling,
    )

    with Session(engine) as session:
        document = session.get(Document, document_id)
        if not document or document.user_id != user_id:
            raise ValueError("Document introuvable ou accès refusé")
        if not document.source_file_path:
            raise ValueError("Aucun fichier source enregistré pour ce document")
        src = Path(document.source_file_path)
        if not src.is_file():
            raise ValueError("Fichier source introuvable sur le disque")
        ld.info(
            "[Réindex] document_id=%s — fichier source : %s",
            document_id,
            src,
        )

        document.processing_status = "processing"
        document.processing_progress = 15
        document.updated_at = datetime.utcnow()
        session.add(document)
        session.commit()

    try:
        pdf_input = ensure_pdf_for_docling(str(src))
        ld.info(
            "[Réindex] document_id=%s — conversion/entrée Docling : %s",
            document_id,
            pdf_input,
        )
        t0 = time.perf_counter()
        markdown_content, llama_docs, docling_doc = process_document_file(pdf_input)
        logger.info(
            "reindex: document_id=%s extraction %.2fs llama_docs=%s",
            document_id,
            time.perf_counter() - t0,
            bool(llama_docs),
        )
        ld.info(
            "[Réindex] document_id=%s — extraction Docling terminée en %.2fs : "
            "markdown_len=%s llama_docs_présent=%s docling_doc_présent=%s. "
            "Sans llama_docs, le chunking hiérarchique Docling est impossible.",
            document_id,
            time.perf_counter() - t0,
            len(markdown_content) if markdown_content else 0,
            bool(llama_docs),
            docling_doc is not None,
        )
    except Exception as e:
        logger.error(
            "reindex: échec process_document_file document_id=%s: %s",
            document_id,
            e,
            exc_info=True,
        )
        with Session(engine) as session:
            d = session.get(Document, document_id)
            if d:
                d.processing_status = "failed"
                d.processing_progress = max(d.processing_progress or 0, 15)
                d.updated_at = datetime.utcnow()
                session.add(d)
                session.commit()
        raise ValueError(f"Échec lecture ou conversion du document: {e}") from e

    if not markdown_content:
        with Session(engine) as session:
            d = session.get(Document, document_id)
            if d:
                d.processing_status = "failed"
                d.processing_progress = max(d.processing_progress or 0, 15)
                d.updated_at = datetime.utcnow()
                session.add(d)
                session.commit()
        raise ValueError("Extraction vide (markdown)")

    chunk_count = 0
    with Session(engine) as session:
        document = session.get(Document, document_id)
        document.content = markdown_content
        document.processing_progress = 55
        document.updated_at = datetime.utcnow()
        session.add(document)
        session.commit()

        images_info: list = []
        if docling_doc:
            try:
                images_info = extract_and_save_images(docling_doc, document_id) or []
            except Exception as img_e:
                logger.warning("reindex: extraction images partielle: %s", img_e)
        ld.info(
            "[Réindex] document_id=%s — images pour Pixtral : %d entrée(s), docling_doc=%s.",
            document_id,
            len(images_info) if images_info else 0,
            docling_doc is not None,
        )

        document = session.get(Document, document_id)
        if llama_docs:
            chunks = create_chunks_for_document_from_docling(
                session,
                document,
                llama_docs,
                generate_embeddings=False,
                images_info=images_info or None,
            )
            logger.info(
                "reindex: document_id=%s chunking=docling_hierarchical chunks=%s",
                document_id,
                len(chunks) if chunks else 0,
            )
        else:
            chunks = create_chunks_for_document(
                session, document, generate_embeddings=False
            )
            logger.info(
                "reindex: document_id=%s chunking=fallback_markdown_ou_fenêtre_fixe chunks=%s",
                document_id,
                len(chunks) if chunks else 0,
            )
        chunk_count = len(chunks) if chunks else 0
        document.processing_progress = 85
        document.updated_at = datetime.utcnow()
        session.add(document)
        session.commit()

    ld.info(
        "[Réindex] document_id=%s — lancement embeddings + KAG (synchrone).",
        document_id,
    )
    complete_document_embeddings_and_kag_sync(document_id)

    logger.info(
        "reindex_library_document terminé document_id=%s chunks=%s",
        document_id,
        chunk_count,
    )
    ld.info(
        "[Réindex] document_id=%s — FIN OK chunks=%s (voir inventaire chunking dans les lignes [Chunking]).",
        document_id,
        chunk_count,
    )
    return {
        "document_id": document_id,
        "chunks": chunk_count,
        "status": "completed",
    }


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
    """Récupère un document par son ID (bibliothèque globale partagée)."""
    statement = select(Document).where(
        Document.id == document_id
    )
    return session.exec(statement).first()


def get_documents_by_folder(session: Session, folder_id: Optional[int], library_id: int, user_id: int) -> List[Document]:
    """Récupère tous les documents d'un dossier (ou racine si folder_id est None)."""
    statement = select(Document).where(
        Document.library_id == library_id,
        Document.folder_id == folder_id
    ).order_by(Document.created_at.desc())
    return list(session.exec(statement).all())


def get_documents_by_library(session: Session, library_id: int, user_id: int) -> List[Document]:
    """Récupère tous les documents d'une bibliothèque."""
    statement = select(Document).where(
        Document.library_id == library_id
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

        # Bibliothèque générale partagée:
        # autoriser le déplacement entre bibliothèques historiques et
        # aligner le document sur la bibliothèque du dossier cible.
        if folder.library_id != document.library_id:
            logger.info(
                "Déplacement inter-bibliothèque du document %d: %s -> %s",
                document_id,
                document.library_id,
                folder.library_id,
            )
            document.library_id = folder.library_id
    
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
    from app.services.kag_graph_service import delete_entities_for_document

    # Empêche un traitement asynchrone tardif de recréer chunks/KAG.
    _mark_document_processing_cancelled(document_id)

    document = get_document_by_id(session, document_id, user_id)
    if not document:
        return False
    
    doc_spaces = get_document_spaces(session, document_id, user_id)
    for doc_space in doc_spaces:
        try:
            delete_entities_for_document(session, document_id, doc_space.space_id)
        except Exception as e:
            logger.warning(
                "Nettoyage KAG incomplet pour document=%s espace=%s: %s",
                document_id,
                doc_space.space_id,
                e,
            )

    delete_chunks_for_document(session, document_id, commit=False)

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
        link = link_document_to_space(session, document_id, space_id, user_id)
        if link is None:
            logger.error(
                "Impossible de lier le document %s à l'espace %s",
                document_id,
                space_id,
            )
            return False

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


def apply_document_spaces_update(
    document_id: int,
    add_space_ids: List[int],
    remove_space_ids: List[int],
    user_id: int,
) -> bool:
    """
    Applique les changements d'association document/espaces dans une session dédiée.
    Utilisé par les workers (Celery/thread) pour ne pas bloquer le process web.
    """
    with Session(engine) as session:
        document = get_document_by_id(session, document_id, user_id)
        if not document:
            logger.error("Document %s non trouvé pour mise à jour des espaces", document_id)
            return False

        if add_space_ids:
            success = add_document_to_spaces(session, document_id, add_space_ids, user_id)
            if not success:
                return False

        if remove_space_ids:
            success = remove_document_from_spaces(
                session, document_id, remove_space_ids, user_id
            )
            if not success:
                return False

        return True


def _process_document_worker():
    """Worker thread : un document à la fois, de bout en bout (pas de chevauchement)."""
    try:
        import os
        if hasattr(os, "nice"):
            os.nice(5)
    except Exception:
        pass

    logger.info("Worker de traitement de documents démarré et en attente de tâches...")
    while True:
        try:
            task = document_task_queue.get(timeout=2)
        except Empty:
            continue

        if task is None:
            document_task_queue.task_done()
            logger.info("Signal d'arrêt reçu, arrêt du worker de documents")
            return

        document_id, file_path = task
        try:
            logger.info("Worker traite le document %d (file globale)", document_id)
            _process_document_for_id(document_id, file_path)
            logger.info(
                "Worker a terminé le document %d (toutes étapes incluses)", document_id
            )
        except Exception as e:
            logger.error(
                "Erreur dans le worker de traitement de documents: %s", e, exc_info=True
            )
        finally:
            document_task_queue.task_done()
            time.sleep(0.5)


def _process_document_for_id(document_id: int, file_path: str):
    """Traite un document pour un ID donné."""
    from app.services.chunk_service import (
        complete_document_embeddings_and_kag_sync,
        create_chunks_for_document,
        create_chunks_for_document_from_docling,
    )

    ld = get_library_document_logger()
    try:
        logger.info("Démarrage du traitement du document %d", document_id)
        ld.info(
            "[Upload/Pipeline] document_id=%s — DÉBUT traitement bibliothèque fichier=%s "
            "(worker thread ou Celery). Étapes : PDF → Docling → stockage → images → chunks → embeddings → KAG.",
            document_id,
            file_path,
        )

        run_embeddings_sync: bool = False

        if _should_abort_processing(document_id):
            logger.info(
                "Traitement annulé avant démarrage effectif pour document %d",
                document_id,
            )
            ld.info(
                "[Upload/Pipeline] document_id=%s — annulé avant démarrage (cancel flag).",
                document_id,
            )
            return

        with Session(engine) as session:
            document = session.get(Document, document_id)
            if not document:
                logger.error("Document %d non trouvé pour traitement", document_id)
                ld.error(
                    "[Upload/Pipeline] document_id=%s — document introuvable en base, arrêt.",
                    document_id,
                )
                return

            document.processing_status = "processing"
            document.processing_progress = 10
            document.updated_at = datetime.utcnow()
            session.add(document)
            session.commit()

            original_ext = Path(file_path).suffix.lower()
            pdf_input_path = file_path
            converted_to_pdf = False
            try:
                pdf_input_path = ensure_pdf_for_docling(file_path)
                converted_to_pdf = pdf_input_path != file_path
                ld.info(
                    "[Upload/Pipeline] document_id=%s — préparation Docling : entrée=%s converti_pdf=%s",
                    document_id,
                    pdf_input_path,
                    converted_to_pdf,
                )
            except Exception as e:
                logger.error(
                    "Conversion vers PDF impossible pour le document %d (%s): %s",
                    document_id,
                    file_path,
                    e,
                    exc_info=True,
                )
                ld.error(
                    "[Upload/Pipeline] document_id=%s — échec ensure_pdf_for_docling : %s",
                    document_id,
                    e,
                    exc_info=True,
                )
                raise

            t_extract = time.perf_counter()
            markdown_content, llama_docs, docling_doc = process_document_file(pdf_input_path)
            extract_s = time.perf_counter() - t_extract
            logger.info(
                "document_id=%s extraction Docling %.2fs markdown_len=%s llama_docs=%s",
                document_id,
                extract_s,
                len(markdown_content) if markdown_content else 0,
                bool(llama_docs),
            )
            ld.info(
                "[Upload/Pipeline] document_id=%s — Docling terminé en %.2fs : markdown_len=%s "
                "llama_docs=%s docling_doc=%s. "
                "Si llama_docs=False après succès markdown, vérifier process_document_file (ex. EPUB fallback).",
                document_id,
                extract_s,
                len(markdown_content) if markdown_content else 0,
                bool(llama_docs),
                docling_doc is not None,
            )

            if not markdown_content:
                document.processing_status = "failed"
                document.processing_progress = max(document.processing_progress or 0, 10)
                document.content = "❌ Erreur lors du traitement du document. Le fichier peut être corrompu ou dans un format non supporté."
                document.updated_at = datetime.utcnow()
                session.add(document)
                session.commit()
                logger.error("Échec du traitement du document %d", document_id)
                ld.error(
                    "[Upload/Pipeline] document_id=%s — markdown vide après Docling, statut failed.",
                    document_id,
                )
                return

            if _should_abort_processing(document_id):
                logger.info(
                    "Traitement annulé après extraction pour document %d",
                    document_id,
                )
                ld.info(
                    "[Upload/Pipeline] document_id=%s — annulé après extraction.",
                    document_id,
                )
                return

            output_ext = Path(pdf_input_path).suffix.lower() or ".bin"
            permanent_output_path = Path(f"media/documents/{document_id}{output_ext}")
            permanent_output_path.parent.mkdir(parents=True, exist_ok=True)

            import shutil

            # Retenter proprement sur relances/erreurs précédentes
            if permanent_output_path.exists():
                permanent_output_path.unlink()

            if converted_to_pdf:
                original_permanent_path = Path(
                    f"media/documents/{document_id}{original_ext}"
                )
                original_permanent_path.parent.mkdir(parents=True, exist_ok=True)
                if original_permanent_path.exists():
                    original_permanent_path.unlink()
                shutil.move(file_path, str(original_permanent_path))

            shutil.move(pdf_input_path, str(permanent_output_path))
            document.source_file_path = str(permanent_output_path)
            logger.info(
                "Fichier traité déplacé vers chemin permanent: %s",
                permanent_output_path,
            )
            ld.info(
                "[Upload/Pipeline] document_id=%s — fichier enregistré sous %s",
                document_id,
                permanent_output_path,
            )

            document.content = markdown_content
            document.processing_status = "processing"
            document.processing_progress = 55
            document.updated_at = datetime.utcnow()
            session.add(document)
            session.commit()

            logger.info("Document traité avec succès pour document_id %d (%d caractères extraits)", document_id, len(markdown_content))

            images_info: list = []
            if docling_doc:
                images_info = extract_and_save_images(docling_doc, document_id) or []
                if images_info:
                    logger.info("%d image(s) extraite(s) pour le document %d", len(images_info), document_id)
                ld.info(
                    "[Upload/Pipeline] document_id=%s — extraction images : %d fichier(s) pour appariement Pixtral.",
                    document_id,
                    len(images_info) if images_info else 0,
                )

            try:
                if _should_abort_processing(document_id):
                    logger.info(
                        "Traitement annulé avant création des chunks pour document %d",
                        document_id,
                    )
                    ld.info(
                        "[Upload/Pipeline] document_id=%s — annulé avant chunking.",
                        document_id,
                    )
                    return

                t_chunk = time.perf_counter()
                if llama_docs:
                    chunks = create_chunks_for_document_from_docling(
                        session,
                        document,
                        llama_docs,
                        generate_embeddings=False,
                        images_info=images_info or None,
                    )
                    chunk_s = time.perf_counter() - t_chunk
                    logger.info(
                        "document_id=%s chunks=%s stratégie=docling_hierarchical durée_chunking=%.2fs",
                        document_id,
                        len(chunks),
                        chunk_s,
                    )
                    ld.info(
                        "[Upload/Pipeline] document_id=%s — branche chunking : llama_docs présent "
                        "(voir [Chunking]/[DoclingNodeParser] pour stratégie réelle et inventaire). "
                        "durée_chunking=%.2fs chunks_retournés=%d",
                        document_id,
                        chunk_s,
                        len(chunks),
                    )
                else:
                    chunks = create_chunks_for_document(session, document, generate_embeddings=False)
                    chunk_s = time.perf_counter() - t_chunk
                    logger.info(
                        "document_id=%s chunks=%s stratégie=fallback_markdown_ou_fenêtre_fixe durée_chunking=%.2fs",
                        document_id,
                        len(chunks),
                        chunk_s,
                    )
                    ld.warning(
                        "[Upload/Pipeline] document_id=%s — llama_docs absent : chunking FALLBACK uniquement "
                        "(pas de DoclingNodeParser). durée=%.2fs chunks=%d",
                        document_id,
                        chunk_s,
                        len(chunks),
                    )

                document.processing_progress = 75
                document.updated_at = datetime.utcnow()
                session.add(document)
                session.commit()

                if chunks:
                    document.processing_progress = 85
                    document.updated_at = datetime.utcnow()
                    session.add(document)
                    session.commit()
                    run_embeddings_sync = True
                else:
                    document.processing_status = "completed"
                    document.processing_progress = 100
                    document.updated_at = datetime.utcnow()
                    session.add(document)
                    session.commit()
                    ld.warning(
                        "[Upload/Pipeline] document_id=%s — aucun chunk produit : pas d'embeddings/KAG, "
                        "document marqué completed.",
                        document_id,
                    )

            except Exception as e:
                logger.error("Erreur lors de la création des chunks pour le document %d: %s", document_id, e, exc_info=True)
                ld.error(
                    "[Upload/Pipeline] document_id=%s — erreur lors de la création des chunks : %s",
                    document_id,
                    e,
                    exc_info=True,
                )
                document.processing_status = "failed"
                document.processing_progress = max(document.processing_progress or 0, 55)
                document.updated_at = datetime.utcnow()
                session.add(document)
                session.commit()
                run_embeddings_sync = False

        if run_embeddings_sync:
            if _should_abort_processing(document_id):
                logger.info(
                    "Traitement annulé avant embeddings/KAG pour document %d",
                    document_id,
                )
                ld.info(
                    "[Upload/Pipeline] document_id=%s — annulé avant embeddings/KAG.",
                    document_id,
                )
                return
            ld.info(
                "[Upload/Pipeline] document_id=%s — enchaînement embeddings + KAG.",
                document_id,
            )
            complete_document_embeddings_and_kag_sync(document_id)
            ld.info(
                "[Upload/Pipeline] document_id=%s — FIN pipeline bibliothèque (succès attendu si pas d'erreur amont).",
                document_id,
            )

    except Exception as e:
        logger.error("Erreur lors du traitement du document %d: %s", document_id, e, exc_info=True)
        ld.error(
            "[Upload/Pipeline] document_id=%s — ERREUR non gérée : %s",
            document_id,
            e,
            exc_info=True,
        )
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
    finally:
        # Nettoyage défensif: si le document n'existe plus, on retire l'annulation.
        try:
            with Session(engine) as session:
                if session.get(Document, document_id) is None:
                    _clear_document_processing_cancelled(document_id)
        except Exception:
            pass


def enqueue_library_document_thread(document_id: int, file_path: str):
    """Ajoute un document à la file thread globale (sans Celery)."""
    try:
        with Session(engine) as session:
            document = session.get(Document, document_id)
            if not document:
                logger.error("Document %d non trouvé pour ajout à la queue", document_id)
                return
    except Exception as e:
        logger.error(
            "Erreur lors de la vérification du document %d: %s", document_id, e, exc_info=True
        )
        return

    _ensure_document_workers()

    with _document_queue_lock:
        document_task_queue.put((document_id, file_path))
        queue_size = document_task_queue.qsize()

    logger.info(
        "✅ Document %d ajouté à la file globale (taille: %d)", document_id, queue_size
    )
    get_library_document_logger().info(
        "[Queue] document_id=%s — entrée file thread worker (position file ~%d) fichier=%s",
        document_id,
        queue_size,
        file_path,
    )


def process_document_async(document_id: int, file_path: str):
    """Délègue à Celery ou à la file thread selon TASK_BACKEND_MODE."""
    from app.services.task_dispatch import dispatch_library_document

    dispatch_library_document(document_id, file_path)


def _ensure_document_workers():
    """S'assurer que les workers de traitement de documents sont démarrés."""
    global document_workers

    with _document_workers_lock:
        if not document_workers or not any(w.is_alive() for w in document_workers):
            document_workers = []
            # Toujours 1 worker : un document va au bout (Docling → embeddings → KAG) avant le suivant.
            num_workers = 1
            for i in range(num_workers):
                worker = threading.Thread(target=_process_document_worker, daemon=True)
                worker.start()
                document_workers.append(worker)
                logger.info("Worker de traitement de documents %d démarré", i + 1)
