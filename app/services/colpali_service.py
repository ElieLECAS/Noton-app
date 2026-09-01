import logging
import threading
import time
from typing import List, Optional
from PIL import Image
import torch
from app.config import settings

logger = logging.getLogger(__name__)

_colpali_model = None
_colpali_processor = None
_colpali_lock = threading.Lock()


def _patch_transformers_chat_template_lookup():
    """Neutralise un bug de `transformers` >= 4.57 qui fait planter TOUT chargement de
    processeur dont le dépôt a un dossier ``additional_chat_templates/`` (ex :
    vidore/colqwen2-v1.0) :

    ``list_repo_templates`` renvoie déjà les noms de fichiers AVEC l'extension
    (``"vision.jinja"``), mais ``ProcessorMixin.get_processor_dict`` leur rajoute un
    ``.jinja`` une seconde fois (``f"{CHAT_TEMPLATE_DIR}/{template}.jinja"``). Le chemin
    résolu (``additional_chat_templates/vision.jinja.jinja``) n'existe jamais dans le
    dépôt : ``cached_file(..., _raise_exceptions_for_missing_entries=False)`` rend
    ``None``, puis ``open(None, ...)`` lève ``TypeError: expected str, bytes or
    os.PathLike object, not NoneType``.

    ColPali/ColQwen2 ne font jamais de chat template (juste des embeddings image et
    texte) : on court-circuite la recherche plutôt que d'attendre un correctif upstream
    ou de rétrograder `transformers`, utilisé ailleurs dans l'application.
    """
    import transformers.processing_utils as processing_utils

    if getattr(processing_utils.list_repo_templates, "_patched_no_templates", False):
        return

    def _no_templates(*_args, **_kwargs):
        return []

    _no_templates._patched_no_templates = True
    processing_utils.list_repo_templates = _no_templates


def get_colpali_model():
    """Lazily loads and returns the ColPali/ColQwen2 model and processor.

    Le rechargement se déclenche si L'UN OU L'AUTRE manque, et tout échec remet les
    DEUX globales à ``None`` avant de se propager. Sans ça, un modèle chargé avec succès
    suivi d'un processeur en échec (bug transitoire, panne réseau…) laissait le worker
    Celery — qui vit des heures et ne recharge jamais le module — bloqué à vie avec
    ``_colpali_model`` posé et ``_colpali_processor`` à ``None`` : la garde initiale
    (``if _colpali_model is None``) sautait tout le bloc de chargement pour toujours
    renvoyer ce couple incohérent, d'où ``'NoneType' object has no attribute
    'process_images'`` sur CHAQUE document suivant, jusqu'au redémarrage du worker.
    """
    global _colpali_model, _colpali_processor
    with _colpali_lock:
        if _colpali_model is None or _colpali_processor is None:
            try:
                _patch_transformers_chat_template_lookup()
                model_name = settings.COLPALI_MODEL_NAME
                logger.info(f"Loading ColPali model: {model_name}...")

                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

                # Load the correct class based on the model type
                if "colqwen" in model_name.lower():
                    from colpali_engine.models import ColQwen2, ColQwen2Processor
                    # Pass device_map explicitly to avoid the 'meta' device bug.
                    # On CPU, device_map='cpu' materializes weights directly to RAM.
                    # On CUDA, device_map='cuda' (or 'auto') materializes weights to GPU.
                    model = ColQwen2.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        device_map=device
                    )
                    processor = ColQwen2Processor.from_pretrained(model_name)
                elif "colsmol" in model_name.lower() or "idefics" in model_name.lower():
                    from colpali_engine.models import ColIdefics3, ColIdefics3Processor
                    model = ColIdefics3.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        device_map=device
                    )
                    processor = ColIdefics3Processor.from_pretrained(model_name)
                else:
                    from colpali_engine.models import ColPali, ColPaliProcessor
                    model = ColPali.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        device_map=device
                    )
                    processor = ColPaliProcessor.from_pretrained(model_name)

                _colpali_model, _colpali_processor = model, processor
                logger.info(f"ColPali model loaded on device: {_colpali_model.device}")
            except Exception:
                _colpali_model, _colpali_processor = None, None
                raise

    return _colpali_model, _colpali_processor

def embed_pdf_pages_colpali(pdf_path: str, document_id: Optional[int] = None) -> List[List[List[float]]]:
    """
    Renders PDF pages as images and extracts multi-vector embeddings using ColPali.
    Returns: List of page-level patch embeddings: [num_pages, num_patches, 128]
    """
    from pdf2image import convert_from_path
    
    logger.info(f"Rendering PDF pages for ColPali from: {pdf_path}")
    # Render PDF pages to PIL images (DPI 150 is typically used for ColPali)
    images = convert_from_path(pdf_path, dpi=150)
    
    model, processor = get_colpali_model()
    all_embeddings = []
    num_pages = len(images)
    
    from sqlmodel import Session
    from app.database import engine
    from app.models.document import Document
    
    batch_size = 3
    for i in range(0, num_pages, batch_size):
        batch_images = images[i : i + batch_size]
        current_page_last = i + len(batch_images)
        logger.info(f"ColPali embedding pages {i + 1}-{current_page_last}/{num_pages}...")
        
        # Update progress and page counters in DB
        if document_id is not None:
            try:
                with Session(engine) as sess:
                    doc = sess.get(Document, document_id)
                    if doc:
                        pct = int(current_page_last / num_pages * 100)
                        doc.processing_progress = pct
                        doc.phase_status_json = {
                            "current_page": current_page_last,
                            "total_pages": num_pages
                        }
                        sess.add(doc)
                        sess.commit()
            except Exception as db_err:
                logger.warning(f"Failed to update progress in DB for document {document_id}: {db_err}")
                
        # Process and prepare image tensor using colpali-engine processor interface
        inputs = processor.process_images(batch_images).to(model.device)
        
        with torch.no_grad():
            embeddings = model(**inputs)  # shape: (batch_size, num_patches, dim)
            
        # Move each page in the batch to CPU, convert to float32 and to lists
        for idx in range(len(batch_images)):
            page_vectors = embeddings[idx].cpu().float().numpy().tolist()
            all_embeddings.append(page_vectors)
        
    logger.info(f"Completed ColPali embedding for {num_pages} pages")
    return all_embeddings


def embed_query_colpali(query: str) -> List[List[float]]:
    """
    Embeds query text using ColPali.
    Returns: Query token embeddings: [num_query_tokens, 128]
    """
    _t0 = time.perf_counter()
    model, processor = get_colpali_model()
    _t_load = time.perf_counter()

    # Process query text using colpali-engine processor interface
    inputs = processor.process_queries([query]).to(model.device)

    with torch.no_grad():
        embeddings = model(**inputs)  # shape: (1, num_query_tokens, dim)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        mask = attention_mask[0] == 1
        query_vectors = embeddings[0][mask].cpu().float().numpy().tolist()
    else:
        query_vectors = embeddings[0].cpu().float().numpy().tolist()
    # [PERF] encodage requête = tax fixe par tour RAG (forward ColQwen2 sur CPU).
    logger.info(
        "[PERF][colpali] encode requête — %d tokens, %.2fs (dont chargement modèle %.2fs) device=%s",
        len(query_vectors),
        time.perf_counter() - _t0,
        _t_load - _t0,
        model.device,
    )
    return query_vectors
