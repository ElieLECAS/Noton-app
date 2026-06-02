import os
import logging
import threading
from typing import List, Optional
from PIL import Image
import torch
from app.config import settings

logger = logging.getLogger(__name__)

_colpali_model = None
_colpali_processor = None
_colpali_lock = threading.Lock()

def get_colpali_model():
    """Lazily loads and returns the ColPali/ColQwen2 model and processor."""
    global _colpali_model, _colpali_processor
    with _colpali_lock:
        if _colpali_model is None:
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
                _colpali_model = ColQwen2.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=device
                )
                _colpali_processor = ColQwen2Processor.from_pretrained(model_name)
            else:
                from colpali_engine.models import ColPali, ColPaliProcessor
                _colpali_model = ColPali.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=device
                )
                _colpali_processor = ColPaliProcessor.from_pretrained(model_name)
                
            logger.info(f"ColPali model loaded on device: {_colpali_model.device}")
        
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
    model, processor = get_colpali_model()
    
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
    return query_vectors


def sync_document_colpali_embeddings(document_id: int):
    """
    Checks if ColPali is enabled, reads the document from DB, 
    generates ColPali page embeddings, and inserts them into LanceDB.
    """
    if not settings.COLPALI_ENABLED:
        return
        
    from app.models.document import Document
    from app.models.document_chunk import DocumentChunk
    from app.services.lancedb_service import insert_colpali_patches_batch_lancedb
    from sqlmodel import Session, select
    from app.database import engine
    
    with Session(engine) as session:
        document = session.get(Document, document_id)
        if not document or not document.source_file_path:
            return
            
        pdf_path = document.source_file_path
        if not os.path.exists(pdf_path):
            logger.warning(f"Source file path not found for ColPali: {pdf_path}")
            return
            
        try:
            # 1. Generate ColPali page-level embeddings
            page_embeddings = embed_pdf_pages_colpali(pdf_path, document_id=document_id)
            
            # 2. Get document chunks
            statement = select(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.is_leaf == True
            )
            chunks = list(session.exec(statement).all())
            
            # Map chunk IDs to page indices and prepare batch list
            chunk_patches_list = []
            for chunk in chunks:
                meta = chunk.metadata_json or {}
                page_no = meta.get("page_no") or meta.get("page_start")
                if page_no is not None:
                    page_idx = int(page_no) - 1
                    if 0 <= page_idx < len(page_embeddings):
                        chunk_patches_list.append((chunk.id, page_embeddings[page_idx]))
            
            if chunk_patches_list:
                insert_colpali_patches_batch_lancedb(document_id, chunk_patches_list)
                
            logger.info(f"ColPali embeddings generated and synced for document {document_id}")
        except Exception as e:
            logger.error(f"Error generating ColPali embeddings for document {document_id}: {e}", exc_info=True)
