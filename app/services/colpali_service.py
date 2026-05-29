import os
import logging
from typing import List, Optional
from PIL import Image
import torch
from app.config import settings

logger = logging.getLogger(__name__)

_colpali_model = None
_colpali_processor = None

def get_colpali_model():
    """Lazily loads and returns the ColPali/ColQwen2 model and processor."""
    global _colpali_model, _colpali_processor
    if _colpali_model is None:
        model_name = settings.COLPALI_MODEL_NAME
        logger.info(f"Loading ColPali model: {model_name}...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Load the correct class based on the model type
        if "colqwen" in model_name.lower():
            from colpali_engine.models import ColQwen2, ColQwen2Processor
            if device == "cuda":
                _colpali_model = ColQwen2.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map="auto"
                )
            else:
                _colpali_model = ColQwen2.from_pretrained(
                    model_name,
                    torch_dtype=dtype
                ).to(device)
            _colpali_processor = ColQwen2Processor.from_pretrained(model_name)
        else:
            from colpali_engine.models import ColPali, ColPaliProcessor
            if device == "cuda":
                _colpali_model = ColPali.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map="auto"
                )
            else:
                _colpali_model = ColPali.from_pretrained(
                    model_name,
                    torch_dtype=dtype
                ).to(device)
            _colpali_processor = ColPaliProcessor.from_pretrained(model_name)
            
        logger.info(f"ColPali model loaded on device: {_colpali_model.device}")
        
    return _colpali_model, _colpali_processor

def embed_pdf_pages_colpali(pdf_path: str) -> List[List[List[float]]]:
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
    
    for idx, img in enumerate(images):
        logger.info(f"ColPali embedding page {idx + 1}/{len(images)}...")
        # Process and prepare image tensor using colpali-engine processor interface
        inputs = processor.process_images([img]).to(model.device)
        
        with torch.no_grad():
            embeddings = model(**inputs)  # shape: (1, num_patches, dim)
            
        # Move to CPU, convert to float32 and to lists
        page_vectors = embeddings[0].cpu().float().numpy().tolist()
        all_embeddings.append(page_vectors)
        
    logger.info(f"Completed ColPali embedding for {len(images)} pages")
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
    from app.services.lancedb_service import insert_colpali_patches_lancedb
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
            page_embeddings = embed_pdf_pages_colpali(pdf_path)
            
            # 2. Get document chunks
            statement = select(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.is_leaf == True
            )
            chunks = list(session.exec(statement).all())
            
            # Map chunk IDs to page indices
            for chunk in chunks:
                meta = chunk.metadata_json or {}
                page_no = meta.get("page_no") or meta.get("page_start")
                if page_no is not None:
                    page_idx = int(page_no) - 1
                    if 0 <= page_idx < len(page_embeddings):
                        insert_colpali_patches_lancedb(document_id, chunk.id, page_embeddings[page_idx])
            logger.info(f"ColPali embeddings generated and synced for document {document_id}")
        except Exception as e:
            logger.error(f"Error generating ColPali embeddings for document {document_id}: {e}", exc_info=True)
