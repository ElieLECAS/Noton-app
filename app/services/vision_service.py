"""
Service de description d'images avec GPT-4o Vision.

Ce module permet de générer des descriptions textuelles des images/schémas
extraits des documents, afin de les indexer dans le système RAG.
"""

import base64
import logging
from pathlib import Path
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


async def describe_image(
    image_path: str,
    context: str = "",
    caption: str = "",
) -> Optional[str]:
    """
    Génère une description textuelle d'une image avec GPT-4o Vision.
    
    Args:
        image_path: Chemin vers le fichier image
        context: Contexte du document (ex: section/heading parent)
        caption: Légende existante de l'image si disponible
        
    Returns:
        Description textuelle de l'image ou None en cas d'erreur
    """
    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY non configurée, description d'image ignorée")
        return None
    
    path = Path(image_path)
    if not path.exists():
        logger.error("Image non trouvée: %s", image_path)
        return None
    
    try:
        with open(path, "rb") as f:
            image_data = f.read()
        
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        
        # Déterminer le type MIME
        suffix = path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/png")
        
        # Construire le prompt
        prompt_parts = [
            "Décris ce schéma ou cette image technique de manière détaillée et structurée.",
            "Inclus :",
            "- Les éléments visuels principaux (formes, connexions, flux)",
            "- Le texte visible dans l'image",
            "- Les relations entre les différents éléments",
            "- L'objectif ou le concept illustré",
        ]
        
        if context:
            prompt_parts.append(f"\nContexte du document : {context}")
        
        if caption:
            prompt_parts.append(f"\nLégende existante : {caption}")
        
        prompt_parts.append("\nRéponds en français de manière concise mais complète.")
        
        prompt = "\n".join(prompt_parts)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]
        
        vision_model = getattr(settings, "VISION_MODEL", "gpt-4o")
        
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": vision_model,
                    "messages": messages,
                    "max_tokens": 1500,
                    "temperature": 0.3,
                },
            )
            
            if response.status_code != 200:
                logger.error(
                    "Erreur API OpenAI Vision (%d): %s",
                    response.status_code,
                    response.text,
                )
                return None
            
            data = response.json()
            description = data["choices"][0]["message"]["content"]
            
            logger.debug(
                "Image décrite avec succès: %s (%d caractères)",
                image_path,
                len(description),
            )
            
            return description
            
    except httpx.TimeoutException:
        logger.error("Timeout lors de la description de l'image: %s", image_path)
        return None
    except Exception as e:
        logger.error(
            "Erreur lors de la description de l'image %s: %s",
            image_path,
            e,
            exc_info=True,
        )
        return None


async def describe_images_batch(
    images_info: list,
    context: str = "",
) -> list:
    """
    Décrit plusieurs images en séquence.
    
    Args:
        images_info: Liste de dictionnaires avec 'path' et optionnellement 'caption'
        context: Contexte global du document
        
    Returns:
        Liste des images_info avec le champ 'description' ajouté
    """
    results = []
    
    for img in images_info:
        image_path = img.get("path")
        caption = img.get("caption", "")
        
        description = await describe_image(
            image_path=image_path,
            context=context,
            caption=caption,
        )
        
        img_with_desc = dict(img)
        img_with_desc["description"] = description
        results.append(img_with_desc)
        
        if description:
            logger.info(
                "✅ Image décrite: %s",
                Path(image_path).name if image_path else "unknown",
            )
    
    return results
