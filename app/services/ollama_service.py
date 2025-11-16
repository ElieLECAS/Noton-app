import httpx
from typing import List, Dict, Optional
from app.config import settings


async def get_available_models() -> List[str]:
    """Récupérer la liste des modèles Ollama disponibles"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
            
            # Extraire les noms des modèles
            models = []
            if "models" in data:
                for model in data["models"]:
                    if "name" in model:
                        models.append(model["name"])
            
            return models
    except Exception as e:
        print(f"Erreur lors de la récupération des modèles Ollama: {e}")
        return []


async def chat(message: str, model: str, context: Optional[List[Dict]] = None) -> Dict:
    """Envoyer un message au chatbot Ollama"""
    try:
        payload = {
            "model": model,
            "messages": context or [{"role": "user", "content": message}],
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json=payload
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Erreur lors de l'appel à Ollama: {e}")
        raise


async def chat_stream(message: str, model: str, context: Optional[List[Dict]] = None):
    """Envoyer un message au chatbot Ollama avec streaming"""
    try:
        payload = {
            "model": model,
            "messages": context or [{"role": "user", "content": message}],
            "stream": True
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield line
    except Exception as e:
        print(f"Erreur lors du streaming Ollama: {e}")
        raise

