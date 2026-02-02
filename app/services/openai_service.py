import httpx
import json
from typing import List, Dict, Optional
from app.config import settings


async def get_available_models() -> List[str]:
    """Récupérer la liste des modèles OpenAI disponibles"""
    if not settings.OPENAI_API_KEY:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}"
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Filtrer les modèles GPT disponibles
            models = []
            if "data" in data:
                for model in data["data"]:
                    model_id = model.get("id", "")
                    if model_id.startswith("gpt-") and "instruct" not in model_id.lower():
                        models.append(model_id)
            
            # Ajouter les modèles depuis les settings s'ils sont configurés et ne sont pas déjà dans la liste
            if settings.OPENAI_MODEL:
                for model in settings.OPENAI_MODEL:
                    if model not in models:
                        models.append(model)
            
            # Trier et retourner les modèles
            return sorted(models)
    except Exception as e:
        print(f"Erreur lors de la récupération des modèles OpenAI: {e}")
        return []


async def chat(message: str, model: str, context: Optional[List[Dict]] = None) -> Dict:
    """Envoyer un message au chatbot OpenAI"""
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY n'est pas configurée")
    
    try:
        # Construire les messages
        messages = []
        if context:
            messages.extend(context)
        else:
            messages.append({"role": "user", "content": message})
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Erreur lors de l'appel à OpenAI: {e}")
        raise


async def chat_stream(message: str, model: str, context: Optional[List[Dict]] = None):
    """Envoyer un message au chatbot OpenAI avec streaming"""
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY n'est pas configurée")
    
    try:
        # Construire les messages
        messages = []
        if context:
            messages.extend(context)
        else:
            messages.append({"role": "user", "content": message})
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        # OpenAI renvoie des lignes au format "data: {...}" ou "data: [DONE]"
                        if line.startswith("data: "):
                            data_str = line[6:]  # Enlever "data: "
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                # Convertir au format Ollama pour compatibilité
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        # Formater comme Ollama: {"message": {"content": "..."}}
                                        ollama_format = {
                                            "message": {
                                                "content": delta["content"]
                                            }
                                        }
                                        yield json.dumps(ollama_format)
                            except json.JSONDecodeError:
                                pass
    except Exception as e:
        print(f"Erreur lors du streaming OpenAI: {e}")
        raise


async def generate_image(
    prompt: str,
    model: Optional[str] = None,
    size: str = "1024x1024",
    quality: str = "standard",
) -> List[Dict]:
    """
    Générer une ou plusieurs images via l'API OpenAI (DALL-E 3).
    Retourne une liste de dicts avec 'url' (ou 'b64_json' selon response_format).
    """
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY n'est pas configurée")

    model = model or settings.OPENAI_IMAGE_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "quality": quality,
        "response_format": "url",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])

