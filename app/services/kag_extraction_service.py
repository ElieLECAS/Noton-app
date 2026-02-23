"""
Service d'extraction d'entités KAG (Knowledge Augmented Generation).

Extrait les entités techniques des chunks via LLM configurable (OpenAI ou Ollama)
pour enrichir le graphe de connaissances et améliorer le RAG.
"""

import json
import re
import unicodedata
import logging
import asyncio
from typing import List, Dict, Optional
from app.config import settings

logger = logging.getLogger(__name__)

ENTITY_TYPES = ["equipement", "procedure", "parametre", "composant", "reference", "lieu"]

PARENT_SUMMARY_PROMPT_TEMPLATE = """Analyse ce passage de document technique et retourne UNIQUEMENT un JSON valide sans markdown.

Format attendu:
{{"summary": "résumé en 1-2 phrases", "generated_questions": ["question 1", "question 2", "question 3"]}}

Règles:
- Le résumé doit capturer l'intention métier de la section (ce que cette section apporte, pas comment elle est structurée)
- Les 3 questions doivent simuler des questions réelles d'un technicien ou d'un technico-commercial face au document
- Questions courtes et précises (interrogatives, sans réponse)
- Retourne UNIQUEMENT le JSON, sans texte avant ni après

Passage:
{content}

JSON:"""

EXTRACTION_PROMPT_TEMPLATE = """Extrais les entités techniques de ce texte.
Types possibles: {entity_types}

Règles:
- Retourne UNIQUEMENT un JSON valide, sans markdown ni commentaires
- Maximum 10 entités par chunk
- Importance entre 0.0 et 1.0 (1.0 = très important)
- Noms courts et précis (pas de phrases)

Format attendu:
[{{"name": "nom_entité", "type": "type", "importance": 0.8}}]

Texte:
{chunk_content}

JSON:"""


def normalize_entity_name(name: str) -> str:
    """
    Normalise un nom d'entité pour la déduplication.
    
    - Lowercase
    - Suppression des accents
    - Suppression des caractères spéciaux
    - Trim des espaces
    """
    if not name:
        return ""
    normalized = name.lower().strip()
    normalized = unicodedata.normalize("NFD", normalized)
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    normalized = re.sub(r"[^a-z0-9\s-]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _parse_llm_response(response_text: str) -> List[Dict]:
    """
    Parse la réponse LLM en liste d'entités.
    Gère les cas où le LLM ajoute du markdown ou du texte autour du JSON.
    """
    if not response_text:
        return []
    
    text = response_text.strip()
    
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_json = False
        for line in lines:
            if line.startswith("```") and not in_json:
                in_json = True
                continue
            elif line.startswith("```") and in_json:
                break
            elif in_json:
                json_lines.append(line)
        text = "\n".join(json_lines)
    
    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if json_match:
        text = json_match.group()
    
    try:
        entities = json.loads(text)
        if not isinstance(entities, list):
            logger.warning("Réponse LLM n'est pas une liste: %s", type(entities))
            return []
        
        valid_entities = []
        for e in entities:
            if not isinstance(e, dict):
                continue
            name = e.get("name", "").strip()
            entity_type = e.get("type", "").strip().lower()
            importance = e.get("importance", 1.0)
            
            if not name or len(name) < 2:
                continue
            if entity_type not in ENTITY_TYPES:
                entity_type = "composant"
            if not isinstance(importance, (int, float)):
                importance = 1.0
            importance = max(0.0, min(1.0, float(importance)))
            if entity_type == "reference":
                importance = 1.0
            
            valid_entities.append({
                "name": name,
                "type": entity_type,
                "importance": importance,
            })
        
        return valid_entities[:10]
    
    except json.JSONDecodeError as e:
        logger.warning("Erreur parsing JSON LLM: %s - Réponse: %s", e, text[:200])
        return []


def _parse_summary_questions_response(response_text: str) -> Optional[Dict]:
    """
    Parse la réponse LLM pour le résumé + questions d'un chunk parent.
    Retourne un dict {"summary": str, "generated_questions": [str, str, str]} ou None.
    """
    if not response_text:
        return None

    text = response_text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_json = False
        for line in lines:
            if line.startswith("```") and not in_json:
                in_json = True
                continue
            elif line.startswith("```") and in_json:
                break
            elif in_json:
                json_lines.append(line)
        text = "\n".join(json_lines)

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        text = json_match.group()

    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return None

        summary = data.get("summary", "").strip()
        questions = data.get("generated_questions", [])

        if not summary:
            return None
        if not isinstance(questions, list):
            questions = []
        questions = [str(q).strip() for q in questions if q and str(q).strip()][:3]

        return {"summary": summary, "generated_questions": questions}
    except json.JSONDecodeError as e:
        logger.warning("Erreur parsing JSON résumé parent: %s - Réponse: %s", e, text[:200])
        return None


async def generate_parent_summary_questions(content: str) -> Optional[Dict]:
    """
    Génère un résumé + 3 questions pour un chunk parent (section) via LLM.

    Args:
        content: Contenu textuel du chunk parent

    Returns:
        Dict {"summary": str, "generated_questions": [str, str, str]} ou None en cas d'échec
    """
    if not content or len(content.strip()) < 30:
        return None

    content_truncated = content[:3000]
    prompt = PARENT_SUMMARY_PROMPT_TEMPLATE.format(content=content_truncated)

    provider = settings.KAG_EXTRACTION_PROVIDER.lower()
    model = settings.KAG_EXTRACTION_MODEL

    try:
        if provider == "openai":
            from app.services import openai_service
            response = await openai_service.chat(
                message=prompt,
                model=model,
                context=[{"role": "user", "content": prompt}],
            )
            raw = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        elif provider == "ollama":
            from app.services import ollama_service
            response = await ollama_service.chat(
                message=prompt,
                model=model,
                context=[{"role": "user", "content": prompt}],
            )
            raw = response.get("message", {}).get("content", "")
        else:
            logger.error("Provider KAG inconnu pour enrichissement parent: %s", provider)
            return None

        result = _parse_summary_questions_response(raw)
        if result:
            logger.debug(
                "Enrichissement parent OK (provider=%s): summary=%d chars, questions=%d",
                provider,
                len(result["summary"]),
                len(result["generated_questions"]),
            )
        return result

    except Exception as e:
        logger.error("Erreur génération résumé parent: %s", e, exc_info=True)
        return None


def generate_parent_summary_questions_sync(content: str) -> Optional[Dict]:
    """
    Version synchrone de generate_parent_summary_questions.
    Utilisée dans les workers de background (même pattern que extract_entities_sync).
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    generate_parent_summary_questions(content),
                )
                return future.result(timeout=60)
        else:
            return loop.run_until_complete(generate_parent_summary_questions(content))
    except RuntimeError:
        return asyncio.run(generate_parent_summary_questions(content))
    except Exception as e:
        logger.warning("Génération résumé parent échouée: %s", e)
        return None


async def extract_entities_from_chunk(chunk_content: str) -> List[Dict]:
    """
    Extrait les entités d'un chunk via LLM.
    
    Args:
        chunk_content: Contenu textuel du chunk
        
    Returns:
        Liste de dicts {"name": str, "type": str, "importance": float}
    """
    if not chunk_content or len(chunk_content.strip()) < 20:
        return []
    
    content_truncated = chunk_content[:2000]
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        entity_types=", ".join(ENTITY_TYPES),
        chunk_content=content_truncated,
    )
    
    provider = settings.KAG_EXTRACTION_PROVIDER.lower()
    model = settings.KAG_EXTRACTION_MODEL
    
    try:
        if provider == "openai":
            from app.services import openai_service
            response = await openai_service.chat(
                message=prompt,
                model=model,
                context=[{"role": "user", "content": prompt}],
            )
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        elif provider == "ollama":
            from app.services import ollama_service
            response = await ollama_service.chat(
                message=prompt,
                model=model,
                context=[{"role": "user", "content": prompt}],
            )
            content = response.get("message", {}).get("content", "")
        
        else:
            logger.error("Provider KAG inconnu: %s", provider)
            return []
        
        entities = _parse_llm_response(content)
        logger.debug(
            "Extraction KAG: %d entités extraites (provider=%s, model=%s)",
            len(entities),
            provider,
            model,
        )
        return entities
    
    except Exception as e:
        logger.error("Erreur extraction KAG: %s", e, exc_info=True)
        return []


async def extract_entities_from_query(query_text: str) -> List[str]:
    """
    Extrait les entités mentionnées dans une requête utilisateur.
    Version simplifiée pour le retrieval (juste les noms).
    
    Args:
        query_text: Texte de la requête
        
    Returns:
        Liste des noms d'entités normalisés
    """
    entities = await extract_entities_from_chunk(query_text)
    return [normalize_entity_name(e["name"]) for e in entities if e.get("name")]


def extract_entities_sync(chunk_content: str) -> List[Dict]:
    """
    Version synchrone de extract_entities_from_chunk.
    Utile pour les workers de background.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    extract_entities_from_chunk(chunk_content)
                )
                return future.result(timeout=60)
        else:
            return loop.run_until_complete(extract_entities_from_chunk(chunk_content))
    except RuntimeError:
        return asyncio.run(extract_entities_from_chunk(chunk_content))


def extract_entities_from_query_sync(query_text: str) -> List[str]:
    """
    Version synchrone de extract_entities_from_query.
    Retourne les noms d'entités normalisés de la requête (pour pivot KAG).
    """
    if not query_text or not query_text.strip():
        return []
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    extract_entities_from_query(query_text),
                )
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(extract_entities_from_query(query_text))
    except RuntimeError:
        return asyncio.run(extract_entities_from_query(query_text))
    except Exception as e:
        logger.warning("Extraction entités requête échouée: %s", e)
        return []
