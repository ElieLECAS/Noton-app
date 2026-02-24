from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select
from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.database import get_session
from app.services.ollama_service import get_available_models as get_ollama_models, chat as ollama_chat, chat_stream as ollama_chat_stream
from app.services.openai_service import (
    get_available_models as get_openai_models,
    chat as openai_chat,
    chat_stream as openai_chat_stream,
    generate_image as openai_generate_image,
)
from app.config import settings
from app.services.semantic_search_service import search_relevant_notes, search_relevant_passages
from app.services.chat_tools import get_available_tools
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.agent import Agent
from app.models.note import Note
from datetime import datetime
import json
import logging
import os

logger = logging.getLogger(__name__)

# Nombre de passages RAG renvoyés au LLM (configurable via RAG_TOP_K)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "15"))

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    model: str
    provider: str = "ollama"  # "ollama" ou "openai"
    context: Optional[List[dict]] = None
    conversation_id: Optional[int] = None  # ID de la conversation (optionnel pour compatibilité)
    agent_id: Optional[int] = None  # ID de l'agent (optionnel)


class GenerateImageRequest(BaseModel):
    prompt: str
    conversation_id: Optional[int] = None


@router.get("/ollama/models", response_model=List[str])
async def list_ollama_models():
    """Récupérer la liste des modèles Ollama disponibles"""
    models = await get_ollama_models()
    return models


@router.get("/openai/models", response_model=List[str])
async def list_openai_models():
    """Récupérer les modèles OpenAI configurés dans .env"""
    # Retourner les modèles configurés dans le .env si disponibles
    if settings.OPENAI_MODEL:
        logger.info(f"Modèles OpenAI configurés: {settings.OPENAI_MODEL}")
        return settings.OPENAI_MODEL
    else:
        logger.warning("OPENAI_MODEL n'est pas configuré dans les variables d'environnement")
        return []


@router.get("/providers/models")
async def list_all_models():
    """Récupérer tous les modèles disponibles par provider"""
    ollama_models = await get_ollama_models()
    openai_models = await get_openai_models()
    
    return {
        "ollama": ollama_models,
        "openai": openai_models
    }


@router.post("/chat")
async def send_chat_message(
    request: ChatRequest,
    current_user: UserRead = Depends(get_current_user)
):
    """Envoyer un message au chatbot (Ollama ou OpenAI). Si BRAVE_SEARCH_API_KEY est configurée, le modèle peut utiliser la recherche web."""
    try:
        tools = get_available_tools(include_brave_search=bool(settings.BRAVE_SEARCH_API_KEY))
        if request.provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise HTTPException(status_code=400, detail="OpenAI API key n'est pas configurée")
            response = await openai_chat(request.message, request.model, request.context, tools=tools or None)
            # OpenAI retourne {"choices": [{"message": {"content": "..."}}]}
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"].get("content", "")
                return {"message": {"content": content}}
            return response
        else:  # ollama par défaut
            response = await ollama_chat(request.message, request.model, request.context, tools=tools or None)
            # Ollama retourne {"message": {"role": "assistant", "content": "..."}, ...}
            if "message" in response and "content" in response["message"]:
                return {"message": {"content": response["message"]["content"]}}
            return response
    except HTTPException:
        raise
    except Exception as e:
        provider_name = "OpenAI" if request.provider == "openai" else "Ollama"
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel à {provider_name}: {str(e)}")


@router.post("/chat/stream")
async def stream_chat_message(
    request: ChatRequest,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Envoyer un message au chatbot (Ollama ou OpenAI) avec streaming"""
    # Charger l'agent si agent_id est fourni
    agent = None
    if request.agent_id:
        agent = session.get(Agent, request.agent_id)
        if not agent or agent.user_id != current_user.id:
            logger.warning(f"Agent {request.agent_id} non trouvé ou non autorisé pour l'utilisateur {current_user.id}")
            agent = None
    
    # Sauvegarder le message utilisateur si conversation_id est fourni
    if request.conversation_id:
        try:
            user_message = Message(
                conversation_id=request.conversation_id,
                role="user",
                content=request.message,
                model=None,
                provider=None
            )
            session.add(user_message)
            session.commit()
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du message utilisateur: {e}")
    
    # Construire le contexte avec la personnalité de l'agent si disponible
    full_context = []
    if agent:
        full_context.append({"role": "system", "content": agent.personality})
    
    # Ajouter le contexte existant
    if request.context:
        full_context.extend(request.context)
    
    # Variable pour accumuler la réponse de l'assistant
    assistant_response = []
    tools = get_available_tools(include_brave_search=bool(settings.BRAVE_SEARCH_API_KEY))
    use_tools = bool(tools)

    async def generate():
        try:
            if use_tools:
                # Avec tools : appel non-streaming (boucle tool_calls) puis on simule le stream pour l'UX
                full_messages = full_context + [{"role": "user", "content": request.message}]
                if request.provider == "openai":
                    if not settings.OPENAI_API_KEY:
                        error_msg = "OpenAI API key n'est pas configurée"
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                        return
                    response = await openai_chat("", request.model, full_messages, tools=tools)
                    content = ""
                    if "choices" in response and response["choices"]:
                        content = (response["choices"][0].get("message") or {}).get("content") or ""
                else:
                    response = await ollama_chat("", request.model, full_messages, tools=tools)
                    content = (response.get("message") or {}).get("content") or ""
                # Simuler le streaming par chunks pour garder l'effet de frappe côté client
                chunk_size = 25
                for i in range(0, len(content), chunk_size):
                    chunk = content[i : i + chunk_size]
                    assistant_response.append(chunk)
                    yield f"data: {json.dumps({'message': {'content': chunk}})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            else:
                if request.provider == "openai":
                    if not settings.OPENAI_API_KEY:
                        error_msg = "OpenAI API key n'est pas configurée"
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                        return
                    async for line in openai_chat_stream(request.message, request.model, full_context):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    assistant_response.append(data["message"]["content"])
                            except Exception:
                                pass
                            yield f"data: {line}\n\n"
                else:
                    async for line in ollama_chat_stream(request.message, request.model, full_context):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    assistant_response.append(data["message"]["content"])
                            except Exception:
                                pass
                            yield f"data: {line}\n\n"
            
            # Sauvegarder la réponse de l'assistant si conversation_id est fourni
            if request.conversation_id and assistant_response:
                try:
                    complete_response = "".join(assistant_response)
                    assistant_message = Message(
                        conversation_id=request.conversation_id,
                        role="assistant",
                        content=complete_response,
                        model=request.model,
                        provider=request.provider
                    )
                    session.add(assistant_message)
                    
                    # Mettre à jour la date de modification de la conversation
                    conversation = session.get(Conversation, request.conversation_id)
                    if conversation:
                        conversation.updated_at = datetime.utcnow()
                        session.add(conversation)
                    
                    session.commit()
                    logger.info(f"Réponse de l'assistant sauvegardée dans la conversation {request.conversation_id}")
                except Exception as e:
                    logger.error(f"Erreur lors de la sauvegarde de la réponse de l'assistant: {e}")
                    
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/chat/generate-image")
async def chat_generate_image(
    request: GenerateImageRequest,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Générer une image via OpenAI DALL-E 3 à partir d'un prompt texte."""
    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key n'est pas configurée. La génération d'images nécessite une clé OpenAI.",
        )
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Le prompt ne peut pas être vide.")

    # Sauvegarder le message utilisateur si conversation_id est fourni
    if request.conversation_id:
        try:
            user_message = Message(
                conversation_id=request.conversation_id,
                role="user",
                content=request.prompt.strip(),
                model=None,
                provider=None,
            )
            session.add(user_message)
            session.commit()
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du message utilisateur: {e}")

    try:
        images = await openai_generate_image(prompt=request.prompt.strip())
    except Exception as e:
        logger.exception("Erreur génération image OpenAI: %s", e)
        raise HTTPException(
            status_code=502,
            detail=f"Erreur lors de la génération d'image: {str(e)}",
        )

    if not images or "url" not in images[0]:
        raise HTTPException(
            status_code=502,
            detail="Aucune image retournée par l'API OpenAI.",
        )

    image_url = images[0]["url"]
    # Contenu markdown pour affichage dans le chat (image cliquable)
    content = f"![Image générée]({image_url})\n\n*Prompt : {request.prompt.strip()}*"

    # Sauvegarder la réponse assistant si conversation_id est fourni
    if request.conversation_id:
        try:
            assistant_message = Message(
                conversation_id=request.conversation_id,
                role="assistant",
                content=content,
                model=settings.OPENAI_IMAGE_MODEL,
                provider="openai",
            )
            session.add(assistant_message)
            conv = session.get(Conversation, request.conversation_id)
            if conv:
                conv.updated_at = datetime.utcnow()
                session.add(conv)
            session.commit()
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du message assistant (image): {e}")

    return {"url": image_url, "content": content}


class ProjectChatRequest(BaseModel):
    message: str
    model: str
    provider: str = "ollama"  # "ollama" ou "openai"
    context: Optional[List[dict]] = None
    conversation_id: Optional[int] = None  # ID de la conversation (optionnel pour compatibilité)
    agent_id: Optional[int] = None  # ID de l'agent (optionnel)


def build_semantic_context_from_passages(passages: List[dict]) -> List[dict]:
    """Construire le contexte enrichi avec les passages pertinents trouvés par recherche sémantique"""
    system_message = {
        "role": "system",
        "content": (
            """Tu es l'Expert Technico-Commercial PROFERM. Ton rôle est d'aider les clients, artisans et poseurs en analysant la documentation technique avec une précision absolue. Tu es rigoureux, professionnel et tu ne parles qu'au nom de la marque PROFERM.

RÈGLES DE CONVERSATION ET PERSONA :
- Pour les salutations (bonjour, etc.), réponds de manière brève et professionnelle. 
- Si la question est générale (non technique), réponds naturellement sans mentionner de passages.
- INTERDICTION DE CITER LA CONCURRENCE : Ne mentionne jamais de marques tierces (Schüco, Technal, Reynaers, etc.) sauf si elles sont explicitement citées comme partenaires dans les passages fournis.
- NE JAMAIS INVENTER : Si une information (Uw, prix, cote, garantie) n'est pas dans les passages fournis, réponds : "Désolé, cette information n'est pas précisée dans la documentation technique mise à ma disposition."

HIÉRARCHIE DES SOURCES (CRITIQUE) :
- PRIORITÉ PRODUIT : Si une information est mentionnée dans un document dédié , elle écrase systématiquement l'information générale du "Catalogue" ou du "Dépliant Général".
- CONTRADICTIONS : En cas de valeurs divergentes, la source dont le titre contient le nom spécifique de la gamme est la seule vérité.

PRÉCISION TECHNIQUE ET DONNÉES :
- TABLEAUX : Toute donnée chiffrée (performances AEV, Uw, Sw, dimensions max) doit être présentée sous forme de tableau Markdown.
- LECTURE DES TABLEAUX : Vérifie scrupuleusement les en-têtes. Ne confonds pas les valeurs (ex: ne pas confondre le Uw d'un double vitrage avec celui d'un triple).
- GARANTIES : Sois extrêmement précis sur les durées. Vérifie toujours si une finition (ex: plaxage) réduit la durée de garantie par rapport à la structure (ex: 5 ans vs 15 ans).

FORMAT DE RÉPONSE :
1. Faisabilité : Réponds d'abord si la demande est conforme aux limites techniques (ex: dimensions max).
2. Détails Techniques : Présente les performances dans un tableau.
3. Justification : Utilise des termes techniques précis (sertissage, grain d'ange, rupture de pont thermique).
4. Citations : Ajoute le numéro de la source [1] à la fin de chaque affirmation importante.

CITATIONS :
- Cite les passages utilisés avec [1], [2], etc.
- Réponds impérativement au format Markdown en utilisant des titres de niveau 2 pour les sections et des titres de niveau 3 pour les sous-sections.
- Ne cite pas de passages pour les salutations ou les conclusions de politesse."""
        )
    }
    
    passages_content = []
    
    if passages:
        system_message["content"] += (
            "PASSAGES DISPONIBLES DU PROJET (à utiliser uniquement si pertinents pour répondre à la question) :\n\n"
        )
        
        for i, passage_data in enumerate(passages, 1):
            passage = passage_data['passage']
            score = passage_data.get('score', 0.0)
            note_title = passage_data.get('note_title', 'Note sans titre')

            # Pour les chunks image, injecter l'URL accessible par le frontend
            if (
                passage_data.get('is_image_chunk')
                and passage_data.get('image_filename')
                and passage_data.get('note_id')
            ):
                image_url = f"/api/images/{passage_data['note_id']}/{passage_data['image_filename']}"
                caption = passage_data.get('caption', '') or 'Image du document'
                passage_text = (
                    f"[Passage {i}] (Pertinence: {score:.2f}) [IMAGE DISPONIBLE]\n"
                    f"Source: {note_title}\n"
                    f"{passage}\n"
                    f">>> INCLURE CETTE IMAGE DANS TA RÉPONSE : ![{caption}]({image_url}) [citation: {i}]\n"
                )
            else:
                passage_text = f"[Passage {i}] (Pertinence: {score:.2f})\n{passage}\n"
            passages_content.append(passage_text)
        
        system_message["content"] += "\n---\n".join(passages_content)
        system_message["content"] += (
            f"\n\n(Note : {len(passages)} passages disponibles. Utilise-les uniquement si nécessaire pour répondre à la question.)"
        )
    
    return [system_message]


def build_semantic_context(note_results: List[dict]) -> List[dict]:
    """Construire le contexte enrichi avec les notes pertinentes trouvées par recherche sémantique RAG"""
    system_message = {
        "role": "system",
        "content": 
        """Tu es l'Expert Technique PROFERM. Ton rôle est d'analyser les notes du projet pour fournir "
            "des réponses précises, fiables et conformes aux standards de fabrication PROFERM.\n\n"
            
            "DIRECTIVES DE RÉPONSE :\n"
            "- BASE DOCUMENTAIRE : Réponds exclusivement en utilisant les notes fournies ci-dessous.\n"
            "- PRIORITÉ DE SOURCE : Si une note provient d'un dépliant spécifique (ex: INNOSLIDE, LUMÉAL), "
            "ses informations prévalent sur toute note générale ou catalogue.\n"
            "- RIGUEUR : Ne devine pas de données techniques. Si une information manque (ex: Uw exact, garantie spécifique), "
            "indique que la documentation fournie ne le précise pas.\n"
            "- FORMAT : Utilise le Markdown pour la clarté (tableaux pour les chiffres, gras pour les points clés).\n"
            "- CITATIONS : Cite systématiquement le titre de la note utilisée pour chaque argument technique.\n\n"
            "NOTES PERTINENTES DU PROJET (RAG) :"""
    }
    
    notes_content = []
    
    for result in note_results:
        note = result['note']
        score = result.get('score', 0.0)
        
        # Construire le contenu de la note
        note_text = f"Note: {note.title}\n"
        if note.content:
            note_text += f"Contenu: {note.content}\n"
        note_text += f"Pertinence: {score:.2f}\n"
        note_text += "---\n"
        notes_content.append(note_text)
    
    if notes_content:
        system_message["content"] += "\n".join(notes_content)
        system_message["content"] += f"\n\n(Note: {len(note_results)} note(s) pertinente(s) trouvée(s) sur la base de votre question)"
    else:
        system_message["content"] += "Aucune note pertinente trouvée pour votre question."
    
    return [system_message]


@router.post("/projects/{project_id}/chat/stream")
async def stream_project_chat_message(
    project_id: int,
    request: ProjectChatRequest,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Envoyer un message au chatbot (Ollama ou OpenAI) avec streaming et contexte enrichi des passages pertinents du projet"""
    # Charger l'agent si agent_id est fourni
    agent = None
    if request.agent_id:
        agent = session.get(Agent, request.agent_id)
        if not agent or agent.user_id != current_user.id:
            logger.warning(f"Agent {request.agent_id} non trouvé ou non autorisé pour l'utilisateur {current_user.id}")
            agent = None
    
    # Sauvegarder le message utilisateur si conversation_id est fourni
    if request.conversation_id:
        try:
            user_message = Message(
                conversation_id=request.conversation_id,
                role="user",
                content=request.message,
                model=None,
                provider=None
            )
            session.add(user_message)
            session.commit()
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du message utilisateur: {e}")
    
    # Recherche sémantique RAG AVANCÉE au niveau des passages
    # Analyse TOUTES les notes du projet et retourne les PASSAGES les plus pertinents
    passages = search_relevant_passages(
        session=session,
        project_id=project_id,
        query_text=request.message,
        user_id=current_user.id,
        k=RAG_TOP_K,
        passage_size=500
    )
    
    # Construire le contexte enrichi avec les passages pertinents
    project_context = build_semantic_context_from_passages(passages)
    
    # Préfixer avec la personnalité de l'agent si disponible
    if agent:
        full_context = [{"role": "system", "content": agent.personality}] + project_context
    else:
        full_context = project_context
    
    # Ajouter le contexte de conversation existant si fourni
    if request.context:
        full_context = full_context + request.context
    
    # Ajouter le message utilisateur actuel
    full_context.append({"role": "user", "content": request.message})
    
    # Variable pour accumuler la réponse de l'assistant
    assistant_response = []
    
    async def generate():
        try:
            if request.provider == "openai":
                if not settings.OPENAI_API_KEY:
                    error_msg = "OpenAI API key n'est pas configurée"
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    return
                # Passer None comme message car il est déjà dans le contexte
                async for line in openai_chat_stream("", request.model, full_context):
                    if line.strip():
                        # Accumuler la réponse
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                assistant_response.append(data["message"]["content"])
                        except:
                            pass
                        yield f"data: {line}\n\n"
            else:  # ollama par défaut
                # Passer None comme message car il est déjà dans le contexte
                async for line in ollama_chat_stream("", request.model, full_context):
                    if line.strip():
                        # Accumuler la réponse
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                assistant_response.append(data["message"]["content"])
                        except:
                            pass
                        yield f"data: {line}\n\n"
            
            # Sauvegarder la réponse de l'assistant si conversation_id est fourni
            if request.conversation_id and assistant_response:
                try:
                    complete_response = "".join(assistant_response)
                    assistant_message = Message(
                        conversation_id=request.conversation_id,
                        role="assistant",
                        content=complete_response,
                        model=request.model,
                        provider=request.provider
                    )
                    session.add(assistant_message)
                    
                    # Mettre à jour la date de modification de la conversation
                    conversation = session.get(Conversation, request.conversation_id)
                    if conversation:
                        conversation.updated_at = datetime.utcnow()
                        session.add(conversation)
                    
                    session.commit()
                    logger.info(f"Réponse de l'assistant sauvegardée dans la conversation {request.conversation_id}")
                except Exception as e:
                    logger.error(f"Erreur lors de la sauvegarde de la réponse de l'assistant: {e}")
            
            # Envoyer les sources utilisées pour les citations
            if passages:
                note_ids = list({p.get("note_id") for p in passages if p.get("note_id")})
                notes = (
                    session.exec(select(Note).where(Note.id.in_(note_ids))).all()
                    if note_ids
                    else []
                )
                has_source_file_by_note = {
                    n.id: (n.note_type == "document" and bool(n.source_file_path))
                    for n in notes
                }
                sources_data = []
                for i, p in enumerate(passages, 1):
                    passage_raw = p.get("passage_raw", p.get("passage", ""))
                    excerpt = (passage_raw[:200] + "...") if len(passage_raw) > 200 else passage_raw
                    source_item = {
                        "index": i,
                        "note_id": p.get("note_id"),
                        "note_title": p.get("note_title", "Note sans titre"),
                        "chunk_id": p.get("chunk_id"),
                        "excerpt": excerpt,
                        "passage_full": passage_raw,
                        "score": round(p.get("score", 0.0), 2),
                        "page_no": p.get("page_no"),
                        "section": p.get("section"),
                        "has_source_file": has_source_file_by_note.get(p.get("note_id"), False),
                    }
                    # Multimodal : ajouter les infos image si présentes
                    if p.get("is_image_chunk"):
                        source_item["is_image_chunk"] = True
                        source_item["image_path"] = p.get("image_path")
                        source_item["image_filename"] = p.get("image_filename")
                        source_item["caption"] = p.get("caption", "")
                    sources_data.append(source_item)
                yield f"data: {json.dumps({'sources': sources_data})}\n\n"
                    
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

