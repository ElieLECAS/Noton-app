import logging
import threading
from typing import Optional
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def _send_webhook_sync(
    document_id: int,
    document_title: str,
    status: str,
    chunks_count: Optional[int] = None,
    error_message: Optional[str] = None,
):
    webhook_url = settings.DISCORD_WEBHOOK_URL
    if not webhook_url:
        logger.debug("Discord webhook URL is not configured. Skipping notification.")
        return

    # Construire le payload d'embed
    if status == "completed":
        title = "✅ Document traité avec succès"
        color = 3066993  # Vert
        description = f"Le document **{document_title}** a été traité avec succès et est prêt à l'emploi."
        fields = [
            {"name": "ID du Document", "value": str(document_id), "inline": True},
            {"name": "Statut", "value": "Terminé", "inline": True},
        ]
        if chunks_count is not None:
            fields.append({"name": "Chunks générés", "value": str(chunks_count), "inline": True})
    else:
        title = "❌ Échec du traitement du document"
        color = 15158332  # Rouge
        description = f"Le traitement du document **{document_title}** a échoué."
        fields = [
            {"name": "ID du Document", "value": str(document_id), "inline": True},
            {"name": "Statut", "value": "Échoué", "inline": True},
        ]
        if error_message:
            # Tronquer à 1000 caractères pour respecter les limites d'embed Discord (1024 max pour un champ)
            truncated_error = str(error_message)[:1000]
            fields.append(
                {"name": "Détails de l'erreur", "value": f"```\n{truncated_error}\n```", "inline": False}
            )

    embed = {
        "title": title,
        "description": description,
        "color": color,
        "fields": fields,
        "footer": {"text": "Noton Document Processor"},
    }

    payload = {
        "username": "Noton Notification",
        "embeds": [embed],
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(webhook_url, json=payload)
            if response.status_code >= 400:
                logger.error(
                    "Échec de l'envoi de la notification Discord. Code statut : %d, réponse : %s",
                    response.status_code,
                    response.text,
                )
            else:
                logger.info(
                    "Notification Discord envoyée avec succès pour le document_id=%s, statut=%s",
                    document_id,
                    status,
                )
    except Exception as e:
        logger.exception("Erreur lors de l'envoi de la notification Discord : %s", e)


def notify_document_status(
    document_id: int,
    document_title: str,
    status: str,
    chunks_count: Optional[int] = None,
    error_message: Optional[str] = None,
):
    """
    Envoie une notification sur Discord pour indiquer le statut du traitement d'un document.
    S'exécute dans un thread séparé pour ne pas bloquer le traitement principal.
    """
    if not settings.DISCORD_WEBHOOK_URL:
        return

    thread = threading.Thread(
        target=_send_webhook_sync,
        args=(document_id, document_title, status, chunks_count, error_message),
        daemon=True,
    )
    thread.start()
