import time
from unittest import mock
import pytest
import httpx

from app.config import settings
from app.services.discord_service import notify_document_status, _send_webhook_sync


def test_notify_discord_no_webhook():
    # Arrange: ensure webhook URL is None/unset
    with mock.patch.object(settings, "DISCORD_WEBHOOK_URL", None):
        # Act & Assert: Should return immediately and not call httpx
        with mock.patch("httpx.Client") as mock_client_cls:
            notify_document_status(
                document_id=42,
                document_title="Test Document",
                status="completed",
                chunks_count=15,
            )
            # Since notify_document_status runs in a thread, we wait a tiny bit to make sure thread executes (though it exits immediately)
            time.sleep(0.1)
            mock_client_cls.assert_not_called()


def test_notify_discord_success():
    webhook_url = "https://discord.com/api/webhooks/mock_test_webhook"
    
    with mock.patch.object(settings, "DISCORD_WEBHOOK_URL", webhook_url):
        mock_response = mock.Mock(spec=httpx.Response)
        mock_response.status_code = 200
        
        mock_client = mock.MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        
        with mock.patch("httpx.Client", return_value=mock_client):
            # We call the synchronous helper directly to avoid thread race conditions in testing
            _send_webhook_sync(
                document_id=101,
                document_title="Budget 2026.pdf",
                status="completed",
                chunks_count=57,
            )
            
            # Assert client was created and post called
            mock_client.post.assert_called_once()
            called_url, called_kwargs = mock_client.post.call_args
            
            assert called_url[0] == webhook_url
            payload = called_kwargs["json"]
            assert payload["username"] == "Noton Notification"
            assert len(payload["embeds"]) == 1
            
            embed = payload["embeds"][0]
            assert embed["title"] == "✅ Document traité avec succès"
            assert "Budget 2026.pdf" in embed["description"]
            
            # Check fields
            fields = embed["fields"]
            assert any(f["name"] == "ID du Document" and f["value"] == "101" for f in fields)
            assert any(f["name"] == "Statut" and f["value"] == "Terminé" for f in fields)
            assert any(f["name"] == "Chunks générés" and f["value"] == "57" for f in fields)


def test_notify_discord_failure():
    webhook_url = "https://discord.com/api/webhooks/mock_test_webhook"
    
    with mock.patch.object(settings, "DISCORD_WEBHOOK_URL", webhook_url):
        mock_response = mock.Mock(spec=httpx.Response)
        mock_response.status_code = 200
        
        mock_client = mock.MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        
        with mock.patch("httpx.Client", return_value=mock_client):
            # We call the synchronous helper directly
            _send_webhook_sync(
                document_id=102,
                document_title="Rapport_Cassé.pdf",
                status="failed",
                error_message="Mistral OCR API timeout after 300s",
            )
            
            # Assert client was created and post called
            mock_client.post.assert_called_once()
            called_url, called_kwargs = mock_client.post.call_args
            
            assert called_url[0] == webhook_url
            payload = called_kwargs["json"]
            
            embed = payload["embeds"][0]
            assert embed["title"] == "❌ Échec du traitement du document"
            assert "Rapport_Cassé.pdf" in embed["description"]
            
            # Check fields
            fields = embed["fields"]
            assert any(f["name"] == "ID du Document" and f["value"] == "102" for f in fields)
            assert any(f["name"] == "Statut" and f["value"] == "Échoué" for f in fields)
            assert any(f["name"] == "Détails de l'erreur" and "Mistral OCR API timeout" in f["value"] for f in fields)
