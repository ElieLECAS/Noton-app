import pytest
import httpx
import json
from unittest import mock
from app.services.mistral_service import _post_json_with_retry, chat_stream
from app.services.lancedb_service import insert_colpali_patches_lancedb, get_colpali_table


@pytest.mark.asyncio
async def test_post_json_with_retry_network_error_retry():
    # Test that _post_json_with_retry retries on httpx.RequestError and eventually succeeds
    mock_client = mock.AsyncMock(spec=httpx.AsyncClient)
    
    # First call raises ReadError (subclass of RequestError), second call returns a 200 OK response
    mock_response = mock.Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "Hello"}}]}
    
    mock_client.post.side_effect = [
        httpx.ReadError("Simulated read error"),
        mock_response
    ]
    
    with mock.patch("app.services.mistral_service.MAX_RETRIES", 3), \
         mock.patch("app.services.mistral_service.RETRY_BACKOFF_BASE_SECONDS", 0.01):
        
        response = await _post_json_with_retry(
            client=mock_client,
            url="https://api.mistral.ai/v1/chat/completions",
            headers={},
            payload={}
        )
        
        assert response.status_code == 200
        assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_post_json_with_retry_network_error_exhausted():
    # Test that _post_json_with_retry raises after MAX_RETRIES failures
    mock_client = mock.AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = httpx.ConnectError("Simulated connection error")
    
    with mock.patch("app.services.mistral_service.MAX_RETRIES", 2), \
         mock.patch("app.services.mistral_service.RETRY_BACKOFF_BASE_SECONDS", 0.01):
        
        with pytest.raises(httpx.ConnectError):
            await _post_json_with_retry(
                client=mock_client,
                url="https://api.mistral.ai/v1/chat/completions",
                headers={},
                payload={}
            )
        
        assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_chat_stream_network_error_retry_before_yield():
    # Test that chat_stream retries the connection if a network error occurs before yielding anything
    mock_client = mock.AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = mock.Mock()
    mock_client.__aenter__.return_value = mock_client
    
    # Mock settings.MISTRAL_API_KEY
    with mock.patch("app.services.mistral_service.settings") as mock_settings:
        mock_settings.MISTRAL_API_KEY = "test-key"
        mock_settings.MISTRAL_BASE_URL = "https://api.mistral.ai"
        mock_settings.MAX_COMPLETION_TOKENS = 100
        
        # We need mock_client.stream context manager
        mock_stream_ctx_success = mock.AsyncMock()
        mock_response = mock.Mock()
        mock_response.status_code = 200
        
        # Async generator for lines
        async def mock_aiter_lines():
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]})
            yield "data: [DONE]"
            
        mock_response.aiter_lines = mock_aiter_lines
        mock_stream_ctx_success.__aenter__.return_value = mock_response
        
        # First call to stream raises ReadError, second succeeds
        mock_client.stream.side_effect = [
            httpx.ReadError("Connection lost"),
            mock_stream_ctx_success
        ]
        
        with mock.patch("httpx.AsyncClient", return_value=mock_client), \
             mock.patch("app.services.mistral_service.MAX_RETRIES", 3), \
             mock.patch("app.services.mistral_service.RETRY_BACKOFF_BASE_SECONDS", 0.01):
            
            chunks = []
            async for chunk in chat_stream("Hello", "mistral-small"):
                chunks.append(json.loads(chunk))
                
            assert len(chunks) == 1
            assert chunks[0]["message"]["content"] == "Hello"
            assert mock_client.stream.call_count == 2


@pytest.mark.asyncio
async def test_chat_stream_network_error_no_retry_after_yield():
    # Test that chat_stream does NOT retry and raises error if connection drops after some chunks have been yielded
    mock_client = mock.AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = mock.Mock()
    mock_client.__aenter__.return_value = mock_client
    
    with mock.patch("app.services.mistral_service.settings") as mock_settings:
        mock_settings.MISTRAL_API_KEY = "test-key"
        mock_settings.MISTRAL_BASE_URL = "https://api.mistral.ai"
        mock_settings.MAX_COMPLETION_TOKENS = 100
        
        mock_stream_ctx = mock.AsyncMock()
        mock_response = mock.Mock()
        mock_response.status_code = 200
        
        # Generator yields one item then raises ReadError
        async def mock_aiter_lines_error():
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "First Part"}, "finish_reason": None}]})
            raise httpx.ReadError("Mid-stream failure")
            
        mock_response.aiter_lines = mock_aiter_lines_error
        mock_stream_ctx.__aenter__.return_value = mock_response
        
        mock_client.stream.return_value = mock_stream_ctx
        
        with mock.patch("httpx.AsyncClient", return_value=mock_client), \
             mock.patch("app.services.mistral_service.MAX_RETRIES", 3), \
             mock.patch("app.services.mistral_service.RETRY_BACKOFF_BASE_SECONDS", 0.01):
            
            chunks = []
            with pytest.raises(httpx.ReadError):
                async for chunk in chat_stream("Hello", "mistral-small"):
                    chunks.append(json.loads(chunk))
            
            # Should have yielded the first chunk
            assert len(chunks) == 1
            assert chunks[0]["message"]["content"] == "First Part"
            # Should NOT have retried (call_count must be exactly 1)
            assert mock_client.stream.call_count == 1


def test_insert_colpali_patches_index_creation():
    # Test that insert_colpali_patches_lancedb calls table.create_index
    mock_table = mock.Mock()
    
    with mock.patch("app.services.lancedb_service.get_colpali_table", return_value=mock_table):
        insert_colpali_patches_lancedb(
            document_id=123,
            chunk_id=456,
            patch_vectors=[[0.1] * 128, [0.2] * 128]
        )
        
        # Verify it deleted, added, and called create_index
        mock_table.delete.assert_called_once_with("chunk_id = 456")
        mock_table.add.assert_called_once()
        mock_table.create_index.assert_called_once_with(
            vector_column_name="vector",
            index_type="IVF_SQ",
            metric="cosine"
        )


def test_insert_colpali_patches_batch_index_creation():
    # Test that insert_colpali_patches_batch_lancedb calls table.delete on document level and table.create_index
    mock_table = mock.Mock()
    from app.services.lancedb_service import insert_colpali_patches_batch_lancedb
    
    with mock.patch("app.services.lancedb_service.get_colpali_table", return_value=mock_table):
        insert_colpali_patches_batch_lancedb(
            document_id=123,
            chunk_patches_list=[
                (456, [[0.1] * 128]),
                (457, [[0.2] * 128])
            ]
        )
        
        # Verify it deleted document-wide, added, and called create_index
        mock_table.delete.assert_called_once_with("document_id = 123")
        mock_table.add.assert_called_once()
        mock_table.create_index.assert_called_once_with(
            vector_column_name="vector",
            index_type="IVF_SQ",
            metric="cosine"
        )
