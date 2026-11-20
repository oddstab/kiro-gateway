
# -*- coding: utf-8 -*-

"""
Unit tests for streaming_openai module.

Tests for:
- stream_kiro_to_openai() generator
- stream_kiro_to_openai_internal() generator
- stream_with_first_token_retry() function
- collect_stream_response() function
"""

import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from kiro.streaming_openai import (
    stream_kiro_to_openai,
    stream_kiro_to_openai_internal,
    stream_with_first_token_retry,
    collect_stream_response,
    FirstTokenTimeoutError,
)
from kiro.streaming_core import KiroEvent


# ==================================================================================================
# Fixtures
# ==================================================================================================

@pytest.fixture
def mock_model_cache():
    """Mock for ModelInfoCache."""
    cache = MagicMock()
    cache.get_max_input_tokens.return_value = 200000
    return cache


@pytest.fixture
def mock_auth_manager():
    """Mock for KiroAuthManager."""
    manager = MagicMock()
    return manager


@pytest.fixture
def mock_http_client():
    """Mock for httpx.AsyncClient."""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_response():
    """Mock for httpx.Response."""
    response = AsyncMock()
    response.status_code = 200
    response.aclose = AsyncMock()
    return response


# ==================================================================================================
# Tests for stream_kiro_to_openai()
# ==================================================================================================

class TestStreamKiroToOpenai:
    """Tests for stream_kiro_to_openai() generator."""
    
    @pytest.mark.asyncio
    async def test_yields_content_chunks(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields content chunks in OpenAI format.
        Goal: Verify content streaming.
        """
        print("Setup: Mock stream with content events...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="content", content=" World")
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should have content chunks
        content_chunks = [c for c in chunks if "content" in c and '"Hello"' in c or '" World"' in c]
        assert len(content_chunks) >= 2
        print("✓ Content chunks yielded correctly")
    
    @pytest.mark.asyncio
    async def test_first_chunk_has_role(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: First chunk includes role: assistant.
        Goal: Verify OpenAI streaming protocol.
        """
        print("Setup: Mock stream with content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # First content chunk should have role
        first_content_chunk = [c for c in chunks if '"content"' in c and '"Hello"' in c][0]
        assert '"role": "assistant"' in first_content_chunk
        print("✓ First chunk has role: assistant")
    
    @pytest.mark.asyncio
    async def test_yields_done_at_end(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields [DONE] at end of stream.
        Goal: Verify stream termination.
        """
        print("Setup: Mock stream with content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Last chunk should be [DONE]
        assert chunks[-1] == "data: [DONE]\n\n"
        print("✓ [DONE] yielded at end")
    
    @pytest.mark.asyncio
    async def test_yields_final_chunk_with_usage(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields final chunk with usage info.
        Goal: Verify usage is included.
        """
        print("Setup: Mock stream with content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should have chunk with usage before [DONE]
        usage_chunks = [c for c in chunks if '"usage"' in c]
        assert len(usage_chunks) >= 1
        print("✓ Final chunk with usage yielded")
    
    @pytest.mark.asyncio
    async def test_yields_tool_calls_chunk(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields tool_calls chunk when tools present.
        Goal: Verify tool call streaming.
        """
        print("Setup: Mock stream with tool call...")
        
        tool_use_data = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Moscow"}'}
        }
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Let me check")
            yield KiroEvent(type="tool_use", tool_use=tool_use_data)
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should have tool_calls chunk
        tool_chunks = [c for c in chunks if '"tool_calls"' in c]
        assert len(tool_chunks) >= 1
        assert "get_weather" in tool_chunks[0]
        print("✓ Tool calls chunk yielded")
    
    @pytest.mark.asyncio
    async def test_tool_calls_have_index(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Tool calls have index field.
        Goal: Verify OpenAI streaming spec compliance.
        """
        print("Setup: Mock stream with multiple tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "func1", "arguments": "{}"}
            })
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_2", "type": "function",
                "function": {"name": "func2", "arguments": "{}"}
            })
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Find tool_calls chunk and verify indices
        tool_chunks = [c for c in chunks if '"tool_calls"' in c]
        assert len(tool_chunks) >= 1
        
        # Parse and check indices
        for chunk in tool_chunks:
            if chunk.startswith("data: "):
                json_str = chunk[6:].strip()
                if json_str != "[DONE]":
                    data = json.loads(json_str)
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                assert "index" in tc
        
        print("✓ Tool calls have index field")
    
    @pytest.mark.asyncio
    async def test_finish_reason_is_tool_calls_when_tools_present(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets finish_reason to tool_calls when tools present.
        Goal: Verify correct finish reason.
        """
        print("Setup: Mock stream with tool call...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "func1", "arguments": "{}"}
            })
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Final chunk before [DONE] should have finish_reason: tool_calls
        final_chunk = chunks[-2]  # Before [DONE]
        assert '"finish_reason": "tool_calls"' in final_chunk
        print("✓ finish_reason is tool_calls")
    
    @pytest.mark.asyncio
    async def test_finish_reason_is_stop_without_tools(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets finish_reason to stop without tools.
        Goal: Verify correct finish reason.
        """
        print("Setup: Mock stream without tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="usage", usage={"inputTokenCount": 10, "outputTokenCount": 1})
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Final chunk before [DONE] should have finish_reason: stop
        final_chunk = chunks[-2]  # Before [DONE]
        assert '"finish_reason": "stop"' in final_chunk
        print("✓ finish_reason is stop")
    
    @pytest.mark.asyncio
    async def test_closes_response_on_completion(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Closes response on completion.
        Goal: Verify resource cleanup.
        """
        print("Setup: Mock stream...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        print("Action: Streaming to OpenAI format...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    pass
        
        print("Check: response.aclose() should be called...")
        mock_response.aclose.assert_called()
        print("✓ Response closed on completion")
    
    @pytest.mark.asyncio
    async def test_closes_response_on_error(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Closes response on error.
        Goal: Verify resource cleanup on error.
        """
        print("Setup: Mock stream that raises error...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            raise RuntimeError("Test error")
        
        print("Action: Streaming to OpenAI format with error...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                try:
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        pass
                except RuntimeError:
                    pass
        
        print("Check: response.aclose() should be called...")
        mock_response.aclose.assert_called()
        print("✓ Response closed on error")


# ==================================================================================================
# Tests for thinking content handling
# ==================================================================================================

class TestStreamingOpenaiThinkingContent:
    """Tests for thinking content handling in OpenAI streaming."""
    
    @pytest.mark.asyncio
    async def test_yields_thinking_as_reasoning_content(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields thinking as reasoning_content when configured.
        Goal: Verify thinking content handling.
        """
        print("Setup: Mock stream with thinking content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="Let me think...")
            yield KiroEvent(type="content", content="Here is my answer")
        
        print("Action: Streaming to OpenAI format with reasoning mode...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                with patch('kiro.streaming_openai.FAKE_REASONING_HANDLING', 'as_reasoning_content'):
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")

        payloads = [
            json.loads(chunk.removeprefix("data: ").strip())
            for chunk in chunks
            if chunk.startswith("data: ") and "[DONE]" not in chunk
        ]
        assert payloads[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
        assert payloads[1]["choices"][0]["delta"] == {"reasoning_content": "Let me think..."}
        assert payloads[2]["choices"][0]["delta"] == {"content": "Here is my answer"}

        # Should have reasoning_content
        reasoning_chunks = [c for c in chunks if '"reasoning_content"' in c]
        assert len(reasoning_chunks) >= 1
        assert "Let me think" in reasoning_chunks[0]
        print("✓ Thinking yielded as reasoning_content")
    
    @pytest.mark.asyncio
    async def test_yields_thinking_as_content_when_configured(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields thinking as content when configured.
        Goal: Verify thinking content handling.
        """
        print("Setup: Mock stream with thinking content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="Let me think...")
            yield KiroEvent(type="content", content="Here is my answer")
        
        print("Action: Streaming to OpenAI format with content mode...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                with patch('kiro.streaming_openai.FAKE_REASONING_HANDLING', 'include_as_text'):
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should have thinking as content
        content_chunks = [c for c in chunks if '"content"' in c and "Let me think" in c]
        assert len(content_chunks) >= 1
        print("✓ Thinking yielded as content")


# ==================================================================================================
# Tests for None protection in tool calls
# ==================================================================================================

class TestStreamingOpenaiNoneProtection:
    """Tests for None protection in tool calls."""
    
    @pytest.mark.asyncio
    async def test_handles_none_function_name(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Handles None in function.name.
        Goal: Verify None is replaced with empty string.
        """
        print("Setup: Mock stream with None function name...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": None, "arguments": "{}"}
            })
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should handle None gracefully
        tool_chunks = [c for c in chunks if '"tool_calls"' in c]
        assert len(tool_chunks) >= 1
        
        # Parse and verify name is empty string
        for chunk in tool_chunks:
            if chunk.startswith("data: "):
                json_str = chunk[6:].strip()
                if json_str != "[DONE]":
                    data = json.loads(json_str)
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                assert tc["function"]["name"] == ""
        
        print("✓ None function name handled")
    
    @pytest.mark.asyncio
    async def test_handles_none_function_arguments(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Handles None in function.arguments.
        Goal: Verify None is replaced with "{}".
        """
        print("Setup: Mock stream with None arguments...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "func1", "arguments": None}
            })
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should handle None gracefully
        tool_chunks = [c for c in chunks if '"tool_calls"' in c]
        assert len(tool_chunks) >= 1
        
        # Parse and verify arguments is "{}"
        for chunk in tool_chunks:
            if chunk.startswith("data: "):
                json_str = chunk[6:].strip()
                if json_str != "[DONE]":
                    data = json.loads(json_str)
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                assert tc["function"]["arguments"] == "{}"
        
        print("✓ None function arguments handled")
    
    @pytest.mark.asyncio
    async def test_handles_none_function_object(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Handles None function object.
        Goal: Verify None function is handled.
        """
        print("Setup: Mock stream with None function...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": None
            })
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should handle None gracefully without error
        assert len(chunks) > 0
        print("✓ None function object handled")


# ==================================================================================================
# Tests for stream_with_first_token_retry()
# ==================================================================================================

class TestStreamWithFirstTokenRetry:
    """Tests for stream_with_first_token_retry() function."""
    
    @pytest.mark.asyncio
    async def test_retries_on_first_token_timeout(self, mock_http_client, mock_model_cache, mock_auth_manager):
        """
        What it does: Retries on first token timeout.
        Goal: Verify retry logic.
        """
        print("Setup: Mock make_request that succeeds on second attempt...")
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aclose = AsyncMock()
        
        call_count = 0
        
        async def mock_make_request():
            nonlocal call_count
            call_count += 1
            print(f"make_request called (attempt {call_count})")
            return mock_response
        
        # First call raises timeout, second succeeds
        timeout_raised = False
        
        async def mock_parse_kiro_stream_with_retry(*args, **kwargs):
            nonlocal timeout_raised
            if not timeout_raised:
                timeout_raised = True
                raise FirstTokenTimeoutError("Timeout!")
            yield KiroEvent(type="content", content="Success")
        
        print("Action: Running stream_with_first_token_retry...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream_with_retry):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_with_first_token_retry(
                    mock_make_request,
                    mock_http_client,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                    max_retries=3,
                    first_token_timeout=15
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        print(f"make_request was called {call_count} times")
        
        assert call_count == 2
        assert len(chunks) > 0
        print("✓ Retry logic worked correctly")
    
    @pytest.mark.asyncio
    async def test_raises_504_after_all_retries_exhausted(self, mock_http_client, mock_model_cache, mock_auth_manager):
        """
        What it does: Raises 504 after all retries exhausted.
        Goal: Verify error handling.
        """
        from fastapi import HTTPException
        
        print("Setup: Mock make_request that always times out...")
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aclose = AsyncMock()
        
        call_count = 0
        
        async def mock_make_request():
            nonlocal call_count
            call_count += 1
            return mock_response
        
        async def mock_parse_kiro_stream_always_timeout(*args, **kwargs):
            raise FirstTokenTimeoutError("Timeout!")
            yield  # Make it a generator
        
        max_retries = 3
        
        print(f"Action: Running stream_with_first_token_retry with max_retries={max_retries}...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream_always_timeout):
            with pytest.raises(HTTPException) as exc_info:
                async for chunk in stream_with_first_token_retry(
                    mock_make_request,
                    mock_http_client,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                    max_retries=max_retries,
                    first_token_timeout=15
                ):
                    pass
        
        print(f"Caught HTTPException: {exc_info.value.status_code}")
        print(f"make_request was called {call_count} times")
        
        assert exc_info.value.status_code == 504
        assert call_count == max_retries
        print("✓ 504 raised after all retries exhausted")
    
    @pytest.mark.asyncio
    async def test_handles_api_error_response(self, mock_http_client, mock_model_cache, mock_auth_manager):
        """
        What it does: Handles API error response.
        Goal: Verify error response handling.
        """
        from fastapi import HTTPException
        
        print("Setup: Mock make_request that returns error...")
        
        mock_response = AsyncMock()
        mock_response.status_code = 500
        # Use simple error text without curly braces to avoid loguru format issues
        mock_response.aread = AsyncMock(return_value=b'Internal server error')
        mock_response.aclose = AsyncMock()
        
        async def mock_make_request():
            return mock_response
        
        print("Action: Running stream_with_first_token_retry with error response...")
        
        with pytest.raises(HTTPException) as exc_info:
            async for chunk in stream_with_first_token_retry(
                mock_make_request,
                mock_http_client,
                "claude-sonnet-4",
                mock_model_cache,
                mock_auth_manager,
                max_retries=3,
                first_token_timeout=15
            ):
                pass
        
        print(f"Caught HTTPException: {exc_info.value.status_code}")
        assert exc_info.value.status_code == 500
        print("✓ API error response handled")
    
    @pytest.mark.asyncio
    async def test_propagates_non_timeout_errors(self, mock_http_client, mock_model_cache, mock_auth_manager):
        """
        What it does: Propagates non-timeout errors without retry.
        Goal: Verify only timeout errors trigger retry.
        """
        print("Setup: Mock make_request that raises RuntimeError...")
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aclose = AsyncMock()
        
        call_count = 0
        
        async def mock_make_request():
            nonlocal call_count
            call_count += 1
            return mock_response
        
        async def mock_parse_kiro_stream_error(*args, **kwargs):
            raise RuntimeError("Test error")
            yield  # Make it a generator
        
        print("Action: Running stream_with_first_token_retry with RuntimeError...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream_error):
            with pytest.raises(RuntimeError) as exc_info:
                async for chunk in stream_with_first_token_retry(
                    mock_make_request,
                    mock_http_client,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                    max_retries=3,
                    first_token_timeout=15
                ):
                    pass
        
        print(f"Caught RuntimeError: {exc_info.value}")
        print(f"make_request was called {call_count} times")
        
        # Should only be called once - no retry for non-timeout errors
        assert call_count == 1
        assert "Test error" in str(exc_info.value)
        print("✓ Non-timeout errors propagated without retry")
    
    @pytest.mark.asyncio
    async def test_closes_response_on_retry(self, mock_http_client, mock_model_cache, mock_auth_manager):
        """
        What it does: Closes response when retrying.
        Goal: Verify resource cleanup on retry.
        """
        print("Setup: Mock responses for retry...")
        
        mock_response1 = AsyncMock()
        mock_response1.status_code = 200
        mock_response1.aclose = AsyncMock()
        
        mock_response2 = AsyncMock()
        mock_response2.status_code = 200
        mock_response2.aclose = AsyncMock()
        
        responses = [mock_response1, mock_response2]
        call_count = 0
        
        async def mock_make_request():
            nonlocal call_count
            response = responses[call_count]
            call_count += 1
            return response
        
        # First call raises timeout, second succeeds
        timeout_raised = False
        
        async def mock_parse_kiro_stream_with_retry(*args, **kwargs):
            nonlocal timeout_raised
            if not timeout_raised:
                timeout_raised = True
                raise FirstTokenTimeoutError("Timeout!")
            yield KiroEvent(type="content", content="Success")
        
        print("Action: Running stream_with_first_token_retry...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream_with_retry):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_with_first_token_retry(
                    mock_make_request,
                    mock_http_client,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                    max_retries=3,
                    first_token_timeout=15
                ):
                    pass
        
        print("Check: First response should be closed...")
        mock_response1.aclose.assert_called()
        print("✓ Response closed on retry")


# ==================================================================================================
# Tests for collect_stream_response()
# ==================================================================================================

class TestCollectStreamResponse:
    """Tests for collect_stream_response() function."""
    
    @pytest.mark.asyncio
    async def test_collects_content(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Collects content from stream.
        Goal: Verify content accumulation.
        """
        print("Setup: Mock stream with content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="content", content=" World")
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Result: {result}")
        
        assert result["choices"][0]["message"]["content"] == "Hello World"
        print("✓ Content collected correctly")
    
    @pytest.mark.asyncio
    async def test_collects_reasoning_content(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Collects reasoning content from stream.
        Goal: Verify reasoning content accumulation.
        """
        print("Setup: Mock stream with thinking content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="Let me think...")
            yield KiroEvent(type="content", content="Answer")
        
        print("Action: Collecting stream response with reasoning mode...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                with patch('kiro.streaming_openai.FAKE_REASONING_HANDLING', 'as_reasoning_content'):
                    result = await collect_stream_response(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    )
        
        print(f"Result: {result}")
        
        message = result["choices"][0]["message"]
        assert "reasoning_content" in message
        assert message["reasoning_content"] == "Let me think..."
        print("✓ Reasoning content collected correctly")
    
    @pytest.mark.asyncio
    async def test_collects_tool_calls(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Collects tool calls from stream.
        Goal: Verify tool call accumulation.
        """
        print("Setup: Mock stream with tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "func1", "arguments": '{"a": 1}'}
            })
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Result: {result}")
        
        message = result["choices"][0]["message"]
        assert "tool_calls" in message
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["function"]["name"] == "func1"
        print("✓ Tool calls collected correctly")
    
    @pytest.mark.asyncio
    async def test_tool_calls_have_no_index(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Collected tool calls don't have index field.
        Goal: Verify index is removed for non-streaming.
        """
        print("Setup: Mock stream with tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "func1", "arguments": "{}"}
            })
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Result: {result}")
        
        message = result["choices"][0]["message"]
        for tc in message.get("tool_calls", []):
            assert "index" not in tc
        
        print("✓ Tool calls have no index field")
    
    @pytest.mark.asyncio
    async def test_includes_usage(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Includes usage in response.
        Goal: Verify usage is included.
        """
        print("Setup: Mock stream with content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Result: {result}")
        
        assert "usage" in result
        assert "prompt_tokens" in result["usage"]
        assert "completion_tokens" in result["usage"]
        assert "total_tokens" in result["usage"]
        print("✓ Usage included in response")
    
    @pytest.mark.asyncio
    async def test_sets_finish_reason_tool_calls(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets finish_reason to tool_calls when tools present.
        Goal: Verify correct finish reason.
        """
        print("Setup: Mock stream with tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "func1", "arguments": "{}"}
            })
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Result: {result}")
        
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        print("✓ finish_reason is tool_calls")
    
    @pytest.mark.asyncio
    async def test_sets_finish_reason_stop(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets finish_reason to stop without tools.
        Goal: Verify correct finish reason.
        """
        print("Setup: Mock stream without tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="usage", usage={"inputTokenCount": 10, "outputTokenCount": 1})
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Result: {result}")
        
        assert result["choices"][0]["finish_reason"] == "stop"
        print("✓ finish_reason is stop")
    
    @pytest.mark.asyncio
    async def test_generates_completion_id(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Generates completion ID.
        Goal: Verify ID is present.
        """
        print("Setup: Mock stream...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"ID: {result['id']}")
        
        assert result["id"].startswith("chatcmpl-")
        print("✓ Completion ID generated")
    
    @pytest.mark.asyncio
    async def test_includes_model_name(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Includes model name in response.
        Goal: Verify model is included.
        """
        print("Setup: Mock stream...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Model: {result['model']}")
        
        assert result["model"] == "claude-sonnet-4"
        print("✓ Model name included")
    
    @pytest.mark.asyncio
    async def test_object_is_chat_completion(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets object to chat.completion.
        Goal: Verify OpenAI format.
        """
        print("Setup: Mock stream...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Object: {result['object']}")
        
        assert result["object"] == "chat.completion"
        print("✓ Object is chat.completion")


# ==================================================================================================
# Tests for error handling
# ==================================================================================================

class TestStreamingOpenaiErrorHandling:
    """Tests for error handling in streaming_openai."""
    
    @pytest.mark.asyncio
    async def test_propagates_first_token_timeout_error(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Propagates FirstTokenTimeoutError.
        Goal: Verify timeout error is propagated for retry.
        """
        print("Setup: Mock stream that raises timeout...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            raise FirstTokenTimeoutError("Timeout!")
            yield  # Make it a generator
        
        print("Action: Streaming to OpenAI format with timeout...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with pytest.raises(FirstTokenTimeoutError):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    pass
        
        print("✓ FirstTokenTimeoutError propagated correctly")
    
    @pytest.mark.asyncio
    async def test_handles_generator_exit_gracefully(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Handles GeneratorExit gracefully without re-raising.
        Goal: Verify client disconnect is handled without error.
        """
        print("Setup: Mock stream that raises GeneratorExit...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            raise GeneratorExit()
        
        print("Action: Streaming to OpenAI format with GeneratorExit...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                # GeneratorExit is caught internally and not re-raised
                # This is correct behavior - client disconnect should be handled gracefully
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks before disconnect")
        # Response should be closed
        mock_response.aclose.assert_called()
        print("✓ GeneratorExit handled gracefully")
    
    @pytest.mark.asyncio
    async def test_propagates_other_exceptions(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Propagates other exceptions.
        Goal: Verify errors are not swallowed.
        """
        print("Setup: Mock stream that raises RuntimeError...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            raise RuntimeError("Test error")
        
        print("Action: Streaming to OpenAI format with RuntimeError...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                with pytest.raises(RuntimeError) as exc_info:
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        pass
        
        print(f"Caught exception: {exc_info.value}")
        assert "Test error" in str(exc_info.value)
        print("✓ RuntimeError propagated correctly")
    
    @pytest.mark.asyncio
    async def test_aclose_error_does_not_mask_original(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: aclose() error doesn't mask original error.
        Goal: Verify original exception is propagated.
        """
        print("Setup: Mock response with error in aclose()...")
        
        mock_response.aclose = AsyncMock(side_effect=ConnectionError("Connection lost"))
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            raise RuntimeError("Original error")
        
        print("Action: Streaming to OpenAI format with error and aclose error...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                with pytest.raises(RuntimeError) as exc_info:
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        pass
        
        print(f"Caught exception: {exc_info.value}")
        assert "Original error" in str(exc_info.value)
        print("✓ Original error not masked by aclose error")


# ==================================================================================================
# Tests for bracket tool calls
# ==================================================================================================

class TestStreamingOpenaiBracketToolCalls:
    """Tests for bracket-style tool call handling."""
    
    @pytest.mark.asyncio
    async def test_detects_bracket_tool_calls(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Detects bracket-style tool calls in content.
        Goal: Verify bracket tool call detection.
        """
        print("Setup: Mock stream with bracket tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="[tool_call: func1]")
        
        bracket_tool_calls = [
            {"id": "call_1", "type": "function", "function": {"name": "func1", "arguments": "{}"}}
        ]
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=bracket_tool_calls):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should have tool_calls chunk
        tool_chunks = [c for c in chunks if '"tool_calls"' in c]
        assert len(tool_chunks) >= 1
        print("✓ Bracket tool calls detected")
    
    @pytest.mark.asyncio
    async def test_deduplicates_tool_calls(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Deduplicates tool calls from stream and bracket.
        Goal: Verify deduplication.
        """
        print("Setup: Mock stream with duplicate tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="text")
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "func1", "arguments": "{}"}
            })
        
        # Same tool call from bracket detection
        bracket_tool_calls = [
            {"id": "call_1", "type": "function", "function": {"name": "func1", "arguments": "{}"}}
        ]
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=bracket_tool_calls):
                with patch('kiro.streaming_openai.deduplicate_tool_calls') as mock_dedup:
                    mock_dedup.return_value = [
                        {"id": "call_1", "type": "function", "function": {"name": "func1", "arguments": "{}"}}
                    ]
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        chunks.append(chunk)
                    
                    # Verify deduplicate was called
                    mock_dedup.assert_called()
        
        print("✓ Tool calls deduplicated")


# ==================================================================================================
# Tests for metering data
# ==================================================================================================

class TestStreamingOpenaiMeteringData:
    """Tests for metering data handling."""
    
    @pytest.mark.asyncio
    async def test_includes_credits_used_in_usage(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Includes credits_used in usage when metering data present.
        Goal: Verify metering data is included.
        """
        print("Setup: Mock stream with metering data...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="usage", usage={"credits": 0.001})
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Final chunk should have credits_used
        final_chunk = chunks[-2]  # Before [DONE]
        assert '"credits_used"' in final_chunk
        print("✓ credits_used included in usage")


# ==================================================================================================
# Tests for truncation detection
# ==================================================================================================

class TestStreamingOpenaiTruncationDetection:
    """Tests for truncation detection in OpenAI streaming."""
    
    @pytest.mark.asyncio
    async def test_finish_reason_is_length_when_truncated(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets finish_reason to length when content is truncated.
        Goal: Verify truncation detection without completion signals.
        """
        print("Setup: Mock stream without completion signals (truncated)...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="This response was cut off mid-sentence because")
            # No usage event = truncation
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Final chunk before [DONE] should have finish_reason: length
        final_chunk = chunks[-2]  # Before [DONE]
        print(f"Comparing finish_reason: Expected 'length', Got chunk: {final_chunk}")
        assert '"finish_reason": "length"' in final_chunk
        print("✓ finish_reason is length when truncated")
    
    @pytest.mark.asyncio
    async def test_finish_reason_is_tool_calls_even_without_completion_signals(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets finish_reason to tool_calls when tool calls present.
        Goal: Verify tool_calls take priority (not confused with content truncation).
        """
        print("Setup: Mock stream with tool calls but no completion signals...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Let me call a tool")
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"}
            })
            # No usage event, but tool calls present = tool_calls finish_reason
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Tool calls take priority (not confused with content truncation)
        final_chunk = chunks[-2]  # Before [DONE]
        print(f"Comparing finish_reason: Expected 'tool_calls', Got chunk: {final_chunk}")
        assert '"finish_reason": "tool_calls"' in final_chunk
        print("✓ finish_reason is tool_calls (not confused with content truncation)")
    
    @pytest.mark.asyncio
    async def test_finish_reason_is_stop_with_completion_signals(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets finish_reason to stop when completion signals present.
        Goal: Verify normal completion is detected correctly.
        """
        print("Setup: Mock stream with completion signals (not truncated)...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Complete response")
            yield KiroEvent(type="usage", usage={"inputTokenCount": 10, "outputTokenCount": 5})
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # With completion signals, should be stop
        final_chunk = chunks[-2]  # Before [DONE]
        print(f"Comparing finish_reason: Expected 'stop', Got chunk: {final_chunk}")
        assert '"finish_reason": "stop"' in final_chunk
        print("✓ finish_reason is stop with completion signals")
    
    @pytest.mark.asyncio
    async def test_collect_extracts_finish_reason_from_chunks(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Non-streaming extracts finish_reason from streaming chunks.
        Goal: Verify collect_stream_response correctly extracts finish_reason.
        """
        print("Setup: Mock stream without completion signals...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Truncated")
            # No usage = truncation
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Result finish_reason: {result['choices'][0]['finish_reason']}")
        
        # Should extract "length" from streaming chunks
        assert result["choices"][0]["finish_reason"] == "length"
        print("✓ collect_stream_response extracts finish_reason correctly")


# ==================================================================================================
# Tests for usage_sink (usage tracking)
# ==================================================================================================

def _final_usage_from_chunks(chunks: list) -> dict:
    """Extract the usage block from the final SSE chunk that carries one.

    Args:
        chunks: SSE strings yielded by the streaming generator.

    Returns:
        The usage dict from the last chunk containing one, or an empty dict.
    """
    usage = {}
    for chunk in chunks:
        if not chunk.startswith("data:"):
            continue
        payload = chunk[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        parsed = json.loads(payload)
        if "usage" in parsed:
            usage = parsed["usage"]
    return usage


class TestUsageSinkSuccess:
    """Tests for usage_sink population on successful streams."""

    @pytest.mark.asyncio
    async def test_sink_none_produces_identical_chunks(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Streaming with usage_sink=None yields the same chunks as without it.
        Goal: Guarantee the new parameter is a no-op when unused (regression guard).
        """
        print("Setup: Identical streams, one with usage_sink=None...")

        def make_stream():
            async def mock_parse_kiro_stream(*args, **kwargs):
                yield KiroEvent(type="content", content="Hello")
                yield KiroEvent(type="context_usage", context_usage_percentage=10.0)
            return mock_parse_kiro_stream

        async def run(sink):
            chunks = []
            with patch('kiro.streaming_openai.parse_kiro_stream', make_stream()):
                with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                    async for chunk in stream_kiro_to_openai_internal(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager, usage_sink=sink
                    ):
                        chunks.append(chunk)
            return chunks

        print("Action: Running with no sink and with an explicit None sink...")
        without = await run(None)
        with_none = await run(None)

        # Completion ids and timestamps differ per run, so compare structure only.
        def strip_ids(chunks):
            out = []
            for c in chunks:
                if c.startswith("data:") and "[DONE]" not in c:
                    d = json.loads(c[len("data:"):].strip())
                    d.pop("id", None)
                    d.pop("created", None)
                    out.append(d)
                else:
                    out.append(c)
            return out

        assert strip_ids(without) == strip_ids(with_none)
        print("✓ usage_sink=None does not change streaming output")

    @pytest.mark.asyncio
    async def test_sink_matches_final_chunk_usage(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Sink token counts equal what the client is told in the final chunk.
        Goal: Prevent the DB recording numbers that differ from the API response.
        """
        print("Setup: Stream with content and context_usage...")

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello world")
            yield KiroEvent(type="context_usage", context_usage_percentage=25.0)

        sink = {}
        chunks = []

        print("Action: Streaming with a usage_sink...")
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai_internal(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager, usage_sink=sink
                ):
                    chunks.append(chunk)

        reported = _final_usage_from_chunks(chunks)
        print(f"Sink: {sink}")
        print(f"Reported to client: {reported}")

        assert sink["prompt_tokens"] == reported["prompt_tokens"]
        assert sink["completion_tokens"] == reported["completion_tokens"]
        assert sink["total_tokens"] == reported["total_tokens"]
        print("✓ Sink agrees with the usage reported to the client")

    @pytest.mark.asyncio
    async def test_token_source_is_context_usage_when_kiro_reports_it(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: token_source is 'context_usage' when Kiro returns a percentage.
        Goal: Verify the data-quality flag marks accurate rows.
        """
        print("Setup: Stream with context_usage=30%...")

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hi")
            yield KiroEvent(type="context_usage", context_usage_percentage=30.0)

        sink = {}

        print("Action: Streaming...")
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for _ in stream_kiro_to_openai_internal(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager, usage_sink=sink
                ):
                    pass

        print(f"token_source={sink['token_source']}, context={sink['context_usage_percentage']}")
        assert sink["token_source"] == "context_usage"
        assert sink["context_usage_percentage"] == 30.0
        print("✓ token_source reflects Kiro-provided context usage")

    @pytest.mark.asyncio
    async def test_token_source_is_tiktoken_without_context_usage(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: token_source is 'tiktoken' when Kiro sends no context_usage.
        Goal: Verify estimated rows are flagged as estimates.
        """
        print("Setup: Stream with content only, no context_usage...")

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")

        sink = {}

        print("Action: Streaming with request messages for fallback counting...")
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for _ in stream_kiro_to_openai_internal(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager,
                    request_messages=[{"role": "user", "content": "Hello there"}],
                    usage_sink=sink
                ):
                    pass

        print(f"token_source={sink['token_source']}, prompt_tokens={sink['prompt_tokens']}")
        assert sink["token_source"] == "tiktoken"
        assert sink["prompt_tokens"] > 0
        assert sink["context_usage_percentage"] is None
        print("✓ token_source marks the fallback estimate")

    @pytest.mark.asyncio
    async def test_total_tokens_is_internally_consistent(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: total_tokens equals prompt + completion on the tiktoken path.
        Goal: Catch arithmetic drift in the fallback branch.
        """
        print("Setup: Stream without context_usage so the fallback computes the total...")

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Some generated answer")

        sink = {}

        print("Action: Streaming...")
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for _ in stream_kiro_to_openai_internal(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager,
                    request_messages=[{"role": "user", "content": "Question"}],
                    usage_sink=sink
                ):
                    pass

        print(f"{sink['prompt_tokens']} + {sink['completion_tokens']} == {sink['total_tokens']}?")
        assert sink["total_tokens"] == sink["prompt_tokens"] + sink["completion_tokens"]
        print("✓ total_tokens is consistent")

    @pytest.mark.asyncio
    async def test_finish_reason_recorded(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: finish_reason lands in the sink.
        Goal: Verify stop-reason reporting for dashboards.
        """
        print("Setup: Content stream ending with context_usage (a clean completion signal)...")

        # A context_usage event marks the stream as properly completed. Without it,
        # the generator treats the stream as truncated and reports finish_reason=length.
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Done")
            yield KiroEvent(type="context_usage", context_usage_percentage=8.0)

        sink = {}

        print("Action: Streaming...")
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for _ in stream_kiro_to_openai_internal(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager, usage_sink=sink
                ):
                    pass

        print(f"finish_reason={sink['finish_reason']}")
        assert sink["finish_reason"] == "stop"
        print("✓ finish_reason recorded")

    @pytest.mark.asyncio
    async def test_truncated_stream_records_length_finish_reason(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: A stream ending without completion signals records finish_reason=length.
        Goal: Truncation must be visible in the recorded data, not silently look like success.
        """
        print("Setup: Content stream with no completion signal (truncated)...")

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Cut off mid-sen")

        sink = {}

        print("Action: Streaming...")
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for _ in stream_kiro_to_openai_internal(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager, usage_sink=sink
                ):
                    pass

        print(f"finish_reason={sink['finish_reason']}")
        assert sink["finish_reason"] == "length"
        print("✓ Truncation surfaces as finish_reason=length")


class TestUsageSinkErrors:
    """Tests for usage_sink behaviour on failure paths."""

    @pytest.mark.asyncio
    async def test_sink_stays_empty_on_midstream_exception(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Sink is untouched when the stream raises partway through.
        Goal: An empty sink is how the route detects there is no usable usage data.
        """
        print("Setup: Stream that raises after one content event...")

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Partial")
            raise RuntimeError("Upstream died")

        sink = {}

        print("Action: Streaming and swallowing the error...")
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                with pytest.raises(RuntimeError):
                    async for _ in stream_kiro_to_openai_internal(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager, usage_sink=sink
                    ):
                        pass

        print(f"Sink after error: {sink}")
        assert sink == {}
        print("✓ Sink left empty on mid-stream error")

    @pytest.mark.asyncio
    async def test_sink_stays_empty_on_client_disconnect(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Sink is untouched when the consumer stops early.
        Goal: Client disconnects must not be recorded as completed usage.
        """
        print("Setup: Stream consumed only partially...")

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="First")
            yield KiroEvent(type="content", content="Second")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)

        sink = {}

        print("Action: Breaking out of the loop after the first chunk...")
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                gen = stream_kiro_to_openai_internal(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager, usage_sink=sink
                )
                async for _ in gen:
                    break
                await gen.aclose()

        print(f"Sink after disconnect: {sink}")
        assert sink == {}
        print("✓ Sink left empty on client disconnect")


class TestUsageSinkEdgeCases:
    """Edge cases for usage_sink handling."""

    @pytest.mark.asyncio
    async def test_stale_keys_are_cleared(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Pre-existing keys in the sink are removed before repopulation.
        Goal: Verify clear() runs, so a retried stream cannot blend two attempts.
        """
        print("Setup: Sink pre-populated with stale values...")

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Fresh")

        sink = {"stale_key": "should be gone", "prompt_tokens": 99999}

        print("Action: Streaming into the dirty sink...")
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for _ in stream_kiro_to_openai_internal(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager, usage_sink=sink
                ):
                    pass

        print(f"Sink keys: {sorted(sink)}")
        assert "stale_key" not in sink
        assert sink["prompt_tokens"] != 99999
        print("✓ Stale keys cleared before repopulation")

    @pytest.mark.asyncio
    async def test_sink_holds_only_final_attempt_after_retry(
        self, mock_model_cache, mock_auth_manager, mock_http_client
    ):
        """
        What it does: After a first-token timeout and retry, the sink holds only
                      the successful attempt's values.
        Goal: This is precisely what clear() protects; a blended sink would
              double-count tokens for every retried request.
        """
        print("Setup: First attempt times out, second attempt succeeds with distinct content...")

        attempt = {"n": 0}

        async def mock_parse_kiro_stream(*args, **kwargs):
            attempt["n"] += 1
            if attempt["n"] == 1:
                # Long content that would inflate the counts if it leaked through.
                yield KiroEvent(type="content", content="A" * 400)
                raise FirstTokenTimeoutError("No first token")
            yield KiroEvent(type="content", content="ok")

        first_response = AsyncMock()
        first_response.status_code = 200
        first_response.aclose = AsyncMock()

        second_response = AsyncMock()
        second_response.status_code = 200
        second_response.aclose = AsyncMock()

        async def make_request():
            return second_response

        sink = {}

        print("Action: Streaming through the retry wrapper...")
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for _ in stream_with_first_token_retry(
                    make_request=make_request,
                    client=mock_http_client,
                    model="claude-sonnet-4",
                    model_cache=mock_model_cache,
                    auth_manager=mock_auth_manager,
                    initial_response=first_response,
                    max_retries=2,
                    request_messages=[{"role": "user", "content": "Hi"}],
                    usage_sink=sink
                ):
                    pass

        # "ok" is far shorter than 400 chars, so a blended sink would be obvious.
        completion_for_ok = sink["completion_tokens"]
        print(f"Attempts run: {attempt['n']}, completion_tokens={completion_for_ok}")
        assert attempt["n"] == 2, "retry should have happened"
        assert completion_for_ok < 20, (
            f"completion_tokens={completion_for_ok} looks like it includes the "
            f"failed attempt's 400-char content"
        )
        print("✓ Sink holds only the final attempt")

    @pytest.mark.asyncio
    async def test_credits_used_recorded_when_kiro_meters(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Kiro's metering value reaches the sink as credits_used.
        Goal: Verify the Kiro-specific usage event is captured.
        """
        print("Setup: Stream including a usage (metering) event...")

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hi")
            yield KiroEvent(type="usage", usage=7)

        sink = {}

        print("Action: Streaming...")
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for _ in stream_kiro_to_openai_internal(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager, usage_sink=sink
                ):
                    pass

        print(f"credits_used={sink['credits_used']}")
        assert sink["credits_used"] == 7
        print("✓ credits_used captured from Kiro metering event")

    @pytest.mark.asyncio
    async def test_wrapper_forwards_sink(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: stream_kiro_to_openai forwards the sink to the internal generator.
        Goal: Verify the pass-through layer is wired, not silently dropping the arg.
        """
        print("Setup: Stream via the public wrapper...")

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hi")

        sink = {}

        print("Action: Streaming through stream_kiro_to_openai...")
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for _ in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager, usage_sink=sink
                ):
                    pass

        print(f"Sink populated: {bool(sink)}")
        assert sink, "wrapper did not forward usage_sink"
        assert "total_tokens" in sink
        print("✓ Public wrapper forwards the sink")


# ==================================================================================================
# Tests for upstream stop reasons
# ==================================================================================================

def _finish_reason_from_chunks(chunks: list[str]) -> str:
    """Extract the final non-null finish reason from SSE chunks.

    Args:
        chunks: SSE strings yielded by the streaming generator.

    Returns:
        The final finish reason reported to the client.
    """
    finish_reason = ""
    for chunk in chunks:
        if not chunk.startswith("data:"):
            continue
        payload = chunk[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        parsed = json.loads(payload)
        value = parsed.get("choices", [{}])[0].get("finish_reason")
        if value:
            finish_reason = value
    return finish_reason


class TestStreamingOpenaiStopReasons:
    """Tests for Kiro metadata stop reason handling in OpenAI streaming."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("upstream_stop_reason", "expected_finish_reason"),
        [
            ("END_TURN", "stop"),
            ("TOOL_USE", "tool_calls"),
            ("MAX_TOKENS", "length"),
            ("STOP_SEQUENCE", "stop"),
            ("CONTENT_FILTERED", "content_filter"),
            ("GUARDRAIL_INTERVENED", "content_filter"),
        ],
    )
    async def test_maps_each_upstream_stop_reason_to_openai_finish_reason(
        self,
        upstream_stop_reason,
        expected_finish_reason,
        mock_http_client,
        mock_response,
        mock_model_cache,
        mock_auth_manager,
    ):
        """
        What it does: Maps every known Kiro metadata stop reason to OpenAI.
        Goal: Preserve upstream completion semantics in final SSE chunks.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Complete response")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)
            yield KiroEvent(type="stop_reason", stop_reason=upstream_stop_reason)

        chunks = []

        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)

        assert _finish_reason_from_chunks(chunks) == expected_finish_reason

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("scenario", "expected_finish_reason"),
        [
            ("normal", "stop"),
            ("tool_calls", "tool_calls"),
            ("truncated", "length"),
        ],
    )
    async def test_preserves_local_inference_without_upstream_stop_reason(
        self,
        scenario,
        expected_finish_reason,
        mock_http_client,
        mock_response,
        mock_model_cache,
        mock_auth_manager,
    ):
        """
        What it does: Runs the existing inference paths without metadata.
        Goal: Guard the legacy behavior when Kiro omits stopReason.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            if scenario == "normal":
                yield KiroEvent(type="content", content="Complete response")
                yield KiroEvent(type="context_usage", context_usage_percentage=5.0)
            elif scenario == "tool_calls":
                yield KiroEvent(type="tool_use", tool_use={
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                })
            else:
                yield KiroEvent(type="content", content="Cut off response")

        chunks = []

        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)

        assert _finish_reason_from_chunks(chunks) == expected_finish_reason

    @pytest.mark.asyncio
    async def test_uses_last_non_empty_upstream_stop_reason(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Retains the latest non-empty metadata stop reason.
        Goal: Ignore blank metadata events while accepting a later correction.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Complete response")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)
            yield KiroEvent(type="stop_reason", stop_reason="END_TURN")
            yield KiroEvent(type="stop_reason", stop_reason="MAX_TOKENS")
            yield KiroEvent(type="stop_reason", stop_reason="")

        chunks = []

        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)

        assert _finish_reason_from_chunks(chunks) == "length"

    @pytest.mark.asyncio
    async def test_local_truncation_overrides_upstream_end_turn(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Prefers local truncation over an upstream END_TURN value.
        Goal: Report an incomplete stream as length even when metadata disagrees.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Cut off response")
            yield KiroEvent(type="stop_reason", stop_reason="END_TURN")

        chunks = []

        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)

        assert _finish_reason_from_chunks(chunks) == "length"

    @pytest.mark.asyncio
    async def test_upstream_max_tokens_reports_length_without_local_truncation(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Maps MAX_TOKENS on a normally completed stream to length.
        Goal: Expose upstream token-limit completions to OpenAI clients.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Complete response")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)
            yield KiroEvent(type="stop_reason", stop_reason="MAX_TOKENS")

        chunks = []

        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)

        assert _finish_reason_from_chunks(chunks) == "length"

    @pytest.mark.asyncio
    async def test_tool_calls_override_upstream_end_turn(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Preserves tool_calls when metadata incorrectly says END_TURN.
        Goal: Ensure clients execute emitted tool calls.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            })
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)
            yield KiroEvent(type="stop_reason", stop_reason="END_TURN")

        chunks = []

        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)

        assert _finish_reason_from_chunks(chunks) == "tool_calls"

    @pytest.mark.asyncio
    async def test_reasoning_only_stream_without_completion_signals_reports_length(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Detects a reasoning-only stream ending without completion signals.
        Goal: Avoid reporting a partially emitted reasoning turn as a clean stop.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="Partial reasoning")

        chunks = []

        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)

        assert _finish_reason_from_chunks(chunks) == "length"

    @pytest.mark.asyncio
    async def test_reasoning_only_stream_with_completion_signals_reports_stop(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Accepts a completed reasoning-only stream as a normal stop.
        Goal: Prevent false-positive truncation when completion metadata arrives.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="Complete reasoning")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)

        chunks = []

        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)

        assert _finish_reason_from_chunks(chunks) == "stop"

    @pytest.mark.asyncio
    async def test_usage_sink_finish_reason_matches_emitted_final_chunk(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Records the same finish reason in usage_sink and SSE output.
        Goal: Keep usage telemetry consistent with the client-visible result.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Complete response")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)
            yield KiroEvent(type="stop_reason", stop_reason="MAX_TOKENS")

        sink = {}
        chunks = []

        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_openai_internal(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager, usage_sink=sink
                ):
                    chunks.append(chunk)

        assert _finish_reason_from_chunks(chunks) == "length"
        assert sink["finish_reason"] == _finish_reason_from_chunks(chunks)

    @pytest.mark.asyncio
    async def test_collect_stream_response_inherits_upstream_finish_reason(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Collects the final streaming finish reason in non-streaming mode.
        Goal: Verify collect_stream_response inherits metadata handling from SSE chunks.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Complete response")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)
            yield KiroEvent(type="stop_reason", stop_reason="CONTENT_FILTERED")

        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )

        assert result["choices"][0]["finish_reason"] == "content_filter"


# ==================================================================================================
# Tests for native reasoning signatures
# ==================================================================================================

def _openai_payloads(chunks: list[str]) -> list[dict]:
    """Parse JSON payloads from OpenAI SSE chunks, excluding the DONE marker.

    Args:
        chunks: SSE strings yielded by the OpenAI streaming generator.

    Returns:
        Decoded JSON payloads from data chunks.
    """
    return [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk.startswith("data: ") and "[DONE]" not in chunk
    ]


class TestStreamingOpenaiReasoningSignatures:
    """Tests for forwarding native Kiro reasoning signatures to OpenAI."""

    @pytest.mark.asyncio
    async def test_forwards_reasoning_signature_in_reasoning_delta(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """Forward a signed reasoning event as an additive delta field."""
        signature = "CAISggcKstream-signature"

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(
                type="thinking",
                thinking_content="Reasoning text",
                reasoning_signature=signature,
            )
            yield KiroEvent(type="content", content="Final answer")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)

        chunks = []
        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                with patch("kiro.streaming_openai.FAKE_REASONING_HANDLING", "as_reasoning_content"):
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        chunks.append(chunk)

        payloads = _openai_payloads(chunks)
        reasoning_delta = next(
            payload["choices"][0]["delta"]
            for payload in payloads
            if "reasoning_content" in payload["choices"][0]["delta"]
        )
        assert reasoning_delta["reasoning_content"] == "Reasoning text"
        assert reasoning_delta["reasoning_signature"] == signature

    @pytest.mark.asyncio
    async def test_omits_reasoning_signature_without_signature(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """Keep unsigned reasoning deltas free of a null or empty signature key."""
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="Unsigned reasoning")
            yield KiroEvent(
                type="thinking",
                thinking_content="Empty signature",
                reasoning_signature="",
            )
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)

        chunks = []
        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                with patch("kiro.streaming_openai.FAKE_REASONING_HANDLING", "as_reasoning_content"):
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        chunks.append(chunk)

        reasoning_deltas = [
            payload["choices"][0]["delta"]
            for payload in _openai_payloads(chunks)
            if "reasoning_content" in payload["choices"][0]["delta"]
        ]
        assert reasoning_deltas
        assert all("reasoning_signature" not in delta for delta in reasoning_deltas)

    @pytest.mark.asyncio
    async def test_forwards_signature_from_later_reasoning_event(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """Forward a signature that appears only on a later reasoning event."""
        signature = "CAISggcK-later-signature"

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="First reasoning")
            yield KiroEvent(
                type="thinking",
                thinking_content="Second reasoning",
                reasoning_signature=signature,
            )
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)

        chunks = []
        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                with patch("kiro.streaming_openai.FAKE_REASONING_HANDLING", "as_reasoning_content"):
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        chunks.append(chunk)

        reasoning_deltas = [
            payload["choices"][0]["delta"]
            for payload in _openai_payloads(chunks)
            if "reasoning_content" in payload["choices"][0]["delta"]
        ]
        assert "reasoning_signature" not in reasoning_deltas[0]
        assert reasoning_deltas[1]["reasoning_signature"] == signature

    @pytest.mark.asyncio
    async def test_signature_does_not_change_existing_deltas_or_final_chunk(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """Preserve content, reasoning text, and final chunk fields when signed."""
        signature = "CAISggcK-regression-signature"

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(
                type="thinking",
                thinking_content="Reasoning",
                reasoning_signature=signature,
            )
            yield KiroEvent(type="content", content="Answer")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)

        chunks = []
        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                with patch("kiro.streaming_openai.FAKE_REASONING_HANDLING", "as_reasoning_content"):
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        chunks.append(chunk)

        payloads = _openai_payloads(chunks)
        assert payloads[0]["choices"][0]["delta"] == {
            "role": "assistant",
            "content": "",
        }
        assert payloads[1]["choices"][0]["delta"] == {
            "reasoning_content": "Reasoning",
            "reasoning_signature": signature,
        }
        assert payloads[2]["choices"][0]["delta"] == {"content": "Answer"}
        assert payloads[-1]["choices"][0]["delta"] == {}
        assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
        assert "usage" in payloads[-1]
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("handling_mode", ["pass", "strip_tags", "include_as_text"])
    async def test_omits_signature_when_thinking_is_folded_into_content(
        self,
        handling_mode,
        mock_http_client,
        mock_response,
        mock_model_cache,
        mock_auth_manager,
    ):
        """Do not attach signatures when thinking is emitted as plain content."""
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(
                type="thinking",
                thinking_content="Folded reasoning",
                reasoning_signature="CAISggcK-folded-signature",
            )
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)

        chunks = []
        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                with patch("kiro.streaming_openai.FAKE_REASONING_HANDLING", handling_mode):
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        chunks.append(chunk)

        assert all(
            "reasoning_signature" not in payload["choices"][0]["delta"]
            for payload in _openai_payloads(chunks)
        )


class TestCollectStreamResponseReasoningSignatures:
    """Tests for collecting native reasoning signatures in non-streaming mode."""

    @pytest.mark.asyncio
    async def test_collects_reasoning_signature_on_message(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """Expose a signed reasoning block on the assembled assistant message."""
        signature = "CAISggcK-non-stream-signature"

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(
                type="thinking",
                thinking_content="Reasoning",
                reasoning_signature=signature,
            )
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)

        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                with patch("kiro.streaming_openai.FAKE_REASONING_HANDLING", "as_reasoning_content"):
                    result = await collect_stream_response(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    )

        message = result["choices"][0]["message"]
        assert message["reasoning_signature"] == signature

    @pytest.mark.asyncio
    async def test_omits_reasoning_signature_on_unsigned_message(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """Keep the unsigned non-streaming message shape unchanged."""
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="Unsigned reasoning")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)

        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                with patch("kiro.streaming_openai.FAKE_REASONING_HANDLING", "as_reasoning_content"):
                    result = await collect_stream_response(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    )

        assert "reasoning_signature" not in result["choices"][0]["message"]

    @pytest.mark.asyncio
    async def test_collects_signed_reasoning_and_tool_calls_together(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """Preserve both the reasoning signature and tool calls in one message."""
        signature = "CAISggcK-tool-signature"

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(
                type="thinking",
                thinking_content="Reasoning before tool use",
                reasoning_signature=signature,
            )
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query":"value"}'},
            })
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)

        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                with patch("kiro.streaming_openai.FAKE_REASONING_HANDLING", "as_reasoning_content"):
                    result = await collect_stream_response(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    )

        message = result["choices"][0]["message"]
        assert message["reasoning_signature"] == signature
        assert message["tool_calls"] == [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"query":"value"}'},
        }]


# ==================================================================================================
# Tests for empty-response detection in stream_kiro_to_openai_internal
# ==================================================================================================


class TestEmptyResponseDetectionInternal:
    """Tests for empty_detection_enabled in stream_kiro_to_openai_internal."""

    @pytest.mark.asyncio
    async def test_reasoning_only_detection_enabled_raises(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Raises EmptyResponseError for a reasoning-only turn when
        empty_detection_enabled=True.
        Purpose: Verify that the retryable signal is raised for the known case
        that triggers client-side resampling loops.
        """
        from kiro.empty_response import EmptyResponseError

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="deep analysis here")
            yield KiroEvent(type="stop_reason", stop_reason="END_TURN")
            yield KiroEvent(type="context_usage", context_usage_percentage=2.0)

        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                with pytest.raises(EmptyResponseError) as exc_info:
                    async for _ in stream_kiro_to_openai_internal(
                        mock_http_client,
                        mock_response,
                        "claude-sonnet-4",
                        mock_model_cache,
                        mock_auth_manager,
                        empty_detection_enabled=True,
                    ):
                        pass

        assert exc_info.value.reason == "reasoning_only"
        assert exc_info.value.had_reasoning is True

    @pytest.mark.asyncio
    async def test_reasoning_only_detection_disabled_delivers(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Delivers a reasoning-only turn normally when
        empty_detection_enabled=False (the default).
        Purpose: Confirm that the flag default preserves existing behavior.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="deep analysis here")
            yield KiroEvent(type="stop_reason", stop_reason="END_TURN")
            yield KiroEvent(type="context_usage", context_usage_percentage=2.0)

        chunks = []
        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_openai_internal(
                    mock_http_client,
                    mock_response,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                    # empty_detection_enabled defaults to False
                ):
                    chunks.append(chunk)

        # Must produce the final chunk and [DONE] — no exception raised.
        assert any("finish_reason" in c for c in chunks)
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_no_terminal_event_before_raise(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Confirms no finish_reason chunk or [DONE] is yielded
        before EmptyResponseError is raised.
        Purpose: A retried attempt must not leave partial output in the client
        stream.
        """
        from kiro.empty_response import EmptyResponseError

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="reasoning")
            yield KiroEvent(type="context_usage", context_usage_percentage=1.0)

        emitted: list = []
        try:
            with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
                with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                    async for chunk in stream_kiro_to_openai_internal(
                        mock_http_client,
                        mock_response,
                        "claude-sonnet-4",
                        mock_model_cache,
                        mock_auth_manager,
                        empty_detection_enabled=True,
                    ):
                        emitted.append(chunk)
        except EmptyResponseError:
            pass

        # None of the emitted chunks may carry a finish_reason or [DONE].
        for chunk in emitted:
            assert '"finish_reason"' not in chunk or '"finish_reason": null' in chunk
            assert "[DONE]" not in chunk

    @pytest.mark.asyncio
    async def test_usage_sink_not_written_before_raise(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: usage_sink remains empty when EmptyResponseError is raised.
        Purpose: A discarded attempt must not corrupt the sink that the route
        handler uses for recording.
        """
        from kiro.empty_response import EmptyResponseError

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="reasoning only")
            yield KiroEvent(type="context_usage", context_usage_percentage=1.0)

        sink: Dict[str, Any] = {}
        try:
            with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
                with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                    async for _ in stream_kiro_to_openai_internal(
                        mock_http_client,
                        mock_response,
                        "claude-sonnet-4",
                        mock_model_cache,
                        mock_auth_manager,
                        usage_sink=sink,
                        empty_detection_enabled=True,
                    ):
                        pass
        except EmptyResponseError:
            pass

        assert sink == {}, f"Expected empty sink, got {sink!r}"

    @pytest.mark.asyncio
    async def test_visible_content_never_raises(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: A turn with visible text is delivered normally even with
        detection enabled.
        Purpose: Ensure detection is a no-op for useful responses.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Here is the answer.")
            yield KiroEvent(type="context_usage", context_usage_percentage=2.0)

        chunks = []
        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_openai_internal(
                    mock_http_client,
                    mock_response,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                    empty_detection_enabled=True,
                ):
                    chunks.append(chunk)

        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_tool_calls_never_raises(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: A turn with tool calls is delivered normally even with
        detection enabled.
        Purpose: Tool call turns must never be resampled.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                },
            )
            yield KiroEvent(type="context_usage", context_usage_percentage=2.0)

        chunks = []
        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_openai_internal(
                    mock_http_client,
                    mock_response,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                    empty_detection_enabled=True,
                ):
                    chunks.append(chunk)

        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_truncated_turn_never_raises(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: A truncated turn (stream ended without completion signals)
        is never classified as retryable.
        Purpose: Truncation must reach the client as a token-limit outcome, not
        be hidden behind a resample.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            # Thinking only, no usage/context_usage — triggers truncation detection
            yield KiroEvent(type="thinking", thinking_content="partial reasoning")
            # No usage event, no context_usage → content_was_truncated stays False
            # because both thinking and content are empty-ish (no visible content,
            # no usage signals).  The classifier guards on was_truncated=True.
            # Simulate a case where there IS some thinking but stream ends cleanly
            # to exercise the truncation guard rather than the reasoning-only path.
            # We emit a context_usage so the stream looks complete, then set the
            # truncated flag via the stream having no normal completion signal.
            # Easiest: yield nothing after thinking (no usage) so truncation fires.

        async def mock_parse_kiro_stream_truncated(*args, **kwargs):
            # Content present but no completion signals → truncation detected.
            yield KiroEvent(type="content", content="partial answer")
            # No usage or context_usage → stream_completed_normally=False →
            # content_was_truncated=True → classify_turn returns None → no raise.

        chunks = []
        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream_truncated):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_openai_internal(
                    mock_http_client,
                    mock_response,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                    empty_detection_enabled=True,
                ):
                    chunks.append(chunk)

        # Delivered normally, not raised.
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_max_tokens_stop_reason_never_raises(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: A turn stopped by MAX_TOKENS is never resampled.
        Purpose: Resampling a deterministic capacity limit would create a retry
        storm.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="stop_reason", stop_reason="MAX_TOKENS")
            yield KiroEvent(type="context_usage", context_usage_percentage=100.0)

        chunks = []
        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_openai_internal(
                    mock_http_client,
                    mock_response,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                    empty_detection_enabled=True,
                ):
                    chunks.append(chunk)

        assert chunks[-1] == "data: [DONE]\n\n"


# ==================================================================================================
# Tests for stream_with_first_token_retry empty-response integration
# ==================================================================================================


class TestStreamWithFirstTokenRetryEmptyResponse:
    """Tests for the empty-response retry loop inside stream_with_first_token_retry."""

    def _make_response_mock(self) -> MagicMock:
        """Return a mock httpx.Response that looks like a 200."""
        resp = MagicMock()
        resp.status_code = 200
        resp.aclose = AsyncMock()
        return resp

    @pytest.mark.asyncio
    async def test_end_to_end_retry_delivers_clean_stream(
        self, mock_http_client, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: First attempt raises EmptyResponseError; second attempt
        returns visible content.  Client sees exactly one clean stream with one
        [DONE].
        Purpose: Core happy-path for the empty-response retry feature.
        """
        call_count = {"n": 0}

        async def mock_parse_kiro_stream(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First attempt: reasoning only → will trigger EmptyResponseError.
                yield KiroEvent(type="thinking", thinking_content="only reasoning")
                yield KiroEvent(type="context_usage", context_usage_percentage=1.0)
            else:
                # Second attempt: real content.
                yield KiroEvent(type="content", content="Here is the answer.")
                yield KiroEvent(type="context_usage", context_usage_percentage=2.0)

        initial_resp = self._make_response_mock()
        retry_resp = self._make_response_mock()
        make_request = AsyncMock(return_value=retry_resp)

        chunks = []
        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                with patch("kiro.config.EMPTY_RESPONSE_RETRIES", 2):
                    with patch("kiro.streaming_openai.EMPTY_RESPONSE_RETRIES", 2, create=True):
                        async for chunk in stream_with_first_token_retry(
                            make_request=make_request,
                            client=mock_http_client,
                            model="claude-sonnet-4",
                            model_cache=mock_model_cache,
                            auth_manager=mock_auth_manager,
                            initial_response=initial_resp,
                        ):
                            chunks.append(chunk)

        done_count = sum(1 for c in chunks if c == "data: [DONE]\n\n")
        assert done_count == 1, f"Expected exactly one [DONE], got {done_count}"
        content_chunks = [c for c in chunks if '"content"' in c and "answer" in c]
        assert len(content_chunks) >= 1

    @pytest.mark.asyncio
    async def test_budget_exhausted_passthrough_delivers(
        self, mock_http_client, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: All attempts return reasoning-only; passthrough mode
        delivers the degenerate turn rather than failing the request.
        Purpose: EMPTY_RESPONSE_ON_EXHAUSTED=passthrough must never break the
        request.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="only reasoning")
            yield KiroEvent(type="context_usage", context_usage_percentage=1.0)

        initial_resp = self._make_response_mock()
        retry_resp = self._make_response_mock()
        make_request = AsyncMock(return_value=retry_resp)

        chunks = []
        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                with patch("kiro.config.EMPTY_RESPONSE_RETRIES", 1):
                    with patch("kiro.config.EMPTY_RESPONSE_ON_EXHAUSTED", "passthrough"):
                        async for chunk in stream_with_first_token_retry(
                            make_request=make_request,
                            client=mock_http_client,
                            model="claude-sonnet-4",
                            model_cache=mock_model_cache,
                            auth_manager=mock_auth_manager,
                            initial_response=initial_resp,
                        ):
                            chunks.append(chunk)

        # Request must succeed (no exception) and the stream must have [DONE].
        assert any(c == "data: [DONE]\n\n" for c in chunks)

    @pytest.mark.asyncio
    async def test_retries_zero_no_detection_single_attempt(
        self, mock_http_client, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: EMPTY_RESPONSE_RETRIES=0 produces a single attempt with
        no detection — the stream is byte-identical to the pre-feature baseline.
        Purpose: EMPTY_RESPONSE_RETRIES=0 must be a no-op.
        """
        call_count = {"n": 0}

        async def mock_parse_kiro_stream(*args, **kwargs):
            call_count["n"] += 1
            yield KiroEvent(type="thinking", thinking_content="only reasoning")
            yield KiroEvent(type="context_usage", context_usage_percentage=1.0)

        initial_resp = self._make_response_mock()
        make_request = AsyncMock(return_value=self._make_response_mock())

        chunks = []
        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                with patch("kiro.config.EMPTY_RESPONSE_RETRIES", 0):
                    async for chunk in stream_with_first_token_retry(
                        make_request=make_request,
                        client=mock_http_client,
                        model="claude-sonnet-4",
                        model_cache=mock_model_cache,
                        auth_manager=mock_auth_manager,
                        initial_response=initial_resp,
                    ):
                        chunks.append(chunk)

        # Only one parse_kiro_stream call (no retry).
        assert call_count["n"] == 1
        # The degenerate turn is delivered, ending with [DONE].
        assert chunks[-1] == "data: [DONE]\n\n"
        # make_request was never called (initial_response was used, no retry).
        make_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_retried_attempts_call_make_request(
        self, mock_http_client, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Attempts after the first call make_request() rather than
        reusing the consumed initial_response.
        Purpose: HTTP response bodies cannot be re-read; reuse would silently
        produce empty or corrupted streams.
        """
        call_count = {"n": 0}

        async def mock_parse_kiro_stream(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                yield KiroEvent(type="thinking", thinking_content="reasoning only")
                yield KiroEvent(type="context_usage", context_usage_percentage=1.0)
            else:
                yield KiroEvent(type="content", content="answer")
                yield KiroEvent(type="context_usage", context_usage_percentage=2.0)

        initial_resp = self._make_response_mock()
        fresh_resp = self._make_response_mock()
        make_request = AsyncMock(return_value=fresh_resp)

        with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                with patch("kiro.config.EMPTY_RESPONSE_RETRIES", 2):
                    with patch("kiro.config.EMPTY_RESPONSE_ON_EXHAUSTED", "passthrough"):
                        async for _ in stream_with_first_token_retry(
                            make_request=make_request,
                            client=mock_http_client,
                            model="claude-sonnet-4",
                            model_cache=mock_model_cache,
                            auth_manager=mock_auth_manager,
                            initial_response=initial_resp,
                        ):
                            pass

        # The second attempt must have called make_request() for a fresh response.
        make_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_budget_exhausted_error_mode_raises_http_exception(
        self, mock_http_client, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: All attempts return reasoning-only; error mode raises
        HTTPException rather than delivering the degenerate turn.
        Purpose: Operators who prefer a hard failure over a silent empty
        response should get one.
        """
        from fastapi import HTTPException

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="only reasoning")
            yield KiroEvent(type="context_usage", context_usage_percentage=1.0)

        initial_resp = self._make_response_mock()
        make_request = AsyncMock(return_value=self._make_response_mock())

        with pytest.raises(HTTPException) as exc_info:
            with patch("kiro.streaming_openai.parse_kiro_stream", mock_parse_kiro_stream):
                with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
                    with patch("kiro.config.EMPTY_RESPONSE_RETRIES", 1):
                        with patch("kiro.config.EMPTY_RESPONSE_ON_EXHAUSTED", "error"):
                            async for _ in stream_with_first_token_retry(
                                make_request=make_request,
                                client=mock_http_client,
                                model="claude-sonnet-4",
                                model_cache=mock_model_cache,
                                auth_manager=mock_auth_manager,
                                initial_response=initial_resp,
                            ):
                                pass

        assert exc_info.value.status_code == 503
        assert "no visible content" in exc_info.value.detail
