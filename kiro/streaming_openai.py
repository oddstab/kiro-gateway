# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/oddstab/kiro-gateway
# Copyright (C) 2025 oddstab
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Streaming logic for converting Kiro stream to OpenAI format.

Contains generators for:
- Converting AWS SSE to OpenAI SSE
- Forming streaming chunks
- Processing tool calls in stream

Uses streaming_core.py for parsing Kiro stream into unified KiroEvent objects.
"""

import json
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Awaitable, Dict, Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro.parsers import parse_bracket_tool_calls, deduplicate_tool_calls
from kiro.stop_reasons import resolve_finish_reason
from kiro.utils import generate_completion_id
from kiro.config import (
    FIRST_TOKEN_TIMEOUT,
    FIRST_TOKEN_MAX_RETRIES,
    FAKE_REASONING_HANDLING,
    WEB_SEARCH_ENABLED,
)
from kiro.mcp_tools import client_provides_web_search
from kiro.tokenizer import count_tokens, count_message_tokens, count_tools_tokens

# Import from streaming_core - reuse shared parsing logic
from kiro.streaming_core import (
    parse_kiro_stream,
    FirstTokenTimeoutError,
    KiroEvent,
    calculate_tokens_from_context_usage,
    normalize_token_source,
    stream_with_first_token_retry as stream_with_first_token_retry_core,
)

if TYPE_CHECKING:
    from kiro.auth import KiroAuthManager
    from kiro.cache import ModelInfoCache

# Import debug_logger for logging
try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


# Re-export FirstTokenTimeoutError for backward compatibility
__all__ = ['FirstTokenTimeoutError', 'stream_kiro_to_openai', 'stream_with_first_token_retry', 'collect_stream_response']


async def stream_kiro_to_openai_internal(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
    conversation_id: Optional[str] = None,
    usage_sink: Optional[Dict[str, Any]] = None,
    *,
    empty_detection_enabled: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Internal generator for converting Kiro stream to OpenAI format.

    Parses AWS SSE stream and converts events to OpenAI chat.completion.chunk.
    Supports tool calls and usage calculation.

    IMPORTANT: This function raises FirstTokenTimeoutError if first token
    is not received within first_token_timeout seconds.

    When ``empty_detection_enabled`` is True, the function classifies the
    completed turn before emitting any terminal event (tool-calls chunk,
    final chunk, or ``[DONE]``). A reasoning-only or no-visible-content turn
    raises :class:`kiro.empty_response.EmptyResponseError` so that the outer
    :func:`stream_with_empty_response_retry` wrapper can resample. The raise
    guarantees that no terminal SSE event has been sent to the client and
    ``usage_sink`` has not been written, so a fresh attempt starts cleanly.

    Args:
        client: HTTP client (for connection management)
        response: HTTP response with data stream
        model: Model name to include in response
        model_cache: Model cache for getting token limits
        auth_manager: Authentication manager
        first_token_timeout: First token wait timeout (seconds)
        request_messages: Original request messages (for fallback token counting)
        request_tools: Original request tools (for fallback token counting)
        conversation_id: Stable conversation ID for truncation recovery (optional)
        usage_sink: Optional mutable dict filled with final token counts so the
            caller (route handler) can record usage after the stream completes.
            Left untouched on error paths and client disconnects, so an empty
            sink signals that no usable usage data was produced.
        empty_detection_enabled: When True, raise EmptyResponseError for a
            completed turn with no client-visible content or tool calls instead
            of delivering it. Must be False on the final attempt so degenerate
            turns are always delivered rather than permanently failing.

    Yields:
        Strings in SSE format: "data: {...}\\n\\n" or "data: [DONE]\\n\\n".
        In ``as_reasoning_content`` mode, thinking deltas may also contain
        ``reasoning_signature`` when Kiro provides one.

    Raises:
        FirstTokenTimeoutError: If first token not received within timeout
        EmptyResponseError: If the turn has no visible content and
            ``empty_detection_enabled`` is True.

    Example:
        >>> async for chunk in stream_kiro_to_openai_internal(client, response, "claude-sonnet-4", cache, auth):
        ...     print(chunk)
        data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...}

        data: [DONE]
    """
    completion_id = generate_completion_id()
    created_time = int(time.time())
    first_chunk = True
    
    metering_data = None
    context_usage_percentage = None
    # Upstream stop reason from Kiro metadataEvent (last non-empty value wins)
    upstream_stop_reason: Optional[str] = None
    full_content = ""
    full_thinking_content = ""  # Accumulated thinking content for non-streaming
    last_reasoning_signature: Optional[str] = None
    
    streaming_error_occurred = False
    tool_calls_from_stream = []
    
    try:
        # Use streaming_core.parse_kiro_stream for unified event parsing
        # This handles AWS SSE parsing, first token timeout, and thinking parser
        async for event in parse_kiro_stream(response, first_token_timeout):
            if event.type == "content" and event.content:
                # Accumulate content for bracket tool call detection
                full_content += event.content
                
                # Format as OpenAI chunk
                delta = {"content": event.content}
                if first_chunk:
                    delta["role"] = "assistant"
                    first_chunk = False
                
                openai_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                }
                
                chunk_text = f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"
                
                if debug_logger:
                    debug_logger.log_modified_chunk(chunk_text.encode('utf-8'))
                
                yield chunk_text
            
            elif event.type == "thinking" and event.thinking_content:
                # Accumulate thinking content
                full_thinking_content += event.thinking_content

                # Emit opening chunk before first reasoning delta so clients
                # that expect a clean role+content initialization see it.
                if first_chunk:
                    opening_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(opening_chunk, ensure_ascii=False)}\n\n"
                    first_chunk = False

                # Send as reasoning_content or content based on mode
                if FAKE_REASONING_HANDLING == "as_reasoning_content":
                    delta = {"reasoning_content": event.thinking_content}
                    if event.reasoning_signature:
                        if last_reasoning_signature is None:
                            logger.debug("Captured reasoning signature from Kiro stream")
                        last_reasoning_signature = event.reasoning_signature
                        delta["reasoning_signature"] = last_reasoning_signature
                else:
                    # Folded thinking is plain content, so it has no structured
                    # reasoning channel where a signature can be represented.
                    delta = {"content": event.thinking_content}
                
                openai_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                }
                
                chunk_text = f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"
                
                if debug_logger:
                    debug_logger.log_modified_chunk(chunk_text.encode('utf-8'))
                
                yield chunk_text

            elif event.type == "web_search" and event.web_search:
                # Format search results as text content (OpenAI has no structured search blocks)
                ws_data = event.web_search
                ws_query = ws_data.get("query", "")
                ws_results = ws_data.get("results", [])

                parts = [f"Search results for \"{ws_query}\":\n"]
                for i, r in enumerate(ws_results, 1):
                    title = r.get("title", "")
                    url = r.get("url", "")
                    snippet = r.get("snippet", "")
                    parts.append(f"{i}. [{title}]({url})")
                    if snippet:
                        parts.append(f"   {snippet}")
                    parts.append("")

                search_text = "\n".join(parts)
                full_content += search_text

                delta = {"content": search_text}
                if first_chunk:
                    delta["role"] = "assistant"
                    first_chunk = False

                openai_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                }

                chunk_text = f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"

                if debug_logger:
                    debug_logger.log_modified_chunk(chunk_text.encode('utf-8'))

                yield chunk_text

            elif event.type == "tool_use" and event.tool_use:
                tool = event.tool_use
                
                # Extract tool name safely (handle None/missing fields)
                tool_name = ""
                if tool:
                    tool_name = (tool.get("function") or {}).get("name", "") or tool.get("name", "")
                
                # ==============================================================================
                # WebSearch Support - Path B: MCP Tool Emulation (Streaming Interception)
                # ==============================================================================
                
                # INTERCEPT web_search tool calls (Path B - MCP emulation).
                #
                # Two conditions must hold:
                #  - WEB_SEARCH_ENABLED, and
                #  - the CLIENT did not send its own web_search tool.
                #
                # The second condition is what keeps raw tags out of Grok Build:
                # it executes web_search itself (POST /v1/responses ->
                # kiro/grok_web_search.py). If we intercepted its tool call we
                # would swallow it and answer with `<web_search>` tagged text,
                # which the client renders verbatim in the chat window.
                # Otherwise the call flows back untouched and the client runs it.
                if (
                    WEB_SEARCH_ENABLED
                    and tool_name == "web_search"
                    and not client_provides_web_search(request_tools)
                ):
                    from kiro.mcp_tools import generate_search_summary
                    from kiro.web_search_provider import call_web_search

                    logger.info("Intercepted web_search tool call (Path B - gateway emulation)")
                    
                    # Parse tool_input
                    tool_input = tool.get("function", {}).get("arguments", {}) or tool.get("input", {})
                    if isinstance(tool_input, str):
                        try:
                            tool_input = json.loads(tool_input)
                        except json.JSONDecodeError:
                            tool_input = {}
                    
                    # Extract query
                    query = tool_input.get("query", "")
                    if not query:
                        logger.warning("web_search called without query, skipping provider call")
                        # Continue with normal tool_use processing
                    else:
                        logger.debug(f"WebSearch query (Path B): {query}")

                        mcp_tool_use_id, results = await call_web_search(
                            query, auth_manager
                        )

                        if results is None:
                            logger.error("Configured provider failed for web_search")
                            # Continue with normal tool_use processing (will show error to user)
                        else:
                            # Emit summary as content chunks (OpenAI format)
                            summary = generate_search_summary(query, results)
                            
                            # Send content chunks
                            chunk_size = 100
                            for i in range(0, len(summary), chunk_size):
                                content_chunk = summary[i:i + chunk_size]
                                
                                delta = {"content": content_chunk}
                                if first_chunk:
                                    delta["role"] = "assistant"
                                    first_chunk = False
                                
                                openai_chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": model,
                                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                                }
                                
                                chunk_text = f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"
                                
                                if debug_logger:
                                    debug_logger.log_modified_chunk(chunk_text.encode('utf-8'))
                                
                                yield chunk_text
                            
                            # Accumulate for token counting
                            full_content += summary
                            
                            # Skip normal tool_use processing
                            continue
                
                # Collect tool calls from stream (normal tools, not web_search)
                tool_calls_from_stream.append(event.tool_use)
            
            elif event.type == "usage" and event.usage:
                metering_data = event.usage
            
            elif event.type == "context_usage" and event.context_usage_percentage is not None:
                context_usage_percentage = event.context_usage_percentage

            elif event.type == "stop_reason" and event.stop_reason:
                upstream_stop_reason = event.stop_reason
        
        # Track completion signals for truncation detection
        received_usage = metering_data is not None
        received_context_usage = context_usage_percentage is not None
        stream_completed_normally = received_usage or received_context_usage
        
        # Check bracket-style tool calls in full content
        bracket_tool_calls = parse_bracket_tool_calls(full_content)
        all_tool_calls = tool_calls_from_stream + bracket_tool_calls
        all_tool_calls = deduplicate_tool_calls(all_tool_calls)
        
        # Detect content truncation (missing completion signals)
        content_was_truncated = (
            not stream_completed_normally and
            len(full_content) + len(full_thinking_content) > 0 and
            not all_tool_calls  # Don't confuse with tool call truncation
        )
        
        if content_was_truncated:
            from kiro.config import TRUNCATION_RECOVERY
            logger.error(
                f"Content truncated by Kiro API: stream ended without completion signals, "
                f"length={len(full_content)} chars. "
                f"{'Model will be notified automatically about truncation.' if TRUNCATION_RECOVERY else 'Set TRUNCATION_RECOVERY=true in .env to auto-notify model about truncation.'}"
            )
        
        # Determine finish_reason from local truncation, upstream metadata, and local inference
        finish_reason = resolve_finish_reason(
            upstream_stop_reason,
            truncated_value="length",
            tool_calls_value="tool_calls",
            default_value="stop",
            was_truncated=content_was_truncated,
            has_tool_calls=bool(all_tool_calls),
            api_label="openai",
        )
        
        # Count completion_tokens (output) using tiktoken
        completion_tokens = count_tokens(full_content + full_thinking_content)
        
        # Calculate total_tokens based on context_usage_percentage from Kiro API
        # context_usage shows TOTAL percentage of context usage (input + output)
        prompt_tokens, total_tokens, prompt_source, total_source = calculate_tokens_from_context_usage(
            context_usage_percentage, completion_tokens, model_cache, model
        )
        
        # Fallback: Kiro API didn't return context_usage, use tiktoken
        # Count prompt_tokens from original messages
        # IMPORTANT: Don't apply correction coefficient for prompt_tokens,
        # as it was calibrated for completion_tokens
        if prompt_source == "unknown" and request_messages:
            prompt_tokens = count_message_tokens(request_messages, apply_claude_correction=False)
            if request_tools:
                prompt_tokens += count_tools_tokens(request_tools, apply_claude_correction=False)
            total_tokens = prompt_tokens + completion_tokens
            prompt_source = "tiktoken"
            total_source = "tiktoken"
        
        # Empty-response detection: raise before any terminal event so the outer
        # stream_with_empty_response_retry wrapper can resample cleanly.
        # Truncation-recovery bookkeeping, usage_sink, tool-calls chunk, final
        # chunk, and [DONE] are all intentionally below this block so that a
        # discarded attempt never leaves partial state in the client stream.
        if empty_detection_enabled:
            from kiro.empty_response import classify_turn, EmptyResponseError

            empty_reason = classify_turn(
                content=full_content,
                thinking_content=full_thinking_content,
                tool_calls=all_tool_calls if all_tool_calls else None,
                kiro_stop_reason=upstream_stop_reason,
                was_truncated=content_was_truncated,
            )
            if empty_reason is not None:
                raise EmptyResponseError(
                    reason=empty_reason,
                    had_reasoning=bool(full_thinking_content.strip()),
                    content_len=len(full_content),
                    tool_call_count=len(all_tool_calls),
                    kiro_stop_reason=upstream_stop_reason,
                    completion_tokens=completion_tokens,
                    model=model,
                )

        # Send tool calls if present
        if all_tool_calls:
            logger.debug(f"Processing {len(all_tool_calls)} tool calls for streaming response")
            
            # Add required index field to each tool_call
            # according to OpenAI API specification for streaming
            indexed_tool_calls = []
            for idx, tc in enumerate(all_tool_calls):
                # Extract function with None protection
                func = tc.get("function") or {}
                # Use "or" for protection against explicit None in values
                tool_name = func.get("name") or ""
                tool_args = func.get("arguments") or "{}"
                
                logger.debug(f"Tool call [{idx}] '{tool_name}': id={tc.get('id')}, args_length={len(tool_args)}")
                
                indexed_tc = {
                    "index": idx,
                    "id": tc.get("id"),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args
                    }
                }
                indexed_tool_calls.append(indexed_tc)
            
            tool_calls_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": indexed_tool_calls},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(tool_calls_chunk, ensure_ascii=False)}\n\n"
        
        # Save truncation info for recovery (tracked by stable identifiers)
        from kiro.truncation_recovery import should_inject_recovery
        from kiro.truncation_state import save_tool_truncation, save_content_truncation
        
        if should_inject_recovery():
            # Save tool truncations (tracked by tool_call_id)
            truncated_count = 0
            for tc in all_tool_calls:
                if tc.get('_truncation_detected'):
                    save_tool_truncation(
                        tool_call_id=tc['id'],
                        tool_name=tc['function']['name'],
                        truncation_info=tc['_truncation_info']
                    )
                    truncated_count += 1
            
            # Save content truncation (tracked by content hash)
            if content_was_truncated:
                save_content_truncation(full_content)
            
            if truncated_count > 0 or content_was_truncated:
                logger.info(
                    f"Truncation detected: {truncated_count} tool(s), "
                    f"content={content_was_truncated}. Will be handled when client sends next request."
                )
        
        # Final chunk with usage
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        }
        
        if metering_data:
            final_chunk["usage"]["credits_used"] = metering_data
        
        # Expose final token counts to the caller for usage recording.
        # clear() before update() is required, not stylistic: stream_with_first_token_retry
        # may re-run this generator after a first token timeout, and the sink must end up
        # holding only the final attempt's values instead of a blend of both attempts.
        if usage_sink is not None:
            usage_sink.clear()
            usage_sink.update({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "context_usage_percentage": context_usage_percentage,
                "credits_used": metering_data,
                "token_source": normalize_token_source(total_source),
                "finish_reason": finish_reason,
            })

        # Log final token values being sent to client
        logger.debug(
            f"[Usage] {model}: "
            f"prompt_tokens={prompt_tokens} ({prompt_source}), "
            f"completion_tokens={completion_tokens} (tiktoken), "
            f"total_tokens={total_tokens} ({total_source})"
        )
        
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        
    except FirstTokenTimeoutError:
        streaming_error_occurred = True
        # Propagate timeout up for retry
        raise
    except GeneratorExit:
        # Client disconnected - this is normal, don't log as error
        logger.debug("Client disconnected (GeneratorExit)")
        streaming_error_occurred = True
    except Exception as e:
        streaming_error_occurred = True
        # Log exception type and message for better diagnostics
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else "(empty message)"
        logger.error(
            f"Error during streaming: [{error_type}] {error_msg}",
            exc_info=True
        )
        # Propagate error up for proper handling in routes_openai.py
        raise
    finally:
        # Always close response
        try:
            await response.aclose()
        except Exception as close_error:
            logger.debug(f"Error closing response: {close_error}")
        
        if streaming_error_occurred:
            logger.debug("Streaming completed with error")
        else:
            logger.debug("Streaming completed successfully")


async def stream_kiro_to_openai(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
    usage_sink: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[str, None]:
    """
    Generator for converting Kiro stream to OpenAI format.
    
    This is a wrapper over stream_kiro_to_openai_internal that does NOT retry.
    Retry logic is implemented in stream_with_first_token_retry.
    
    Args:
        client: HTTP client (for connection management)
        response: HTTP response with data stream
        model: Model name to include in response
        model_cache: Model cache for getting token limits
        auth_manager: Authentication manager
        request_messages: Original request messages (for fallback token counting)
        request_tools: Original request tools (for fallback token counting)
        usage_sink: Optional mutable dict filled with final token counts
            (see stream_kiro_to_openai_internal)
    
    Yields:
        Strings in SSE format. In ``as_reasoning_content`` mode, thinking
        deltas may include ``reasoning_signature`` when Kiro provides one.
    """
    async for chunk in stream_kiro_to_openai_internal(
        client, response, model, model_cache, auth_manager,
        request_messages=request_messages,
        request_tools=request_tools,
        usage_sink=usage_sink
    ):
        yield chunk


async def stream_with_first_token_retry(
    make_request: Callable[[], Awaitable[httpx.Response]],
    client: httpx.AsyncClient,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    initial_response: Optional[httpx.Response] = None,
    max_retries: int = FIRST_TOKEN_MAX_RETRIES,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
    usage_sink: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """
    Streaming with automatic retry on first token timeout and empty responses.

    Wraps two independent retry budgets:

    - **Outer** (:func:`~kiro.streaming_core.stream_with_empty_response_retry`):
      resamples completed turns that had no client-visible content (e.g.
      reasoning-only turns).  Budget is ``EMPTY_RESPONSE_RETRIES + 1`` total
      attempts.  Zero disables detection and leaves behavior identical to the
      pre-feature baseline.

    - **Inner** (:func:`~kiro.streaming_core.stream_with_first_token_retry_core`):
      retries when the model is slow to emit its first token.  Each
      empty-response attempt gets its own fresh first-token-timeout budget,
      because an empty completion and a dead connection are different failures.

    The ``initial_response`` may only be consumed once (an HTTP response body
    cannot be re-read).  Attempts after the first call ``make_request()`` for a
    fresh response.

    ``EMPTY_RESPONSE_ON_EXHAUSTED`` controls what happens when all
    empty-response attempts are used up:

    - ``"passthrough"`` (default): the final attempt delivers the degenerate
      turn as-is; the request always succeeds.
    - ``"error"``: raises :class:`fastapi.HTTPException` with a message
      explaining the situation and suggesting a retry or model change.

    Args:
        make_request: Factory that opens a new streaming HTTP request.
        client: HTTP client used by the internal generator.
        model: Model name to include in response chunks.
        model_cache: Model cache for token-limit lookups.
        auth_manager: Authentication manager.
        initial_response: Optional pre-validated 200 response for the first
            attempt.  Consumed exactly once; later attempts call
            ``make_request()``.
        max_retries: Maximum first-token-timeout attempts per empty-response
            attempt.
        first_token_timeout: Seconds to wait for the first token.
        request_messages: Original messages for fallback token counting.
        request_tools: Original tools for fallback token counting.
        usage_sink: Optional mutable dict filled with final token counts.
            On retry the sink is fully overwritten so it reflects only the
            attempt that produced the delivered response.

    Yields:
        Strings in SSE format.  In ``as_reasoning_content`` mode, thinking
        deltas may include ``reasoning_signature`` when Kiro provides one.

    Raises:
        HTTPException: After exhausting first-token-timeout retries, or when
            ``EMPTY_RESPONSE_ON_EXHAUSTED`` is ``"error"`` and all
            empty-response attempts are used up.

    Example:
        >>> async def make_req():
        ...     return await http_client.request_with_retry("POST", url, payload, stream=True)
        >>> response = await make_req()
        >>> async for chunk in stream_with_first_token_retry(
        ...     make_req, client, model, cache, auth, initial_response=response
        ... ):
        ...     print(chunk)
    """
    from kiro.config import EMPTY_RESPONSE_RETRIES, EMPTY_RESPONSE_ON_EXHAUSTED
    from kiro.empty_response import EmptyResponseError
    from kiro.streaming_core import stream_with_empty_response_retry

    # EMPTY_RESPONSE_RETRIES=0 means "feature off": single attempt, no detection.
    # Otherwise the outer wrapper runs EMPTY_RESPONSE_RETRIES+1 total attempts.
    empty_max_attempts: int = 1 if EMPTY_RESPONSE_RETRIES == 0 else EMPTY_RESPONSE_RETRIES + 1
    on_error_mode: bool = EMPTY_RESPONSE_ON_EXHAUSTED == "error"

    def create_http_error(status_code: int, error_text: str) -> HTTPException:
        """Create HTTPException for HTTP errors."""
        return HTTPException(
            status_code=status_code,
            detail=f"Upstream API error: {error_text}",
        )

    def create_timeout_error(retries: int, timeout: float) -> HTTPException:
        """Create HTTPException for timeout errors."""
        return HTTPException(
            status_code=504,
            detail=f"Model did not respond within {timeout}s after {retries} attempts. Please try again.",
        )

    # Track how many times the outer runner has been called so we only pass
    # initial_response on the very first call.  An HTTP response body cannot be
    # re-read, so later calls must open a fresh connection via make_request().
    _call_count: Dict[str, int] = {"n": 0}

    async def attempt_runner(
        empty_attempt: int,
        is_final_attempt: bool,
    ) -> AsyncGenerator[str, None]:
        """
        Run one empty-response attempt with its own first-token-timeout budget.

        ``empty_detection_enabled`` is True unless this is the final attempt in
        passthrough mode, in which case the degenerate turn must be delivered
        rather than raising.  In error mode detection stays on for the final
        attempt so the EmptyResponseError propagates and can be converted to an
        HTTPException by the caller.

        Args:
            empty_attempt: One-based attempt number from the outer wrapper.
            is_final_attempt: True when this is the last allowed attempt.

        Yields:
            SSE strings.

        Raises:
            EmptyResponseError: When detection is enabled and the turn is empty.
            HTTPException: On transport/timeout failure.
        """
        _call_count["n"] += 1
        # Pass initial_response only on the very first call.
        resp_for_attempt: Optional[httpx.Response] = (
            initial_response if _call_count["n"] == 1 else None
        )

        # For passthrough mode, disarm detection on the final attempt so the
        # degenerate turn is delivered rather than failing the request.
        # For error mode, keep detection on all attempts so EmptyResponseError
        # propagates out of stream_with_empty_response_retry.
        detection_enabled: bool = not is_final_attempt or on_error_mode

        async def stream_processor(resp: httpx.Response) -> AsyncGenerator[str, None]:
            """Adapt a raw HTTP response into SSE strings."""
            async for chunk in stream_kiro_to_openai_internal(
                client,
                resp,
                model,
                model_cache,
                auth_manager,
                first_token_timeout=first_token_timeout,
                request_messages=request_messages,
                request_tools=request_tools,
                usage_sink=usage_sink,
                empty_detection_enabled=detection_enabled,
            ):
                yield chunk

        async for chunk in stream_with_first_token_retry_core(
            make_request=make_request,
            stream_processor=stream_processor,
            initial_response=resp_for_attempt,
            max_retries=max_retries,
            first_token_timeout=first_token_timeout,
            on_http_error=create_http_error,
            on_all_retries_failed=create_timeout_error,
        ):
            yield chunk

    if empty_max_attempts <= 1:
        # Feature is disabled: single attempt with detection always off.
        async for chunk in attempt_runner(1, True):
            yield chunk
        return

    try:
        async for chunk in stream_with_empty_response_retry(
            attempt_runner=attempt_runner,
            max_attempts=empty_max_attempts,
        ):
            yield chunk
    except EmptyResponseError as exc:
        # Only reachable in error mode (detection stays on for the final attempt).
        logger.error(
            "Empty response budget exhausted "
            f"(reason={exc.reason!r}, model={exc.model!r}, "
            f"attempts={empty_max_attempts}); returning error to client."
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"The model returned no visible content after {empty_max_attempts} attempt(s) "
                f"(reason: {exc.reason}). "
                "Please retry your request or try a different model."
            ),
        )



async def collect_stream_response(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None
) -> dict:
    """
    Collect full response from streaming stream.
    
    Used for non-streaming mode - collects all chunks
    and forms a single response.
    
    Args:
        client: HTTP client
        response: HTTP response with stream
        model: Model name
        model_cache: Model cache
        auth_manager: Authentication manager
        request_messages: Original request messages (for fallback token counting)
        request_tools: Original request tools (for fallback token counting)
    
    Returns:
        Dictionary with full response in OpenAI chat.completion format. A
        non-empty ``reasoning_signature`` is included on the assistant message
        when signed reasoning was present in the stream.
    """
    full_content = ""
    full_reasoning_content = ""
    last_reasoning_signature: Optional[str] = None
    final_usage = None
    tool_calls = []
    finish_reason = "stop"  # Default fallback
    completion_id = generate_completion_id()
    
    async for chunk_str in stream_kiro_to_openai(
        client,
        response,
        model,
        model_cache,
        auth_manager,
        request_messages=request_messages,
        request_tools=request_tools
    ):
        if not chunk_str.startswith("data:"):
            continue
        
        data_str = chunk_str[len("data:"):].strip()
        if not data_str or data_str == "[DONE]":
            continue
        
        try:
            chunk_data = json.loads(data_str)
            
            # Extract data from chunk
            delta = chunk_data.get("choices", [{}])[0].get("delta", {})
            if "content" in delta:
                full_content += delta["content"]
            if "reasoning_content" in delta:
                full_reasoning_content += delta["reasoning_content"]
            reasoning_signature = delta.get("reasoning_signature")
            if reasoning_signature:
                if last_reasoning_signature is None:
                    logger.debug("Captured reasoning signature while collecting stream")
                last_reasoning_signature = reasoning_signature
            if "tool_calls" in delta:
                tool_calls.extend(delta["tool_calls"])
            
            # Extract finish_reason from chunk (streaming already calculated it correctly)
            finish_reason_from_chunk = chunk_data.get("choices", [{}])[0].get("finish_reason")
            if finish_reason_from_chunk:
                finish_reason = finish_reason_from_chunk
            
            # Save usage from last chunk
            if "usage" in chunk_data:
                final_usage = chunk_data["usage"]
                
        except (json.JSONDecodeError, IndexError):
            continue
    
    # Form final response
    message = {"role": "assistant", "content": full_content}
    if full_reasoning_content:
        message["reasoning_content"] = full_reasoning_content
    if last_reasoning_signature:
        message["reasoning_signature"] = last_reasoning_signature
    if tool_calls:
        # For non-streaming response remove index field from tool_calls,
        # as it's only required for streaming chunks
        cleaned_tool_calls = []
        for tc in tool_calls:
            # Extract function with None protection
            func = tc.get("function") or {}
            cleaned_tc = {
                "id": tc.get("id"),
                "type": tc.get("type", "function"),
                "function": {
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}")
                }
            }
            cleaned_tool_calls.append(cleaned_tc)
        message["tool_calls"] = cleaned_tool_calls
    
    # Form usage for response
    usage = final_usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    # Log token info for debugging (non-streaming uses same logs from streaming)
    
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": usage
    }