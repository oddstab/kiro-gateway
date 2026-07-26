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
Converters for transforming Anthropic Messages API format to Kiro format.

This module is an adapter layer that converts Anthropic-specific formats
to the unified format used by converters_core.py.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from kiro.config import HIDDEN_MODELS, MODEL_ALIASES
from kiro.model_resolver import get_model_id_for_kiro
from kiro.models_anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessage,
    AnthropicTool,
)
from kiro.converters_core import (
    UnifiedMessage,
    UnifiedTool,
    ThinkingConfig,
    get_native_reasoning_format,
    normalize_native_reasoning_effort,
    build_kiro_payload,
    extract_text_content,
    extract_images_from_content,
)


def convert_anthropic_content_to_text(content: Any) -> str:
    """
    Extracts text content from Anthropic message content.

    Anthropic content can be:
    - String: "Hello, world!"
    - List of content blocks: [{"type": "text", "text": "Hello"}]
    - Server-side tool blocks (server_tool_use, web_search_tool_result)

    Args:
        content: Anthropic message content

    Returns:
        Extracted text content
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "web_search_tool_result":
                    # 提取搜尋結果摘要文字
                    text_parts.append(_extract_web_search_result_text(block))
                elif block_type == "server_tool_use":
                    # server_tool_use 本身沒有有用的文字，跳過
                    pass
            elif hasattr(block, "type"):
                if block.type == "text":
                    text_parts.append(block.text)
        return "".join(text_parts)

    return str(content) if content else ""


def extract_reasoning_from_anthropic_content(content: Any) -> tuple[Optional[str], Optional[str]]:
    """Extract thinking text and signature from assistant content blocks."""
    if not isinstance(content, list):
        return None, None

    parts: List[str] = []
    signature: Optional[str] = None
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "thinking":
                parts.append(str(block.get("thinking", "")))
                signature = block.get("signature") or signature
        elif getattr(block, "type", None) == "thinking":
            parts.append(str(getattr(block, "thinking", "")))
            signature = getattr(block, "signature", None) or signature

    text = "".join(parts)
    return (text or None), signature


def _extract_web_search_result_text(block: dict) -> str:
    """
    Extracts readable text from a web_search_tool_result block.

    The block contains a list of web_search_result items with title/url/content.
    We format them as a readable summary for the model.

    Args:
        block: web_search_tool_result content block

    Returns:
        Formatted text summary of search results
    """
    results = block.get("content", [])
    if not isinstance(results, list):
        return ""

    parts = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "web_search_result":
            title = item.get("title", "")
            url = item.get("url", "")
            snippet = item.get("encrypted_content", "") or item.get("content", "")
            if title or snippet:
                entry = f"[{title}]({url})" if url else title
                if snippet:
                    entry += f"\n{snippet}"
                parts.append(entry)

    if not parts:
        return ""
    return "\n\n".join(parts)


def extract_system_prompt(system: Any) -> str:
    """
    Extracts system prompt text from Anthropic system field.

    Anthropic API supports system in two formats:
    1. String: "You are helpful"
    2. List of content blocks: [{"type": "text", "text": "...", "cache_control": {...}}]

    The second format is used for prompt caching with cache_control.
    We extract only the text, ignoring cache_control (not supported by Kiro).

    Args:
        system: System prompt in string or list format

    Returns:
        Extracted system prompt as string
    """
    if system is None:
        return ""

    if isinstance(system, str):
        return system

    if isinstance(system, list):
        text_parts = []
        for block in system:
            if isinstance(block, dict):
                # Handle {"type": "text", "text": "...", "cache_control": {...}}
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif hasattr(block, "type") and block.type == "text":
                # Handle Pydantic model
                text_parts.append(getattr(block, "text", ""))
        return "\n".join(text_parts)

    return str(system)


def extract_tool_results_from_anthropic_content(content: Any) -> List[Dict[str, Any]]:
    """
    Extracts tool results from Anthropic message content.

    Looks for content blocks with type="tool_result".

    Args:
        content: Anthropic message content (list of content blocks)

    Returns:
        List of tool results in unified format
    """
    tool_results = []

    if not isinstance(content, list):
        return tool_results

    for block in content:
        block_type = None
        tool_use_id = None
        result_content = ""

        if isinstance(block, dict):
            block_type = block.get("type")
            tool_use_id = block.get("tool_use_id")
            result_content = block.get("content", "")
        elif hasattr(block, "type"):
            block_type = block.type
            tool_use_id = getattr(block, "tool_use_id", None)
            result_content = getattr(block, "content", "")

        if block_type == "tool_result" and tool_use_id:
            # Convert content to text if it's a list
            if isinstance(result_content, list):
                result_content = extract_text_content(result_content)
            elif not isinstance(result_content, str):
                result_content = str(result_content) if result_content else ""

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_content or "(empty result)",
                }
            )

    return tool_results


def extract_images_from_tool_results(content: Any) -> List[Dict[str, Any]]:
    """
    Extracts images from tool_result content blocks.

    Tool results in Anthropic format can contain images (e.g., screenshots from browser tools).
    This function extracts those images so they can be passed to the model.

    Args:
        content: Anthropic message content (list of content blocks)

    Returns:
        List of images in unified format: [{"media_type": "image/jpeg", "data": "base64..."}]
    """
    images: List[Dict[str, Any]] = []

    if not isinstance(content, list):
        return images

    for block in content:
        block_type = None
        result_content = None

        if isinstance(block, dict):
            block_type = block.get("type")
            result_content = block.get("content")
        elif hasattr(block, "type"):
            block_type = block.type
            result_content = getattr(block, "content", None)

        if block_type == "tool_result" and isinstance(result_content, list):
            # Extract images from the tool_result's content
            tool_result_images = extract_images_from_content(result_content)
            images.extend(tool_result_images)

    if images:
        logger.debug(f"Extracted {len(images)} image(s) from tool_result content")

    return images


def extract_tool_uses_from_anthropic_content(content: Any) -> List[Dict[str, Any]]:
    """
    Extracts tool uses from Anthropic assistant message content.

    Looks for content blocks with type="tool_use".

    Args:
        content: Anthropic message content (list of content blocks)

    Returns:
        List of tool calls in unified format
    """
    tool_calls = []

    if not isinstance(content, list):
        return tool_calls

    for block in content:
        block_type = None
        tool_id = None
        tool_name = None
        tool_input = {}

        if isinstance(block, dict):
            block_type = block.get("type")
            tool_id = block.get("id")
            tool_name = block.get("name")
            tool_input = block.get("input", {})
        elif hasattr(block, "type"):
            block_type = block.type
            tool_id = getattr(block, "id", None)
            tool_name = getattr(block, "name", None)
            tool_input = getattr(block, "input", {})

        if block_type in ("tool_use", "server_tool_use") and tool_id and tool_name:
            # server_tool_use represents a COMPLETED server-side operation (e.g. web_search
            # that already executed). It must NOT be converted to a tool_call because:
            # 1. The client won't (and shouldn't) send a tool_result for it
            # 2. Kiro API requires every tool_use to have a following tool_result
            # 3. Including it causes: "tool_use ids found without tool_result blocks" (400)
            if block_type == "server_tool_use":
                continue
            tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_input
                        if isinstance(tool_input, str)
                        else tool_input,
                    },
                }
            )

    return tool_calls


def convert_anthropic_messages(
    messages: List[AnthropicMessage],
) -> List[UnifiedMessage]:
    """
    Converts Anthropic messages to unified format.

    Handles:
    - Text content (string or list of text blocks)
    - Tool use blocks (assistant messages)
    - Tool result blocks (user messages)

    Args:
        messages: List of Anthropic messages

    Returns:
        List of messages in unified format
    """

    unified_messages = []
    total_tool_calls = 0
    total_tool_results = 0
    total_images = 0

    for msg in messages:
        role = msg.role
        content = msg.content

        # Extract text content
        text_content = convert_anthropic_content_to_text(content)

        # Extract tool-related data and images based on role
        tool_calls = None
        tool_results = None
        images = None
        reasoning_content = None
        reasoning_signature = None

        if role == "assistant":
            # Assistant messages may contain tool_use and thinking blocks.
            tool_calls = extract_tool_uses_from_anthropic_content(content)
            reasoning_content, reasoning_signature = extract_reasoning_from_anthropic_content(content)
            if tool_calls:
                total_tool_calls += len(tool_calls)

        elif role == "user":
            # User messages may contain tool_result blocks and images
            tool_results = extract_tool_results_from_anthropic_content(content)
            if tool_results:
                total_tool_results += len(tool_results)

            # Extract images from user messages (both top-level and inside tool_results)
            images = extract_images_from_content(content)

            # Also extract images from inside tool_result content blocks
            # (e.g., screenshots returned by browser MCP tools)
            tool_result_images = extract_images_from_tool_results(content)
            if tool_result_images:
                if images:
                    images.extend(tool_result_images)
                else:
                    images = tool_result_images

            if images:
                total_images += len(images)

        unified_msg = UnifiedMessage(
            role=role,
            content=text_content,
            tool_calls=tool_calls if tool_calls else None,
            tool_results=tool_results if tool_results else None,
            images=images if images else None,
            reasoning_content=reasoning_content,
            reasoning_signature=reasoning_signature,
        )
        unified_messages.append(unified_msg)

    # Log summary if any tool content or images were found
    if total_tool_calls > 0 or total_tool_results > 0 or total_images > 0:
        logger.debug(
            f"Converted {len(messages)} Anthropic messages: "
            f"{total_tool_calls} tool_calls, {total_tool_results} tool_results, {total_images} images"
        )

    return unified_messages


def convert_anthropic_tools(
    tools: Optional[List[AnthropicTool]],
) -> Optional[List[UnifiedTool]]:
    """
    Converts Anthropic tools to unified format.

    Args:
        tools: List of Anthropic tools

    Returns:
        List of tools in unified format, or None if no tools
    """
    if not tools:
        return None

    unified_tools = []
    for tool in tools:
        # Handle both dict and Pydantic model
        if isinstance(tool, dict):
            name = tool.get("name", "")
            description = tool.get("description")
            input_schema = tool.get("input_schema", {})
        else:
            name = tool.name
            description = tool.description
            input_schema = tool.input_schema

        unified_tools.append(
            UnifiedTool(name=name, description=description, input_schema=input_schema)
        )

    return unified_tools if unified_tools else None


def extract_thinking_config_from_anthropic(
    request: AnthropicMessagesRequest,
    model_id: str = "",
    model_info: Optional[Dict[str, Any]] = None,
) -> ThinkingConfig:
    """
    Extract native Kiro reasoning fields or configure the fake fallback.

    Native fields require an explicit client request AND model support confirmed
    by get_native_reasoning_format() (AWS additionalModelRequestFieldsSchema for
    dynamic models, explicit whitelist for static/hidden ones). Models without
    that support fall back to prompt-based fake reasoning, so a client sending
    `thinking` to e.g. claude-haiku-4.5 never produces
    additionalModelRequestFields.

    Args:
        request: Anthropic MessagesRequest
        model_id: Resolved Kiro model ID
        model_info: Optional ListAvailableModels metadata

    Returns:
        ThinkingConfig for the core layer
    """
    thinking = request.thinking if isinstance(request.thinking, dict) else None
    output_config = request.output_config if isinstance(request.output_config, dict) else None

    if thinking and thinking.get("type") == "disabled":
        return ThinkingConfig(enabled=False)

    native_requested = thinking is not None or output_config is not None
    native_format = (
        get_native_reasoning_format(model_id, model_info)
        if native_requested
        else None
    )

    if native_format == "output_config":
        native_thinking = dict(thinking or {})
        thinking_type = native_thinking.pop("type", None)
        native_thinking.pop("budget_tokens", None)
        native_thinking["type"] = "adaptive" if thinking_type in (None, "enabled") else thinking_type
        native_thinking.setdefault("display", "summarized")
        native_output_config = dict(output_config or {})
        raw_effort = native_output_config.pop("effort", None)
        effort = normalize_native_reasoning_effort(raw_effort, model_id, model_info)
        if effort and effort != "none":
            native_output_config["effort"] = effort
        native_fields: Dict[str, Any] = {"thinking": native_thinking}
        if native_output_config:
            native_fields["output_config"] = native_output_config
        return ThinkingConfig(enabled=True, native_fields=native_fields)

    if native_format == "reasoning" and output_config and output_config.get("effort"):
        effort = normalize_native_reasoning_effort(
            output_config["effort"], model_id, model_info
        )
        if effort and effort != "none":
            return ThinkingConfig(
                enabled=True,
                native_fields={"reasoning": {"effort": effort}},
            )

    budget = thinking.get("budget_tokens") if thinking else None
    if budget:
        logger.debug(f"Using fake reasoning fallback with budget={budget}")
    return ThinkingConfig(enabled=True, budget_tokens=budget)


def anthropic_to_kiro(
    request: AnthropicMessagesRequest,
    conversation_id: str,
    profile_arn: str,
    model_info: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Converts Anthropic Messages API request to Kiro API payload.

    This is the main entry point for Anthropic → Kiro conversion.

    Key differences from OpenAI:
    - System prompt is a separate field (not in messages)
    - Content can be string or list of content blocks
    - Tool format uses input_schema instead of parameters

    Args:
        request: Anthropic MessagesRequest
        conversation_id: Unique conversation ID
        profile_arn: AWS CodeWhisperer profile ARN

    Returns:
        Payload dictionary for POST request to Kiro API

    Raises:
        ValueError: If there are no messages to send
    """
    # Convert messages to unified format
    unified_messages = convert_anthropic_messages(request.messages)

    # Convert tools to unified format
    unified_tools = convert_anthropic_tools(request.tools)

    # System prompt is already separate in Anthropic format!
    # It can be a string or list of content blocks (for prompt caching)
    system_prompt = extract_system_prompt(request.system)

    # Get model ID for Kiro API (normalizes + resolves hidden models)
    # Pass-through principle: we normalize and send to Kiro, Kiro decides if valid
    resolved_model = MODEL_ALIASES.get(request.model, request.model)
    model_id = get_model_id_for_kiro(resolved_model, HIDDEN_MODELS)

    # Prefer native Kiro reasoning when explicitly requested and supported.
    thinking_config = extract_thinking_config_from_anthropic(
        request,
        model_id=model_id,
        model_info=model_info,
    )

    logger.debug(
        f"Converting Anthropic request: model={request.model} -> {model_id}, "
        f"messages={len(unified_messages)}, tools={len(unified_tools) if unified_tools else 0}, "
        f"system_prompt_length={len(system_prompt)}, "
        f"thinking_enabled={thinking_config.enabled}, thinking_budget={thinking_config.budget_tokens}"
    )

    # Use core function to build payload
    result = build_kiro_payload(
        messages=unified_messages,
        system_prompt=system_prompt,
        model_id=model_id,
        tools=unified_tools,
        conversation_id=conversation_id,
        profile_arn=profile_arn,
        thinking_config=thinking_config,
    )

    return result.payload
