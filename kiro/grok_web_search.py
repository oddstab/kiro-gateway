# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
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
Grok Build web_search proxy (OpenAI Responses API format).

Grok Build's client-side web_search tool sends POST /v1/responses requests in
OpenAI Responses API format with a {"type": "web_search"} tool. This module
detects those requests, runs a real Kiro MCP web search via call_kiro_mcp_api,
and returns the results in the Responses API shape that Grok Build's
async-openai parser accepts.

Unlike the Path A / Path B handlers in mcp_tools.py (which emit results as
<web_search> tagged text inside chat/messages responses), this returns a
native-looking Responses object: URLs are embedded directly in output_text and
annotations are kept empty ([]) to exactly match the shape the parser accepts
(a populated url_citation annotation with a mismatched schema, or an extra
usage object with the wrong nested fields, makes the whole Response parse fail
-- HTTP 200 but the tool reports "failed").
"""

import time
import uuid
from datetime import datetime
from typing import Any, Optional

from loguru import logger


def is_grok_web_search_request(body: dict[str, Any]) -> bool:
    """Detect a Grok Build web_search Responses API request.

    Grok Build sends:
        {
            "model": "...",
            "input": "search query",
            "tools": [{"type": "web_search", ...}],
            "store": false,
            ...
        }

    Args:
        body: Parsed JSON request body.

    Returns:
        True if a web_search tool is present in the request.
    """
    tools = body.get("tools") or []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type", "")).strip().lower()
        # Accept "web_search" and any versioned/preview variant
        # (e.g. "web_search_preview", "web_search_2025xxxx").
        if tool_type.startswith("web_search"):
            return True
        # Some clients express it as a named function tool.
        if tool_type == "function" and tool.get("name") == "web_search":
            return True
    return False


def extract_query_from_responses_input(body: dict[str, Any]) -> str:
    """Extract the search query from the Responses API `input` field.

    The `input` can be a plain string (most common for web_search) or a list
    of input items (message objects).

    Args:
        body: Parsed JSON request body.

    Returns:
        The extracted query string, or "" if none found.
    """
    input_field = body.get("input", "")

    # Simple string input
    if isinstance(input_field, str):
        return input_field.strip()

    # Array of input items
    if isinstance(input_field, list):
        for item in reversed(input_field):
            if isinstance(item, dict):
                # Message item with content
                if item.get("type") == "message" and item.get("role") == "user":
                    content = item.get("content", [])
                    if isinstance(content, str):
                        return content.strip()
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "input_text":
                                return block.get("text", "").strip()
                # Direct input_text item
                elif item.get("type") == "input_text":
                    return item.get("text", "").strip()

    return str(input_field).strip() if input_field else ""


def build_responses_payload(model: str, query: str, results: Optional[dict]) -> dict[str, Any]:
    """Build an OpenAI Responses API object from Kiro MCP search results.

    Args:
        model: Model id echoed back in the response.
        query: The original search query.
        results: Parsed Kiro MCP result dict (with a "results" list), or None.

    Returns:
        A dict in OpenAI Responses API format that Grok Build accepts.
    """
    text_parts: list[str] = []

    result_items = (results or {}).get("results", []) if results else []
    for i, r in enumerate(result_items, 1):
        title = r.get("title", "") or "Untitled"
        url = r.get("url", "") or ""
        snippet = r.get("snippet", "") or ""
        published_ms = r.get("publishedDate")

        entry = f"[{i}] {title}\n"
        if url:
            entry += f"{url}\n"
        if published_ms:
            try:
                dt = datetime.fromtimestamp(published_ms / 1000)
                entry += f"Published: {dt.strftime('%d %b %Y %H:%M:%S')}\n"
            except (ValueError, OSError, TypeError):
                pass
        if snippet:
            entry += f"{snippet}\n"
        text_parts.append(entry)

    full_text = "\n".join(text_parts) if text_parts else "No search results found."

    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": msg_id,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": full_text,
                        # Kept empty on purpose -- see module docstring.
                        "annotations": [],
                    }
                ],
            }
        ],
    }
