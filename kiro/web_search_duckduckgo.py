# -*- coding: utf-8 -*-

"""
DuckDuckGo web search provider.

Used as fallback for models that cannot use Kiro MCP web_search (e.g. Grok).
Returns the same (tool_use_id, results_dict) tuple as call_kiro_mcp_api.
"""

import asyncio
import uuid
from typing import Dict, Optional, Tuple

from loguru import logger

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None


async def call_duckduckgo(query: str, max_results: int = 10) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Search DuckDuckGo and return results in the same format as call_kiro_mcp_api.

    Args:
        query: Search query
        max_results: Max results to return

    Returns:
        (tool_use_id, results_dict) or (None, None) on failure
    """
    if DDGS is None:
        logger.error("duckduckgo-search not installed")
        return None, None

    try:
        raw = await asyncio.to_thread(_ddg_search, query, max_results)
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return None, None

    if raw is None:
        return None, None

    tool_use_id = f"srvtoolu_{uuid.uuid4().hex[:32]}"
    results = {
        "results": [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "publishedDate": None,
            }
            for r in raw
        ],
        "totalResults": len(raw),
        "query": query,
    }

    logger.debug(f"DuckDuckGo returned {len(raw)} results for: {query}")
    return tool_use_id, results


def _ddg_search(query: str, max_results: int):
    """Sync wrapper — runs in thread via asyncio.to_thread."""
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))
