# -*- coding: utf-8 -*-

"""DuckDuckGo web search provider.

Selected with ``WEB_SEARCH_PROVIDER=duckduckgo`` (or ``ddg``). Returns the same
``(tool_use_id, results_dict)`` tuple as the Kiro MCP provider.
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException
except ImportError:
    DDGS = None
    DDGSException = RuntimeError

DUCKDUCKGO_BACKEND = "duckduckgo"
WebSearchResult = Tuple[Optional[str], Optional[Dict[str, Any]]]


async def call_duckduckgo(query: str, max_results: int = 10) -> WebSearchResult:
    """Search DuckDuckGo and normalize results to the Kiro MCP shape.

    Args:
        query: Search query.
        max_results: Maximum number of results to return.

    Returns:
        A tool-use ID and normalized result dictionary, or ``(None, None)``
        when the provider is unavailable or the search fails.
    """
    if DDGS is None:
        logger.error("ddgs is not installed")
        return None, None

    try:
        raw_results = await asyncio.to_thread(_ddg_search, query, max_results)
    except DDGSException as error:
        logger.error(f"DuckDuckGo search failed: {error}")
        return None, None

    normalized_results = [
        {
            "title": result.get("title", ""),
            "url": result.get("href", ""),
            "snippet": result.get("body", ""),
            "publishedDate": None,
        }
        for result in raw_results
        if isinstance(result, dict)
    ]
    tool_use_id = f"srvtoolu_{uuid.uuid4().hex[:32]}"
    results: Dict[str, Any] = {
        "results": normalized_results,
        "totalResults": len(normalized_results),
        "query": query,
    }

    logger.debug(
        f"DuckDuckGo returned {len(normalized_results)} results for: {query}"
    )
    return tool_use_id, results


def _ddg_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Run the synchronous DuckDuckGo client inside a worker thread.

    Args:
        query: Search query.
        max_results: Maximum number of results to return.

    Returns:
        Raw DuckDuckGo result dictionaries.
    """
    with DDGS() as ddgs:
        return list(
            ddgs.text(
                query,
                backend=DUCKDUCKGO_BACKEND,
                max_results=max_results,
            )
        )
