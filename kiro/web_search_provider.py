# -*- coding: utf-8 -*-

"""Select the configured backend for gateway-executed web searches."""

from typing import Any, Dict, Optional, Tuple

from loguru import logger

from kiro.config import (
    WEB_SEARCH_PROVIDER,
    WEB_SEARCH_PROVIDER_DUCKDUCKGO,
    WEB_SEARCH_PROVIDER_KIRO,
)
from kiro.mcp_tools import call_kiro_mcp_api
from kiro.web_search_duckduckgo import call_duckduckgo

WebSearchResult = Tuple[Optional[str], Optional[Dict[str, Any]]]


def web_search_requires_kiro_auth() -> bool:
    """Return whether the configured provider needs a Kiro auth manager.

    Returns:
        True when searches use the Kiro MCP backend.
    """
    return WEB_SEARCH_PROVIDER == WEB_SEARCH_PROVIDER_KIRO


async def call_web_search(
    query: str,
    auth_manager: Optional[Any] = None,
) -> WebSearchResult:
    """Run a web search using the configured provider.

    Args:
        query: Search query supplied by the client or model.
        auth_manager: Kiro auth manager. Required only by the ``kiro`` provider.

    Returns:
        A tool-use ID and normalized results, or ``(None, None)`` on failure.
    """
    logger.info(f"Executing web search with provider={WEB_SEARCH_PROVIDER}")

    if WEB_SEARCH_PROVIDER == WEB_SEARCH_PROVIDER_DUCKDUCKGO:
        return await call_duckduckgo(query)

    if auth_manager is None:
        logger.error("Kiro web search requires an initialized auth manager")
        return None, None

    return await call_kiro_mcp_api(query, auth_manager)
