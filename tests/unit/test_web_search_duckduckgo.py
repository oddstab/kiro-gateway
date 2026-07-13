# -*- coding: utf-8 -*-

"""Tests for DuckDuckGo web search provider."""

import pytest
from unittest.mock import patch, MagicMock

from kiro.web_search_duckduckgo import call_duckduckgo, _ddg_search


@pytest.mark.asyncio
async def test_call_duckduckgo_returns_expected_format():
    """DDG results match the (tool_use_id, results_dict) contract."""
    fake_results = [
        {"title": "Example", "href": "https://example.com", "body": "A snippet"},
        {"title": "Other", "href": "https://other.com", "body": "Another"},
    ]

    with patch("kiro.web_search_duckduckgo._ddg_search", return_value=fake_results):
        tool_use_id, results = await call_duckduckgo("test query")

    assert tool_use_id is not None
    assert tool_use_id.startswith("srvtoolu_")
    assert results["query"] == "test query"
    assert results["totalResults"] == 2
    assert results["results"][0]["title"] == "Example"
    assert results["results"][0]["url"] == "https://example.com"
    assert results["results"][0]["snippet"] == "A snippet"
    assert results["results"][1]["title"] == "Other"


@pytest.mark.asyncio
async def test_call_duckduckgo_handles_exception():
    """Returns (None, None) on search failure."""
    with patch("kiro.web_search_duckduckgo._ddg_search", side_effect=Exception("network")):
        tool_use_id, results = await call_duckduckgo("fail query")

    assert tool_use_id is None
    assert results is None


@pytest.mark.asyncio
async def test_call_duckduckgo_missing_library():
    """Returns (None, None) if duckduckgo-search not installed."""
    with patch("kiro.web_search_duckduckgo.DDGS", None):
        tool_use_id, results = await call_duckduckgo("no lib")

    assert tool_use_id is None
    assert results is None
