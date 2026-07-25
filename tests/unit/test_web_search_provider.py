# -*- coding: utf-8 -*-

"""Tests for configured web-search provider dispatch."""

from unittest.mock import AsyncMock

import pytest

import kiro.web_search_provider as provider


SEARCH_RESULTS = {
    "results": [
        {
            "title": "Example",
            "url": "https://example.com",
            "snippet": "Example result",
            "publishedDate": None,
        }
    ],
    "totalResults": 1,
    "query": "test query",
}


class TestWebSearchProviderDispatch:
    """Verify that each configured provider reaches only its own backend."""

    @pytest.mark.asyncio
    async def test_kiro_provider_calls_mcp_with_auth_manager(self, monkeypatch):
        """The Kiro provider must pass the selected account to Kiro MCP."""
        auth_manager = object()
        kiro_search = AsyncMock(return_value=("srvtoolu_kiro", SEARCH_RESULTS))
        duckduckgo_search = AsyncMock()
        monkeypatch.setattr(provider, "WEB_SEARCH_PROVIDER", provider.WEB_SEARCH_PROVIDER_KIRO)
        monkeypatch.setattr(provider, "call_kiro_mcp_api", kiro_search)
        monkeypatch.setattr(provider, "call_duckduckgo", duckduckgo_search)

        result = await provider.call_web_search("test query", auth_manager)

        assert result == ("srvtoolu_kiro", SEARCH_RESULTS)
        kiro_search.assert_awaited_once_with("test query", auth_manager)
        duckduckgo_search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_duckduckgo_provider_does_not_use_kiro_auth(self, monkeypatch):
        """DuckDuckGo must work without selecting or using a Kiro account."""
        kiro_search = AsyncMock()
        duckduckgo_search = AsyncMock(
            return_value=("srvtoolu_duckduckgo", SEARCH_RESULTS)
        )
        monkeypatch.setattr(
            provider,
            "WEB_SEARCH_PROVIDER",
            provider.WEB_SEARCH_PROVIDER_DUCKDUCKGO,
        )
        monkeypatch.setattr(provider, "call_kiro_mcp_api", kiro_search)
        monkeypatch.setattr(provider, "call_duckduckgo", duckduckgo_search)

        result = await provider.call_web_search("test query")

        assert result == ("srvtoolu_duckduckgo", SEARCH_RESULTS)
        duckduckgo_search.assert_awaited_once_with("test query")
        kiro_search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_kiro_provider_without_auth_fails_closed(self, monkeypatch):
        """Kiro searches without an initialized account must not call either backend."""
        kiro_search = AsyncMock()
        duckduckgo_search = AsyncMock()
        monkeypatch.setattr(provider, "WEB_SEARCH_PROVIDER", provider.WEB_SEARCH_PROVIDER_KIRO)
        monkeypatch.setattr(provider, "call_kiro_mcp_api", kiro_search)
        monkeypatch.setattr(provider, "call_duckduckgo", duckduckgo_search)

        result = await provider.call_web_search("test query")

        assert result == (None, None)
        kiro_search.assert_not_awaited()
        duckduckgo_search.assert_not_awaited()


class TestWebSearchProviderAuthentication:
    """Verify account requirements exposed to API routes."""

    @pytest.mark.parametrize(
        ("configured_provider", "expected"),
        [
            (provider.WEB_SEARCH_PROVIDER_KIRO, True),
            (provider.WEB_SEARCH_PROVIDER_DUCKDUCKGO, False),
        ],
    )
    def test_only_kiro_requires_auth(
        self,
        monkeypatch,
        configured_provider,
        expected,
    ):
        """Only the Kiro MCP backend should trigger Kiro account selection."""
        monkeypatch.setattr(provider, "WEB_SEARCH_PROVIDER", configured_provider)

        assert provider.web_search_requires_kiro_auth() is expected
