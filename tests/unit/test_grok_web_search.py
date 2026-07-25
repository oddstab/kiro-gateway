# -*- coding: utf-8 -*-

"""
Tests for the Grok Build web_search integration (OpenAI Responses API path).

Grok Build's web_search tool does NOT emit `<web_search>` tagged text for the
gateway to parse. It performs a real backend API call, verified against
grok-build's source:

    crates/codegen/xai-grok-tools/src/implementations/web_search/client.rs
        WebSearchClient::search()
            POST {base_url}/responses
            body: {model, input, tools:[{type:"web_search",...}], store,
                   temperature, top_p, max_output_tokens}
            reply parsed as async-openai rs::Response -> output_text()
                                                      -> extract_citations()

    crates/codegen/xai-grok-shell/src/agent/config.rs
        resolve_web_search_sampling_config()
            1. find_model_by_id(models, models.web_search) -> reuse that entry's
               base_url + api_key  => request reaches THIS gateway
            2. id not in the list -> endpoints.resolve_inference_base_url()
               => request goes to xAI upstream, gateway never sees it

Tests cover:
- The client-side search proxy is not advertised as a Kiro chat model
- Request detection and query extraction for the real Grok Build request body
- Responses payload shape, including url_citation annotations and offsets
- No `<web_search>` tag leakage on this path
"""

import pytest

from kiro.grok_web_search import (
    build_responses_payload,
    extract_query_from_responses_input,
    is_grok_web_search_request,
)


SEARCH_PROXY_MODEL = "kiro-search-proxy"

# The exact request Grok Build sends (client.rs::search, lines 112-136).
GROK_BUILD_REQUEST = {
    "model": SEARCH_PROXY_MODEL,
    "input": "kiro gateway web search",
    "tools": [{"type": "web_search", "filters": {"allowed_domains": None}}],
    "store": False,
    "temperature": 0.1,
    "top_p": 0.95,
    "max_output_tokens": 8192,
}

KIRO_MCP_RESULTS = {
    "results": [
        {
            "title": "Web tools",
            "url": "https://kiro.dev/docs/chat/webtools/",
            "snippet": "Web access capabilities.",
            "publishedDate": 1700000000000,
        },
        {
            "title": "kiro-web-search PyPI",
            "url": "https://pypi.org/project/kiro-web-search/",
            "snippet": "A minimal MCP server.",
            "publishedDate": None,
        },
    ],
    "totalResults": 2,
}


def _output_text(payload: dict) -> str:
    """Mirror of async-openai rs::Response::output_text()."""
    parts = []
    for item in payload.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    parts.append(block.get("text", ""))
    return "".join(parts)


def _annotations(payload: dict) -> list:
    return payload["output"][0]["content"][0]["annotations"]


def _extract_citations(payload: dict) -> list:
    """Mirror of grok-build WebSearchClient::extract_citations (client.rs:280).

    Collects url_citation annotation URLs and dedupes, preserving order.
    """
    urls = []
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for block in item.get("content", []):
            if block.get("type") != "output_text":
                continue
            for ann in block.get("annotations", []):
                if ann.get("type") == "url_citation" and ann.get("url"):
                    urls.append(ann["url"])
    seen = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


class TestSearchProxyIsClientOnly:
    """The search proxy must never masquerade as a Kiro chat model."""

    def test_shipped_aliases_contain_no_grok_models_or_search_proxy(self):
        """
        What it does: Inspects the aliases shipped by the gateway.
        Purpose: Removed Grok-to-Opus mappings must not reappear in /v1/models.
        """
        from kiro.config import MODEL_ALIASES

        assert SEARCH_PROXY_MODEL not in MODEL_ALIASES
        assert not any(model_id.startswith("grok-") for model_id in MODEL_ALIASES)

    @pytest.mark.asyncio
    async def test_model_resolver_does_not_advertise_search_proxy(self):
        """
        What it does: Builds the real resolver from the shipped aliases.
        Purpose: The proxy is configured by Grok Build, not advertised as chat.
        """
        from kiro.cache import ModelInfoCache
        from kiro.config import MODEL_ALIASES
        from kiro.model_resolver import ModelResolver

        cache = ModelInfoCache()
        await cache.update([{"modelId": "claude-opus-4.6"}])
        resolver = ModelResolver(cache=cache, aliases=MODEL_ALIASES)

        available_models = resolver.get_available_models()
        assert SEARCH_PROXY_MODEL not in available_models
        assert not any(model_id.startswith("grok-") for model_id in available_models)


class TestRequestDetection:
    """is_grok_web_search_request / extract_query_from_responses_input."""

    def test_detects_real_grok_build_request(self):
        """
        What it does: Feeds the exact body from client.rs to the detector.
        Purpose: The primary happy path; a miss returns HTTP 501 to Grok Build.
        """
        assert is_grok_web_search_request(GROK_BUILD_REQUEST) is True

    @pytest.mark.parametrize(
        "tool_type",
        ["web_search", "web_search_preview", "web_search_2025_01_01"],
    )
    def test_detects_versioned_tool_types(self, tool_type):
        """
        What it does: Verifies versioned/preview web_search variants match.
        Purpose: Grok Build may bump the tool type; prefix matching covers it.
        """
        assert is_grok_web_search_request({"tools": [{"type": tool_type}]}) is True

    def test_detects_function_named_web_search(self):
        """
        What it does: Verifies a function-style tool named web_search matches.
        Purpose: Some clients express the tool as a named function.
        """
        assert is_grok_web_search_request(
            {"tools": [{"type": "function", "name": "web_search"}]}
        ) is True

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"tools": []},
            {"tools": [{"type": "code_interpreter"}]},
            {"tools": [{"type": "function", "name": "read_file"}]},
            {"tools": ["not-a-dict"]},
            {"tools": None},
        ],
    )
    def test_rejects_non_web_search_requests(self, body):
        """
        What it does: Verifies unrelated/malformed tool lists do not match.
        Purpose: The endpoint must not hijack other Responses API traffic.
        """
        assert is_grok_web_search_request(body) is False

    def test_extracts_query_from_string_input(self):
        """
        What it does: Extracts the query from Grok Build's plain string input.
        Purpose: This is the shape client.rs actually sends (.input(query)).
        """
        assert extract_query_from_responses_input(GROK_BUILD_REQUEST) == (
            "kiro gateway web search"
        )

    def test_extracts_query_from_message_items(self):
        """
        What it does: Extracts from the structured input-items form.
        Purpose: Responses API also permits an input array.
        """
        body = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "  spaced query  "}],
                }
            ]
        }
        assert extract_query_from_responses_input(body) == "spaced query"

    @pytest.mark.parametrize("body", [{}, {"input": ""}, {"input": "   "}, {"input": []}])
    def test_missing_query_returns_empty(self, body):
        """
        What it does: Verifies empty/blank inputs yield "".
        Purpose: The route turns "" into an actionable HTTP 400.
        """
        assert extract_query_from_responses_input(body) == ""


class TestResponsesPayload:
    """build_responses_payload must satisfy async-openai's rs::Response."""

    def test_payload_has_required_response_envelope(self):
        """
        What it does: Verifies the top-level Response fields exist.
        Purpose: A missing field fails deserialization -> HTTP 200 but the
                 tool reports "failed".
        """
        payload = build_responses_payload(SEARCH_PROXY_MODEL, "q", KIRO_MCP_RESULTS)

        assert payload["object"] == "response"
        assert payload["status"] == "completed"
        assert payload["model"] == SEARCH_PROXY_MODEL
        assert payload["id"].startswith("resp_")
        assert isinstance(payload["created_at"], int)
        assert isinstance(payload["output"], list) and payload["output"]

    def test_output_text_carries_every_result(self):
        """
        What it does: Verifies titles, URLs and snippets all reach output_text.
        Purpose: output_text() is the only content Grok Build shows the model,
                 so nothing may be dropped.
        """
        payload = build_responses_payload(SEARCH_PROXY_MODEL, "q", KIRO_MCP_RESULTS)
        text = _output_text(payload)

        for result in KIRO_MCP_RESULTS["results"]:
            assert result["title"] in text
            assert result["url"] in text
            assert result["snippet"] in text

    def test_no_web_search_tag_leak(self):
        """
        What it does: Verifies this path never emits literal <web_search> tags.
        Purpose: Grok Build renders output_text verbatim.
        """
        payload = build_responses_payload(SEARCH_PROXY_MODEL, "q", KIRO_MCP_RESULTS)
        text = _output_text(payload)

        assert "<web_search>" not in text
        assert "</web_search>" not in text

    def test_citations_are_extractable(self):
        """
        What it does: Runs grok-build's extract_citations logic on the payload.
        Purpose: Annotations must be readable by the real consumer, in order.
        """
        payload = build_responses_payload(SEARCH_PROXY_MODEL, "q", KIRO_MCP_RESULTS)

        assert _extract_citations(payload) == [
            "https://kiro.dev/docs/chat/webtools/",
            "https://pypi.org/project/kiro-web-search/",
        ]

    def test_annotation_offsets_point_at_their_own_result(self):
        """
        What it does: Slices output_text with each annotation's index range.
        Purpose: Offsets must select the span describing that very result,
                 otherwise citations render against the wrong text.
        """
        payload = build_responses_payload(SEARCH_PROXY_MODEL, "q", KIRO_MCP_RESULTS)
        text = _output_text(payload)

        annotations = _annotations(payload)
        assert len(annotations) == len(KIRO_MCP_RESULTS["results"])

        for annotation in annotations:
            start, end = annotation["start_index"], annotation["end_index"]
            assert 0 <= start < end <= len(text), f"bad range {start}..{end}"
            span = text[start:end]
            assert annotation["url"] in span
            assert annotation["title"] in span

    def test_annotation_schema_matches_grok_build_fixtures(self):
        """
        What it does: Verifies each annotation has exactly the expected keys.
        Purpose: An unknown/renamed field can fail rs::Response parsing, which
                 is why annotations were previously left empty.
        """
        payload = build_responses_payload(SEARCH_PROXY_MODEL, "q", KIRO_MCP_RESULTS)

        for annotation in _annotations(payload):
            assert set(annotation) == {
                "type",
                "url",
                "title",
                "start_index",
                "end_index",
            }
            assert annotation["type"] == "url_citation"
            assert isinstance(annotation["start_index"], int)
            assert isinstance(annotation["end_index"], int)

    def test_result_without_url_gets_no_annotation_but_stays_in_text(self):
        """
        What it does: Verifies a URL-less result is skipped for citations only.
        Purpose: An empty url would produce a useless citation, but the text
                 content must still be preserved.
        """
        results = {
            "results": [
                {"title": "No link", "url": "", "snippet": "still visible"},
                {"title": "Linked", "url": "https://x.example", "snippet": "s"},
            ]
        }
        payload = build_responses_payload(SEARCH_PROXY_MODEL, "q", results)
        text = _output_text(payload)

        assert _extract_citations(payload) == ["https://x.example"]
        assert "No link" in text
        assert "still visible" in text

    @pytest.mark.parametrize("results", [None, {}, {"results": []}])
    def test_empty_results_still_parse(self, results):
        """
        What it does: Verifies the no-results case returns a valid Response.
        Purpose: A backend miss must not break deserialization.
        """
        payload = build_responses_payload(SEARCH_PROXY_MODEL, "q", results)

        assert payload["status"] == "completed"
        assert _output_text(payload) == "No search results found."
        assert _annotations(payload) == []

    def test_invalid_published_date_is_skipped(self):
        """
        What it does: Verifies an out-of-range timestamp does not raise.
        Purpose: Malformed backend data must not 500 the endpoint, and offsets
                 must stay consistent with the text actually emitted.
        """
        results = {
            "results": [
                {
                    "title": "Bad date",
                    "url": "https://x.example",
                    "snippet": "s",
                    "publishedDate": 10**18,
                }
            ]
        }
        payload = build_responses_payload(SEARCH_PROXY_MODEL, "q", results)
        text = _output_text(payload)

        assert "Published:" not in text
        annotation = _annotations(payload)[0]
        assert annotation["url"] in text[annotation["start_index"] : annotation["end_index"]]

    def test_missing_title_falls_back_to_untitled(self):
        """
        What it does: Verifies a result without a title renders "Untitled".
        Purpose: Avoid an empty markdown-ish entry, and keep the annotation
                 title consistent with the text.
        """
        results = {"results": [{"url": "https://x.example", "snippet": "s"}]}
        payload = build_responses_payload(SEARCH_PROXY_MODEL, "q", results)
        text = _output_text(payload)

        assert "Untitled" in text
        annotation = _annotations(payload)[0]
        assert annotation["title"] == "Untitled"
        assert annotation["title"] in text[annotation["start_index"] : annotation["end_index"]]


class TestResponsesEndpoint:
    """Drive POST /v1/responses through the real FastAPI route.

    Only the Kiro MCP backend is faked; routing, auth and payload building are
    the shipped code.
    """

    @staticmethod
    def _client(monkeypatch, backend_results=KIRO_MCP_RESULTS):
        """Build a TestClient whose app has one fake account and a fake search."""
        from unittest.mock import AsyncMock, MagicMock

        from fastapi import FastAPI

        from kiro.routes_openai import router

        async def _fake_search(query, auth_manager=None):
            return "srvtoolu_fake", backend_results

        monkeypatch.setattr("kiro.routes_openai.call_web_search", _fake_search)

        app = FastAPI()
        app.include_router(router)

        account = MagicMock()
        account.auth_manager = MagicMock()
        manager = MagicMock()
        manager.get_first_account.return_value = account
        manager.get_next_account = AsyncMock(return_value=account)
        app.state.account_manager = manager
        app.state.account_system = False

        from fastapi.testclient import TestClient

        return TestClient(app)

    @staticmethod
    def _auth_headers():
        from kiro.config import PROXY_API_KEY

        return {"Authorization": f"Bearer {PROXY_API_KEY}"}

    def test_real_grok_build_request_returns_parseable_response(self, monkeypatch):
        """
        What it does: POSTs the exact Grok Build body to the real route.
        Purpose: End-to-end proof that the endpoint answers 200 with a body
                 whose output_text and citations are both usable.
        """
        client = self._client(monkeypatch)
        response = client.post(
            "/v1/responses", json=GROK_BUILD_REQUEST, headers=self._auth_headers()
        )

        assert response.status_code == 200
        payload = response.json()
        text = _output_text(payload)

        assert "Web tools" in text
        assert "kiro.dev/docs/chat/webtools" in text
        assert "<web_search>" not in text
        assert _extract_citations(payload) == [
            "https://kiro.dev/docs/chat/webtools/",
            "https://pypi.org/project/kiro-web-search/",
        ]

    def test_requires_api_key(self, monkeypatch):
        """
        What it does: POSTs without an Authorization header.
        Purpose: The endpoint must not be an unauthenticated search proxy.
        """
        client = self._client(monkeypatch)
        response = client.post("/v1/responses", json=GROK_BUILD_REQUEST)

        assert response.status_code == 401

    def test_non_web_search_request_is_rejected(self, monkeypatch):
        """
        What it does: POSTs a Responses request with an unrelated tool.
        Purpose: Only web_search is implemented here; anything else gets 501
                 rather than a misleading empty result.
        """
        client = self._client(monkeypatch)
        response = client.post(
            "/v1/responses",
            json={"model": "m", "input": "hi", "tools": [{"type": "code_interpreter"}]},
            headers=self._auth_headers(),
        )

        assert response.status_code == 501

    def test_missing_query_returns_400(self, monkeypatch):
        """
        What it does: POSTs a web_search request with a blank input.
        Purpose: Actionable error instead of an empty search.
        """
        client = self._client(monkeypatch)
        response = client.post(
            "/v1/responses",
            json={"model": "m", "input": "   ", "tools": [{"type": "web_search"}]},
            headers=self._auth_headers(),
        )

        assert response.status_code == 400

    def test_backend_failure_returns_502(self, monkeypatch):
        """
        What it does: Makes the Kiro MCP search return no results at all.
        Purpose: A backend failure must surface as an error, not as a
                 successful-looking empty Response.
        """
        from unittest.mock import AsyncMock, MagicMock

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from kiro.routes_openai import router

        async def _failing_search(query, auth_manager=None):
            return None, None

        monkeypatch.setattr("kiro.routes_openai.call_web_search", _failing_search)

        app = FastAPI()
        app.include_router(router)
        account = MagicMock()
        account.auth_manager = MagicMock()
        manager = MagicMock()
        manager.get_first_account.return_value = account
        manager.get_next_account = AsyncMock(return_value=account)
        app.state.account_manager = manager
        app.state.account_system = False

        with TestClient(app) as client:
            response = client.post(
                "/v1/responses", json=GROK_BUILD_REQUEST, headers=self._auth_headers()
            )

        assert response.status_code == 502
        assert "WEB_SEARCH_PROVIDER" in response.json()["error"]["message"]

    def test_duckduckgo_mode_does_not_require_kiro_account(self, monkeypatch):
        """DDG-backed Responses searches must not touch Kiro account selection."""
        from unittest.mock import AsyncMock, MagicMock

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import kiro.routes_openai as routes_openai

        search = AsyncMock(return_value=("srvtoolu_ddg", KIRO_MCP_RESULTS))
        monkeypatch.setattr(
            routes_openai,
            "web_search_requires_kiro_auth",
            lambda: False,
        )
        monkeypatch.setattr(routes_openai, "call_web_search", search)

        app = FastAPI()
        app.include_router(routes_openai.router)
        manager = MagicMock()
        app.state.account_manager = manager
        app.state.account_system = False

        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json=GROK_BUILD_REQUEST,
                headers=self._auth_headers(),
            )

        assert response.status_code == 200
        manager.get_first_account.assert_not_called()
        manager.get_next_account.assert_not_called()
        search.assert_awaited_once_with("kiro gateway web search", None)

    def test_kiro_mode_without_account_returns_503(self, monkeypatch):
        """Kiro MCP searches must fail before dispatch when no account is usable."""
        from unittest.mock import AsyncMock, MagicMock

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import kiro.routes_openai as routes_openai

        search = AsyncMock()
        monkeypatch.setattr(
            routes_openai,
            "web_search_requires_kiro_auth",
            lambda: True,
        )
        monkeypatch.setattr(routes_openai, "call_web_search", search)

        app = FastAPI()
        app.include_router(routes_openai.router)
        manager = MagicMock()
        manager.get_first_account.return_value = None
        app.state.account_manager = manager
        app.state.account_system = False

        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json=GROK_BUILD_REQUEST,
                headers=self._auth_headers(),
            )

        assert response.status_code == 503
        assert "Kiro web search" in response.json()["error"]["message"]
        search.assert_not_awaited()

    def test_model_list_does_not_advertise_search_proxy(self, monkeypatch):
        """
        What it does: Calls the real /v1/models route and inspects every id.
        Purpose: The client-side Responses route must not masquerade as chat.
        """
        from unittest.mock import MagicMock

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from kiro.routes_openai import router

        app = FastAPI()
        app.include_router(router)

        resolver = MagicMock()
        resolver.get_available_models.return_value = ["claude-opus-4.6"]
        account = MagicMock()
        account.model_resolver = resolver
        manager = MagicMock()
        manager.get_first_account.return_value = account
        manager.get_model_context_window.return_value = 1_000_000
        app.state.account_manager = manager
        app.state.account_system = False

        with TestClient(app) as client:
            response = client.get("/v1/models", headers=self._auth_headers())

        assert response.status_code == 200
        model_ids = {model["id"] for model in response.json()["data"]}
        assert SEARCH_PROXY_MODEL not in model_ids
        assert not any(model_id.startswith("grok-") for model_id in model_ids)


class TestGrokBuildConfigTomlContract:
    """The client-side Grok Build config must route search to this gateway."""

    REQUIRED_TOML = """
        [models]
        web_search = "kiro-search-proxy"

        [model.kiro-search-proxy]
        model = "kiro-search-proxy"
        base_url = "http://localhost:8000/v1"
        env_key = "GROK_CODE_XAI_API_KEY"
        api_backend = "responses"
        context_window = 1000000
    """

    @staticmethod
    def _parse(toml_text: str) -> dict:
        import textwrap
        import tomllib

        return tomllib.loads(textwrap.dedent(toml_text))

    @staticmethod
    def _find_model_by_id(models: dict, model_id: str):
        """Mirror Grok Build's model lookup by key and model field."""
        if model_id in models:
            return models[model_id]
        for entry in models.values():
            if isinstance(entry, dict) and entry.get("model") == model_id:
                return entry
        return None

    def test_required_stanza_resolves_to_the_gateway(self):
        """The documented client entry must resolve to /v1/responses."""
        cfg = self._parse(self.REQUIRED_TOML)
        search_model = cfg["models"]["web_search"]

        assert search_model == SEARCH_PROXY_MODEL
        entry = self._find_model_by_id(cfg.get("model", {}), search_model)
        assert entry is not None
        assert entry["model"] == SEARCH_PROXY_MODEL
        assert entry["base_url"].rstrip("/") + "/responses" == (
            "http://localhost:8000/v1/responses"
        )

    def test_proxy_name_is_a_valid_bare_toml_key(self):
        """The hyphenated proxy id must parse without quoted dotted-key traps."""
        cfg = self._parse(self.REQUIRED_TOML)

        assert SEARCH_PROXY_MODEL in cfg["model"]

    def test_missing_base_url_cannot_route_to_gateway(self):
        """A client model entry without base_url is insufficient."""
        cfg = self._parse(
            """
            [models]
            web_search = "kiro-search-proxy"

            [model.kiro-search-proxy]
            model = "kiro-search-proxy"
            """
        )
        entry = self._find_model_by_id(cfg["model"], cfg["models"]["web_search"])

        assert entry is not None
        assert "base_url" not in entry

    def test_env_example_documents_client_side_proxy(self):
        """The example env file must explain the client-only search route."""
        from pathlib import Path

        text = Path(".env.example").read_text(encoding="utf-8")

        assert "[model.kiro-search-proxy]" in text
        assert "not advertise it in /v1/models" in text

    def test_readme_toml_example_is_valid_and_resolves(self):
        """The README's copy-paste TOML must route search to this gateway."""
        import re
        from pathlib import Path

        readme = Path("README.md").read_text(encoding="utf-8")
        blocks = [
            block
            for block in re.findall(r"```toml\n(.*?)```", readme, re.DOTALL)
            if "web_search" in block
        ]
        assert blocks, "README must document the config.toml web_search stanza"

        cfg = self._parse(blocks[0])
        search_model = cfg["models"]["web_search"]
        assert search_model == SEARCH_PROXY_MODEL

        entry = self._find_model_by_id(cfg.get("model", {}), search_model)
        assert entry is not None
        assert entry["base_url"] == "http://localhost:8000/v1"

    @pytest.mark.parametrize(
        "readme_path",
        [
            "README.md",
            "docs/en/README.md",
            "docs/ja/README.md",
            "docs/ko/README.md",
        ],
    )
    def test_readmes_document_ddg_for_claude_code_and_grok_build(
        self,
        readme_path,
    ):
        """Every README must explain how both clients reach Gateway DDG."""
        from pathlib import Path

        text = Path(readme_path).read_text(encoding="utf-8")

        assert "WEB_SEARCH_PROVIDER=ddg" in text
        assert "--disallowedTools WebSearch" in text
        assert 'web_search = "kiro-search-proxy"' in text
        assert "duckduckgo" in text
        assert "duckgo" in text


class TestAliasContextWindowInheritance:
    """Alias ids are absent from the Kiro cache, so they must resolve first."""

    def test_alias_inherits_context_window_from_alias_target(self):
        """
        What it does: Drives the real AccountManager.get_model_context_window
                      with an alias that only exists as a MODEL_ALIASES key.
        Purpose: Without the resolve-then-lookup fallback this returns None and
                 the client silently applies its own default context window.
        """
        from unittest.mock import MagicMock

        from kiro.account_manager import AccountManager

        real_model = "claude-opus-4.6"

        cache = MagicMock()
        cache.is_valid_model.side_effect = lambda mid: mid == real_model
        cache.get_max_input_tokens.side_effect = (
            lambda mid: 1_000_000 if mid == real_model else None
        )

        resolver = MagicMock()
        resolver.resolve.return_value = MagicMock(internal_id=real_model)

        account = MagicMock()
        account.model_cache = cache
        account.model_resolver = resolver

        manager = AccountManager.__new__(AccountManager)
        manager._accounts = {"acct": account}

        alias_model = "my-opus"

        # Direct hit still works.
        assert manager.get_model_context_window(real_model) == 1_000_000
        # A generic alias resolves through the resolver, then hits the cache.
        assert manager.get_model_context_window(alias_model) == 1_000_000

    def test_unresolvable_model_returns_none(self):
        """
        What it does: Makes the resolver raise for an unknown id.
        Purpose: A resolver error must not propagate out of a metadata lookup.
        """
        from unittest.mock import MagicMock

        from kiro.account_manager import AccountManager

        cache = MagicMock()
        cache.is_valid_model.return_value = False
        resolver = MagicMock()
        resolver.resolve.side_effect = ValueError("unknown model")

        account = MagicMock()
        account.model_cache = cache
        account.model_resolver = resolver

        manager = AccountManager.__new__(AccountManager)
        manager._accounts = {"acct": account}

        assert manager.get_model_context_window("totally-unknown") is None
