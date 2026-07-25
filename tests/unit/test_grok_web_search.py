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
- The web_search model id is advertised by /v1/models (branch 1 above)
- Request detection and query extraction for the real Grok Build request body
- Responses payload shape, including url_citation annotations and offsets
- No `<web_search>` tag leakage on this path
"""

import pytest

from kiro.config import GROK_WEB_SEARCH_MODEL, MODEL_ALIASES
from kiro.grok_web_search import (
    build_responses_payload,
    extract_query_from_responses_input,
    is_grok_web_search_request,
)


# The exact request Grok Build sends (client.rs::search, lines 112-136).
GROK_BUILD_REQUEST = {
    "model": GROK_WEB_SEARCH_MODEL,
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


class TestWebSearchModelIsAdvertised:
    """Branch 1 of resolve_web_search_sampling_config must win.

    If the web_search model id is absent from /v1/models, Grok Build resolves
    it to xAI's own inference endpoint and this gateway is bypassed entirely.
    """

    def test_web_search_model_is_registered_as_alias(self):
        """
        What it does: Re-imports kiro.config fresh and checks the live dict.
        Purpose: Alias keys are what /v1/models advertises, so this is the hook
                 that keeps find_model_by_id() succeeding in Grok Build. Reading
                 the module attribute (not a test-local import) means removing
                 the registration in config.py actually fails this test.
        """
        import importlib

        import kiro.config as config_module

        importlib.reload(config_module)
        assert config_module.GROK_WEB_SEARCH_MODEL in config_module.MODEL_ALIASES

    def test_web_search_model_maps_to_a_real_kiro_model(self):
        """
        What it does: Verifies the alias target is a non-empty concrete model.
        Purpose: A dangling alias would make chat requests for the id fail.
        """
        import importlib

        import kiro.config as config_module

        importlib.reload(config_module)
        target = config_module.MODEL_ALIASES[config_module.GROK_WEB_SEARCH_MODEL]
        assert isinstance(target, str) and target.strip()
        assert target != config_module.GROK_WEB_SEARCH_MODEL, (
            "alias must not point at itself"
        )

    def test_default_web_search_model_matches_grok_build_default(self):
        """
        What it does: Pins the default id to grok-build's compiled-in default.
        Purpose: grok-build's default_web_search_model()
                 (xai-grok-workspace/src/session/tool_config.rs) returns
                 "grok-4.20-multi-agent" unless GROK_WEB_SEARCH_MODEL overrides
                 it. A drift here silently sends searches to xAI instead.
        """
        import importlib
        import os
        from unittest.mock import patch

        import kiro.config as config_module

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROK_WEB_SEARCH_MODEL", None)
            importlib.reload(config_module)
            assert config_module.GROK_WEB_SEARCH_MODEL == "grok-4.20-multi-agent"

    def test_web_search_model_is_env_overridable(self):
        """
        What it does: Sets GROK_WEB_SEARCH_MODEL and reloads the config.
        Purpose: Users with a customised Grok Build models.web_search must be
                 able to match it without editing source.
        """
        import importlib
        import os
        from unittest.mock import patch

        import kiro.config as config_module

        with patch.dict(os.environ, {"GROK_WEB_SEARCH_MODEL": "my-search-model"}):
            importlib.reload(config_module)
            assert config_module.GROK_WEB_SEARCH_MODEL == "my-search-model"
            assert "my-search-model" in config_module.MODEL_ALIASES

        importlib.reload(config_module)

    @pytest.mark.asyncio
    async def test_model_resolver_advertises_and_resolves_the_id(self):
        """
        What it does: Drives the real ModelResolver with the shipped aliases.
        Purpose: Proves the id appears in the /v1/models list AND resolves to a
                 usable Kiro id, which is exactly what Grok Build's
                 find_model_by_id() lookup needs.
        """
        import importlib

        import kiro.config as config_module
        from kiro.cache import ModelInfoCache
        from kiro.model_resolver import ModelResolver, normalize_model_name

        importlib.reload(config_module)
        ws_model = config_module.GROK_WEB_SEARCH_MODEL
        aliases = config_module.MODEL_ALIASES
        target = aliases[ws_model]

        cache = ModelInfoCache()
        await cache.update([{"modelId": target}])
        resolver = ModelResolver(cache=cache, aliases=aliases)

        # Advertised in /v1/models -> Grok Build's lookup finds it.
        assert ws_model in resolver.get_available_models()

        # Resolves onward to the alias target (after the usual normalization,
        # e.g. "claude-opus-4-6[1m]" -> "claude-opus-4.6").
        resolution = resolver.resolve(ws_model)
        assert resolution.internal_id == normalize_model_name(target)
        assert resolution.internal_id != ws_model


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
        payload = build_responses_payload(GROK_WEB_SEARCH_MODEL, "q", KIRO_MCP_RESULTS)

        assert payload["object"] == "response"
        assert payload["status"] == "completed"
        assert payload["model"] == GROK_WEB_SEARCH_MODEL
        assert payload["id"].startswith("resp_")
        assert isinstance(payload["created_at"], int)
        assert isinstance(payload["output"], list) and payload["output"]

    def test_output_text_carries_every_result(self):
        """
        What it does: Verifies titles, URLs and snippets all reach output_text.
        Purpose: output_text() is the only content Grok Build shows the model,
                 so nothing may be dropped.
        """
        payload = build_responses_payload(GROK_WEB_SEARCH_MODEL, "q", KIRO_MCP_RESULTS)
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
        payload = build_responses_payload(GROK_WEB_SEARCH_MODEL, "q", KIRO_MCP_RESULTS)
        text = _output_text(payload)

        assert "<web_search>" not in text
        assert "</web_search>" not in text

    def test_citations_are_extractable(self):
        """
        What it does: Runs grok-build's extract_citations logic on the payload.
        Purpose: Annotations must be readable by the real consumer, in order.
        """
        payload = build_responses_payload(GROK_WEB_SEARCH_MODEL, "q", KIRO_MCP_RESULTS)

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
        payload = build_responses_payload(GROK_WEB_SEARCH_MODEL, "q", KIRO_MCP_RESULTS)
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
        payload = build_responses_payload(GROK_WEB_SEARCH_MODEL, "q", KIRO_MCP_RESULTS)

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
        payload = build_responses_payload(GROK_WEB_SEARCH_MODEL, "q", results)
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
        payload = build_responses_payload(GROK_WEB_SEARCH_MODEL, "q", results)

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
        payload = build_responses_payload(GROK_WEB_SEARCH_MODEL, "q", results)
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
        payload = build_responses_payload(GROK_WEB_SEARCH_MODEL, "q", results)
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

        async def _fake_search(query, auth_manager):
            return "srvtoolu_fake", backend_results

        monkeypatch.setattr("kiro.routes_openai.call_kiro_mcp_api", _fake_search)

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

        async def _failing_search(query, auth_manager):
            return None, None

        monkeypatch.setattr("kiro.routes_openai.call_kiro_mcp_api", _failing_search)

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

    def test_model_list_advertises_web_search_model_with_context_window(
        self, monkeypatch
    ):
        """
        What it does: Calls the real /v1/models route and inspects the entry.
        Purpose: Grok Build reads this list to resolve its web_search model; the
                 id must be present, and contextWindow must carry the alias
                 target's real value rather than being omitted.
        """
        from unittest.mock import MagicMock

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from kiro.routes_openai import router

        app = FastAPI()
        app.include_router(router)

        resolver = MagicMock()
        resolver.get_available_models.return_value = [
            "claude-opus-4.6",
            GROK_WEB_SEARCH_MODEL,
        ]
        account = MagicMock()
        account.model_resolver = resolver
        manager = MagicMock()
        manager.get_first_account.return_value = account
        manager.get_model_context_window.side_effect = (
            lambda model_id: 1_000_000 if model_id == GROK_WEB_SEARCH_MODEL else 200_000
        )
        app.state.account_manager = manager
        app.state.account_system = False

        with TestClient(app) as client:
            response = client.get("/v1/models", headers=self._auth_headers())

        assert response.status_code == 200
        entries = {m["id"]: m for m in response.json()["data"]}
        assert GROK_WEB_SEARCH_MODEL in entries, (
            "web_search model missing -> Grok Build would fall back to xAI's "
            "endpoint and bypass this gateway"
        )
        assert entries[GROK_WEB_SEARCH_MODEL]["contextWindow"] == 1_000_000


class TestGrokBuildConfigTomlContract:
    """The client half of the fix: Grok Build's config.toml must point here.

    Advertising the model id is necessary but not sufficient. Grok Build only
    reuses this gateway's base_url when its own config names the model AND
    defines a `[model."<id>"]` block whose base_url is the gateway. These tests
    pin the TOML shape that satisfies `find_model_by_id` + `resolve_credentials`
    so a regression (or a doc copy-paste error) is caught here.
    """

    # The config.toml stanza this gateway requires. Documented in .env.example.
    REQUIRED_TOML = """
        [models]
        web_search = "grok-4.20-multi-agent"

        [model."grok-4.20-multi-agent"]
        model = "grok-4.20-multi-agent"
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
        """Mirror of grok-build find_model_by_id (config.rs:3316).

        Looks up by table key, then by the entry's `model` field.
        """
        if model_id in models:
            return models[model_id]
        for entry in models.values():
            if isinstance(entry, dict) and entry.get("model") == model_id:
                return entry
        return None

    def test_required_stanza_resolves_to_the_gateway(self):
        """
        What it does: Parses the documented stanza and runs the lookup on it.
        Purpose: Proves the config we tell users to write actually satisfies
                 find_model_by_id and yields the gateway's base_url.
        """
        cfg = self._parse(self.REQUIRED_TOML)
        ws_id = cfg["models"]["web_search"]

        entry = self._find_model_by_id(cfg.get("model", {}), ws_id)
        assert entry is not None, "find_model_by_id must locate the web_search model"
        assert entry["base_url"] == "http://localhost:8000/v1"
        # WebSearchClient::search appends /responses to base_url.
        assert entry["base_url"].rstrip("/") + "/responses" == (
            "http://localhost:8000/v1/responses"
        )

    def test_stanza_model_id_matches_gateway_default(self):
        """
        What it does: Cross-checks the stanza id against GROK_WEB_SEARCH_MODEL.
        Purpose: A mismatch between the two sides silently reintroduces the bug.
        """
        import importlib

        import kiro.config as config_module

        importlib.reload(config_module)
        cfg = self._parse(self.REQUIRED_TOML)

        assert cfg["models"]["web_search"] == config_module.GROK_WEB_SEARCH_MODEL

    def test_unquoted_key_breaks_lookup(self):
        """
        What it does: Parses the same block with an UNQUOTED table key.
        Purpose: Regression guard for a real trap hit while fixing this. The id
                 contains a dot, so `[model.grok-4.20-multi-agent]` becomes the
                 nested table model.grok-4 -> "20-multi-agent". The key must be
                 quoted or find_model_by_id fails and Grok Build silently falls
                 back to xAI's endpoint.
        """
        broken = self._parse(
            """
            [models]
            web_search = "grok-4.20-multi-agent"

            [model.grok-4.20-multi-agent]
            base_url = "http://localhost:8000/v1"
            """
        )

        # Parsed as a nested table, so the flat key does not exist...
        assert "grok-4.20-multi-agent" not in broken["model"]
        assert "grok-4" in broken["model"]
        # ...and there is no `model` field at the top level to rescue it either.
        assert self._find_model_by_id(broken["model"], "grok-4.20-multi-agent") is None

    def test_missing_base_url_would_bypass_the_gateway(self):
        """
        What it does: Parses a stanza whose model block omits base_url.
        Purpose: Documents that the entry alone is not enough -- without
                 base_url, resolve_credentials has no gateway URL to reuse.
        """
        cfg = self._parse(
            """
            [models]
            web_search = "grok-4.20-multi-agent"

            [model."grok-4.20-multi-agent"]
            model = "grok-4.20-multi-agent"
            """
        )
        entry = self._find_model_by_id(cfg["model"], cfg["models"]["web_search"])

        assert entry is not None
        assert "base_url" not in entry, (
            "a model entry without base_url cannot route the search to the gateway"
        )

    def test_env_example_documents_the_web_search_model(self):
        """
        What it does: Greps .env.example for the GROK_WEB_SEARCH_MODEL guidance.
        Purpose: The client-side config.toml requirement is invisible from the
                 gateway alone, so the docs must state it.
        """
        from pathlib import Path

        text = Path(".env.example").read_text(encoding="utf-8")

        assert "GROK_WEB_SEARCH_MODEL" in text
        assert "/v1/responses" in text

    def test_readme_toml_example_is_valid_and_resolves(self):
        """
        What it does: Extracts the TOML block from README.md, parses it, and
                      runs the same find_model_by_id lookup against it.
        Purpose: Users copy-paste this block. If it drifts (or loses the quotes
                 around the dotted key) their web_search silently goes to xAI,
                 so the doc example must stay executable.
        """
        import importlib
        import re
        from pathlib import Path

        import kiro.config as config_module

        importlib.reload(config_module)
        readme = Path("README.md").read_text(encoding="utf-8")

        blocks = [
            block
            for block in re.findall(r"```toml\n(.*?)```", readme, re.DOTALL)
            if "web_search" in block
        ]
        assert blocks, "README must document the config.toml web_search stanza"

        cfg = self._parse(blocks[0])
        ws_id = cfg["models"]["web_search"]
        assert ws_id == config_module.GROK_WEB_SEARCH_MODEL

        entry = self._find_model_by_id(cfg.get("model", {}), ws_id)
        assert entry is not None, (
            "README example must survive find_model_by_id -- check the dotted "
            "table key is quoted"
        )
        assert entry["base_url"].startswith("http://localhost:8000")


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

        # Direct hit still works.
        assert manager.get_model_context_window(real_model) == 1_000_000
        # Alias resolves through the resolver, then hits the cache.
        assert manager.get_model_context_window(GROK_WEB_SEARCH_MODEL) == 1_000_000

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
