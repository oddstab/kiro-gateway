# -*- coding: utf-8 -*-

"""
Streaming-level leak tests for the <web_search> tag.

這些測試驅動「真正的」串流 generator（stream_kiro_to_openai_internal /
stream_kiro_to_anthropic），只把 httpx transport（byte 來源）換成假的。
Kiro 的 content 事件是靠掃描 `{"content":"..."}` JSON pattern 抓取的
（見 kiro/parsers.py AwsEventStreamParser.EVENT_PATTERNS），所以我們用
json.dumps 造出貨真價實的上游 byte stream，parser 與 generator 全程真跑。

驗證重點：無論 Grok 吐的 <web_search> 區塊是否閉合，客戶端輸出裡都
不得出現字面 `<web_search>` / `</web_search>` 標籤。
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from kiro.streaming_openai import stream_kiro_to_openai_internal
from kiro.streaming_anthropic import stream_kiro_to_anthropic


# 回報截圖的真實樣本：未閉合 tag、開頭空格、粗體標題、彎引號 query、中文。
REAL_LEAKED_SAMPLE = (
    '<web_search> Search results for "\u7570\u74b0 \u6700\u65b0\u89d2\u8272":\n\n'
    '1. Title: **\u7570\u74b0 Patches and Updates \u00b7 SteamDB**\n'
    '   Published: 09 Jul 2026 19:26:06\n'
    '   URL: https://steamdb.info/app/4706890/patchnotes/\n'
    '   NTE is a supernatural open-world RPG developed by Hotta Studio.\n\n'
    '2. Title: **Download \u7570\u74b0 on PC with MEmu**\n'
    '   Published: 10 Apr 2026 08:00:00\n'
    '   URL: https://www.memuplay.com/how-to-play.html\n'
    '   The all-new MEmu 9 is the best way to play.\n'
)


def _make_kiro_bytes(text: str) -> list:
    """把一段模型輸出文字切成兩個 Kiro content 事件的 byte 區塊。

    模擬串流：故意在 <web_search> 標籤中間切開，驗證跨 chunk 也不洩漏。
    """
    split = len(text) // 2
    return [
        json.dumps({"content": text[:split]}).encode("utf-8"),
        json.dumps({"content": text[split:]}).encode("utf-8"),
    ]


def _fake_response(byte_chunks: list):
    """造一個假的 httpx.Response，只實作 parse_kiro_stream 會用到的介面。"""
    response = AsyncMock()
    response.status_code = 200
    response.aclose = AsyncMock()

    async def _aiter_bytes():
        for c in byte_chunks:
            yield c

    # aiter_bytes 必須是「回傳 async iterator 的一般函式」，不是 coroutine
    response.aiter_bytes = lambda *a, **k: _aiter_bytes()
    return response


def _mock_cache():
    cache = MagicMock()
    cache.get_max_input_tokens.return_value = 200000
    return cache


@pytest.mark.asyncio
async def test_openai_stream_no_raw_web_search_tag_leak():
    """OpenAI 串流：真實未閉合樣本，輸出零標籤洩漏且有格式化搜尋文字。"""
    response = _fake_response(_make_kiro_bytes(REAL_LEAKED_SAMPLE))
    out = []
    async for chunk in stream_kiro_to_openai_internal(
        AsyncMock(), response, "claude-sonnet-4", _mock_cache(), MagicMock(),
    ):
        out.append(chunk)
    blob = "".join(out)

    assert "<web_search>" not in blob
    assert "</web_search>" not in blob
    # OpenAI 路徑把搜尋結果格式化成文字內容
    assert "Search results for" in blob
    assert "steamdb.info" in blob
    # 粗體標記不該外洩到標題
    assert "**\u7570\u74b0 Patches" not in blob


@pytest.mark.asyncio
async def test_anthropic_stream_no_raw_web_search_tag_leak():
    """Anthropic 串流：真實未閉合樣本，輸出零標籤洩漏且產生結構化搜尋區塊。"""
    response = _fake_response(_make_kiro_bytes(REAL_LEAKED_SAMPLE))
    out = []
    async for chunk in stream_kiro_to_anthropic(
        response, "claude-sonnet-4", _mock_cache(), MagicMock(),
    ):
        out.append(chunk)
    blob = "".join(out)

    assert "<web_search>" not in blob
    assert "</web_search>" not in blob
    # Anthropic 路徑產生結構化 server_tool_use + web_search_tool_result 區塊
    assert "server_tool_use" in blob
    assert "web_search_tool_result" in blob


@pytest.mark.asyncio
async def test_openai_stream_unclosed_unparseable_no_leak():
    """OpenAI 串流：未閉合且無法解析成結果的區塊也不得洩漏原始標籤。"""
    response = _fake_response(_make_kiro_bytes("<web_search> Searching the web, please wait"))
    out = []
    async for chunk in stream_kiro_to_openai_internal(
        AsyncMock(), response, "claude-sonnet-4", _mock_cache(), MagicMock(),
    ):
        out.append(chunk)
    blob = "".join(out)

    assert "<web_search>" not in blob
    assert "</web_search>" not in blob


# ==================================================================================================
# Path B 攔截 vs. 客戶端自帶 web_search
#
# 這是使用者截圖裡標籤外洩的真正來源。閘道器會自動注入一個 web_search 工具，
# 模型呼叫它，Path B 再攔截並把 generate_search_summary() 的「帶標籤」文字直接
# 塞進回應內容 —— 那段文字是在 WebSearchParser 之後才注入的，所以 parser 完全
# 攔不到。Grok Build 本身就有 client-side web_search（打 POST /v1/responses），
# 因此閘道器必須讓路。
#
# 這些測試驅動真正的 generator，只替換搜尋後端。
# ==================================================================================================

# Grok Build 送出的形態：web_search 是它的 client-side function tool 之一
GROK_BUILD_TOOLS = [
    {"type": "function", "function": {"name": "read_file", "parameters": {}}},
    {"type": "function", "function": {"name": "web_search", "parameters": {}}},
]
ANTHROPIC_CLIENT_TOOLS = [
    {"name": "read_file", "input_schema": {}},
    {"name": "web_search", "input_schema": {}},
]

BACKEND_RESULTS = {
    "results": [
        {
            "title": "LEY DE CONCURSOS Y QUIEBRAS",
            "url": "https://servicios.infoleg.gob.ar/x",
            "snippet": "Sancionada: Julio 20 de 1995.",
            "publishedDate": 1548115200000,
        }
    ],
    "totalResults": 1,
}


def _make_tool_call_bytes(query: str) -> list:
    """造出上游的 web_search tool call 事件 byte（Path B 的觸發來源）。

    AwsEventStreamParser 靠掃描 `{"name":...}` / `{"stop":...}` 抓 tool call
    （見 kiro/parsers.py EVENT_PATTERNS），這裡直接產生那兩個 frame。
    """
    return [
        json.dumps(
            {"name": "web_search", "toolUseId": "tooluse_x", "input": {"query": query}}
        ).encode("utf-8"),
        json.dumps({"stop": True}).encode("utf-8"),
    ]


@pytest.fixture
def fake_search_backend(monkeypatch):
    """Replace the configured provider while exercising real stream code."""
    async def _fake_search(query, auth_manager=None):
        return "srvtoolu_x", BACKEND_RESULTS

    monkeypatch.setattr("kiro.web_search_provider.call_web_search", _fake_search)


@pytest.mark.asyncio
async def test_openai_client_web_search_tool_is_not_hijacked(fake_search_backend):
    """OpenAI：客戶端自帶 web_search 時，不攔截、不洩漏標籤、原樣傳回 tool call。

    這條就是使用者截圖的回歸測試。
    """
    out = []
    async for chunk in stream_kiro_to_openai_internal(
        AsyncMock(), _fake_response(_make_tool_call_bytes("LCQ 是什麼 縮寫")),
        "kiro-search-proxy", _mock_cache(), MagicMock(),
        request_tools=GROK_BUILD_TOOLS,
    ):
        out.append(chunk)
    blob = "".join(out)

    # 零標籤外洩
    assert "<web_search>" not in blob
    assert "</web_search>" not in blob
    # 閘道器沒有自己去搜尋
    assert "LEY DE CONCURSOS" not in blob
    # tool call 原樣傳回，交給 Grok Build 自己執行（POST /v1/responses）
    assert "tool_calls" in blob
    assert "web_search" in blob


@pytest.mark.asyncio
async def test_anthropic_client_web_search_tool_is_not_hijacked(fake_search_backend):
    """Anthropic：與 OpenAI 路徑行為一致（專案要求兩個 API 對齊）。"""
    out = []
    async for chunk in stream_kiro_to_anthropic(
        _fake_response(_make_tool_call_bytes("LCQ")), "claude-opus-4.6",
        _mock_cache(), MagicMock(), request_tools=ANTHROPIC_CLIENT_TOOLS,
    ):
        out.append(chunk)
    blob = "".join(out)

    assert "<web_search>" not in blob
    assert "</web_search>" not in blob
    assert "LEY DE CONCURSOS" not in blob
    assert "tool_use" in blob
    assert "web_search" in blob


@pytest.mark.asyncio
async def test_openai_path_b_still_intercepts_without_client_tool(fake_search_backend):
    """OpenAI：客戶端沒帶 web_search 時，Path B 仍照舊攔截（維持相容）。

    這些客戶端沒有自己的搜尋能力，靠閘道器代跑，所以不能一併關掉。
    """
    out = []
    async for chunk in stream_kiro_to_openai_internal(
        AsyncMock(), _fake_response(_make_tool_call_bytes("LCQ")),
        "claude-opus-4.6", _mock_cache(), MagicMock(),
        request_tools=[{"type": "function", "function": {"name": "read_file"}}],
    ):
        out.append(chunk)
    blob = "".join(out)

    assert "LEY DE CONCURSOS" in blob, "Path B 應該仍為無搜尋能力的客戶端代跑搜尋"


@pytest.mark.asyncio
async def test_anthropic_path_b_still_intercepts_without_client_tool(
    fake_search_backend,
):
    """Anthropic：客戶端沒帶 web_search 時同樣維持攔截行為。"""
    out = []
    async for chunk in stream_kiro_to_anthropic(
        _fake_response(_make_tool_call_bytes("LCQ")), "claude-opus-4.6",
        _mock_cache(), MagicMock(), request_tools=[{"name": "read_file"}],
    ):
        out.append(chunk)
    blob = "".join(out)

    assert "LEY DE CONCURSOS" in blob


@pytest.mark.asyncio
async def test_openai_native_web_search_tool_type_is_respected(fake_search_backend):
    """OpenAI：原生 server-side 形態 {"type": "web_search"} 也算客戶端自帶。"""
    out = []
    async for chunk in stream_kiro_to_openai_internal(
        AsyncMock(), _fake_response(_make_tool_call_bytes("LCQ")),
        "kiro-search-proxy", _mock_cache(), MagicMock(),
        request_tools=[{"type": "web_search_20250305"}],
    ):
        out.append(chunk)
    blob = "".join(out)

    assert "<web_search>" not in blob
    assert "LEY DE CONCURSOS" not in blob
