# -*- coding: utf-8 -*-

"""
Unit tests for web_search_parser module.
"""

import pytest
from kiro.web_search_parser import WebSearchParser, WebSearchResult


class TestWebSearchParser:
    """Tests for WebSearchParser."""

    def test_passthrough_when_no_tag(self):
        """Normal text without <web_search> passes through unchanged."""
        parser = WebSearchParser()
        result = parser.feed("Hello world, no search here.")
        assert result.regular_content == "Hello world, no search here."
        assert result.search_results is None

    def test_detects_complete_tag_in_single_chunk(self):
        """Parses a complete <web_search>...</web_search> in one chunk."""
        parser = WebSearchParser()
        content = (
            'Before text. <web_search> Search results for "Python":\n\n'
            '1. Title: Python.org\n'
            '   Published: 01 Jan 2026\n'
            '   URL: https://python.org\n'
            '   The official Python site.\n\n'
            '2. Title: Python Tutorial\n'
            '   URL: https://docs.python.org/tutorial\n'
            '   Official tutorial.\n'
            '</web_search> After text.'
        )
        result = parser.feed(content)
        assert result.regular_content == "Before text. "
        assert result.search_results is not None
        assert len(result.search_results) == 2
        assert result.search_results[0].title == "Python.org"
        assert result.search_results[0].url == "https://python.org"
        assert result.search_results[0].snippet == "The official Python site."
        assert result.search_results[1].title == "Python Tutorial"
        assert result.query == "Python"
        assert result.tool_use_id.startswith("srvtoolu_")

        # After text should be in tail buffer, flush with finalize
        final = parser.finalize()
        assert final.regular_content == " After text."

    def test_tag_split_across_chunks(self):
        """Handles tag split across multiple feed() calls."""
        parser = WebSearchParser()

        r1 = parser.feed("Start. <web_")
        # Should buffer the partial tag, emit "Start. "
        assert r1.regular_content == "Start. "

        r2 = parser.feed("search> Search results for \"test\":\n\n1. Title: Result\n   URL: https://example.com\n   Snippet here.\n</web_search>")
        assert r2.search_results is not None
        assert len(r2.search_results) == 1
        assert r2.search_results[0].title == "Result"
        assert r2.query == "test"

    def test_close_tag_split_across_chunks(self):
        """Handles close tag split across chunks."""
        parser = WebSearchParser()

        r1 = parser.feed('<web_search> Search results for "q":\n\n1. Title: T\n   URL: http://x.com\n   S\n</web_')
        assert r1.buffering is True

        r2 = parser.feed("search>Done")
        assert r2.search_results is not None
        assert r2.search_results[0].title == "T"

        final = parser.finalize()
        assert final.regular_content == "Done"

    def test_unclosed_unparseable_tag_strips_markers_on_finalize(self):
        """未閉合且解析不出結果的 tag：絕不把 <web_search> 標籤原樣洩漏。

        這是回報截圖的核心症狀。舊版會原樣吐出標籤，現在必須剝掉標記，
        只保留內部文字。
        """
        parser = WebSearchParser()
        r = parser.feed("<web_search> Some content without close")
        assert r.buffering is True

        final = parser.finalize()
        assert "<web_search>" not in (final.regular_content or "")
        assert "</web_search>" not in (final.regular_content or "")
        assert "Some content without close" in final.regular_content

    def test_no_results_produces_empty_list(self):
        """Tag with no parseable results returns empty list."""
        parser = WebSearchParser()
        r = parser.feed('<web_search> Search results for "empty":\n\n</web_search>')
        assert r.search_results is not None
        assert len(r.search_results) == 0
        assert r.query == "empty"

    def test_extracts_query_with_unicode_quotes(self):
        """Handles unicode curly quotes around query."""
        parser = WebSearchParser()
        r = parser.feed('<web_search> Search results for “test query”:\n\n1. Title: R\n   URL: http://x\n   S\n</web_search>')
        assert r.query == "test query"

    def test_multiple_tags_in_sequence(self):
        """Multiple search blocks in one stream."""
        parser = WebSearchParser()

        r1 = parser.feed('Text1 <web_search> Search results for "a":\n\n1. Title: A\n   URL: http://a\n   Sa\n</web_search>')
        assert r1.regular_content == "Text1 "
        assert r1.search_results[0].title == "A"

        # finalize to flush tail
        f1 = parser.finalize()

        # In practice the parser would be reused for next chunk
        parser2 = WebSearchParser()
        r2 = parser2.feed(' Between <web_search> Search results for "b":\n\n1. Title: B\n   URL: http://b\n   Sb\n</web_search> End')
        assert r2.regular_content == " Between "
        assert r2.search_results[0].title == "B"

    # ---- 回報截圖的真實洩漏情境 ----

    # 真實 Grok 樣本：無閉合 tag、開頭空格、粗體標題、彎引號 query、中文
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

    def _collect(self, chunks):
        """跑完 feed + finalize，回傳 (洩漏文字, 結果, query)。"""
        parser = WebSearchParser()
        leaked = []
        results = []
        query = None
        for c in chunks:
            r = parser.feed(c)
            if r.regular_content:
                leaked.append(r.regular_content)
            if r.search_results is not None:
                results, query = r.search_results, r.query
        f = parser.finalize()
        if f.regular_content:
            leaked.append(f.regular_content)
        if f.search_results is not None:
            results, query = f.search_results, f.query
        return "".join(leaked), results, query

    def test_real_leaked_sample_single_chunk_no_leak(self):
        """回報樣本（未閉合）單塊餵入：零標籤洩漏且結果解析出來。"""
        leaked, results, query = self._collect([self.REAL_LEAKED_SAMPLE])
        assert "<web_search>" not in leaked and "</web_search>" not in leaked
        assert query == "\u7570\u74b0 \u6700\u65b0\u89d2\u8272"
        assert len(results) == 2
        assert results[0].title == "\u7570\u74b0 Patches and Updates \u00b7 SteamDB"
        assert results[0].url == "https://steamdb.info/app/4706890/patchnotes/"

    def test_real_leaked_sample_split_chunks_no_leak(self):
        """回報樣本被切成多塊（模擬串流）：一樣零洩漏。"""
        s = self.REAL_LEAKED_SAMPLE
        leaked, results, query = self._collect([s[:60], s[60:150], s[150:]])
        assert "<web_search>" not in leaked and "</web_search>" not in leaked
        assert len(results) == 2

    def test_markdown_bold_stripped_from_title(self):
        """標題外圍的 ** 粗體標記被剝掉。"""
        parser = WebSearchParser()
        r = parser.feed('<web_search> Search results for "q":\n\n1. Title: **Bold Title**\n   URL: http://x\n   S\n</web_search>')
        assert r.search_results[0].title == "Bold Title"

    def test_unclosed_unparseable_no_raw_tag_leak(self):
        """未閉合且無法解析成結果：剝掉標籤，不洩漏原始 <web_search>。"""
        leaked, results, _ = self._collect(['<web_search> Searching the web, please wait'])
        assert "<web_search>" not in leaked
        assert "Searching the web, please wait" in leaked
