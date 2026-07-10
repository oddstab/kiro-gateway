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

    def test_unclosed_tag_flushed_on_finalize(self):
        """Unclosed tag is emitted as raw text on finalize."""
        parser = WebSearchParser()
        r = parser.feed("<web_search> Some content without close")
        assert r.buffering is True

        final = parser.finalize()
        assert "<web_search>" in final.regular_content
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
