# -*- coding: utf-8 -*-

"""
Web search tag parser for streaming responses.

Detects <web_search>...</web_search> tags in Grok model output (via Kiro API)
and parses the structured text into search results.

Grok outputs search results as plain text wrapped in XML tags:

    <web_search> Search results for "query":

    1. Title: Example
       Published: 10 Jul 2026 00:10:40
       URL: https://example.com
       Snippet text here.

    2. Title: Another
       ...
    </web_search>

This parser buffers the tag content, parses it into structured results,
and signals the streaming layer to emit proper SSE content blocks.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger


@dataclass
class WebSearchResult:
    """Single search result entry."""
    title: str = ""
    url: str = ""
    published: str = ""
    snippet: str = ""


@dataclass
class WebSearchParseResult:
    """Result of feeding content to the web search parser."""
    # 要直接輸出的文字（tag 之前或之後的部分）
    regular_content: Optional[str] = None
    # 解析完成的搜尋結果
    search_results: Optional[List[WebSearchResult]] = None
    # 搜尋 query（從 tag 內容提取）
    query: Optional[str] = None
    # 生成的 tool_use_id
    tool_use_id: Optional[str] = None
    # tag 是否仍在 buffer 中（尚未關閉）
    buffering: bool = False


class WebSearchParser:
    """
    Detects and parses <web_search>...</web_search> tags in streaming text.

    Unlike ThinkingParser (which only detects at response start), this parser
    can detect <web_search> tags anywhere in the response since Grok may
    output text before and after search results.
    """

    OPEN_TAG = "<web_search>"
    CLOSE_TAG = "</web_search>"

    def __init__(self):
        self._buffer = ""
        self._in_tag = False
        # buffer 尾端可能是不完整的 tag 開頭
        self._tail_buffer = ""

    def feed(self, content: str) -> WebSearchParseResult:
        """
        Process a chunk of streaming content.

        Returns:
            WebSearchParseResult indicating what to do with the content.
        """
        if not content:
            return WebSearchParseResult()

        text = self._tail_buffer + content
        self._tail_buffer = ""

        if self._in_tag:
            return self._handle_in_tag(text)
        else:
            return self._handle_outside_tag(text)

    def _handle_outside_tag(self, text: str) -> WebSearchParseResult:
        """Look for opening tag in text."""
        idx = text.find(self.OPEN_TAG)

        if idx != -1:
            # Found opening tag
            before = text[:idx]
            after = text[idx + len(self.OPEN_TAG):]
            self._in_tag = True
            self._buffer = after

            # Check if closing tag is already in buffer
            close_idx = self._buffer.find(self.CLOSE_TAG)
            if close_idx != -1:
                return self._complete_tag(before)

            return WebSearchParseResult(
                regular_content=before if before else None,
                buffering=True,
            )

        # No opening tag found — but tail might be partial "<web_search"
        # Buffer potential partial tag at the end
        potential_start = self._find_partial_tag_start(text)
        if potential_start is not None:
            self._tail_buffer = text[potential_start:]
            emit = text[:potential_start]
            return WebSearchParseResult(
                regular_content=emit if emit else None,
            )

        return WebSearchParseResult(regular_content=text)

    def _handle_in_tag(self, text: str) -> WebSearchParseResult:
        """Accumulate content inside the tag, looking for close."""
        self._buffer += text

        close_idx = self._buffer.find(self.CLOSE_TAG)
        if close_idx != -1:
            return self._complete_tag(None)

        return WebSearchParseResult(buffering=True)

    def _complete_tag(self, before_text: Optional[str]) -> WebSearchParseResult:
        """Tag closed — parse results."""
        close_idx = self._buffer.find(self.CLOSE_TAG)
        tag_content = self._buffer[:close_idx]
        after_close = self._buffer[close_idx + len(self.CLOSE_TAG):]

        self._in_tag = False
        self._buffer = ""

        # 剩餘文字放回 tail buffer（可能還有更多 tag）
        if after_close:
            self._tail_buffer = after_close

        # 解析搜尋結果
        query, results = self._parse_search_content(tag_content)
        tool_use_id = f"srvtoolu_{uuid.uuid4().hex[:24]}"

        return WebSearchParseResult(
            regular_content=before_text if before_text else None,
            search_results=results,
            query=query or "",
            tool_use_id=tool_use_id,
            buffering=False,
        )

    def finalize(self) -> WebSearchParseResult:
        """
        Flush any remaining buffered content (e.g. unclosed tag or tail).

        If an unclosed <web_search> tag has parseable results, treat it as
        a complete search block (Grok often omits the closing tag when it
        also emits a bracket-style tool call).
        """
        if self._in_tag and self._buffer.strip():
            # Attempt to parse unclosed buffer as search results
            buf_content = self._buffer
            query, results = self._parse_search_content(buf_content)
            self._in_tag = False
            self._buffer = ""

            if results:
                tool_use_id = f"srvtoolu_{uuid.uuid4().hex[:24]}"
                remaining = self._tail_buffer
                self._tail_buffer = ""
                return WebSearchParseResult(
                    regular_content=remaining if remaining else None,
                    search_results=results,
                    query=query or "",
                    tool_use_id=tool_use_id,
                    buffering=False,
                )
            else:
                # No parseable results — emit as raw text
                remaining = self.OPEN_TAG + buf_content + self._tail_buffer
                self._tail_buffer = ""
                return WebSearchParseResult(regular_content=remaining)

        remaining = ""
        if self._in_tag:
            remaining = self.OPEN_TAG + self._buffer
            self._in_tag = False
            self._buffer = ""
        if self._tail_buffer:
            remaining += self._tail_buffer
            self._tail_buffer = ""

        if remaining:
            return WebSearchParseResult(regular_content=remaining)
        return WebSearchParseResult()

    def _find_partial_tag_start(self, text: str) -> Optional[int]:
        """
        Check if the end of text could be the start of '<web_search>'.

        Returns the index where the partial match starts, or None.
        """
        tag = self.OPEN_TAG
        # Check suffixes of text against prefixes of tag
        for i in range(1, min(len(tag), len(text)) + 1):
            if text[-i:] == tag[:i]:
                return len(text) - i
        return None

    @staticmethod
    def _parse_search_content(content: str) -> tuple:
        """
        Parse the text inside <web_search>...</web_search>.

        Returns:
            (query, list of WebSearchResult)
        """
        lines = content.strip().split("\n")
        query = ""
        results: List[WebSearchResult] = []

        # 第一行通常是 'Search results for "query":'
        if lines:
            m = re.match(r'Search results for ["“](.+?)["”]', lines[0].strip())
            if m:
                query = m.group(1)
                lines = lines[1:]

        # 解析編號結果
        current: Optional[WebSearchResult] = None
        snippet_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 新的編號項目: "1. Title: ..."
            num_match = re.match(r'^\d+\.\s+Title:\s*(.+)', stripped)
            if num_match:
                # 儲存前一個
                if current is not None:
                    current.snippet = " ".join(snippet_lines).strip()
                    results.append(current)
                    snippet_lines = []
                current = WebSearchResult(title=num_match.group(1).strip())
                continue

            if current is not None:
                if stripped.startswith("Published:"):
                    current.published = stripped[len("Published:"):].strip()
                elif stripped.startswith("URL:"):
                    current.url = stripped[len("URL:"):].strip()
                else:
                    snippet_lines.append(stripped)

        # 最後一個
        if current is not None:
            current.snippet = " ".join(snippet_lines).strip()
            results.append(current)

        return query, results
