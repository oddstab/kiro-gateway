# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/oddstab/kiro-gateway
# Copyright (C) 2025 oddstab
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Classify completed Kiro turns that have no client-visible result.

This module centralizes the decision whether a completed response is safe to
resample. It deliberately leaves truncation and deterministic upstream stops
visible to the client, which can explain and handle those outcomes directly.
"""

from typing import Any, List, Optional

from loguru import logger

from kiro.stop_reasons import (
    KIRO_CONTENT_FILTERED,
    KIRO_GUARDRAIL_INTERVENED,
    KIRO_MAX_TOKENS,
)


EMPTY_REASON_REASONING_ONLY: str = "reasoning_only"
"""The model emitted reasoning but no visible content or tool calls."""

EMPTY_REASON_NO_VISIBLE_CONTENT: str = "no_visible_content"
"""The model emitted no visible content, tool calls, or reasoning."""

_EXPLANATORY_STOP_REASONS = frozenset(
    {
        KIRO_MAX_TOKENS,
        KIRO_CONTENT_FILTERED,
        KIRO_GUARDRAIL_INTERVENED,
    }
)


def classify_turn(
    content: str,
    thinking_content: str,
    tool_calls: Optional[List[Any]],
    kiro_stop_reason: Optional[str],
    was_truncated: bool,
) -> Optional[str]:
    """
    Classify whether a completed turn should be resampled for an empty result.

    Decision order preserves outcomes that the client can interpret:

    1. Visible content or tool calls are already usable and must be delivered.
    2. Locally detected truncation must remain a token-limit result, rather
       than being hidden behind a resample.
    3. Definite upstream stop reasons explain a deterministic outcome, so
       resampling would create a retry storm.
    4. Reasoning without a visible result is a retryable reasoning-only turn.
    5. All other empty turns have no visible content at all.

    Args:
        content: Client-visible text assembled from the model response.
        thinking_content: Reasoning text assembled from the model response.
        tool_calls: Tool calls assembled from the model response, if any.
        kiro_stop_reason: Raw upstream Kiro stop reason, if supplied.
        was_truncated: Whether the gateway detected stream truncation.

    Returns:
        A retryable empty-response reason, or None when the turn must be
        delivered to the client unchanged.

    Examples:
        >>> classify_turn("", "analysis", [], "END_TURN", False)
        'reasoning_only'
        >>> classify_turn("answer", "", [], None, False) is None
        True
    """
    # Visible text or a tool call is a usable result and must not be resampled.
    if content.strip() or tool_calls:
        return None

    # Local truncation must reach the client as a token-limit outcome.
    if was_truncated:
        return None

    normalized_stop_reason = (
        kiro_stop_reason.strip().upper() if kiro_stop_reason else ""
    )

    # Deterministic upstream stops need to remain visible instead of retried.
    if normalized_stop_reason in _EXPLANATORY_STOP_REASONS:
        return None

    # Reasoning without visible output is retryable, unlike a useful response.
    if thinking_content.strip():
        return EMPTY_REASON_REASONING_ONLY

    return EMPTY_REASON_NO_VISIBLE_CONTENT


class EmptyResponseError(Exception):
    """
    Signal a retryable completed response with no client-visible result.

    Args:
        reason: Classifier reason for the degenerate response.
        had_reasoning: Whether the response contained non-whitespace reasoning.
        content_len: Length of client-visible content before classification.
        tool_call_count: Number of assembled tool calls before classification.
        kiro_stop_reason: Raw upstream Kiro stop reason, if supplied.
        completion_tokens: Upstream completion token count, if supplied.
        model: Model identifier used for the response, if supplied.
    """

    def __init__(
        self,
        reason: str,
        had_reasoning: Optional[bool] = None,
        content_len: Optional[int] = None,
        tool_call_count: Optional[int] = None,
        kiro_stop_reason: Optional[str] = None,
        completion_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Initialize the retry signal with diagnostics for a single log line.

        Args:
            reason: Classifier reason for the degenerate response.
            had_reasoning: Whether the response contained non-whitespace reasoning.
            content_len: Length of client-visible content before classification.
            tool_call_count: Number of assembled tool calls before classification.
            kiro_stop_reason: Raw upstream Kiro stop reason, if supplied.
            completion_tokens: Upstream completion token count, if supplied.
            model: Model identifier used for the response, if supplied.
        """
        self.reason: str = reason
        self.had_reasoning: Optional[bool] = had_reasoning
        self.content_len: Optional[int] = content_len
        self.tool_call_count: Optional[int] = tool_call_count
        self.kiro_stop_reason: Optional[str] = kiro_stop_reason
        self.completion_tokens: Optional[int] = completion_tokens
        self.model: Optional[str] = model

        message = (
            "Empty response detected "
            f"(reason={reason!r}, had_reasoning={had_reasoning!r}, "
            f"content_len={content_len!r}, tool_call_count={tool_call_count!r}, "
            f"kiro_stop_reason={kiro_stop_reason!r}, "
            f"completion_tokens={completion_tokens!r}, model={model!r})"
        )
        super().__init__(message)


def log_empty_response(
    reason: str,
    had_reasoning: Optional[bool] = None,
    content_len: Optional[int] = None,
    tool_call_count: Optional[int] = None,
    kiro_stop_reason: Optional[str] = None,
    completion_tokens: Optional[int] = None,
    model: Optional[str] = None,
    attempt: Optional[int] = None,
    retry_budget: Optional[int] = None,
) -> None:
    """
    Emit one consistent error log line for a retryable empty response.

    Args:
        reason: Classifier reason for the degenerate response.
        had_reasoning: Whether the response contained non-whitespace reasoning.
        content_len: Length of client-visible content before classification.
        tool_call_count: Number of assembled tool calls before classification.
        kiro_stop_reason: Raw upstream Kiro stop reason, if supplied.
        completion_tokens: Upstream completion token count, if supplied.
        model: Model identifier used for the response, if supplied.
        attempt: One-based completed generation attempt number, if known.
        retry_budget: Number of additional resamples allowed, if known.

    Returns:
        None. The helper only emits a log entry.
    """
    resample_follows: Optional[bool] = (
        attempt <= retry_budget
        if attempt is not None and retry_budget is not None
        else None
    )

    logger.error(
        "Empty response detected: reason={reason!r}, "
        "had_reasoning={had_reasoning!r}, content_len={content_len!r}, "
        "tool_call_count={tool_call_count!r}, "
        "kiro_stop_reason={kiro_stop_reason!r}, "
        "completion_tokens={completion_tokens!r}, model={model!r}, "
        "attempt={attempt!r}, retry_budget={retry_budget!r}, "
        "resample_follows={resample_follows!r}",
        reason=reason,
        had_reasoning=had_reasoning,
        content_len=content_len,
        tool_call_count=tool_call_count,
        kiro_stop_reason=kiro_stop_reason,
        completion_tokens=completion_tokens,
        model=model,
        attempt=attempt,
        retry_budget=retry_budget,
        resample_follows=resample_follows,
    )
