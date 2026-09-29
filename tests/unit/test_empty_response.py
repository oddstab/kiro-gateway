# -*- coding: utf-8 -*-

"""
Unit tests for empty-response classification and diagnostics.
"""

from typing import Optional
from unittest.mock import patch

import pytest

from kiro.empty_response import (
    EMPTY_REASON_NO_VISIBLE_CONTENT,
    EMPTY_REASON_REASONING_ONLY,
    EmptyResponseError,
    classify_turn,
    log_empty_response,
)
from kiro.stop_reasons import (
    KIRO_CONTENT_FILTERED,
    KIRO_END_TURN,
    KIRO_GUARDRAIL_INTERVENED,
    KIRO_MAX_TOKENS,
    KIRO_STOP_SEQUENCE,
    KIRO_TOOL_USE,
)


class TestTurnClassification:
    """Tests for retryable empty-response classification."""

    def test_visible_content_is_delivered_without_resampling(
        self: "TestTurnClassification",
    ) -> None:
        """
        What it does: Classifies a response with visible content.
        Purpose: Ensure useful client-visible text is never retried.
        """
        result = classify_turn(
            content="Visible answer",
            thinking_content="hidden reasoning",
            tool_calls=[],
            kiro_stop_reason=None,
            was_truncated=False,
        )

        assert result is None

    def test_whitespace_content_is_not_treated_as_visible(
        self: "TestTurnClassification",
    ) -> None:
        """
        What it does: Classifies whitespace-only visible content.
        Purpose: Ensure whitespace cannot suppress an empty-response retry.
        """
        result = classify_turn(
            content=" \t\n ",
            thinking_content="",
            tool_calls=[],
            kiro_stop_reason=None,
            was_truncated=False,
        )

        assert result == EMPTY_REASON_NO_VISIBLE_CONTENT

    def test_tool_calls_are_delivered_without_resampling(
        self: "TestTurnClassification",
    ) -> None:
        """
        What it does: Classifies an empty-text response carrying a tool call.
        Purpose: Ensure executable tool calls remain visible to the client.
        """
        result = classify_turn(
            content="",
            thinking_content="",
            tool_calls=[{"id": "call_1"}],
            kiro_stop_reason=None,
            was_truncated=False,
        )

        assert result is None

    def test_truncated_response_is_delivered_without_resampling(
        self: "TestTurnClassification",
    ) -> None:
        """
        What it does: Classifies an otherwise degenerate truncated response.
        Purpose: Preserve the client-visible token-limit finish reason.
        """
        result = classify_turn(
            content="",
            thinking_content="hidden reasoning",
            tool_calls=[],
            kiro_stop_reason=KIRO_END_TURN,
            was_truncated=True,
        )

        assert result is None

    @pytest.mark.parametrize(
        "kiro_stop_reason",
        [
            KIRO_MAX_TOKENS,
            KIRO_MAX_TOKENS.lower(),
            f" {KIRO_MAX_TOKENS} ",
            f" {KIRO_MAX_TOKENS.lower()} ",
            KIRO_CONTENT_FILTERED,
            KIRO_CONTENT_FILTERED.lower(),
            f" {KIRO_CONTENT_FILTERED} ",
            f" {KIRO_CONTENT_FILTERED.lower()} ",
            KIRO_GUARDRAIL_INTERVENED,
            KIRO_GUARDRAIL_INTERVENED.lower(),
            f" {KIRO_GUARDRAIL_INTERVENED} ",
            f" {KIRO_GUARDRAIL_INTERVENED.lower()} ",
        ],
    )
    def test_explanatory_upstream_stop_reasons_are_not_resampled(
        self: "TestTurnClassification",
        kiro_stop_reason: str,
    ) -> None:
        """
        What it does: Classifies explanatory Kiro stop reasons in mixed formatting.
        Purpose: Avoid retry storms for deterministic upstream outcomes.
        """
        result = classify_turn(
            content="",
            thinking_content="hidden reasoning",
            tool_calls=[],
            kiro_stop_reason=kiro_stop_reason,
            was_truncated=False,
        )

        assert result is None

    @pytest.mark.parametrize(
        "kiro_stop_reason",
        [
            KIRO_END_TURN,
            KIRO_TOOL_USE,
            KIRO_STOP_SEQUENCE,
            "UNKNOWN_REASON",
            "",
            None,
        ],
    )
    def test_non_explanatory_stop_reasons_preserve_reasoning_only_classification(
        self: "TestTurnClassification",
        kiro_stop_reason: Optional[str],
    ) -> None:
        """
        What it does: Classifies non-deterministic or absent upstream stop reasons.
        Purpose: Ensure reasoning-only turns remain retryable in these cases.
        """
        result = classify_turn(
            content="",
            thinking_content="hidden reasoning",
            tool_calls=[],
            kiro_stop_reason=kiro_stop_reason,
            was_truncated=False,
        )

        assert result == EMPTY_REASON_REASONING_ONLY

    def test_empty_turn_without_reasoning_is_no_visible_content(
        self: "TestTurnClassification",
    ) -> None:
        """
        What it does: Classifies a response with no usable fields.
        Purpose: Differentiate total emptiness from reasoning-only output.
        """
        result = classify_turn(
            content="",
            thinking_content="",
            tool_calls=[],
            kiro_stop_reason=None,
            was_truncated=False,
        )

        assert result == EMPTY_REASON_NO_VISIBLE_CONTENT

    def test_whitespace_reasoning_is_not_reasoning_only(
        self: "TestTurnClassification",
    ) -> None:
        """
        What it does: Classifies whitespace-only reasoning with no visible output.
        Purpose: Ensure empty reasoning is not mislabeled as useful reasoning.
        """
        result = classify_turn(
            content="",
            thinking_content=" \n\t ",
            tool_calls=[],
            kiro_stop_reason=None,
            was_truncated=False,
        )

        assert result == EMPTY_REASON_NO_VISIBLE_CONTENT

    def test_none_tool_calls_are_handled_as_empty(
        self: "TestTurnClassification",
    ) -> None:
        """
        What it does: Classifies a response with no tool-call collection.
        Purpose: Ensure callers can pass None without triggering an exception.
        """
        result = classify_turn(
            content="",
            thinking_content="",
            tool_calls=None,
            kiro_stop_reason=None,
            was_truncated=False,
        )

        assert result == EMPTY_REASON_NO_VISIBLE_CONTENT

    def test_decision_order_preserves_client_interpretable_outcomes(
        self: "TestTurnClassification",
    ) -> None:
        """
        What it does: Exercises the ordered classifier precedence chain.
        Purpose: Prevent later checks from overriding a more important outcome.
        """
        assert classify_turn(
            content="Visible answer",
            thinking_content="hidden reasoning",
            tool_calls=[],
            kiro_stop_reason=KIRO_MAX_TOKENS,
            was_truncated=True,
        ) is None
        assert classify_turn(
            content="",
            thinking_content="hidden reasoning",
            tool_calls=[],
            kiro_stop_reason=KIRO_MAX_TOKENS,
            was_truncated=True,
        ) is None
        assert classify_turn(
            content="",
            thinking_content="hidden reasoning",
            tool_calls=[],
            kiro_stop_reason=KIRO_MAX_TOKENS,
            was_truncated=False,
        ) is None
        assert classify_turn(
            content="",
            thinking_content="hidden reasoning",
            tool_calls=[],
            kiro_stop_reason=KIRO_END_TURN,
            was_truncated=False,
        ) == EMPTY_REASON_REASONING_ONLY


class TestEmptyResponseError:
    """Tests for the retryable empty-response signal."""

    def test_error_preserves_diagnostics_and_renders_reason(
        self: "TestEmptyResponseError",
    ) -> None:
        """
        What it does: Creates an error with every diagnostic field.
        Purpose: Keep retry diagnostics available to routes and logs.
        """
        error = EmptyResponseError(
            reason=EMPTY_REASON_REASONING_ONLY,
            had_reasoning=True,
            content_len=0,
            tool_call_count=0,
            kiro_stop_reason=KIRO_END_TURN,
            completion_tokens=42,
            model="claude-sonnet-4.5",
        )

        assert error.reason == EMPTY_REASON_REASONING_ONLY
        assert error.had_reasoning is True
        assert error.content_len == 0
        assert error.tool_call_count == 0
        assert error.kiro_stop_reason == KIRO_END_TURN
        assert error.completion_tokens == 42
        assert error.model == "claude-sonnet-4.5"
        assert EMPTY_REASON_REASONING_ONLY in str(error)


class TestEmptyResponseLogging:
    """Tests for uniform empty-response logging."""

    @pytest.mark.parametrize(
        ("attempt", "retry_budget", "expected_resample_follows"),
        [(1, 2, True), (3, 2, False)],
    )
    def test_log_empty_response_reports_resample_decision(
        self: "TestEmptyResponseLogging",
        attempt: int,
        retry_budget: int,
        expected_resample_follows: bool,
    ) -> None:
        """
        What it does: Emits an empty-response log record at different retry points.
        Purpose: Make it clear from one line whether a resample follows.
        """
        with patch("kiro.empty_response.logger.error") as mock_error:
            log_empty_response(
                reason=EMPTY_REASON_REASONING_ONLY,
                had_reasoning=True,
                content_len=0,
                tool_call_count=0,
                kiro_stop_reason=KIRO_END_TURN,
                completion_tokens=42,
                model="claude-sonnet-4.5",
                attempt=attempt,
                retry_budget=retry_budget,
            )

        mock_error.assert_called_once()
        assert "resample_follows" in mock_error.call_args.args[0]
        assert (
            mock_error.call_args.kwargs["resample_follows"]
            is expected_resample_follows
        )
