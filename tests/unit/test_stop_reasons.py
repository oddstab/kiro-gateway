# -*- coding: utf-8 -*-

"""
Unit tests for stop_reasons module.

Tests the mapping of Kiro stop reasons to OpenAI and Anthropic formats.
"""

import pytest
from kiro.stop_reasons import (
    to_openai_finish_reason,
    to_anthropic_stop_reason,
    resolve_finish_reason,
    KIRO_END_TURN,
    KIRO_TOOL_USE,
    KIRO_MAX_TOKENS,
    KIRO_STOP_SEQUENCE,
    KIRO_CONTENT_FILTERED,
    KIRO_GUARDRAIL_INTERVENED,
)


# Keyword bundles matching how each streaming layer calls the resolver, so a
# test never silently checks the wrong API's vocabulary.
OPENAI_VALUES = {
    "truncated_value": "length",
    "tool_calls_value": "tool_calls",
    "default_value": "stop",
    "api_label": "openai",
}
ANTHROPIC_VALUES = {
    "truncated_value": "max_tokens",
    "tool_calls_value": "tool_use",
    "default_value": "end_turn",
    "api_label": "anthropic",
}


class TestToOpenaiFinishReasonSuccess:
    """Tests for to_openai_finish_reason successful mappings."""

    def test_maps_end_turn_to_stop(self):
        """
        What it does: Maps END_TURN to OpenAI stop.
        Goal: Ensure END_TURN is recognized as normal completion.
        """
        print("Setup: END_TURN reason...")
        result = to_openai_finish_reason(KIRO_END_TURN)

        print(f"Comparing result: Expected 'stop', Got '{result}'")
        assert result == "stop"

    def test_maps_tool_use_to_tool_calls(self):
        """
        What it does: Maps TOOL_USE to OpenAI tool_calls.
        Goal: Ensure TOOL_USE maps to tool completion.
        """
        print("Setup: TOOL_USE reason...")
        result = to_openai_finish_reason(KIRO_TOOL_USE)

        print(f"Comparing result: Expected 'tool_calls', Got '{result}'")
        assert result == "tool_calls"

    def test_maps_max_tokens_to_length(self):
        """
        What it does: Maps MAX_TOKENS to OpenAI length.
        Goal: Ensure MAX_TOKENS maps to length truncation.
        """
        print("Setup: MAX_TOKENS reason...")
        result = to_openai_finish_reason(KIRO_MAX_TOKENS)

        print(f"Comparing result: Expected 'length', Got '{result}'")
        assert result == "length"

    def test_maps_stop_sequence_to_stop(self):
        """
        What it does: Maps STOP_SEQUENCE to OpenAI stop.
        Goal: Ensure STOP_SEQUENCE maps correctly.
        """
        print("Setup: STOP_SEQUENCE reason...")
        result = to_openai_finish_reason(KIRO_STOP_SEQUENCE)

        print(f"Comparing result: Expected 'stop', Got '{result}'")
        assert result == "stop"

    def test_maps_content_filtered_to_content_filter(self):
        """
        What it does: Maps CONTENT_FILTERED to OpenAI content_filter.
        Goal: Ensure content filtering is recognized.
        """
        print("Setup: CONTENT_FILTERED reason...")
        result = to_openai_finish_reason(KIRO_CONTENT_FILTERED)

        print(f"Comparing result: Expected 'content_filter', Got '{result}'")
        assert result == "content_filter"

    def test_maps_guardrail_intervened_to_content_filter(self):
        """
        What it does: Maps GUARDRAIL_INTERVENED to OpenAI content_filter.
        Goal: Ensure guardrail triggers map to content_filter.
        """
        print("Setup: GUARDRAIL_INTERVENED reason...")
        result = to_openai_finish_reason(KIRO_GUARDRAIL_INTERVENED)

        print(f"Comparing result: Expected 'content_filter', Got '{result}'")
        assert result == "content_filter"


class TestToOpenaiFinishReasonEdgeCases:
    """Tests for to_openai_finish_reason edge cases."""

    def test_handles_none_input(self):
        """
        What it does: Handles None input.
        Goal: Ensure None returns None (no opinion).
        """
        print("Setup: None input...")
        result = to_openai_finish_reason(None)

        print(f"Comparing result: Expected None, Got {result}")
        assert result is None

    def test_handles_empty_string(self):
        """
        What it does: Handles empty string.
        Goal: Ensure empty string returns None.
        """
        print("Setup: Empty string...")
        result = to_openai_finish_reason("")

        print(f"Comparing result: Expected None, Got {result}")
        assert result is None

    def test_handles_whitespace_only(self):
        """
        What it does: Handles whitespace-only string.
        Goal: Ensure whitespace returns None.
        """
        print("Setup: Whitespace string...")
        result = to_openai_finish_reason("   \t\n  ")

        print(f"Comparing result: Expected None, Got {result}")
        assert result is None

    def test_handles_lowercase_input(self):
        """
        What it does: Handles lowercase input.
        Goal: Ensure case-insensitive matching.
        """
        print("Setup: Lowercase END_TURN...")
        result = to_openai_finish_reason("end_turn")

        print(f"Comparing result: Expected 'stop', Got '{result}'")
        assert result == "stop"

    def test_handles_mixed_case_input(self):
        """
        What it does: Handles mixed case input.
        Goal: Ensure mixed case is normalized.
        """
        print("Setup: Mixed case Tool_Use...")
        result = to_openai_finish_reason("Tool_Use")

        print(f"Comparing result: Expected 'tool_calls', Got '{result}'")
        assert result == "tool_calls"

    def test_handles_input_with_surrounding_whitespace(self):
        """
        What it does: Handles input with surrounding whitespace.
        Goal: Ensure whitespace is trimmed.
        """
        print("Setup: Input with whitespace...")
        result = to_openai_finish_reason("  MAX_TOKENS  ")

        print(f"Comparing result: Expected 'length', Got '{result}'")
        assert result == "length"

    def test_handles_unknown_value(self):
        """
        What it does: Handles unknown stop reason value.
        Goal: Ensure unknown values return None (no opinion).
        """
        print("Setup: Unknown value...")
        result = to_openai_finish_reason("UNKNOWN_REASON")

        print(f"Comparing result: Expected None, Got {result}")
        assert result is None

    def test_handles_typo_in_value(self):
        """
        What it does: Handles typo in stop reason.
        Goal: Ensure typos return None gracefully.
        """
        print("Setup: Typo in reason...")
        result = to_openai_finish_reason("END_TRN")  # Typo: TRN instead of TURN

        print(f"Comparing result: Expected None, Got {result}")
        assert result is None


class TestToAnthropicStopReasonSuccess:
    """Tests for to_anthropic_stop_reason successful mappings."""

    def test_maps_end_turn_to_end_turn(self):
        """
        What it does: Maps END_TURN to Anthropic end_turn.
        Goal: Ensure END_TURN maps correctly.
        """
        print("Setup: END_TURN reason...")
        result = to_anthropic_stop_reason(KIRO_END_TURN)

        print(f"Comparing result: Expected 'end_turn', Got '{result}'")
        assert result == "end_turn"

    def test_maps_tool_use_to_tool_use(self):
        """
        What it does: Maps TOOL_USE to Anthropic tool_use.
        Goal: Ensure TOOL_USE maps correctly.
        """
        print("Setup: TOOL_USE reason...")
        result = to_anthropic_stop_reason(KIRO_TOOL_USE)

        print(f"Comparing result: Expected 'tool_use', Got '{result}'")
        assert result == "tool_use"

    def test_maps_max_tokens_to_max_tokens(self):
        """
        What it does: Maps MAX_TOKENS to Anthropic max_tokens.
        Goal: Ensure MAX_TOKENS maps correctly.
        """
        print("Setup: MAX_TOKENS reason...")
        result = to_anthropic_stop_reason(KIRO_MAX_TOKENS)

        print(f"Comparing result: Expected 'max_tokens', Got '{result}'")
        assert result == "max_tokens"

    def test_maps_stop_sequence_to_stop_sequence(self):
        """
        What it does: Maps STOP_SEQUENCE to Anthropic stop_sequence.
        Goal: Ensure STOP_SEQUENCE maps correctly.
        """
        print("Setup: STOP_SEQUENCE reason...")
        result = to_anthropic_stop_reason(KIRO_STOP_SEQUENCE)

        print(f"Comparing result: Expected 'stop_sequence', Got '{result}'")
        assert result == "stop_sequence"

    def test_maps_content_filtered_to_refusal(self):
        """
        What it does: Maps CONTENT_FILTERED to Anthropic refusal.
        Goal: Ensure content filtering maps to refusal.
        """
        print("Setup: CONTENT_FILTERED reason...")
        result = to_anthropic_stop_reason(KIRO_CONTENT_FILTERED)

        print(f"Comparing result: Expected 'refusal', Got '{result}'")
        assert result == "refusal"

    def test_maps_guardrail_intervened_to_refusal(self):
        """
        What it does: Maps GUARDRAIL_INTERVENED to Anthropic refusal.
        Goal: Ensure guardrail triggers map to refusal.
        """
        print("Setup: GUARDRAIL_INTERVENED reason...")
        result = to_anthropic_stop_reason(KIRO_GUARDRAIL_INTERVENED)

        print(f"Comparing result: Expected 'refusal', Got '{result}'")
        assert result == "refusal"


class TestToAnthropicStopReasonEdgeCases:
    """Tests for to_anthropic_stop_reason edge cases."""

    def test_handles_none_input(self):
        """
        What it does: Handles None input.
        Goal: Ensure None returns None (no opinion).
        """
        print("Setup: None input...")
        result = to_anthropic_stop_reason(None)

        print(f"Comparing result: Expected None, Got {result}")
        assert result is None

    def test_handles_empty_string(self):
        """
        What it does: Handles empty string.
        Goal: Ensure empty string returns None.
        """
        print("Setup: Empty string...")
        result = to_anthropic_stop_reason("")

        print(f"Comparing result: Expected None, Got {result}")
        assert result is None

    def test_handles_whitespace_only(self):
        """
        What it does: Handles whitespace-only string.
        Goal: Ensure whitespace returns None.
        """
        print("Setup: Whitespace string...")
        result = to_anthropic_stop_reason("   \t\n  ")

        print(f"Comparing result: Expected None, Got {result}")
        assert result is None

    def test_handles_lowercase_input(self):
        """
        What it does: Handles lowercase input.
        Goal: Ensure case-insensitive matching.
        """
        print("Setup: Lowercase end_turn...")
        result = to_anthropic_stop_reason("end_turn")

        print(f"Comparing result: Expected 'end_turn', Got '{result}'")
        assert result == "end_turn"

    def test_handles_mixed_case_input(self):
        """
        What it does: Handles mixed case input.
        Goal: Ensure mixed case is normalized.
        """
        print("Setup: Mixed case Max_Tokens...")
        result = to_anthropic_stop_reason("Max_Tokens")

        print(f"Comparing result: Expected 'max_tokens', Got '{result}'")
        assert result == "max_tokens"

    def test_handles_input_with_surrounding_whitespace(self):
        """
        What it does: Handles input with surrounding whitespace.
        Goal: Ensure whitespace is trimmed.
        """
        print("Setup: Input with whitespace...")
        result = to_anthropic_stop_reason("  TOOL_USE  ")

        print(f"Comparing result: Expected 'tool_use', Got '{result}'")
        assert result == "tool_use"

    def test_handles_unknown_value(self):
        """
        What it does: Handles unknown stop reason value.
        Goal: Ensure unknown values return None (no opinion).
        """
        print("Setup: Unknown value...")
        result = to_anthropic_stop_reason("UNKNOWN_REASON")

        print(f"Comparing result: Expected None, Got {result}")
        assert result is None

    def test_handles_typo_in_value(self):
        """
        What it does: Handles typo in stop reason.
        Goal: Ensure typos return None gracefully.
        """
        print("Setup: Typo in reason...")
        result = to_anthropic_stop_reason("TOOL_USE_ERROR")  # Typo

        print(f"Comparing result: Expected None, Got {result}")
        assert result is None


class TestMappingConsistency:
    """Tests for consistency between mappings."""

    def test_both_map_all_known_reasons(self):
        """
        What it does: Verifies both functions handle the same known reasons.
        Goal: Ensure no reason is missing from either mapping.
        """
        print("Setup: Known Kiro reasons...")
        known_reasons = [
            KIRO_END_TURN,
            KIRO_TOOL_USE,
            KIRO_MAX_TOKENS,
            KIRO_STOP_SEQUENCE,
            KIRO_CONTENT_FILTERED,
            KIRO_GUARDRAIL_INTERVENED,
        ]

        print("Action: Checking both functions handle each reason...")
        for reason in known_reasons:
            openai_result = to_openai_finish_reason(reason)
            anthropic_result = to_anthropic_stop_reason(reason)

            print(f"  {reason}: OpenAI={openai_result}, Anthropic={anthropic_result}")
            assert openai_result is not None, f"OpenAI doesn't map {reason}"
            assert anthropic_result is not None, f"Anthropic doesn't map {reason}"

        print("✓ All known reasons are mapped by both functions")

    def test_both_return_none_for_unknown(self):
        """
        What it does: Both functions return None for unknown reasons.
        Goal: Ensure consistent "no opinion" behavior.
        """
        print("Setup: Unknown reason...")
        unknown = "UNKNOWN_FUTURE_REASON"

        print("Action: Checking both return None...")
        openai_result = to_openai_finish_reason(unknown)
        anthropic_result = to_anthropic_stop_reason(unknown)

        print(f"Comparing: OpenAI returned {openai_result}, Anthropic returned {anthropic_result}")
        assert openai_result is None
        assert anthropic_result is None


class TestResolveFinishReasonPrecedence:
    """Tests for the precedence order implemented by resolve_finish_reason."""

    @pytest.mark.parametrize("values,expected", [
        (OPENAI_VALUES, "length"),
        (ANTHROPIC_VALUES, "max_tokens"),
    ])
    def test_local_truncation_overrides_upstream_end_turn(self, values, expected):
        """
        What it does: Local truncation beats an upstream END_TURN.
        Goal: A stream that died without completion signals is ground truth,
              so a cheerful upstream stop reason must not mask it.
        """
        print("Setup: upstream says END_TURN but the gateway detected truncation...")
        result = resolve_finish_reason(
            KIRO_END_TURN, was_truncated=True, has_tool_calls=False, **values
        )

        print(f"Comparing result: Expected '{expected}', Got '{result}'")
        assert result == expected

    @pytest.mark.parametrize("values,expected", [
        (OPENAI_VALUES, "length"),
        (ANTHROPIC_VALUES, "max_tokens"),
    ])
    def test_upstream_max_tokens_is_reported_without_local_truncation(self, values, expected):
        """
        What it does: Upstream MAX_TOKENS surfaces even when the gateway saw a
                      structurally complete stream.
        Goal: This is the case the whole change exists for — clients must be
              told the answer was cut off instead of seeing a clean finish.
        """
        print("Setup: upstream MAX_TOKENS, no local truncation...")
        result = resolve_finish_reason(
            KIRO_MAX_TOKENS, was_truncated=False, has_tool_calls=False, **values
        )

        print(f"Comparing result: Expected '{expected}', Got '{result}'")
        assert result == expected

    @pytest.mark.parametrize("values,expected", [
        (OPENAI_VALUES, "content_filter"),
        (ANTHROPIC_VALUES, "refusal"),
    ])
    def test_upstream_content_filter_is_reported(self, values, expected):
        """
        What it does: CONTENT_FILTERED reaches the client as a filter signal.
        Goal: A deliberately content-less turn must be distinguishable from a
              degenerate empty one, or clients retry it pointlessly.
        """
        print("Setup: upstream CONTENT_FILTERED...")
        result = resolve_finish_reason(
            KIRO_CONTENT_FILTERED, was_truncated=False, has_tool_calls=False, **values
        )

        print(f"Comparing result: Expected '{expected}', Got '{result}'")
        assert result == expected


class TestResolveFinishReasonToolCallGuard:
    """Tests that a response carrying tool calls never loses its tool reason."""

    @pytest.mark.parametrize("upstream", [
        KIRO_END_TURN,
        KIRO_MAX_TOKENS,
        KIRO_STOP_SEQUENCE,
        KIRO_CONTENT_FILTERED,
        KIRO_GUARDRAIL_INTERVENED,
    ])
    @pytest.mark.parametrize("values,expected", [
        (OPENAI_VALUES, "tool_calls"),
        (ANTHROPIC_VALUES, "tool_use"),
    ])
    def test_tool_calls_win_over_any_disagreeing_upstream_value(
        self, upstream, values, expected
    ):
        """
        What it does: Tool calls present + upstream disagreeing → tool reason.
        Goal: Clients stop executing tools when the finish reason is not the
              tool one, so this guard protects the agent loop itself.
        """
        print(f"Setup: tool calls present, upstream says {upstream}...")
        result = resolve_finish_reason(
            upstream, was_truncated=False, has_tool_calls=True, **values
        )

        print(f"Comparing result: Expected '{expected}', Got '{result}'")
        assert result == expected

    @pytest.mark.parametrize("values,expected", [
        (OPENAI_VALUES, "tool_calls"),
        (ANTHROPIC_VALUES, "tool_use"),
    ])
    def test_upstream_tool_use_agrees_with_tool_calls(self, values, expected):
        """
        What it does: Upstream TOOL_USE with tool calls resolves to the tool value.
        Goal: Confirm the agreeing case takes the normal path, not the guard.
        """
        print("Setup: tool calls present, upstream agrees (TOOL_USE)...")
        result = resolve_finish_reason(
            KIRO_TOOL_USE, was_truncated=False, has_tool_calls=True, **values
        )

        print(f"Comparing result: Expected '{expected}', Got '{result}'")
        assert result == expected

    @pytest.mark.parametrize("values,expected", [
        (OPENAI_VALUES, "length"),
        (ANTHROPIC_VALUES, "max_tokens"),
    ])
    def test_truncation_still_wins_when_tool_calls_present(self, values, expected):
        """
        What it does: Local truncation outranks even the tool-call guard.
        Goal: Truncated tool arguments are unusable, so the client must learn
              the turn was cut off rather than try to execute a partial call.
        """
        print("Setup: truncated stream that also produced tool calls...")
        result = resolve_finish_reason(
            KIRO_TOOL_USE, was_truncated=True, has_tool_calls=True, **values
        )

        print(f"Comparing result: Expected '{expected}', Got '{result}'")
        assert result == expected


class TestResolveFinishReasonFallback:
    """Tests that an absent or unusable upstream value changes nothing."""

    @pytest.mark.parametrize("upstream", [None, "", "   ", "UNKNOWN_FUTURE_REASON", "end turn"])
    @pytest.mark.parametrize("values,expected", [
        (OPENAI_VALUES, "stop"),
        (ANTHROPIC_VALUES, "end_turn"),
    ])
    def test_unusable_upstream_falls_back_to_default(self, upstream, values, expected):
        """
        What it does: None/empty/unknown upstream → the caller's default value.
        Goal: Regression guard. Behavior with no usable upstream value must be
              byte-for-byte what the gateway did before this feature existed.
        """
        print(f"Setup: unusable upstream value {upstream!r}, no tool calls...")
        result = resolve_finish_reason(
            upstream, was_truncated=False, has_tool_calls=False, **values
        )

        print(f"Comparing result: Expected '{expected}', Got '{result}'")
        assert result == expected

    @pytest.mark.parametrize("upstream", [None, "", "UNKNOWN_FUTURE_REASON"])
    @pytest.mark.parametrize("values,expected", [
        (OPENAI_VALUES, "tool_calls"),
        (ANTHROPIC_VALUES, "tool_use"),
    ])
    def test_unusable_upstream_falls_back_to_tool_inference(
        self, upstream, values, expected
    ):
        """
        What it does: Unusable upstream + tool calls → the tool reason.
        Goal: Regression guard for the other branch of the old inference.
        """
        print(f"Setup: unusable upstream value {upstream!r}, tool calls present...")
        result = resolve_finish_reason(
            upstream, was_truncated=False, has_tool_calls=True, **values
        )

        print(f"Comparing result: Expected '{expected}', Got '{result}'")
        assert result == expected

    @pytest.mark.parametrize("upstream,expected", [
        ("max_tokens", "length"),
        ("  MAX_TOKENS  ", "length"),
        ("Max_Tokens", "length"),
    ])
    def test_upstream_value_hygiene_is_tolerated(self, upstream, expected):
        """
        What it does: Case and whitespace variations still resolve.
        Goal: Upstream string hygiene is not guaranteed; a stray space must not
              silently downgrade a truncation signal to a clean finish.
        """
        print(f"Setup: upstream value {upstream!r}...")
        result = resolve_finish_reason(
            upstream, was_truncated=False, has_tool_calls=False, **OPENAI_VALUES
        )

        print(f"Comparing result: Expected '{expected}', Got '{result}'")
        assert result == expected

    def test_never_returns_none(self):
        """
        What it does: Every input combination yields a usable string.
        Goal: Callers put this value straight on the wire, so None would
              produce a malformed response instead of a wrong-but-valid one.
        """
        print("Setup: sweeping every flag combination and several upstreams...")
        upstreams = [None, "", "WEIRD", KIRO_END_TURN, KIRO_TOOL_USE, KIRO_MAX_TOKENS]

        for values in (OPENAI_VALUES, ANTHROPIC_VALUES):
            for upstream in upstreams:
                for was_truncated in (True, False):
                    for has_tool_calls in (True, False):
                        result = resolve_finish_reason(
                            upstream,
                            was_truncated=was_truncated,
                            has_tool_calls=has_tool_calls,
                            **values,
                        )
                        assert isinstance(result, str) and result, (
                            f"empty result for upstream={upstream!r}, "
                            f"truncated={was_truncated}, tools={has_tool_calls}"
                        )

        print("✓ All combinations returned a non-empty string")

    def test_openai_and_anthropic_vocabularies_never_leak(self):
        """
        What it does: Each API only ever receives its own vocabulary.
        Goal: A copy-paste slip at a call site would send e.g. 'end_turn' to an
              OpenAI client, which clients reject or misread.
        """
        print("Setup: collecting every resolvable value per API...")
        openai_allowed = {"stop", "length", "tool_calls", "content_filter"}
        anthropic_allowed = {"end_turn", "max_tokens", "tool_use", "stop_sequence", "refusal"}
        upstreams = [
            None, "", "WEIRD", KIRO_END_TURN, KIRO_TOOL_USE, KIRO_MAX_TOKENS,
            KIRO_STOP_SEQUENCE, KIRO_CONTENT_FILTERED, KIRO_GUARDRAIL_INTERVENED,
        ]

        for values, allowed in ((OPENAI_VALUES, openai_allowed), (ANTHROPIC_VALUES, anthropic_allowed)):
            for upstream in upstreams:
                for was_truncated in (True, False):
                    for has_tool_calls in (True, False):
                        result = resolve_finish_reason(
                            upstream,
                            was_truncated=was_truncated,
                            has_tool_calls=has_tool_calls,
                            **values,
                        )
                        assert result in allowed, (
                            f"{values['api_label']} got out-of-vocabulary "
                            f"'{result}' for upstream={upstream!r}"
                        )

        print("✓ No vocabulary leaked between the two APIs")
