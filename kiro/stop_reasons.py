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
Mapping of Kiro API stop reasons to client API formats.

This module provides the single source of truth for mapping upstream Kiro
stop reason values (from metadataEvent stopReason field) to the equivalent
stop_reason/finish_reason values used by OpenAI and Anthropic APIs.

Observed Kiro stop reason values:
- END_TURN: Normal completion
- TOOL_USE: Model called a tool
- MAX_TOKENS: Hit token limit
- STOP_SEQUENCE: Hit stop sequence
- CONTENT_FILTERED: Content policy violation
- GUARDRAIL_INTERVENED: Guardrail triggered

This mapping is API-agnostic and used by both streaming layers.
"""

from typing import Optional
from loguru import logger

# Known Kiro stop reason values
KIRO_END_TURN = "END_TURN"
KIRO_TOOL_USE = "TOOL_USE"
KIRO_MAX_TOKENS = "MAX_TOKENS"
KIRO_STOP_SEQUENCE = "STOP_SEQUENCE"
KIRO_CONTENT_FILTERED = "CONTENT_FILTERED"
KIRO_GUARDRAIL_INTERVENED = "GUARDRAIL_INTERVENED"

# Kiro values that indicate the answer is short for a reason the user must be
# able to see in the logs, rather than an ordinary completion.
_NOTEWORTHY_KIRO_STOP_REASONS = frozenset({
    KIRO_MAX_TOKENS,
    KIRO_CONTENT_FILTERED,
    KIRO_GUARDRAIL_INTERVENED,
})


def to_openai_finish_reason(kiro_stop_reason: Optional[str]) -> Optional[str]:
    """
    Convert Kiro stop reason to OpenAI finish_reason.

    Maps Kiro's upstream stop reasons to the equivalent OpenAI API values.

    Args:
        kiro_stop_reason: Stop reason from Kiro metadataEvent, or None

    Returns:
        OpenAI finish_reason string, or None if mapping unavailable.
        Returns None for empty/unknown values to signal "no opinion" — caller
        falls back to existing inference logic.

    Mapping:
    - END_TURN → "stop"
    - TOOL_USE → "tool_calls"
    - MAX_TOKENS → "length"
    - STOP_SEQUENCE → "stop"
    - CONTENT_FILTERED → "content_filter"
    - GUARDRAIL_INTERVENED → "content_filter"

    Examples:
        >>> to_openai_finish_reason("END_TURN")
        'stop'
        >>> to_openai_finish_reason("TOOL_USE")
        'tool_calls'
        >>> to_openai_finish_reason("UNKNOWN_VALUE")
        None
        >>> to_openai_finish_reason(None)
        None
    """
    if not kiro_stop_reason:
        return None

    # Normalize: strip whitespace and uppercase
    normalized = kiro_stop_reason.strip().upper()

    # Map to OpenAI finish_reason
    mapping = {
        KIRO_END_TURN: "stop",
        KIRO_TOOL_USE: "tool_calls",
        KIRO_MAX_TOKENS: "length",
        KIRO_STOP_SEQUENCE: "stop",
        KIRO_CONTENT_FILTERED: "content_filter",
        KIRO_GUARDRAIL_INTERVENED: "content_filter",
    }

    result = mapping.get(normalized)

    if result is None and normalized:
        # Log unknown non-empty value for discoverability
        logger.debug(f"Unknown Kiro stop reason: {normalized}")

    return result


def to_anthropic_stop_reason(kiro_stop_reason: Optional[str]) -> Optional[str]:
    """
    Convert Kiro stop reason to Anthropic stop_reason.

    Maps Kiro's upstream stop reasons to the equivalent Anthropic API values.

    Args:
        kiro_stop_reason: Stop reason from Kiro metadataEvent, or None

    Returns:
        Anthropic stop_reason string, or None if mapping unavailable.
        Returns None for empty/unknown values to signal "no opinion" — caller
        falls back to existing inference logic.

    Mapping:
    - END_TURN → "end_turn"
    - TOOL_USE → "tool_use"
    - MAX_TOKENS → "max_tokens"
    - STOP_SEQUENCE → "stop_sequence"
    - CONTENT_FILTERED → "refusal"
    - GUARDRAIL_INTERVENED → "refusal"

    Examples:
        >>> to_anthropic_stop_reason("END_TURN")
        'end_turn'
        >>> to_anthropic_stop_reason("TOOL_USE")
        'tool_use'
        >>> to_anthropic_stop_reason("UNKNOWN_VALUE")
        None
        >>> to_anthropic_stop_reason(None)
        None
    """
    if not kiro_stop_reason:
        return None

    # Normalize: strip whitespace and uppercase
    normalized = kiro_stop_reason.strip().upper()

    # Map to Anthropic stop_reason
    mapping = {
        KIRO_END_TURN: "end_turn",
        KIRO_TOOL_USE: "tool_use",
        KIRO_MAX_TOKENS: "max_tokens",
        KIRO_STOP_SEQUENCE: "stop_sequence",
        KIRO_CONTENT_FILTERED: "refusal",
        KIRO_GUARDRAIL_INTERVENED: "refusal",
    }

    result = mapping.get(normalized)

    if result is None and normalized:
        # Log unknown non-empty value for discoverability
        logger.debug(f"Unknown Kiro stop reason: {normalized}")

    return result


def resolve_finish_reason(
    kiro_stop_reason: Optional[str],
    truncated_value: str,
    tool_calls_value: str,
    default_value: str,
    was_truncated: bool,
    has_tool_calls: bool,
    api_label: str,
) -> str:
    """
    Resolve the final finish/stop reason reported to the client.

    Single implementation of the precedence shared by both client APIs and by
    both the streaming and non-streaming paths, so the four call sites cannot
    drift apart:

    1. Locally detected truncation wins. A stream that died without completion
       signals is ground truth regardless of what upstream claimed.
    2. The mapped upstream Kiro stop reason, when it maps to a known value.
    3. The caller's existing inference (tool calls, otherwise the default).

    Args:
        kiro_stop_reason: Raw upstream Kiro stopReason, or None when absent.
        truncated_value: Client value meaning "cut off by a token limit"
            ("length" for OpenAI, "max_tokens" for Anthropic).
        tool_calls_value: Client value meaning "the model called tools"
            ("tool_calls" for OpenAI, "tool_use" for Anthropic).
        default_value: Client value for an ordinary completion ("stop" for
            OpenAI, "end_turn" for Anthropic).
        was_truncated: Whether the gateway itself detected truncation.
        has_tool_calls: Whether the assembled response carries tool calls.
        api_label: Short API name used only in log lines ("openai"/"anthropic").

    Returns:
        The finish/stop reason string to report to the client.

    Examples:
        >>> resolve_finish_reason("MAX_TOKENS", "length", "tool_calls", "stop",
        ...                       False, False, "openai")
        'length'
        >>> resolve_finish_reason(None, "length", "tool_calls", "stop",
        ...                       False, True, "openai")
        'tool_calls'
    """
    if was_truncated:
        return truncated_value

    normalized = kiro_stop_reason.strip().upper() if kiro_stop_reason else ""

    mapped = (
        to_openai_finish_reason(kiro_stop_reason)
        if truncated_value == "length"
        else to_anthropic_stop_reason(kiro_stop_reason)
    )

    inferred = tool_calls_value if has_tool_calls else default_value

    if mapped is None:
        return inferred

    # A response carrying tool calls must never be reported with a non-tool
    # finish reason: clients stop executing the tools when they see one. Keep
    # this guard even though upstream usually agrees; disagreement has been
    # observed when a turn both calls tools and hits another stop condition.
    if has_tool_calls and mapped != tool_calls_value:
        logger.debug(
            f"[{api_label}] Upstream stopReason '{normalized}' maps to "
            f"'{mapped}' but tool calls are present: reporting "
            f"'{tool_calls_value}'"
        )
        return tool_calls_value

    if normalized in _NOTEWORTHY_KIRO_STOP_REASONS:
        logger.info(
            f"[{api_label}] Kiro reported stopReason '{normalized}': "
            f"reporting '{mapped}' to the client"
        )
    elif mapped != inferred:
        logger.debug(
            f"[{api_label}] Upstream stopReason '{normalized}' decided the "
            f"outcome: '{mapped}' (local inference would have said "
            f"'{inferred}')"
        )

    return mapped
