# -*- coding: utf-8 -*-

"""
Tests for kiro/model_capabilities.py - native reasoning capability detection.

This module is the single source of truth for "does this model accept
additionalModelRequestFields?". The tests below are deliberately hostile: the
bug they guard against is claude-haiku-4.5 being advertised as effort-capable
because its name contains "claude", which made AWS answer
400 additionalModelRequestFields is not supported for this model
(REQUEST_BODY_INVALID).
"""

import json

import pytest

from kiro.model_capabilities import (
    MODEL_SOURCE_DYNAMIC,
    MODEL_SOURCE_KEY,
    MODEL_SOURCE_STATIC,
    NATIVE_REASONING_SCHEMA_KEY,
    STATIC_NATIVE_REASONING_MODELS,
    is_dynamic_metadata,
    parse_native_reasoning_schema,
    resolve_native_reasoning_format,
)


def _dynamic(schema=..., **extra):
    """Build dynamic (AWS) metadata, optionally without any schema key."""
    info = {"modelId": "test-model", MODEL_SOURCE_KEY: MODEL_SOURCE_DYNAMIC}
    if schema is not ...:
        info[NATIVE_REASONING_SCHEMA_KEY] = schema
    info.update(extra)
    return info


# ==================================================================================================
# Tests for parse_native_reasoning_schema
# ==================================================================================================

class TestParseNativeReasoningSchemaSuccess:
    """Schemas that legitimately advertise a native reasoning protocol."""

    def test_reasoning_property_maps_to_gpt_style(self):
        """
        What it does: Parses a schema exposing properties.reasoning.
        Purpose: GPT-5.6 style models must keep using {"reasoning": {...}}.
        """
        schema = {"properties": {"reasoning": {"type": "object"}}}
        assert parse_native_reasoning_schema(schema) == "reasoning"

    def test_output_config_property_maps_to_claude_style(self):
        """
        What it does: Parses a schema exposing properties.output_config.
        Purpose: Claude models that really support effort keep native fields.
        """
        schema = {"properties": {"output_config": {"type": "object"}}}
        assert parse_native_reasoning_schema(schema) == "output_config"

    def test_json_string_schema_is_parsed(self):
        """
        What it does: Parses a schema delivered as a JSON-encoded string.
        Purpose: AWS returns this field as a string in some responses.
        """
        schema = json.dumps({"properties": {"output_config": {}}})
        assert parse_native_reasoning_schema(schema) == "output_config"

    def test_reasoning_wins_when_both_properties_present(self):
        """
        What it does: Resolves a schema exposing both properties.
        Purpose: Deterministic precedence; GPT-style matches Kiro's own shape.
        """
        schema = {"properties": {"reasoning": {}, "output_config": {}}}
        assert parse_native_reasoning_schema(schema) == "reasoning"

    def test_extra_properties_do_not_break_detection(self):
        """
        What it does: Parses a schema with unrelated sibling properties.
        Purpose: Unknown AWS fields must not hide a real capability.
        """
        schema = {
            "type": "object",
            "properties": {"anthropic_beta": {}, "output_config": {}},
        }
        assert parse_native_reasoning_schema(schema) == "output_config"


class TestParseNativeReasoningSchemaFailClosed:
    """Everything ambiguous or malformed must resolve to "not supported"."""

    @pytest.mark.parametrize(
        "schema",
        [
            None,
            {},
            {"properties": {}},
            {"properties": None},
            {"properties": []},
            {"properties": "reasoning"},
            {"properties": {"thinking": {}}},
            {"properties": {"budget_tokens": {}}},
            {"reasoning": {}},
            {"output_config": {}},
            "",
            "null",
            "not json at all",
            '{"properties": {"reasoning": {}}',  # truncated JSON
            "[]",
            "42",
            [],
            42,
            True,
        ],
    )
    def test_rejects_unusable_schema(self, schema):
        """
        What it does: Feeds malformed/empty/unrelated schemas to the parser.
        Purpose: Fail closed - never invent a capability from bad metadata.
        """
        assert parse_native_reasoning_schema(schema) is None

    def test_json_string_of_list_is_rejected(self):
        """
        What it does: Parses valid JSON that is not an object.
        Purpose: Valid JSON is not the same as a valid schema.
        """
        assert parse_native_reasoning_schema('[{"properties": {"reasoning": {}}}]') is None


# ==================================================================================================
# Tests for is_dynamic_metadata
# ==================================================================================================

class TestIsDynamicMetadata:
    """Provenance detection decides which capability rule set applies."""

    def test_dynamic_marker_is_dynamic(self):
        """
        What it does: Checks metadata tagged as coming from AWS.
        Purpose: AWS schema must be treated as authoritative.
        """
        assert is_dynamic_metadata(_dynamic(schema={"properties": {}})) is True

    def test_static_marker_is_not_dynamic(self):
        """
        What it does: Checks metadata tagged as a local static definition.
        Purpose: Static entries have no schema, so whitelist rules apply.
        """
        info = {"modelId": "claude-haiku-4.5", MODEL_SOURCE_KEY: MODEL_SOURCE_STATIC}
        assert is_dynamic_metadata(info) is False

    def test_untagged_metadata_defaults_to_dynamic(self):
        """
        What it does: Checks metadata without a provenance marker.
        Purpose: Unknown provenance gets the stricter (schema-only) treatment.
        """
        assert is_dynamic_metadata({"modelId": "claude-haiku-4.5"}) is True

    @pytest.mark.parametrize("model_info", [None, "claude-haiku-4.5", [], 0])
    def test_non_dict_metadata_is_not_dynamic(self, model_info):
        """
        What it does: Passes non-dict values as metadata.
        Purpose: Absent metadata must route to the whitelist, not crash.
        """
        assert is_dynamic_metadata(model_info) is False


# ==================================================================================================
# Tests for resolve_native_reasoning_format
# ==================================================================================================

class TestResolveNativeReasoningFormatDynamic:
    """AWS metadata is authoritative: schema or nothing."""

    def test_haiku_without_schema_has_no_native_reasoning(self):
        """
        What it does: Resolves claude-haiku-4.5 with AWS metadata but no schema.
        Purpose: The exact bug - name-based guessing produced a 400 from AWS.
        """
        info = _dynamic(modelId="claude-haiku-4.5", tokenLimits={"maxInputTokens": 200000})
        assert resolve_native_reasoning_format("claude-haiku-4.5", info) is None

    @pytest.mark.parametrize(
        "schema",
        [None, {}, {"properties": {}}, {"properties": {"thinking": {}}}, "broken json"],
    )
    def test_dynamic_claude_with_unusable_schema_fails_closed(self, schema):
        """
        What it does: Resolves a Claude model whose AWS schema is unusable.
        Purpose: A Claude name must never re-enable native fields.
        """
        info = _dynamic(schema=schema, modelId="claude-haiku-4.5")
        assert resolve_native_reasoning_format("claude-haiku-4.5", info) is None

    def test_dynamic_metadata_overrides_static_whitelist(self):
        """
        What it does: Resolves a whitelisted model whose AWS schema is empty.
        Purpose: AWS truth beats our local table; the whitelist is fallback only.
        """
        info = _dynamic(schema={"properties": {}}, modelId="claude-opus-4.8")
        assert resolve_native_reasoning_format("claude-opus-4.8", info) is None

    def test_dynamic_schema_enables_model_absent_from_whitelist(self):
        """
        What it does: Resolves an unlisted model whose AWS schema supports it.
        Purpose: New Kiro models get native reasoning without a code change.
        """
        info = _dynamic(
            schema={"properties": {"output_config": {}}},
            modelId="claude-brand-new-9",
        )
        assert resolve_native_reasoning_format("claude-brand-new-9", info) == "output_config"

    def test_untagged_dynamic_metadata_still_uses_schema(self):
        """
        What it does: Resolves metadata lacking a provenance marker.
        Purpose: Callers building model_info by hand keep schema-only semantics.
        """
        info = {
            "modelId": "gpt-5.6-sol",
            NATIVE_REASONING_SCHEMA_KEY: {"properties": {"reasoning": {}}},
        }
        assert resolve_native_reasoning_format("gpt-5.6-sol", info) == "reasoning"

    def test_untagged_metadata_without_schema_fails_closed(self):
        """
        What it does: Resolves untagged metadata that carries no schema.
        Purpose: Route-supplied cache entries must not fall back to family names.
        """
        info = {"modelId": "claude-haiku-4.5", "tokenLimits": {"maxInputTokens": 200000}}
        assert resolve_native_reasoning_format("claude-haiku-4.5", info) is None


class TestResolveNativeReasoningFormatStatic:
    """Static/hidden/pass-through models use the explicit whitelist only."""

    def test_haiku_is_not_whitelisted(self):
        """
        What it does: Resolves claude-haiku-4.5 with no metadata at all.
        Purpose: Offline FALLBACK_MODELS mode must not claim effort support.
        """
        assert resolve_native_reasoning_format("claude-haiku-4.5", None) is None
        assert "claude-haiku-4.5" not in STATIC_NATIVE_REASONING_MODELS

    def test_whitelisted_claude_model_uses_output_config(self):
        """
        What it does: Resolves a whitelisted Claude model without metadata.
        Purpose: Offline mode keeps working for verified models.
        """
        assert resolve_native_reasoning_format("claude-opus-4.8", None) == "output_config"

    def test_whitelisted_gpt_model_uses_reasoning(self):
        """
        What it does: Resolves a whitelisted GPT model without metadata.
        Purpose: GPT-style protocol survives the offline path.
        """
        assert resolve_native_reasoning_format("gpt-5.6-sol", None) == "reasoning"

    def test_static_marker_ignores_any_attached_schema(self):
        """
        What it does: Resolves static metadata that also carries a schema.
        Purpose: A locally fabricated schema must not grant capabilities.
        """
        info = {
            "modelId": "claude-haiku-4.5",
            MODEL_SOURCE_KEY: MODEL_SOURCE_STATIC,
            NATIVE_REASONING_SCHEMA_KEY: {"properties": {"output_config": {}}},
        }
        assert resolve_native_reasoning_format("claude-haiku-4.5", info) is None

    @pytest.mark.parametrize(
        "client_name",
        [
            "claude-opus-4-8",
            "claude-opus-4-8-20251001",
            "claude-opus-4.8[1m]",
            "CLAUDE-OPUS-4-8",
        ],
    )
    def test_whitelist_lookup_normalizes_client_names(self, client_name):
        """
        What it does: Resolves client-side spellings of a whitelisted model.
        Purpose: Capability must not depend on which name format a client sends.
        """
        assert resolve_native_reasoning_format(client_name, None) == "output_config"

    @pytest.mark.parametrize(
        "model_id",
        [
            "claude-haiku-4.5",
            "claude-haiku-4-5",
            "claude-sonnet-4",
            "auto",
            "deepseek-3.2",
            "glm-5",
            "minimax-m2.5",
            "qwen3-coder-next",
            "totally-unknown-model",
            "",
        ],
    )
    def test_unverified_models_get_no_native_reasoning(self, model_id):
        """
        What it does: Resolves models absent from the whitelist.
        Purpose: Broad family guessing stays gone; unknown means unsupported.
        """
        assert resolve_native_reasoning_format(model_id, None) is None

    @pytest.mark.parametrize("model_id", [None, 42, [], {}])
    def test_non_string_model_id_is_safe(self, model_id):
        """
        What it does: Resolves invalid model id types with no metadata.
        Purpose: Capability lookup must never raise on odd caller input.
        """
        assert resolve_native_reasoning_format(model_id, None) is None

    def test_alias_inherits_target_from_static_metadata(self):
        """
        What it does: Resolves an alias whose static cache entry is its target.
        Purpose: In offline mode an alias must inherit the target's capability,
                 not be judged by the alias display name (which is never listed).
        """
        info = {
            "modelId": "claude-opus-4.8",
            MODEL_SOURCE_KEY: MODEL_SOURCE_STATIC,
        }
        assert resolve_native_reasoning_format("my-opus", info) == "output_config"

    def test_alias_to_unsupported_target_stays_unsupported(self):
        """
        What it does: Resolves an alias pointing at a non-whitelisted target.
        Purpose: Alias indirection must never upgrade an unsupported model.
        """
        info = {
            "modelId": "claude-haiku-4.5",
            MODEL_SOURCE_KEY: MODEL_SOURCE_STATIC,
        }
        assert resolve_native_reasoning_format("my-haiku", info) is None

    def test_static_metadata_without_model_id_falls_back_to_argument(self):
        """
        What it does: Resolves static metadata missing its modelId key.
        Purpose: A partial cache entry must not lose the caller's model id.
        """
        info = {MODEL_SOURCE_KEY: MODEL_SOURCE_STATIC}
        assert resolve_native_reasoning_format("claude-opus-4.8", info) == "output_config"
        assert resolve_native_reasoning_format("claude-haiku-4.5", info) is None


class TestStaticWhitelistIntegrity:
    """Guardrails on the whitelist table itself."""

    def test_all_values_are_known_formats(self):
        """
        What it does: Checks every whitelist value.
        Purpose: A typo would silently produce an invalid payload key.
        """
        assert set(STATIC_NATIVE_REASONING_MODELS.values()) <= {
            "reasoning",
            "output_config",
        }

    def test_keys_are_normalized_ids(self):
        """
        What it does: Checks whitelist keys survive normalization unchanged.
        Purpose: A non-normalized key would never be matched at lookup time.
        """
        from kiro.model_resolver import normalize_model_name

        for model_id in STATIC_NATIVE_REASONING_MODELS:
            assert normalize_model_name(model_id) == model_id
