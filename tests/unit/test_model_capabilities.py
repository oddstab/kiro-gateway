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
    get_native_effort_spec,
    get_native_reasoning_efforts,
    is_dynamic_metadata,
    parse_native_reasoning_schema,
    resolve_native_reasoning_format,
)

# Verbatim ListAvailableModels payloads captured from the live Kiro API. These
# are the contract the gateway must follow, so the tests assert against them
# instead of against locally invented model tables.
OFFICIAL_FOUR_LEVEL_SCHEMA = {
    "type": "object",
    "properties": {
        "thinking": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["adaptive", "disabled"]},
                "display": {"type": "string", "enum": ["summarized", "omitted"]},
            },
            "required": ["type"],
        },
        "output_config": {
            "type": "object",
            "properties": {
                "effort": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "max"],
                    "default": "high",
                }
            },
        },
        "max_tokens": {"type": "integer", "minimum": 1024, "maximum": 64000},
    },
    "additionalProperties": False,
}

OFFICIAL_FIVE_LEVEL_SCHEMA = {
    "type": "object",
    "properties": {
        "thinking": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["adaptive", "disabled"]},
                "display": {"type": "string", "enum": ["summarized", "omitted"]},
            },
            "required": ["type"],
        },
        "output_config": {
            "type": "object",
            "properties": {
                "effort": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "xhigh", "max"],
                    "default": "high",
                }
            },
        },
        "max_tokens": {"type": "integer", "minimum": 1024, "maximum": 128000},
    },
    "additionalProperties": False,
}

OFFICIAL_FIVE_LEVEL_SCHEMA_XHIGH_DEFAULT = {
    "type": "object",
    "properties": {
        "output_config": {
            "type": "object",
            "properties": {
                "effort": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "xhigh", "max"],
                    "default": "xhigh",
                }
            },
        }
    },
    "additionalProperties": False,
}

OFFICIAL_GPT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["standard", "pro"], "default": "standard"},
                "effort": {
                    "type": "string",
                    "enum": ["none", "low", "medium", "high", "xhigh", "max"],
                    "default": "high",
                },
            },
        }
    },
    "additionalProperties": False,
}


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
    """Anything not carrying official metadata must resolve to unsupported."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "claude-opus-4.8",
            "gpt-5.6-sol",
            "claude-haiku-4.5",
            "claude-sonnet-4",
            "auto",
            "deepseek-3.2",
            "totally-unknown-model",
            "",
        ],
    )
    def test_no_metadata_means_no_native_reasoning(self, model_id):
        """
        What it does: Resolves every kind of model id with no metadata at all.
        Purpose: Capability must come from the official schema, never a name.
        """
        assert resolve_native_reasoning_format(model_id, None) is None

    def test_static_marker_ignores_any_attached_schema(self):
        """
        What it does: Resolves static metadata that also carries a schema.
        Purpose: A locally fabricated schema must not grant capabilities.
        """
        info = {
            "modelId": "claude-opus-4.8",
            MODEL_SOURCE_KEY: MODEL_SOURCE_STATIC,
            NATIVE_REASONING_SCHEMA_KEY: OFFICIAL_FIVE_LEVEL_SCHEMA,
        }
        assert resolve_native_reasoning_format("claude-opus-4.8", info) is None

    @pytest.mark.parametrize("model_id", [None, 42, [], {}])
    def test_non_string_model_id_is_safe(self, model_id):
        """
        What it does: Resolves invalid model id types with no metadata.
        Purpose: Capability lookup must never raise on odd caller input.
        """
        assert resolve_native_reasoning_format(model_id, None) is None

    def test_alias_uses_official_schema_of_its_cache_entry(self):
        """
        What it does: Resolves an alias whose cached entry is the real target.
        Purpose: Aliases inherit capability from official data, not their name.
        """
        info = _dynamic(schema=OFFICIAL_FIVE_LEVEL_SCHEMA, modelId="claude-opus-4.8")
        assert resolve_native_reasoning_format("my-opus", info) == "output_config"

    def test_alias_without_official_schema_stays_unsupported(self):
        """
        What it does: Resolves an alias whose target advertises no schema.
        Purpose: Alias indirection must never invent a capability.
        """
        info = _dynamic(schema=None, modelId="claude-haiku-4.5")
        assert resolve_native_reasoning_format("my-haiku", info) is None


# ==================================================================================================
# Tests for the official effort enum readers
# ==================================================================================================

class TestGetNativeEffortSpec:
    """
    The effort enum must be read verbatim from official model metadata.

    The enums below are the real ListAvailableModels payloads: Claude 4.6
    advertises four values, Claude 4.7/4.8/5 advertise five (including xhigh),
    and the gpt-5.6 family advertises six (including none).
    """

    @pytest.mark.parametrize(
        ("model_id", "schema", "expected_values", "expected_default"),
        [
            (
                "claude-opus-4.6",
                OFFICIAL_FOUR_LEVEL_SCHEMA,
                ("low", "medium", "high", "max"),
                "high",
            ),
            (
                "claude-opus-4.7",
                OFFICIAL_FIVE_LEVEL_SCHEMA_XHIGH_DEFAULT,
                ("low", "medium", "high", "xhigh", "max"),
                "xhigh",
            ),
            (
                "claude-opus-4.8",
                OFFICIAL_FIVE_LEVEL_SCHEMA,
                ("low", "medium", "high", "xhigh", "max"),
                "high",
            ),
            (
                "gpt-5.6-sol",
                OFFICIAL_GPT_SCHEMA,
                ("none", "low", "medium", "high", "xhigh", "max"),
                "high",
            ),
        ],
    )
    def test_reads_official_enum_and_default_verbatim(
        self, model_id, schema, expected_values, expected_default
    ):
        """
        What it does: Parses each real official schema shape.
        Purpose: Order, membership and default must survive untouched.
        """
        spec = get_native_effort_spec(model_id, _dynamic(schema=schema, modelId=model_id))

        assert spec.values == expected_values
        assert spec.default == expected_default
        assert get_native_reasoning_efforts(model_id, _dynamic(schema=schema)) == frozenset(
            expected_values
        )

    def test_json_string_schema_is_supported(self):
        """
        What it does: Supplies the official schema as a JSON string.
        Purpose: AWS returns this field encoded in some responses.
        """
        info = _dynamic(schema=json.dumps(OFFICIAL_FOUR_LEVEL_SCHEMA))
        spec = get_native_effort_spec("claude-opus-4.6", info)

        assert spec.values == ("low", "medium", "high", "max")

    @pytest.mark.parametrize(
        "model_id",
        ["claude-opus-4.6", "claude-opus-4.8", "gpt-5.6-sol", "unknown-model"],
    )
    def test_absent_metadata_yields_no_effort_spec(self, model_id):
        """
        What it does: Requests an effort spec with no official metadata.
        Purpose: Without official data the gateway must claim nothing.
        """
        assert get_native_effort_spec(model_id, None) is None
        assert get_native_reasoning_efforts(model_id, None) is None

    def test_static_metadata_is_never_trusted(self):
        """
        What it does: Attaches a real schema to a static cache entry.
        Purpose: Only ListAvailableModels data may define capabilities.
        """
        info = {
            "modelId": "claude-opus-4.8",
            MODEL_SOURCE_KEY: MODEL_SOURCE_STATIC,
            NATIVE_REASONING_SCHEMA_KEY: OFFICIAL_FIVE_LEVEL_SCHEMA,
        }
        assert get_native_effort_spec("claude-opus-4.8", info) is None

    @pytest.mark.parametrize(
        "schema",
        [
            None,
            {},
            {"properties": {}},
            {"properties": {"output_config": {}}},
            {"properties": {"output_config": {"properties": {}}}},
            {"properties": {"output_config": {"properties": {"effort": {}}}}},
            {"properties": {"output_config": {"properties": {"effort": {"enum": []}}}}},
            {
                "properties": {
                    "output_config": {"properties": {"effort": {"enum": ["high", 1]}}}
                }
            },
            {
                "properties": {
                    "output_config": {"properties": {"effort": {"enum": ["", "high"]}}}
                }
            },
            "not json",
            '{"properties": {"output_config": {}}',
            [],
            42,
        ],
    )
    def test_unusable_schema_yields_no_effort_spec(self, schema):
        """
        What it does: Feeds malformed or incomplete schemas to the reader.
        Purpose: Fail closed rather than inventing an effort vocabulary.
        """
        assert get_native_effort_spec("claude-opus-4.8", _dynamic(schema=schema)) is None

    def test_default_outside_enum_is_discarded(self):
        """
        What it does: Parses a schema whose default is not in its own enum.
        Purpose: A contradictory default must not become a usable value.
        """
        schema = {
            "properties": {
                "output_config": {
                    "properties": {
                        "effort": {"enum": ["low", "high"], "default": "xhigh"}
                    }
                }
            }
        }
        spec = get_native_effort_spec("claude-test", _dynamic(schema=schema))

        assert spec.values == ("low", "high")
        assert spec.default is None

    def test_gpt_effort_is_read_from_the_reasoning_container(self):
        """
        What it does: Parses the official gpt-5.6 reasoning container.
        Purpose: The enum lives under a different key for GPT-style models.
        """
        info = _dynamic(schema=OFFICIAL_GPT_SCHEMA, modelId="gpt-5.6-luna")

        assert get_native_effort_spec("gpt-5.6-luna", info).values[0] == "none"


class TestNoModelNamesAreHardcoded:
    """
    Capability decisions must never be keyed on model names or versions.

    Regression: a 4.6-specific table produced correct behaviour for 4.6 while
    silently degrading, or wrongly trusting, every other model.
    """

    def test_module_has_no_model_specific_tables(self):
        """
        What it does: Inspects the module for model-keyed capability tables.
        Purpose: Official metadata is the only permitted source.
        """
        import kiro.model_capabilities as capabilities

        assert not hasattr(capabilities, "STATIC_NATIVE_REASONING_MODELS")
        assert not hasattr(capabilities, "STATIC_NATIVE_REASONING_EFFORTS")

    def test_capability_source_files_do_not_mention_model_versions(self):
        """
        What it does: Greps the capability and conversion modules for versions.
        Purpose: A version literal is how the hardcoding regression returns.
        """
        import re
        from pathlib import Path

        import kiro.converters_core as converters_core
        import kiro.model_capabilities as capabilities
        import kiro.routes_openai as routes_openai

        version_pattern = re.compile(r"(claude|gpt)-[a-z0-9.\-]*\d", re.IGNORECASE)
        for module in (capabilities, converters_core, routes_openai):
            source = Path(module.__file__).read_text(encoding="utf-8")
            code_only = "\n".join(
                line for line in source.splitlines() if not line.strip().startswith("#")
            )
            for match in version_pattern.finditer(code_only):
                # Docstrings may reference models; executable logic may not.
                line = code_only[
                    code_only.rfind("\n", 0, match.start()) + 1 : match.end()
                ]
                assert '"' in line or "'" not in line, (
                    f"{module.__name__} appears to branch on model name: {line!r}"
                )
