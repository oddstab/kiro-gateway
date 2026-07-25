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
Model capability detection for Kiro Gateway.

Single source of truth for "does this model support Kiro native reasoning?".
Both the /v1/models advertisement (what the client UI shows) and the request
converters (what we actually send upstream) resolve capabilities through this
module, so the advertised metadata can never drift from runtime behaviour.

Capability resolution is provenance aware:

- Dynamic models (from ListAvailableModels) are decided **only** by their
  ``additionalModelRequestFieldsSchema``. That schema is AWS's own contract and
  is therefore authoritative: a missing, null, empty, malformed or unrelated
  schema means "no native reasoning" (fail closed). Model-name family guessing
  is never applied here, because guessing is exactly what made Kiro Gateway
  send ``additionalModelRequestFields`` to claude-haiku-4.5 and get
  ``400 additionalModelRequestFields is not supported for this model
  (REQUEST_BODY_INVALID)``.
- Static models (FALLBACK_MODELS when ListAvailableModels is unreachable, plus
  hidden/manually configured models) carry no AWS schema at all. For those the
  only permitted source is the explicit whitelist below. Anything not listed is
  treated as unsupported.
"""

import json
from typing import Any, Dict, Optional

from loguru import logger

from kiro.model_resolver import normalize_model_name

# ==================================================================================================
# Native Reasoning Formats
# ==================================================================================================

# GPT-style native reasoning: additionalModelRequestFields = {"reasoning": {...}}
NATIVE_REASONING_FORMAT_REASONING: str = "reasoning"

# Claude-style native reasoning: additionalModelRequestFields =
# {"thinking": {...}, "output_config": {...}}
NATIVE_REASONING_FORMAT_OUTPUT_CONFIG: str = "output_config"

# Key of the AWS-provided JSON Schema describing additionalModelRequestFields.
NATIVE_REASONING_SCHEMA_KEY: str = "additionalModelRequestFieldsSchema"

# Schema properties that identify each native reasoning protocol, in priority
# order (a model exposing both is treated as GPT-style, matching Kiro's own
# request shape for the gpt-5.6 family).
_SCHEMA_PROPERTY_TO_FORMAT = (
    (NATIVE_REASONING_FORMAT_REASONING, NATIVE_REASONING_FORMAT_REASONING),
    (NATIVE_REASONING_FORMAT_OUTPUT_CONFIG, NATIVE_REASONING_FORMAT_OUTPUT_CONFIG),
)

# ==================================================================================================
# Metadata Provenance
# ==================================================================================================

# Marker injected by ModelInfoCache so capability resolution can tell AWS
# metadata apart from locally configured entries.
MODEL_SOURCE_KEY: str = "_metadata_source"

# Metadata came from ListAvailableModels -> its schema is authoritative.
MODEL_SOURCE_DYNAMIC: str = "dynamic"

# Metadata came from FALLBACK_MODELS / HIDDEN_MODELS -> whitelist only.
MODEL_SOURCE_STATIC: str = "static"

# ==================================================================================================
# Static Capability Whitelist
# ==================================================================================================

# Native reasoning support for models with no AWS metadata (offline fallback
# list, hidden models, unknown pass-through ids).
#
# Rules for this table:
# - Keys are normalized Kiro model ids (dot format, see normalize_model_name).
# - Entries are explicit. No families, no prefixes, no "claude means Claude
#   style" heuristics. A model absent from this table gets no native reasoning
#   fields, which is always safe: the prompt-based fake reasoning fallback still
#   provides reasoning when FAKE_REASONING is enabled.
# - Deliberately absent: claude-haiku-4.5 (AWS rejects
#   additionalModelRequestFields for it), claude-sonnet-4, auto, and the
#   non-Anthropic models (deepseek/glm/minimax/qwen) whose schemas we have not
#   verified.
# - When ListAvailableModels is reachable this table is irrelevant: the AWS
#   schema wins.
STATIC_NATIVE_REASONING_MODELS: Dict[str, str] = {
    # Claude models that accept {"thinking": ..., "output_config": ...}
    "claude-sonnet-4.5": NATIVE_REASONING_FORMAT_OUTPUT_CONFIG,
    "claude-sonnet-4.6": NATIVE_REASONING_FORMAT_OUTPUT_CONFIG,
    "claude-sonnet-5": NATIVE_REASONING_FORMAT_OUTPUT_CONFIG,
    "claude-opus-4.5": NATIVE_REASONING_FORMAT_OUTPUT_CONFIG,
    "claude-opus-4.6": NATIVE_REASONING_FORMAT_OUTPUT_CONFIG,
    "claude-opus-4.7": NATIVE_REASONING_FORMAT_OUTPUT_CONFIG,
    "claude-opus-4.8": NATIVE_REASONING_FORMAT_OUTPUT_CONFIG,
    "claude-opus-5": NATIVE_REASONING_FORMAT_OUTPUT_CONFIG,
    # GPT-5.6 family accepts {"reasoning": {...}}
    "gpt-5.6-sol": NATIVE_REASONING_FORMAT_REASONING,
    "gpt-5.6-terra": NATIVE_REASONING_FORMAT_REASONING,
    "gpt-5.6-luna": NATIVE_REASONING_FORMAT_REASONING,
}


def parse_native_reasoning_schema(schema: Any) -> Optional[str]:
    """
    Read the native reasoning format out of an AWS request-fields schema.

    Args:
        schema: Value of additionalModelRequestFieldsSchema. May be a dict, a
            JSON-encoded string, or anything else (all non-conforming values
            are rejected).

    Returns:
        NATIVE_REASONING_FORMAT_REASONING, NATIVE_REASONING_FORMAT_OUTPUT_CONFIG,
        or None when the schema does not advertise a supported reasoning field.

    Examples:
        >>> parse_native_reasoning_schema({"properties": {"reasoning": {}}})
        'reasoning'
        >>> parse_native_reasoning_schema('{"properties": {"output_config": {}}}')
        'output_config'
        >>> parse_native_reasoning_schema({"properties": {}}) is None
        True
        >>> parse_native_reasoning_schema("not json") is None
        True
    """
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            logger.warning(
                f"Ignoring invalid {NATIVE_REASONING_SCHEMA_KEY} JSON: "
                "treating model as not supporting native reasoning"
            )
            return None

    if not isinstance(schema, dict):
        return None

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None

    for property_name, native_format in _SCHEMA_PROPERTY_TO_FORMAT:
        if property_name in properties:
            return native_format

    return None


def is_dynamic_metadata(model_info: Optional[Dict[str, Any]]) -> bool:
    """
    Check whether metadata came from AWS ListAvailableModels.

    Unmarked metadata is treated as dynamic on purpose: dynamic handling is the
    strict one (schema or nothing), so an unknown provenance can never unlock
    native reasoning by accident.

    Args:
        model_info: Model metadata dict, or None.

    Returns:
        True when the metadata should be judged by its AWS schema.
    """
    if not isinstance(model_info, dict):
        return False
    return model_info.get(MODEL_SOURCE_KEY, MODEL_SOURCE_DYNAMIC) != MODEL_SOURCE_STATIC


def resolve_native_reasoning_format(
    model_id: str,
    model_info: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Resolve the native reasoning protocol for a model.

    This is the only place allowed to answer that question. It is used both by
    the /v1/models capability advertisement and by the OpenAI/Anthropic request
    converters (streaming and non-streaming), so the UI and the upstream
    payload can never disagree.

    Args:
        model_id: Resolved Kiro model id (aliases must already be resolved by
            the caller so a model inherits its target's capabilities).
        model_info: Metadata for that model, when available.

    Returns:
        "reasoning" for GPT-style additionalModelRequestFields, "output_config"
        for Claude-style fields, or None when native reasoning must not be sent.
    """
    if is_dynamic_metadata(model_info):
        # AWS metadata is the contract. No schema -> no native reasoning.
        native_format = parse_native_reasoning_schema(
            model_info.get(NATIVE_REASONING_SCHEMA_KEY)
        )
        if native_format is None:
            logger.debug(
                f"Model '{model_id}' has no {NATIVE_REASONING_SCHEMA_KEY} entry for "
                "reasoning/output_config: native reasoning disabled"
            )
        return native_format

    # Static/hidden/pass-through model: explicit whitelist only.
    #
    # Prefer the cached entry's modelId over the caller-supplied id. For an alias
    # (grok-* etc.) the cache entry is the alias *target*, so this is what makes
    # an alias inherit its target's capability in offline mode instead of being
    # judged by its own display name.
    lookup_id = model_id
    if isinstance(model_info, dict):
        cached_id = model_info.get("modelId")
        if isinstance(cached_id, str) and cached_id:
            lookup_id = cached_id

    if not isinstance(lookup_id, str) or not lookup_id:
        return None

    native_format = STATIC_NATIVE_REASONING_MODELS.get(normalize_model_name(lookup_id))
    if native_format is None:
        logger.debug(
            f"Model '{lookup_id}' has no AWS metadata and is not in the static "
            "native reasoning whitelist: native reasoning disabled"
        )
    return native_format
