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

Capability resolution is schema-only:

- Models returned by ``ListAvailableModels`` are decided **only** by their
  ``additionalModelRequestFieldsSchema``. That schema is AWS's contract and is
  authoritative for both the native request shape and each model's exact
  ``effort`` enum.
- Missing, null, empty, malformed, unrelated, or locally fabricated schemas
  mean "no native reasoning effort" (fail closed). Static fallback and hidden
  model entries never infer capabilities from a model name.

This prevents the gateway from sending values that one model rejects merely
because another model in the same family accepts them.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Tuple

from loguru import logger

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

# Marker injected by ModelInfoCache. Only dynamic entries originate from the
# official ListAvailableModels response and may grant native capabilities.
MODEL_SOURCE_KEY: str = "_metadata_source"
MODEL_SOURCE_DYNAMIC: str = "dynamic"
MODEL_SOURCE_STATIC: str = "static"


def _parse_schema_object(schema: Any) -> Optional[Dict[str, Any]]:
    """Decode an AWS additional-model-fields schema into an object."""
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            logger.warning(
                f"Ignoring invalid {NATIVE_REASONING_SCHEMA_KEY} JSON: "
                "treating model as not supporting native reasoning"
            )
            return None

    return schema if isinstance(schema, dict) else None


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
    schema_object = _parse_schema_object(schema)
    if schema_object is None:
        return None

    properties = schema_object.get("properties")
    if not isinstance(properties, dict):
        return None

    for property_name, native_format in _SCHEMA_PROPERTY_TO_FORMAT:
        if property_name in properties:
            return native_format

    return None


def is_dynamic_metadata(model_info: Optional[Dict[str, Any]]) -> bool:
    """
    Check whether metadata came from AWS ListAvailableModels.

    Entries explicitly tagged static are never trusted. Untagged metadata is
    treated as dynamic because dynamic handling is the strict path: an official
    schema is still required before any capability is granted.

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
    Resolve the native reasoning protocol from official model metadata.

    Args:
        model_id: Resolved Kiro model id, used only for diagnostics.
        model_info: Metadata cached from ListAvailableModels.

    Returns:
        ``reasoning`` or ``output_config`` when the official schema advertises
        that protocol; otherwise None.
    """
    if not is_dynamic_metadata(model_info):
        logger.debug(
            f"Model '{model_id}' has no official ListAvailableModels metadata: "
            "native reasoning disabled"
        )
        return None

    native_format = parse_native_reasoning_schema(
        model_info.get(NATIVE_REASONING_SCHEMA_KEY)
    )
    if native_format is None:
        logger.debug(
            f"Model '{model_id}' has no {NATIVE_REASONING_SCHEMA_KEY} entry for "
            "reasoning/output_config: native reasoning disabled"
        )
    return native_format


@dataclass(frozen=True)
class NativeEffortSpec:
    """
    One model's official reasoning effort contract.

    Attributes:
        values: Effort values in the exact order AWS advertises them.
        default: The schema's own ``default`` value, when present.
    """

    values: Tuple[str, ...]
    default: Optional[str] = None

    @property
    def value_set(self) -> FrozenSet[str]:
        """Return the accepted effort values as a set."""
        return frozenset(self.values)


def get_native_effort_spec(
    model_id: str,
    model_info: Optional[Dict[str, Any]] = None,
) -> Optional[NativeEffortSpec]:
    """
    Read one model's official effort enum out of its AWS schema.

    Args:
        model_id: Resolved Kiro model id, used only for diagnostics.
        model_info: Metadata cached from ListAvailableModels.

    Returns:
        The official NativeEffortSpec, or None when AWS advertises no effort
        enum for this model.
    """
    if not is_dynamic_metadata(model_info):
        return None

    schema = _parse_schema_object(model_info.get(NATIVE_REASONING_SCHEMA_KEY))
    if schema is None:
        return None

    native_format = parse_native_reasoning_schema(schema)
    properties = schema.get("properties")
    container_schema = (
        properties.get(native_format)
        if isinstance(properties, dict) and native_format
        else None
    )
    container_properties = (
        container_schema.get("properties")
        if isinstance(container_schema, dict)
        else None
    )
    effort_schema = (
        container_properties.get("effort")
        if isinstance(container_properties, dict)
        else None
    )
    if not isinstance(effort_schema, dict):
        return None

    effort_enum = effort_schema.get("enum")
    if not isinstance(effort_enum, list) or not effort_enum:
        return None
    if not all(isinstance(value, str) and value for value in effort_enum):
        logger.warning(
            f"Model '{model_id}' advertises a non-string effort enum: "
            "ignoring native effort support"
        )
        return None

    default = effort_schema.get("default")
    if not isinstance(default, str) or default not in effort_enum:
        default = None

    return NativeEffortSpec(values=tuple(effort_enum), default=default)


def get_native_reasoning_efforts(
    model_id: str,
    model_info: Optional[Dict[str, Any]] = None,
) -> Optional[FrozenSet[str]]:
    """
    Return the official effort values accepted by one model.

    Args:
        model_id: Resolved Kiro model id.
        model_info: Metadata cached from ListAvailableModels.

    Returns:
        Accepted effort values, or None when AWS advertises no effort enum.
    """
    spec = get_native_effort_spec(model_id, model_info)
    return spec.value_set if spec else None
