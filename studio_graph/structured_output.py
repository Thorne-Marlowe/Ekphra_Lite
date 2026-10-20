"""LM Studio/OpenAI-compatible JSON Schema request configuration.

The schemas require the durable application contract while permitting the
storyboarder to retain its existing optional visual fields.  They are passed
directly as Chat Completions ``response_format`` rather than relying on a
provider-specific parser or free-form JSON repair.
"""
from __future__ import annotations

from difflib import get_close_matches
import re
from typing import Any, Callable

from .narrative_contract import NARRATIVE_RESOLUTIONS


STORY_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "language", "tone", "trope", "characters", "beats"],
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "language": {"type": "string", "minLength": 1},
        "tone": {"type": "string", "minLength": 1},
        "trope": {"type": "string", "minLength": 1},
        "creative_intent": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "genre": {"type": "string"},
                "tone": {"type": "string"},
                # ``auto`` directs Planner inference; it is never an inference.
                "narrative_resolution": {"enum": sorted(NARRATIVE_RESOLUTIONS - {"auto"})},
            },
        },
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "name"],
                "properties": {
                    "id": {"type": "string", "minLength": 1, "pattern": r"^\S(?:.*\S)?$"},
                    "name": {"type": "string", "minLength": 1},
                    "role": {"type": "string"},
                    "goal": {"type": "string"},
                    "personality_contrast": {"type": "string"},
                    "fixed_visual_identity": {"type": "string"},
                },
            },
        },
        "beats": {
            "type": "array",
            "items": {
                "type": "object",
                # This object is closed so malformed near-canonical keys cannot
                # silently bypass downstream constraint transfer.
                "additionalProperties": False,
                "required": ["panel", "event"],
                "properties": {
                    "panel": {
                        "anyOf": [
                            {"type": "integer", "minimum": 1},
                            {"type": "string", "pattern": "^[1-9][0-9]*$"},
                        ],
                    },
                    "role": {"type": "string"},
                    "event": {"type": "string", "minLength": 1},
                    "primary_subject": {"type": "string"},
                    "secondary_subject": {"type": "string"},
                    "visual_hierarchy": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                    "required_subjects": {"type": "array", "items": {"type": "string"}},
                    "required_action": {"type": "string", "minLength": 1},
                    "required_props": {"type": "array", "items": {"type": "string"}},
                    "required_text": {"type": "array", "items": {"type": "string"}},
                    "dialogue_intent": {"type": "string"},
                    "payoff": {"type": "string"},
                },
            },
        },
    },
}


def _json_type_matches(value: Any, expected: str) -> bool:
    """Match JSON Schema primitive types without treating booleans as numbers."""
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__


def _field_path(parent: str, field: str) -> str:
    return f"{parent}.{field}" if parent else field


def _unknown_field_error(parent: str, field: str, canonical_fields: set[str]) -> str:
    path = _field_path(parent, field)
    matches = get_close_matches(field.strip(), sorted(canonical_fields), n=1, cutoff=0.6)
    suggestion = f" Did you mean {matches[0]}?" if matches else ""
    return f"{path}: Unknown field.{suggestion}"


def _validate_schema_value(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    """Validate the JSON Schema subset used by the Story Planner contract."""
    errors: list[str] = []
    if "anyOf" in schema:
        if any(not _validate_schema_value(value, option, path) for option in schema["anyOf"]):
            return []
        return [f"{path or 'story_plan'}: Value does not match any accepted schema form."]

    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(str(item) for item in schema["enum"])
        return [f"{path or 'story_plan'}: Must be one of: {allowed}."]

    expected_type = schema.get("type")
    if expected_type and not _json_type_matches(value, expected_type):
        return [
            f"{path or 'story_plan'}: Expected {expected_type}, received {_json_type_name(value)}."
        ]

    if expected_type == "object":
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{_field_path(path, required)}: Missing required field.")
        additional_properties = schema.get("additionalProperties", True)
        if additional_properties is False:
            canonical_fields = set(properties)
            for field in value:
                if field not in properties:
                    errors.append(_unknown_field_error(path, str(field), canonical_fields))
        for field, child_value in value.items():
            child_schema = properties.get(field)
            if child_schema is not None:
                errors.extend(_validate_schema_value(child_value, child_schema, _field_path(path, field)))
            elif isinstance(additional_properties, dict):
                errors.extend(_validate_schema_value(child_value, additional_properties, _field_path(path, str(field))))

    elif expected_type == "array":
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value, start=1):
                errors.extend(_validate_schema_value(item, item_schema, f"{path}[{index}]"))

    elif expected_type == "string":
        if len(value) < schema.get("minLength", 0) or (schema.get("minLength", 0) and not value.strip()):
            errors.append(f"{path}: Must be a non-empty string.")
        pattern = schema.get("pattern")
        if pattern and re.fullmatch(pattern, value) is None:
            errors.append(f"{path}: Does not match the required format.")

    elif expected_type in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: Must be at least {schema['minimum']}.")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: Must be at most {schema['maximum']}.")
    return errors


def validate_story_plan_contract(story_plan: Any) -> list[str]:
    """Validate one canonical Planner result independently of model enforcement."""
    errors = _validate_schema_value(story_plan, STORY_PLAN_SCHEMA, "")
    if not isinstance(story_plan, dict):
        return errors

    characters = story_plan.get("characters")
    if isinstance(characters, list):
        seen_ids: set[str] = set()
        for index, character in enumerate(characters, start=1):
            if not isinstance(character, dict) or not isinstance(character.get("id"), str):
                continue
            character_id = character["id"].strip()
            if not character_id:
                continue
            if character_id in seen_ids:
                errors.append(f"characters[{index}].id: Character IDs must be unique.")
            seen_ids.add(character_id)
    return errors


def validate_storyboard_contract(storyboard: Any) -> list[str]:
    """Validate a parsed Storyboard Artist result before normalization.

    This deliberately validates only the static JSON contract.  Cross-field
    requirements such as registered character IDs, requested page count, and
    adult-only safety remain the responsibility of deterministic storyboard
    validation after approved constraints and dialogue are merged.
    """
    return _validate_schema_value(storyboard, STORYBOARD_SCHEMA, "")

DIALOGUE_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": ["dialogues"],
    "properties": {
        "dialogues": {"type": "array", "items": {"type": "object", "additionalProperties": True, "required": ["panel", "character_id", "text", "intent", "emotion"]}},
    },
}

_STRING_OR_STRING_LIST = {
    "anyOf": [
        {"type": "string"},
        {"type": "array", "items": {"type": "string"}},
    ]
}

_CHARACTER_POSITION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "x": {"anyOf": [{"enum": ["left", "center", "right"]}, {"type": "number", "minimum": 0, "maximum": 1}]},
        "y": {"anyOf": [{"enum": ["top", "middle", "bottom"]}, {"type": "number", "minimum": 0, "maximum": 1}]},
        "anchor": {"enum": ["center"]},
    },
}

_CHARACTER_ACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action_en": {"type": "string"},
        "pose_en": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "interaction": {"type": "string"},
    },
}

_DIALOGUE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["character_id", "text"],
    "properties": {
        "character_id": {"type": "string"},
        "text": {"type": "string"},
        "emotion": {"type": "string"},
        # Retained for generated and saved dialogue metadata even though
        # lettering currently uses character_id, text, and emotion.
        "intent": {"type": "string"},
    },
}

_PANEL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "panel_prompt_en", "active_char_ids", "dialogues"],
    "properties": {
        "id": {"type": "string"},
        "panel_prompt_en": {"type": "string"},
        "panel_negative_en": {"type": "string"},
        "description_en": {"type": "string"},
        "background_prompt_en": {"type": "string"},
        "active_char_ids": {"type": "array", "items": {"type": "string"}},
        "dialogues": {"type": "array", "items": _DIALOGUE_SCHEMA},
        "primary_subject": {"type": "string"},
        "secondary_subject": {"type": "string"},
        "visual_hierarchy": _STRING_OR_STRING_LIST,
        "required_subjects": {
            "anyOf": [
                _STRING_OR_STRING_LIST,
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "primary": {"type": "string"},
                        "secondary": {"type": "string"},
                    },
                },
            ],
        },
        "required_action": {"type": "string"},
        "required_props": _STRING_OR_STRING_LIST,
        "required_text": _STRING_OR_STRING_LIST,
        "character_positions": {
            "type": "object",
            "additionalProperties": _CHARACTER_POSITION_SCHEMA,
        },
        "character_actions": {
            "type": "object",
            "additionalProperties": _CHARACTER_ACTION_SCHEMA,
        },
    },
}

STORYBOARD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["characters", "pages"],
    "properties": {
        "title": {"type": "string"},
        # Narrative-contract metadata has nested provenance fields; preserve
        # those and other durable metadata without making it model-structural.
        "metadata": {"type": "object", "additionalProperties": True},
        "background": {
            "type": "object",
            "additionalProperties": False,
            "required": ["prompt_en"],
            "properties": {
                "prompt_en": {"type": "string"},
                "negative_en": {"type": "string"},
                "seed": {"type": "integer"},
            },
        },
        "characters": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "base_prompt_en"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "base_prompt_en": {"type": "string"},
                    "base_negative_en": {"type": "string"},
                    "camera_angle": {"type": "string"},
                    "camera_distance": {"type": "string"},
                    "seed": {"type": "integer"},
                },
            },
        },
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["page_number", "panels"],
                "properties": {
                    "page_number": {"type": "integer"},
                    "panels": {"type": "array", "items": _PANEL_SCHEMA},
                },
            },
        },
    },
}


def json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON Schema response format accepted by LM Studio.

    ``strict`` is intentionally omitted: not every OpenAI-compatible runtime
    implements that OpenAI extension, while LM Studio accepts this standard
    ``json_schema`` envelope.
    """
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


def create_structured_llm(
    model: str,
    base_url: str | None,
    schema_name: str,
    schema: dict[str, Any],
    client_factory: Callable[..., Any] | None = None,
):
    """Create an explicit schema-constrained LangChain chat model."""
    if client_factory is None:
        # Keep schema construction testable without importing the optional
        # FastAPI runtime dependency. Production agents use ChatOpenAI here.
        from langchain_openai import ChatOpenAI
        client_factory = ChatOpenAI
    return client_factory(
        model=model,
        temperature=0.7,
        base_url=base_url,
        model_kwargs={"response_format": json_schema_response_format(schema_name, schema)},
    )
