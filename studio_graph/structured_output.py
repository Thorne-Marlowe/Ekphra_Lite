"""LM Studio/OpenAI-compatible JSON Schema request configuration.

The schemas require the durable application contract while permitting the
storyboarder to retain its existing optional visual fields.  They are passed
directly as Chat Completions ``response_format`` rather than relying on a
provider-specific parser or free-form JSON repair.
"""
from __future__ import annotations

from typing import Any, Callable


STORY_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": ["title", "language", "tone", "trope", "characters", "beats"],
    "properties": {
        "title": {"type": "string"}, "language": {"type": "string"},
        "tone": {"type": "string"}, "trope": {"type": "string"},
        "characters": {"type": "array", "items": {"type": "object", "additionalProperties": True, "required": ["id", "name"]}},
        "beats": {"type": "array", "items": {"type": "object", "additionalProperties": True, "required": ["panel", "event"]}},
    },
}

DIALOGUE_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": ["dialogues"],
    "properties": {
        "dialogues": {"type": "array", "items": {"type": "object", "additionalProperties": True, "required": ["panel", "character_id", "text", "intent", "emotion"]}},
    },
}

STORYBOARD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": ["characters", "pages"],
    "properties": {
        "characters": {"type": "object", "additionalProperties": {"type": "object", "additionalProperties": True, "required": ["name", "base_prompt_en"]}},
        "pages": {"type": "array", "items": {"type": "object", "additionalProperties": True, "required": ["page_number", "panels"], "properties": {
            "page_number": {"type": "integer"},
            "panels": {"type": "array", "items": {"type": "object", "additionalProperties": True, "required": ["id", "panel_prompt_en", "active_char_ids", "dialogues"]}},
        }}},
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
