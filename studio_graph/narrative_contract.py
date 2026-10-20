"""Small, provider-independent narrative-intent contract helpers.

This module deliberately records *what the user selected*, rather than trying
to infer genre or narrative intent from arbitrary prose.  Prompt guidance that
causes agents to follow the contract is a later phase; keeping the data model
here deterministic makes that separation explicit.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


NARRATIVE_RESOLUTIONS = frozenset({
    "auto",
    "punchline",
    "cliffhanger",
    "revelation",
    "transition",
    "emotional_beat",
    "deliberate_nonresolution",
})

INTENT_DIMENSIONS = ("genre", "tone", "narrative_resolution")


class NarrativeContractError(ValueError):
    """Raised when the explicit, structured contract is malformed."""


def normalize_requested_intent(value: dict[str, Any] | None) -> dict[str, str]:
    """Validate and normalize explicitly selected creative-intent values.

    Empty strings are treated as omitted selections.  ``auto`` is valid only
    for narrative resolution and is deliberately retained as a *requested*
    value; it is never copied to effective intent.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise NarrativeContractError("creative intent must be an object")

    unknown = set(value) - set(INTENT_DIMENSIONS)
    if unknown:
        raise NarrativeContractError(
            "creative intent contains unsupported field(s): " + ", ".join(sorted(unknown))
        )

    normalized: dict[str, str] = {}
    for field in ("genre", "tone"):
        raw = value.get(field)
        if raw is None:
            continue
        if not isinstance(raw, str):
            raise NarrativeContractError(f"creative intent {field} must be a string")
        cleaned = raw.strip()
        if cleaned:
            normalized[field] = cleaned

    resolution = value.get("narrative_resolution")
    if resolution is not None:
        if not isinstance(resolution, str) or resolution not in NARRATIVE_RESOLUTIONS:
            allowed = ", ".join(sorted(NARRATIVE_RESOLUTIONS))
            raise NarrativeContractError(
                f"creative intent narrative_resolution must be one of: {allowed}"
            )
        normalized["narrative_resolution"] = resolution
    return normalized


def normalize_inferred_intent(value: dict[str, Any] | None) -> dict[str, str]:
    """Normalize actual Planner inference without inventing missing values."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise NarrativeContractError("inferred creative intent must be an object")
    inferred = normalize_requested_intent(value)
    # ``auto`` is a user-selection instruction, never a Planner conclusion.
    if inferred.get("narrative_resolution") == "auto":
        inferred.pop("narrative_resolution")
    return inferred


def resolve_creative_intent(
    requested: dict[str, Any] | None,
    inferred: dict[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """Return requested, inferred, effective intent, and per-field provenance.

    Explicit values win.  ``auto`` delegates only that dimension to actual
    Planner inference.  There is intentionally no default resolution.
    """
    requested_normalized = normalize_requested_intent(requested)
    inferred_normalized = normalize_inferred_intent(inferred)
    effective: dict[str, str] = {}
    sources: dict[str, str] = {}

    for field in INTENT_DIMENSIONS:
        selected = requested_normalized.get(field)
        if selected is not None and not (field == "narrative_resolution" and selected == "auto"):
            effective[field] = selected
            sources[field] = "user_selection"
        elif field in inferred_normalized:
            effective[field] = inferred_normalized[field]
            sources[field] = "planner_inference"

    return requested_normalized, inferred_normalized, effective, sources


def conflict_warning(
    dimension: str,
    *,
    selection: str,
    prose_summary: str,
) -> dict[str, str]:
    """Build the stable warning shape for a reliably detected conflict.

    Conflict detection itself is intentionally not attempted here: arbitrary
    prose cannot be classified safely with keyword matching.  A later phase
    may supply a reliable source of these warnings.
    """
    if dimension not in INTENT_DIMENSIONS:
        raise NarrativeContractError(f"unsupported creative-intent dimension: {dimension}")
    return {
        "code": "creative_intent_conflict",
        "dimension": dimension,
        "selection": selection,
        "prose_summary": prose_summary,
        "message": (
            f"The explicit {dimension} selection is retained, but it may conflict "
            "with the supplied prose. Review the storyboard before rendering."
        ),
    }


def creative_intent_metadata(
    requested: dict[str, Any] | None,
    inferred: dict[str, Any] | None = None,
    warnings: list[dict[str, str]] | None = None,
) -> dict[str, Any] | None:
    """Create optional durable storyboard metadata from actual known values."""
    requested_value, inferred_value, effective, sources = resolve_creative_intent(requested, inferred)
    warning_values = deepcopy(warnings or [])
    if not (requested_value or inferred_value or effective or warning_values):
        return None
    return {
        "requested": requested_value,
        "inferred": inferred_value,
        "effective": effective,
        "source": sources,
        "warnings": warning_values,
    }


def resolved_creative_intent_metadata(
    requested: dict[str, Any] | None,
    inferred: dict[str, Any] | None,
    effective: dict[str, Any] | None,
    sources: dict[str, Any] | None,
    warnings: list[dict[str, str]] | None = None,
) -> dict[str, Any] | None:
    """Serialize the already-resolved state used by the graph.

    Revalidate that state against the one resolver so a stale or independently
    recomputed value cannot be written to the final storyboard.
    """
    requested_value, inferred_value, expected_effective, expected_sources = resolve_creative_intent(
        requested, inferred
    )
    if effective is not None and effective != expected_effective:
        raise NarrativeContractError("effective creative intent disagrees with requested and inferred intent")
    if sources is not None and sources != expected_sources:
        raise NarrativeContractError("creative-intent provenance disagrees with requested and inferred intent")
    warning_values = deepcopy(warnings or [])
    if not (requested_value or inferred_value or expected_effective or warning_values):
        return None
    return {
        "requested": requested_value,
        "inferred": inferred_value,
        "effective": deepcopy(expected_effective),
        "source": deepcopy(expected_sources),
        "warnings": warning_values,
    }


def attach_creative_intent_metadata(schema: dict[str, Any], intent: dict[str, Any] | None) -> None:
    """Attach the contract without disturbing unrelated storyboard metadata."""
    if intent is None:
        return
    metadata = schema.get("metadata")
    if metadata is None:
        metadata = {}
        schema["metadata"] = metadata
    if not isinstance(metadata, dict):
        raise NarrativeContractError("storyboard metadata must be an object to store creative intent")
    metadata["creative_intent"] = deepcopy(intent)
