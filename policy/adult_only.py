"""Adult-only text policy for storyboard and diffusion-prompt integrity.

This module intentionally evaluates authored text and outgoing prompt payloads,
not rendered pixels. It is also suitable for validating a future exported
image-metadata payload before it is written to disk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# Keep age-related vocabulary in one place. Patterns use token boundaries so
# ordinary words such as "childcare" are not rejected by a substring match.
PROHIBITED_AGE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("childlike", re.compile(r"\bchild[\s-]?like\b", re.IGNORECASE)),
    ("child/children", re.compile(r"\bchild(?:ren)?\b", re.IGNORECASE)),
    ("kid/kids", re.compile(r"\bkids?\b", re.IGNORECASE)),
    ("teen", re.compile(r"\bteen(?:s|age(?:r)?s?|aged)?\b", re.IGNORECASE)),
    ("adolescent", re.compile(r"\badolescen(?:t|ts|ce)\b", re.IGNORECASE)),
    # "minor conflict" is ordinary narrative prose; reject the age noun form
    # instead of treating every adjective use of "minor" as an age reference.
    ("minor", re.compile(r"\b(?:a|an|the)\s+minor\b|\bminors\b", re.IGNORECASE)),
    ("underage", re.compile(r"\bunder[\s-]?age(?:d)?\b", re.IGNORECASE)),
    ("preteen", re.compile(r"\bpre[\s-]?teens?\b", re.IGNORECASE)),
    ("prepubescent", re.compile(r"\bpre[\s-]?pubescent\b", re.IGNORECASE)),
    ("toddler", re.compile(r"\btoddlers?\b", re.IGNORECASE)),
    ("infant", re.compile(r"\binfants?\b", re.IGNORECASE)),
    ("baby", re.compile(r"\bbab(?:y|ies)\b", re.IGNORECASE)),
    ("juvenile", re.compile(r"\bjuveniles?\b", re.IGNORECASE)),
    ("school-age figure", re.compile(r"\bschool(?:boy|girl|child|children)\b", re.IGNORECASE)),
    ("young or little minor", re.compile(r"\b(?:young|little)\s+(?:boy|girl|child|children|teen(?:ager)?)\b", re.IGNORECASE)),
    ("under 18", re.compile(r"\b(?:under|below)\s*18\b", re.IGNORECASE)),
    ("age below 18", re.compile(r"\bage\s*(?:[0-9]|1[0-7])\b", re.IGNORECASE)),
    ("age below 18", re.compile(r"\baged\s+(?:[0-9]|1[0-7])\b", re.IGNORECASE)),
    ("age below 18", re.compile(r"\b(?:[0-9]|1[0-7])\s*[- ]?years?[- ]?old\b", re.IGNORECASE)),
    ("age below 18", re.compile(r"\b(?:[0-9]|1[0-7])\s*y\s*/\s*o\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class PolicyViolation:
    path: str
    term: str
    matched_text: str

    def message(self) -> str:
        return (
            f"Adult-only policy violation at {self.path}: prohibited age-related "
            f"term '{self.matched_text}' ({self.term}). All intentionally depicted "
            "human or humanlike figures must be unambiguously adults aged 18 or older."
        )


def _text_violations(text: str, path: str) -> list[PolicyViolation]:
    violations: list[PolicyViolation] = []
    for term, pattern in PROHIBITED_AGE_PATTERNS:
        for match in pattern.finditer(text):
            violations.append(PolicyViolation(path, term, match.group(0)))
    return violations


def find_prohibited_age_references(value: Any, path: str = "payload") -> list[PolicyViolation]:
    """Recursively find prohibited text while retaining the exact field path."""
    if isinstance(value, str):
        return _text_violations(value, path)
    if isinstance(value, dict):
        violations: list[PolicyViolation] = []
        for key, item in value.items():
            item_path = f"{path}.{key}"
            # Optional structured metadata may represent age numerically rather
            # than in a prose description.
            if str(key).casefold() in {"age", "age_years"} and _is_minor_age_value(item):
                violations.append(PolicyViolation(item_path, "age below 18", str(item)))
            else:
                violations.extend(find_prohibited_age_references(item, item_path))
        return violations
    if isinstance(value, (list, tuple)):
        violations: list[PolicyViolation] = []
        for index, item in enumerate(value):
            violations.extend(find_prohibited_age_references(item, f"{path}[{index}]"))
        return violations
    return []


def _is_minor_age_value(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 0 <= value < 18
    if isinstance(value, str) and re.fullmatch(r"(?:[0-9]|1[0-7])(?:\.0+)?", value.strip()):
        return True
    return False


def adult_only_errors(value: Any, path: str = "payload") -> list[str]:
    return [violation.message() for violation in find_prohibited_age_references(value, path)]


def validate_metadata_payload(payload: Any, path: str = "metadata") -> None:
    """Reject a future persisted metadata payload before it can be written."""
    errors = adult_only_errors(payload, path)
    if errors:
        raise ValueError(" | ".join(errors))


def _workflow_string_payload(workflow: dict[str, Any]) -> dict[str, str]:
    """Flatten every string in the bound API payload with diagnostic paths."""
    strings: dict[str, str] = {}

    def collect(value: Any, path: str) -> None:
        if isinstance(value, str):
            strings[path] = value
        elif isinstance(value, dict):
            for key, item in value.items():
                collect(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                collect(item, f"{path}[{index}]")

    for node_id, node in workflow.items():
        node_path = f"workflow node '{node_id}'"
        if not isinstance(node, dict):
            collect(node, node_path)
            continue
        for field_name, value in node.items():
            if field_name == "inputs" and isinstance(value, dict):
                for input_name, input_value in value.items():
                    collect(input_value, f"{node_path} input '{input_name}'")
            else:
                collect(value, f"{node_path} field '{field_name}'")
    return strings


def validate_diffusion_prompts(
    *, panel_id: str, positive_prompt: str, negative_prompt: str, bound_workflow: dict[str, Any]
) -> None:
    """Validate exact text conditioning immediately before ComfyUI submission."""
    payload: dict[str, Any] = {
        f"panel '{panel_id}' positive prompt": positive_prompt,
        f"panel '{panel_id}' negative prompt": negative_prompt,
    }
    payload.update({f"panel '{panel_id}' {path}": value for path, value in _workflow_string_payload(bound_workflow).items()})
    validate_metadata_payload(payload, path="diffusion payload")
